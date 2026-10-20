// Compile: g++ -O3 -mavx2 -mf16c -mfma -std=c++20 pixi_infer.cpp -o pixi_infer.exe
#include <immintrin.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>



#endif




#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <chrono>

// ============================================================================
// 1. PACKED QUANTIZATION STRUCTURES & SIMD KERNELS
// ============================================================================

#pragma pack(push, 1)
struct BlockQ8_0 {
    uint16_t d;       // IEEE 754 FP16 scale (2 bytes)
    int8_t   qs[32];  // 32 quantized int8 weights (32 bytes)
};
#pragma pack(pop)

static_assert(sizeof(BlockQ8_0) == 34, "Alignment error: BlockQ8_0 must be strictly 34 bytes.");

// Horizontal sum of an 8-lane AVX2 float register
inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Dequantizes a single Q8_0 row into an FP32 buffer (used for token embedding lookup)
inline void dequantize_row_q8_0(const BlockQ8_0* __restrict x, float* __restrict y, int k) {
    const int nb = k / 32;
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d);
        const __m256 vd = _mm256_set1_ps(d);
        for (int j = 0; j < 4; ++j) {
            __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(x[i].qs + j * 8));
            __m256i q32 = _mm256_cvtepi8_epi32(q8);
            __m256 wf = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), vd);
            _mm256_storeu_ps(y + i * 32 + j * 8, wf);
        }
    }
}

// Fused Q8_0 x FP32 Matrix-Vector Multiplication: out[M] = W[M, K] * inp[K]
void matvec_q8_0_f32(
    float* __restrict out,
    const BlockQ8_0* __restrict W,
    const float* __restrict inp,
    int M,
    int K
) {
    const int nb = K / 32;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < M; ++row) {
        const BlockQ8_0* __restrict row_blocks = W + row * nb;
        __m256 acc = _mm256_setzero_ps();

        for (int b = 0; b < nb; ++b) {
            // Convert FP16 scale to FP32 using F16C hardware intrinsic
            const float d = _cvtsh_ss(row_blocks[b].d);
            const __m256 vd = _mm256_set1_ps(d);
            const int8_t* __restrict qs = row_blocks[b].qs;
            const float* __restrict xp = inp + b * 32;

            // Unroll 32 elements as 4 chunks of 8 lanes
            __m256 block_sum = _mm256_setzero_ps();
            for (int j = 0; j < 4; ++j) {
                __m128i q8  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + j * 8));
                __m256i q32 = _mm256_cvtepi8_epi32(q8);
                __m256  wf  = _mm256_cvtepi32_ps(q32);
                __m256  xf  = _mm256_loadu_ps(xp + j * 8);
                block_sum   = _mm256_fmadd_ps(wf, xf, block_sum);
            }
            acc = _mm256_fmadd_ps(vd, block_sum, acc);
        }
        out[row] = hsum256_ps(acc);
    }
}

// AVX2 RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
void rmsnorm_f32(float* out, const float* x, const float* weight, int size, float eps = 1e-5f) {
    __m256 sum_sq = _mm256_setzero_ps();
    for (int i = 0; i < size; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum_sq = _mm256_fmadd_ps(vx, vx, sum_sq);
    }
    const float mean_sq = hsum256_ps(sum_sq) / static_cast<float>(size);
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    const __m256 vscale = _mm256_set1_ps(inv_rms);

    for (int i = 0; i < size; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw));
    }
}

// ============================================================================
// 2. ZERO-COPY GGUF v3 MEMORY-MAPPED LOADER
// ============================================================================

struct TensorView {
    uint32_t ggml_type;
    const void* data;
};

class GGUFMappedLoader {
public:
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMapping = NULL;
#else
    int fd = -1;
#endif
    size_t file_size = 0;
    const uint8_t* base_ptr = nullptr;
    std::unordered_map<std::string, TensorView> tensors;

    explicit GGUFMappedLoader(const std::string& path) {
#ifdef _WIN32
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            std::cerr << "Failed to open GGUF file: " << path << "\n";
            std::exit(1);
        }
        LARGE_INTEGER size;
        if (!GetFileSizeEx(hFile, &size)) {
            std::cerr << "Failed to get file size for: " << path << "\n";
            CloseHandle(hFile);
            std::exit(1);
        }
        file_size = static_cast<size_t>(size.QuadPart);
        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMapping) {
            std::cerr << "CreateFileMapping failed for: " << path << "\n";
            CloseHandle(hFile);
            std::exit(1);
        }
        base_ptr = static_cast<const uint8_t*>(MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
        if (!base_ptr) {
            std::cerr << "MapViewOfFile failed for: " << path << "\n";
            CloseHandle(hMapping);
            CloseHandle(hFile);
            std::exit(1);
        }
#else
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            perror("Failed to open GGUF file");
            std::exit(1);
        }
        struct stat st{};
        fstat(fd, &st);
        file_size = st.st_size;

        base_ptr = static_cast<const uint8_t*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (base_ptr == MAP_FAILED) {
            perror("mmap failed");
            std::exit(1);
        }
#endif
        parse_header();
    }

    ~GGUFMappedLoader() {
#ifdef _WIN32
        if (base_ptr) UnmapViewOfFile(base_ptr);
        if (hMapping) CloseHandle(hMapping);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (base_ptr && base_ptr != MAP_FAILED) munmap(const_cast<uint8_t*>(base_ptr), file_size);
        if (fd >= 0) close(fd);
#endif
    }

    template <typename T>
    const T* get(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            std::cerr << "Fatal: Missing tensor in GGUF: " << name << "\n";
            std::exit(1);
        }
        return reinterpret_cast<const T*>(it->second.data);
    }

private:
    template <typename T>
    T read_val(size_t& offset) const {
        T val;
        std::memcpy(&val, base_ptr + offset, sizeof(T));
        offset += sizeof(T);
        return val;
    }

    std::string read_string(size_t& offset) const {
        uint64_t len = read_val<uint64_t>(offset);
        std::string s(reinterpret_cast<const char*>(base_ptr + offset), len);
        offset += len;
        return s;
    }

    void parse_header() {
        size_t offset = 0;
        if (std::memcmp(base_ptr, "GGUF", 4) != 0) {
            std::cerr << "Invalid GGUF magic bytes.\n";
            std::exit(1);
        }
        offset += 4;
        uint32_t version = read_val<uint32_t>(offset);
        if (version != 3) {
            std::cerr << "Unsupported GGUF version: " << version << "\n";
            std::exit(1);
        }

        uint64_t tensor_count = read_val<uint64_t>(offset);
        uint64_t kv_count     = read_val<uint64_t>(offset);

        // Skip KV pairs (supports uint32=4, float32=6, string=8 emitted by our exporter)
        for (uint64_t i = 0; i < kv_count; ++i) {
            std::string key = read_string(offset);
            uint32_t vtype  = read_val<uint32_t>(offset);
            if (vtype == 8) {
                read_string(offset);
            } else if (vtype == 4 || vtype == 6) {
                offset += 4;
            } else {
                std::cerr << "Unsupported GGUF metadata type " << vtype << " for key " << key << "\n";
                std::exit(1);
            }
        }

        struct RawInfo {
            std::string name;
            uint32_t type;
            uint64_t rel_offset;
        };
        std::vector<RawInfo> infos;
        infos.reserve(tensor_count);

        for (uint64_t i = 0; i < tensor_count; ++i) {
            std::string name = read_string(offset);
            uint32_t n_dims  = read_val<uint32_t>(offset);
            for (uint32_t d = 0; d < n_dims; ++d) read_val<uint64_t>(offset);
            uint32_t type       = read_val<uint32_t>(offset);
            uint64_t rel_offset = read_val<uint64_t>(offset);
            infos.push_back({name, type, rel_offset});
        }

        // Align data section start to 32 bytes
        const size_t alignment = 32;
        size_t pad = (alignment - (offset % alignment)) % alignment;
        size_t data_base = offset + pad;

        for (const auto& info : infos) {
            tensors[info.name] = {info.type, base_ptr + data_base + info.rel_offset};
        }
    }
};

// ============================================================================
// 3. MICROCHAT-40M ARCHITECTURE & CONTIGUOUS ARENA
// ============================================================================

struct Config {
    static constexpr int vocab_size  = 16384;
    static constexpr int d_model     = 512;
    static constexpr int n_layers    = 10;
    static constexpr int n_heads     = 8;
    static constexpr int head_dim    = 64;
    static constexpr int d_ff        = 1408;
    static constexpr int max_seq_len = 2048;
    static constexpr float rope_theta = 10000.0f;
    static constexpr float norm_eps   = 1e-5f;
};

struct LayerWeights {
    const float*    attn_norm;
    const BlockQ8_0* q_proj;
    const BlockQ8_0* k_proj;
    const BlockQ8_0* v_proj;
    const BlockQ8_0* o_proj;
    const float*    ffn_norm;
    const BlockQ8_0* gate_proj;
    const BlockQ8_0* up_proj;
    const BlockQ8_0* down_proj;
};

class PixiEngine {
public:
    GGUFMappedLoader loader;
    const BlockQ8_0* embed_tokens; // Tied with lm_head
    const float*     final_norm;
    LayerWeights     layers[Config::n_layers];

    // Pre-allocated activation & KV-cache arena (zero heap allocations during inference)
    std::vector<float> x;
    std::vector<float> xb;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn_out;
    std::vector<float> gate_buf;
    std::vector<float> up_buf;
    std::vector<float> logits;

    // KV Cache: [n_layers, max_seq_len, d_model]
    std::vector<float> k_cache;
    std::vector<float> v_cache;

    // Precomputed RoPE tables: [max_seq_len, head_dim / 2]
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;

    explicit PixiEngine(const std::string& gguf_path) : loader(gguf_path) {
        embed_tokens = loader.get<BlockQ8_0>("embed_tokens.weight");
        final_norm   = loader.get<float>("norm.weight");

        for (int l = 0; l < Config::n_layers; ++l) {
            std::string p = "layers." + std::to_string(l) + ".";
            layers[l].attn_norm = loader.get<float>(p + "attn_norm.weight");
            layers[l].q_proj    = loader.get<BlockQ8_0>(p + "attn.q_proj.weight");
            layers[l].k_proj    = loader.get<BlockQ8_0>(p + "attn.k_proj.weight");
            layers[l].v_proj    = loader.get<BlockQ8_0>(p + "attn.v_proj.weight");
            layers[l].o_proj    = loader.get<BlockQ8_0>(p + "attn.o_proj.weight");
            layers[l].ffn_norm  = loader.get<float>(p + "ffn_norm.weight");
            layers[l].gate_proj = loader.get<BlockQ8_0>(p + "mlp.gate_proj.weight");
            layers[l].up_proj   = loader.get<BlockQ8_0>(p + "mlp.up_proj.weight");
            layers[l].down_proj = loader.get<BlockQ8_0>(p + "mlp.down_proj.weight");
        }

        x.resize(Config::d_model);
        xb.resize(Config::d_model);
        q.resize(Config::d_model);
        k.resize(Config::d_model);
        v.resize(Config::d_model);
        attn_out.resize(Config::d_model);
        gate_buf.resize(Config::d_ff);
        up_buf.resize(Config::d_ff);
        logits.resize(Config::vocab_size);

        k_cache.resize(Config::n_layers * Config::max_seq_len * Config::d_model, 0.0f);
        v_cache.resize(Config::n_layers * Config::max_seq_len * Config::d_model, 0.0f);

        init_rope();
    }

    void init_rope() {
        const int half_dim = Config::head_dim / 2;
        rope_cos.resize(Config::max_seq_len * half_dim);
        rope_sin.resize(Config::max_seq_len * half_dim);

        for (int pos = 0; pos < Config::max_seq_len; ++pos) {
            for (int i = 0; i < half_dim; ++i) {
                float freq = 1.0f / std::pow(Config::rope_theta, static_cast<float>(2 * i) / Config::head_dim);
                float angle = pos * freq;
                rope_cos[pos * half_dim + i] = std::cos(angle);
                rope_sin[pos * half_dim + i] = std::sin(angle);
            }
        }
    }

    // Matches PyTorch's rotate_half layout: pairs (i, i + head_dim/2)
    inline void apply_rope_half(float* vec, int pos) const {
        const int half = Config::head_dim / 2;
        const float* cos_ptr = rope_cos.data() + pos * half;
        const float* sin_ptr = rope_sin.data() + pos * half;

        for (int h = 0; h < Config::n_heads; ++h) {
            float* head_vec = vec + h * Config::head_dim;
            for (int i = 0; i < half; ++i) {
                float x1 = head_vec[i];
                float x2 = head_vec[i + half];
                float c  = cos_ptr[i];
                float s  = sin_ptr[i];
                head_vec[i]        = x1 * c - x2 * s;
                head_vec[i + half] = x2 * c + x1 * s;
            }
        }
    }

    const float* forward(int token_id, int pos) {
        // 1. Dequantize token embedding row into activation vector x
        const int blocks_per_row = Config::d_model / 32;
        dequantize_row_q8_0(embed_tokens + token_id * blocks_per_row, x.data(), Config::d_model);

        // 2. Transformer Decoder Layers
        for (int l = 0; l < Config::n_layers; ++l) {
            const auto& lw = layers[l];

            // Pre-Attention RMSNorm
            rmsnorm_f32(xb.data(), x.data(), lw.attn_norm, Config::d_model, Config::norm_eps);

            // Q, K, V Projections
            matvec_q8_0_f32(q.data(), lw.q_proj, xb.data(), Config::d_model, Config::d_model);
            matvec_q8_0_f32(k.data(), lw.k_proj, xb.data(), Config::d_model, Config::d_model);
            matvec_q8_0_f32(v.data(), lw.v_proj, xb.data(), Config::d_model, Config::d_model);

            // Apply Rotary Position Embeddings
            apply_rope_half(q.data(), pos);
            apply_rope_half(k.data(), pos);

            // Append K and V to layer KV Cache
            size_t cache_offset = (static_cast<size_t>(l) * Config::max_seq_len + pos) * Config::d_model;
            std::memcpy(k_cache.data() + cache_offset, k.data(), Config::d_model * sizeof(float));
            std::memcpy(v_cache.data() + cache_offset, v.data(), Config::d_model * sizeof(float));

            // Multi-Head Attention over cached sequence [0 ... pos]
            const float scale = 1.0f / std::sqrt(static_cast<float>(Config::head_dim));
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < Config::n_heads; ++h) {
                const float* q_head = q.data() + h * Config::head_dim;
                float* out_head     = attn_out.data() + h * Config::head_dim;

                // Thread-local stack buffer for attention scores (max 2048 floats = 8 KB)
                float scores[Config::max_seq_len];
                float max_score = -1e30f;

                for (int t = 0; t <= pos; ++t) {
                    const float* k_head = k_cache.data() + (static_cast<size_t>(l) * Config::max_seq_len + t) * Config::d_model + h * Config::head_dim;
                    __m256 dot_acc = _mm256_setzero_ps();
                    for (int d = 0; d < Config::head_dim; d += 8) {
                        dot_acc = _mm256_fmadd_ps(_mm256_loadu_ps(q_head + d), _mm256_loadu_ps(k_head + d), dot_acc);
                    }
                    float s = hsum256_ps(dot_acc) * scale;
                    scores[t] = s;
                    if (s > max_score) max_score = s;
                }

                // Numerically stable Softmax
                float sum_exp = 0.0f;
                for (int t = 0; t <= pos; ++t) {
                    scores[t] = std::exp(scores[t] - max_score);
                    sum_exp += scores[t];
                }
                const float inv_sum = 1.0f / sum_exp;

                // Weighted sum of V
                std::memset(out_head, 0, Config::head_dim * sizeof(float));
                for (int t = 0; t <= pos; ++t) {
                    const float prob = scores[t] * inv_sum;
                    const __m256 vprob = _mm256_set1_ps(prob);
                    const float* v_head = v_cache.data() + (static_cast<size_t>(l) * Config::max_seq_len + t) * Config::d_model + h * Config::head_dim;

                    for (int d = 0; d < Config::head_dim; d += 8) {
                        __m256 vo = _mm256_loadu_ps(out_head + d);
                        __m256 vv = _mm256_loadu_ps(v_head + d);
                        _mm256_storeu_ps(out_head + d, _mm256_fmadd_ps(vprob, vv, vo));
                    }
                }
            }

            // Output Projection + Residual Connection
            matvec_q8_0_f32(xb.data(), lw.o_proj, attn_out.data(), Config::d_model, Config::d_model);
            for (int i = 0; i < Config::d_model; ++i) x[i] += xb[i];

            // Pre-FFN RMSNorm
            rmsnorm_f32(xb.data(), x.data(), lw.ffn_norm, Config::d_model, Config::norm_eps);

            // SwiGLU MLP: down_proj(SiLU(gate_proj(x)) * up_proj(x))
            matvec_q8_0_f32(gate_buf.data(), lw.gate_proj, xb.data(), Config::d_ff, Config::d_model);
            matvec_q8_0_f32(up_buf.data(),   lw.up_proj,   xb.data(), Config::d_ff, Config::d_model);

            for (int i = 0; i < Config::d_ff; ++i) {
                float g = gate_buf[i];
                float silu = g / (1.0f + std::exp(-g));
                gate_buf[i] = silu * up_buf[i];
            }

            matvec_q8_0_f32(xb.data(), lw.down_proj, gate_buf.data(), Config::d_model, Config::d_ff);
            for (int i = 0; i < Config::d_model; ++i) x[i] += xb[i];
        }

        // 3. Final RMSNorm & Tied LM Head Logits
        rmsnorm_f32(x.data(), x.data(), final_norm, Config::d_model, Config::norm_eps);
        matvec_q8_0_f32(logits.data(), embed_tokens, x.data(), Config::vocab_size, Config::d_model);

        return logits.data();
    }
};

// ============================================================================
// 4. CLI DRIVER & BENCHMARK HARNESS
// ============================================================================

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : "models/pixi-40m-q8_0.gguf";
    std::cout << "[Pixi-40M] Mapping GGUF binary: " << model_path << "\n";

    PixiEngine engine(model_path);

    // ChatML formatted synthetic prompt tokens (<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n)
    std::vector<int> prompt_tokens = {4, 312, 10, 891, 442, 5, 10, 4, 519, 10};
    const int max_new_tokens = 128;
    const int im_end_id = 5;

    auto t_start = std::chrono::high_resolution_clock::now();
    int pos = 0;
    const float* logits = nullptr;

    // Prefill Phase
    for (int tok : prompt_tokens) {
        logits = engine.forward(tok, pos++);
    }

    // Autoregressive Decode Phase (Greedy Argmax for deterministic verification)
    std::cout << "[Pixi-40M] Generated Token IDs: ";
    int generated_count = 0;
    for (int step = 0; step < max_new_tokens && pos < Config::max_seq_len; ++step) {
        int next_token = static_cast<int>(std::max_element(logits, logits + Config::vocab_size) - logits);
        std::cout << next_token << " " << std::flush;
        ++generated_count;

        if (next_token == im_end_id) break;
        logits = engine.forward(next_token, pos++);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double tok_per_sec = (prompt_tokens.size() + generated_count) / (elapsed_ms / 1000.0);

    std::cout << "\n[Pixi-40M] Completed " << generated_count << " tokens in "
              << elapsed_ms << " ms (" << tok_per_sec << " tok/s)\n";
    return 0;
}