# 📋 Pixi-40M: Master Engineering Plan & Execution Log

> **A comprehensive record of architectural design, parameter budgeting, bug audits, prototype validation, hardware tuning, dataset sharding, and production roadmap for building a custom ~40.5M parameter English chatbot model.**

---

## 📌 Executive Summary

* **Model Name**: Pixi-40M
* **Architecture**: Modern Decoder-Only Transformer (Llama / Gemma style)
* **Target Parameter Count**: Exactly **40,512,000 parameters** (~40.5M)
* **Primary Objective**: Ultra-low latency, highly coherent English conversational assistance on edge hardware and consumer GPUs.
* **Target Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) + Intel Core i5-12450HX (12 Threads) + 16GB RAM.

---

## 📐 Phase 1: Architectural Formulation & Mathematical Parameter Budget

To achieve efficient dialogue modeling on consumer hardware, every tensor dimension is chosen to hit an exact parameter budget of 40,512,000 parameters.

### Structural Hyperparameters

* **Vocabulary Size ($V$)**: $16,384$ ($2^{14}$, optimal power-of-two for GPU Tensor Cores).
* **Hidden Dimension ($d_{\text{model}}$)**: $512$
* **Number of Layers ($L$)**: $10$
* **Attention Heads ($H$)**: $8$ ($\text{Head Dim} = 64$)
* **Key-Value Heads ($H_{kv}$)**: $8$ (Multi-Head Attention)
* **Feedforward Intermediate Dimension ($d_{\text{ff}}$)**: $1,408$ ($\approx \frac{8}{3} \times d_{\text{model}}$)
* **Maximum Context Length**: $1,024$ tokens (expandable to $2,048$)
* **Position Representation**: Rotary Position Embeddings (RoPE, $\theta = 10,000$)
* **Normalization**: Root Mean Square Normalization ($\text{RMSNorm}$, $\epsilon = 10^{-5}$)
* **Weight Tying**: Enabled (`lm_head.weight == embed_tokens.weight`)

### Matrix-by-Matrix Parameter Breakdown

$$\begin{aligned}
\text{Token Embeddings} &= V \times d_{\text{model}} = 16,384 \times 512 = \mathbf{8,388,608} \\
\text{Attention Q, K, V, O Projections} &= 4 \times (d_{\text{model}} \times d_{\text{model}}) = 4 \times (512 \times 512) = 1,048,576 \\
\text{SwiGLU Gate, Up, Down Projections} &= 3 \times (d_{\text{model}} \times d_{\text{ff}}) = 3 \times (512 \times 1,408) = 2,162,688 \\
\text{RMSNorms (Attn Norm + FFN Norm)} &= 2 \times d_{\text{model}} = 2 \times 512 = 1,024 \\
\text{Total Parameters Per Layer} &= 1,048,576 + 2,162,688 + 1,024 = \mathbf{3,212,288} \\
\text{Total for 10 Layers} &= 10 \times 3,212,288 = \mathbf{32,122,880} \\
\text{Final RMSNorm} &= d_{\text{model}} = \mathbf{512} \\
\text{LM Head (Weight Tied)} &= 0\ \text{additional params} \\
\mathbf{Grand\ Total} &= \mathbf{8,388,608} + \mathbf{32,122,880} + \mathbf{512} = \mathbf{40,512,000}
\end{aligned}$$

---

## 🔤 Phase 2: Custom BPE Tokenizer Construction

A custom Byte-Level Byte-Pair Encoding (BPE) tokenizer was trained with exact special token alignment for conversational control.

* **Special Token IDs**:
  * `<|pad|>`: ID `0`
  * `<|bos|>`: ID `1`
  * `<|eos|>`: ID `2`
  * `<|unk|>`: ID `3`
  * `<|im_start|>`: ID `4` (ChatML message boundary start)
  * `<|im_end|>`: ID `5` (ChatML message boundary end)
* **Vocabulary Alignment**: Implemented automatic reserved-token padding (`<|extra_id_i|>`) up to exactly $16,384$ tokens to guarantee complete compatibility with the model's embedding matrix.
* **Artifact**: [`tokenizer/tokenizer.json`](file:///c:/workspace/pixi/tokenizer/tokenizer.json) (~3.0 MB).

---

## 🛠️ Phase 3: Comprehensive Codebase Audit & Bug Resolutions

A rigorous audit of the code was performed, identifying and fixing 8 critical bugs across Python and C++ implementations:

1. **C++ Zero-Copy Memory Mapping on Windows (`pixi_infer.cpp`)**:
   * *Problem*: The file relied on POSIX-exclusive `<sys/mman.h>`, `mmap`, and `unistd.h`, failing compilation on Windows with MSYS2/MinGW.
   * *Solution*: Implemented a clean cross-platform `#ifdef _WIN32` memory-mapped loader using native Windows APIs (`CreateFileA`, `CreateFileMappingA`, `MapViewOfFile`, `UnmapViewOfFile`, `CloseHandle`), retaining POSIX `mmap` fallback for Linux/macOS.

2. **SIMD AVX2 / FMA Compiler Flag Incompatibility**:
   * *Problem*: `_mm256_fmadd_ps` calls failed to inline with `target specific option mismatch` error during compilation.
   * *Solution*: Added `-mfma` and `-mf16c` compiler flags to properly enable hardware Fused Multiply-Add instructions.

3. **In-place Pointer Aliasing in RMSNorm SIMD Kernel**:
   * *Problem*: `rmsnorm_f32(x.data(), x.data(), ...)` passed identical pointers to two `__restrict` parameters, causing undefined compiler optimization behavior.
   * *Solution*: Removed `__restrict` from `rmsnorm_f32` to ensure strict standard compliance.

4. **RoPE Tensor Device Synchronization (`model.py`)**:
   * *Problem*: `cos` and `sin` RoPE cache tables were cast to `q.dtype` but omitted device synchronization, causing device mismatch errors on GPU execution.
   * *Solution*: Explicitly synced tensors via `.to(device=q.device, dtype=q.dtype)`.

5. **Safe KV Cache Length Slicing (`model.py`)**:
   * *Problem*: `kv_caches[0][0]` crashed when passed empty or unitialized cache objects.
   * *Solution*: Added length and null-check guards before inspecting cached sequence dimension.

6. **Causal Token Shift in Sequence Log-Probability Computation (`trainer.py`)**:
   * *Problem*: `compute_sequence_logprobs` gathered unshifted labels directly from next-token logits, introducing an off-by-one label misalignment.
   * *Solution*: Implemented proper causal slicing (`shift_logits = logits[:, :-1, :]` and `shift_labels = labels[:, 1:]`).

7. **Optimizer Fused Guard for Non-CUDA Hardware (`trainer.py`)**:
   * *Problem*: Passing `fused=True` to `torch.optim.AdamW` on CPU caused runtime exceptions.
   * *Solution*: Guarded with `fused = torch.cuda.is_available() and any(p.is_cuda for p in model.parameters())`.

8. **GGUF Exporter Missing Import & Dual EOS Handling (`export_and_infer.py`)**:
   * *Problem*: `os.path.getsize()` threw `NameError: name 'os' is not defined`.
   * *Solution*: Added missing `import os` and configured inference to cleanly stop on either `<|im_end|>` (ID 5) or `<|eos|>` (ID 2).

---

## 🧪 Phase 4: Prototype Verification & Dual Inference Benchmarks

The corrected codebase was subjected to end-to-end verification via [`scripts/verify_pipeline.py`](file:///c:/workspace/pixi/scripts/verify_pipeline.py):

* **Supervised Fine-Tuning Run**: 15 steps with masked ChatML loss (`labels = -100` on prompt/user tokens).
  * Step 1 Loss: **9.8467**
  * Step 8 Loss: **6.3771**
  * Step 12 Loss: **5.6356**
* **GGUF Q8_0 Serialization**: Saved full 40.5M model into a standalone **41.09 MB** container (`models/pixi-40m-q8_0.gguf`).
* **Python Autoregressive Decode**: Achieved **87.8 tokens/second** on CPU with KV caching.
* **Native C++ AVX2/FMA Inference Engine**: Compiled and verified `cpp/pixi_infer.exe`, achieving **272.9 tokens/second** on CPU.

---

## ⚡ Phase 5: Target Hardware Profiling (RTX 3050 6GB)

Hardware profiling was conducted for the target setup:
* **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (6,144 MiB VRAM, Compute Capability 8.6, Ampere).
* **CPU**: Intel Core i5-12450HX (8 cores / 12 threads).
* **RAM**: 16 GB DDR4/DDR5.

### VRAM Budget Allocation (Training at Sequence Length 1,024)

| Component | Precision | Memory |
| :--- | :--- | :--- |
| **Model Weights** | BF16 / FP16 | ~81 MB |
| **Gradients** | BF16 / FP16 | ~81 MB |
| **AdamW Optimizer States ($m, v$)** | FP32 | ~324 MB |
| **Forward Activations (Batch 8, Seq 1024)** | BF16 | ~1.5 GB |
| **PyTorch CUDA Context & Buffers** | — | ~450 MB |
| **Total Estimated VRAM Usage** | — | **~2.43 GB (Leaves >3.5 GB free!)** |

### Optimized Pre-training Configuration

* **Micro-Batch Size**: $8$
* **Gradient Accumulation Steps**: $8$
* **Effective Batch Size**: $8 \times 8 = 64$ sequences ($\mathbf{65,536\ \text{tokens/step}}$)
* **Precision**: `torch.bfloat16` with native Ampere Tensor Core execution
* **Context Window**: $1,024$ tokens

---

## 📦 Phase 6: Pre-training Data Acquisition & Sharding

1. **Source Corpus**: Extracted and streamed educational, filtered web text from `HuggingFaceFW/fineweb-edu` (sample-10BT).
2. **Binary Pre-Tokenization**: Tokenized using [`tokenizer/tokenizer.json`](file:///c:/workspace/pixi/tokenizer/tokenizer.json) and packed into flat `uint16` memory-mapped arrays.
3. **Completed Dataset Shards**:
   * **Total Tokens Prepared**: **500,001,792 tokens** (~500M tokens).
   * **Total Sequences ($T=1024$)**: **488,283 sequences**.
   * **Location**: [`data/pretrain_bins/shard_000.bin`](file:///c:/workspace/pixi/data/pretrain_bins) through `shard_010.bin` (11 binary shards, 1.0 GB total on disk).
   * **Data Access**: Zero-copy memory mapping (`np.memmap`) via `ShardedDirectoryDataset`, eliminating RAM limits and CPU loading bottlenecks.

---

## 🗂️ Phase 7: Production Repository Architecture

The project has been refactored into clean, modular directories:

```text
pixi/
├── configs/               # Hyperparameter & architecture specifications
│   └── model_config.json
├── src/                   # Reusable library modules
│   ├── __init__.py
│   ├── model.py           # Transformer architecture definition
│   ├── data_pipeline.py   # Tokenizer training & dataset streaming classes
│   ├── trainer.py         # Optimizer, cosine schedule, SFT & DPO loss
│   └── export_and_infer.py# GGUF Q8_0 exporter & KV-cache generation
├── scripts/               # Entrypoint & automation scripts
│   ├── prepare_data.py    # Dataset downloading & binary sharding
│   ├── train_pretrain.py  # Pre-training execution module
│   └── verify_pipeline.py # End-to-end verification pipeline
├── cpp/                   # C++ native inference engine
│   ├── pixi_infer.cpp# AVX2/FMA/F16C SIMD inference source
│   ├── pixi_infer.exe# Compiled native runner (270+ tok/s)
│   └── compile.bat        # Windows compiler script
├── tokenizer/             # Vocabulary & tokenizer assets
│   └── tokenizer.json     # 16,384-vocab Byte-Level BPE tokenizer
├── models/                # Quantized model containers
│   └── pixi-40m-q8_0.gguf
├── checkpoints/           # Training states and weights
│   ├── pretrain/          # Pre-training checkpoint directory
│   └── pixi_40m.pt   # Verified prototype weights
├── data/                  # Pre-training shards & instruction sets
│   ├── pretrain_bins/     # 11 binary shards (500M tokens)
│   ├── chat_sft.jsonl     # ChatML instruction samples
│   └── raw_corpus.txt     # Raw text corpus
├── requirements.txt       # Python dependencies
├── .gitignore             # Exclusion rules for large binaries and caches
├── plan.md                # This comprehensive engineering log
├── README.md              # Public repository documentation
├── chat.py                # Interactive terminal chat client
└── chat_base.py           # Base model testing script
```

---

## 🚀 Phase 8: Execution Instructions & Subsequent Roadmap

### 1. Pre-training: COMPLETED

* Checkpoint: checkpoints/pretrain/pixi_step_10000.pt (486 MB)

* Final Loss: **1.3700** (converged from 9.70 initial cross-entropy).

* Result: Fluent English base text representation across grammar, vocabulary, and world knowledge.



### 2. Phase 9: Supervised Fine-Tuning (SFT) - READY

* Run conversational fine-tuning:

  `powershell

  python scripts/train_sft.py

  `

* Trains on data/chat_sft.jsonl using [ChatMLDataset](file:///c:/workspace/pixi/src/data_pipeline.py) with prompt masking (loss computed only on assistant tokens).

* Saves fine-tuned checkpoint: checkpoints/pixi_chat_sft.pt.

* Automatically exports quantized GGUF model: models/pixi-40m-q8_0.gguf.



### 3. Phase 10: Interactive Chat & Native Inference

* **Terminal Chat (Token Streaming)**:

  `powershell

  python chat.py

  `

* **C++ AVX2 Native Runner (270+ tok/s)**:

  `powershell

  .\cpp\pixi_infer.exe models\pixi-40m-q8_0.gguf

  `



### 4. Phase 11: Direct Preference Optimization (DPO) & Future Scaling

* Align conversational preferences using [dpo_train_step](file:///c:/workspace/pixi/src/trainer.py) on paired chosen vs. rejected responses to eliminate repetitive loops and enhance helpfulness.

