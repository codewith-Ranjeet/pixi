import os
import struct
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
try:
    from .model import Pixi, MicroChat
except ImportError:
    from model import Pixi, MicroChat

@torch.inference_mode()
def generate_chat(
    model: Pixi,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_id: int = 5,
    device: Optional[str] = None,
) -> list[int]:
    """Autoregressive KV-cached decoding with Top-P (Nucleus) sampling."""
    model.eval()
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    
    # Prefill phase
    logits, _, kv_caches = model(input_tensor)
    next_token_logits = logits[:, -1, :]
    generated = []

    for _ in range(max_new_tokens):
        if temperature > 0.0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens above top_p threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        generated.append(token_id)
        if token_id == eos_id or token_id == 2:
            break

        # Decode step: pass only the 1 newly generated token + KV cache
        logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]

    return generated

def quantize_q8_0(weight_fp32: np.ndarray) -> bytes:
    """
    Quantizes a 2D weight matrix into GGUF Q8_0 format (block size = 32).
    Each block of 32 floats is stored as: [fp16 scale (2 bytes)] + [32 x int8 (32 bytes)] = 34 bytes.
    """
    flat = weight_fp32.reshape(-1, 32)
    amax = np.max(np.abs(flat), axis=1, keepdims=True)
    scale = amax / 127.0
    inv_scale = np.where(scale == 0, 0.0, 1.0 / scale)
    quants = np.clip(np.round(flat * inv_scale), -127, 127).astype(np.int8)
    scale_fp16 = scale.astype(np.float16)

    # Interleave 2-byte fp16 scale and 32-byte int8 payload per block
    num_blocks = flat.shape[0]
    out = np.empty(num_blocks, dtype=[("d", "<f2"), ("qs", "i1", (32,))])
    out["d"] = scale_fp16.squeeze(-1)
    out["qs"] = quants
    return out.tobytes()

def export_to_gguf_q8_0(model: Pixi, out_path: str = "models/pixi-40m-q8_0.gguf"):
    """Writes a GGUF v3 binary container with Q8_0 quantized 2D weights and FP32 1D norms."""
    state_dict = {k: v.detach().cpu().to(torch.float32).numpy() for k, v in model.state_dict().items()}
    # Exclude tied lm_head.weight from redundant serialization
    if model.config.tie_word_embeddings and "lm_head.weight" in state_dict:
        del state_dict["lm_head.weight"]

    GGUF_MAGIC = b"GGUF"
    GGUF_VERSION = 3
    ALIGNMENT = 32

    def pack_string(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    # Metadata KV pairs (uint32 value type: 8=string, 4=uint32, 6=float32)
    kv_pairs = [
        ("general.architecture", 8, pack_string("llama")),
        ("llama.context_length", 4, struct.pack("<I", model.config.max_seq_len)),
        ("llama.embedding_length", 4, struct.pack("<I", model.config.d_model)),
        ("llama.block_count", 4, struct.pack("<I", model.config.n_layers)),
        ("llama.feed_forward_length", 4, struct.pack("<I", model.config.d_ff)),
        ("llama.attention.head_count", 4, struct.pack("<I", model.config.n_heads)),
        ("llama.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", model.config.rms_norm_eps)),
        ("llama.rope.freq_base", 6, struct.pack("<f", model.config.rope_theta)),
    ]

    tensor_infos = []
    tensor_payloads = []
    current_offset = 0

    for name, arr in state_dict.items():
        is_2d = (arr.ndim == 2)
        ggml_type = 8 if is_2d else 0  # 8 = GGML_TYPE_Q8_0, 0 = GGML_TYPE_F32
        payload = quantize_q8_0(arr) if is_2d else arr.astype(np.float32).tobytes()

        # GGUF dimensions are written in reverse order compared to PyTorch shape
        dims = arr.shape[::-1]
        info_bytes = pack_string(name) + struct.pack("<I", len(dims))
        for d in dims:
            info_bytes += struct.pack("<Q", d)
        info_bytes += struct.pack("<I", ggml_type) + struct.pack("<Q", current_offset)

        tensor_infos.append(info_bytes)
        # Pad payload to 32-byte alignment
        pad = (ALIGNMENT - (len(payload) % ALIGNMENT)) % ALIGNMENT
        padded_payload = payload + (b"\x00" * pad)
        tensor_payloads.append(padded_payload)
        current_offset += len(padded_payload)

    with open(out_path, "wb") as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensor_infos)))
        f.write(struct.pack("<Q", len(kv_pairs)))

        for key, vtype, val_bytes in kv_pairs:
            f.write(pack_string(key))
            f.write(struct.pack("<I", vtype))
            f.write(val_bytes)

        for info in tensor_infos:
            f.write(info)

        # Pad header to 32-byte alignment before writing raw tensor data section
        pos = f.tell()
        header_pad = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT
        f.write(b"\x00" * header_pad)

        for payload in tensor_payloads:
            f.write(payload)

    print(f"Exported GGUF Q8_0 binary to {out_path} ({os.path.getsize(out_path) / (1024**2):.2f} MB)")