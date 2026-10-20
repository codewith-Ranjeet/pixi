import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PixiConfig:
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8
    d_ff: int = 1408
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

MicroChatConfig = PixiConfig  # Backward-compatible alias

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization computed in fp32 to prevent underflow/overflow."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed.to(orig_dtype))

def precompute_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0, device: torch.device = None):
    """Precomputes cos and sin tables for Rotary Position Embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim // 2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to query and key tensors of shape (B, H, T, D)."""
    # Reshape cos/sin for broadcasting: (1, 1, T, D)
    cos = cos.unsqueeze(0).unsqueeze(0).to(device=q.device, dtype=q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(device=q.device, dtype=q.dtype)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

class CausalSelfAttention(nn.Module):
    def __init__(self, config: PixiConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_kv_cache = (k, v)

        # Dispatches to FlashAttention-2 or Memory-Efficient Attention on supported GPUs
        is_causal = (kv_cache is None) and (T > 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), new_kv_cache

class SwiGLUMLP(nn.Module):
    def __init__(self, config: PixiConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: PixiConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_kv_cache

class Pixi(nn.Module):
    def __init__(self, config: PixiConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope_cache(config.max_seq_len, config.head_dim, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Depth-scaled initialization for residual projections (1 / sqrt(2 * n_layers))
        residual_scale = 0.02 / math.sqrt(2 * config.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=residual_scale)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual_scale)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        B, T = input_ids.shape
        past_len = kv_caches[0][0].size(2) if (kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None) else 0
        assert past_len + T <= self.config.max_seq_len, "Sequence length exceeds RoPE cache bounds."

        cos = self.rope_cos[past_len : past_len + T].to(input_ids.device)
        sin = self.rope_sin[past_len : past_len + T].to(input_ids.device)

        x = self.embed_tokens(input_ids)
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, cos, sin, kv_cache=cache)
            new_kv_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return logits, loss, new_kv_caches



MicroChat = Pixi  # Backward-compatible alias

if __name__ == "__main__":


    cfg = PixiConfig()
    model = Pixi(cfg)
    total_params = model.count_parameters()
    print(f"Verified Pixi-40M Parameter Count: {total_params:,}")
    assert total_params == 40_512_000, f"Parameter mismatch! Expected 40,512,000, got {total_params}"