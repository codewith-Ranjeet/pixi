import math
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from .model import Pixi, PixiConfig, MicroChat, MicroChatConfig
except ImportError:
    from model import Pixi, PixiConfig, MicroChat, MicroChatConfig

def build_optimizer(model: Pixi, lr: float = 6e-4, weight_decay: float = 0.1) -> torch.optim.AdamW:
    """Separates 2D projection matrices (decayed) from 1D RMSNorm parameters (non-decayed)."""
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    use_fused = torch.cuda.is_available() and any(p.is_cuda for p in model.parameters())
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

def get_cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train_causal_or_sft(
    model: Pixi,
    dataloader: DataLoader,
    max_steps: int = 10000,
    grad_accum_steps: int = 4,
    max_lr: float = 6e-4,
    device: Optional[str] = None,
):
    """Unified loop for Pre-training and SFT (SFT simply passes -100 masked targets via ChatMLDataset)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    optimizer = build_optimizer(model, lr=max_lr)
    warmup_steps = int(0.05 * max_steps)
    min_lr = max_lr * 0.1

    data_iter = iter(dataloader)
    for step in range(max_steps):
        lr = get_cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                _, loss, _ = model(x, labels=y)
                loss = loss / grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:05d} | Loss: {accum_loss:.4f} | LR: {lr:.2e}")

def compute_sequence_logprobs(model: Pixi, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes summed log-probabilities over non-masked response tokens for DPO."""
    logits, _, _ = model(input_ids)
    if labels.size(1) == input_ids.size(1):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
    else:
        shift_logits = logits[:, :labels.size(1), :].contiguous()
        shift_labels = labels.contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Replace -100 with 0 for gather, then zero out masked positions
    valid_mask = (shift_labels != -100)
    safe_labels = torch.where(valid_mask, shift_labels, torch.zeros_like(shift_labels))
    token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * valid_mask).sum(dim=-1)

def dpo_train_step(
    policy_model: Pixi,
    ref_model: Pixi,
    optimizer: torch.optim.Optimizer,
    chosen_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
) -> float:
    """Executes a single Direct Preference Optimization (DPO) update step."""
    policy_model.train()
    ref_model.eval()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=chosen_ids.device.type, dtype=torch.bfloat16):
        policy_chosen_logps = compute_sequence_logprobs(policy_model, chosen_ids, chosen_labels)
        policy_rejected_logps = compute_sequence_logprobs(policy_model, rejected_ids, rejected_labels)

        with torch.no_grad():
            ref_chosen_logps = compute_sequence_logprobs(ref_model, chosen_ids, chosen_labels)
            ref_rejected_logps = compute_sequence_logprobs(ref_model, rejected_ids, rejected_labels)

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        loss = -F.logsigmoid(beta * logits).mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()