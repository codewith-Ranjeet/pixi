"""
Pre-training Script for Pixi-40M
Optimized for NVIDIA RTX 3050 6GB (Ampere)
"""
import math
import os
import sys
import time
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import Pixi, PixiConfig
from src.data_pipeline import ShardedDirectoryDataset
from src.trainer import build_optimizer, get_cosine_lr

def pretrain():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Running on: {device.upper()}")
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")

    # ----------------- Hyperparameters (Tuned for RTX 3050 6GB) -----------------
    seq_len = 1024            # Full 1024 token context window
    micro_batch_size = 8      # Micro-batch fits within ~2.5 GB VRAM
    grad_accum_steps = 8      # Effective batch size = 64 sequences (65,536 tokens / step)
    max_steps = 10000         # 10,000 steps * 65,536 tokens = ~655M tokens
    warmup_steps = 500        # Linear warmup
    max_lr = 6e-4
    min_lr = 6e-5
    save_every = 1000         # Checkpoint every 1,000 steps
    data_glob = os.path.join(PROJECT_ROOT, "data", "pretrain_bins", "shard_*.bin")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "pretrain")
    # ----------------------------------------------------------------------------

    # Instantiate Model
    config = PixiConfig(vocab_size=16384, d_model=512, n_layers=10, n_heads=8, d_ff=1408, max_seq_len=seq_len)
    model = Pixi(config).to(device)
    print(f"[Model] Initialized Pixi-40M: {model.count_parameters():,} parameters")

    optimizer = build_optimizer(model, lr=max_lr, weight_decay=0.1)

    # Ampere natively accelerates bfloat16
    autocast_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(autocast_dtype == torch.float16 and device == "cuda"))

    dataset = ShardedDirectoryDataset(data_glob, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True, drop_last=True)
    data_iter = iter(loader)

    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()
    running_loss = 0.0
    t0 = time.time()

    print(f"\n[Training] Starting Pre-training for {max_steps:,} steps...")
    for step in range(1, max_steps + 1):
        lr = get_cosine_lr(step - 1, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss, _ = model(x, labels=y)
                loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += accum_loss

        # Log progress every 25 steps
        if step % 25 == 0:
            avg_loss = running_loss / 25
            ppl = math.exp(min(avg_loss, 20.0))
            elapsed = time.time() - t0
            tok_per_sec = (25 * grad_accum_steps * micro_batch_size * seq_len) / elapsed
            print(f"Step {step:05d}/{max_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s")
            running_loss = 0.0
            t0 = time.time()

        # Save Checkpoint
        if step % save_every == 0 or step == max_steps:
            ckpt_file = os.path.join(checkpoint_dir, f"pixi_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "loss": accum_loss,
            }, ckpt_file)
            print(f"--> Saved checkpoint: {ckpt_file}")

    print("\nPre-training completed successfully!")

if __name__ == "__main__":
    pretrain()

    pretrain()