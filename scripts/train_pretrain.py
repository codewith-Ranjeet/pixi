"""
Pre-training Script for Pixi-40M
Optimized for NVIDIA RTX 3050 6GB (Ampere)
"""
import glob
import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from src.model import Pixi, PixiConfig, MicroChat, MicroChatConfig
except ImportError:
    from model import Pixi, PixiConfig, MicroChat, MicroChatConfig
try:
    from src.trainer import build_optimizer, get_cosine_lr
except ImportError:
    from train import build_optimizer, get_cosine_lr

class ShardedDirectoryDataset(Dataset):
    """Memory-mapped reader iterating over pre-tokenized binary shards."""
    def __init__(self, shard_glob: str, seq_len: int = 1024):
        self.seq_len = seq_len
        self.files = sorted(glob.glob(shard_glob))
        assert len(self.files) > 0, f"No binary shards found at {shard_glob}"

        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.shard_samples = [(len(s) - 1) // seq_len for s in self.shards]
        self.total_samples = sum(self.shard_samples)
        print(f"Loaded {len(self.files)} shards with {self.total_samples:,} sequences ({self.total_samples * seq_len:,} tokens).")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        curr = 0
        for shard, num_samples in zip(self.shards, self.shard_samples):
            if idx < curr + num_samples:
                start = (idx - curr) * self.seq_len
                end = start + self.seq_len + 1
                chunk = torch.from_numpy(shard[start:end].astype(np.int64))
                return chunk[:-1], chunk[1:]
            curr += num_samples
        raise IndexError("Sample index out of bounds")

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
    # ----------------------------------------------------------------------------

    # Instantiate Model
    config = PixiConfig(vocab_size=16384, d_model=512, n_layers=10, n_heads=8, d_ff=1408, max_seq_len=seq_len)
    model = Pixi(config).to(device)
    print(f"[Model] Initialized Pixi-40M: {model.count_parameters():,} parameters")

    optimizer = build_optimizer(model, lr=max_lr, weight_decay=0.1)

    # Ampere natively accelerates bfloat16
    autocast_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(autocast_dtype == torch.float16 and device == "cuda"))

    dataset = ShardedDirectoryDataset("data/pretrain_bins/shard_*.bin", seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True, drop_last=True)
    data_iter = iter(loader)

    os.makedirs("checkpoints/pretrain", exist_ok=True)
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
            ckpt_file = f"checkpoints/pretrain/pixi_step_{step}.pt"
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