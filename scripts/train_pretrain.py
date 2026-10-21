"""Pretrain Pixi from checked token shards with held-out loss tracking."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import ShardedDirectoryDataset
from src.model import Pixi, PixiConfig
from src.trainer import build_optimizer, get_cosine_lr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_shards(files: list[Path], tokenizer: Tokenizer) -> None:
    """Check token ranges and a few decoded samples before spending GPU time."""
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise ValueError("Tokenizer does not define <|eos|>.")
    for path in files:
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        if not len(tokens):
            raise ValueError(f"Empty token shard: {path}")
        maximum = 0
        for start in range(0, len(tokens), 4_000_000):
            maximum = max(maximum, int(tokens[start : start + 4_000_000].max()))
        if maximum >= vocab_size:
            raise ValueError(f"{path.name} contains token ID {maximum}, outside vocabulary size {vocab_size}.")

        # A wrong tokenizer often turns natural-language shards into punctuation/noise.
        starts = sorted({min(max(0, len(tokens) // 3), max(0, len(tokens) - 256)),
                         min(max(0, 2 * len(tokens) // 3), max(0, len(tokens) - 256))})
        for start in starts:
            sample = tokenizer.decode(tokens[start : start + 256].astype(np.int64).tolist())
            visible = sum(ch.isprintable() or ch in "\r\n\t" for ch in sample)
            if not sample or visible / len(sample) < 0.90:
                raise ValueError(
                    f"Shard {path.name} does not decode as readable text with the active tokenizer. "
                    "Rebuild token shards with tokenizer/tokenizer.json before training."
                )
    print(f"[Data check] {len(files)} shards have in-range IDs and decode as readable text.")


def evaluate(model, loader, device, autocast_dtype, max_batches):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=device == "cuda"):
                _, loss, _ = model(x, labels=y)
            n = y.numel()
            loss_sum += loss.item() * n
            token_count += n
    model.train()
    if not token_count:
        raise ValueError("Validation loader produced no tokens.")
    return loss_sum / token_count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "pretrain_bins"))
    parser.add_argument("--tokenizer", default=str(PROJECT_ROOT / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints" / "pretrain_next"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Resume an interrupted run from its checkpoint.")
    parser.add_argument("--check-only", action="store_true", help="Validate tokenizer/shards without initializing or training a model.")
    return parser.parse_args()


def pretrain(args=None):
    args = args or parse_args()
    data_dir = Path(args.data_dir).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if min(args.steps, args.save_every, args.eval_every, args.eval_batches,
           args.micro_batch_size, args.grad_accum_steps, args.seq_len) <= 0:
        raise ValueError("Training sizes, step counts, and intervals must be positive.")
    if not tokenizer_path.is_file() or not data_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer or data directory not found: {tokenizer_path} / {data_dir}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_ids = {name: tokenizer.token_to_id(name) for name in
                 ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|im_start|>", "<|im_end|>")}
    if any(value is None for value in token_ids.values()):
        raise ValueError(f"Tokenizer is missing required control tokens: {token_ids}")
    if tokenizer.get_vocab_size() != 16_384:
        raise ValueError(f"Expected 16,384 tokenizer entries, got {tokenizer.get_vocab_size()}.")

    all_files = sorted(data_dir.glob("shard_*.bin"))
    usable_files = [p for p in all_files if p.stat().st_size // 2 > args.seq_len]
    if len(usable_files) < 2:
        raise ValueError("Need at least two full shards so one can be held out for validation.")
    manifest_path = data_dir / "manifest.json"
    tokenizer_hash = sha256_file(tokenizer_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError("Shard manifest tokenizer hash differs from the active tokenizer; rebuild the shards.")
        named_validation = manifest.get("validation_shard")
        validation_file = data_dir / named_validation if named_validation else usable_files[-1]
        if validation_file not in usable_files:
            raise ValueError(f"Manifest validation shard is missing or too small: {validation_file}")
    else:
        validation_file = usable_files[-1]
        print("[Data check] Legacy shard set has no tokenizer manifest; using the last full shard as hold-out.")
    inspect_shards(usable_files, tokenizer)

    train_dataset = ShardedDirectoryDataset(
        str(data_dir / "shard_*.bin"), seq_len=args.seq_len, exclude_files=[str(validation_file)]
    )
    validation_dataset = ShardedDirectoryDataset(str(validation_file), seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.micro_batch_size,
                              shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation_dataset, batch_size=args.micro_batch_size,
                                   shuffle=False, drop_last=False, pin_memory=torch.cuda.is_available())
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("The selected train/validation shards do not contain complete batches.")
    print(f"[Data] train={len(train_dataset):,} sequences from {len(train_dataset.files)} shards; "
          f"validation={len(validation_dataset):,} sequences from {validation_file.name}")
    if args.check_only:
        print("[Ready] Data/tokenizer checks passed. No model was initialized and no training was run.")
        return

    run_config = {
        "steps": args.steps, "save_every": args.save_every, "eval_every": args.eval_every,
        "eval_batches": args.eval_batches, "micro_batch_size": args.micro_batch_size,
        "grad_accum_steps": args.grad_accum_steps, "seq_len": args.seq_len,
        "seed": args.seed, "tokenizer_sha256": tokenizer_hash,
        "train_shards": [p.name for p in train_dataset.files],
        "validation_shard": validation_file.name,
    }
    start_step = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = Path(args.resume).resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        checkpoint_dir = resume_checkpoint.parent
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        saved_run = checkpoint.get("run_config", {})
        for key in ("steps", "micro_batch_size", "grad_accum_steps", "seq_len", "tokenizer_sha256", "train_shards", "validation_shard"):
            if saved_run.get(key) != run_config.get(key):
                raise ValueError(f"Cannot resume: {key} differs from the saved run configuration.")
        start_step = int(checkpoint["step"])
        if start_step >= args.steps:
            raise ValueError(f"Checkpoint is already at step {start_step}; requested total is {args.steps}.")
    elif checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"Checkpoint directory is non-empty; refusing to overwrite: {checkpoint_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PixiConfig(vocab_size=tokenizer.get_vocab_size(), d_model=512, n_layers=10,
                        n_heads=8, d_ff=1408, max_seq_len=args.seq_len)
    model = Pixi(config).to(device)
    if resume_checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = build_optimizer(model, lr=6e-4, weight_decay=0.1)
    if resume_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    max_lr, min_lr = 6e-4, 6e-5
    warmup_steps = min(500, max(1, int(0.05 * args.steps)))
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir / "metrics.csv"
    if not resume_checkpoint and metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {metrics_path}")
    if resume_checkpoint and not metrics_path.exists():
        raise FileNotFoundError(f"Cannot resume without the run metrics file: {metrics_path}")
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        if start_step == 0:
            metrics_writer.writerow(["step", "train_loss", "validation_loss", "learning_rate", "tokens_seen"])
            metrics_file.flush()

        print(f"[Device] {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
        print(f"[Model] {model.count_parameters():,} parameters | vocab {tokenizer.get_vocab_size()} | seq {args.seq_len}")
        print(f"[Tokens] planned={args.steps * args.micro_batch_size * args.grad_accum_steps * args.seq_len:,}; "
              f"tokenizer SHA-256={tokenizer_hash}")

        model.train()
        data_iter = iter(train_loader)
        last_val = float("nan")
        running = 0.0
        window_steps = 0
        started = time.time()
        for step in range(start_step + 1, args.steps + 1):
            lr = get_cosine_lr(step - 1, warmup_steps, args.steps, max_lr, min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(args.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                    _, loss, _ = model(x, labels=y)
                    loss = loss / args.grad_accum_steps
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}.")
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step_loss += loss.item()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            running += step_loss
            window_steps += 1
            should_log = step % 25 == 0 or step == args.steps
            should_eval = step % args.eval_every == 0 or step == args.steps
            if should_eval:
                last_val = evaluate(model, validation_loader, device, dtype, args.eval_batches)
            if should_log or should_eval:
                avg_loss = running / max(window_steps, 1)
                tokens_seen = step * args.micro_batch_size * args.grad_accum_steps * args.seq_len
                print(f"step {step:05d}/{args.steps} | train {avg_loss:.4f} | "
                      f"validation {last_val:.4f} | lr {lr:.2e} | tokens {tokens_seen:,}")
                if should_eval:
                    metrics_writer.writerow([step, f"{avg_loss:.6f}", f"{last_val:.6f}", f"{lr:.9g}", tokens_seen])
                    metrics_file.flush()
                running = 0.0
                window_steps = 0

            if step % args.save_every == 0 or step == args.steps:
                checkpoint_path = checkpoint_dir / f"pixi_step_{step}.pt"
                if checkpoint_path.exists():
                    raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
                temporary_path = checkpoint_path.with_suffix(".pt.tmp")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "loss": step_loss,
                    "validation_loss": last_val,
                    "tokenizer_sha256": tokenizer_hash,
                    "run_config": run_config,
                }, temporary_path)
                os.replace(temporary_path, checkpoint_path)
                print(f"[Checkpoint] {checkpoint_path}")
        print(f"[Done] elapsed {time.time() - started:.1f}s; metrics: {metrics_path}")


if __name__ == "__main__":
    pretrain()
