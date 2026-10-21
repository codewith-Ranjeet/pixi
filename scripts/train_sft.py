"""Supervised fine-tuning for Pixi-40M using ChatML conversation data."""

import os
import sys
import time
import hashlib
import argparse
import random

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline import ChatMLDataset
from src.export_and_infer import export_to_gguf_q8_0, generate_chat
from src.model import Pixi, PixiConfig
from src.trainer import build_optimizer, get_cosine_lr


def run_sft(
    base_checkpoint=None,
    data_path=None,
    validation_data_path=None,
    tokenizer_path=None,
    output_checkpoint=None,
    output_gguf=None,
    epochs=1,
    batch_size=4,
    learning_rate=3e-5,
    max_seq_len=512,
    data_source="HuggingFaceH4/ultrachat_200k",
    seed=2026,
):
    base_checkpoint = base_checkpoint or os.path.join(PROJECT_ROOT, "checkpoints", "old_pretrain", "pixi_step_10000_session3.pt")
    data_path = data_path or os.path.join(PROJECT_ROOT, "data", "chat_sft_ultrachat_train.jsonl")
    validation_data_path = validation_data_path or os.path.join(PROJECT_ROOT, "data", "chat_sft_ultrachat_validation.jsonl")
    tokenizer_path = tokenizer_path or os.path.join(PROJECT_ROOT, "tokenizer", "tokenizer.json")
    output_checkpoint = output_checkpoint or os.path.join(PROJECT_ROOT, "checkpoints", "pixi_chat_sft_ultrachat.pt")
    for path in (base_checkpoint, data_path, validation_data_path, tokenizer_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if os.path.exists(output_checkpoint) or (output_gguf and os.path.exists(output_gguf)):
        raise FileExistsError("Refusing to overwrite an SFT checkpoint or GGUF; choose new output paths.")
    if epochs < 1 or batch_size < 1 or max_seq_len < 2:
        raise ValueError("epochs, batch_size, and max_seq_len must be positive (max_seq_len >= 2).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data_generator = torch.Generator().manual_seed(seed)
    print(f"[Device] {device.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"[Base] {base_checkpoint}")
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    with open(tokenizer_path, "rb") as tokenizer_file:
        tokenizer_hash = hashlib.sha256(tokenizer_file.read()).hexdigest()
    source_note = os.path.splitext(data_path)[0] + "_SOURCE.md"
    sft_revision = None
    if os.path.isfile(source_note):
        with open(source_note, "r", encoding="utf-8") as source_file:
            for line in source_file:
                if line.startswith("- Immutable dataset revision:"):
                    sft_revision = line.split("`", 2)[1]
                    break
    checkpoint_tokenizer_hash = ckpt.get("tokenizer_sha256")
    if checkpoint_tokenizer_hash and checkpoint_tokenizer_hash != tokenizer_hash:
        raise ValueError("Base checkpoint tokenizer hash differs from tokenizer/tokenizer.json.")
    config = ckpt.get("config", PixiConfig(vocab_size=16384, d_model=512, n_layers=10, n_heads=8, d_ff=1408, max_seq_len=1024))
    model = Pixi(config).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    train_data = ChatMLDataset(data_path, tokenizer_path=tokenizer_path, max_len=max_seq_len)
    val_data = ChatMLDataset(validation_data_path, tokenizer_path=tokenizer_path, max_len=max_seq_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, generator=data_generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False)
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Train and validation sets must each provide at least one batch.")
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    print(f"[Data] {len(train_data)} train, {len(val_data)} validation conversations; {epochs} epoch(s), {total_steps} steps")

    optimizer = build_optimizer(model, lr=learning_rate, weight_decay=0.01)
    amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp_dtype == torch.float16))
    global_step = 0
    final_train_loss = float("nan")
    validation_loss = float("nan")
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_token_count = 0
        for x, y in train_loader:
            lr = get_cosine_lr(global_step, warmup_steps, total_steps, learning_rate, min_lr=learning_rate * 0.1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
                _, loss, _ = model(x, labels=y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {global_step + 1}: {loss.item()}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            supervised_tokens = (y != -100).sum().item()
            train_sum += loss.item() * supervised_tokens
            train_token_count += supervised_tokens
            global_step += 1
            if global_step % 25 == 0 or global_step == total_steps:
                print(f"step {global_step}/{total_steps}: loss={loss.item():.4f}, lr={lr:.2e}")
        final_train_loss = train_sum / max(train_token_count, 1)

        model.eval()
        loss_sum = 0.0
        token_count = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
                    _, loss, _ = model(x, labels=y)
                n = (y != -100).sum().item()
                loss_sum += loss.item() * n
                token_count += n
        validation_loss = loss_sum / max(token_count, 1)
        print(f"epoch {epoch}: train loss={final_train_loss:.4f}; validation loss={validation_loss:.4f}")

    tok = Tokenizer.from_file(tokenizer_path)
    test_prompt = "<|im_start|>system\nYou are Pixi, a friendly and helpful English AI assistant.<|im_end|>\n<|im_start|>user\nHello! Who are you?<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(test_prompt).ids
    with torch.inference_mode():
        response = generate_chat(model, ids, max_new_tokens=64, temperature=0.0, top_p=0.9, device=device)
    print(f"[Sample] {tok.decode(response)}")

    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "loss": final_train_loss,
        "validation_loss": validation_loss,
        "tokenizer_sha256": tokenizer_hash,
        "sft_data": data_source,
        "sft_data_revision": sft_revision,
        "train_examples": len(train_data),
        "validation_examples": len(val_data),
        "base_checkpoint": os.path.abspath(base_checkpoint),
        "sft_epochs": epochs,
        "sft_learning_rate": learning_rate,
        "sft_batch_size": batch_size,
        "sft_max_seq_len": max_seq_len,
        "sft_seed": seed,
    }, output_checkpoint)
    print(f"[Saved] {output_checkpoint}")
    if output_gguf:
        os.makedirs(os.path.dirname(output_gguf), exist_ok=True)
        export_to_gguf_q8_0(model, out_path=output_gguf)
        print(f"[Saved] {output_gguf}")
    print(f"[Time] {time.time() - started:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for Pixi-40M.")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--validation-data", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--output-gguf", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--data-source", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run_sft(
        base_checkpoint=args.base_checkpoint,
        data_path=args.data,
        validation_data_path=args.validation_data,
        tokenizer_path=args.tokenizer,
        output_checkpoint=args.output_checkpoint,
        output_gguf=args.output_gguf,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        data_source=args.data_source,
        seed=args.seed,
    )
