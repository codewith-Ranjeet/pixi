"""
Supervised Fine-Tuning (SFT / Chat Tuning) for Pixi-40M
Transforms the pre-trained base model into an interactive conversational AI.
"""

import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import Pixi, PixiConfig
from src.data_pipeline import ChatMLDataset
from src.trainer import build_optimizer, get_cosine_lr
from src.export_and_infer import export_to_gguf_q8_0, generate_chat
from tokenizers import Tokenizer

def run_sft(
    base_checkpoint=None,
    data_path=None,
    tokenizer_path=None,
    output_checkpoint=None,
    output_gguf=None,
    epochs=5,
    batch_size=4,
    learning_rate=3e-5,
    max_seq_len=256,
):
    if base_checkpoint is None:
        base_checkpoint = os.path.join(PROJECT_ROOT, "checkpoints", "pretrain", "pixi_step_10000.pt")
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "chat_sft.jsonl")
    if tokenizer_path is None:
        tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer", "tokenizer.json")
    if output_checkpoint is None:
        output_checkpoint = os.path.join(PROJECT_ROOT, "checkpoints", "pixi_chat_sft.pt")
    if output_gguf is None:
        output_gguf = os.path.join(PROJECT_ROOT, "models", "pixi-40m-q8_0.gguf")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("🤖 Pixi-40M: Supervised Instruction Fine-Tuning (SFT)")
    print("=" * 70)
    print(f"[Device] Running on: {device.upper()}")
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")

    # 1. Load Pre-trained Base Model
    print(f"\n[1/5] Loading pre-trained base model: {base_checkpoint}...")
    assert os.path.exists(base_checkpoint), f"Base checkpoint not found at {base_checkpoint}!"
    ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", PixiConfig(vocab_size=16384, d_model=512, n_layers=10, n_heads=8, d_ff=1408, max_seq_len=1024))
    model = Pixi(config).to(device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"  Successfully loaded base weights! Pre-training loss was: {ckpt.get('loss', 'N/A'):.4f}")

    # 2. Prepare Conversational Dataset with Masked Loss
    print(f"\n[2/5] Preparing ChatML conversational dataset from {data_path}...")
    dataset = ChatMLDataset(data_path, tokenizer_path=tokenizer_path, max_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_steps = len(loader) * epochs
    warmup_steps = int(0.1 * total_steps)

    print(f"  Loaded {len(dataset)} dialogue sequences | {epochs} epochs | {total_steps} total steps")
    print(f"  Prompt masking enabled: Loss is ONLY calculated on assistant responses.")

    # 3. Optimizer & Training Loop
    optimizer = build_optimizer(model, lr=learning_rate, weight_decay=0.01)
    autocast_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(autocast_dtype == torch.float16 and device == "cuda"))

    print(f"\n[3/5] Starting Fine-Tuning (Peak LR: {learning_rate:.1e})...")
    model.train()
    global_step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            global_step += 1
            lr = get_cosine_lr(global_step - 1, warmup_steps, total_steps, learning_rate, min_lr=learning_rate * 0.1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss, _ = model(x, labels=y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()

            if global_step % 10 == 0 or global_step == total_steps:
                print(f"  Epoch {epoch}/{epochs} | Step {global_step:03d}/{total_steps} | SFT Loss: {loss.item():.4f} | LR: {lr:.2e}")

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"--> Completed Epoch {epoch}/{epochs} | Average Loss: {avg_epoch_loss:.4f}")

    elapsed = time.time() - t0
    print(f"\n[4/5] Fine-Tuning completed in {elapsed:.1f}s ({elapsed/total_steps:.2f}s per step)")

    # 4. Save Fine-Tuned PyTorch Checkpoint
    os.makedirs(os.path.dirname(output_checkpoint), exist_ok=True)
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "loss": avg_epoch_loss
    }
    torch.save(save_dict, output_checkpoint)
    print(f"  Saved SFT checkpoint: {output_checkpoint} ({os.path.getsize(output_checkpoint)/(1024**2):.2f} MB)")

    # 5. Export to Quantized GGUF Container
    print(f"\n[5/5] Exporting fine-tuned model to GGUF Q8_0: {output_gguf}...")
    os.makedirs(os.path.dirname(output_gguf), exist_ok=True)
    export_to_gguf_q8_0(model, out_path=output_gguf)

    # 6. Quick Verification Prompt
    tok = Tokenizer.from_file(tokenizer_path)
    test_prompt = "<|im_start|>user\nHello! Who are you?<|im_end|>\n<|im_start|>assistant\n"
    test_ids = tok.encode(test_prompt).ids
    print(f"\n[Verification] Generating response to: '{test_prompt.strip()}'...")
    response_ids = generate_chat(model, test_ids, max_new_tokens=64, temperature=0.7, top_p=0.9, device=device)
    print(f"[Pixi Response]: {tok.decode(response_ids)}")
    print("=" * 70)
    print("🎉 SFT Training & GGUF Export Complete! You can now chat with Pixi!")
    print("=" * 70)

if __name__ == "__main__":
    run_sft()

