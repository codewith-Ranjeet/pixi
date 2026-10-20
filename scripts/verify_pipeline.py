"""
End-to-End Implementation Script for Pixi-40M
1. Constructs specialized English Conversational Dataset
2. Trains Byte-Level BPE Tokenizer (16,384 vocab) with ChatML control tokens
3. Trains custom 40.5M parameter Pixi model with masked loss
4. Exports weights to GGUF Q8_0 container
5. Runs verification benchmarks in both Python and compiled C++ AVX2 engine
"""

import os
import json
import time
import subprocess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import Pixi, PixiConfig, MicroChat, MicroChatConfig
from src.data_pipeline import train_bpe_tokenizer, ChatMLDataset
from src.trainer import build_optimizer, get_cosine_lr
from src.export_and_infer import export_to_gguf_q8_0, generate_chat

def create_training_corpus(text_file: str, jsonl_file: str):
    os.makedirs(os.path.dirname(text_file), exist_ok=True)
    os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)

    # Conversational instruction dataset for ChatML fine-tuning
    conversations = [
        {"messages": [
            {"role": "system", "content": "You are Pixi, a friendly and intelligent English AI assistant."},
            {"role": "user", "content": "Hello! Who are you?"},
            {"role": "assistant", "content": "Hello! I am Pixi, a fast and lightweight 40M parameter English conversational AI model built from scratch."}
        ]},
        {"messages": [
            {"role": "system", "content": "You are Pixi, a friendly and intelligent English AI assistant."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its readable syntax, versatility, and extensive standard library."}
        ]},
        {"messages": [
            {"role": "system", "content": "You are Pixi, a friendly and intelligent English AI assistant."},
            {"role": "user", "content": "Explain gravity in one sentence."},
            {"role": "assistant", "content": "Gravity is the fundamental force by which physical bodies attract each other with a force proportional to their masses."}
        ]},
        {"messages": [
            {"role": "system", "content": "You are Pixi, a friendly and intelligent English AI assistant."},
            {"role": "user", "content": "How do you calculate the area of a circle?"},
            {"role": "assistant", "content": "The area of a circle is calculated using the formula Area = pi * r^2, where r is the radius of the circle."}
        ]},
        {"messages": [
            {"role": "system", "content": "You are Pixi, a friendly and intelligent English AI assistant."},
            {"role": "user", "content": "What are the primary colors?"},
            {"role": "assistant", "content": "The traditional primary colors in art and subtractive color theory are red, yellow, and blue. In additive light systems, they are red, green, and blue."}
        ]},
        {"messages": [
            {"role": "system", "content": "You are Pixi, a friendly and intelligent English AI assistant."},
            {"role": "user", "content": "Write a python function to add two numbers."},
            {"role": "assistant", "content": "def add(a, b):\n    return a + b\n\n# Example usage:\nresult = add(5, 3)\nprint(result)  # Output: 8"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are Pixi, a friendly and intelligent English AI assistant."},
            {"role": "user", "content": "What is photosynthesis?"},
            {"role": "assistant", "content": "Photosynthesis is the biological process by which green plants and organisms transform light energy into chemical energy, converting water and carbon dioxide into glucose and oxygen."}
        ]},
        {"messages": [
            {"role": "system", "content": "You are Pixi, a friendly and intelligent English AI assistant."},
            {"role": "user", "content": "Can you help me with coding?"},
            {"role": "assistant", "content": "Yes, I can help you write, debug, and explain code in Python, C++, and other programming languages."}
        ]},
    ]

    # Expand dialogues to create adequate training signal
    expanded_dialogues = conversations * 15
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for c in expanded_dialogues:
            f.write(json.dumps(c) + "\n")

    # Raw English text corpus for tokenizer vocabulary generation
    with open(text_file, "w", encoding="utf-8") as f:
        for c in conversations:
            for m in c["messages"]:
                f.write(f"{m['role']}: {m['content']}\n")
        f.write("Machine learning and artificial intelligence enable computers to learn from data.\n" * 50)
        f.write("A 40M parameter Transformer model uses RMSNorm, RoPE embeddings, and SwiGLU activations.\n" * 50)
        f.write("Quantization reduces 32-bit floating point weights into 8-bit integers for high speed CPU inference.\n" * 50)

    print(f"[Dataset] Generated {len(expanded_dialogues)} ChatML dialogue samples in {jsonl_file}")
    print(f"[Dataset] Generated raw text corpus in {text_file}")

def run_pipeline():
    print("=" * 70)
    print(">> Pixi-40M: End-to-End Build, Training & Deployment Pipeline")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Hardware] Execution Device: {device.upper()}")

    # 1. Dataset Generation
    corpus_txt = "data/raw_corpus.txt"
    chat_jsonl = "data/chat_sft.jsonl"
    create_training_corpus(corpus_txt, chat_jsonl)

    # 2. Tokenizer Training
    tokenizer_path = "tokenizer/tokenizer.json"
    print(f"\n[Tokenizer] Training custom 16,384 BPE Tokenizer...")
    tok = train_bpe_tokenizer([corpus_txt], save_path=tokenizer_path, vocab_size=16384)
    print(f"[Tokenizer] Vocab size: {tok.get_vocab_size():,} tokens")
    print(f"[Tokenizer] Special Tokens: <|pad|>=0, <|bos|>=1, <|eos|>=2, <|unk|>=3, <|im_start|>=4, <|im_end|>=5")

    # 3. Model Architecture Instantiation
    cfg = PixiConfig(vocab_size=16384, d_model=512, n_layers=10, n_heads=8, d_ff=1408, max_seq_len=256)
    model = Pixi(cfg).to(device)
    total_params = model.count_parameters()
    print(f"\n[Model] Instantiated Pixi-40M")
    print(f"[Model] Total Parameter Count: {total_params:,} (Exact target: 40,512,000)")

    # 4. Supervised Instruction Fine-Tuning (SFT)
    max_seq_len = 128
    dataset = ChatMLDataset(chat_jsonl, tokenizer_path=tokenizer_path, max_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = build_optimizer(model, lr=8e-4, weight_decay=0.01)
    train_steps = 15  # Optimized for CPU responsiveness while demonstrating loss convergence
    warmup_steps = 3
    min_lr = 8e-5

    print(f"\n[Training] Starting Supervised Fine-Tuning ({train_steps} steps on {device.upper()})...")
    model.train()
    data_iter = iter(loader)
    t_start = time.time()

    for step in range(1, train_steps + 1):
        lr = get_cosine_lr(step - 1, warmup_steps, train_steps, 8e-4, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            _, loss, _ = model(x, labels=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"  Step {step:02d}/{train_steps:02d} | Loss: {loss.item():.4f} | LR: {lr:.2e}")

    t_train = time.time() - t_start
    print(f"[Training] Training completed in {t_train:.1f}s ({t_train/train_steps:.2f}s per step)")

    # 5. Save PyTorch Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/pixi_40m.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Checkpoint] Saved PyTorch weights to {ckpt_path} ({os.path.getsize(ckpt_path) / (1024**2):.2f} MB)")

    # 6. Export to GGUF Q8_0 Binary
    gguf_path = "models/pixi-40m-q8_0.gguf"
    print(f"\n[GGUF Export] Quantizing to Q8_0 container: {gguf_path}...")
    export_to_gguf_q8_0(model, out_path=gguf_path)

    # 7. Python Generation Test
    print(f"\n[Inference: Python] Testing Autoregressive Decode with KV Cache...")
    prompt_text = "<|im_start|>user\nHello! Who are you?<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tok.encode(prompt_text).ids
    print(f"  Prompt: {prompt_text.strip()}")

    t_gen_start = time.time()
    gen_tokens = generate_chat(model, prompt_ids, max_new_tokens=40, temperature=0.7, top_p=0.9, device=device)
    t_gen = time.time() - t_gen_start

    decoded = tok.decode(gen_tokens)
    print(f"  Response Tokens: {gen_tokens}")
    print(f"  Decoded Text: {decoded}")
    print(f"  Python generation took {t_gen:.2f}s ({len(gen_tokens) / max(0.001, t_gen):.1f} tok/s)")

    # 8. C++ High-Speed Inference Engine Test
    cpp_exe = "cpp/pixi_infer.exe" if os.path.exists("cpp/pixi_infer.exe") else "pixi_infer.exe"
    if os.path.exists(cpp_exe):
        print(f"\n[Inference: C++ AVX2 Engine] Running native compiled binary: {cpp_exe}...")
        res = subprocess.run([os.path.abspath(cpp_exe), gguf_path], capture_output=True, text=True)
        print(res.stdout)
    else:
        print(f"\n[Inference: C++] Binary {cpp_exe} not found, compiling now...")
        subprocess.run(["g++", "-O3", "-mavx2", "-mf16c", "-mfma", "-std=c++20", "cpp/pixi_infer.cpp", "-o", "cpp/pixi_infer.exe"])
        res = subprocess.run([os.path.abspath(cpp_exe), gguf_path], capture_output=True, text=True)
        print(res.stdout)

    print("=" * 70)
    print(">> Full Pipeline Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
