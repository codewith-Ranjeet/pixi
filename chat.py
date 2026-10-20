"""
Interactive Terminal Chat Interface for Pixi-40M
Supports multi-turn conversation, streaming token generation, and customizable parameters.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.model import Pixi, PixiConfig

DEFAULT_SYSTEM_PROMPT = "You are Pixi, a friendly, concise, and intelligent English AI assistant."

def load_pixi_model(checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'!")

    print(f"Loading weights from: {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", PixiConfig(
        vocab_size=16384,
        d_model=512,
        n_layers=10,
        n_heads=8,
        d_ff=1408,
        max_seq_len=1024,
    ))
    model = Pixi(config).to(device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, config

@torch.inference_mode()
def stream_generate(
    model: Pixi,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cpu",
):
    """Generates tokens autoregressively with KV caching and yields decoded text chunks."""
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, _, kv_caches = model(input_tensor)
    next_token_logits = logits[:, -1, :]

    im_end_id = tokenizer.token_to_id("<|im_end|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    generated_ids = []

    for _ in range(max_new_tokens):
        if temperature > 0.0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

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
        if token_id in (im_end_id, eos_id):
            break

        generated_ids.append(token_id)
        # Yield decoded text token
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        yield token_str

        # KV-cached next step
        logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]

    return generated_ids

def main():
    parser = argparse.ArgumentParser(description="Chat interactively with Pixi-40M in your terminal.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling (default: 0.9)")
    parser.add_argument("--max_tokens", type=int, default=150, help="Max new tokens per reply")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Select checkpoint: Prefer SFT checkpoint, fall back to pre-trained checkpoint
    checkpoint_path = args.checkpoint
    is_base_model = False
    if checkpoint_path is None:
        if os.path.exists("checkpoints/pixi_chat_sft.pt"):
            checkpoint_path = "checkpoints/pixi_chat_sft.pt"
        elif os.path.exists("checkpoints/pretrain/pixi_step_10000.pt"):
            checkpoint_path = "checkpoints/pretrain/pixi_step_10000.pt"
            is_base_model = True
        else:
            print("❌ No checkpoint found in checkpoints/! Run pre-training or SFT first.")
            sys.exit(1)

    print("=" * 65)
    print("  ✨ Welcome to Pixi-40M Interactive Chat ✨")
    print("=" * 65)
    print(f"• Hardware: {device.upper()}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"• Model: {checkpoint_path}")
    if is_base_model:
        print("  [Note: This is the raw base pre-trained model. Run 'python run_sft.py'")
        print("   to fine-tune Pixi into a dedicated conversational assistant!]")
    print(f"• Sampling: Temperature={args.temp}, Top-P={args.top_p}")
    print("• Commands: /clear (reset chat), /temp <val>, /exit or /quit")
    print("=" * 65)

    # Load tokenizer and model
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model, config = load_pixi_model(checkpoint_path, device=device)

    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    history = []
    temp = args.temp
    top_p = args.top_p

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit"):
            print("Goodbye!")
            break

        if user_input.lower() in ("/clear", "/reset"):
            history.clear()
            print("🧹 Conversation reset.")
            continue

        if user_input.lower().startswith("/temp"):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    temp = float(parts[1])
                    print(f"⚙️ Temperature set to {temp}")
                except ValueError:
                    print("Invalid temperature. Example: /temp 0.7")
            else:
                print(f"Current temperature: {temp}")
            continue

        if user_input.lower() == "/help":
            print("Commands:")
            print("  /clear      - Reset conversation history")
            print("  /temp <val> - Change sampling temperature")
            print("  /exit       - Quit the chat")
            continue

        # Add to history
        history.append({"role": "user", "content": user_input})

        # Format full conversation with ChatML
        prompt = f"{im_start}system\n{DEFAULT_SYSTEM_PROMPT}{im_end}\n"
        for turn in history:
            prompt += f"{im_start}{turn['role']}\n{turn['content']}{im_end}\n"
        prompt += f"{im_start}assistant\n"

        prompt_ids = tokenizer.encode(prompt).ids

        # Safeguard sequence length
        if len(prompt_ids) + args.max_tokens > config.max_seq_len:
            # Drop older turns if exceeding context
            while len(history) > 1 and len(prompt_ids) + args.max_tokens > config.max_seq_len:
                history.pop(0)
                prompt = f"{im_start}system\n{DEFAULT_SYSTEM_PROMPT}{im_end}\n"
                for turn in history:
                    prompt += f"{im_start}{turn['role']}\n{turn['content']}{im_end}\n"
                prompt += f"{im_start}assistant\n"
                prompt_ids = tokenizer.encode(prompt).ids

        print("\nPixi > ", end="", flush=True)
        assistant_reply = ""
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=args.max_tokens,
            temperature=temp,
            top_p=top_p,
            device=device,
        ):
            print(chunk, end="", flush=True)
            assistant_reply += chunk
        print()

        history.append({"role": "assistant", "content": assistant_reply.strip()})

if __name__ == "__main__":
    main()

