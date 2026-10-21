"""
Interactive Terminal Chat Interface for Pixi-40M
Uses multi-turn ChatML prompts matching Pixi's prepared SFT conversation data.
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
from src.chat_tools import (
    CalculatorError,
    calculate_query,
    format_fraction,
    get_builtin_response,
    load_fact_store,
    looks_like_fact_query,
    retrieve_fact,
)

DEFAULT_SYSTEM_PROMPT = "You are Pixi, a friendly and helpful English AI assistant."

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
    emitted_text = ""

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
        # Decode the full byte-level BPE prefix. Decoding each ID separately
        # can split UTF-8 characters and emit replacement glyphs (for example,
        # Spanish accents). Hold any incomplete trailing character until the
        # next token completes it.
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
        stable_text = decoded.rstrip("\ufffd")
        if stable_text.startswith(emitted_text):
            chunk = stable_text[len(emitted_text):]
            if chunk:
                yield chunk
                emitted_text = stable_text

        # KV-cached next step
        logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]

    if generated_ids:
        final_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        if final_text.startswith(emitted_text):
            tail = final_text[len(emitted_text):]
            if tail:
                yield tail
    return generated_ids

def main():
    parser = argparse.ArgumentParser(description="Chat interactively with Pixi-40M in your terminal.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0, greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling (default: 0.9)")
    parser.add_argument("--max_tokens", type=int, default=150, help="Max new tokens per reply")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--facts", type=str, default="data/facts.json", help="Local JSON fact library")
    parser.add_argument("--no-facts", action="store_true", help="Disable local fact retrieval")
    parser.add_argument(
        "--allow-unverified-facts",
        action="store_true",
        help="Let Pixi attempt factual questions that have no local fact match (may be wrong)",
    )
    parser.add_argument("--no-calculator", action="store_true", help="Disable the safe arithmetic calculator")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prefer the latest dataset-specific SFT checkpoint, then legacy models.
    checkpoint_path = args.checkpoint
    is_base_model = False
    if checkpoint_path is None:
        if os.path.exists("checkpoints/pixi_chat_sft_session10_final_v9.pt"):
            checkpoint_path = "checkpoints/pixi_chat_sft_session10_final_v9.pt"
        elif os.path.exists("checkpoints/pixi_chat_sft_ultrachat.pt"):
            checkpoint_path = "checkpoints/pixi_chat_sft_ultrachat.pt"
        elif os.path.exists("checkpoints/pixi_chat_sft_pretrain_next.pt"):
            checkpoint_path = "checkpoints/pixi_chat_sft_pretrain_next.pt"
        elif os.path.exists("checkpoints/pixi_chat_sft_oasst1.pt"):
            checkpoint_path = "checkpoints/pixi_chat_sft_oasst1.pt"
        elif os.path.exists("checkpoints/old_pretrain/pixi_step_10000_session3.pt"):
            checkpoint_path = "checkpoints/old_pretrain/pixi_step_10000_session3.pt"
            is_base_model = True
        elif os.path.exists("checkpoints/pretrain/pixi_step_10000.pt"):
            checkpoint_path = "checkpoints/pretrain/pixi_step_10000.pt"
            is_base_model = True
        else:
            print("❌ No checkpoint found in checkpoints/! Run pre-training or SFT first.")
            sys.exit(1)

    facts = [] if args.no_facts else load_fact_store(args.facts)

    print("=" * 65)
    print("  ✨ Welcome to Pixi-40M Interactive Chat ✨")
    print("=" * 65)
    print(f"• Hardware: {device.upper()}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"• Model: {checkpoint_path}")
    tool_status = []
    if not args.no_calculator:
        tool_status.append("safe calculator")
    if not args.no_facts:
        tool_status.append(f"local fact lookup ({len(facts)} entries)")
    print(f"• Tools: {', '.join(tool_status) if tool_status else 'disabled'}")
    if is_base_model:
        print("  [Note: This is the raw base pre-trained model. Run 'python run_sft.py'")
        print("   to fine-tune Pixi into a dedicated conversational assistant!]")
    print(f"• Sampling: Temperature={args.temp}, Top-P={args.top_p}")
    print("• Commands: /clear (reset conversation), /temp <val>, /exit or /quit")
    print("=" * 65)

    # Load tokenizer and model
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model, config = load_pixi_model(checkpoint_path, device=device)

    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    temp = args.temp
    top_p = args.top_p
    history = []

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
            print("Conversation reset.")
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
            print("  /clear      - Clear conversation context")
            print("  /temp <val> - Change sampling temperature")
            print("  /exit       - Quit the chat")
            print("Tools:")
            print("  Basic arithmetic is calculated locally.")
            print("  Matching facts are returned from the local fact file with a source.")
            print("  Unmatched factual questions are declined unless --allow-unverified-facts is set.")
            continue

        builtin_response = get_builtin_response(user_input)
        if builtin_response is not None:
            history.append(("user", user_input))
            print(f"\nPixi > {builtin_response}")
            history.append(("assistant", builtin_response))
            continue

        if not args.no_calculator:
            try:
                calculation = calculate_query(user_input)
            except CalculatorError as exc:
                history.append(("user", user_input))
                tool_response = f"I can't calculate that safely: {exc}"
                print(f"\nPixi 🧮 > {tool_response}")
                history.append(("assistant", tool_response))
                continue
            if calculation is not None:
                expression, value = calculation
                answer = format_fraction(value)
                history.append(("user", user_input))
                tool_response = f"{expression} = {answer} (calculated locally)"
                print(f"\nPixi 🧮 > {tool_response}")
                history.append(("assistant", tool_response))
                continue

        matched_fact = retrieve_fact(user_input, facts) if facts else None
        if matched_fact is not None:
            history.append(("user", user_input))
            tool_response = (
                f"{matched_fact['answer']}\n"
                f"Source: {matched_fact['source_title']} — {matched_fact['source_url']}"
            )
            print(f"\nPixi 🔎 > {tool_response}")
            history.append(("assistant", tool_response))
            continue

        unverified_fact = bool(facts) and looks_like_fact_query(user_input)

        if unverified_fact and not args.allow_unverified_facts:
            history.append(("user", user_input))
            tool_response = (
                "I don't have a verified local fact for that question, so I won't guess. "
                f"You can add a sourced entry to {args.facts}, or rerun with "
                "--allow-unverified-facts to let Pixi try an unverified answer."
            )
            print(f"\nPixi 🔎 > {tool_response}")
            history.append(("assistant", tool_response))
            continue

        history.append(("user", user_input))
        generation_budget = max(1, min(args.max_tokens, config.max_seq_len - 1))
        max_prompt_tokens = config.max_seq_len - generation_budget
        system_ids = tokenizer.encode(f"{im_start}system\n{DEFAULT_SYSTEM_PROMPT}{im_end}\n").ids
        # Keep the most recent complete turns that fit the context window.
        kept_turns = []
        for user_text, assistant_text in reversed([
            (history[i][1], history[i + 1][1])
            for i in range(0, len(history) - 1, 2)
            if history[i][0] == "user" and history[i + 1][0] == "assistant"
        ]):
            turn = f"{im_start}user\n{user_text}{im_end}\n{im_start}assistant\n{assistant_text}{im_end}\n"
            candidate = tokenizer.encode(turn).ids
            if len(system_ids) + sum(len(ids) for ids in kept_turns) + len(candidate) > max_prompt_tokens:
                break
            kept_turns.insert(0, candidate)
        current_user_ids = tokenizer.encode(f"{im_start}user\n{user_input}{im_end}\n{im_start}assistant\n").ids
        while kept_turns and len(system_ids) + sum(len(ids) for ids in kept_turns) + len(current_user_ids) > max_prompt_tokens:
            kept_turns.pop(0)
        if len(system_ids) + len(current_user_ids) > max_prompt_tokens:
            prefix = tokenizer.encode(f"{im_start}user\n").ids
            suffix = tokenizer.encode(f"{im_end}\n{im_start}assistant\n").ids
            content = tokenizer.encode(user_input).ids
            budget = max(0, max_prompt_tokens - len(system_ids) - len(prefix) - len(suffix))
            current_user_ids = prefix + (content[-budget:] if budget else []) + suffix
        prompt_ids = system_ids + [token for turn_ids in kept_turns for token in turn_ids] + current_user_ids

        print("\nPixi > ", end="", flush=True)
        if unverified_fact:
            print("[No verified local match; this model reply is unverified.] ", end="", flush=True)
        response_parts = []
        for chunk in stream_generate(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=generation_budget,
            temperature=temp,
            top_p=top_p,
            device=device,
        ):
            response_parts.append(chunk)
            print(chunk, end="", flush=True)
        print()
        history.append(("assistant", "".join(response_parts)))


if __name__ == "__main__":
    main()

