"""
Train a real Byte-Level BPE tokenizer for Pixi using FineWeb-Edu.

Run from the Pixi project root:
    python scripts/train_tokenizer.py

This writes tokenizer/tokenizer_new.json and leaves the active tokenizer unchanged.
Requires:
    pip install tokenizers datasets tqdm
"""
import argparse
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
]


def text_iterator(dataset_name, dataset_config, max_documents):
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
    )

    accepted = 0
    for example in dataset:
        text = example.get("text", "")
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue

        yield text
        accepted += 1
        if accepted >= max_documents:
            break


def main():
    parser = argparse.ArgumentParser(description="Train Pixi's Byte-Level BPE tokenizer.")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--max-documents", type=int, default=100_000)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "tokenizer" / "tokenizer_new.json"),
    )
    args = parser.parse_args()

    if args.max_documents <= 0:
        raise ValueError("--max-documents must be greater than zero.")
    if args.vocab_size <= len(SPECIAL_TOKENS) + 256:
        raise ValueError("--vocab-size is too small for the special tokens and byte alphabet.")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training tokenizer on up to {args.max_documents:,} documents...")
    tokenizer.train_from_iterator(
        text_iterator(args.dataset, args.config, args.max_documents),
        trainer=trainer,
    )

    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    print(f"Tokenizer vocabulary size: {actual_vocab_size:,}")

    if actual_vocab_size != args.vocab_size:
        raise RuntimeError(
            f"Expected vocabulary size {args.vocab_size:,}, got {actual_vocab_size:,}. "
            "Try increasing --max-documents before using this tokenizer."
        )

    # Confirm the six special tokens exist and have stable, retrievable IDs.
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(token_id is None for token_id in special_ids.values()):
        raise RuntimeError(f"Missing special token IDs: {special_ids}")

    # Sanity-check ordinary text. Round-trip should preserve these samples exactly.
    test_texts = [
        "Hello, how are you?",
        "The capital of France is Paris.",
        "Transformers use attention mechanisms to process sequences.",
        "Pixi is a small language model built from scratch.",
        "Numbers: 12345; punctuation: !?.,; and code: print('hello')",
        "Line one.\nLine two.\n",
    ]
    for text in test_texts:
        encoded = tokenizer.encode(text)
        if not encoded.ids:
            raise RuntimeError(f"Tokenizer returned no IDs for: {text!r}")
        if min(encoded.ids) < 0 or max(encoded.ids) >= actual_vocab_size:
            raise RuntimeError(f"Out-of-range token ID for: {text!r}")
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        if decoded != text:
            raise RuntimeError(
                f"Round-trip failed.\nOriginal: {text!r}\nDecoded:  {decoded!r}"
            )

    tokenizer.save(str(output_path))

    # Reload the serialized file and verify it behaves the same way.
    reloaded = Tokenizer.from_file(str(output_path))
    for text in test_texts:
        if reloaded.decode(reloaded.encode(text).ids, skip_special_tokens=False) != text:
            raise RuntimeError("Saved tokenizer failed the round-trip verification.")

    print("\nTokenizer training and sanity checks passed.")
    print(f"Saved to: {output_path}")
    print("Special token IDs:")
    for token, token_id in special_ids.items():
        print(f"  {token}: {token_id}")
    print("\nDo not start pretraining yet. First inspect this tokenizer and update the data pipeline.")


if __name__ == "__main__":
    main()
