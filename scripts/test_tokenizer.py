"""Lightweight round-trip checks for the active Pixi tokenizer."""

from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"

TEST_TEXTS = [
    "Hello, how are you?",
    "The capital of France is Paris.",
    "Transformers use attention mechanisms to process sequences.",
    "Pixi is a small language model built from scratch.",
    "Numbers: 12345; punctuation: !?.,; and code: print('hello')",
    "Line one.\nLine two.\n",
]


def main() -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    for text in TEST_TEXTS:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        if decoded != text:
            raise AssertionError(f"Tokenizer round trip failed: {text!r} -> {decoded!r}")
        if any(token_id < 0 or token_id >= tokenizer.get_vocab_size() for token_id in encoded.ids):
            raise AssertionError(f"Tokenizer produced an out-of-range ID for {text!r}")
    for token in ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|im_start|>", "<|im_end|>"):
        if tokenizer.token_to_id(token) is None:
            raise AssertionError(f"Missing special token: {token}")
    print(f"Tokenizer checks passed: {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
