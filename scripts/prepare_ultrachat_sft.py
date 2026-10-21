"""Prepare a bounded, deduplicated UltraChat 200k subset for Pixi SFT."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
SYSTEM = "You are Pixi, a friendly and helpful English AI assistant."
TRAIN_OUT = ROOT / "data" / "chat_sft_ultrachat_train.jsonl"
VALID_OUT = ROOT / "data" / "chat_sft_ultrachat_validation.jsonl"
SOURCE_OUT = ROOT / "data" / "chat_sft_ultrachat_SOURCE.md"


def message_cost(message: dict, tokenizer: Tokenizer) -> int:
    return (
        1 + len(tokenizer.encode(f"{message['role']}\n").ids)
        + len(tokenizer.encode(message["content"]).ids)
        + 1 + len(tokenizer.encode("\n").ids)
    )


def pack_conversation(row: dict, tokenizer: Tokenizer, max_tokens: int) -> dict | None:
    raw = row.get("messages")
    if not isinstance(raw, list) or not raw:
        return None
    messages = [{"role": "system", "content": SYSTEM}]
    used = message_cost(messages[0], tokenizer)
    i = 0
    while i + 1 < len(raw):
        user, assistant = raw[i], raw[i + 1]
        if not isinstance(user, dict) or not isinstance(assistant, dict):
            break
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            break
        user_text, assistant_text = user.get("content"), assistant.get("content")
        if not isinstance(user_text, str) or not isinstance(assistant_text, str):
            break
        user_text, assistant_text = user_text.strip(), assistant_text.strip()
        if len(user_text) < 3 or len(assistant_text) < 12:
            break
        user_msg = {"role": "user", "content": user_text}
        assistant_msg = {"role": "assistant", "content": assistant_text}
        pair_cost = message_cost(user_msg, tokenizer) + message_cost(assistant_msg, tokenizer)
        if used + pair_cost <= max_tokens:
            messages.extend((user_msg, assistant_msg))
            used += pair_cost
            i += 2
            continue

        # Keep the first user turn and a token-bounded assistant prefix when a
        # single answer is longer than the model's SFT context.
        if len(messages) == 1:
            user_cost = message_cost(user_msg, tokenizer)
            assistant_overhead = message_cost({"role": "assistant", "content": ""}, tokenizer)
            budget = max_tokens - used - user_cost - assistant_overhead
            if budget < 16:
                return None
            answer_ids = tokenizer.encode(assistant_text).ids[:budget]
            answer_prefix = tokenizer.decode(answer_ids).strip()
            while answer_prefix and (
                used + user_cost + message_cost(
                    {"role": "assistant", "content": answer_prefix}, tokenizer
                ) > max_tokens
            ):
                excess = (
                    used + user_cost + message_cost(
                        {"role": "assistant", "content": answer_prefix}, tokenizer
                    ) - max_tokens
                )
                answer_ids = answer_ids[: max(0, len(answer_ids) - excess - 2)]
                answer_prefix = tokenizer.decode(answer_ids).strip()
            if len(answer_prefix) < 12:
                return None
            messages.extend((user_msg, {"role": "assistant", "content": answer_prefix}))
        break

    if len(messages) < 3 or messages[-1]["role"] != "assistant":
        return None
    if sum(message_cost(message, tokenizer) for message in messages) > max_tokens:
        raise AssertionError("Packed conversation exceeds the configured token limit.")
    return {"messages": messages}


def signature(conversation: dict) -> str:
    parts = []
    for message in conversation["messages"]:
        if message["role"] == "system":
            continue
        normalized = re.sub(r"\s+", " ", message["content"].casefold()).strip()
        parts.append(f"{message['role']}:{normalized}")
    text = "\n".join(parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect(split: str, revision: str, tokenizer: Tokenizer, limit: int,
            max_tokens: int, seed: int, scan_limit: int) -> list[dict]:
    rows = load_dataset(DATASET_ID, split=split, revision=revision, streaming=True)
    if split == "train_sft":
        rows = rows.shuffle(seed=seed, buffer_size=20_000)
    seen = set()
    result = []
    for index, row in enumerate(rows):
        if index >= scan_limit or len(result) >= limit:
            break
        conversation = pack_conversation(row, tokenizer, max_tokens)
        if conversation is None:
            continue
        key = signature(conversation)
        if key in seen:
            continue
        seen.add(key)
        result.append(conversation)
    if len(result) < limit:
        print(f"[Data] {split}: found {len(result):,} valid unique rows within scan limit {scan_limit:,}.")
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    train_limit, validation_limit = 10_000, 1_000
    max_tokens, seed = 512, 2026
    if any(path.exists() for path in (TRAIN_OUT, VALID_OUT, SOURCE_OUT)):
        raise FileExistsError("UltraChat outputs already exist; refusing to overwrite them.")

    tokenizer_path = ROOT / "tokenizer" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    revision = HfApi().repo_info(DATASET_ID, repo_type="dataset").sha
    print(f"[Source] {DATASET_ID}@{revision}")
    train = collect("train_sft", revision, tokenizer, train_limit, max_tokens,
                    seed, scan_limit=150_000)
    validation = collect("test_sft", revision, tokenizer, validation_limit,
                         max_tokens, seed, scan_limit=30_000)
    if len(train) < 1_000 or len(validation) < 100:
        raise RuntimeError(
            f"Prepared subset is too small ({len(train)} train, {len(validation)} validation); "
            "outputs were not written."
        )

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VALID_OUT, validation)
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    SOURCE_OUT.write_text(
        "# Pixi UltraChat SFT data provenance\n\n"
        f"- Source: https://huggingface.co/datasets/{DATASET_ID}\n"
        f"- Immutable dataset revision: `{revision}`\n"
        "- Dataset card license: MIT. The card describes the dialogues as generated by ChatGPT; inspect samples and applicable source terms before redistribution.\n"
        "- Official splits used: `train_sft` for training and `test_sft` for validation; no split mixing.\n"
        f"- Filtering: alternating user/assistant turns, non-empty English text as supplied by the source, complete turn packing within {max_tokens} tokenizer tokens, deduplicated conversations; long first answers are token-truncated to fit.\n"
        f"- System prompt: `{SYSTEM}`\n"
        f"- Prepared counts: {len(train)} train, {len(validation)} validation.\n"
        f"- Tokenizer SHA-256: `{tokenizer_hash}`\n"
        f"- Sampling seed: {seed}; maximums: {train_limit} train, {validation_limit} validation.\n"
        "- Rebuild with `python scripts/prepare_ultrachat_sft.py`.\n",
        encoding="utf-8",
    )
    print(f"[Ready] train={len(train):,}; validation={len(validation):,}; max tokens={max_tokens}")
    print(f"[Wrote] {TRAIN_OUT}\n[Wrote] {VALID_OUT}\n[Wrote] {SOURCE_OUT}")


if __name__ == "__main__":
    main()
