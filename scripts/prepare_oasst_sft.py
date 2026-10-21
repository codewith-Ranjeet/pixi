"""Prepare a small, filtered OASST1 SFT dataset for Pixi.

Pinned source: OpenAssistant/oasst1 at fdf72ae0827c1cda404aff25b6603abec9e3399b.
This writes separate train and validation JSONL files and never overwrites by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
REPO = "OpenAssistant/oasst1"
SYSTEM = "You are Pixi, a friendly and intelligent English AI assistant."
FILES = {
    "train": "data/train-00000-of-00001-b42a775f407cee45.parquet",
    "validation": "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
}
OUTPUTS = {
    "train": ROOT / "data" / "chat_sft_oasst1_train.jsonl",
    "validation": ROOT / "data" / "chat_sft_oasst1_validation.jsonl",
}


def row_is_usable(row: dict) -> bool:
    if (
        row.get("lang") != "en"
        or row.get("tree_state") != "ready_for_export"
        or row.get("deleted")
        or row.get("synthetic")
        or row.get("review_result") is not True
        or not isinstance(row.get("text"), str)
        or not row["text"].strip()
    ):
        return False

    labels = row.get("labels") or {}
    if isinstance(labels, dict):
        names = labels.get("name", []) or []
        values = labels.get("value", []) or []
        label_values = dict(zip(names, values))
        excluded = ("spam", "lang_mismatch", "pii", "not_appropriate", "hate_speech", "sexual_content")
        for name in excluded:
            try:
                if float(label_values.get(name, 0) or 0) > 0:
                    return False
            except (TypeError, ValueError):
                return False
    return True


def quality_value(row: dict) -> float:
    labels = row.get("labels") or {}
    if not isinstance(labels, dict):
        return 0.0
    values = dict(zip(labels.get("name", []) or [], labels.get("value", []) or []))
    try:
        return float(values.get("quality", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def serialized_token_count(messages: list[dict], tokenizer: Tokenizer) -> int:
    im_start = tokenizer.token_to_id("<|im_start|>")
    im_end = tokenizer.token_to_id("<|im_end|>")
    newline_ids = tokenizer.encode("\n").ids
    total = 0
    for message in messages:
        total += 1 + len(tokenizer.encode(f"{message['role']}\n").ids)
        total += len(tokenizer.encode(message["content"]).ids) + 1 + len(newline_ids)
    return total


def build_conversations(rows: list[dict], tokenizer: Tokenizer, max_tokens: int) -> list[dict]:
    children: dict[str, list[dict]] = defaultdict(list)
    roots = []
    for row in rows:
        parent = row.get("parent_id")
        if isinstance(parent, str) and parent:
            children[parent].append(row)
        elif row_is_usable(row):
            roots.append(row)

    conversations = []
    seen = set()
    for root in roots:
        if root.get("role") != "prompter":
            continue
        path = [root]
        current = root
        visited = {root["message_id"]}
        while True:
            options = [
                child for child in children.get(current["message_id"], [])
                if row_is_usable(child) and child["message_id"] not in visited
            ]
            expected_role = "assistant" if current["role"] == "prompter" else "prompter"
            options = [child for child in options if child.get("role") == expected_role]
            if not options:
                break

            if expected_role == "assistant":
                def rank_value(child):
                    try:
                        value = float(child.get("rank"))
                    except (TypeError, ValueError):
                        return None
                    return None if math.isnan(value) else value

                ranked = [(child, rank_value(child)) for child in options]
                ranked = [(child, rank) for child, rank in ranked if rank is not None]
                if ranked:
                    best_rank = min(rank for _, rank in ranked)
                    options = [child for child, rank in ranked if rank == best_rank]
                elif len(options) > 1:
                    # Do not choose arbitrarily among unranked assistant alternatives.
                    break

            current = sorted(
                options,
                key=lambda child: (-quality_value(child), child.get("created_date") or "", child["message_id"]),
            )[0]
            path.append(current)
            visited.add(current["message_id"])

        if len(path) < 2 or path[-1].get("role") != "assistant":
            continue
        if any(path[i].get("role") == path[i + 1].get("role") for i in range(len(path) - 1)):
            continue

        messages = [{"role": "system", "content": SYSTEM}]
        messages.extend(
            {"role": "user" if row["role"] == "prompter" else "assistant", "content": row["text"].strip()}
            for row in path
        )
        if serialized_token_count(messages, tokenizer) > max_tokens:
            continue

        signature = "\n".join(f"{m['role']}:{re.sub(r'\s+', ' ', m['content'].casefold()).strip()}" for m in messages)
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        conversations.append({"messages": messages})
    return conversations


def write_jsonl(path: Path, rows: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it.")
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reviewed, English OASST1 SFT train/validation JSONL.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.max_train <= 0:
        raise ValueError("--max-tokens and --max-train must be positive.")
    if not args.overwrite and any(path.exists() for path in OUTPUTS.values()):
        raise FileExistsError("Prepared OASST output exists; pass --overwrite to replace it.")

    tokenizer_path = ROOT / "tokenizer" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cache_dir = ROOT / "data" / "_oasst_hub_cache"
    prepared = {}
    for split, filename in FILES.items():
        source_path = hf_hub_download(
            repo_id=REPO,
            filename=filename,
            repo_type="dataset",
            revision=REVISION,
            cache_dir=str(cache_dir),
        )
        table_rows = pq.read_table(source_path).to_pylist()
        conversations = build_conversations(table_rows, tokenizer, args.max_tokens)
        if split == "train":
            random.Random(args.seed).shuffle(conversations)
            conversations = conversations[: args.max_train]
        prepared[split] = conversations
        print(f"{split}: {len(conversations)} conversations")

    for split, rows in prepared.items():
        write_jsonl(OUTPUTS[split], rows, args.overwrite)

    source_note = ROOT / "data" / "chat_sft_oasst1_SOURCE.md"
    if source_note.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {source_note}; pass --overwrite to replace it.")
    source_note.write_text(
        "# Pixi SFT data provenance\n\n"
        f"- Source: [{REPO}](https://huggingface.co/datasets/{REPO})\n"
        f"- Pinned revision: `{REVISION}`\n"
        "- License: Apache-2.0 (see the source repository LICENSE).\n"
        "- Split: official `train` and `validation` parquet files; trees remain in their original split.\n"
        "- Filters: English, ready-for-export, reviewed, non-deleted, non-synthetic; remove spam, language-mismatch, PII, inappropriate, hate-speech, and sexual-content flagged rows; at assistant branch points keep the best available rank; deduplicate complete conversations; cap at 512 tokenizer tokens.\n"
        f"- Prepared on: 2026-10-21.\n"
        f"- Prepared counts: train {len(prepared['train'])}; validation {len(prepared['validation'])}.\n"
        "- Rebuild with `python scripts/prepare_oasst_sft.py`; outputs are separate train and validation files.\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUTS['train']} and {OUTPUTS['validation']}")


if __name__ == "__main__":
    main()
