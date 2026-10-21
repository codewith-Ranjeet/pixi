"""Stream FineWeb-Edu text into protected uint16 token shards."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset_shards(dataset_name, dataset_config, tokenizer_path, output_dir,
                         target_tokens, shard_size):
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if target_tokens <= 0 or shard_size <= 0:
        raise ValueError("target_tokens and shard_size must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    special_ids = {token: tokenizer.token_to_id(token) for token in
                   ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|im_start|>", "<|im_end|>")}
    if any(value is None for value in special_ids.values()):
        raise ValueError(f"Tokenizer is missing required control tokens: {special_ids}")
    if max(special_ids.values()) >= 65_536 or vocab_size > 65_536:
        raise ValueError("Tokenizer IDs do not fit the uint16 shard format.")

    print(f"Streaming {dataset_name} ({dataset_config}) using tokenizer SHA-256 {sha256_file(tokenizer_path)}")
    dataset = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    buffer = np.empty(shard_size, dtype=np.uint16)
    used = 0
    total_tokens = 0
    shard_index = 0
    shard_names = []
    progress = tqdm(total=target_tokens, desc="Tokenizing and sharding", unit=" tokens")

    for example in dataset:
        text = example.get("text", "")
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue
        ids = tokenizer.encode(text).ids
        ids.append(special_ids["<|eos|>"])
        if not ids or max(ids) >= vocab_size:
            raise ValueError("Tokenizer produced an invalid token ID.")

        offset = 0
        while offset < len(ids):
            count = min(shard_size - used, len(ids) - offset)
            buffer[used : used + count] = ids[offset : offset + count]
            used += count
            offset += count
            total_tokens += count
            if used == shard_size:
                name = f"shard_{shard_index:03d}.bin"
                buffer.tofile(output_dir / name)
                shard_names.append(name)
                shard_index += 1
                used = 0
        progress.update(len(ids))
        if total_tokens >= target_tokens:
            break

    if used:
        name = f"shard_{shard_index:03d}.bin"
        buffer[:used].tofile(output_dir / name)
        shard_names.append(name)
    progress.close()
    if not shard_names:
        raise RuntimeError("Dataset stream produced no usable token shards.")

    full_shards = [name for name in shard_names if (output_dir / name).stat().st_size // 2 > 1024]
    manifest = {
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": "train stream; final full shard reserved by train_pretrain.py for validation",
        "tokenizer_path": str(tokenizer_path.relative_to(PROJECT_ROOT)),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "vocab_size": vocab_size,
        "special_token_ids": special_ids,
        "dtype": "uint16",
        "target_tokens": target_tokens,
        "actual_tokens": total_tokens,
        "shard_size_tokens": shard_size,
        "shards": shard_names,
        "validation_shard": full_shards[-1] if len(full_shards) >= 2 else None,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {total_tokens:,} tokens in {len(shard_names)} shards to {output_dir}")
    if manifest["validation_shard"] is None:
        print("Warning: fewer than two full shards; create more data before pretraining with validation.")
    else:
        print(f"Held-out validation shard: {manifest['validation_shard']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--tokenizer", default=str(PROJECT_ROOT / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "pretrain_bins"))
    parser.add_argument("--target-tokens", type=int, default=500_000_000)
    parser.add_argument("--shard-size", type=int, default=50_000_000)
    args = parser.parse_args()
    build_dataset_shards(args.dataset, args.config, args.tokenizer, args.output_dir,
                         args.target_tokens, args.shard_size)


if __name__ == "__main__":
    main()
