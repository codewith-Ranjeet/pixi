"""
Downloads FineWeb-Edu / TinyStories from Hugging Face,
tokenizes using tokenizer.json, and streams into uint16 binary shards.
"""
import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

def build_dataset_shards(
    dataset_name="HuggingFaceFW/fineweb-edu",
    dataset_config="sample-10BT",
    tokenizer_path = None,
    output_dir="data/pretrain_bins",
    target_tokens=500_000_000,    # 500M tokens for a strong base model (adjust as needed)
    shard_size=50_000_000         # 50M tokens per shard (~100 MB per file)
):
    if tokenizer_path is None:
        tokenizer_path = "tokenizer/tokenizer.json" if os.path.exists("tokenizer/tokenizer.json") else "tokenizer.json"
    os.makedirs(output_dir, exist_ok=True)
    tok = Tokenizer.from_file(tokenizer_path)
    eos_id = tok.token_to_id("<|eos|>")

    print(f"Streaming dataset: {dataset_name} ({dataset_config})...")
    ds = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)

    buffer = []
    total_tokens = 0
    shard_idx = 0

    pbar = tqdm(total=target_tokens, desc="Tokenizing & Sharding", unit=" tokens")

    for example in ds:
        text = example.get("text", "")
        if not text or len(text.strip()) < 50:
            continue

        token_ids = tok.encode(text).ids + [eos_id]
        buffer.extend(token_ids)
        total_tokens += len(token_ids)
        pbar.update(len(token_ids))

        if len(buffer) >= shard_size:
            shard_file = os.path.join(output_dir, f"shard_{shard_idx:03d}.bin")
            arr = np.array(buffer[:shard_size], dtype=np.uint16)
            arr.tofile(shard_file)
            buffer = buffer[shard_size:]
            shard_idx += 1

        if total_tokens >= target_tokens:
            break

    # Save remaining tokens
    if buffer:
        shard_file = os.path.join(output_dir, f"shard_{shard_idx:03d}.bin")
        np.array(buffer, dtype=np.uint16).tofile(shard_file)

    pbar.close()
    print(f"\nDone! Extracted {total_tokens:,} tokens across {shard_idx + 1} shards in {output_dir}")

if __name__ == "__main__":
    build_dataset_shards()