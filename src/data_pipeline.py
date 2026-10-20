import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

SPECIAL_TOKENS = [
    "<|pad|>",       # ID 0
    "<|bos|>",       # ID 1
    "<|eos|>",       # ID 2
    "<|unk|>",       # ID 3
    "<|im_start|>",  # ID 4
    "<|im_end|>",    # ID 5
]

def train_bpe_tokenizer(corpus_files: list[str], save_path: str = "tokenizer.json", vocab_size: int = 16384):
    """Trains a byte-level BPE tokenizer with exact ChatML special token alignment."""
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train(files=corpus_files, trainer=trainer)
    cur_size = tokenizer.get_vocab_size()
    if cur_size < vocab_size:
        tokenizer.add_special_tokens([f"<|extra_{i}|>" for i in range(vocab_size - cur_size)])
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(save_path)
    return tokenizer

def build_pretrain_bin(tokenizer_path: str, text_files: list[str], bin_path: str):
    """Encodes raw text into a flat uint16 binary shard for zero-copy memmap loading."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|eos|>")
    
    buffer = []
    out_dir = os.path.dirname(bin_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(bin_path, "wb") as f_out:
        for filepath in text_files:
            with open(filepath, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    ids = tokenizer.encode(line).ids + [eos_id]
                    buffer.extend(ids)
                    if len(buffer) >= 1_000_000:
                        np.array(buffer, dtype=np.uint16).tofile(f_out)
                        buffer.clear()
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f_out)

class ShardedMemmapDataset(Dataset):
    """Zero-copy memory-mapped dataset for causal pre-training."""
    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.num_samples = max(0, (len(self.data) - 1) // seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return chunk[:-1], chunk[1:]

class ChatMLDataset(Dataset):
    """
    Formats multi-turn conversations using ChatML and builds a strict loss mask.
    Only assistant tokens (and their closing <|im_end|>) contribute to the cross-entropy loss.
    All system/user prompt tokens are masked with -100.
    """
    def __init__(self, jsonl_path: str, tokenizer_path: str, max_len: int = 1024):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.pad_id = self.tokenizer.token_to_id("<|pad|>")
        self.im_start_id = self.tokenizer.token_to_id("<|im_start|>")
        self.im_end_id = self.tokenizer.token_to_id("<|im_end|>")
        self.nl_ids = self.tokenizer.encode("\n").ids
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.conversations = [json.loads(line)["messages"] for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        messages = self.conversations[idx]
        input_ids = []
        labels = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            header_ids = [self.im_start_id] + self.tokenizer.encode(f"{role}\n").ids
            body_ids = self.tokenizer.encode(content).ids + [self.im_end_id] + self.nl_ids
            
            input_ids.extend(header_ids + body_ids)
            if role == "assistant":
                # Mask the header, supervise the assistant body + <|im_end|> + newline
                labels.extend([-100] * len(header_ids) + body_ids)
            else:
                labels.extend([-100] * (len(header_ids) + len(body_ids)))

        # Truncate or pad to max_len + 1 for causal shift
        input_ids = input_ids[:self.max_len + 1]
        labels = labels[:self.max_len + 1]
        
        pad_len = (self.max_len + 1) - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_id] * pad_len)
            labels.extend([-100] * pad_len)

        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y