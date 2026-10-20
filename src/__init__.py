"""
Pixi (40M Parameter English Chatbot) Core Package
"""

from .model import Pixi, PixiConfig, MicroChat, MicroChatConfig
from .data_pipeline import train_bpe_tokenizer, ShardedMemmapDataset, ChatMLDataset
from .trainer import build_optimizer, get_cosine_lr, train_causal_or_sft, dpo_train_step, compute_sequence_logprobs
from .export_and_infer import export_to_gguf_q8_0, generate_chat

__all__ = [
    "Pixi",
    "PixiConfig",
    "MicroChat",
    "MicroChatConfig",
    "train_bpe_tokenizer",
    "ShardedMemmapDataset",
    "ChatMLDataset",
    "build_optimizer",
    "get_cosine_lr",
    "train_causal_or_sft",
    "dpo_train_step",
    "compute_sequence_logprobs",
    "export_to_gguf_q8_0",
    "generate_chat",
]
