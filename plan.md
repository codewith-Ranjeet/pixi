# Pixi project status

This file tracks the current state and known limitations. Older prototype claims and the original pipeline notes were removed because they referred to scripts and artifacts that could overwrite active data or no longer exist.

## Current artifacts

- Architecture: decoder-only Transformer, 40,512,000 parameters, vocabulary size 16,384, context length 1,024.
- Active tokenizer: `tokenizer/tokenizer.json`. Existing checkpoints and pretraining shards are tied to this tokenizer.
- Previous base checkpoint: `checkpoints/pretrain/pixi_step_10000.pt` (retained; new runs do not overwrite it).
- Next pretraining run output: `checkpoints/pretrain_next/`.
- Pretraining data: 500,001,792 prepared token IDs in `data/pretrain_bins/`.
- SFT source: pinned OpenAssistant/oasst1; see `data/chat_sft_oasst1_SOURCE.md`.
- SFT splits: 1,252 train conversations and 67 validation conversations.
- Selected SFT checkpoint: `checkpoints/pixi_chat_sft_oasst1.pt` (one epoch).
- Selected GGUF export: `models/pixi-40m-chat-oasst1-q8_0.gguf`.

## Known limitation

The selected SFT checkpoint has lower held-out loss than the base checkpoint on the prepared validation split (2.80 vs. 3.18), but generation remains repetitive and unreliable. The project is not ready to be described or used as a capable conversational assistant. Do not infer chat quality from training loss alone.

## Pretraining readiness

- Run `python scripts/train_pretrain.py --check-only` before training. It checks token ranges, sampled decoding under the active tokenizer, and reserves the final full shard for validation.
- The existing shard set has no original tokenizer-hash manifest. Readability checks pass under the current tokenizer; the next run records the tokenizer hash and data split in its checkpoints.
- The new trainer writes to a separate output directory, logs held-out loss to `metrics.csv`, and refuses to overwrite existing run artifacts.

## Safe workflow

1. Run `python scripts/test_tokenizer.py` after tokenizer changes.
2. Rebuild pretraining shards only with the tokenizer that will train the model.
3. Use a separate validation set for SFT and save outputs under new names.
4. Inspect generated replies on held-out prompts before treating an SFT run as an improvement.
5. `scripts/verify_pipeline.py` was removed because it recreated the toy SFT data and overwrote the tokenizer and prototype checkpoint.
