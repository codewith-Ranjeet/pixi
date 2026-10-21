# Pixi-40M

Pixi is an experimental decoder-only Transformer implemented in PyTorch. It has about 40.5 million parameters and a 1,024-token context window. **The current model is a research prototype and does not yet generate reliable conversational answers.** Its latest SFT checkpoint still repeats phrases; a lower validation loss has not translated into good chat quality.

## Project structure

```text
configs/model_config.json             Model configuration
src/                                  Model, data pipeline, optimizer, GGUF export
scripts/prepare_data.py               Build pretraining token shards
scripts/prepare_oasst_sft.py          Prepare OASST1 train and validation splits
scripts/train_pretrain.py             Pretrain with held-out loss reporting
scripts/train_sft.py                  Fine-tune from the final pretraining checkpoint
scripts/train_tokenizer.py            Train a separate candidate tokenizer
scripts/test_tokenizer.py             Check the active tokenizer's round trips
chat.py                               Interactive multi-turn chat
chat_base.py                          Base-model text completion
tokenizer/tokenizer.json              Active tokenizer; keep paired with checkpoints/data
data/pretrain_bins/                   Tokenized pretraining corpus
data/chat_sft_oasst1_*.jsonl          Filtered SFT train/validation data
checkpoints/pretrain_next/pixi_step_10000.pt
                                      Final base-model checkpoint
checkpoints/pixi_chat_sft_pretrain_next.pt
                                      SFT checkpoint produced from the next pretraining run
models/pixi-40m-chat-oasst1-q8_0.gguf
                                      GGUF export of selected SFT checkpoint
cpp/pixi_infer.cpp                    Optional native inference source
```

Large datasets, checkpoints, and model exports are local artifacts and are not committed. Do not substitute a tokenizer without rebuilding the token shards and training a compatible model.

## Setup and run

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python chat.py
```

The chat script loads `checkpoints/pixi_chat_sft_oasst1.pt` when available. Use `python chat_base.py` to inspect base-model continuation behavior. The current model may repeat text in either mode; temperature changes do not fix that underlying limitation.

## Data and training

The active SFT data is prepared from a pinned OpenAssistant/oasst1 revision. The filtering, license, revision, and row counts are recorded in [data/chat_sft_oasst1_SOURCE.md](data/chat_sft_oasst1_SOURCE.md). Prepared outputs are separate train and validation JSONL files. The preparation script refuses to overwrite existing outputs unless explicitly passed `--overwrite`.

The pretraining shards under `data/pretrain_bins/` are checked against the active tokenizer before the trainer initializes. The last full shard is held out for validation; the small trailing shard is ignored. The trainer reports held-out loss, records metrics and the tokenizer hash, and writes checkpoints under the separate `checkpoints/pretrain_next/` directory. It refuses to overwrite existing outputs.

Before committing GPU time, validate the current shards:

```powershell
python scripts/train_pretrain.py --check-only
```

This scans token ranges and decodes sample windows using the active tokenizer. It does not initialize a model or train. If it passes, launch the default 10,000-step run with `python scripts/train_pretrain.py`. This is a long GPU job. The default processes about 655 million training tokens across about 450 million train tokens, cycling the train split as needed; one full shard (about 50 million tokens) is held out for validation.

To prepare fresh shards with a tokenizer-hash manifest, use a new output directory so existing shards remain intact:

```powershell
python scripts/prepare_data.py --output-dir data/pretrain_bins_next
python scripts/train_pretrain.py --data-dir data/pretrain_bins_next
```

After pretraining, the final checkpoint is `checkpoints/pretrain_next/pixi_step_10000.pt`. The SFT trainer uses it by default and writes a new checkpoint/export with separate names; chat prefers that result when it exists.

To fine-tune, first choose new output names or move the existing selected artifacts, because the trainer intentionally refuses to overwrite them:

```powershell
python scripts/train_sft.py
```

SFT uses one epoch by default and reports held-out loss. Loss alone is not a chat-quality guarantee. The selected checkpoint currently has a validation loss of about 2.80, compared with about 3.18 for the base checkpoint on the same split, but generated replies remain repetitive.

## Configuration and native inference

Model dimensions are in [configs/model_config.json](configs/model_config.json). To build the optional C++ runner on a system with a compatible `g++`, run `cpp/compile.bat`; the generated executable is ignored by Git.

## License

Project code is licensed under Apache-2.0; see [LICENSE](LICENSE). Dataset licensing and provenance are documented separately in the SFT source note.
