# Pixi-40M

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Parameters](https://img.shields.io/badge/Parameters-40.5M-orange.svg)](#model-architecture)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Pixi is a small decoder-only Transformer implemented in PyTorch and trained from scratch as an educational experiment. It has about 40.5 million parameters and a 1,024-token context window. **It is not a reliable general-purpose assistant:** its language-model checkpoints can repeat text and give inaccurate answers. Lower training or validation loss has not reliably translated into better responses.

The chat program adds a restricted local calculator and sourced local fact lookup. Those tools handle supported arithmetic and facts separately from the model; other questions still depend on Pixi's limited language ability.

## Model architecture

| Specification | Value |
| --- | ---: |
| Parameters | 40,512,000 |
| Architecture | Decoder-only Transformer |
| Layers | 10 |
| Model width | 512 |
| Attention heads | 8 (head dimension 64) |
| Feed-forward width | 1,408 (SwiGLU) |
| Vocabulary | 16,384 Byte-Level BPE tokens |
| Context window | 1,024 tokens |
| Position encoding | RoPE, theta 10,000 |
| Normalization | RMSNorm, epsilon 1e-5 |
| Embeddings | Input and output weights tied |

The active tokenizer is `tokenizer/tokenizer.json`. Checkpoints and tokenized data depend on this tokenizer; changing it requires compatible data and model weights.

## Hardware and scope

The project has been run on a Windows laptop with an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM. Inference can also run on CPU, though it will be slower. Memory and speed depend on the selected checkpoint, sequence length, batch size, and PyTorch build; this README does not claim fixed throughput or memory figures.

The old C++ inference prototype and GGUF exports have been removed. Python chat and training do not need a `cpp/` or `models/` directory.

## Project layout

```text
configs/model_config.json                           #Architecture configuration
src/model.py                                        #Transformer implementation
src/data_pipeline.py                                #Pretraining and ChatML data loaders
src/trainer.py                                      #Optimizer and learning-rate schedule
src/chat_tools.py                                   #Safe calculator and local fact lookup
src/export_and_infer.py                             #PyTorch generation and optional GGUF exporter
scripts/train_sft.py                                #Supervised fine-tuning
scripts/prepare_ultrachat_sft.py                    #Prepare the pinned UltraChat subset
scripts/evaluate_sft.py                             #Deterministic response benchmark
scripts/test_tokenizer.py                           #Tokenizer round-trip check
scripts/prepare_data.py                             #Legacy pretraining-data preparation
scripts/train_pretrain.py                           #Legacy pretraining trainer
chat.py                                             #Interactive chat with local tools
chat_base.py                                        #Base-model text completion
tokenizer/tokenizer.json                            #Active tokenizer
data/facts.json                                     #Sourced local facts and capitals
data/chat_sft_ultrachat_*.jsonl                     #Prepared SFT data (local, ignored by Git)
data/chat_sft_ultrachat_SOURCE.md                   #SFT dataset provenance
data/pretrain_bins/                                 #Legacy pretraining shards
checkpoints/old_pretrain/                           #Finished base-model checkpoints
checkpoints/pixi_chat_sft_session10_final_v9.pt     #Latest tested SFT checkpoint
result/                                             #Local training logs and response samples
```

Datasets, checkpoints, and results are local artifacts and are not committed to Git. The pretraining shards are not needed for chat or future SFT from a retained checkpoint. Since no further pretraining is planned, `data/pretrain_bins/` may be removed to reclaim disk space; keep `checkpoints/old_pretrain/` if you may fine-tune again.

## Install and run chat

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python chat.py
```

When no checkpoint is specified, `chat.py` prefers `checkpoints/pixi_chat_sft_session10_final_v9.pt`. Select a checkpoint and sampling options explicitly when comparing models:

```powershell
python chat.py --checkpoint .\checkpoints\pixi_chat_sft_session10_final_v9.pt --temp 0 --top_p 0.9 --max_tokens 100
```

`--temp 0` uses greedy decoding. Higher temperatures make token selection less predictable; `--top_p` limits sampling to a probability-ranked nucleus, and `--max_tokens` limits reply length. These settings can change the output style but cannot fix model limitations. Use `python chat_base.py` to see base-model text completion rather than chat behavior.

## Calculator and fact lookup

`chat.py` uses a restricted local calculator for supported arithmetic expressions and common operation phrasing. It does not execute user-provided Python code. Unsupported word problems go to the model and may be answered incorrectly.

Fact lookup reads [`data/facts.json`](data/facts.json), which contains sourced common facts and country capitals. Pixi does not access the internet at chat time. When a recognized fact is not in the local file, the default behavior is to decline rather than guess. `--allow-unverified-facts` lets the model try, with its answer marked unverified.

```powershell
python chat.py --checkpoint .\checkpoints\pixi_chat_sft_session10_final_v9.pt --allow-unverified-facts
```

Use `--facts path\to\facts.json` to select another fact file, `--no-facts` to disable fact lookup, or `--no-calculator` to disable arithmetic. Check sources before adding facts.

## Supervised fine-tuning

The prepared SFT dataset is a bounded subset of [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). Its source note records the pinned revision, filtering, sample counts, and tokenizer hash. The dataset card describes its dialogues as generated by ChatGPT; inspect examples and do not assume they are factually correct.

The prepared train and validation files are already present locally. To recreate them, first move or remove the existing outputs; the preparation script refuses to overwrite them:

```powershell
python scripts/prepare_ultrachat_sft.py
```

Run the default one-epoch SFT job:

```powershell
python scripts/train_sft.py
```

By default, the trainer starts from `checkpoints/old_pretrain/pixi_step_10000_session3.pt`, uses the UltraChat train and validation files, and saves `checkpoints/pixi_chat_sft_ultrachat.pt`. It refuses to overwrite an existing output checkpoint. This command does not overwrite `checkpoints/pixi_chat_sft_session10_final_v9.pt`.

Evaluate checkpoints with the same decoding settings and benchmark suite. The benchmark is a small smoke test, not evidence of broad reliability:

```powershell
python scripts/evaluate_sft.py --checkpoint .\checkpoints\pixi_chat_sft_session10_final_v9.pt --suite targeted --output .\result\SFT\session10_targeted.json
python scripts/evaluate_sft.py --checkpoint .\checkpoints\pixi_chat_sft_session10_final_v9.pt --suite generalization --output .\result\SFT\session10_generalization.json
```

Keep each run's training log and prompt-response samples in `result/` for comparison. The directory is ignored by Git, so these records remain local unless you choose to archive them separately.

## Pretraining

Pretraining is not part of the recommended workflow. `scripts/prepare_data.py`, `scripts/train_pretrain.py`, and `data/pretrain_bins/` are retained legacy components. Running the trainer is a long GPU job; do not launch it unless you intentionally decide to resume pretraining and have checked the data and outputs first. Inference and SFT from a saved base checkpoint do not require the pretraining shards.

## License

Project code is licensed under the Apache License 2.0; see [LICENSE](LICENSE). Dataset licensing and preparation details are documented in `data/chat_sft_ultrachat_SOURCE.md`.
