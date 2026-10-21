# Pixi-40M

Pixi is an experimental decoder-only Transformer implemented in PyTorch. It has about 40.5 million parameters and a 1,024-token context window. **It is a research prototype, not a reliable general-purpose assistant.** The latest SFT checkpoint can repeat text and produce inaccurate answers. A lower training or validation loss does not establish chat quality.

## Project contents

```text
configs/model_config.json                  Model configuration
src/                                       Model, data pipeline, trainer, and GGUF exporter
scripts/train_sft.py                       Supervised fine-tuning
scripts/prepare_ultrachat_sft.py            Prepare the pinned UltraChat subset
scripts/evaluate_sft.py                    Deterministic prompt benchmark
scripts/test_tokenizer.py                  Tokenizer round-trip check
scripts/prepare_data.py                    Legacy pretraining-data preparation
scripts/train_pretrain.py                  Legacy pretraining trainer
chat.py                                    Interactive chat with calculator and fact lookup
chat_base.py                               Base-model text completion
tokenizer/tokenizer.json                   Active tokenizer; keep paired with checkpoints
data/facts.json                            Local sourced facts used by chat.py
data/chat_sft_ultrachat_*.jsonl            Local SFT data (ignored by Git)
data/chat_sft_ultrachat_SOURCE.md           SFT dataset provenance
data/pretrain_bins/                         Legacy pretraining shards; not needed for chat or SFT
checkpoints/old_pretrain/                   Finished base-model checkpoints
checkpoints/pixi_chat_sft_session10_final_v9.pt
                                           Latest tested SFT checkpoint
result/                                    Local training and prompt-response records (ignored by Git)
```

The C++ inference prototype was removed. Python chat and training do not require GGUF exports or a `models/` directory. Large datasets and checkpoints are local artifacts and are not committed. Do not replace the tokenizer without also preparing compatible data and training compatible weights.

## Install and chat

Run these commands from the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python chat.py
```

With no `--checkpoint`, `chat.py` prefers `checkpoints/pixi_chat_sft_session10_final_v9.pt`. To select a checkpoint explicitly:

```powershell
python chat.py --checkpoint .\checkpoints\pixi_chat_sft_session10_final_v9.pt
```

Use `python chat_base.py` to see raw base-model text completion. Temperature changes sampling but do not fix repetition or capability limits.

## Calculator and local fact lookup

`chat.py` uses a restricted local calculator for supported arithmetic expressions and simple addition, subtraction, multiplication, and division phrasing. It does not execute user-provided Python code. Unsupported word problems are sent to the model and may be answered incorrectly.

Fact lookup reads [`data/facts.json`](data/facts.json), which contains sourced common facts and country capitals. Pixi does not fetch facts from the internet at chat time. If a recognized fact has no local match, the default behavior is to decline rather than guess. `--allow-unverified-facts` permits a model answer marked unverified.

Examples:

```powershell
python chat.py --checkpoint .\checkpoints\pixi_chat_sft_session10_final_v9.pt --temp 0
python chat.py --checkpoint .\checkpoints\pixi_chat_sft_session10_final_v9.pt --allow-unverified-facts
```

Use `--facts path\to\facts.json` to select another fact library, `--no-facts` to disable fact lookup, or `--no-calculator` to disable arithmetic. Verify sources before adding entries to `facts.json`.

## SFT

The prepared SFT set is a bounded subset of [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). The source note records its pinned revision, filtering, counts, and tokenizer hash. The dialogues are synthetic according to the dataset card; inspect examples and do not assume their factual correctness.

The prepared files are already present locally. To prepare them again, the output paths must not already exist; the script refuses to overwrite files:

```powershell
python scripts/prepare_ultrachat_sft.py
```

To run the default one-epoch SFT job:

```powershell
python scripts/train_sft.py
```

By default, SFT starts from `checkpoints/old_pretrain/pixi_step_10000_session3.pt`, uses the UltraChat train/validation JSONL files, and saves `checkpoints/pixi_chat_sft_ultrachat.pt`. It refuses to overwrite an existing output checkpoint. The SFT dataset and base checkpoint must be present. The already trained `pixi_chat_sft_session10_final_v9.pt` is not overwritten by this command.

Compare checkpoints using the same evaluation suite and decoding settings. For example:

```powershell
python scripts/evaluate_sft.py --checkpoint .\checkpoints\pixi_chat_sft_session10_final_v9.pt --suite targeted --output .\result\SFT\session10_targeted.json
python scripts/evaluate_sft.py --checkpoint .\checkpoints\pixi_chat_sft_session10_final_v9.pt --suite generalization --output .\result\SFT\session10_generalization.json
```

The benchmark is a small smoke test, not proof of broad reliability. Preserve training logs and prompt-response samples in `result/` for each run; those local records are ignored by Git.

## Pretraining assets

Pretraining is not part of the recommended workflow for this project. The existing `data/pretrain_bins/` shards and `scripts/prepare_data.py` / `scripts/train_pretrain.py` are legacy assets and are not needed for inference or future SFT from the retained base checkpoint. Since no more pretraining is planned, the shard directory may be removed to recover disk space. Keep `checkpoints/old_pretrain/` if you may fine-tune again; those checkpoints are SFT starting points.

## License

Project code is licensed under Apache-2.0; see [LICENSE](LICENSE). Dataset licensing and provenance are documented in the UltraChat source note.
