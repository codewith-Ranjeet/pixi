# 🤖 Pixi-40M

> **An ultra-compact, high-performance ~40.5 Million parameter English conversational language model built from scratch in PyTorch with a custom zero-copy AVX2 C++ inference engine.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-20%20(AVX2%2FFMA)-00599C.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Parameters](https://img.shields.io/badge/Model%20Size-40.5M%20Params-orange.svg)](#-model-architecture--specs)
[![VRAM](https://img.shields.io/badge/VRAM%20Required-%3C3GB%20(RTX%203050)-purple.svg)](#-hardware-requirements)

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Model Architecture & Specs](#-model-architecture--specs)
3. [Hardware Requirements](#-hardware-requirements)
4. [Project Directory Layout](#-project-directory-layout)
5. [Quickstart in 1 Minute](#-quickstart-in-1-minute)
6. [Step-by-Step Guide: From Scratch to Chatbot](#-step-by-step-guide-from-scratch-to-chatbot)
   * [Step 1: Environment Setup](#step-1-environment-setup)
   * [Step 2: Tokenizer Training](#step-2-tokenizer-training)
   * [Step 3: Pre-training Data Acquisition & Sharding](#step-3-pre-training-data-acquisition--sharding)
   * [Step 4: Launching Pre-training on GPU](#step-4-launching-pre-training-on-gpu)
   * [Step 5: Supervised Fine-Tuning (SFT)](#step-5-supervised-fine-tuning-sft)
   * [Step 6: Alignment via DPO](#step-6-alignment-via-dpo)
   * [Step 7: Quantization to GGUF Q8_0](#step-7-quantization-to-gguf-q8_0)
   * [Step 8: High-Speed Native C++ Inference](#step-8-high-speed-native-c-inference)
7. [Inference Performance Benchmarks](#-inference-performance-benchmarks)
8. [Troubleshooting & FAQ](#-troubleshooting--faq)
9. [License](#-license)

---

## 🌟 Overview

**Pixi-40M** is a compact, decoder-only causal language model engineered specifically for low-latency English conversational assistance, edge devices, and personal AI systems. 

Unlike massive LLMs that require high-end server clusters, Pixi-40M:
* Fits completely inside **~2.5 GB of VRAM** during training (comfortably trainable on a consumer **RTX 3050 6GB** laptop GPU).
* Quantizes into a tiny **~41 MB GGUF container** in 8-bit precision (`Q8_0`).
* Generates tokens at **270+ tokens/second on a standard laptop CPU** using our custom AVX2/FMA zero-copy C++ inference engine.
* Employs modern LLM architectural standards: **Root Mean Square Normalization (RMSNorm)**, **Rotary Position Embeddings (RoPE)**, **SwiGLU feedforward blocks**, and **weight tying**.

---

## 📐 Model Architecture & Specs

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **40,512,000** | Exact verified parameter count (~40.5M) |
| **Hidden Dimension ($d_{\text{model}}$)** | **512** | Model width across all layers |
| **Layers ($L$)** | **10** | Transformer decoder blocks |
| **Attention Heads ($H$)** | **8** | Multi-Head Attention (head_dim = 64) |
| **Intermediate Dimension ($d_{\text{ff}}$)** | **1,408** | SwiGLU projection dimension ($\approx \frac{8}{3} \times d_{\text{model}}$) |
| **Vocabulary Size ($V$)** | **16,384** | Byte-Level BPE with ChatML control tokens |
| **Context Window ($T$)** | **1,024** | Tokens (expandable to 2,048) |
| **Positional Embeddings** | **RoPE** | Rotary Position Embeddings ($\theta = 10,000$) |
| **Normalization** | **RMSNorm** | Pre-normalization with $\epsilon = 10^{-5}$ |
| **LM Head** | **Weight-Tied** | Shared weights with input token embeddings |

$$\mathbf{Parameters} = (16,384 \times 512) + 10 \times \left[4(512^2) + 3(512 \times 1,408) + 2(512)\right] + 512 = \mathbf{40,512,000}$$

---

## 💻 Hardware Requirements

Pixi-40M is engineered to run on accessible consumer hardware:

* **Recommended GPU**: NVIDIA GeForce RTX 3050 (6GB) / RTX 3060 / RTX 4060 / any Ampere or newer GPU with $\ge 4\text{ GB}$ VRAM.
* **CPU Fallback**: 8-core CPU (e.g. Intel Core i5-12450HX or AMD Ryzen 5/7).
* **RAM**: 8 GB minimum (16 GB recommended).
* **Disk Space**: ~3 GB (includes pre-training binary shards, model checkpoints, and tokenizer).

---

## 📂 Project Directory Layout

```text
pixi/
├── configs/                     # Architectural and training configurations
│   └── model_config.json        # JSON specifications of model dimensions
├── src/                         # Core Python package modules
│   ├── __init__.py              # Package entrypoint
│   ├── model.py                 # Pure PyTorch Pixi Transformer architecture
│   ├── data_pipeline.py         # Tokenizer training & memory-mapped dataset loaders
│   ├── trainer.py               # AdamW optimizer, cosine schedule, SFT & DPO logic
│   └── export_and_infer.py      # GGUF Q8_0 exporter & KV-cache generation logic
├── scripts/                     # Standalone automation & workflow scripts
│   ├── prepare_data.py          # Hugging Face dataset downloader and binary shard packager
│   ├── train_pretrain.py        # Pre-training execution script for RTX 3050
│   └── verify_pipeline.py       # End-to-end prototype demo and test script
├── cpp/                         # High-performance native C++ inference engine
│   ├── pixi_infer.cpp      # Zero-copy AVX2/FMA/F16C SIMD runner
│   ├── pixi_infer.exe      # Compiled Windows executable
│   └── compile.bat              # One-click C++ compiler batch script
├── tokenizer/                   # Vocabulary assets
│   └── tokenizer.json           # 16,384-token BPE tokenizer with ChatML tokens
├── models/                      # Quantized model binaries
│   └── pixi-40m-q8_0.gguf  # 41.09 MB standalone quantized model
├── checkpoints/                 # Model checkpoints
│   ├── pretrain/                # Pre-training checkpoint snapshots
│   └── pixi_40m.pt         # Initial verified model weights (154.5 MB)
├── data/                        # Datasets & binary token shards
│   ├── pretrain_bins/           # 500M token pre-training binary shards
│   │   ├── shard_000.bin (100 MB)
│   │   ├── ...
│   │   └── shard_010.bin (100 MB)
│   ├── chat_sft.jsonl           # ChatML instruction fine-tuning dataset
│   └── raw_corpus.txt           # Raw English text corpus
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusion rules for large binaries
├── plan.md                      # Detailed engineering log & milestone audit
├── README.md                    # Project documentation (this file)
└── run_pretrain.py              # Root launcher for pre-training
```

---

## ⚡ Quickstart in 1 Minute

Clone the repository and install the dependencies:

```powershell
# 1. Clone the repository
git clone https://github.com/your-username/pixi-40m.git
cd pixi-40m

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Instant Test Inference

* **Python KV-Cached Generator**:
  ```powershell
  python -c "from src.model import Pixi, PixiConfig; from src.export_and_infer import generate_chat; from tokenizers import Tokenizer; import torch; tok = Tokenizer.from_file('tokenizer/tokenizer.json'); m = Pixi(PixiConfig()); m.load_state_dict(torch.load('checkpoints/pixi_40m.pt', map_location='cpu')); print(tok.decode(generate_chat(m, tok.encode('<|im_start|>user\nHello! Who are you?<|im_end|>\n<|im_start|>assistant\n').ids, device='cpu')))"
  ```

* **Native C++ Engine (270+ tok/s on CPU)**:
  ```powershell
  .\cpp\pixi_infer.exe models\pixi-40m-q8_0.gguf
  ```

---

## 🛠️ Step-by-Step Guide: From Scratch to Chatbot

### Step 1: Environment Setup

Ensure your GPU is detected with CUDA acceleration:

```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> [!TIP]
> If CUDA is reported as `False`, install the CUDA-enabled PyTorch build for your system:
> ```powershell
> pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu126
> ```

---

### Step 2: Tokenizer Training

Pixi uses a Byte-Level Byte-Pair Encoding (BPE) tokenizer with $16,384$ tokens and dedicated ChatML delimiters:
* `<|pad|>`: ID `0`
* `<|bos|>`: ID `1`
* `<|eos|>`: ID `2`
* `<|unk|>`: ID `3`
* `<|im_start|>`: ID `4`
* `<|im_end|>`: ID `5`

To train or retrain the tokenizer on your custom text corpus:
```python
from src.data_pipeline import train_bpe_tokenizer
tok = train_bpe_tokenizer(["data/raw_corpus.txt"], save_path="tokenizer/tokenizer.json", vocab_size=16384)
```

---

### Step 3: Pre-training Data Acquisition & Sharding

To prevent CPU data loader bottlenecks during pre-training, raw text is streamed and packed into memory-mapped `uint16` binary shards:

```powershell
python scripts\prepare_data.py
```

This extracts text from `HuggingFaceFW/fineweb-edu`, tokenizes it, and outputs binary shards to `data/pretrain_bins/` (~500 Million tokens across 11 shards).

---

### Step 4: Launching Pre-training on GPU

Launch causal next-token pre-training on your RTX 3050 GPU:

```powershell
python run_pretrain.py
```

* **Batch Size**: 8 micro-batch $\times$ 8 gradient accumulation steps = **65,536 tokens per optimizer step**.
* **Sequence Length**: 1,024 tokens.
* **Precision**: `torch.bfloat16` with native Ampere Tensor Cores.
* **Loss Trajectory**:
  * Step 1: Initial loss starts near $\sim 9.70$ ($\ln(16384)$).
  * Step 500: Drops to $\sim 5.5 - 6.0$ (learning common words and syntax).
  * Step 2,500: Drops to $\sim 3.8$ (learning grammar and sentence coherence).
  * Step 10,000: Settles between $\sim 2.6$ and $3.0$ (fluent English text generation).
* Checkpoints are automatically saved to `checkpoints/pretrain/`.

---

### Step 5: Supervised Fine-Tuning (SFT)

Transform the base pre-trained model into an obedient conversational agent using **ChatML formatting**:

```text
<|im_start|>system
You are Pixi, a helpful English AI assistant.<|im_end|>
<|im_start|>user
What causes rain?<|im_end|>
<|im_start|>assistant
Rain occurs when atmospheric water vapor condenses into water droplets that fall under gravity.<|im_end|>
```

> [!IMPORTANT]
> **Prompt Masking**: [`ChatMLDataset`](file:///c:/workspace/pixi/src/data_pipeline.py) automatically masks all system and user prompt tokens with `labels = -100`. Loss is computed **strictly on the assistant's tokens and closing `<|im_end|>`**, preventing the model from regurgitating user inputs.

---

### Step 6: Alignment via DPO

Align the model with human preferences using Direct Preference Optimization:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

Use [`dpo_train_step()`](file:///c:/workspace/pixi/src/trainer.py) on paired chosen ($y_w$) and rejected ($y_l$) responses to suppress repetitive loops and eliminate hallucinations.

---

### Step 7: Quantization to GGUF Q8_0

Convert your trained PyTorch checkpoint into a standalone 8-bit quantized GGUF v3 container:

```python
import torch
from src.model import Pixi, PixiConfig
from src.export_and_infer import export_to_gguf_q8_0

model = Pixi(PixiConfig())
model.load_state_dict(torch.load("checkpoints/pixi_40m.pt", map_location="cpu"))
export_to_gguf_q8_0(model, out_path="models/pixi-40m-q8_0.gguf")
```

The resulting file is only **41.09 MB**, containing all quantized weights and metadata.

---

### Step 8: High-Speed Native C++ Inference

Our native C++ inference engine (`cpp/pixi_infer.cpp`) uses:
* Zero-copy memory mapping (`MapViewOfFile` on Windows, `mmap` on POSIX).
* AVX2, FMA, and F16C hardware vectorization intrinsics.
* Static activation arena with **zero heap allocations** during token decoding.

#### Recompile C++ Engine (Optional)
```powershell
cd cpp
.\compile.bat
cd ..
```

#### Run Fast CLI Chatbot:
```powershell
.\cpp\pixi_infer.exe models\pixi-40m-q8_0.gguf
```

---

## 📊 Inference Performance Benchmarks

*Tested on Intel Core i5-12450HX CPU (No GPU required during inference)*:

| Inference Engine | Memory Model | Quantization | Latency per Token | Generation Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Python (PyTorch SDPA)** | PyTorch Tensors | FP32 / BF16 | ~11.4 ms / token | **~88 tok/s** |
| **Native C++ (AVX2 / FMA)** | Zero-Copy Mapped | **Q8_0 (8-bit)** | **~3.6 ms / token** | **~273 tok/s** |

---

## ❓ Troubleshooting & FAQ

### 1. `CUDA Available: False`
* **Fix**: Ensure the CUDA-enabled wheel of PyTorch is installed:
  ```powershell
  pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu126
  ```

### 2. Loss is `NaN` or exploding during pre-training
* **Fix**: Increase warmup steps (e.g. from 500 to 1,500) and verify gradient clipping is enabled (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`).

### 3. Repetition loops ("the the the...")
* **Fix**: In [`generate_chat()`](file:///c:/workspace/pixi/src/export_and_infer.py), set `temperature = 0.7`, `top_p = 0.9`, and ensure the model has trained on sufficient tokens ($\ge 500\text{M}$ tokens).

---

## 📜 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
