# Simple Image Editing Script with diffusers

# diffusersを使った単一スクリプトでAI画像編集を行うコードサンプル

「画像ファイルを引数に取って、編集した画像を出力」 する用途を想定しています。

> **対象環境:** GeForce RTX 3xxx (VRAM 12GB) 程度の環境で動作させることを目指してモデルを選定・調整しています。より高性能なGPUでは `--no-offload` や高解像度での処理も可能ですが、蒸留されていないモデルを選定すべきでしょう。

### 仕様

* 入力：画像（複数画像対応）
* 最大サイズ：2048x2048
* 幅は8の倍数、高さは16の倍数でパディングします（余白で調整）
* 低VRAMの場合：`transformer.set_offload(...num_blocks_on_gpu=...)` + `enable_sequential_cpu_offload()`

### コマンドラインオプション

* `--prompt "..."`：プロンプトを指定（省略時はソースコード内の `PROMPT` を使用）
* `--pre-resize [0.3m|1m|2m]`：アスペクト比維持で総画素数を縮小
* `--no-offload`：オフロード無効（速いがVRAMに余裕が必要）
* `--seed NNN`：乱数シード指定（省略時はランダム）
* `--progress`：ダウンロード進捗表示
* `--mem-log`：メモリ使用量表示
* `--ref {file}`：参照画像ファイル指定（枚数はモデルに依存）
* `--lora {file}` `--lora-scale N`：LoRAファイル指定（メモリが足りないので動作未確認）
* `--t2i` `--size WxH`：text-to-imageモード

> **Tip:** 常に同じプロンプトを使用する場合は、スクリプト内の `PROMPT = "..."` 行を直接編集してください。

---

## Qwen-Image-Edit-2509 Lightning (nunchaku版)

`simple_image_edit_nunchaku_qwen.py`

### 概要

nunchaku版Lightningを使って `Qwen/Qwen-Image-Edit-2509` を高速に動かします。
サンプル構成（scheduler / model_path / VRAMに応じた offload 分岐）に沿って実装しています。

* モデル: [nunchaku-ai/nunchaku-qwen-image-edit-2509](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509)（デフォルト: lightning-251115、約12GB）
* ベース: `Qwen/Qwen-Image-Edit-2509`
* 8ステップ推論

### 実行例

```powershell
py .\simple_image_edit_nunchaku_qwen.py .\sample.png
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --prompt "Remove text and enhance image quality."
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --pre-resize 2m --progress --mem-log
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --no-offload
```

---

## Qwen-Image-Edit-Rapid GGUF量子化版

`simple_image_edit_gguf_qwen.py`

### 概要

Qwen-Image-Edit-Rapid-AIO-V23 の GGUF 量子化モデルを使った画像編集スクリプトです。nunchaku版よりは導入が容易です。

* モデル: [Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF](https://huggingface.co/Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF)（デフォルト: v23 Q3_K、約10GB）
* ベース: `Qwen/Qwen-Image-Edit-2511`
* 4ステップ推論

### 実行例

```powershell
py .\simple_image_edit_gguf_qwen.py .\sample.png
py .\simple_image_edit_gguf_qwen.py .\sample.png --prompt "Enhance quality." --seed 42
py .\simple_image_edit_gguf_qwen.py .\sample.png --pre-resize 2m --offload --progress
py .\simple_image_edit_gguf_qwen.py .\sample.png --gguf-file v23/Qwen-Rapid-NSFW-v23_Q2_K.gguf
```

---

## 必要環境

* OS: Windows / Linux（CUDAが使えること）
* GPU: NVIDIA 推奨
* Python: 3.10〜3.12 推奨（3.11が無難）
* PyTorch: **CUDA対応版**（※`cu130` などの表記は、あなたの環境で入っている PyTorch の CUDA ビルドに依存します）

### 依存ライブラリ（目安）

* torch（CUDA版）
* diffusers
* transformers, accelerate, safetensors
* pillow
* huggingface_hub
* psutil（`--mem-log` を使う場合）
* nunchaku (nunchaku版)
* gguf (gguf版)

---

## インストール例（Windows / PowerShell）

### 1) venv 作成

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
```

### 2) CUDA対応 PyTorch のインストール

PyTorch は**CUDA対応版**を明示的にインストールする必要があります。`--index-url` を指定しないと CPU 版がインストールされる場合があります。

> ※ CUDA 13.0 の場合（RTX 30xx〜50xx）

```powershell
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

> ※ CUDA 12.8 の場合

```powershell
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

利用可能なCUDAバージョンは [PyTorch公式](https://pytorch.org/get-started/locally/) で確認してください。インストール後、以下で確認できます:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 3) diffusers のインストール

nunchaku版・GGUF版は `diffusers==0.36.x` が必要です。

```powershell
pip install "diffusers>=0.36.0,<0.37.0"
```

**重要:** git nightly版（0.37.0.dev）はAPIが変更されており、nunchaku版で `pos_embed` `max_txt_seq_len` `txt_seq_lens` 関連のエラーが発生します。

### 4) nunchaku のインストール（GGUF版のみ使用する場合は不要）

nunchaku はリリースアセットの中からバージョンに合ったコンパイル済のwheelを見つけてpipで導入するのが簡単です。

> ※ Windows、Python 3.11、Torch 2.10 + CUDA 13.0 の場合

```powershell
pip install -U https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl
```

その他の環境は [nunchaku リリースページ](https://github.com/nunchaku-ai/nunchaku/releases) で対応する wheel を探してください。

### 5) その他の依存ライブラリのインストール

```powershell
pip install -U pillow huggingface_hub psutil transformers accelerate safetensors gguf
```

> `psutil` は `--mem-log` 使用時のみ必要、`gguf` は GGUF版のみ必要ですが、まとめて入れても問題ありません。

---

**注意:** 初回起動時にモデルのダウンロードが発生します（**30〜100GB程度**）。ダウンロード先は Hugging Face のキャッシュディレクトリです。

---

## よくある問題と対策

### A) 0%から進まない（特に初回）

* 初回は **モデルのダウンロードとロード**が重く、進捗が止まって見えることがあります。
* `--progress` を付けてダウンロード進捗を出す
* 入力が大きい場合は `--pre-resize 1m` を使う

### B) shape mismatch（width/height と入力画像latentsが合わない）

* 本スクリプトは **入力画像を 2048 上限に収めた上で、端数はパディングで倍数に合わせる**ため、基本的に起きにくいはずです。
* もし自前で `width/height` を固定する改造をした場合は、**画像サイズと width/height を一致**させてください。

### C) OOM（CUDA out of memory）

* `--pre-resize 1m` を使う
* `--no-offload` を付けない（＝オフロード有効のまま）

---

## Tips: キャッシュの管理

初回起動時にダウンロードされたモデルは Hugging Face のキャッシュに保存されます。ディスク容量を確保したい場合は以下のコマンドで削除できます。

```powershell
# pip キャッシュの削除
pip cache purge

# Hugging Face キャッシュの確認
hf cache ls

# 特定モデルの削除
hf cache rm <リビジョンID>
```

キャッシュの場所（デフォルト）:
- Windows: `C:\Users\<ユーザー名>\.cache\huggingface\hub`
- Linux/Mac: `~/.cache/huggingface/hub`

---

## 備考

### Z-Image Turbo (4bit)版 `simple_image_edit_zit.py`

* モデル: [unsloth/Z-Image-Turbo-unsloth-bnb-4bit](https://huggingface.co/unsloth/Z-Image-Turbo-unsloth-bnb-4bit)

テストしましたが要求を満たす性能を発揮させることができませんでした。

### FLUX.2 [klein] 4B 版 `simple_image_edit_flux2_klein.py`

* モデル: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

遜色なく利用できましたが、diffusers最新版(0.37.0dev以降)を要求するため nunchaku版とは共存できません。venvを分ける必要があります。

> `pip install -U git+https://github.com/huggingface/diffusers`

### Qwen-Image-Edit-Rapid-AIO-V23ベースのDiffusers-compatible版 `simple_image_edit_rapid_qwen.py`

* モデル: [prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23)

遜色なく利用できましたが、時間が掛かりすぎます（1枚30分くらいかかった）。

---

## 関連

 - [Qwen-Image-EditをCLIで使う(Diffusers + nunchaku-qwen-image-edit-2509) - ふぁメモ](https://fa.hatenadiary.jp/entry/20260208/1770527361)

---

## English Documentation

Designed to **take a single image file as input and output an edited image**.

> **Target Environment:** Models are selected and tuned to run on GeForce RTX 3xxx (12GB VRAM) class hardware. Higher-end GPUs can use `--no-offload` or process higher resolutions.

### Features

* Input: Single image (official samples support multiple images, but these scripts are for single input)
* `--prompt "..."`: Specify prompt (uses `PROMPT` constant in source if omitted)
* `--pre-resize 1m|2m`: Reduce total pixels while maintaining aspect ratio
* Max size: **Fits within 2048x2048 (downscale only)**
* **Height padded to multiple of 16** (padding, not resizing)
* High VRAM: `enable_model_cpu_offload()`
* Low VRAM: `transformer.set_offload(...num_blocks_on_gpu=...)` + `enable_sequential_cpu_offload()` (follows official sample)
* `--no-offload`: Disable offloading (faster but requires more VRAM)
* `--seed N`: Random seed (random if omitted)
* `--progress` / `--mem-log`

> **Tip:** For a fixed prompt, edit the `PROMPT = "..."` line in the script directly.

---

### Qwen-Image-Edit-2509 Lightning (Nunchaku)

`simple_image_edit_nunchaku_qwen.py`

#### Overview

Runs `Qwen/Qwen-Image-Edit-2509` at high speed using Nunchaku Lightning weights. Implements the official sample structure (scheduler / model_path / VRAM-based offload branching).

* Model: [nunchaku-ai/nunchaku-qwen-image-edit-2509](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509) (default: lightning-251115, ~12GB)
* Base: `Qwen/Qwen-Image-Edit-2509`
* 8-step inference (distilled)

#### Examples

```bash
python simple_image_edit_nunchaku_qwen.py ./sample.png
python simple_image_edit_nunchaku_qwen.py ./sample.png --prompt "Remove text and enhance image quality."
python simple_image_edit_nunchaku_qwen.py ./sample.png --pre-resize 2m --progress --mem-log
python simple_image_edit_nunchaku_qwen.py ./sample.png --no-offload
```

---

### Qwen-Image-Edit-Rapid GGUF Quantized

`simple_image_edit_gguf_qwen.py`

#### Overview

Image editing using GGUF-quantized Qwen-Image-Edit-Rapid-AIO-V23. Unlike the Nunchaku version, **no pre-compiled wheel is needed** — setup is done entirely via pip.

* Model: [Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF](https://huggingface.co/Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF) (default: Q3_K, ~10GB)
* Base: `Qwen/Qwen-Image-Edit-2511`
* 4-step inference (distilled)

#### Examples

```bash
python simple_image_edit_gguf_qwen.py ./sample.png
python simple_image_edit_gguf_qwen.py ./sample.png --prompt "Enhance quality." --seed 42
python simple_image_edit_gguf_qwen.py ./sample.png --pre-resize 2m --offload --progress
python simple_image_edit_gguf_qwen.py ./sample.png --gguf-file v23/Qwen-Rapid-NSFW-v23_Q2_K.gguf
```

---

### Requirements

* OS: Windows / Linux (CUDA required)
* GPU: NVIDIA recommended
* Python: 3.10-3.12 recommended (3.11 is most stable)
* PyTorch: **CUDA version** (e.g., `cu130` depends on your installed PyTorch CUDA build)

#### Dependencies

* torch (CUDA version)
* diffusers
* transformers, accelerate, safetensors
* pillow
* huggingface_hub
* psutil (for `--mem-log`)
* nunchaku (Nunchaku version)
* gguf (GGUF version)

---

### Installation (Windows / PowerShell)

#### 1) Create venv

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
```

#### 2) Install CUDA-enabled PyTorch

You must explicitly install the **CUDA build** of PyTorch. Without `--index-url`, pip may install the CPU-only version.

> CUDA 13.0 (RTX 30xx–50xx):

```powershell
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

> CUDA 12.8:

```powershell
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check available CUDA versions at [pytorch.org](https://pytorch.org/get-started/locally/). Verify after install:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

#### 3) Install diffusers

Nunchaku and GGUF versions require `diffusers==0.36.x`.

```powershell
pip install -U "diffusers>=0.36.0,<0.37.0"
```

**Important:** The git main version (0.37.0.dev) has API changes that cause `pos_embed` / `max_txt_seq_len` / `txt_seq_lens` related errors.

#### 4) Install nunchaku (skip if using GGUF version only)

Install the pre-compiled wheel matching your environment from the release assets.

> Example: Windows, Python 3.11, Torch 2.10 + CUDA 13.0

```powershell
pip install -U https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl
```

For other environments, find the matching wheel on the [nunchaku releases page](https://github.com/nunchaku-ai/nunchaku/releases).

#### 5) Install other dependencies

```powershell
pip install -U pillow huggingface_hub psutil transformers accelerate safetensors gguf
```

> `psutil` is only needed for `--mem-log`, and `gguf` is only needed for the GGUF version, but installing them together is harmless.

---

**Note:** First run will download models (**30-100GB total**). Models are cached in the Hugging Face cache directory.

---

### Troubleshooting

**A) Stuck at 0% (especially first run)**
* First run involves heavy model download and loading, which may appear stuck
* Use `--progress` to show download progress
* Use `--pre-resize 1m` for large inputs

**B) Shape mismatch (width/height vs input image latents)**
* These scripts **cap input to 2048 and pad to multiples**, so this should rarely occur
* If you modify width/height manually, ensure **image size matches width/height**

**C) OOM (CUDA out of memory)**
* Use `--pre-resize 1m`
* Don't use `--no-offload` (keep offload enabled)

---

### Tips: Cache Management

Downloaded models are stored in the Hugging Face cache. To free up disk space:

```bash
# Clear pip cache
pip cache purge

# View Hugging Face cache
hf cache ls

# Delete specific model cache
hf cache rm <revision_id>
```

Default cache locations:
* Windows: `C:\Users\<username>\.cache\huggingface\hub`
* Linux/Mac: `~/.cache/huggingface/hub`

---

### Notes

#### Z-Image Turbo (4bit) `simple_image_edit_zit.py`

* Model: [unsloth/Z-Image-Turbo-unsloth-bnb-4bit](https://huggingface.co/unsloth/Z-Image-Turbo-unsloth-bnb-4bit)

Tested but could not achieve satisfactory performance for our requirements.

#### FLUX.2 [klein] 4B `simple_image_edit_flux2_klein.py`

* Model: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

Worked well, but requires latest diffusers and cannot coexist with the Nunchaku version. Separate venv required.

> `pip install -U git+https://github.com/huggingface/diffusers`

#### Qwen-Image-Edit-Rapid-AIO-V23 (Diffusers-compatible) `simple_image_edit_rapid_qwen.py`

* Model: [prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23](https://huggingface.co/prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23)

Worked well, but too slow (~30 minutes per image).

---

## Credits

* Nunchaku Lightning example structure (scheduler config, model_path format, offload branching) follows the official reference sample

---

2026.02 id-fa
