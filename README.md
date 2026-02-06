# Simple Image Editing Script with diffusers

# diffusersを使った単一スクリプトでAI画像編集を行うコードサンプル

**Qwen-Image-Edit-2509 Lightning (Nunchaku版)**: `simple_image_edit_nunchaku_qwen.py`

**「画像ファイル1枚を引数に取って、編集した画像を出力」**する用途を想定しています。

> **対象環境:** GeForce RTX 3xxx (VRAM 12GB) 程度の環境で動作させることを目指してモデルを選定・調整しています。より高性能なGPUでは `--no-offload` や高解像度での処理も可能です。

> **備考:** Z-Image Turbo (4bit)もテストしましたが要求を満たす性能を発揮させることができませんでした。一応検証したスクリプト(`simple_image_edit_zit.py`)を残してあります。

> **備考:** FLUX.2 [klein] 4B は利用できましたが、diffusers最新版を要求するためnunchaku版Qwen-Image-Editとは共存できません。venvを分ける必要があります。(`simple_image_edit_flux2_klein.py`)
> `pip install -U git+https://github.com/huggingface/diffusers`

---

## Qwen-Image-Edit-2509 Lightning (Nunchaku版)

### 概要

Nunchaku の Lightning 重み（`.safetensors`）を使って `Qwen/Qwen-Image-Edit-2509` を高速に動かします。
サンプル構成（scheduler / model_path / VRAMに応じた offload 分岐）に沿って実装しています。

### 特徴

* 入力：画像1枚（※公式のサンプルコードは複数画像にも対応しますが、本スクリプトは単一入力用）
* `--prompt "..."`：プロンプトを指定（省略時はソースコード内の `PROMPT` を使用）
* `--pre-resize 1m|2m`：アスペクト比維持で総画素数を縮小
* 最大サイズ：**2048x2048 に収める（縮小のみ）**
* **高さを16の倍数にパディング**（リサイズではなく余白で調整）
* VRAMに余裕がある場合：`enable_model_cpu_offload()`
* 低VRAMの場合：`transformer.set_offload(...num_blocks_on_gpu=...)` + `enable_sequential_cpu_offload()`（サンプル踏襲）
* `--no-offload`：オフロード無効（速いがVRAMに余裕が必要）
* `--seed N`：乱数シード指定（省略時はランダム）
* `--progress` / `--mem-log`

> **Tip:** 常に同じプロンプトを使用する場合は、スクリプト内の `PROMPT = "..."` 行を直接編集してください。

### 実行例

```powershell
py .\simple_image_edit_nunchaku_qwen.py .\sample.png
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --prompt "Remove text and enhance image quality."
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --pre-resize 2m --progress --mem-log
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --no-offload
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
* nunchaku

---

## インストール例（Windows / PowerShell）

### 1) venv 作成

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
```

### 2) 依存のインストール（例）

```powershell
pip install -U pillow huggingface_hub psutil transformers accelerate safetensors
pip install "diffusers>=0.36.0,<0.37.0"
```

nunchaku はコンパイル済のwheel版をpipで導入するのが簡単です。

```powershell
py -m pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl
```
> Python 3.11、Torch2.10+cu130の場合

**重要:** Nunchaku 1.2.1 は `diffusers==0.36.x` が必要です。git main版（0.37.0.dev）はAPIが変更されており、`pos_embed`関連のエラーが発生します。

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

## English Documentation

This script is designed to **take a single image file as input and output an edited image**.

> **Target Environment:** Models are selected and tuned to run on GeForce RTX 3xxx (12GB VRAM) class hardware. Higher-end GPUs can use `--no-offload` or process higher resolutions.

> **Note:** Z-Image Turbo (4bit) was also tested but could not achieve satisfactory performance for our requirements. The test script (`simple_image_edit_zit.py`) has been kept for reference.

### Qwen-Image-Edit-2509 Lightning (Nunchaku)

Runs `Qwen/Qwen-Image-Edit-2509` at high speed using Nunchaku Lightning weights (`.safetensors`). Implements the official sample structure (scheduler / model_path / VRAM-based offload branching).

**Features:**
- Input: Single image (official sample supports multiple images, but this script is for single input)
- `--prompt "..."`: Specify prompt (uses `PROMPT` constant in source if omitted)
- `--pre-resize 1m|2m`: Reduce total pixels while maintaining aspect ratio
- Max size: **Fits within 2048x2048 (downscale only)**
- **Height padded to multiple of 16** (padding, not resizing)
- High VRAM: `enable_model_cpu_offload()`
- Low VRAM: `transformer.set_offload(...num_blocks_on_gpu=...)` + `enable_sequential_cpu_offload()` (follows official sample)
- `--no-offload`: Disable offloading (faster but requires more VRAM)
- `--seed N`: Random seed (random if omitted)
- `--progress` / `--mem-log`

> **Tip:** For a fixed prompt, edit the `PROMPT = "..."` line in the script directly.

**Examples:**
```bash
python simple_image_edit_nunchaku_qwen.py ./sample.png
python simple_image_edit_nunchaku_qwen.py ./sample.png --prompt "Remove text and enhance image quality."
python simple_image_edit_nunchaku_qwen.py ./sample.png --pre-resize 2m --progress --mem-log
python simple_image_edit_nunchaku_qwen.py ./sample.png --no-offload
```

### Requirements

- OS: Windows / Linux (CUDA required)
- GPU: NVIDIA recommended
- Python: 3.10-3.12 recommended (3.11 is most stable)
- PyTorch: **CUDA version** (e.g., `cu130` depends on your installed PyTorch CUDA build)

**Dependencies:**
- torch (CUDA version)
- diffusers
- transformers, accelerate, safetensors
- pillow
- huggingface_hub
- psutil (for `--mem-log`)
- nunchaku

### Installation (Windows / PowerShell)

**1) Create venv:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
```

**2) Install dependencies:**
```powershell
pip install -U pillow huggingface_hub psutil transformers accelerate safetensors
pip install "diffusers>=0.36.0,<0.37.0"
```

For Nunchaku, install the pre-compiled wheel:
```powershell
py -m pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl
```
> For Python 3.11, Torch 2.10+cu130

**Important:** Nunchaku 1.2.1 requires `diffusers==0.36.x`. The git main version (0.37.0.dev) has API changes that cause `pos_embed` related errors.

**Note:** First run will download models (**30-100GB total**). Models are cached in the Hugging Face cache directory.

### Troubleshooting

**A) Stuck at 0% (especially first run)**
- First run involves heavy model download and loading, which may appear stuck
- Use `--progress` to show download progress
- Use `--pre-resize 1m` for large inputs

**B) Shape mismatch (width/height vs input image latents)**
- This script **caps input to 2048 and pads to multiples**, so this should rarely occur
- If you modify width/height manually, ensure **image size matches width/height**

**C) OOM (CUDA out of memory)**
- Use `--pre-resize 1m`
- Don't use `--no-offload` (keep offload enabled)

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
- Windows: `C:\Users\<username>\.cache\huggingface\hub`
- Linux/Mac: `~/.cache/huggingface/hub`

---

## Credits

- Nunchaku Lightning example structure (scheduler config, model_path format, offload branching) follows the official reference sample

---

## License

License for scripts in this repository: Add as appropriate (e.g., MIT).
Models and dependencies follow their respective distribution licenses.
