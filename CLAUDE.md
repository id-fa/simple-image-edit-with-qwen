# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python scripts for AI-powered image editing using diffusion models:

### CLI Scripts
1. **simple_image_edit_nunchaku_qwen.py** - Qwen-Image-Edit-2509 Lightning (Nunchaku-optimized) pipeline **(main)**
2. **simple_image_edit_rapid_qwen.py** - Qwen-Image-Edit-Rapid-AIO-V23 (4-step accelerated, no nunchaku)
3. **simple_image_edit_gguf_qwen.py** - Qwen-Image-Edit-Rapid-AIO-V23 GGUF quantized (Q3_K default, low VRAM)
4. **simple_image_edit_flux2_klein.py** - FLUX.2 Klein 4B pipeline (requires separate venv with latest diffusers)
5. **simple_i2v_ltx2_distilled.py** - LTX-2 19B Distilled video generator [diffusers] (fp8 layerwise casting, i2v / flf2v / t2v)
6. **simple_i2v_ltx2_native.py** - LTX-2 19B Distilled video generator [native ltx-pipelines] (fp8 checkpoint direct, i2v / flf2v / t2v)
7. **simple_image_edit_zit.py** - Z-Image Turbo (4bit) img2img pipeline (archived; did not meet quality requirements)

### Web Server Scripts (`server/`)
8. **server/app_nunchaku.py** - Flask web server for Nunchaku Qwen-Image-Edit-2509 Lightning
9. **server/app_gguf.py** - Flask web server for GGUF quantized Qwen-Image-Edit-Rapid-AIO-V23
10. **server/app_aio.py** - Flask web server for Qwen-Image-Edit-Rapid-AIO-V23 (no GGUF/nunchaku dependency)
11. **server/app_comfyui.py** - Flask web server using ComfyUI API backend for AIO workflow (no torch/diffusers dependency)
12. **server/app_comfyui_nunchaku.py** - Flask web server using ComfyUI API backend for Nunchaku workflow (no torch/diffusers dependency)

### Google Colab Notebooks (`notebooks/`)
13. **notebooks/colab_app_gguf.ipynb** - Google Colab notebook for GGUF web server (cloudflared tunnel, A100 recommended)

### Utilities
14. **server/nunchaku_lora_qwen.py** - LoRA loader for NunchakuQwenImageTransformer2DModel (ported from ComfyUI-QwenImageLoraLoader)

### Workflow Templates (`server/comfyui_workflow/`)
15. **server/comfyui_workflow/comfyui_qwen_image_edit_AIO_v23_api.json** - ComfyUI API format workflow for Qwen-Rapid-AIO-NSFW-v23 (used by app_comfyui.py)
16. **server/comfyui_workflow/comfyui_qwen_image_edit_nunchaku_api.json** - ComfyUI API format workflow for Nunchaku Qwen-Image-Edit-2509 Lightning (used by app_comfyui_nunchaku.py)

Image editing scripts take a single image as input (with optional `--ref` reference images) and output an edited version. All image scripts support `--t2i` mode for text-to-image generation. The video script (`simple_i2v_ltx2_distilled.py`) supports i2v (image-to-video), flf2v (first+last frame to video via `--ref`), and t2v (text-to-video via `--t2i`). Prompts can be specified via `--prompt` argument or by editing the `PROMPT` constant in the source.

Web servers provide browser GUI with password protection, job queue (1 processing + 2 waiting), real-time step progress, cancel functionality, and optional gallery mode with password-gated login.

**Target Environment:** GeForce RTX 3xxx (VRAM 12GB) class hardware. Higher-end GPUs can use `--no-offload` or process higher resolutions.

**Important:** Qwen (nunchaku/GGUF) and FLUX.2 Klein/LTX-2 cannot coexist in the same venv due to diffusers version conflicts. Nunchaku and GGUF can share the same venv (`diffusers==0.36.x`). Rapid Qwen, FLUX.2 Klein, and LTX-2 can share the same venv (latest diffusers). ComfyUI servers (`app_comfyui.py`, `app_comfyui_nunchaku.py`) have no torch/diffusers dependency and use their own lightweight venv.

## Running the Scripts

### Qwen-Image-Edit Lightning (Nunchaku) — requires diffusers 0.36.x
```powershell
py .\simple_image_edit_nunchaku_qwen.py .\sample.png
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --prompt "Enhance quality." --seed 42
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --pre-resize 2m --progress
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --ref ref2.png --ref ref3.png --prompt "Combine elements"
py .\simple_image_edit_nunchaku_qwen.py --t2i --prompt "A cat sitting on a windowsill" --seed 42
py .\simple_image_edit_nunchaku_qwen.py --t2i --prompt "A landscape" --size 1536x1024
```

### Qwen-Image-Edit-Rapid-AIO-V23 — requires recent diffusers
```powershell
py .\simple_image_edit_rapid_qwen.py .\sample.png
py .\simple_image_edit_rapid_qwen.py .\sample.png --prompt "Enhance quality." --seed 42
py .\simple_image_edit_rapid_qwen.py .\sample.png --pre-resize 2m --offload --progress
py .\simple_image_edit_rapid_qwen.py .\sample.png --ref ref2.png --prompt "Combine elements"
py .\simple_image_edit_rapid_qwen.py --t2i --prompt "A landscape painting" --size 1536x1024
```

### Qwen-Image-Edit-Rapid GGUF — requires diffusers 0.36.x + gguf
```powershell
py .\simple_image_edit_gguf_qwen.py .\sample.png
py .\simple_image_edit_gguf_qwen.py .\sample.png --prompt "Enhance quality." --seed 42
py .\simple_image_edit_gguf_qwen.py .\sample.png --pre-resize 2m --offload --progress
py .\simple_image_edit_gguf_qwen.py .\sample.png --gguf-file v23/Qwen-Rapid-NSFW-v23_Q2_K.gguf
py .\simple_image_edit_gguf_qwen.py .\sample.png --ref ref2.png --ref ref3.png --pre-resize 1m
py .\simple_image_edit_gguf_qwen.py --t2i --prompt "A mountain scene" --seed 42
```

### FLUX.2 Klein 4B — requires diffusers latest (git main)
```powershell
py .\simple_image_edit_flux2_klein.py .\sample.png
py .\simple_image_edit_flux2_klein.py .\sample.png --prompt "Enhance quality." --seed 42
py .\simple_image_edit_flux2_klein.py .\sample.png --pre-resize 1m --offload --progress
py .\simple_image_edit_flux2_klein.py .\sample.png --ref ref2.png --ref ref3.png --ref ref4.png
py .\simple_image_edit_flux2_klein.py --t2i --prompt "A futuristic city" --size 1024x768
```

### LTX-2 19B Distilled [diffusers] — requires diffusers latest (git main)
```powershell
py .\simple_i2v_ltx2_distilled.py .\sample.png
py .\simple_i2v_ltx2_distilled.py .\sample.png --prompt "Camera slowly pans right" --seed 42
py .\simple_i2v_ltx2_distilled.py .\sample.png --ref last_frame.png --prompt "Smooth transition"
py .\simple_i2v_ltx2_distilled.py .\sample.png --no-stage2 --offload --num-frames 61
py .\simple_i2v_ltx2_distilled.py .\sample.png --size 768x512 --pre-resize 0.3m
py .\simple_i2v_ltx2_distilled.py --t2i --prompt "A cat playing with yarn" --size 768x512 --seed 42
py .\simple_i2v_ltx2_distilled.py .\sample.png --no-fp8  # bf16 (no fp8 casting)
```

### LTX-2 19B Distilled [native] — requires ltx-core + ltx-pipelines
```powershell
# モデルファイルをローカルにダウンロードして使用
py .\simple_i2v_ltx2_native.py .\sample.png --checkpoint models\ltx-2-19b-distilled-fp8.safetensors --upsampler models\ltx-2-spatial-upscaler-x2-1.0.safetensors --gemma-root models\gemma-3-4b-it
py .\simple_i2v_ltx2_native.py .\sample.png --checkpoint models\ltx-2-19b-distilled-fp8.safetensors --upsampler models\ltx-2-spatial-upscaler-x2-1.0.safetensors --gemma-root models\gemma-3-4b-it --prompt "Camera slowly zooms in" --seed 42
py .\simple_i2v_ltx2_native.py .\sample.png --ref last_frame.png --checkpoint ... --upsampler ... --gemma-root ...
py .\simple_i2v_ltx2_native.py --t2i --prompt "A cat playing" --checkpoint ... --upsampler ... --gemma-root ...
py .\simple_i2v_ltx2_native.py .\sample.png --no-audio --checkpoint ... --upsampler ... --gemma-root ...
```

### Common Options
- `--prompt "..."` - Specify prompt (uses `PROMPT` constant if omitted)
- `--seed N` - Random seed for reproducibility (random if omitted)
- `--pre-resize 1m|2m` - Reduce total pixels while maintaining aspect ratio
- `--ref FILE` - Add reference image (repeatable; Qwen: max 2, FLUX.2 Klein: max 3, Z-Image: ignored)
- `--t2i` - Text-to-image mode (no input image required; generates from white blank image). Requires `--prompt`. Incompatible with `--ref` and `--pre-resize`
- `--size WxH` - Output size for `--t2i` mode (default: 1024x1024, e.g. `--size 1536x1024`)
- `--progress` - Show Hugging Face download progress
- `--mem-log` - Display memory usage (requires psutil)
- `--no-offload` - Disable offloading (requires high VRAM)
- `--lora REPO_OR_PATH` - LoRA weights (HF repo ID or local path; Nunchaku: `nunchaku_lora_qwen.py`, others: diffusers API). Can also be set via `LORA` constant in source
- `--lora-weight-name FILE` - Weight file name within HF repo (optional; auto-detected if omitted). Can also be set via `LORA_WEIGHT_NAME` constant
- `--lora-scale N` - LoRA strength (default: 1.0; Nunchaku: `set_lora_strength()`, others: `fuse_lora()` when != 1.0). Can also be set via `LORA_SCALE` constant

### Script-specific Options
- `--offload` (FLUX.2 Klein, Rapid Qwen, GGUF Qwen) - Sequential CPU offload for low VRAM
- `--steps N` (all) - Override inference step count
- `--rank N` (Nunchaku Qwen only) - Nunchaku rank (32 or 128)
- `--num-blocks-on-gpu N` (Nunchaku Qwen only) - Blocks to keep on GPU in low-VRAM mode
- `--guidance-scale N` (FLUX.2 Klein, Rapid Qwen, GGUF Qwen) - Override guidance scale
- `--true-cfg-scale N` (Rapid Qwen, GGUF Qwen) - Override true CFG scale
- `--negative-prompt "..."` (Rapid Qwen, GGUF Qwen) - Override negative prompt
- `--gguf-repo REPO` (GGUF Qwen) - HuggingFace GGUF repo (default: `Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF`)
- `--gguf-file PATH` (GGUF Qwen) - GGUF file within repo (default: `v23/Qwen-Rapid-NSFW-v23_Q3_K.gguf`)
- `--gguf-local PATH` (GGUF Qwen) - Use local GGUF file directly
- `--num-frames N` (LTX-2) - Number of video frames (default: 121)
- `--frame-rate N` (LTX-2) - Video frame rate (default: 24.0)
- `--steps-s2 N` (LTX-2) - Stage 2 inference steps (default: 3)
- `--no-stage2` (LTX-2) - Skip Stage 2 (latent upsample + refinement)
- `--no-fp8` (LTX-2 both) - Disable fp8 quantization (load in bf16)
- `--no-audio` (LTX-2 both) - Output video without audio
- `--ref FILE` (LTX-2 both) - Last frame image for first+last frame to video (single file only)
- `--checkpoint PATH` (LTX-2 native) - Distilled checkpoint .safetensors path
- `--upsampler PATH` (LTX-2 native) - Spatial upsampler .safetensors path
- `--gemma-root DIR` (LTX-2 native) - Gemma 3 text encoder directory path
- `--enhance-prompt` (LTX-2 native) - Enable Gemma prompt enhancement

## Running the Web Servers

### Nunchaku Web Server
```powershell
cd server
python app_nunchaku.py
python app_nunchaku.py --password mysecret --port 8080
python app_nunchaku.py --host 0.0.0.0 --no-progress
python app_nunchaku.py --no-offload --rank 128
python app_nunchaku.py --steps 4              # 4-step model
python app_nunchaku.py --lora "path/to/lora.safetensors"
python app_nunchaku.py --lora "repo_id::weight.safetensors" --lora "another.safetensors"
python app_nunchaku.py --gallery --password mysecret
python app_nunchaku.py --preset "高画質化::Enhance quality." --preset "テキスト除去::Remove all text."
```

### GGUF Web Server
```powershell
cd server
python app_gguf.py
python app_gguf.py --password mysecret --port 8080
python app_gguf.py --gguf-file "v23/Qwen-Rapid-NSFW-v23_Q2_K.gguf"
python app_gguf.py --gguf-local "path/to/model.gguf"
python app_gguf.py --lora "HF_REPO_ID::weights.safetensors" --lora "another.safetensors"
python app_gguf.py --gallery --password mysecret
```

### AIO Web Server
```powershell
cd server
python app_aio.py
python app_aio.py --password mysecret --port 8080
python app_aio.py --offload                        # sequential CPU offload (low VRAM)
python app_aio.py --lora "path/to/lora.safetensors" --lora "another.safetensors"
python app_aio.py --gallery --password mysecret
```

### ComfyUI Web Server — AIO workflow (API backend)
```powershell
cd server
python app_comfyui.py
python app_comfyui.py --password mysecret --port 5000
python app_comfyui.py --comfyui-url http://192.168.1.100:8188
python app_comfyui.py --comfyui-path D:\ComfyUI          # auto-register LoRA path + reboot ComfyUI
python app_comfyui.py --steps 4 --cfg 1.0
python app_comfyui.py --gallery --password mysecret
python app_comfyui.py --preset "高画質化::Enhance quality." --preset "テキスト除去::Remove all text."
```

### ComfyUI Web Server — Nunchaku workflow (API backend)
```powershell
cd server
python app_comfyui_nunchaku.py
python app_comfyui_nunchaku.py --password mysecret --port 5000
python app_comfyui_nunchaku.py --comfyui-url http://192.168.1.100:8188
python app_comfyui_nunchaku.py --comfyui-path D:\ComfyUI
python app_comfyui_nunchaku.py --steps 4 --cfg 1.0
python app_comfyui_nunchaku.py --gallery --password mysecret
python app_comfyui_nunchaku.py --preset "高画質化::Enhance quality." --preset "テキスト除去::Remove all text."
```

### Web Server Common Options
- `--host HOST` - Bind host (default: 127.0.0.1)
- `--port PORT` - Bind port (default: 5000)
- `--password PW` - Generation password (default: "password")
- `--no-progress` - Hide HF download progress (shown by default)
- `--no-offload` - Disable offloading (high VRAM)
- `--lora REPO_OR_PATH` - LoRA weights (repeatable). Format: `path_or_repo` or `repo_id::weight_name`. Multiple `--lora` args register multiple LoRAs
- `--gallery` - Enable gallery mode (show generation history with image reuse)
- `--preset "label::prompt"` - Prompt preset button (repeatable). Omit `label::` for auto-numbered labels (`preset1`, `preset2`, ...)

### Web Server Features
- Browser GUI with password protection
- Job queue: 1 processing + 2 waiting, 3+ returns 503 BUSY
- Real-time step progress via polling (`callback_on_step_end`)
- Cancel running jobs (`pipeline._interrupt`)
- Auto-cleanup: files older than 1 hour removed every 5 minutes
- Pre-resize: 0.3M or 1M pixels
- t2i mode: 1024x1024 fixed size
- Prompt presets (`--preset`): Configurable buttons above prompt textarea. Click to fill prompt with preset text. Confirmation dialog when replacing non-empty prompt
- Prompt clear button (x) next to label for clearing prompt text
- Prompt translation (googletrans): `Translate:` label with `-> EN` / `-> ZH` / `-> JA` buttons
- File input clear buttons (x) for resetting image selections, drag-and-drop image upload
- Continuous mode: Checkbox to auto-set generation result as Img1 for iterative editing. Result area also shows Img1/Img2 radio buttons for immediate reuse without waiting for gallery refresh
- Preview during generation (ComfyUI only): Checkbox to show/hide real-time preview images from ComfyUI's WebSocket binary frames. Requires ComfyUI preview method enabled (Settings → Preview Method). Preview image displayed below progress bar, cleared on completion
- Polling safety: Auto-stops polling after 2 min with no response (timeout) or 3 consecutive auth errors
- Multi-LoRA support: `--lora` repeatable, WebUI shows checkboxes + strength sliders (0.0-2.0, default unchecked). Selected LoRAs applied dynamically per generation. Nunchaku: pipeline reloads when LoRA config changes (CUDA kernels cache buffers, no runtime toggle). AIO/GGUF: `set_adapters()` for instant switching. Incompatible LoRAs are skipped gracefully (no crash)
- Model info display: pipeline, transformer, text encoder class, tokenizer, VAE class, dtype, LoRA
- Error messages: Japanese/English bilingual (i18n)
- Gallery mode (`--gallery`): Browse past generation history, click thumbnails to enlarge, download links on each image, reuse gallery images as input for new generations via radio button selection (Img1/Img2 slots). Selecting a gallery image automatically disables t2i mode. Selected image shown as thumbnail + ID text below the file input
- Gallery login gate: When `--gallery` is enabled, a password login screen is shown first; all gallery/result/input APIs require password authentication via query parameter
- User identification: Each gallery entry shows a hashed user ID (SHA-256 of IP + User-Agent, 8 chars) to distinguish generators. Uses `X-Forwarded-For` header when available (reverse proxy support)
- Gallery entry deletion: Users can delete their own gallery entries (user_hash match required). Deleted entries remain visible as placeholders showing timestamp, user ID, and bilingual deletion message. Files are not removed (cleaned up by auto-cleanup). `DELETE /api/gallery/<job_id>` sets job status to `"hidden"`
- Drawing editor: Click any image (gallery thumbnails, upload previews, generated results) to open a full-screen drawing editor with:
  - Dual-canvas architecture: background layer (original image) + transparent overlay layer (drawings)
  - Tools: Pen (freehand drawing), Brush (pressure-sensitive pen — line width scales with pen pressure), Airbrush (pressure-sensitive spray — radius and density scale with pressure), Eraser (removes overlay pixels only via `destination-out`), Cover (white paint over original image)
  - Color palette: 10 preset colors + custom color picker
  - Line sizes: 1, 2, 4, 8, 14, 24, 64 pixels
  - Undo history (up to 30 steps)
  - Copy & paste: Select tool for region selection, copy (bg/draw/composite), paste with drag-move and corner-resize. Lock ratio checkbox for aspect-ratio-locked resize
  - Save options: composite (bg + overlay merged), line-only (overlay layer only)
  - Draft/pause: Save both layers as a resumable draft with DRAFT label thumbnail. Drafts cannot be used as Img1/Img2
  - Saved drawings appear as thumbnails with Img1/Img2 radio buttons and download/delete controls
  - Private to user (user_hash-based access control)
  - Works in both gallery and non-gallery modes
  - Blank sketch button: Start drawing on a 1024x1024 white canvas without any source image
  - File input auto-clear: Selecting Img1/Img2 via radio buttons clears corresponding file input to avoid ambiguity. Conversely, uploading a file (file picker or drag-and-drop) auto-clears the corresponding gallery slot selection
- Drawing API routes: `POST /api/drawing/save`, `GET /api/drawing/<id>`, `GET /api/drawing/<id>/bg`, `GET /api/drawing/<id>/overlay`, `GET /api/drawings`, `DELETE /api/drawing/<id>`

### Web Server-specific Options
- `--steps N` (Nunchaku only) - Inference steps: 4 or 8 (default: 8)
- `--rank N` (Nunchaku only) - Nunchaku rank (32 or 128)
- `--num-blocks-on-gpu N` (Nunchaku only) - Blocks on GPU in low-VRAM mode
- `--gguf-file PATH` (GGUF only) - GGUF file within HF repo (default: `v23/Qwen-Rapid-NSFW-v23_Q3_K.gguf`)
- `--gguf-local PATH` (GGUF only) - Use local GGUF file directly
- `--offload` (AIO only) - Sequential CPU offload for low VRAM
- `--comfyui-url URL` (ComfyUI only) - ComfyUI API URL (default: `http://127.0.0.1:8188`)
- `--comfyui-path DIR` (ComfyUI only) - ComfyUI installation directory. Auto-registers `server/LoRA/` in `extra_model_paths.yaml` and reboots ComfyUI via Manager API if needed
- `--steps N` (ComfyUI only) - Inference steps (default: 8)
- `--cfg N` (ComfyUI only) - CFG scale (default: 1.0)

**Note:** GGUF server does NOT support `--offload` (sequential CPU offload is incompatible with GGUF tensors). Only default (`enable_model_cpu_offload`) and `--no-offload` are available.

**Note:** AIO server requires significantly more VRAM than GGUF (full bf16 transformer). On Google Colab, use GGUF version instead (A100 recommended).

**Note:** ComfyUI servers (`app_comfyui.py`, `app_comfyui_nunchaku.py`) require a running ComfyUI instance. No torch/diffusers dependency — only flask, requests, pillow, websocket-client. AIO variant requires `Qwen-Rapid-AIO-NSFW-v23.safetensors`, `qwen2.5-vl-7b-instruct-abliterated.safetensors`, and `qwen_image_vae.safetensors`. Nunchaku variant requires `qwen_2.5_vl_7b_fp8_scaled.safetensors`, `qwen_image_vae.safetensors`, and `NunchakuQwenImageLoraStackV3` custom node. LoRA files in `server/LoRA/` must be in ComfyUI's search path (use `--comfyui-path` for auto-registration).

## Environment Setup

Python 3.10-3.12 (3.11 recommended). First run downloads large models.

### venv for Qwen (nunchaku)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -U pip pillow huggingface_hub psutil transformers accelerate safetensors
pip install "diffusers>=0.36.0,<0.37.0"
py -m pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl
```

**Critical:** Nunchaku 1.2.1 requires `diffusers==0.36.x`. Version 0.37+ has breaking API changes causing `pos_embed` errors.

### Adding GGUF support to the Nunchaku venv
```powershell
.\.venv\Scripts\activate
pip install -U gguf
```

GGUF Qwen requires `diffusers>=0.36.0`, so it can share the Nunchaku venv (`diffusers==0.36.x`).

### venv for Rapid Qwen / FLUX.2 Klein / LTX-2 diffusers (requires latest diffusers)
```powershell
py -3.11 -m venv .venv-flux2
.\.venv-flux2\Scripts\activate
pip install -U pip pillow huggingface_hub psutil transformers accelerate safetensors
pip install -U git+https://github.com/huggingface/diffusers
```

### venv for LTX-2 native (official ltx-pipelines)
```powershell
py -3.11 -m venv .venv-ltx2
.\.venv-ltx2\Scripts\activate
pip install -U pip pillow ltx-core ltx-pipelines
```

**Note:** LTX-2 native requires local model files (not auto-downloaded). Download from `https://huggingface.co/Lightricks/LTX-2`:
- `ltx-2-19b-distilled-fp8.safetensors` (checkpoint)
- `ltx-2-spatial-upscaler-x2-1.0.safetensors` (upsampler)
- Gemma 3 text encoder (e.g. `google/gemma-3-4b-it`)

### venv for ComfyUI server (no torch/diffusers needed)
```powershell
py -3.11 -m venv .venv-comfyui
.\.venv-comfyui\Scripts\activate
pip install flask requests pillow websocket-client googletrans==4.0.0rc1
```

**Note:** ComfyUI server delegates all inference to a running ComfyUI instance via API. Requires ComfyUI with `Qwen-Rapid-AIO-NSFW-v23.safetensors`, `qwen2.5-vl-7b-instruct-abliterated.safetensors`, and `qwen_image_vae.safetensors` models. ComfyUI-Manager extension enables auto-reboot for LoRA path registration.

## Architecture

### Image Processing Pipeline

All scripts share this preprocessing flow:

1. **Pre-resize** (optional): Reduce total pixels to 1M/2M while keeping aspect ratio
2. **Max Size Cap**: Scale down to fit within max bounds (image scripts: 2048×2048, LTX-2: 1280×720)
3. **Alignment by Padding**: Pad to multiples (image scripts: 16px, LTX-2: 32px)

### Memory Management

**Qwen Lightning**:
- Nunchaku SVDQ-optimized transformer
- Prefers bf16, falls back to fp16
- Adaptive offload based on GPU memory:
  - High VRAM (>18GB): `enable_model_cpu_offload()`
  - Low VRAM: `transformer.set_offload()` + `enable_sequential_cpu_offload()`
- Warns if incompatible diffusers version detected
- LoRA: via `server/nunchaku_lora_qwen.py` (ported from ComfyUI-QwenImageLoraLoader). Handles QKV fusion, GLU fusion, proj_out split for nunchaku quantized modules. Standard diffusers LoRA files work (no nunchaku-specific format needed). Skips AWQ modulation layers (img_mod/txt_mod) as they are too sensitive
- LoRA runtime switching: Nunchaku CUDA kernels cache internal buffers at initialization — LoRA cannot be toggled at runtime. Web server detects LoRA config changes between requests and reloads the entire pipeline from scratch (destroy + gc + cuda.empty_cache + recreate). Same config skips reload

**Qwen Rapid-AIO-V23**:
- 4-step accelerated transformer (from `prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23`)
- Base pipeline: `Qwen/Qwen-Image-Edit-2511`
- Prefers bf16, falls back to fp16
- No nunchaku dependency; uses standard `QwenImageTransformer2DModel`
- 3-tier offload: default `enable_model_cpu_offload()`, `--offload` for sequential, `--no-offload` for full GPU
- Supports negative prompt and true CFG scale
- LoRA: diffusers API (`load_lora_weights` at startup, `set_adapters` per-request). Incompatible LoRAs skipped gracefully

**Qwen Rapid GGUF**:
- GGUF quantized version of Rapid-AIO-V23 (from `Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF`)
- Default: Q3_K (~10GB), also available: Q2_K (~7.4GB) through Q8_0 (~21.8GB)
- Loaded via `from_single_file` + `GGUFQuantizationConfig`
- Base pipeline: `Qwen/Qwen-Image-Edit-2511`
- Transformer config: local `qwen-image-edit-transformer-config/config.json`
- 3-tier offload: default `enable_model_cpu_offload()`, `--offload` for sequential, `--no-offload` for full GPU
- Requires `gguf` package
- LoRA: diffusers API (`load_lora_weights` at startup, `set_adapters` per-request). Incompatible LoRAs skipped gracefully

**FLUX.2 Klein 4B**:
- 4B parameter rectified flow transformer (distilled)
- Prefers bf16, falls back to fp16
- 3-tier offload: default `enable_model_cpu_offload()`, `--offload` for sequential, `--no-offload` for full GPU
- Model: `black-forest-labs/FLUX.2-klein-4B` (Apache 2.0)
- Text encoder: Mistral3 (via transformers)
- Image passed as conditioning reference via `image` parameter

**LTX-2 19B Distilled [diffusers]** (`simple_i2v_ltx2_distilled.py`):
- 19B parameter DiT-based video generation model (distilled, 8-step Stage1)
- Pipeline: `LTX2ConditionPipeline` (unified for i2v/flf2v/t2v via `LTX2VideoCondition`)
- Two-stage generation: Stage1 (8 steps, `DISTILLED_SIGMA_VALUES`) → latent upsample (2x) → Stage2 (3 steps, `STAGE_2_DISTILLED_SIGMA_VALUES`)
- fp8 via `enable_layerwise_casting(storage_dtype=float8_e4m3fn)` by default (`--no-fp8` to disable)
- Prefers bf16, falls back to fp16
- 3-tier offload: default `enable_model_cpu_offload()`, `--offload` for sequential, `--no-offload` for full GPU
- VAE tiling enabled automatically for Stage 2
- Model: `rootonchair/LTX-2-19b-distilled` (HF diffusers format)
- Output: MP4 with audio (via vocoder at 24kHz, `--no-audio` to disable)
- Size: max 1280x720, aligned to 32px multiples

**LTX-2 19B Distilled [native]** (`simple_i2v_ltx2_native.py`):
- Same model but via official `ltx-pipelines` codebase (not diffusers)
- Pipeline: `DistilledPipeline` (two-stage distilled with built-in spatial upsampler)
- Native fp8 checkpoint support (`ltx-2-19b-distilled-fp8.safetensors` loaded directly)
- No HF `from_pretrained`; requires local model files (checkpoint, upsampler, Gemma)
- Quantization: `QuantizationPolicy.fp8_cast()` by default (`--no-fp8` to disable)
- Image conditioning via `ImageConditioningInput(path, frame_idx, strength, crf)`
- Output: MP4 with audio (via PyAV H.264, `--no-audio` to disable)
- Size: max 1280x768, aligned to 64px multiples
- Install: `pip install ltx-core ltx-pipelines`

**Z-Image Turbo** (archived):
- 4bit quantization via bitsandbytes (unsloth model)
- `--offload` enables CPU offloading

**ComfyUI API Backend — AIO** (`server/app_comfyui.py`):
- No torch/diffusers dependency — delegates inference to running ComfyUI instance
- Workflow template loaded from `server/comfyui_workflow/comfyui_qwen_image_edit_AIO_v23_api.json`
- Workflow node IDs centralized in `WF_NODE` dict for easy remapping when workflow is re-exported
- Images uploaded to ComfyUI via `POST /upload/image`, workflow submitted via `POST /prompt`
- Progress tracking via WebSocket (`websocket-client`); binary frames parsed for preview images, falls back to polling if not installed
- Preview images: ComfyUI binary WebSocket frames (8-byte header + JPEG/PNG) displayed during generation via `/api/preview/<job_id>`. Togglable via "Show preview during generation" checkbox
- Cancel via `POST /interrupt` on ComfyUI API
- Result retrieved from `GET /history/{prompt_id}` → `GET /view`
- Pre-resize: `ImageScaleToTotalPixels` nodes in workflow, megapixels set from pre-resize selection (0.3M/1M), `resolution_steps=16` for model alignment
- LoRA: scans `server/LoRA/` folder, matches against ComfyUI's known LoRAs via `/object_info/LoraLoaderModelOnly`. Workflow has 3 `LoraLoaderModelOnly` slots chained (UNETLoader → slot1 → slot2 → slot3 → ModelSamplingAuraFlow); unused slots are removed from workflow and chain is rewired
- `--comfyui-path`: auto-registers LoRA path in `extra_model_paths.yaml` and reboots ComfyUI via Manager API (`GET /manager/reboot`)
- Steps/CFG: set directly on KSampler node
- HTML template: standalone `server/app_comfyui_template.html` (no app_aio.py dependency), includes ComfyUI-specific UI (preview checkbox, preview area)
- Startup checks: ComfyUI connectivity (60s retry), required model availability (UNET/CLIP/VAE), LoRA path registration

**ComfyUI API Backend — Nunchaku** (`server/app_comfyui_nunchaku.py`):
- Same architecture as AIO variant but for the Nunchaku workflow
- Workflow template loaded from `server/comfyui_workflow/comfyui_qwen_image_edit_nunchaku_api.json`
- `ImageScaleToTotalPixels` for both img1 (93) and img2 (138)
- Prompt set directly on `TextEncodeQwenImageEditPlus` nodes (no separate prompt text node)
- LoRA: via `NunchakuQwenImageLoraStackV3` custom node — `lora_count` + `enabled_N`/`lora_name_N`/`lora_strength_N` (dynamic slots). Scans `server/LoRA/` and matches against ComfyUI's known LoRAs via `/object_info/NunchakuQwenImageLoraStackV3`
- No UNETLoader (nunchaku model loaded by `NunchakuQwenImageDiTLoader` node 133)
- Startup checks: ComfyUI connectivity (60s retry), required model availability (Nunchaku/CLIP/VAE), NunchakuQwenImageLoraStackV3 node availability
- Shares `app_comfyui_template.html` with AIO variant

### Configurable Parameters

Edit constants at top of each script for fixed values:

**simple_image_edit_nunchaku_qwen.py**:
- `PROMPT`, `TRUE_CFG_SCALE` (1.0)
- `NUM_INFERENCE_STEPS` (8), `RANK` (32)
- `NUM_BLOCKS_ON_GPU` (1), `GPU_MEM_THRESHOLD_GB` (18.0)
- `LORA` (None), `LORA_WEIGHT_NAME` (None), `LORA_SCALE` (1.0)

**simple_image_edit_rapid_qwen.py**:
- `PROMPT`, `TRUE_CFG_SCALE` (1.0), `GUIDANCE_SCALE` (1.0), `NEGATIVE_PROMPT` (" ")
- `NUM_INFERENCE_STEPS` (4)
- `TRANSFORMER_ID` (`prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23`)
- `BASE_MODEL_ID` (`Qwen/Qwen-Image-Edit-2511`)
- `LORA` (None), `LORA_WEIGHT_NAME` (None), `LORA_SCALE` (1.0)

**simple_image_edit_gguf_qwen.py**:
- `PROMPT`, `TRUE_CFG_SCALE` (1.0), `GUIDANCE_SCALE` (1.0), `NEGATIVE_PROMPT` (" ")
- `NUM_INFERENCE_STEPS` (4)
- `GGUF_REPO` (`Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF`)
- `GGUF_FILENAME` (`v23/Qwen-Rapid-NSFW-v23_Q3_K.gguf`)
- `TRANSFORMER_CONFIG` (`qwen-image-edit-transformer-config`)
- `BASE_MODEL_ID` (`Qwen/Qwen-Image-Edit-2511`)
- `LORA` (None), `LORA_WEIGHT_NAME` (None), `LORA_SCALE` (1.0)

**simple_image_edit_flux2_klein.py**:
- `PROMPT`, `GUIDANCE_SCALE` (1.0)
- `NUM_INFERENCE_STEPS` (20)
- `MODEL_ID` (`black-forest-labs/FLUX.2-klein-4B`)
- `LORA` (None), `LORA_WEIGHT_NAME` (None), `LORA_SCALE` (1.0)

**simple_i2v_ltx2_distilled.py** (diffusers):
- `PROMPT`, `NEGATIVE_PROMPT`, `GUIDANCE_SCALE` (1.0)
- `NUM_INFERENCE_STEPS_S1` (8), `NUM_INFERENCE_STEPS_S2` (3)
- `NUM_FRAMES` (121), `FRAME_RATE` (24.0)
- `MODEL_ID` (`rootonchair/LTX-2-19b-distilled`)
- `LORA` (None), `LORA_WEIGHT_NAME` (None), `LORA_SCALE` (1.0)

**simple_i2v_ltx2_native.py** (native ltx-pipelines):
- `PROMPT`, `NEGATIVE_PROMPT`
- `NUM_FRAMES` (97), `FRAME_RATE` (25.0)
- `DEFAULT_CHECKPOINT` (None), `DEFAULT_UPSAMPLER` (None), `DEFAULT_GEMMA_ROOT` (None)

**simple_image_edit_zit.py** (archived):
- `PROMPT`, `NEGATIVE_PROMPT`
- `NUM_STEPS` (8), `GUIDANCE_SCALE` (0.0), `STRENGTH` (0.6)
- `LORA` (None), `LORA_WEIGHT_NAME` (None), `LORA_SCALE` (1.0)

### Model Paths

**Qwen**: Dynamically constructed from `--steps` and `--rank`:
```
nunchaku-tech/nunchaku-qwen-image-edit-2509/lightning-251115/
svdq-{precision}_r{rank}-qwen-image-edit-2509-lightning-{steps}steps-251115.safetensors
```

**Qwen Rapid**: Transformer from `prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23`, base `Qwen/Qwen-Image-Edit-2511`

**Qwen GGUF**: GGUF from `Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF/v23/`, config from local `qwen-image-edit-transformer-config/`

**FLUX.2 Klein**: `black-forest-labs/FLUX.2-klein-4B`

**LTX-2 (diffusers)**: `rootonchair/LTX-2-19b-distilled` (fp8 via layerwise casting, latent upsampler: `latent_upsampler` subfolder)

**LTX-2 (native)**: Local files from `Lightricks/LTX-2` HF repo (`ltx-2-19b-distilled-fp8.safetensors`, `ltx-2-spatial-upscaler-x2-1.0.safetensors`, Gemma 3 text encoder)

**Z-Image** (archived): `unsloth/Z-Image-Turbo-unsloth-bnb-4bit`

## Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Use `--pre-resize 1m`, enable offload |
| 0% hang on first run | Model downloading (progress shown by default; CLI scripts use `--progress` to enable) |
| `pos_embed` error (Qwen) | Install `diffusers>=0.36.0,<0.37.0` |
| Transformer load error (Qwen) | Check `--steps` (4/8) and `--rank` (32/128) combo |
| `Flux2KleinPipeline` not found | Need latest diffusers: `pip install -U git+https://github.com/huggingface/diffusers` |
| `GGUFQuantizationConfig` not found | Need `diffusers>=0.36.0` + gguf: `pip install "diffusers>=0.36.0" gguf` |
| GGUF `from_single_file` fails | Ensure `qwen-image-edit-transformer-config/config.json` exists; check diffusers version |
| Qwen (nunchaku/GGUF) + FLUX.2 conflict | Use separate venvs (diffusers version incompatible). Nunchaku and GGUF can share the same 0.36.x venv |
| CUDA OOM with `--ref` | Reference images increase VRAM usage; use `--pre-resize 1m`, reduce `--ref` count, or enable offload |
| LoRA load failure | Verify the LoRA is compatible with the target model architecture (Qwen/FLUX.2/Z-Image); check file path or HF repo ID |
| CUDA OOM with `--lora` | LoRA increases VRAM usage; combine with `--pre-resize 1m` and offload options |
| Nunchaku LoRA format | Uses `nunchaku_lora_qwen.py` (ported from ComfyUI-QwenImageLoraLoader); standard diffusers LoRA .safetensors files work |
| Nunchaku LoRA switching slow | Expected: pipeline reloads from scratch on config change (CUDA kernel limitation). Same config reuses existing pipeline |
| AIO/GGUF LoRA load crash | Some LoRA files lack expected keys (e.g. `img_in.alpha`); server skips incompatible LoRAs and continues |
| GGUF + `--offload` error | `enable_sequential_cpu_offload()` is incompatible with GGUF tensors; use default offload or `--no-offload` |
| `LTX2ConditionPipeline` not found | Need latest diffusers: `pip install -U git+https://github.com/huggingface/diffusers` |
| LTX-2 CUDA OOM | Use `--no-stage2`, `--num-frames 61`, `--size 512x320`, or `--offload` |
| LTX-2 + Qwen venv conflict | LTX-2 requires latest diffusers (same venv as FLUX.2 Klein / Rapid Qwen) |
| ComfyUI connection refused | Ensure ComfyUI is running at the URL specified by `--comfyui-url` |
| ComfyUI model not found | Place model files in ComfyUI's `models/` directory or configure `extra_model_paths.yaml` |
| ComfyUI LoRA not recognized | Use `--comfyui-path` to auto-register `server/LoRA/` in `extra_model_paths.yaml` (triggers ComfyUI reboot) |
| ComfyUI reboot fails | Install ComfyUI-Manager extension for API reboot support (`GET /manager/reboot`) |
| ComfyUI no progress updates | Install `websocket-client`: `pip install websocket-client` |
| ComfyUI nunchaku node missing | Install `NunchakuQwenImageLoraStackV3` custom node in ComfyUI (required by `app_comfyui_nunchaku.py`) |

## Cache Management

Models cached in `~/.cache/huggingface/hub`. To clear:
```bash
pip cache purge
hf cache ls
hf cache rm <revision_id>
```

## Output Naming

- Qwen (Nunchaku): `{input_stem}_filtered.png`
- Qwen (Rapid): `{input_stem}_filtered_rapid.png`
- Qwen (GGUF): `{input_stem}_filtered_gguf.png`
- FLUX.2 Klein: `{input_stem}_filtered_klein.png`
- LTX-2 (diffusers): `{input_stem}_ltx2.mp4`
- LTX-2 (native): `{input_stem}_ltx2n.mp4`
- Z-Image: `{input_stem}_filtered_zit.png`

In `--t2i` mode, image scripts use `t2i` as the stem (e.g. `t2i_filtered.png`), LTX-2 uses `t2v` (e.g. `t2v_ltx2.mp4`), in the current directory.
