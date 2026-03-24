# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python scripts for AI-powered image editing using diffusion models:

### CLI Scripts
1. **simple_image_edit_nunchaku_qwen.py** - Qwen-Image-Edit-2509 Lightning (Nunchaku-optimized) pipeline **(main)**
2. **simple_image_edit_rapid_qwen.py** - Qwen-Image-Edit-Rapid-AIO-V23 (4-step accelerated, no nunchaku)
3. **simple_image_edit_gguf_qwen.py** - Qwen-Image-Edit-Rapid-AIO-V23 GGUF quantized (Q3_K default, low VRAM)
4. **simple_image_edit_flux2_klein.py** - FLUX.2 Klein 4B pipeline (requires separate venv with latest diffusers)
5. **simple_image_edit_zit.py** - Z-Image Turbo (4bit) img2img pipeline (archived; did not meet quality requirements)

### Web Server Scripts (`server/`)
6. **server/app_nunchaku.py** - Flask web server for Nunchaku Qwen-Image-Edit-2509 Lightning
7. **server/app_gguf.py** - Flask web server for GGUF quantized Qwen-Image-Edit-Rapid-AIO-V23

### Utilities
8. **server/nunchaku_lora_qwen.py** - LoRA loader for NunchakuQwenImageTransformer2DModel (ported from ComfyUI-QwenImageLoraLoader)

All CLI scripts take a single image as input (with optional `--ref` reference images) and output an edited version. All scripts also support `--t2i` mode for text-to-image generation without an input image. Prompts can be specified via `--prompt` argument or by editing the `PROMPT` constant in the source.

Web servers provide browser GUI with password protection, job queue (1 processing + 2 waiting), real-time step progress, cancel functionality, and optional gallery mode with password-gated login.

**Target Environment:** GeForce RTX 3xxx (VRAM 12GB) class hardware. Higher-end GPUs can use `--no-offload` or process higher resolutions.

**Important:** Qwen (nunchaku/GGUF) and FLUX.2 Klein cannot coexist in the same venv due to diffusers version conflicts. Nunchaku and GGUF can share the same venv (`diffusers==0.36.x`).

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

## Running the Web Servers

### Nunchaku Web Server
```powershell
cd server
python app_nunchaku.py
python app_nunchaku.py --password mysecret --port 8080
python app_nunchaku.py --host 0.0.0.0 --no-progress
python app_nunchaku.py --no-offload --rank 128
python app_nunchaku.py --steps 4              # 4-step model
python app_nunchaku.py --lora "path/to/lora.safetensors" --lora-scale 0.8
python app_nunchaku.py --gallery --password mysecret
```

### GGUF Web Server
```powershell
cd server
python app_gguf.py
python app_gguf.py --password mysecret --port 8080
python app_gguf.py --gguf-local "path/to/model.gguf"
python app_gguf.py --lora "HF_REPO_ID" --lora-weight-name "weights.safetensors"
python app_gguf.py --gallery --password mysecret
```

### Web Server Common Options
- `--host HOST` - Bind host (default: 127.0.0.1)
- `--port PORT` - Bind port (default: 5000)
- `--password PW` - Generation password (default: "password")
- `--no-progress` - Hide HF download progress (shown by default)
- `--no-offload` - Disable offloading (high VRAM)
- `--lora`, `--lora-weight-name`, `--lora-scale` - LoRA support (same as CLI)
- `--gallery` - Enable gallery mode (show generation history with image reuse)

### Web Server Features
- Browser GUI with password protection
- Job queue: 1 processing + 2 waiting, 3+ returns 503 BUSY
- Real-time step progress via polling (`callback_on_step_end`)
- Cancel running jobs (`pipeline._interrupt`)
- Auto-cleanup: files older than 1 hour removed every 5 minutes
- Pre-resize: 0.3M or 1M pixels
- t2i mode: 1024x1024 fixed size
- Prompt translation (googletrans): `-> EN` / `-> ZH` buttons for prompt translation
- File input clear buttons (x) for resetting image selections, drag-and-drop image upload
- Model info display: pipeline, transformer, text encoder class, tokenizer, VAE class, dtype, LoRA
- Error messages: Japanese/English bilingual (i18n)
- Gallery mode (`--gallery`): Browse past generation history, click thumbnails to enlarge (lightbox), download links on each image, reuse gallery images as input for new generations via radio button selection (Img1/Img2 slots). Selecting a gallery image automatically disables t2i mode
- Gallery login gate: When `--gallery` is enabled, a password login screen is shown first; all gallery/result/input APIs require password authentication via query parameter
- User identification: Each gallery entry shows a hashed user ID (SHA-256 of IP + User-Agent, 8 chars) to distinguish generators. Uses `X-Forwarded-For` header when available (reverse proxy support)
- Gallery entry deletion: Users can delete their own gallery entries (user_hash match required). Deleted entries remain visible as placeholders showing timestamp, user ID, and bilingual deletion message. Files are not removed (cleaned up by auto-cleanup). `DELETE /api/gallery/<job_id>` sets job status to `"hidden"`

### Web Server-specific Options
- `--steps N` (Nunchaku only) - Inference steps: 4 or 8 (default: 8)
- `--rank N` (Nunchaku only) - Nunchaku rank (32 or 128)
- `--num-blocks-on-gpu N` (Nunchaku only) - Blocks on GPU in low-VRAM mode
- `--gguf-local PATH` (GGUF only) - Use local GGUF file directly

**Note:** GGUF server does NOT support `--offload` (sequential CPU offload is incompatible with GGUF tensors). Only default (`enable_model_cpu_offload`) and `--no-offload` are available.

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

### venv for Rapid Qwen / FLUX.2 Klein (requires latest diffusers)
```powershell
py -3.11 -m venv .venv-flux2
.\.venv-flux2\Scripts\activate
pip install -U pip pillow huggingface_hub psutil transformers accelerate safetensors
pip install -U git+https://github.com/huggingface/diffusers
```

## Architecture

### Image Processing Pipeline

All scripts share this preprocessing flow:

1. **Pre-resize** (optional): Reduce total pixels to 1M/2M while keeping aspect ratio
2. **Max Size Cap**: Scale down to fit within 2048×2048 (downscale only)
3. **Alignment by Padding**: Pad to width×16 and height×16 multiples (not resize)

### Memory Management

**Qwen Lightning**:
- Nunchaku SVDQ-optimized transformer
- Prefers bf16, falls back to fp16
- Adaptive offload based on GPU memory:
  - High VRAM (>18GB): `enable_model_cpu_offload()`
  - Low VRAM: `transformer.set_offload()` + `enable_sequential_cpu_offload()`
- Warns if incompatible diffusers version detected
- LoRA: via `server/nunchaku_lora_qwen.py` (ported from ComfyUI-QwenImageLoraLoader). Handles QKV fusion, GLU fusion, proj_out split for nunchaku quantized modules. Standard diffusers LoRA files work (no nunchaku-specific format needed). Skips AWQ modulation layers (img_mod/txt_mod) as they are too sensitive

**Qwen Rapid-AIO-V23**:
- 4-step accelerated transformer (from `prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23`)
- Base pipeline: `Qwen/Qwen-Image-Edit-2511`
- Prefers bf16, falls back to fp16
- No nunchaku dependency; uses standard `QwenImageTransformer2DModel`
- 3-tier offload: default `enable_model_cpu_offload()`, `--offload` for sequential, `--no-offload` for full GPU
- Supports negative prompt and true CFG scale

**Qwen Rapid GGUF**:
- GGUF quantized version of Rapid-AIO-V23 (from `Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF`)
- Default: Q3_K (~10GB), also available: Q2_K (~7.4GB) through Q8_0 (~21.8GB)
- Loaded via `from_single_file` + `GGUFQuantizationConfig`
- Base pipeline: `Qwen/Qwen-Image-Edit-2511`
- Transformer config: local `qwen-image-edit-transformer-config/config.json`
- 3-tier offload: default `enable_model_cpu_offload()`, `--offload` for sequential, `--no-offload` for full GPU
- Requires `gguf` package

**FLUX.2 Klein 4B**:
- 4B parameter rectified flow transformer (distilled)
- Prefers bf16, falls back to fp16
- 3-tier offload: default `enable_model_cpu_offload()`, `--offload` for sequential, `--no-offload` for full GPU
- Model: `black-forest-labs/FLUX.2-klein-4B` (Apache 2.0)
- Text encoder: Mistral3 (via transformers)
- Image passed as conditioning reference via `image` parameter

**Z-Image Turbo** (archived):
- 4bit quantization via bitsandbytes (unsloth model)
- `--offload` enables CPU offloading

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
| GGUF + `--offload` error | `enable_sequential_cpu_offload()` is incompatible with GGUF tensors; use default offload or `--no-offload` |

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
- Z-Image: `{input_stem}_filtered_zit.png`

In `--t2i` mode, output files use `t2i` as the stem (e.g. `t2i_filtered.png`, `t2i_filtered_rapid.png`, etc.) in the current directory.
