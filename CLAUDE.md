# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python scripts for AI-powered image editing using diffusion models:

1. **simple_image_edit_nunchaku_qwen.py** - Qwen-Image-Edit-2509 Lightning (Nunchaku-optimized) pipeline **(main)**
2. **simple_image_edit_flux2_klein.py** - FLUX.2 Klein 4B pipeline (requires separate venv with latest diffusers)
3. **simple_image_edit_zit.py** - Z-Image Turbo (4bit) img2img pipeline (archived; did not meet quality requirements)

All take a single image as input and output an edited version. Prompts can be specified via `--prompt` argument or by editing the `PROMPT` constant in the source.

**Target Environment:** GeForce RTX 3xxx (VRAM 12GB) class hardware. Higher-end GPUs can use `--no-offload` or process higher resolutions.

**Important:** Qwen (nunchaku) and FLUX.2 Klein cannot coexist in the same venv due to diffusers version conflicts.

## Running the Scripts

### Qwen-Image-Edit Lightning (Nunchaku) — requires diffusers 0.36.x
```powershell
py .\simple_image_edit_nunchaku_qwen.py .\sample.png
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --prompt "Enhance quality." --seed 42
py .\simple_image_edit_nunchaku_qwen.py .\sample.png --pre-resize 2m --progress
```

### FLUX.2 Klein 4B — requires diffusers latest (git main)
```powershell
py .\simple_image_edit_flux2_klein.py .\sample.png
py .\simple_image_edit_flux2_klein.py .\sample.png --prompt "Enhance quality." --seed 42
py .\simple_image_edit_flux2_klein.py .\sample.png --pre-resize 1m --offload --progress
```

### Common Options
- `--prompt "..."` - Specify prompt (uses `PROMPT` constant if omitted)
- `--seed N` - Random seed for reproducibility (random if omitted)
- `--pre-resize 1m|2m` - Reduce total pixels while maintaining aspect ratio
- `--progress` - Show Hugging Face download progress
- `--mem-log` - Display memory usage (requires psutil)
- `--no-offload` - Disable offloading (requires high VRAM)

### Script-specific Options
- `--offload` (FLUX.2 Klein) - Sequential CPU offload for low VRAM
- `--steps N` (both) - Override inference step count
- `--rank N` (Qwen only) - Nunchaku rank (32 or 128)
- `--num-blocks-on-gpu N` (Qwen only) - Blocks to keep on GPU in low-VRAM mode
- `--guidance-scale N` (FLUX.2 Klein only) - Override guidance scale

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

### venv for FLUX.2 Klein (separate)
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

**simple_image_edit_flux2_klein.py**:
- `PROMPT`, `GUIDANCE_SCALE` (1.0)
- `NUM_INFERENCE_STEPS` (20)
- `MODEL_ID` (`black-forest-labs/FLUX.2-klein-4B`)

**simple_image_edit_zit.py** (archived):
- `PROMPT`, `NEGATIVE_PROMPT`
- `NUM_STEPS` (8), `GUIDANCE_SCALE` (0.0), `STRENGTH` (0.6)

### Model Paths

**Qwen**: Dynamically constructed from `--steps` and `--rank`:
```
nunchaku-tech/nunchaku-qwen-image-edit-2509/lightning-251115/
svdq-{precision}_r{rank}-qwen-image-edit-2509-lightning-{steps}steps-251115.safetensors
```

**FLUX.2 Klein**: `black-forest-labs/FLUX.2-klein-4B`

**Z-Image** (archived): `unsloth/Z-Image-Turbo-unsloth-bnb-4bit`

## Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Use `--pre-resize 1m`, enable offload |
| 0% hang on first run | Model downloading; use `--progress` |
| `pos_embed` error (Qwen) | Install `diffusers>=0.36.0,<0.37.0` |
| Transformer load error (Qwen) | Check `--steps` (4/8) and `--rank` (32/128) combo |
| `Flux2KleinPipeline` not found | Need latest diffusers: `pip install -U git+https://github.com/huggingface/diffusers` |
| Qwen + FLUX.2 conflict | Use separate venvs (diffusers version incompatible) |

## Cache Management

Models cached in `~/.cache/huggingface/hub`. To clear:
```bash
pip cache purge
hf cache ls
hf cache rm <revision_id>
```

## Output Naming

- Qwen: `{input_stem}_filtered.png`
- FLUX.2 Klein: `{input_stem}_filtered_klein.png`
- Z-Image: `{input_stem}_filtered_zit.png`
