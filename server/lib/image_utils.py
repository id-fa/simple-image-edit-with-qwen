"""
Shared image preprocessing and LoRA argument parsing for diffusers-based servers.

Used by: app_aio.py, app_nunchaku.py, app_gguf.py
Not used by ComfyUI servers (image processing delegated to ComfyUI).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path


def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def pre_resize_to_total_pixels(img, target_pixels: int):
    w, h = img.size
    if w * h <= target_pixels:
        return img
    scale = math.sqrt(target_pixels / (w * h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    print(f"[info] pre-resize: {w}x{h} -> {nw}x{nh}", file=sys.stderr)
    return img.resize((nw, nh))


def fit_and_align(img, max_w: int, max_h: int, w_mult: int, h_mult: int):
    from PIL import Image

    w, h = img.size

    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))

    aw, ah = round_up(w, w_mult), round_up(h, h_mult)
    if aw > max_w or ah > max_h:
        scale = min(max_w / aw, max_h / ah)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))
        aw, ah = round_up(w, w_mult), round_up(h, h_mult)

    if w != aw or h != ah:
        bg = img.getpixel((w - 1, h - 1))
        canvas = Image.new("RGB", (aw, ah), bg)
        canvas.paste(img, (0, 0))
        return canvas

    return img


def preprocess_image(img, pre_resize_target: int | None,
                     max_w: int = 2048, max_h: int = 2048,
                     w_mult: int = 8, h_mult: int = 16):
    if pre_resize_target:
        img = pre_resize_to_total_pixels(img, pre_resize_target)
    img = fit_and_align(img, max_w, max_h, w_mult, h_mult)
    return img


def parse_lora_args(lora_args: list[str] | None) -> list[dict]:
    """Parse --lora arguments and resolve file paths.

    Format: "path_or_repo" or "repo_id::weight_name"
    Returns: list of {"name": str, "path": str, "default_scale": float}
    """
    if not lora_args:
        return []

    registry = []
    for raw in lora_args:
        weight_name = None
        if "::" in raw:
            repo_or_path, weight_name = raw.split("::", 1)
        else:
            repo_or_path = raw

        if os.path.isfile(repo_or_path):
            local_path = repo_or_path
            name = Path(repo_or_path).stem
        else:
            from huggingface_hub import hf_hub_download
            fn = weight_name or "pytorch_lora_weights.safetensors"
            print(f"[info] downloading LoRA: {repo_or_path} / {fn}", file=sys.stderr)
            local_path = hf_hub_download(repo_id=repo_or_path, filename=fn)
            name = repo_or_path.split("/")[-1]
            if weight_name:
                wn_stem = Path(weight_name).stem
                if wn_stem != "pytorch_lora_weights":
                    name += "/" + wn_stem

        base_name = name
        counter = 2
        existing_names = {e["name"] for e in registry}
        while name in existing_names:
            name = f"{base_name}_{counter}"
            counter += 1

        registry.append({"name": name, "path": local_path, "default_scale": 1.0})
        print(f"[info] LoRA registered: {name} -> {local_path}", file=sys.stderr)

    return registry
