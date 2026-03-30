#!/usr/bin/env python3
"""
Web server for Qwen-Image-Edit using ComfyUI API backend (GGUF workflow).
Replaces diffusers pipeline with ComfyUI API calls.
No torch/diffusers dependency — requires only flask, requests, pillow, websocket-client.

Usage:
  python app_comfyui_gguf.py
  python app_comfyui_gguf.py --password mysecret --port 5000
  python app_comfyui_gguf.py --comfyui-url http://192.168.1.100:8188
  python app_comfyui_gguf.py --comfyui-path D:\\ComfyUI   # auto-register LoRA path
  python app_comfyui_gguf.py --gallery --password mysecret
  python app_comfyui_gguf.py --steps 8 --cfg 1.0

Requirements:
  pip install flask requests pillow websocket-client googletrans==4.0.0rc1
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import math
import os
import random
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import requests as http_requests
from flask import Flask, request, jsonify, send_file, render_template_string
from PIL import Image

try:
    import websocket as ws_lib
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    print("[warn] websocket-client not installed. Progress tracking will use polling.", file=sys.stderr)
    print("[info] Install with: pip install websocket-client", file=sys.stderr)


# =======================
# Fixed parameters
# =======================
PROMPT_DEFAULT = (
    "Fix visible seams and misalignment at image boundaries. "
    "Remove all overlaid text, subtitles, and credits. "
    "Remove compression artifacts and upscaling noise. "
    "Preserve the original composition, character identity, pose, and costume exactly."
)

DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0

MAX_W = 2048
MAX_H = 2048
W_MULT = 8
H_MULT = 16

T2I_SIZE = (1024, 1024)

CLEANUP_INTERVAL_SEC = 300  # check every 5 min
MAX_AGE_SEC = 3600          # 1 hour
MAX_QUEUE_WAITING = 2       # max waiting jobs

TMP_DIR = Path(__file__).parent / "tmp"
LORA_DIR = Path(__file__).parent / "LoRA"

# =======================
# Workflow template (loaded from file)
# =======================
_WORKFLOW_PATH = Path(__file__).parent / "comfyui_workflow" / "comfyui_qwen_image_edit_AIO_v23_gguf_api.json"
try:
    with open(_WORKFLOW_PATH, encoding="utf-8") as _f:
        WORKFLOW_TEMPLATE = json.load(_f)
    print(f"[info] workflow template loaded: {_WORKFLOW_PATH}", file=sys.stderr)
except FileNotFoundError:
    print(f"[error] workflow template not found: {_WORKFLOW_PATH}", file=sys.stderr)
    print(f"[error] place workflow JSON at {_WORKFLOW_PATH}", file=sys.stderr)
    WORKFLOW_TEMPLATE = {}

# ----------------------------------------------------------
# Workflow node ID mapping
# When the workflow is re-exported from ComfyUI and node IDs
# change, update ONLY this section.
# ----------------------------------------------------------
WF_NODE = {
    "load_image1":      "78",        # LoadImage - main input
    "load_image2":      "448",       # LoadImage - reference / image2
    "image_scale1":     "449",       # ImageScaleToTotalPixels - img1
    "image_scale2":     "450",       # ImageScaleToTotalPixels - img2
    "prompt_text":      "435",       # PrimitiveStringMultiline - Prompt
    "gguf_loader":      "456",       # UnetLoaderGGUF - GGUF model
    "clip_loader_gguf": "455",       # CLIPLoaderGGUF - GGUF CLIP
    "vae_loader":       "437",       # VAELoader
    "vae_encode":       "443",       # VAEEncode
    "vae_decode":       "444",       # VAEDecode
    "text_neg":         "440",       # TextEncodeQwenImageEditPlus (negative)
    "text_pos":         "442",       # TextEncodeQwenImageEditPlus (positive)
    "model_sampling":   "441",       # ModelSamplingAuraFlow
    "cfg_norm":         "436",       # CFGNorm
    "lora_loader1":     "445",       # LoraLoaderModelOnly (1st, model <- gguf_loader)
    "lora_loader2":     "451",       # LoraLoaderModelOnly (2nd, model <- lora_loader1)
    "lora_loader3":     "452",       # LoraLoaderModelOnly (3rd, model <- lora_loader2)
    "ksampler":         "447",       # KSampler
    "save_image":       "60",        # SaveImage - output
}


# =======================
# Global state
# =======================
app = Flask(__name__)

# Job queue & tracking
job_lock = threading.Lock()
jobs: dict[str, dict] = {}
processing_queue: deque[str] = deque()
current_processing: str | None = None
cancel_requests: set[str] = set()

comfyui_url = "http://127.0.0.1:8188"
comfyui_client_id = uuid.uuid4().hex

server_password = "password"
gallery_enabled = False
prompt_presets: list[dict[str, str]] = []
lora_registry: list[dict] = []   # [{"name", "comfyui_name", "local_path", "default_scale"}]
drawings: dict[str, dict] = {}
model_info: dict[str, str] = {}

configured_steps = DEFAULT_STEPS
configured_cfg = DEFAULT_CFG


# =======================
# ComfyUI API helpers
# =======================
def comfyui_upload_image(image_bytes: bytes, filename: str) -> str:
    """Upload image to ComfyUI input folder. Returns the stored filename."""
    resp = http_requests.post(
        f"{comfyui_url}/upload/image",
        files={"image": (filename, io.BytesIO(image_bytes), "image/png")},
        data={"overwrite": "true"},
    )
    resp.raise_for_status()
    return resp.json().get("name", filename)


def comfyui_submit_prompt(workflow: dict) -> str:
    """Submit workflow to ComfyUI. Returns prompt_id."""
    payload = {"prompt": workflow, "client_id": comfyui_client_id}
    resp = http_requests.post(f"{comfyui_url}/prompt", json=payload)
    if resp.status_code != 200:
        try:
            err_detail = resp.json()
        except Exception:
            err_detail = resp.text[:2000]
        print(f"[error] ComfyUI /prompt returned {resp.status_code}: {json.dumps(err_detail, indent=2, ensure_ascii=False)}", file=sys.stderr)
        resp.raise_for_status()
    result = resp.json()

    if "error" in result:
        detail = result.get("node_errors", result["error"])
        raise RuntimeError(f"ComfyUI error: {detail}")
    node_errors = result.get("node_errors")
    if node_errors:
        raise RuntimeError(f"ComfyUI node errors: {json.dumps(node_errors, ensure_ascii=False)}")

    return result["prompt_id"]


def comfyui_interrupt():
    """Interrupt current ComfyUI execution."""
    try:
        http_requests.post(f"{comfyui_url}/interrupt")
    except Exception as ex:
        print(f"[warn] ComfyUI interrupt failed: {ex}", file=sys.stderr)


def comfyui_get_history(prompt_id: str) -> dict:
    """Get execution history for a prompt."""
    resp = http_requests.get(f"{comfyui_url}/history/{prompt_id}")
    resp.raise_for_status()
    return resp.json()


def comfyui_get_image(filename: str, subfolder: str = "", img_type: str = "output") -> bytes:
    """Download an image from ComfyUI."""
    resp = http_requests.get(
        f"{comfyui_url}/view",
        params={"filename": filename, "subfolder": subfolder, "type": img_type},
    )
    resp.raise_for_status()
    return resp.content


def comfyui_get_available_models() -> dict:
    """Query ComfyUI for available models."""
    info: dict[str, list] = {}
    for node_class, key, field in [
        ("UnetLoaderGGUF", "gguf_models", "unet_name"),
        ("CLIPLoaderGGUF", "clip_gguf_models", "clip_name"),
        ("VAELoader", "vae_models", "vae_name"),
        ("LoraLoaderModelOnly", "lora_models", "lora_name"),
    ]:
        try:
            resp = http_requests.get(f"{comfyui_url}/object_info/{node_class}", timeout=10)
            data = resp.json()
            info[key] = data[node_class]["input"]["required"][field][0]
        except Exception:
            info[key] = []
    return info


def comfyui_wait_for_result(prompt_id: str, job_id: str, timeout: float = 600) -> dict | None:
    """Wait for ComfyUI execution to complete."""
    if HAS_WEBSOCKET:
        return _wait_ws(prompt_id, job_id, timeout)
    return _wait_poll(prompt_id, job_id, timeout)


def _wait_ws(prompt_id: str, job_id: str, timeout: float) -> dict | None:
    """WebSocket-based progress tracking."""
    ws_url = comfyui_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws?clientId={comfyui_client_id}"

    ws = ws_lib.WebSocket()
    ws.settimeout(5.0)
    try:
        ws.connect(ws_url)
    except Exception as ex:
        print(f"[warn] WS connect failed, falling back to polling: {ex}", file=sys.stderr)
        return _wait_poll(prompt_id, job_id, timeout)

    done = False
    error_msg = None
    start = time.time()

    try:
        while time.time() - start < timeout:
            if job_id in cancel_requests:
                comfyui_interrupt()
                return None

            try:
                raw = ws.recv()
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    # Binary frame = preview image from ComfyUI
                    # Format: 4 bytes event type + 4 bytes format + image data
                    if len(raw) > 8:
                        preview_data = raw[8:]
                        with job_lock:
                            if job_id in jobs:
                                jobs[job_id]["preview"] = preview_data
                    continue
                data = json.loads(raw)
                msg_type = data.get("type", "")
                msg_data = data.get("data", {})

                if msg_data.get("prompt_id") != prompt_id:
                    continue

                if msg_type == "progress":
                    step = msg_data.get("value", 0)
                    total = msg_data.get("max", configured_steps)
                    with job_lock:
                        if job_id in jobs:
                            jobs[job_id]["current_step"] = step
                            jobs[job_id]["total_steps"] = total

                elif msg_type == "executing":
                    if msg_data.get("node") is None:
                        done = True
                        break

                elif msg_type == "execution_error":
                    error_msg = msg_data.get("exception_message", "ComfyUI execution error")
                    break

            except ws_lib.WebSocketTimeoutException:
                continue
            except Exception as ex:
                print(f"[warn] WS recv error: {ex}", file=sys.stderr)
                time.sleep(1)
    finally:
        try:
            ws.close()
        except Exception:
            pass

    if error_msg:
        raise RuntimeError(error_msg)
    if not done:
        return None

    history = comfyui_get_history(prompt_id)
    return history.get(prompt_id, {}).get("outputs", {})


def _wait_poll(prompt_id: str, job_id: str, timeout: float) -> dict | None:
    """Polling-based fallback (no progress tracking)."""
    start = time.time()
    while time.time() - start < timeout:
        if job_id in cancel_requests:
            comfyui_interrupt()
            return None
        try:
            history = comfyui_get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                if outputs:
                    return outputs
        except Exception:
            pass
        time.sleep(2)
    return None


# =======================
# Image processing helpers
# =======================
def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def pre_resize_to_total_pixels(img: Image.Image, target_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= target_pixels:
        return img
    scale = math.sqrt(target_pixels / (w * h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    print(f"[info] pre-resize: {w}x{h} -> {nw}x{nh}", file=sys.stderr)
    return img.resize((nw, nh))


def fit_and_align(img: Image.Image, max_w: int, max_h: int, w_mult: int, h_mult: int) -> Image.Image:
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


def preprocess_image(img: Image.Image, pre_resize_target: int | None) -> Image.Image:
    if pre_resize_target:
        img = pre_resize_to_total_pixels(img, pre_resize_target)
    return fit_and_align(img, MAX_W, MAX_H, W_MULT, H_MULT)


# =======================
# extra_model_paths.yaml auto-registration
# =======================
YAML_SECTION_NAME = "simple_image_edit"
YAML_COMMENT = "# Auto-added by app_comfyui_gguf.py — LoRA path for simple-image-edit server"


def ensure_lora_path_in_comfyui(comfyui_path: Path) -> bool:
    """Register server/LoRA/ in ComfyUI's extra_model_paths.yaml if not already present.

    Returns True if the file was modified (ComfyUI restart needed).
    """
    if not LORA_DIR.is_dir():
        return False

    yaml_path = comfyui_path / "extra_model_paths.yaml"
    lora_abs = str(LORA_DIR.resolve()).replace("\\", "/")

    existing_content = ""
    if yaml_path.exists():
        existing_content = yaml_path.read_text(encoding="utf-8")

        # Check if already registered (normalize slashes for comparison)
        normalized = existing_content.replace("\\", "/")
        if lora_abs in normalized:
            print(f"[info] LoRA path already in {yaml_path}", file=sys.stderr)
            return False

    # Build new YAML section
    new_section = f"\n{YAML_COMMENT}\n{YAML_SECTION_NAME}:\n    loras: {lora_abs}/\n"

    # Backup original if it exists and has content
    if existing_content.strip():
        backup_path = yaml_path.with_suffix(".yaml.bak")
        backup_path.write_text(existing_content, encoding="utf-8")
        print(f"[info] backed up {yaml_path} -> {backup_path}", file=sys.stderr)

    # Append section
    with open(yaml_path, "a", encoding="utf-8") as f:
        f.write(new_section)

    print(f"[info] registered LoRA path in {yaml_path}:", file=sys.stderr)
    print(f"[info]   {YAML_SECTION_NAME}.loras: {lora_abs}/", file=sys.stderr)
    return True


def reboot_comfyui_and_wait(timeout: float = 120) -> bool:
    """Reboot ComfyUI via Manager API and wait for it to come back up."""
    print("[info] ComfyUIを再起動しています... / Rebooting ComfyUI...", file=sys.stderr)
    try:
        http_requests.get(f"{comfyui_url}/manager/reboot", timeout=5)
    except Exception:
        pass

    start = time.time()
    while time.time() - start < timeout:
        time.sleep(2)
        try:
            resp = http_requests.get(f"{comfyui_url}/system_stats", timeout=3)
            if resp.status_code == 200:
                elapsed = time.time() - start
                print(f"[info] ComfyUI restarted ({elapsed:.0f}s)", file=sys.stderr)
                return True
        except Exception:
            pass

    print("[error] ComfyUI restart timeout", file=sys.stderr)
    return False


# =======================
# LoRA scanning
# =======================
def scan_lora_folder(comfyui_lora_names: list[str]) -> list[dict]:
    """Scan server/LoRA/ folder and match with ComfyUI's known LoRAs."""
    if not LORA_DIR.is_dir():
        print(f"[info] LoRA directory not found: {LORA_DIR}", file=sys.stderr)
        return []

    registry = []
    for f in sorted(LORA_DIR.iterdir()):
        if f.suffix.lower() not in (".safetensors", ".ckpt", ".pt"):
            continue

        comfyui_name = None
        for cn in comfyui_lora_names:
            cn_base = cn.replace("\\", "/").split("/")[-1]
            if cn_base == f.name or cn == f.name:
                comfyui_name = cn
                break

        if comfyui_name:
            registry.append({
                "name": f.stem,
                "comfyui_name": comfyui_name,
                "local_path": str(f),
                "default_scale": 1.0,
            })
            print(f"[info] LoRA found: {f.stem} -> {comfyui_name}", file=sys.stderr)
        else:
            print(f"[warn] LoRA not recognized by ComfyUI: {f.name}", file=sys.stderr)
            print(f"[warn]   -> Add LoRA directory to ComfyUI extra_model_paths.yaml", file=sys.stderr)

    return registry


# =======================
# Workflow builder
# =======================
def build_workflow(image1_name: str, image2_name: str | None, prompt: str,
                   seed: int | None, loras: list[dict] | None = None,
                   pre_resize: int | None = None) -> dict:
    """Build ComfyUI workflow from template with given parameters.

    Args:
        loras: list of {"comfyui_name": str, "scale": float} (max 3)
        pre_resize: target total pixels (e.g. 1000000, 2000000) for ImageScaleToTotalPixels
    """
    N = WF_NODE  # shorthand
    wf = copy.deepcopy(WORKFLOW_TEMPLATE)

    # Input images
    wf[N["load_image1"]]["inputs"]["image"] = image1_name
    wf[N["load_image2"]]["inputs"]["image"] = image2_name or image1_name

    # Image scale (megapixels) for pre-resize
    if pre_resize:
        megapixels = pre_resize / 1_000_000  # 1000000 -> 1.0, 2000000 -> 2.0
    else:
        megapixels = 1.0
    wf[N["image_scale1"]]["inputs"]["megapixels"] = megapixels
    wf[N["image_scale2"]]["inputs"]["megapixels"] = megapixels

    # Prompt (via PrimitiveStringMultiline -> TextEncodeQwenImageEditPlus)
    wf[N["prompt_text"]]["inputs"]["value"] = prompt

    # Seed, Steps, CFG - directly on KSampler
    wf[N["ksampler"]]["inputs"]["seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf[N["ksampler"]]["inputs"]["steps"] = configured_steps
    wf[N["ksampler"]]["inputs"]["cfg"] = configured_cfg

    # LoRA handling (3 slots in workflow: lora_loader1 -> lora_loader2 -> lora_loader3)
    # Chain: UnetLoaderGGUF(456) -> 445 -> 451 -> 452 -> ModelSamplingAuraFlow(441)
    # Unused nodes are removed and the model chain is rewired.
    active_loras = [l for l in (loras or []) if abs(l.get("scale", 1.0)) > 1e-5]
    active_loras = active_loras[:3]  # max 3

    loader_keys = ["lora_loader1", "lora_loader2", "lora_loader3"]
    last_model_source = [N["gguf_loader"], 0]  # default: direct from UnetLoaderGGUF
    for i, key in enumerate(loader_keys):
        if i < len(active_loras):
            wf[N[key]]["inputs"]["lora_name"] = active_loras[i]["comfyui_name"]
            wf[N[key]]["inputs"]["strength_model"] = active_loras[i]["scale"]
            last_model_source = [N[key], 0]
        else:
            # Remove unused LoRA node from workflow
            del wf[N[key]]

    # Rewire ModelSamplingAuraFlow to the last active node in the chain
    wf[N["model_sampling"]]["inputs"]["model"] = last_model_source

    return wf


# =======================
# Job worker
# =======================
def worker_loop():
    global current_processing
    while True:
        job_id = None
        with job_lock:
            while processing_queue:
                candidate = processing_queue[0]
                if candidate in cancel_requests:
                    processing_queue.popleft()
                    cancel_requests.discard(candidate)
                    jobs[candidate]["status"] = "cancelled"
                    continue
                break

            if processing_queue:
                job_id = processing_queue[0]
                current_processing = job_id
                jobs[job_id]["status"] = "processing"
                jobs[job_id]["current_step"] = 0

        if job_id is None:
            time.sleep(0.5)
            continue

        try:
            job = jobs[job_id]
            image_names: list[str] = []

            if job["t2i"]:
                tw, th = T2I_SIZE
                tw, th = round_up(tw, W_MULT), round_up(th, H_MULT)
                img = Image.new("RGB", (tw, th), (255, 255, 255))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                fname = comfyui_upload_image(buf.getvalue(), f"{job_id}_t2i.png")
                image_names.append(fname)
            else:
                for i, img_path in enumerate(job["input_paths"]):
                    img = Image.open(img_path).convert("RGB")
                    img = preprocess_image(img, job["pre_resize"])
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    fname = comfyui_upload_image(buf.getvalue(), f"{job_id}_in{i}.png")
                    image_names.append(fname)

            # Resolve LoRA selection
            lora_selection = []
            for sel in job.get("loras", []):
                matched = [l for l in lora_registry if l["name"] == sel["name"]]
                if matched:
                    lora_selection.append({
                        "comfyui_name": matched[0]["comfyui_name"],
                        "scale": sel.get("scale", 1.0),
                    })

            # Build and submit workflow
            wf = build_workflow(
                image1_name=image_names[0],
                image2_name=image_names[1] if len(image_names) > 1 else None,
                prompt=job["prompt"],
                seed=job["seed"],
                loras=lora_selection,
                pre_resize=job.get("pre_resize"),
            )
            prompt_id = comfyui_submit_prompt(wf)

            # Wait for result (with progress tracking)
            outputs = comfyui_wait_for_result(prompt_id, job_id)

            if job_id in cancel_requests:
                with job_lock:
                    cancel_requests.discard(job_id)
                    jobs[job_id]["status"] = "cancelled"
            elif outputs is None:
                with job_lock:
                    if jobs[job_id]["status"] != "cancelled":
                        jobs[job_id]["status"] = "error"
                        jobs[job_id]["error"] = "ComfyUI timeout または出力なし / ComfyUI timeout or no output"
            else:
                save_output = outputs.get(WF_NODE["save_image"], {})
                images = save_output.get("images", [])
                if not images:
                    raise RuntimeError("ComfyUIから出力画像がありません / No output images from ComfyUI")

                img_info = images[0]
                img_bytes = comfyui_get_image(
                    img_info["filename"],
                    img_info.get("subfolder", ""),
                    img_info.get("type", "output"),
                )
                out_path = TMP_DIR / f"{job_id}_out.png"
                with open(out_path, "wb") as f:
                    f.write(img_bytes)

                with job_lock:
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["result_path"] = str(out_path)

        except Exception as ex:
            import traceback
            traceback.print_exc()
            with job_lock:
                cancel_requests.discard(job_id)
                if jobs[job_id]["status"] != "cancelled":
                    jobs[job_id]["status"] = "error"
                    jobs[job_id]["error"] = str(ex)
        finally:
            with job_lock:
                if processing_queue and processing_queue[0] == job_id:
                    processing_queue.popleft()
                current_processing = None
            gc.collect()


# =======================
# Cleanup old files
# =======================
def cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL_SEC)
        now = time.time()
        cutoff = now - MAX_AGE_SEC

        to_remove = []
        with job_lock:
            for jid, job in jobs.items():
                if job["created"] < cutoff:
                    to_remove.append(jid)

        for jid in to_remove:
            with job_lock:
                job = jobs.pop(jid, None)
                cancel_requests.discard(jid)
            if job:
                for p in job.get("input_paths", []):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                rp = job.get("result_path")
                if rp:
                    try:
                        os.remove(rp)
                    except OSError:
                        pass

        # Clean old drawings
        draw_remove = []
        for did, d in drawings.items():
            if d["created"] < cutoff:
                draw_remove.append(did)
        for did in draw_remove:
            d = drawings.pop(did, None)
            if d:
                for key in ("path", "bg_path", "overlay_path"):
                    p = d.get(key)
                    if p:
                        try:
                            os.remove(p)
                        except OSError:
                            pass

        # Clean orphaned tmp files
        for f in TMP_DIR.iterdir():
            if f.name == ".gitkeep":
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass


# =======================
# Gallery helper
# =======================
def get_user_hash() -> str:
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not ip:
        ip = request.remote_addr or ""
    ua = request.headers.get("User-Agent", "")
    return hashlib.sha256(f"{ip}:{ua}".encode()).hexdigest()[:8]


def resolve_gallery_ref(ref_str: str, new_job_id: str, slot: int) -> str | None:
    import shutil
    parts = ref_str.split(":")
    if len(parts) < 2:
        return None

    if parts[0] == "drawing":
        drawing_id = parts[1]
        d = drawings.get(drawing_id)
        if not d:
            return None
        src_path = d.get("path")
        if not src_path or not os.path.exists(src_path):
            return None
        dest_path = TMP_DIR / f"{new_job_id}_in{slot}.png"
        shutil.copy2(src_path, dest_path)
        return str(dest_path)

    src_job_id, ref_type = parts[0], parts[1]
    with job_lock:
        src_job = jobs.get(src_job_id)
        if not src_job:
            return None
        if ref_type == "result":
            src_path = src_job.get("result_path")
        elif ref_type == "input" and len(parts) >= 3:
            idx = int(parts[2])
            paths = src_job.get("input_paths", [])
            src_path = paths[idx] if 0 <= idx < len(paths) else None
        else:
            return None
    if not src_path or not os.path.exists(src_path):
        return None
    ext = Path(src_path).suffix.lower() or ".png"
    dest_path = TMP_DIR / f"{new_job_id}_in{slot}{ext}"
    shutil.copy2(src_path, dest_path)
    return str(dest_path)


# =======================
# Routes
# =======================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, gallery_enabled=gallery_enabled,
                                  prompt_presets=prompt_presets)


@app.route("/api/submit", methods=["POST"])
def submit():
    pw = request.form.get("password", "")
    if pw != server_password:
        return jsonify({"error": "パスワードが正しくありません / Invalid password"}), 403

    with job_lock:
        waiting_count = len(processing_queue)
        if waiting_count >= 1 + MAX_QUEUE_WAITING:
            return jsonify({"error": "サーバーがビジー状態です / Server is busy. Please try again later.",
                            "busy": True, "queue_size": waiting_count}), 503

    t2i = request.form.get("t2i") == "1"
    prompt = request.form.get("prompt", "").strip() or PROMPT_DEFAULT
    seed_str = request.form.get("seed", "").strip()
    seed = int(seed_str) if seed_str else None
    pre_resize_str = request.form.get("pre_resize", "1m")
    pre_resize_map = {"1m": 1_000_000, "2m": 2_000_000}
    pre_resize = pre_resize_map.get(pre_resize_str, 1_000_000)

    job_id = uuid.uuid4().hex[:12]
    input_paths = []

    if not t2i:
        f1 = request.files.get("image1")
        f2 = request.files.get("image2")
        g1 = request.form.get("gallery_image1", "").strip()
        g2 = request.form.get("gallery_image2", "").strip()

        if f1 and f1.filename:
            ext1 = Path(f1.filename).suffix.lower() or ".png"
            save1 = TMP_DIR / f"{job_id}_in0{ext1}"
            f1.save(save1)
            input_paths.append(str(save1))
        elif g1:
            resolved = resolve_gallery_ref(g1, job_id, 0)
            if not resolved:
                return jsonify({"error": "ギャラリー画像1の参照が無効または期限切れです / Gallery image 1 reference is invalid or expired"}), 400
            input_paths.append(resolved)
        else:
            return jsonify({"error": "Image 1 を選択してください / Please select Image 1"}), 400

        if f2 and f2.filename:
            ext2 = Path(f2.filename).suffix.lower() or ".png"
            save2 = TMP_DIR / f"{job_id}_in1{ext2}"
            f2.save(save2)
            input_paths.append(str(save2))
        elif g2:
            resolved = resolve_gallery_ref(g2, job_id, 1)
            if resolved:
                input_paths.append(resolved)
    else:
        if not prompt or prompt == PROMPT_DEFAULT:
            return jsonify({"error": "t2iモードではプロンプトの入力が必要です / Prompt is required in t2i mode"}), 400

    # Parse LoRA selection
    lora_selection = []
    loras_raw = request.form.get("loras", "").strip()
    if loras_raw:
        try:
            lora_selection = json.loads(loras_raw)
        except Exception:
            pass

    user_hash = get_user_hash()

    with job_lock:
        jobs[job_id] = {
            "status": "queued",
            "created": time.time(),
            "input_paths": input_paths,
            "result_path": None,
            "error": None,
            "prompt": prompt,
            "seed": seed,
            "pre_resize": pre_resize,
            "t2i": t2i,
            "current_step": 0,
            "total_steps": configured_steps,
            "user_hash": user_hash,
            "loras": lora_selection,
        }
        processing_queue.append(job_id)
        queue_pos = len(processing_queue)

    return jsonify({"job_id": job_id, "queue_position": queue_pos})


@app.route("/api/status/<job_id>")
def status(job_id):
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "ジョブが見つかりません / Job not found"}), 404

        queue_pos = 0
        for i, qid in enumerate(processing_queue):
            if qid == job_id:
                queue_pos = i + 1
                break

        return jsonify({
            "status": job["status"],
            "queue_position": queue_pos,
            "queue_total": len(processing_queue),
            "error": job.get("error"),
            "current_step": job.get("current_step", 0),
            "total_steps": job.get("total_steps", configured_steps),
            "has_preview": "preview" in job,
        })


@app.route("/api/preview/<job_id>")
def preview(job_id):
    with job_lock:
        job = jobs.get(job_id)
        if not job or "preview" not in job:
            return "", 204
        preview_data = job["preview"]
    content_type = "image/jpeg"
    if preview_data[:4] == b'\x89PNG':
        content_type = "image/png"
    return send_file(io.BytesIO(preview_data), mimetype=content_type)


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "ジョブが見つかりません / Job not found"}), 404
        if job["status"] in ("done", "error", "cancelled"):
            return jsonify({"error": "このジョブは既に終了しています / This job has already finished"}), 400
        cancel_requests.add(job_id)
        if job["status"] == "queued":
            try:
                processing_queue.remove(job_id)
            except ValueError:
                pass
            job["status"] = "cancelled"
            cancel_requests.discard(job_id)

    return jsonify({"ok": True, "message": "キャンセルを要求しました / Cancel requested"})


@app.route("/api/result/<job_id>")
def result(job_id):
    if gallery_enabled:
        pw = request.args.get("password", "")
        if pw != server_password:
            return jsonify({"error": "Unauthorized"}), 403
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "ジョブが見つかりません / Job not found"}), 404
        if job["status"] != "done":
            return jsonify({"error": "まだ処理中です / Still processing"}), 400
        rp = job["result_path"]
    if not rp or not os.path.exists(rp):
        return jsonify({"error": "結果ファイルが見つかりません / Result file not found"}), 404
    return send_file(rp, mimetype="image/png", as_attachment=True,
                     download_name=f"result_{job_id}.png")


@app.route("/api/translate", methods=["POST"])
def translate_text():
    text = request.json.get("text", "").strip()
    target = request.json.get("target", "en")
    if not text:
        return jsonify({"error": "テキストが空です / Text is empty"}), 400
    try:
        import asyncio
        from googletrans import Translator
        tr = Translator()
        result = asyncio.run(tr.translate(text, dest=target))
        return jsonify({"translated": result.text, "src": result.src})
    except Exception as ex:
        return jsonify({"error": f"翻訳に失敗しました / Translation failed: {ex}"}), 500


@app.route("/api/gallery")
def gallery():
    if not gallery_enabled:
        return jsonify({"error": "Gallery is disabled"}), 404
    pw = request.args.get("password", "")
    if pw != server_password:
        return jsonify({"error": "Unauthorized"}), 403
    with job_lock:
        items = []
        for jid, job in jobs.items():
            if job["status"] == "done":
                items.append({
                    "job_id": jid,
                    "created": job["created"],
                    "prompt": job.get("prompt", ""),
                    "seed": job.get("seed"),
                    "t2i": job.get("t2i", False),
                    "input_count": len(job.get("input_paths", [])),
                    "user_hash": job.get("user_hash", ""),
                    "deleted": False,
                })
            elif job["status"] == "hidden":
                items.append({
                    "job_id": jid,
                    "created": job["created"],
                    "user_hash": job.get("user_hash", ""),
                    "deleted": True,
                })
    items.sort(key=lambda x: x["created"], reverse=True)
    caller_hash = get_user_hash()
    return jsonify({"items": items, "caller_hash": caller_hash})


@app.route("/api/gallery/<job_id>", methods=["DELETE"])
def gallery_delete(job_id):
    if not gallery_enabled:
        return jsonify({"error": "Gallery is disabled"}), 404
    pw = request.args.get("password", "")
    if pw != server_password:
        return jsonify({"error": "Unauthorized"}), 403
    caller_hash = get_user_hash()
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "ジョブが見つかりません / Job not found"}), 404
        if job.get("user_hash", "") != caller_hash:
            return jsonify({"error": "他のユーザーの履歴は削除できません / Cannot delete another user's entry"}), 403
        job["status"] = "hidden"
    return jsonify({"ok": True})


@app.route("/api/input/<job_id>/<int:idx>")
def serve_input(job_id, idx):
    if gallery_enabled:
        pw = request.args.get("password", "")
        if pw != server_password:
            return jsonify({"error": "Unauthorized"}), 403
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        paths = job.get("input_paths", [])
        if idx < 0 or idx >= len(paths):
            return jsonify({"error": "Input index out of range"}), 404
        path = paths[idx]
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "bmp": "image/bmp"}.get(ext, "image/png")
    return send_file(path, mimetype=mime)


@app.route("/api/model_info")
def get_model_info():
    return jsonify(model_info)


@app.route("/api/loras")
def get_loras():
    return jsonify([{"name": e["name"], "default_scale": e["default_scale"]}
                     for e in lora_registry])


@app.route("/api/queue_info")
def queue_info():
    with job_lock:
        return jsonify({
            "queue_size": len(processing_queue),
            "processing": current_processing is not None,
        })


# =======================
# Drawing API
# =======================
@app.route("/api/drawing/save", methods=["POST"])
def save_drawing():
    if gallery_enabled:
        pw = request.json.get("password", "")
        if pw != server_password:
            return jsonify({"error": "Unauthorized"}), 403

    import base64
    data = request.json
    img_data = data.get("image", "")
    draw_type = data.get("type", "composite")
    source_info = data.get("source", "")

    if not img_data:
        return jsonify({"error": "No image data"}), 400

    if "," in img_data:
        img_data = img_data.split(",", 1)[1]

    drawing_id = uuid.uuid4().hex[:12]
    path = TMP_DIR / f"draw_{drawing_id}.png"

    with open(path, "wb") as f:
        f.write(base64.b64decode(img_data))

    bg_path = None
    overlay_path = None
    if draw_type == "draft":
        bg_data = data.get("bg", "")
        overlay_data = data.get("overlay", "")
        if bg_data:
            if "," in bg_data:
                bg_data = bg_data.split(",", 1)[1]
            bg_path = str(TMP_DIR / f"draw_{drawing_id}_bg.png")
            with open(bg_path, "wb") as f:
                f.write(base64.b64decode(bg_data))
        if overlay_data:
            if "," in overlay_data:
                overlay_data = overlay_data.split(",", 1)[1]
            overlay_path = str(TMP_DIR / f"draw_{drawing_id}_ov.png")
            with open(overlay_path, "wb") as f:
                f.write(base64.b64decode(overlay_data))

    user_hash = get_user_hash()
    drawings[drawing_id] = {
        "user_hash": user_hash,
        "created": time.time(),
        "path": str(path),
        "type": draw_type,
        "source": source_info,
        "bg_path": bg_path,
        "overlay_path": overlay_path,
    }
    return jsonify({"drawing_id": drawing_id})


@app.route("/api/drawing/<drawing_id>")
def serve_drawing(drawing_id):
    if gallery_enabled:
        pw = request.args.get("password", "")
        if pw != server_password:
            return jsonify({"error": "Unauthorized"}), 403
    d = drawings.get(drawing_id)
    if not d:
        return jsonify({"error": "Drawing not found"}), 404
    if d["user_hash"] != get_user_hash():
        return jsonify({"error": "Unauthorized"}), 403
    if not os.path.exists(d["path"]):
        return jsonify({"error": "File not found"}), 404
    return send_file(d["path"], mimetype="image/png")


@app.route("/api/drawing/<drawing_id>/bg")
def serve_drawing_bg(drawing_id):
    if gallery_enabled:
        pw = request.args.get("password", "")
        if pw != server_password:
            return jsonify({"error": "Unauthorized"}), 403
    d = drawings.get(drawing_id)
    if not d:
        return jsonify({"error": "Drawing not found"}), 404
    if d["user_hash"] != get_user_hash():
        return jsonify({"error": "Unauthorized"}), 403
    bg = d.get("bg_path")
    if not bg or not os.path.exists(bg):
        return jsonify({"error": "BG not found"}), 404
    return send_file(bg, mimetype="image/png")


@app.route("/api/drawing/<drawing_id>/overlay")
def serve_drawing_overlay(drawing_id):
    if gallery_enabled:
        pw = request.args.get("password", "")
        if pw != server_password:
            return jsonify({"error": "Unauthorized"}), 403
    d = drawings.get(drawing_id)
    if not d:
        return jsonify({"error": "Drawing not found"}), 404
    if d["user_hash"] != get_user_hash():
        return jsonify({"error": "Unauthorized"}), 403
    ov = d.get("overlay_path")
    if not ov or not os.path.exists(ov):
        return jsonify({"error": "Overlay not found"}), 404
    return send_file(ov, mimetype="image/png")


@app.route("/api/drawings")
def list_drawings():
    if gallery_enabled:
        pw = request.args.get("password", "")
        if pw != server_password:
            return jsonify({"error": "Unauthorized"}), 403
    user_hash = get_user_hash()
    items = []
    for did, d in drawings.items():
        if d["user_hash"] != user_hash:
            continue
        items.append({
            "drawing_id": did,
            "created": d["created"],
            "type": d["type"],
            "source": d["source"],
        })
    items.sort(key=lambda x: x["created"], reverse=True)
    return jsonify({"items": items})


@app.route("/api/drawing/<drawing_id>", methods=["DELETE"])
def delete_drawing(drawing_id):
    if gallery_enabled:
        pw = request.args.get("password", "")
        if pw != server_password:
            return jsonify({"error": "Unauthorized"}), 403
    d = drawings.get(drawing_id)
    if not d:
        return jsonify({"error": "Drawing not found"}), 404
    if d["user_hash"] != get_user_hash():
        return jsonify({"error": "Unauthorized"}), 403
    for key in ("path", "bg_path", "overlay_path"):
        p = d.get(key)
        if p:
            try:
                os.remove(p)
            except OSError:
                pass
    del drawings[drawing_id]
    return jsonify({"ok": True})


# =======================
# HTML Template — loaded from app_comfyui_template.html
# =======================
_TEMPLATE_PATH = Path(__file__).parent / "app_comfyui_template.html"
try:
    HTML_TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")
    print(f"[info] HTML template loaded: {_TEMPLATE_PATH}", file=sys.stderr)
except FileNotFoundError:
    print(f"[error] HTML template not found: {_TEMPLATE_PATH}", file=sys.stderr)
    HTML_TEMPLATE = (
        "<!DOCTYPE html><html><body>"
        "<h1>Error: app_comfyui_template.html not found</h1>"
        "</body></html>"
    )


# =======================
# Entry point
# =======================
def main():
    global server_password, gallery_enabled, comfyui_url, configured_steps, configured_cfg

    ap = argparse.ArgumentParser(description="Qwen Image Edit Web Server (ComfyUI GGUF backend)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    ap.add_argument("--password", default="password", help="Generation password (default: password)")
    ap.add_argument("--comfyui-url", default="http://127.0.0.1:8188",
                    help="ComfyUI API URL (default: http://127.0.0.1:8188)")
    ap.add_argument("--comfyui-path", default=None, metavar="DIR",
                    help="ComfyUI installation directory. "
                         "Auto-registers server/LoRA/ in extra_model_paths.yaml")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"Inference steps (default: {DEFAULT_STEPS})")
    ap.add_argument("--cfg", type=float, default=DEFAULT_CFG,
                    help=f"CFG scale (default: {DEFAULT_CFG})")
    ap.add_argument("--gallery", action="store_true",
                    help="Enable gallery mode (show generation history)")
    ap.add_argument("--preset", action="append", default=[], metavar='"label::prompt"',
                    help="Prompt preset button (repeatable). Format: label::prompt or just prompt")
    args = ap.parse_args()

    server_password = args.password
    gallery_enabled = args.gallery
    comfyui_url = args.comfyui_url.rstrip("/")
    configured_steps = args.steps
    configured_cfg = args.cfg

    # Parse presets
    global prompt_presets
    for i, raw in enumerate(args.preset, 1):
        if "::" in raw:
            label, prompt_text = raw.split("::", 1)
        else:
            label, prompt_text = f"preset{i}", raw
        prompt_presets.append({"label": label.strip(), "prompt": prompt_text.strip()})

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Check ComfyUI connectivity (retry once after 60s)
    print(f"[info] connecting to ComfyUI: {comfyui_url}", file=sys.stderr)
    comfyui_connected = False
    for attempt in range(2):
        try:
            resp = http_requests.get(f"{comfyui_url}/system_stats", timeout=10)
            resp.raise_for_status()
            print("[info] ComfyUI connection OK", file=sys.stderr)
            comfyui_connected = True
            break
        except Exception as ex:
            if attempt == 0:
                print(f"[warn] Cannot connect to ComfyUI: {ex}", file=sys.stderr)
                print("[info] Retrying in 60 seconds...", file=sys.stderr)
                time.sleep(60)
            else:
                print(f"[error] Cannot connect to ComfyUI at {comfyui_url}: {ex}", file=sys.stderr)
                print("[error] Make sure ComfyUI is running.", file=sys.stderr)
    if not comfyui_connected:
        sys.exit(1)

    # Check required models (GGUF model + GGUF CLIP + VAE)
    print("[info] checking available models...", file=sys.stderr)
    available = comfyui_get_available_models()

    gguf_name = WORKFLOW_TEMPLATE.get(WF_NODE["gguf_loader"], {}).get("inputs", {}).get("unet_name", "")
    clip_gguf_name = WORKFLOW_TEMPLATE.get(WF_NODE["clip_loader_gguf"], {}).get("inputs", {}).get("clip_name", "")
    vae_name = WORKFLOW_TEMPLATE.get(WF_NODE["vae_loader"], {}).get("inputs", {}).get("vae_name", "")

    model_download_urls = {
        "gguf_models": [
            "https://huggingface.co/Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF/resolve/main/v23/Qwen-Rapid-NSFW-v23_Q6_K.gguf",
        ],
        "clip_gguf_models": [
            "https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-heretic-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-heretic.Q6_K.gguf",
            "https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-heretic-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-heretic.mmproj-Q8_0.gguf (mmproj: place in same folder)",
        ],
        "vae_models": [
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        ],
    }

    missing = []
    if gguf_name and gguf_name not in available.get("gguf_models", []):
        missing.append(("GGUF", gguf_name, "gguf_models"))
    if clip_gguf_name and clip_gguf_name not in available.get("clip_gguf_models", []):
        missing.append(("CLIP(GGUF)", clip_gguf_name, "clip_gguf_models"))
    if vae_name and vae_name not in available.get("vae_models", []):
        missing.append(("VAE", vae_name, "vae_models"))

    if missing:
        print("[error] Required models not found in ComfyUI:", file=sys.stderr)
        for label, name, model_type in missing:
            print(f"  - {label}: {name}", file=sys.stderr)
            urls = model_download_urls.get(model_type, [])
            for url in urls:
                print(f"    Download: {url}", file=sys.stderr)
        print("[error] Place model files in ComfyUI's models directory.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[info] GGUF: {gguf_name} ✓", file=sys.stderr)
        print(f"[info] CLIP: {clip_gguf_name} ✓", file=sys.stderr)
        print(f"[info] VAE:  {vae_name} ✓", file=sys.stderr)

    # Auto-register LoRA path in ComfyUI's extra_model_paths.yaml
    yaml_modified = False
    if args.comfyui_path:
        comfyui_dir = Path(args.comfyui_path)
        if not comfyui_dir.is_dir():
            print(f"[error] ComfyUI directory not found: {comfyui_dir}", file=sys.stderr)
            sys.exit(1)
        yaml_modified = ensure_lora_path_in_comfyui(comfyui_dir)

    if yaml_modified:
        if reboot_comfyui_and_wait():
            available = comfyui_get_available_models()
        else:
            print("[warn] ComfyUI reboot failed. LoRAs may not be fully available.", file=sys.stderr)
            print("[warn] Restart ComfyUI manually and re-run this server.", file=sys.stderr)

    # Scan LoRA folder
    lora_registry.extend(scan_lora_folder(available.get("lora_models", [])))
    if lora_registry:
        print(f"[info] {len(lora_registry)} LoRA(s) available", file=sys.stderr)

    # Build model info for UI
    model_info["pipeline"] = f"ComfyUI GGUF ({comfyui_url})"
    model_info["transformer"] = Path(gguf_name).stem if gguf_name else "unknown"
    model_info["text_encoder_class"] = clip_gguf_name
    model_info["vae_class"] = vae_name
    model_info["steps"] = f"{configured_steps} (CFG: {configured_cfg})"
    if lora_registry:
        model_info["loras"] = ", ".join(e["name"] for e in lora_registry)

    # Start worker thread
    threading.Thread(target=worker_loop, daemon=True).start()

    # Start cleanup thread
    threading.Thread(target=cleanup_loop, daemon=True).start()

    print(f"[info] server starting at http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[info] password: {args.password}", file=sys.stderr)

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
