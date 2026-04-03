#!/usr/bin/env python3
"""
Web server for Qwen-Image-Edit using ComfyUI API backend (Nunchaku workflow).
Replaces diffusers pipeline with ComfyUI API calls.
No torch/diffusers dependency — requires only flask, requests, pillow, websocket-client.

Usage:
  python app_comfyui_nunchaku.py
  python app_comfyui_nunchaku.py --password mysecret --port 5000
  python app_comfyui_nunchaku.py --comfyui-url http://192.168.1.100:8188
  python app_comfyui_nunchaku.py --comfyui-path D:\\ComfyUI
  python app_comfyui_nunchaku.py --gallery --password mysecret
  python app_comfyui_nunchaku.py --steps 8 --cfg 1.0

Requirements:
  pip install flask requests pillow websocket-client googletrans==4.0.0rc1
"""

from __future__ import annotations

import argparse
import copy
import gc
import io
import json
import math
import random
import sys
import time
import uuid
from pathlib import Path

import requests as http_requests
from PIL import Image

import lib.server_common as common
from lib.server_routes import register_routes

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

LORA_DIR = Path(__file__).parent / "LoRA"

# =======================
# Enhance prompt workflow (optional)
# =======================
_ENHANCE_WF_PATH = Path(__file__).parent / "comfyui_workflow" / "enhance_prompt_api.json"
try:
    with open(_ENHANCE_WF_PATH, encoding="utf-8") as _f:
        ENHANCE_WF_TEMPLATE = json.load(_f)
    HAS_ENHANCE = True
    print(f"[info] enhance workflow loaded: {_ENHANCE_WF_PATH}", file=sys.stderr)
except FileNotFoundError:
    ENHANCE_WF_TEMPLATE = {}
    HAS_ENHANCE = False

# =======================
# Workflow template (loaded from file)
# =======================
_WORKFLOW_PATH = Path(__file__).parent / "comfyui_workflow" / "comfyui_qwen_image_edit_nunchaku_api.json"
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
    "load_image2":      "127",       # LoadImage - reference / image2
    "image_scale1":     "93",        # ImageScaleToTotalPixels - img1
    "image_scale2":     "138",       # ImageScaleToTotalPixels - img2
    "clip_loader":      "140",       # CLIPLoaderGGUF
    "vae_loader":       "39",        # VAELoader
    "vae_encode":       "88",        # VAEEncode
    "vae_decode":       "8",         # VAEDecode
    "text_neg":         "125",       # TextEncodeQwenImageEditPlus (negative)
    "text_pos":         "126",       # TextEncodeQwenImageEditPlus (positive)
    "model_sampling":   "66",        # ModelSamplingAuraFlow
    "cfg_norm":         "75",        # CFGNorm
    "nunchaku_model":   "133",       # NunchakuQwenImageDiTLoader
    "nunchaku_lora":    "136",       # NunchakuQwenImageLoraStackV3
    "ksampler":         "3",         # KSampler
    "save_image":       "60",        # SaveImage - output
}

# Max LoRA slots in NunchakuQwenImageLoraStackV3
# (template has only slot 1; additional slots are added dynamically)
NUNCHAKU_MAX_LORA_SLOTS = 10


# =======================
# ComfyUI-specific state
# =======================
comfyui_url = "http://127.0.0.1:8188"
comfyui_client_id = uuid.uuid4().hex
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
        ("CLIPLoader", "clip_models", "clip_name"),
        ("CLIPLoaderGGUF", "clip_models", "clip_name"),
        ("VAELoader", "vae_models", "vae_name"),
    ]:
        try:
            resp = http_requests.get(f"{comfyui_url}/object_info/{node_class}", timeout=10)
            data = resp.json()
            models = data[node_class]["input"]["required"][field][0]
            if key in info:
                seen = set(info[key])
                info[key].extend(m for m in models if m not in seen)
            else:
                info[key] = models
        except Exception:
            if key not in info:
                info[key] = []
    # NunchakuQwenImageDiTLoader model names
    try:
        resp = http_requests.get(f"{comfyui_url}/object_info/NunchakuQwenImageDiTLoader", timeout=10)
        data = resp.json()
        info["nunchaku_models"] = data["NunchakuQwenImageDiTLoader"]["input"]["required"]["model_name"][0]
    except Exception:
        info["nunchaku_models"] = []
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
    ws_error_count = 0
    WS_ERROR_LIMIT = 10

    try:
        while time.time() - start < timeout:
            if job_id in common.cancel_requests:
                comfyui_interrupt()
                return None

            try:
                raw = ws.recv()
                ws_error_count = 0
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    # Binary frame = preview image from ComfyUI
                    if len(raw) > 8:
                        preview_data = raw[8:]
                        with common.job_lock:
                            if job_id in common.jobs:
                                common.jobs[job_id]["preview"] = preview_data
                    continue
                data = json.loads(raw)
                msg_type = data.get("type", "")
                msg_data = data.get("data", {})

                if msg_data.get("prompt_id") != prompt_id:
                    continue

                if msg_type == "progress":
                    step = msg_data.get("value", 0)
                    total = msg_data.get("max", configured_steps)
                    with common.job_lock:
                        if job_id in common.jobs:
                            common.jobs[job_id]["current_step"] = step
                            common.jobs[job_id]["total_steps"] = total

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
                ws_error_count += 1
                print(f"[warn] WS recv error ({ws_error_count}/{WS_ERROR_LIMIT}): {ex}", file=sys.stderr)
                if ws_error_count >= WS_ERROR_LIMIT:
                    raise RuntimeError("ComfyUI との接続が切断されました / ComfyUI connection lost")
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
        if job_id in common.cancel_requests:
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
YAML_COMMENT = "# Auto-added by app_comfyui_nunchaku.py — LoRA path for simple-image-edit server"


def ensure_lora_path_in_comfyui(comfyui_path: Path) -> bool:
    """Register server/LoRA/ in ComfyUI's extra_model_paths.yaml if not already present."""
    if not LORA_DIR.is_dir():
        return False

    yaml_path = comfyui_path / "extra_model_paths.yaml"
    lora_abs = str(LORA_DIR.resolve()).replace("\\", "/")

    existing_content = ""
    if yaml_path.exists():
        existing_content = yaml_path.read_text(encoding="utf-8")
        normalized = existing_content.replace("\\", "/")
        if lora_abs in normalized:
            print(f"[info] LoRA path already in {yaml_path}", file=sys.stderr)
            return False

    new_section = f"\n{YAML_COMMENT}\n{YAML_SECTION_NAME}:\n    loras: {lora_abs}/\n"

    if existing_content.strip():
        backup_path = yaml_path.with_suffix(".yaml.bak")
        backup_path.write_text(existing_content, encoding="utf-8")
        print(f"[info] backed up {yaml_path} -> {backup_path}", file=sys.stderr)

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
def scan_lora_folder() -> list[dict]:
    """Scan server/LoRA/ folder and register all LoRA files.

    Files are registered by filename directly — the LoRA directory is
    expected to be in ComfyUI's search path via extra_model_paths.yaml.
    """
    if not LORA_DIR.is_dir():
        print(f"[info] LoRA directory not found: {LORA_DIR}", file=sys.stderr)
        return []

    registry = []
    for f in sorted(LORA_DIR.iterdir()):
        if f.suffix.lower() not in (".safetensors", ".ckpt", ".pt"):
            continue

        registry.append({
            "name": f.stem,
            "comfyui_name": f.name,
            "local_path": str(f),
            "default_scale": 1.0,
        })
        print(f"[info] LoRA found: {f.name}", file=sys.stderr)

    return registry


# =======================
# Workflow builder
# =======================
def build_workflow(image1_name: str, image2_name: str | None, prompt: str,
                   seed: int | None, loras: list[dict] | None = None,
                   pre_resize: int | None = None) -> dict:
    """Build ComfyUI workflow from template with given parameters.

    Args:
        loras: list of {"comfyui_name": str, "scale": float} (max 10)
        pre_resize: target total pixels (e.g. 1000000, 2000000) for ImageScaleToTotalPixels
    """
    N = WF_NODE  # shorthand
    wf = copy.deepcopy(WORKFLOW_TEMPLATE)

    # Input images
    wf[N["load_image1"]]["inputs"]["image"] = image1_name
    wf[N["load_image2"]]["inputs"]["image"] = image2_name or image1_name

    # Image scale (megapixels) for pre-resize — img1 and img2
    if pre_resize:
        megapixels = pre_resize / 1_000_000
    else:
        megapixels = 1.0
    wf[N["image_scale1"]]["inputs"]["megapixels"] = megapixels
    wf[N["image_scale2"]]["inputs"]["megapixels"] = megapixels

    # Prompt — directly on TextEncodeQwenImageEditPlus nodes
    wf[N["text_pos"]]["inputs"]["prompt"] = prompt

    # Seed, Steps, CFG — directly on KSampler
    wf[N["ksampler"]]["inputs"]["seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf[N["ksampler"]]["inputs"]["steps"] = configured_steps
    wf[N["ksampler"]]["inputs"]["cfg"] = configured_cfg

    # LoRA handling — NunchakuQwenImageLoraStackV3 (dynamic slots)
    active_loras = [l for l in (loras or []) if abs(l.get("scale", 1.0)) > 1e-5]
    active_loras = active_loras[:NUNCHAKU_MAX_LORA_SLOTS]

    lora_node = wf[N["nunchaku_lora"]]
    lora_count = len(active_loras)
    lora_node["inputs"]["lora_count"] = lora_count
    lora_node["inputs"]["\U0001f522 LoRA Count"] = str(lora_count)
    lora_node["inputs"]["toggle_all"] = lora_count > 0

    # Remove template slot 1 defaults, then set active slots
    for key in list(lora_node["inputs"].keys()):
        if key.startswith(("enabled_", "lora_name_", "lora_strength_")):
            del lora_node["inputs"][key]

    for i, lora in enumerate(active_loras, 1):
        lora_node["inputs"][f"enabled_{i}"] = True
        lora_node["inputs"][f"lora_name_{i}"] = lora["comfyui_name"]
        lora_node["inputs"][f"lora_strength_{i}"] = lora["scale"]

    return wf


# =======================
# Job worker
# =======================
def worker_loop():
    while True:
        job_id = None
        with common.job_lock:
            while common.processing_queue:
                candidate = common.processing_queue[0]
                if candidate in common.cancel_requests:
                    common.processing_queue.popleft()
                    common.cancel_requests.discard(candidate)
                    common.jobs[candidate]["status"] = "cancelled"
                    continue
                break

            if common.processing_queue:
                job_id = common.processing_queue[0]
                common.current_processing = job_id
                common.jobs[job_id]["status"] = "processing"
                common.jobs[job_id]["current_step"] = 0

        if job_id is None:
            time.sleep(0.5)
            continue

        try:
            job = common.jobs[job_id]
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
                matched = [l for l in common.lora_registry if l["name"] == sel["name"]]
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

            if job_id in common.cancel_requests:
                with common.job_lock:
                    common.cancel_requests.discard(job_id)
                    common.jobs[job_id]["status"] = "cancelled"
            elif outputs is None:
                with common.job_lock:
                    if common.jobs[job_id]["status"] != "cancelled":
                        common.jobs[job_id]["status"] = "error"
                        common.jobs[job_id]["error"] = "ComfyUI timeout または出力なし / ComfyUI timeout or no output"
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
                out_path = common.TMP_DIR / f"{job_id}_out.png"
                with open(out_path, "wb") as f:
                    f.write(img_bytes)

                with common.job_lock:
                    common.jobs[job_id]["status"] = "done"
                    common.jobs[job_id]["result_path"] = str(out_path)

                common.persist_job_to_db(job_id, job, str(out_path))

        except Exception as ex:
            import traceback
            traceback.print_exc()
            with common.job_lock:
                common.cancel_requests.discard(job_id)
                if common.jobs[job_id]["status"] != "cancelled":
                    common.jobs[job_id]["status"] = "error"
                    common.jobs[job_id]["error"] = str(ex)
        finally:
            with common.job_lock:
                if common.processing_queue and common.processing_queue[0] == job_id:
                    common.processing_queue.popleft()
                common.current_processing = None
            gc.collect()


# =======================
# Register shared routes & load template
# =======================
HTML_TEMPLATE = common.load_html_template()

if HAS_ENHANCE:
    from lib.comfyui_enhance import run_enhance
    def _enhance_prompt(prompt_text, image_bytes):
        return run_enhance(
            comfyui_upload_image, comfyui_submit_prompt, comfyui_get_history,
            ENHANCE_WF_TEMPLATE, prompt_text, image_bytes,
        )

register_routes(
    server_title="ComfyUI-Nunchaku",
    pre_resize_options=[{"value": "1m", "label": "1M pixels"}, {"value": "2m", "label": "2M pixels"}],
    pre_resize_map={"1m": 1_000_000, "2m": 2_000_000},
    default_pre_resize="1m",
    has_preview=True,
    has_enhance=HAS_ENHANCE,
    enhance_fn=_enhance_prompt if HAS_ENHANCE else None,
    get_total_steps=lambda: configured_steps,
    prompt_default=PROMPT_DEFAULT,
    html_template=HTML_TEMPLATE,
)


# =======================
# Entry point
# =======================
def main():
    global comfyui_url, configured_steps, configured_cfg

    ap = argparse.ArgumentParser(description="Qwen Image Edit Web Server (ComfyUI Nunchaku backend)")
    common.add_common_args(ap)
    ap.add_argument("--comfyui-url", default="http://127.0.0.1:8188",
                    help="ComfyUI API URL (default: http://127.0.0.1:8188)")
    ap.add_argument("--comfyui-path", default=None, metavar="DIR",
                    help="ComfyUI installation directory. "
                         "Auto-registers server/LoRA/ in extra_model_paths.yaml")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"Inference steps (default: {DEFAULT_STEPS})")
    ap.add_argument("--cfg", type=float, default=DEFAULT_CFG,
                    help=f"CFG scale (default: {DEFAULT_CFG})")
    ap.add_argument("--enhance-model", default=None, metavar="NAME",
                    help="GGUF model name for enhance prompt (overrides workflow default)")
    args = ap.parse_args()

    common.apply_common_args(args)
    comfyui_url = args.comfyui_url.rstrip("/")
    configured_steps = args.steps
    configured_cfg = args.cfg

    if args.enhance_model and HAS_ENHANCE:
        ENHANCE_WF_TEMPLATE["58"]["inputs"]["model_name"] = args.enhance_model
        print(f"[info] enhance model: {args.enhance_model}", file=sys.stderr)

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

    # Check required models (Nunchaku model + CLIP + VAE)
    print("[info] checking available models...", file=sys.stderr)
    available = comfyui_get_available_models()

    nunchaku_model_name = WORKFLOW_TEMPLATE.get(WF_NODE["nunchaku_model"], {}).get("inputs", {}).get("model_name", "")
    clip_name = WORKFLOW_TEMPLATE.get(WF_NODE["clip_loader"], {}).get("inputs", {}).get("clip_name", "")
    vae_name = WORKFLOW_TEMPLATE.get(WF_NODE["vae_loader"], {}).get("inputs", {}).get("vae_name", "")

    model_download_urls = {
        "nunchaku_models": [
            "https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509/resolve/main/lightning-251115/svdq-int4_r128-qwen-image-edit-2509-lightning-8steps-251115.safetensors",
            "https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509/resolve/main/lightning-251115/svdq-fp4_r128-qwen-image-edit-2509-lightning-8steps-251115.safetensors",
        ],
        "clip_models": [
            "https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-heretic-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-heretic.Q8_0.gguf",
            "https://huggingface.co/mradermacher/Qwen2.5-VL-7B-Instruct-heretic-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-heretic.mmproj-Q8_0.gguf",
        ],
        "vae_models": [
            "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        ],
    }

    missing = []
    if nunchaku_model_name and nunchaku_model_name not in available.get("nunchaku_models", []):
        missing.append(("Nunchaku", nunchaku_model_name, "nunchaku_models"))
    if clip_name and clip_name not in available.get("clip_models", []):
        missing.append(("CLIP", clip_name, "clip_models"))
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
        print(f"[info] Nunchaku: {nunchaku_model_name} ✓", file=sys.stderr)
        print(f"[info] CLIP: {clip_name} ✓", file=sys.stderr)
        print(f"[info] VAE:  {vae_name} ✓", file=sys.stderr)

    # Check NunchakuQwenImageLoraStackV3 custom node availability
    try:
        resp = http_requests.get(f"{comfyui_url}/object_info/NunchakuQwenImageLoraStackV3", timeout=10)
        if resp.status_code == 200:
            print("[info] NunchakuQwenImageLoraStackV3 node ✓", file=sys.stderr)
        else:
            print("[warn] NunchakuQwenImageLoraStackV3 node not found. Install nunchaku ComfyUI nodes.", file=sys.stderr)
    except Exception:
        print("[warn] Could not verify NunchakuQwenImageLoraStackV3 node availability.", file=sys.stderr)

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
    common.lora_registry.extend(scan_lora_folder())
    if common.lora_registry:
        print(f"[info] {len(common.lora_registry)} LoRA(s) available", file=sys.stderr)

    # Build model info for UI
    common.model_info["pipeline"] = f"ComfyUI Nunchaku ({comfyui_url})"
    common.model_info["transformer"] = Path(nunchaku_model_name).stem if nunchaku_model_name else "unknown"
    common.model_info["text_encoder_class"] = clip_name
    common.model_info["vae_class"] = vae_name
    common.model_info["steps"] = f"{configured_steps} (CFG: {configured_cfg})"
    if common.lora_registry:
        common.model_info["loras"] = ", ".join(e["name"] for e in common.lora_registry)

    common.start_server_threads(worker_loop)

    print(f"[info] server starting at http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[info] password: {args.password}", file=sys.stderr)

    common.app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
