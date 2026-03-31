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
    resp = http_requests.post(
        f"{comfyui_url}/upload/image",
        files={"image": (filename, io.BytesIO(image_bytes), "image/png")},
        data={"overwrite": "true"},
    )
    resp.raise_for_status()
    return resp.json().get("name", filename)


def comfyui_submit_prompt(workflow: dict) -> str:
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
    try:
        http_requests.post(f"{comfyui_url}/interrupt")
    except Exception as ex:
        print(f"[warn] ComfyUI interrupt failed: {ex}", file=sys.stderr)


def comfyui_get_history(prompt_id: str) -> dict:
    resp = http_requests.get(f"{comfyui_url}/history/{prompt_id}")
    resp.raise_for_status()
    return resp.json()


def comfyui_get_image(filename: str, subfolder: str = "", img_type: str = "output") -> bytes:
    resp = http_requests.get(
        f"{comfyui_url}/view",
        params={"filename": filename, "subfolder": subfolder, "type": img_type},
    )
    resp.raise_for_status()
    return resp.content


def comfyui_get_available_models() -> dict:
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
    if HAS_WEBSOCKET:
        return _wait_ws(prompt_id, job_id, timeout)
    return _wait_poll(prompt_id, job_id, timeout)


def _wait_ws(prompt_id: str, job_id: str, timeout: float) -> dict | None:
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
            if job_id in common.cancel_requests:
                comfyui_interrupt()
                return None

            try:
                raw = ws.recv()
                if not raw:
                    continue
                if isinstance(raw, bytes):
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
# Image processing (for t2i blank image creation)
# =======================
def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def preprocess_image(img: Image.Image, pre_resize_target: int | None) -> Image.Image:
    if pre_resize_target:
        w, h = img.size
        if w * h > pre_resize_target:
            scale = math.sqrt(pre_resize_target / (w * h))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            print(f"[info] pre-resize: {w}x{h} -> {nw}x{nh}", file=sys.stderr)
            img = img.resize((nw, nh))
    w, h = img.size
    if w > MAX_W or h > MAX_H:
        scale = min(MAX_W / w, MAX_H / h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))
    aw, ah = round_up(w, W_MULT), round_up(h, H_MULT)
    if aw > MAX_W or ah > MAX_H:
        scale = min(MAX_W / aw, MAX_H / ah)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))
        aw, ah = round_up(w, W_MULT), round_up(h, H_MULT)
    if w != aw or h != ah:
        bg = img.getpixel((w - 1, h - 1))
        canvas = Image.new("RGB", (aw, ah), bg)
        canvas.paste(img, (0, 0))
        return canvas
    return img


# =======================
# extra_model_paths.yaml auto-registration
# =======================
YAML_SECTION_NAME = "simple_image_edit"
YAML_COMMENT = "# Auto-added by app_comfyui_gguf.py — LoRA path for simple-image-edit server"


def ensure_lora_path_in_comfyui(comfyui_path: Path) -> bool:
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
    N = WF_NODE
    wf = copy.deepcopy(WORKFLOW_TEMPLATE)

    wf[N["load_image1"]]["inputs"]["image"] = image1_name
    wf[N["load_image2"]]["inputs"]["image"] = image2_name or image1_name

    megapixels = (pre_resize / 1_000_000) if pre_resize else 1.0
    wf[N["image_scale1"]]["inputs"]["megapixels"] = megapixels
    wf[N["image_scale2"]]["inputs"]["megapixels"] = megapixels

    wf[N["prompt_text"]]["inputs"]["value"] = prompt

    wf[N["ksampler"]]["inputs"]["seed"] = seed if seed is not None else random.randint(0, 2**32 - 1)
    wf[N["ksampler"]]["inputs"]["steps"] = configured_steps
    wf[N["ksampler"]]["inputs"]["cfg"] = configured_cfg

    # LoRA handling (3 slots)
    # Chain: UnetLoaderGGUF(456) -> 445 -> 451 -> 452 -> ModelSamplingAuraFlow(441)
    active_loras = [l for l in (loras or []) if abs(l.get("scale", 1.0)) > 1e-5][:3]

    loader_keys = ["lora_loader1", "lora_loader2", "lora_loader3"]
    last_model_source = [N["gguf_loader"], 0]
    for i, key in enumerate(loader_keys):
        if i < len(active_loras):
            wf[N[key]]["inputs"]["lora_name"] = active_loras[i]["comfyui_name"]
            wf[N[key]]["inputs"]["strength_model"] = active_loras[i]["scale"]
            last_model_source = [N[key], 0]
        else:
            del wf[N[key]]

    wf[N["model_sampling"]]["inputs"]["model"] = last_model_source

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

            wf = build_workflow(
                image1_name=image_names[0],
                image2_name=image_names[1] if len(image_names) > 1 else None,
                prompt=job["prompt"],
                seed=job["seed"],
                loras=lora_selection,
                pre_resize=job.get("pre_resize"),
            )
            prompt_id = comfyui_submit_prompt(wf)

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

register_routes(
    server_title="ComfyUI-GGUF",
    pre_resize_options=[{"value": "1m", "label": "1M pixels"}, {"value": "2m", "label": "2M pixels"}],
    pre_resize_map={"1m": 1_000_000, "2m": 2_000_000},
    default_pre_resize="1m",
    has_preview=True,
    get_total_steps=lambda: configured_steps,
    prompt_default=PROMPT_DEFAULT,
    html_template=HTML_TEMPLATE,
)


# =======================
# Entry point
# =======================
def main():
    global comfyui_url, configured_steps, configured_cfg

    ap = argparse.ArgumentParser(description="Qwen Image Edit Web Server (ComfyUI GGUF backend)")
    common.add_common_args(ap)
    ap.add_argument("--comfyui-url", default="http://127.0.0.1:8188",
                    help="ComfyUI API URL (default: http://127.0.0.1:8188)")
    ap.add_argument("--comfyui-path", default=None, metavar="DIR",
                    help="ComfyUI installation directory. "
                         "Auto-registers server/LoRA/ in extra_model_paths.yaml")
    ap.add_argument("--gguf-model", default=None, metavar="NAME",
                    help="GGUF model name for UnetLoaderGGUF (auto-detected if omitted)")
    ap.add_argument("--clip-model", default=None, metavar="NAME",
                    help="GGUF CLIP model name for CLIPLoaderGGUF (auto-detected if omitted)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"Inference steps (default: {DEFAULT_STEPS})")
    ap.add_argument("--cfg", type=float, default=DEFAULT_CFG,
                    help=f"CFG scale (default: {DEFAULT_CFG})")
    args = ap.parse_args()

    common.apply_common_args(args)
    comfyui_url = args.comfyui_url.rstrip("/")
    configured_steps = args.steps
    configured_cfg = args.cfg

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

    # Resolve models: CLI arg > workflow template default > auto-detect from ComfyUI
    print("[info] checking available models...", file=sys.stderr)
    available = comfyui_get_available_models()

    wf_gguf_default = WORKFLOW_TEMPLATE.get(WF_NODE["gguf_loader"], {}).get("inputs", {}).get("unet_name", "")
    wf_clip_default = WORKFLOW_TEMPLATE.get(WF_NODE["clip_loader_gguf"], {}).get("inputs", {}).get("clip_name", "")
    vae_name = WORKFLOW_TEMPLATE.get(WF_NODE["vae_loader"], {}).get("inputs", {}).get("vae_name", "")

    def _pick_model(cli_arg, wf_default, model_list, label):
        """Select model: CLI arg > workflow default (if available) > first detected .gguf/.safetensors."""
        if cli_arg:
            if cli_arg in model_list:
                return cli_arg
            print(f"[error] {label}: --{label.lower().replace('(','').replace(')','')}-model '{cli_arg}' not found in ComfyUI", file=sys.stderr)
            print(f"  Available: {model_list[:10]}", file=sys.stderr)
            sys.exit(1)
        if wf_default and wf_default in model_list:
            return wf_default
        candidates = [m for m in model_list if m.endswith((".gguf", ".safetensors"))]
        if candidates:
            picked = candidates[0]
            if picked != wf_default:
                print(f"[info] {label}: '{wf_default}' not found, using '{picked}'", file=sys.stderr)
            return picked
        return None

    gguf_name = _pick_model(args.gguf_model, wf_gguf_default, available.get("gguf_models", []), "GGUF")
    clip_gguf_name = _pick_model(args.clip_model, wf_clip_default, available.get("clip_gguf_models", []), "CLIP(GGUF)")

    missing = []
    if not gguf_name:
        missing.append("GGUF (UnetLoaderGGUF) — no .gguf model detected")
    if not clip_gguf_name:
        missing.append("CLIP(GGUF) (CLIPLoaderGGUF) — no .gguf model detected")
    if vae_name and vae_name not in available.get("vae_models", []):
        missing.append(f"VAE: {vae_name}")

    if missing:
        print("[error] Required models not found in ComfyUI:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("[error] Place model files in ComfyUI's models directory.", file=sys.stderr)
        sys.exit(1)

    # Apply selected models to workflow template
    WORKFLOW_TEMPLATE[WF_NODE["gguf_loader"]]["inputs"]["unet_name"] = gguf_name
    WORKFLOW_TEMPLATE[WF_NODE["clip_loader_gguf"]]["inputs"]["clip_name"] = clip_gguf_name

    print(f"[info] GGUF: {gguf_name} ✓", file=sys.stderr)
    print(f"[info] CLIP: {clip_gguf_name} ✓", file=sys.stderr)
    print(f"[info] VAE:  {vae_name} ✓", file=sys.stderr)

    # Auto-register LoRA path
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
    common.lora_registry.extend(scan_lora_folder(available.get("lora_models", [])))
    if common.lora_registry:
        print(f"[info] {len(common.lora_registry)} LoRA(s) available", file=sys.stderr)

    # Build model info
    common.model_info["pipeline"] = f"ComfyUI GGUF ({comfyui_url})"
    common.model_info["transformer"] = Path(gguf_name).stem if gguf_name else "unknown"
    common.model_info["text_encoder_class"] = clip_gguf_name
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
