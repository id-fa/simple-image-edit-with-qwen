#!/usr/bin/env python3
"""
Web server for OpenAI gpt-image-2 via WaveSpeed AI cloud API.
No local GPU/torch/diffusers dependency — uses WaveSpeed API for inference.

Usage:
  python app_wavespeed_gpt.py
  python app_wavespeed_gpt.py --password mysecret --port 5000
  python app_wavespeed_gpt.py --comfyui-url http://127.0.0.1:8188   # enable prompt enhance
  python app_wavespeed_gpt.py --gallery --password mysecret

Requirements:
  pip install flask requests pillow googletrans==4.0.0rc1
  Environment variable: WAVESPEED_API_KEY
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import requests as http_requests
from PIL import Image

import lib.server_common as common
from lib.server_routes import register_routes

# =======================
# Fixed parameters
# =======================
PROMPT_DEFAULT = (
    "Fix visible seams and misalignment at image boundaries. "
    "Remove all overlaid text, subtitles, and credits. "
    "Remove compression artifacts and upscaling noise. "
    "Preserve the original composition, character identity, pose, and costume exactly."
)

T2I_SIZE = (1024, 1024)

# WaveSpeed API
WAVESPEED_BASE_URL = "https://api.wavespeed.ai/api/v3"
WAVESPEED_UPLOAD_URL = f"{WAVESPEED_BASE_URL}/media/upload/binary"
WAVESPEED_EDIT_URL = f"{WAVESPEED_BASE_URL}/openai/gpt-image-2/edit"

# Image max cap (API does not document explicit limits; use a safe upper bound)
MAX_DIM = 3072

# gpt-image-2 supported aspect ratios
ASPECT_RATIOS = {
    "1:1": 1.0,
    "2:3": 2.0 / 3.0,
    "3:2": 3.0 / 2.0,
}

# Polling
POLL_INTERVAL = 2.0
POLL_TIMEOUT = 600.0

# =======================
# Enhance prompt workflow (optional, requires ComfyUI)
# =======================
_ENHANCE_WF_PATH = Path(__file__).parent / "comfyui_workflow" / "enhance_prompt_api.json"
try:
    with open(_ENHANCE_WF_PATH, encoding="utf-8") as _f:
        ENHANCE_WF_TEMPLATE = json.load(_f)
    HAS_ENHANCE_WF = True
except FileNotFoundError:
    ENHANCE_WF_TEMPLATE = {}
    HAS_ENHANCE_WF = False

# =======================
# Runtime state
# =======================
wavespeed_api_key: str = ""
comfyui_url: str | None = None


# =======================
# WaveSpeed API helpers
# =======================
def _ws_headers() -> dict:
    return {"Authorization": f"Bearer {wavespeed_api_key}"}


def wavespeed_upload_image(image_bytes: bytes) -> str:
    """Upload image to WaveSpeed, return download URL."""
    resp = http_requests.post(
        WAVESPEED_UPLOAD_URL,
        headers=_ws_headers(),
        files={"file": ("image.png", io.BytesIO(image_bytes), "image/png")},
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    url = data.get("download_url")
    if not url:
        raise RuntimeError(f"WaveSpeed upload: download_url が取得できません / No download_url in response: {resp.json()}")
    return url


def wavespeed_submit_edit(
    image_urls: list[str],
    prompt: str,
    aspect_ratio: str | None = None,
) -> tuple[str, str]:
    """Submit edit task, return (task_id, poll_url)."""
    body: dict = {
        "images": image_urls,
        "prompt": prompt,
    }
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio

    resp = http_requests.post(
        WAVESPEED_EDIT_URL,
        headers={**_ws_headers(), "Content-Type": "application/json"},
        json=body,
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 200:
        raise RuntimeError(f"WaveSpeed API error: {result.get('message', result)}")

    data = result.get("data", {})
    task_id = data.get("id")
    poll_url = data.get("urls", {}).get("get", f"{WAVESPEED_BASE_URL}/predictions/{task_id}/result")
    if not task_id:
        raise RuntimeError(f"WaveSpeed submit: task ID が取得できません / No task ID: {result}")
    return task_id, poll_url


def wavespeed_poll_result(
    poll_url: str,
    job_id: str,
    timeout: float = POLL_TIMEOUT,
) -> str | None:
    """Poll for result. Returns output image URL or None on timeout.
    Raises RuntimeError on API failure. Checks cancel_requests each iteration."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job_id in common.cancel_requests:
            return None

        resp = http_requests.get(poll_url, headers=_ws_headers())
        resp.raise_for_status()
        data = resp.json().get("data", {})
        status = data.get("status", "")

        if status == "completed":
            outputs = data.get("outputs", [])
            if outputs:
                return outputs[0]
            raise RuntimeError("WaveSpeed: 完了したが出力がありません / Completed but no outputs")

        if status == "failed":
            error = data.get("error", data)
            raise RuntimeError(f"WaveSpeed generation failed: {error}")

        time.sleep(POLL_INTERVAL)

    return None


def wavespeed_download_image(url: str) -> bytes:
    """Download result image from URL."""
    resp = http_requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


# =======================
# Image preprocessing
# =======================
def preprocess_image(img: Image.Image, pre_resize_target: int | None) -> Image.Image:
    """Pre-resize and clamp to safe upload size."""
    if pre_resize_target:
        w, h = img.size
        if w * h > pre_resize_target:
            scale = math.sqrt(pre_resize_target / (w * h))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            print(f"[info] pre-resize: {w}x{h} -> {nw}x{nh}", file=sys.stderr)
            img = img.resize((nw, nh))

    w, h = img.size
    if w > MAX_DIM or h > MAX_DIM:
        scale = min(MAX_DIM / w, MAX_DIM / h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))
        print(f"[info] clamped to max dim: {w}x{h}", file=sys.stderr)

    return img


def pick_aspect_ratio(w: int, h: int) -> str:
    """Pick closest supported aspect ratio for given image dimensions."""
    ratio = w / h
    return min(ASPECT_RATIOS.items(), key=lambda kv: abs(math.log(kv[1]) - math.log(ratio)))[0]


# =======================
# ComfyUI helpers (for enhance only)
# =======================
def _comfyui_upload_image(image_bytes: bytes, filename: str) -> str:
    resp = http_requests.post(
        f"{comfyui_url}/upload/image",
        files={"image": (filename, io.BytesIO(image_bytes), "image/png")},
        data={"overwrite": "true"},
    )
    resp.raise_for_status()
    return resp.json().get("name", filename)


def _comfyui_submit_prompt(workflow: dict) -> str:
    payload = {"prompt": workflow}
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
    return result["prompt_id"]


def _comfyui_get_history(prompt_id: str) -> dict:
    resp = http_requests.get(f"{comfyui_url}/history/{prompt_id}")
    resp.raise_for_status()
    return resp.json()


# =======================
# Worker loop
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
            image_urls: list[str] = []
            aspect_ratio: str | None = None

            if job["t2i"]:
                tw, th = T2I_SIZE
                img = Image.new("RGB", (tw, th), (255, 255, 255))
                buf = io.BytesIO()
                try:
                    img.save(buf, format="PNG")
                    url = wavespeed_upload_image(buf.getvalue())
                    image_urls.append(url)
                finally:
                    buf.close()
                    img.close()
                aspect_ratio = "1:1"
            else:
                first_size: tuple[int, int] | None = None
                for img_path in job["input_paths"]:
                    img = Image.open(img_path).convert("RGB")
                    img = preprocess_image(img, job["pre_resize"])
                    if first_size is None:
                        first_size = img.size
                    buf = io.BytesIO()
                    try:
                        img.save(buf, format="PNG")
                        url = wavespeed_upload_image(buf.getvalue())
                        image_urls.append(url)
                    finally:
                        buf.close()
                        img.close()
                if first_size is not None:
                    aspect_ratio = pick_aspect_ratio(*first_size)

            task_id, poll_url = wavespeed_submit_edit(
                image_urls=image_urls,
                prompt=job["prompt"],
                aspect_ratio=aspect_ratio,
            )
            print(f"[info] WaveSpeed task submitted: {task_id} (aspect_ratio={aspect_ratio})", file=sys.stderr)

            output_url = wavespeed_poll_result(poll_url, job_id)

            if job_id in common.cancel_requests:
                with common.job_lock:
                    common.cancel_requests.discard(job_id)
                    common.jobs[job_id]["status"] = "cancelled"
            elif output_url is None:
                with common.job_lock:
                    if common.jobs[job_id]["status"] != "cancelled":
                        common.jobs[job_id]["status"] = "error"
                        common.jobs[job_id]["error"] = "WaveSpeed API タイムアウト / WaveSpeed API timeout"
            else:
                img_bytes = wavespeed_download_image(output_url)
                out_path = common.TMP_DIR / f"{job_id}_out.png"
                with open(out_path, "wb") as f:
                    f.write(img_bytes)

                with common.job_lock:
                    common.jobs[job_id]["status"] = "done"
                    common.jobs[job_id]["result_path"] = str(out_path)
                    common.jobs[job_id]["current_step"] = 1

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

# Enhance wiring (deferred to main() for comfyui_url availability)
_enhance_fn = None
effective_has_enhance = False


# =======================
# Entry point
# =======================
def main():
    global wavespeed_api_key, comfyui_url, _enhance_fn, effective_has_enhance

    ap = argparse.ArgumentParser(description="OpenAI gpt-image-2 Web Server (WaveSpeed AI API)")
    common.add_common_args(ap)
    ap.add_argument("--comfyui-url", default=None,
                    help="ComfyUI API URL for prompt enhancement (optional, e.g. http://127.0.0.1:8188)")
    ap.add_argument("--enhance-model", default=None, metavar="NAME",
                    help="GGUF model name for enhance prompt (overrides workflow default)")
    args = ap.parse_args()

    common.apply_common_args(args)

    # Validate API key
    wavespeed_api_key = os.environ.get("WAVESPEED_API_KEY", "")
    if not wavespeed_api_key:
        print("[error] WAVESPEED_API_KEY 環境変数が設定されていません / WAVESPEED_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    print(f"[info] WaveSpeed API key: ...{wavespeed_api_key[-4:]}", file=sys.stderr)

    # ComfyUI for enhance (optional)
    if args.comfyui_url and HAS_ENHANCE_WF:
        comfyui_url = args.comfyui_url.rstrip("/")
        print(f"[info] connecting to ComfyUI for enhance: {comfyui_url}", file=sys.stderr)
        try:
            resp = http_requests.get(f"{comfyui_url}/system_stats", timeout=10)
            resp.raise_for_status()
            print("[info] ComfyUI connection OK (enhance enabled)", file=sys.stderr)

            if args.enhance_model:
                ENHANCE_WF_TEMPLATE["58"]["inputs"]["model_name"] = args.enhance_model
                print(f"[info] enhance model: {args.enhance_model}", file=sys.stderr)

            from lib.comfyui_enhance import run_enhance
            def _make_enhance_fn():
                def fn(prompt_text, image_bytes, image2_bytes=None):
                    return run_enhance(
                        _comfyui_upload_image, _comfyui_submit_prompt, _comfyui_get_history,
                        ENHANCE_WF_TEMPLATE, prompt_text, image_bytes, image2_bytes,
                    )
                return fn
            _enhance_fn = _make_enhance_fn()
            effective_has_enhance = True
        except Exception as ex:
            print(f"[warn] ComfyUI connection failed: {ex}", file=sys.stderr)
            print("[warn] Prompt enhancement disabled.", file=sys.stderr)
    elif args.comfyui_url and not HAS_ENHANCE_WF:
        print("[warn] enhance_prompt_api.json not found. Prompt enhancement disabled.", file=sys.stderr)

    # Register routes
    register_routes(
        server_title="WaveSpeed gpt-image-2",
        pre_resize_options=[
            {"value": "1m", "label": "1M pixels"},
            {"value": "2m", "label": "2M pixels"},
        ],
        pre_resize_map={"1m": 1_000_000, "2m": 2_000_000},
        default_pre_resize="1m",
        has_preview=False,
        has_enhance=effective_has_enhance,
        enhance_fn=_enhance_fn,
        get_total_steps=lambda: 1,
        prompt_default=PROMPT_DEFAULT,
        html_template=HTML_TEMPLATE,
    )

    # Model info for UI
    common.model_info["pipeline"] = "WaveSpeed AI API"
    common.model_info["transformer"] = "OpenAI gpt-image-2 (Edit)"
    common.model_info["api"] = WAVESPEED_EDIT_URL

    common.start_server_threads(worker_loop)

    print(f"[info] server starting at http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[info] password: {args.password}", file=sys.stderr)

    common.app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
