#!/usr/bin/env python3
"""
Web server for Qwen-Image-Edit-Rapid-AIO-V23 (4-step accelerated).
Based on simple_image_edit_rapid_qwen.py.

Usage:
  python app_aio.py
  python app_aio.py --password mysecret --port 8080
  python app_aio.py --host 0.0.0.0 --no-progress
  python app_aio.py --no-offload       # full GPU (high VRAM)
  python app_aio.py --offload          # sequential CPU offload (low VRAM)

No nunchaku dependency. Requires recent diffusers with QwenImageEditPlusPipeline.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template_string

# =======================
# Fixed parameters
# =======================
PROMPT_DEFAULT = "Fix visible seams and misalignment at image boundaries. Remove all overlaid text, subtitles, and credits. Remove compression artifacts and upscaling noise. Preserve the original composition, character identity, pose, and costume exactly."
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
NEGATIVE_PROMPT = " "
NUM_INFERENCE_STEPS = 4

MAX_W = 2048
MAX_H = 2048
W_MULT = 8
H_MULT = 16

TRANSFORMER_ID = "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23"
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

T2I_SIZE = (1024, 1024)

CLEANUP_INTERVAL_SEC = 300  # check every 5 min
MAX_AGE_SEC = 3600          # 1 hour

MAX_QUEUE_WAITING = 2       # max waiting jobs (excluding the one being processed)

TMP_DIR = Path(__file__).parent / "tmp"

# =======================
# Global state
# =======================
app = Flask(__name__)

# Job queue & tracking
job_lock = threading.Lock()
jobs: dict[str, dict] = {}          # job_id -> {status, created, result_path, error, current_step, total_steps, ...}
processing_queue: deque[str] = deque()  # job_ids waiting or processing
current_processing: str | None = None
cancel_requests: set[str] = set()   # job_ids requested to cancel

pipeline_ref = {"pipe": None, "dtype": None}
pipeline_load_lock = threading.Lock()
model_info: dict[str, str] = {}

server_password = "password"
gallery_enabled = False
prompt_presets: list[dict[str, str]] = []  # [{"label": "...", "prompt": "..."}]
lora_registry: list[dict] = []  # [{"name": str, "path": str, "default_scale": float}]
drawings: dict[str, dict] = {}  # drawing_id -> {user_hash, created, path, type, source}


# =======================
# Image processing helpers (from rapid_qwen)
# =======================
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


def preprocess_image(img, pre_resize_target: int | None):
    if pre_resize_target:
        img = pre_resize_to_total_pixels(img, pre_resize_target)
    img = fit_and_align(img, MAX_W, MAX_H, W_MULT, H_MULT)
    return img


# =======================
# LoRA argument parsing
# =======================
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


# =======================
# Pipeline management
# =======================
def load_pipeline(progress: bool = True, no_offload: bool = False,
                  offload: bool = False):
    """Load Rapid AIO V23 pipeline once (thread-safe).

    Offload strategy (3-tier):
      no_offload=True:  pipeline.to("cuda")
      offload=True:     enable_sequential_cpu_offload() (slow, low VRAM)
      default:          enable_model_cpu_offload()
    """
    with pipeline_load_lock:
        if pipeline_ref["pipe"] is not None:
            return pipeline_ref["pipe"]

        import torch
        from diffusers.models import QwenImageTransformer2DModel
        from diffusers import QwenImageEditPlusPipeline

        if progress:
            try:
                from huggingface_hub.utils import enable_progress_bars
                enable_progress_bars()
            except Exception:
                pass
        else:
            try:
                from huggingface_hub.utils import disable_progress_bars
                disable_progress_bars()
            except Exception:
                pass

        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        dtype = torch.bfloat16 if bf16_ok else torch.float16

        # Load transformer from rapid model
        print(f"[info] loading transformer: {TRANSFORMER_ID}", file=sys.stderr)
        t0 = time.time()
        transformer = QwenImageTransformer2DModel.from_pretrained(
            TRANSFORMER_ID, torch_dtype=dtype,
        )
        print(f"[info] transformer loaded ({time.time()-t0:.1f}s)", file=sys.stderr)

        # Load pipeline with base model + rapid transformer
        print(f"[info] loading pipeline: {BASE_MODEL_ID}", file=sys.stderr)
        t1 = time.time()
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_ID, transformer=transformer, torch_dtype=dtype,
        )
        print(f"[info] pipeline loaded ({time.time()-t1:.1f}s)", file=sys.stderr)

        # Offload strategy (must be set before loading LoRA adapters)
        if no_offload:
            print("[info] --no-offload: pipeline.to('cuda')", file=sys.stderr)
            pipe.to("cuda")
        elif offload:
            print("[info] --offload: enable_sequential_cpu_offload()", file=sys.stderr)
            pipe.enable_sequential_cpu_offload()
        else:
            print("[info] enable_model_cpu_offload()", file=sys.stderr)
            pipe.enable_model_cpu_offload()

        # Pre-load all registered LoRAs with adapter names
        failed_loras = []
        for entry in lora_registry:
            print(f"[info] loading LoRA adapter: {entry['name']}", file=sys.stderr)
            try:
                pipe.load_lora_weights(entry["path"], adapter_name=entry["name"])
                print(f"[info] LoRA adapter loaded: {entry['name']}", file=sys.stderr)
            except Exception as ex:
                print(f"[warn] LoRA adapter failed: {entry['name']}: {ex}", file=sys.stderr)
                failed_loras.append(entry["name"])
        if failed_loras:
            lora_registry[:] = [e for e in lora_registry if e["name"] not in failed_loras]
        if lora_registry:
            pipe.set_adapters([e["name"] for e in lora_registry],
                              [e["default_scale"] for e in lora_registry])

        pipeline_ref["pipe"] = pipe
        pipeline_ref["dtype"] = dtype

        # Collect model info for UI display
        model_info["pipeline"] = BASE_MODEL_ID
        model_info["transformer"] = TRANSFORMER_ID
        model_info["dtype"] = str(dtype).replace("torch.", "")
        model_info["steps"] = str(NUM_INFERENCE_STEPS)
        try:
            model_info["text_encoder_class"] = pipe.text_encoder.__class__.__name__
        except Exception:
            pass
        try:
            model_info["tokenizer"] = pipe.tokenizer.__class__.__name__
        except Exception:
            pass
        try:
            model_info["vae_class"] = pipe.vae.__class__.__name__
        except Exception:
            pass
        if lora_registry:
            model_info["loras"] = ", ".join(e["name"] for e in lora_registry)

        print("[info] pipeline ready", file=sys.stderr)
        return pipe


def apply_job_loras(lora_selection: list[dict]):
    """Apply selected LoRAs for this job (diffusers set_adapters)."""
    pipe = pipeline_ref["pipe"]
    if pipe is None or not lora_registry:
        return

    valid_names = {e["name"] for e in lora_registry}
    active_names = []
    active_scales = []
    for sel in lora_selection:
        if sel["name"] in valid_names and abs(sel.get("scale", 1.0)) > 1e-5:
            active_names.append(sel["name"])
            active_scales.append(sel.get("scale", 1.0))

    if active_names:
        pipe.set_adapters(active_names, active_scales)
        print(f"[info] LoRA set: {list(zip(active_names, active_scales))}", file=sys.stderr)
    else:
        pipe.set_adapters([], [])
        print("[info] LoRA cleared", file=sys.stderr)


def run_inference(pipe, images: list, prompt: str, seed: int | None, job_id: str):
    import torch

    gen = None
    if seed is not None:
        gen = torch.Generator("cuda").manual_seed(seed)

    def on_step_end(pipeline, step_index, timestep, callback_kwargs):
        with job_lock:
            jobs[job_id]["current_step"] = step_index + 1
        # Check cancel request
        if job_id in cancel_requests:
            pipeline._interrupt = True
        return callback_kwargs

    output = pipe(
        image=images,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
        true_cfg_scale=TRUE_CFG_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=gen,
        callback_on_step_end=on_step_end,
    )
    return output.images[0]


# =======================
# Job worker
# =======================
def worker_loop():
    global current_processing
    while True:
        job_id = None
        with job_lock:
            # Remove queued jobs that were cancelled before processing
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

        cancelled = False
        try:
            job = jobs[job_id]
            pipe = pipeline_ref["pipe"]

            from PIL import Image

            image_list = []
            for img_path in job["input_paths"]:
                img = Image.open(img_path).convert("RGB")
                img = preprocess_image(img, job["pre_resize"])
                image_list.append(img)

            if job["t2i"]:
                tw, th = T2I_SIZE
                tw, th = round_up(tw, W_MULT), round_up(th, H_MULT)
                image_list = [Image.new("RGB", (tw, th), (255, 255, 255))]

            # Apply LoRA selection for this job
            if lora_registry:
                apply_job_loras(job.get("loras", []))

            result_img = run_inference(pipe, image_list, job["prompt"], job["seed"], job_id)

            # Check if was cancelled during inference
            if job_id in cancel_requests:
                cancelled = True
                with job_lock:
                    cancel_requests.discard(job_id)
                    jobs[job_id]["status"] = "cancelled"
            else:
                out_path = TMP_DIR / f"{job_id}_out.png"
                result_img.save(out_path)
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
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


# =======================
# Cleanup old files
# =======================
def cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL_SEC)
        now = time.time()
        cutoff = now - MAX_AGE_SEC

        # Clean job entries and their files
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

        # Clean up old drawings
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

        # Also clean orphaned files in tmp
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
    """Generate a short hash from client IP (X-Forwarded-For preferred) + User-Agent."""
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not ip:
        ip = request.remote_addr or ""
    ua = request.headers.get("User-Agent", "")
    raw = f"{ip}:{ua}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def resolve_gallery_ref(ref_str: str, new_job_id: str, slot: int) -> str | None:
    """Parse 'job_id:input:0', 'job_id:result', or 'drawing:id' and copy file to new job's input slot."""
    import shutil
    parts = ref_str.split(":")
    if len(parts) < 2:
        return None

    # Handle drawing references
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
    # Password check
    pw = request.form.get("password", "")
    if pw != server_password:
        return jsonify({"error": "パスワードが正しくありません / Invalid password"}), 403

    # Queue capacity check
    with job_lock:
        waiting_count = len(processing_queue)
        # processing_queue includes the one being processed + waiting
        # max total in queue = 1 (processing) + MAX_QUEUE_WAITING (waiting)
        if waiting_count >= 1 + MAX_QUEUE_WAITING:
            return jsonify({"error": "サーバーがビジー状態です / Server is busy. Please try again later.",
                            "busy": True, "queue_size": waiting_count}), 503

    t2i = request.form.get("t2i") == "1"
    prompt = request.form.get("prompt", "").strip() or PROMPT_DEFAULT
    seed_str = request.form.get("seed", "").strip()
    seed = int(seed_str) if seed_str else None
    pre_resize_str = request.form.get("pre_resize", "0.3m")

    pre_resize_map = {"0.3m": 300_000, "1m": 1_000_000}
    pre_resize = pre_resize_map.get(pre_resize_str, 300_000)

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
    import json as _json
    lora_selection = []
    loras_raw = request.form.get("loras", "").strip()
    if loras_raw:
        try:
            lora_selection = _json.loads(loras_raw)
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
            "total_steps": NUM_INFERENCE_STEPS,
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

        # Calculate queue position
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
            "total_steps": job.get("total_steps", NUM_INFERENCE_STEPS),
        })


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "ジョブが見つかりません / Job not found"}), 404
        if job["status"] in ("done", "error", "cancelled"):
            return jsonify({"error": "このジョブは既に終了しています / This job has already finished"}), 400

        cancel_requests.add(job_id)

        # If still queued (not yet processing), remove from queue immediately
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


@app.route("/api/input/<job_id>/<int:index>")
def serve_input(job_id, index):
    if gallery_enabled:
        pw = request.args.get("password", "")
        if pw != server_password:
            return jsonify({"error": "Unauthorized"}), 403
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        paths = job.get("input_paths", [])
        if index < 0 or index >= len(paths):
            return jsonify({"error": "Input index out of range"}), 404
        path = paths[index]
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

    # For drafts, also save bg and overlay layers for resume
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
# HTML Template
# =======================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen Image Edit Server (AIO)</title>
<style>
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0f0f13;
  color: #e0e0e0;
  margin: 0; padding: 20px;
  min-height: 100vh;
}
.container { max-width: 720px; margin: 0 auto; }
h1 {
  text-align: center; font-size: 1.4rem;
  color: #a78bfa; margin-bottom: 24px;
}
.card {
  background: #1a1a24;
  border: 1px solid #2a2a3a;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 16px;
}
label {
  display: block; font-size: 0.85rem;
  color: #9ca3af; margin-bottom: 4px; margin-top: 12px;
}
label:first-child { margin-top: 0; }
input[type="text"], input[type="number"], input[type="password"],
input[type="file"], select, textarea {
  width: 100%; padding: 8px 12px;
  background: #0f0f13; border: 1px solid #3a3a4a;
  border-radius: 6px; color: #e0e0e0;
  font-size: 0.9rem;
}
textarea { resize: vertical; min-height: 60px; font-family: inherit; }
input[type="file"] { padding: 6px; }
select { appearance: auto; }
.checkbox-row {
  display: flex; align-items: center; gap: 8px;
  margin-top: 12px;
}
.checkbox-row input[type="checkbox"] { width: 18px; height: 18px; }
.checkbox-row label { margin: 0; }
.btn-row { display: flex; gap: 10px; align-items: center; margin-top: 20px; }
.lora-section { margin-top: 12px; }
.lora-section-label { font-size: 0.85rem; color: #a78bfa; margin-bottom: 6px; }
.lora-item { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
.lora-item input[type="checkbox"] { margin: 0; accent-color: #7c3aed; }
.lora-item .lora-name { font-size: 0.8rem; color: #d1d5db; min-width: 80px; }
.lora-item input[type="range"] { flex: 1; max-width: 160px; accent-color: #7c3aed; }
.lora-item .lora-scale-val { font-size: 0.75rem; color: #9ca3af; min-width: 30px; text-align: right; }
button {
  padding: 10px 28px;
  background: #7c3aed; color: #fff;
  border: none; border-radius: 8px;
  font-size: 1rem; cursor: pointer;
  transition: background 0.2s;
}
button:hover { background: #6d28d9; }
button:disabled { background: #4a4a5a; cursor: not-allowed; }
button.cancel-btn {
  background: #dc2626; padding: 10px 20px;
}
button.cancel-btn:hover { background: #b91c1c; }
button.cancel-btn:disabled { background: #4a4a5a; }
.status-box {
  margin-top: 16px; padding: 16px;
  background: #12121a; border: 1px solid #2a2a3a;
  border-radius: 8px; display: none;
}
.status-box.visible { display: block; }
.status-text { font-size: 0.9rem; }
.status-text.error { color: #f87171; }
.status-text.cancelled { color: #fb923c; }
.status-text.done { color: #34d399; }
.status-text.processing { color: #fbbf24; }
.status-text.queued { color: #60a5fa; }
.progress-bar-container {
  margin-top: 8px; height: 8px;
  background: #2a2a3a; border-radius: 4px;
  overflow: hidden; display: none;
}
.progress-bar-container.visible { display: block; }
.progress-bar {
  height: 100%; background: #7c3aed;
  border-radius: 4px; transition: width 0.3s ease;
  width: 0%;
}
.result-area { text-align: center; margin-top: 12px; }
.result-area img {
  max-width: 100%; max-height: 512px;
  border-radius: 8px; border: 1px solid #3a3a4a;
}
.result-area a {
  display: inline-block; margin-top: 8px;
  color: #a78bfa; text-decoration: none;
}
.result-area a:hover { text-decoration: underline; }
.queue-badge {
  display: inline-block; padding: 4px 12px;
  background: #1e1e2e; border: 1px solid #3a3a4a;
  border-radius: 16px; font-size: 0.8rem;
  color: #9ca3af; margin-bottom: 12px;
}
.model-info {
  background: #12121a; border: 1px solid #2a2a3a;
  border-radius: 8px; padding: 12px 16px;
  margin-bottom: 16px; font-size: 0.8rem;
  color: #9ca3af; line-height: 1.6;
}
.model-info .mi-row { display: flex; gap: 8px; }
.model-info .mi-label { color: #6b7280; min-width: 110px; flex-shrink: 0; }
.model-info .mi-val { color: #d1d5db; word-break: break-all; }
.preset-row { display: flex; gap: 6px; margin-bottom: 6px; flex-wrap: wrap; }
.preset-btn {
  padding: 3px 10px; font-size: 0.75rem;
  background: #2a2a3a; border: 1px solid #4a4a5a;
  border-radius: 4px; color: #c4b5fd; cursor: pointer;
  transition: background 0.2s;
}
.preset-btn:hover { background: #3a3a4a; }
.translate-row {
  display: flex; gap: 8px; margin-top: 6px;
}
.translate-row button {
  padding: 4px 14px; font-size: 0.8rem;
  background: #2a2a3a; border: 1px solid #3a3a4a;
  border-radius: 6px; color: #d1d5db; cursor: pointer;
  transition: background 0.2s;
}
.translate-row button:hover { background: #3a3a4a; }
.translate-row button:disabled { opacity: 0.5; cursor: not-allowed; }
.image-slot { margin-top: 12px; }
.file-row { display: flex; gap: 6px; align-items: center; }
.file-row input[type="file"] { flex: 1; }
.file-clear-btn {
  padding: 2px 8px; font-size: 1rem; line-height: 1;
  background: #2a2a3a; border: 1px solid #3a3a4a;
  border-radius: 6px; color: #9ca3af; cursor: pointer;
  transition: background 0.2s;
}
.file-clear-btn:hover { background: #3a3a4a; color: #f87171; }
.image-slot.dragover { outline: 2px dashed #a78bfa; outline-offset: -2px; border-radius: 8px; background: rgba(167,139,250,0.08); }
.image-slot .preview-thumb {
  max-height: 120px; border-radius: 6px;
  border: 1px solid #3a3a4a; margin-top: 6px; display: none;
}
/* Drawing Editor */
#drawingEditor {
  display: none; position: fixed; inset: 0; z-index: 1000;
  background: rgba(0,0,0,0.95);
  flex-direction: column; align-items: center;
}
#drawingEditor.visible { display: flex; }
.de-toolbar {
  width: 100%; padding: 6px 12px;
  background: #1a1a24; border-bottom: 1px solid #2a2a3a;
  display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  z-index: 1;
}
.de-group { display: flex; gap: 3px; align-items: center; }
.de-group-label { font-size: 0.7rem; color: #6b7280; margin-right: 2px; white-space: nowrap; }
.de-sep { width: 1px; height: 24px; background: #3a3a4a; margin: 0 4px; }
.de-tool-btn {
  padding: 3px 8px; font-size: 0.75rem;
  background: #2a2a3a; border: 1px solid #4a4a5a;
  border-radius: 4px; color: #d1d5db; cursor: pointer;
  white-space: nowrap;
}
.de-tool-btn.active { background: #7c3aed; border-color: #7c3aed; color: #fff; }
.de-tool-btn:hover { background: #3a3a4a; }
.de-color-swatch {
  width: 20px; height: 20px; border-radius: 3px;
  border: 2px solid transparent; cursor: pointer;
}
.de-color-swatch.active { border-color: #fff; }
.de-color-swatch:hover { border-color: #9ca3af; }
.de-size-btn {
  padding: 1px 6px; font-size: 0.7rem;
  background: #2a2a3a; border: 1px solid #4a4a5a;
  border-radius: 3px; color: #d1d5db; cursor: pointer;
}
.de-size-btn.active { background: #7c3aed; border-color: #7c3aed; color: #fff; }
.de-action-btn {
  padding: 3px 10px; font-size: 0.75rem;
  background: #2a2a3a; border: 1px solid #4a4a5a;
  border-radius: 4px; color: #d1d5db; cursor: pointer;
  white-space: nowrap;
}
.de-action-btn:hover { background: #3a3a4a; }
.de-action-btn.save { background: #065f46; border-color: #10b981; color: #d1fae5; }
.de-action-btn.save:hover { background: #047857; }
.de-action-btn.close { background: #7f1d1d; border-color: #ef4444; color: #fecaca; }
.de-action-btn.close:hover { background: #991b1b; }
.de-canvas-area {
  flex: 1; display: flex; align-items: center; justify-content: center;
  overflow: auto; width: 100%;
}
.de-canvas-wrap { position: relative; display: inline-block; }
#bgCanvas {
  display: block; max-width: 95vw; max-height: calc(100vh - 80px);
}
#drawCanvas {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%;
  cursor: crosshair; touch-action: none;
}
#selCanvas {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
}
#selCanvas.active { pointer-events: auto; cursor: crosshair; touch-action: none; }
#selCanvas.pasting { pointer-events: auto; cursor: move; touch-action: none; }
.de-copy-menu {
  position: fixed; z-index: 1010;
  background: #1a1a2e; border: 1px solid #4a4a6a; border-radius: 6px;
  padding: 4px; display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.de-copy-menu button {
  display: block; width: 100%; padding: 5px 14px; font-size: 0.75rem;
  background: #2a2a3a; border: 1px solid #4a4a5a; border-radius: 3px;
  color: #d1d5db; cursor: pointer; text-align: left; margin: 2px 0;
  white-space: nowrap;
}
.de-copy-menu button:hover { background: #3a3a5a; }
.de-paste-bar {
  display: none; padding: 3px 8px;
  background: #1a2a1a; border: 1px solid #10b981; border-radius: 4px;
  color: #d1fae5; font-size: 0.7rem; align-items: center; gap: 6px;
}
.de-paste-bar.active { display: flex; }
/* Drawings section */
#drawingsSection { display: none; }
.drawings-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.drawing-card {
  background: #12121a; border: 1px solid #2a2a3a;
  border-radius: 6px; padding: 6px; width: 130px;
  display: flex; flex-direction: column; align-items: center; gap: 3px;
}
.drawing-card.draft { border-color: #4a4a1a; }
.drawing-card.draft .drawing-thumb { border-color: #8b8b00; }
.drawing-draft-label { font-size: 0.6rem; color: #eab308; font-weight: bold; }
.drawing-thumb {
  max-width: 118px; max-height: 118px;
  border-radius: 4px; border: 1px solid #3a3a4a;
}
.drawing-info { font-size: 0.65rem; color: #6b7280; text-align: center; }
.drawing-actions { display: flex; gap: 4px; }
.drawing-actions a, .drawing-actions button {
  font-size: 0.65rem; color: #a78bfa; background: none;
  border: 1px solid #3a3a4a; border-radius: 3px;
  padding: 1px 5px; cursor: pointer; text-decoration: none;
}
.drawing-actions button:hover, .drawing-actions a:hover { background: #2a2a3a; }
.drawing-radio-row {
  font-size: 0.65rem; color: #9ca3af;
  display: flex; gap: 4px; justify-content: center;
}
.drawing-radio-row label { display: flex; align-items: center; gap: 2px; cursor: pointer; }
.drawing-radio-row input[type="radio"] { width: 12px; height: 12px; }
.image-slot .preview-thumb[src] { cursor: pointer; }
.image-slot .preview-thumb[src]:hover { outline: 2px solid #a78bfa; outline-offset: 2px; }
.blank-sketch-btn {
  padding: 3px 10px; font-size: 0.75rem;
  background: #2a2a3a; border: 1px solid #4a4a5a;
  border-radius: 4px; color: #c4b5fd; cursor: pointer;
  margin-top: 8px;
}
.blank-sketch-btn:hover { background: #3a3a4a; }
.gallery-item {
  background: #12121a; border: 1px solid #2a2a3a;
  border-radius: 8px; padding: 12px;
}
.gallery-meta { font-size: 0.75rem; color: #6b7280; margin-bottom: 4px; display: flex; align-items: center; flex-wrap: wrap; }
.gallery-delete-btn { background: none; border: 1px solid #ef4444; color: #ef4444; cursor: pointer; font-size: 0.85rem; line-height: 1; padding: 1px 5px; margin-left: 8px; border-radius: 3px; }
.gallery-delete-btn:hover { background: #ef4444; color: #fff; }
.gallery-deleted { opacity: 0.5; }
.gallery-deleted-msg { font-size: 0.8rem; color: #9ca3af; font-style: italic; margin-top: 4px; }
.gallery-prompt {
  font-size: 0.8rem; color: #9ca3af; margin-bottom: 8px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 100%; cursor: pointer;
}
.gallery-prompt.expanded { white-space: normal; }
.gallery-images { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; }
.gallery-img-box { display: inline-flex; flex-direction: column; align-items: center; gap: 2px; }
.gallery-thumb {
  max-height: 150px; max-width: 210px;
  border-radius: 6px; border: 1px solid #3a3a4a;
  cursor: pointer; transition: transform 0.15s;
}
.gallery-thumb:hover { transform: scale(1.05); }
.gallery-radio-row {
  font-size: 0.7rem; color: #9ca3af;
  display: flex; gap: 6px; justify-content: center;
}
.gallery-radio-row label { display: flex; align-items: center; gap: 2px; cursor: pointer; }
.gallery-radio-row input[type="radio"] { width: 14px; height: 14px; }
.gallery-img-label { font-size: 0.65rem; color: #6b7280; }
.gallery-dl { font-size: 0.7rem; }
.gallery-dl a { color: #a78bfa; text-decoration: none; }
.gallery-dl a:hover { text-decoration: underline; }
#loginGate {
  max-width: 360px; margin: 80px auto; text-align: center;
}
#loginGate h1 { margin-bottom: 24px; }
#loginGate .login-error { color: #f87171; font-size: 0.85rem; margin-top: 8px; display: none; }
</style>
</head>
<body>
{% if gallery_enabled %}
<div id="loginGate" class="container">
  <h1 style="color:#a78bfa; font-size:1.4rem;">Qwen Image Edit Server (AIO)</h1>
  <div class="card">
    <label for="loginPassword">Password</label>
    <input type="password" id="loginPassword" placeholder="Enter password">
    <div class="btn-row" style="justify-content:center;">
      <button type="button" id="loginBtn">Login</button>
    </div>
    <div class="login-error" id="loginError">パスワードが正しくありません / Invalid password</div>
  </div>
</div>
{% endif %}
<div class="container" id="mainContent" {% if gallery_enabled %}style="display:none"{% endif %}>
  <h1>Qwen Image Edit Server (AIO)</h1>

  <div class="queue-badge" id="queueBadge">Queue: --</div>
  <div class="model-info" id="modelInfo">Loading model info...</div>

  <form id="editForm" class="card" enctype="multipart/form-data">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" required>

    <div class="checkbox-row">
      <input type="checkbox" id="t2i" name="t2i">
      <label for="t2i">Text-to-Image (t2i) mode (1024x1024)</label>
    </div>

    <div id="imageSection">
      <div class="image-slot">
        <label for="image1">Image 1</label>
        <div class="file-row"><input type="file" id="image1" name="image1" accept="image/*"><button type="button" class="file-clear-btn" onclick="clearFileInput('image1','preview1')">&#215;</button></div>
        <img class="preview-thumb" id="preview1">
        <div id="galleryInd1" style="font-size:0.75rem; color:#a78bfa; margin-top:2px;"></div>
      </div>
      <div class="image-slot">
        <label for="image2">Image 2 / REF (optional)</label>
        <div class="file-row"><input type="file" id="image2" name="image2" accept="image/*"><button type="button" class="file-clear-btn" onclick="clearFileInput('image2','preview2')">&#215;</button></div>
        <img class="preview-thumb" id="preview2">
        <div id="galleryInd2" style="font-size:0.75rem; color:#a78bfa; margin-top:2px;"></div>
      </div>
      <button type="button" class="blank-sketch-btn" onclick="openBlankCanvas()">+ Blank Sketch</button>
    </div>

    <label for="prompt">Prompt (blank = default)</label>
    {% if prompt_presets %}
    <div class="preset-row">
      {% for p in prompt_presets %}
      <button type="button" class="preset-btn" data-prompt="{{ p.prompt }}">{{ p.label }}</button>
      {% endfor %}
    </div>
    {% endif %}
    <textarea id="prompt" name="prompt" placeholder="Fix visible seams and misalignment..."></textarea>
    <div class="translate-row">
      <button type="button" id="translateEn">&rarr; EN</button>
      <button type="button" id="translateZh">&rarr; ZH</button>
    </div>

    <label for="pre_resize">Pre-resize</label>
    <select id="pre_resize" name="pre_resize">
      <option value="0.3m">0.3M pixels</option>
      <option value="1m">1M pixels</option>
    </select>

    <label for="seed">Seed (blank = random)</label>
    <input type="number" id="seed" name="seed" placeholder="optional">

    <div class="lora-section" id="loraSection" style="display:none;">
      <div class="lora-section-label">LoRA</div>
      <div id="loraList"></div>
    </div>

    <div class="btn-row">
      <button type="submit" id="submitBtn">Generate</button>
      <button type="button" id="cancelBtn" class="cancel-btn" style="display:none;">Cancel</button>
    </div>
  </form>

  <div class="status-box" id="statusBox">
    <div class="status-text" id="statusText"></div>
    <div class="progress-bar-container" id="progressBarContainer">
      <div class="progress-bar" id="progressBar"></div>
    </div>
    <div class="result-area" id="resultArea"></div>
  </div>

  <div class="card" id="drawingsSection">
    <h2 style="font-size:1.1rem; color:#a78bfa; margin:0 0 12px 0;">Drawings</h2>
    <div class="drawings-grid" id="drawingsList"></div>
  </div>

  {% if gallery_enabled %}
  <div class="card" id="gallerySection">
    <h2 style="font-size:1.1rem; color:#a78bfa; margin:0 0 16px 0;">Gallery</h2>
    <div id="galleryList" style="display:flex; flex-direction:column; gap:16px;">
      <div style="color:#6b7280; font-size:0.85rem;">No completed jobs yet.</div>
    </div>
  </div>
  {% endif %}

  <div id="drawingEditor">
    <div class="de-toolbar">
      <div class="de-group">
        <button class="de-tool-btn active" data-tool="pen">Pen</button>
        <button class="de-tool-btn" data-tool="eraser">Eraser</button>
        <button class="de-tool-btn" data-tool="overwrite">Cover</button>
        <button class="de-tool-btn" data-tool="select">Select</button>
      </div>
      <div class="de-sep"></div>
      <div class="de-group" id="deColorGroup">
        <span class="de-group-label">Color:</span>
      </div>
      <div class="de-sep"></div>
      <div class="de-group" id="deSizeGroup">
        <span class="de-group-label">Size:</span>
      </div>
      <div class="de-sep"></div>
      <div class="de-group">
        <button class="de-action-btn" id="dePasteBtn" onclick="startPaste()" disabled style="background:#1a1a3a;border-color:#6366f1;color:#a5b4fc;">Paste</button>
        <button class="de-action-btn" onclick="drawUndo()">Undo</button>
        <button class="de-action-btn" onclick="pauseDrawing()" style="background:#4a3a1a;border-color:#eab308;color:#fef08a;">Pause</button>
        <button class="de-action-btn save" onclick="saveDrawingToServer('composite')">Save(+bg)</button>
        <button class="de-action-btn save" onclick="saveDrawingToServer('overlay')">Save(line)</button>
        <button class="de-action-btn close" onclick="closeDrawingEditor()">Close</button>
      </div>
      <div class="de-paste-bar" id="dePasteBar">
        <span>Paste mode — drag to move, corners to resize</span>
        <button class="de-action-btn save" onclick="confirmPaste()">Confirm</button>
        <button class="de-action-btn close" onclick="cancelPaste()">Cancel</button>
      </div>
    </div>
    <div class="de-canvas-area">
      <div class="de-canvas-wrap">
        <canvas id="bgCanvas"></canvas>
        <canvas id="drawCanvas"></canvas>
        <canvas id="selCanvas"></canvas>
      </div>
    </div>
    <div class="de-copy-menu" id="deCopyMenu">
      <button onclick="deCopyRegion('bg')">Copy (bg)</button>
      <button onclick="deCopyRegion('draw')">Copy (draw)</button>
      <button onclick="deCopyRegion('composite')">Copy (+bg)</button>
    </div>
  </div>
</div>

<script>
let sessionPassword = '';
let selectedSlot1 = null;
let selectedSlot2 = null;

// Login gate
(function() {
  const loginGate = document.getElementById('loginGate');
  if (!loginGate) return; // gallery not enabled
  const loginBtn = document.getElementById('loginBtn');
  const loginPw = document.getElementById('loginPassword');
  const loginErr = document.getElementById('loginError');
  function doLogin() {
    const pw = loginPw.value;
    fetch('/api/gallery?password=' + encodeURIComponent(pw))
      .then(r => {
        if (r.status === 403) { loginErr.style.display = 'block'; return; }
        sessionPassword = pw;
        loginGate.style.display = 'none';
        document.getElementById('mainContent').style.display = 'block';
        document.getElementById('password').value = pw;
        initApp();
      }).catch(() => { loginErr.style.display = 'block'; });
  }
  loginBtn.addEventListener('click', doLogin);
  loginPw.addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
})();

// If gallery not enabled, init immediately
if (!document.getElementById('loginGate')) { initApp(); }

function clearFileInput(inputId, previewId) {
  const inp = document.getElementById(inputId);
  const prev = document.getElementById(previewId);
  if (inp) inp.value = '';
  if (prev) { prev.src = ''; prev.style.display = 'none'; }
}

function initApp() {

const form = document.getElementById('editForm');
const statusBox = document.getElementById('statusBox');
const statusText = document.getElementById('statusText');
const resultArea = document.getElementById('resultArea');
const submitBtn = document.getElementById('submitBtn');
const cancelBtn = document.getElementById('cancelBtn');
const t2iCheck = document.getElementById('t2i');
const imageSection = document.getElementById('imageSection');
const image1Input = document.getElementById('image1');
const image2Input = document.getElementById('image2');
const preview1 = document.getElementById('preview1');
const preview2 = document.getElementById('preview2');
const queueBadge = document.getElementById('queueBadge');
const modelInfoEl = document.getElementById('modelInfo');
const progressBarContainer = document.getElementById('progressBarContainer');
const progressBar = document.getElementById('progressBar');

let pollTimer = null;
let queueTimer = null;
let activeJobId = null;

// Model info
fetch('/api/model_info').then(r => r.json()).then(d => {
  const rows = [];
  const labels = {
    pipeline: 'Pipeline', transformer: 'Transformer',
    text_encoder_class: 'Text Encoder', tokenizer: 'Tokenizer', vae_class: 'VAE',
    dtype: 'Dtype', steps: 'Steps',
    loras: 'LoRAs'
  };
  const order = ['pipeline','transformer','text_encoder_class','tokenizer','vae_class','dtype','steps','loras'];
  for (const k of order) {
    if (d[k]) rows.push(`<div class="mi-row"><span class="mi-label">${labels[k] || k}</span><span class="mi-val">${d[k]}</span></div>`);
  }
  modelInfoEl.innerHTML = rows.join('') || 'No model info';
}).catch(() => { modelInfoEl.textContent = 'Failed to load model info'; });

// LoRA list
fetch('/api/loras').then(r => r.json()).then(loras => {
  if (!loras || loras.length === 0) return;
  const section = document.getElementById('loraSection');
  const list = document.getElementById('loraList');
  if (!section || !list) return;
  section.style.display = 'block';
  let html = '';
  for (const lora of loras) {
    const id = 'lora_' + lora.name.replace(/[^a-zA-Z0-9_]/g, '_');
    html += '<div class="lora-item">'
      + '<input type="checkbox" id="' + id + '" data-lora="' + lora.name + '">'
      + '<label class="lora-name" for="' + id + '">' + lora.name + '</label>'
      + '<input type="range" min="0" max="2" step="0.05" value="' + lora.default_scale + '" '
      + 'id="' + id + '_scale" oninput="this.nextElementSibling.textContent=this.value">'
      + '<span class="lora-scale-val">' + lora.default_scale + '</span>'
      + '</div>';
  }
  list.innerHTML = html;
}).catch(() => {});

// t2i toggle
t2iCheck.addEventListener('change', () => {
  imageSection.style.display = t2iCheck.checked ? 'none' : 'block';
});

// image preview
function setupPreview(input, preview) {
  input.addEventListener('change', () => {
    if (input.files && input.files[0]) {
      preview.src = URL.createObjectURL(input.files[0]);
      preview.style.display = 'block';
    } else {
      preview.style.display = 'none';
    }
  });
  preview.addEventListener('click', () => {
    if (typeof openDrawingEditor === 'function' && preview.src && preview.style.display !== 'none') {
      openDrawingEditor(preview.src);
    }
  });
  preview.title = 'Click to open in drawing editor';
}
setupPreview(image1Input, preview1);
setupPreview(image2Input, preview2);

// Drag and drop
function setupDrop(slotEl, inputEl, previewEl) {
  slotEl.addEventListener('dragover', e => { e.preventDefault(); slotEl.classList.add('dragover'); });
  slotEl.addEventListener('dragleave', () => slotEl.classList.remove('dragover'));
  slotEl.addEventListener('drop', e => {
    e.preventDefault(); slotEl.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      inputEl.files = files;
      previewEl.src = URL.createObjectURL(files[0]);
      previewEl.style.display = 'block';
    }
  });
}
document.querySelectorAll('.image-slot').forEach(slot => {
  const inp = slot.querySelector('input[type="file"]');
  const prev = slot.querySelector('.preview-thumb');
  if (inp && prev) setupDrop(slot, inp, prev);
});

// Translate
async function doTranslate(target) {
  const promptEl = document.getElementById('prompt');
  const text = promptEl.value.trim();
  if (!text) return;
  const btns = document.querySelectorAll('.translate-row button');
  btns.forEach(b => b.disabled = true);
  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, target})
    });
    const d = await res.json();
    if (d.error) { alert(d.error); return; }
    promptEl.value = d.translated;
  } catch (e) { alert('Translation error: ' + e.message); }
  finally { btns.forEach(b => b.disabled = false); }
}
document.getElementById('translateEn').addEventListener('click', () => doTranslate('en'));
document.getElementById('translateZh').addEventListener('click', () => doTranslate('zh-cn'));

// Preset buttons
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('prompt').value = btn.getAttribute('data-prompt');
  });
});

// queue info polling
function updateQueueInfo() {
  fetch('/api/queue_info')
    .then(r => r.json())
    .then(d => {
      const s = d.processing ? `Queue: ${d.queue_size} (processing)` : `Queue: ${d.queue_size}`;
      queueBadge.textContent = s;
    })
    .catch(() => {});
}
queueTimer = setInterval(updateQueueInfo, 10000);
updateQueueInfo();

function resetUI() {
  submitBtn.disabled = false;
  cancelBtn.style.display = 'none';
  cancelBtn.disabled = false;
  activeJobId = null;
  progressBarContainer.classList.remove('visible');
  progressBar.style.width = '0%';
  updateQueueInfo();
}

// cancel
cancelBtn.addEventListener('click', async () => {
  if (!activeJobId) return;
  cancelBtn.disabled = true;
  try {
    await fetch(`/api/cancel/${activeJobId}`, { method: 'POST' });
  } catch (err) {
    // ignore
  }
});

// submit
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }

  const fd = new FormData();
  fd.append('password', document.getElementById('password').value);
  fd.append('prompt', document.getElementById('prompt').value);
  fd.append('pre_resize', document.getElementById('pre_resize').value);
  fd.append('seed', document.getElementById('seed').value);

  if (t2iCheck.checked) {
    fd.append('t2i', '1');
  } else {
    const hasFile1 = image1Input.files && image1Input.files[0];
    const hasFile2 = image2Input.files && image2Input.files[0];
    if (hasFile1) {
      fd.append('image1', image1Input.files[0]);
    } else if (selectedSlot1) {
      fd.append('gallery_image1', selectedSlot1);
    } else {
      alert('Image 1 を選択してください / Please select Image 1 (file or gallery)');
      return;
    }
    if (hasFile2) {
      fd.append('image2', image2Input.files[0]);
    } else if (selectedSlot2) {
      fd.append('gallery_image2', selectedSlot2);
    }
  }

  // Collect LoRA selection
  const loraChecks = document.querySelectorAll('#loraList input[type="checkbox"]');
  if (loraChecks.length > 0) {
    const sel = [];
    loraChecks.forEach(cb => {
      if (cb.checked) {
        const scaleEl = document.getElementById(cb.id + '_scale');
        sel.push({name: cb.dataset.lora, scale: parseFloat(scaleEl ? scaleEl.value : '1.0')});
      }
    });
    fd.append('loras', JSON.stringify(sel));
  }

  submitBtn.disabled = true;
  cancelBtn.style.display = 'inline-block';
  cancelBtn.disabled = false;
  statusBox.classList.add('visible');
  statusText.className = 'status-text';
  statusText.textContent = 'Submitting...';
  resultArea.innerHTML = '';
  progressBarContainer.classList.remove('visible');
  progressBar.style.width = '0%';

  try {
    const res = await fetch('/api/submit', { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok) {
      statusText.className = 'status-text error';
      statusText.textContent = data.error || 'Error';
      resetUI();
      return;
    }

    const jobId = data.job_id;
    activeJobId = jobId;
    statusText.className = 'status-text queued';
    statusText.textContent = `Queued (position: ${data.queue_position})`;

    // poll status
    pollTimer = setInterval(async () => {
      try {
        const sr = await fetch(`/api/status/${jobId}`);
        const sd = await sr.json();

        if (sd.status === 'queued') {
          statusText.className = 'status-text queued';
          statusText.textContent = `Queued (position: ${sd.queue_position} / ${sd.queue_total})`;
          progressBarContainer.classList.remove('visible');
        } else if (sd.status === 'processing') {
          statusText.className = 'status-text processing';
          const step = sd.current_step || 0;
          const total = sd.total_steps || 8;
          statusText.textContent = `Processing... (step ${step}/${total})`;
          progressBarContainer.classList.add('visible');
          progressBar.style.width = `${Math.round((step / total) * 100)}%`;
        } else if (sd.status === 'done') {
          clearInterval(pollTimer); pollTimer = null;
          statusText.className = 'status-text done';
          statusText.textContent = 'Done!';
          progressBarContainer.classList.add('visible');
          progressBar.style.width = '100%';
          const rPw = sessionPassword ? '?password=' + encodeURIComponent(sessionPassword) : '';
          const rUrl = `/api/result/${jobId}${rPw}`;
          resultArea.innerHTML = `
            <img src="${rUrl}" alt="result" style="cursor:pointer;" title="Click to open in drawing editor" onclick="openDrawingEditor('${rUrl}')">
            <br>
            <a href="${rUrl}" download="result_${jobId}.png">Download</a>
          `;
          resetUI();
        } else if (sd.status === 'cancelled') {
          clearInterval(pollTimer); pollTimer = null;
          statusText.className = 'status-text cancelled';
          statusText.textContent = 'Cancelled';
          resetUI();
        } else if (sd.status === 'error') {
          clearInterval(pollTimer); pollTimer = null;
          statusText.className = 'status-text error';
          statusText.textContent = 'Error: ' + (sd.error || 'unknown');
          resetUI();
        }
      } catch (err) {
        // keep polling
      }
    }, 3000);

  } catch (err) {
    statusText.className = 'status-text error';
    statusText.textContent = 'Network error: ' + err.message;
    resetUI();
  }
});

} // end initApp

// === Gallery ===
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// === Drawing Editor ===
const DE_COLORS = ['#000000','#ffffff','#ff0000','#ff8800','#ffff00','#00cc00','#0088ff','#8800ff','#ff00ff','#884400'];
const DE_SIZES = [1, 2, 4, 8, 14, 24, 64];
let deCurrentTool = 'pen';
let deCurrentColor = '#ff0000';
let deLineWidth = 4;
let deIsDrawing = false;
let deHistory = [];
let deMaxHistory = 30;
let deSourceUrl = '';

// === Copy & Paste state ===
let deClipboard = null;
let deSelecting = false;
let deSelStart = null;
let deSelEnd = null;
let dePasteMode = false;
let dePasteX = 0, dePasteY = 0;
let dePasteW = 0, dePasteH = 0;
let dePasteDrag = null;
let dePasteResize = null;

function initDrawingEditor() {
  const colorGroup = document.getElementById('deColorGroup');
  const sizeGroup = document.getElementById('deSizeGroup');
  if (!colorGroup || !sizeGroup) return;

  DE_COLORS.forEach((color, i) => {
    const sw = document.createElement('div');
    sw.className = 'de-color-swatch' + (color === deCurrentColor ? ' active' : '');
    sw.style.background = color;
    sw.onclick = () => {
      deCurrentColor = color;
      colorGroup.querySelectorAll('.de-color-swatch').forEach(s => s.classList.remove('active'));
      sw.classList.add('active');
    };
    colorGroup.appendChild(sw);
  });
  const picker = document.createElement('input');
  picker.type = 'color'; picker.value = deCurrentColor;
  picker.style.cssText = 'width:20px;height:20px;border:none;padding:0;cursor:pointer;background:none;';
  picker.oninput = () => {
    deCurrentColor = picker.value;
    colorGroup.querySelectorAll('.de-color-swatch').forEach(s => s.classList.remove('active'));
  };
  colorGroup.appendChild(picker);

  DE_SIZES.forEach((size, i) => {
    const btn = document.createElement('button');
    btn.className = 'de-size-btn' + (size === deLineWidth ? ' active' : '');
    btn.textContent = size;
    btn.onclick = () => {
      deLineWidth = size;
      sizeGroup.querySelectorAll('.de-size-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    };
    sizeGroup.appendChild(btn);
  });

  document.querySelectorAll('.de-tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (dePasteMode) return;
      deCurrentTool = btn.dataset.tool;
      document.querySelectorAll('.de-tool-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const c = document.getElementById('drawCanvas');
      const sc = document.getElementById('selCanvas');
      if (deCurrentTool === 'select') {
        if (c) c.style.cursor = 'default';
        if (sc) sc.classList.add('active');
      } else {
        if (c) c.style.cursor = deCurrentTool === 'pen' ? 'crosshair' : 'cell';
        if (sc) { sc.classList.remove('active'); hideCopyMenu(); clearSelCanvas(); }
      }
    });
  });
}

function openLightbox(url) {
  openDrawingEditor(url);
}

function openDrawingEditor(imageUrl) {
  const editor = document.getElementById('drawingEditor');
  if (!editor) return;
  const bgCanvas = document.getElementById('bgCanvas');
  const drawCanvas = document.getElementById('drawCanvas');
  const selCanvas = document.getElementById('selCanvas');

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = function() {
    bgCanvas.width = drawCanvas.width = selCanvas.width = img.naturalWidth;
    bgCanvas.height = drawCanvas.height = selCanvas.height = img.naturalHeight;
    bgCanvas.getContext('2d').drawImage(img, 0, 0);
    drawCanvas.getContext('2d').clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    selCanvas.getContext('2d').clearRect(0, 0, selCanvas.width, selCanvas.height);
    deHistory = [];
    deSourceUrl = imageUrl;
    dePasteMode = false;
    editor.style.display = 'flex';
    deDrawSaveState();
  };
  img.src = imageUrl;
}

function closeDrawingEditor() {
  const editor = document.getElementById('drawingEditor');
  if (editor) editor.style.display = 'none';
  cancelPaste();
  hideCopyMenu();
  clearSelCanvas();
}

async function pauseDrawing() {
  const bgCanvas = document.getElementById('bgCanvas');
  const drawCanvas = document.getElementById('drawCanvas');
  if (!bgCanvas || !drawCanvas) return;

  const tmp = document.createElement('canvas');
  tmp.width = bgCanvas.width; tmp.height = bgCanvas.height;
  const tmpCtx = tmp.getContext('2d');
  tmpCtx.drawImage(bgCanvas, 0, 0);
  tmpCtx.drawImage(drawCanvas, 0, 0);

  const body = {
    image: tmp.toDataURL('image/png'),
    bg: bgCanvas.toDataURL('image/png'),
    overlay: drawCanvas.toDataURL('image/png'),
    type: 'draft',
    source: deSourceUrl
  };
  if (sessionPassword) body.password = sessionPassword;

  try {
    const res = await fetch('/api/drawing/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    closeDrawingEditor();
    updateDrawings();
  } catch(e) { alert('Save error: ' + e.message); }
}

function resumeDrawing(drawingId) {
  const editor = document.getElementById('drawingEditor');
  if (!editor) return;
  const bgCanvas = document.getElementById('bgCanvas');
  const drawCanvas = document.getElementById('drawCanvas');
  const selCanvas = document.getElementById('selCanvas');
  const pw = sessionPassword ? '?password=' + encodeURIComponent(sessionPassword) : '';

  const bgImg = new Image();
  bgImg.crossOrigin = 'anonymous';
  bgImg.onload = function() {
    bgCanvas.width = drawCanvas.width = selCanvas.width = bgImg.naturalWidth;
    bgCanvas.height = drawCanvas.height = selCanvas.height = bgImg.naturalHeight;
    bgCanvas.getContext('2d').drawImage(bgImg, 0, 0);
    selCanvas.getContext('2d').clearRect(0, 0, selCanvas.width, selCanvas.height);

    const ovImg = new Image();
    ovImg.crossOrigin = 'anonymous';
    ovImg.onload = function() {
      drawCanvas.getContext('2d').clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      drawCanvas.getContext('2d').drawImage(ovImg, 0, 0);
      deHistory = [];
      deSourceUrl = '/api/drawing/' + drawingId + '/bg' + pw;
      editor.style.display = 'flex';
      deDrawSaveState();
    };
    ovImg.onerror = function() {
      drawCanvas.getContext('2d').clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      deHistory = [];
      deSourceUrl = '/api/drawing/' + drawingId + '/bg' + pw;
      editor.style.display = 'flex';
      deDrawSaveState();
    };
    ovImg.src = '/api/drawing/' + drawingId + '/overlay' + pw;
  };
  bgImg.src = '/api/drawing/' + drawingId + '/bg' + pw;
}

function openBlankCanvas() {
  const editor = document.getElementById('drawingEditor');
  if (!editor) return;
  const bgCanvas = document.getElementById('bgCanvas');
  const drawCanvas = document.getElementById('drawCanvas');
  const selCanvas = document.getElementById('selCanvas');
  const w = 1024, h = 1024;
  bgCanvas.width = drawCanvas.width = selCanvas.width = w;
  bgCanvas.height = drawCanvas.height = selCanvas.height = h;
  const bgCtx = bgCanvas.getContext('2d');
  bgCtx.fillStyle = '#ffffff';
  bgCtx.fillRect(0, 0, w, h);
  drawCanvas.getContext('2d').clearRect(0, 0, w, h);
  selCanvas.getContext('2d').clearRect(0, 0, w, h);
  deHistory = [];
  deSourceUrl = '';
  dePasteMode = false;
  editor.style.display = 'flex';
  deDrawSaveState();
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    if (dePasteMode) { cancelPaste(); return; }
    const menu = document.getElementById('deCopyMenu');
    if (menu && menu.style.display === 'block') { hideCopyMenu(); clearSelCanvas(); return; }
    closeDrawingEditor();
  }
  if (e.key === 'Enter' && dePasteMode) { confirmPaste(); }
});

function deGetCoords(e) {
  const c = document.getElementById('drawCanvas');
  if (!c) return {x:0, y:0};
  const r = c.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (c.width / r.width),
    y: (e.clientY - r.top) * (c.height / r.height)
  };
}

function deDrawSaveState() {
  const c = document.getElementById('drawCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  if (deHistory.length >= deMaxHistory) deHistory.shift();
  deHistory.push(ctx.getImageData(0, 0, c.width, c.height));
}

function drawUndo() {
  if (deHistory.length <= 1) return;
  deHistory.pop();
  const c = document.getElementById('drawCanvas');
  if (!c) return;
  c.getContext('2d').putImageData(deHistory[deHistory.length - 1], 0, 0);
}

// === Copy & Paste functions ===
function deGetSelCoords(e) {
  const c = document.getElementById('selCanvas');
  if (!c) return {x:0, y:0};
  const r = c.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (c.width / r.width),
    y: (e.clientY - r.top) * (c.height / r.height)
  };
}

function deGetSelRect() {
  if (!deSelStart || !deSelEnd) return null;
  return {
    x: Math.min(deSelStart.x, deSelEnd.x),
    y: Math.min(deSelStart.y, deSelEnd.y),
    w: Math.abs(deSelEnd.x - deSelStart.x),
    h: Math.abs(deSelEnd.y - deSelStart.y)
  };
}

function deDrawSelRect() {
  const c = document.getElementById('selCanvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const r = deGetSelRect();
  if (!r || r.w < 2 || r.h < 2) return;
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  ctx.strokeRect(r.x, r.y, r.w, r.h);
  ctx.restore();
}

function deCopyRegion(source) {
  const r = deGetSelRect();
  if (!r || r.w < 2 || r.h < 2) return;
  const bgCanvas = document.getElementById('bgCanvas');
  const drawCanvas = document.getElementById('drawCanvas');
  const tmp = document.createElement('canvas');
  tmp.width = Math.round(r.w); tmp.height = Math.round(r.h);
  const tmpCtx = tmp.getContext('2d');
  const sx = Math.round(r.x), sy = Math.round(r.y), sw = tmp.width, sh = tmp.height;
  if (source === 'bg') {
    tmpCtx.drawImage(bgCanvas, sx, sy, sw, sh, 0, 0, sw, sh);
  } else if (source === 'draw') {
    tmpCtx.drawImage(drawCanvas, sx, sy, sw, sh, 0, 0, sw, sh);
  } else {
    tmpCtx.drawImage(bgCanvas, sx, sy, sw, sh, 0, 0, sw, sh);
    tmpCtx.drawImage(drawCanvas, sx, sy, sw, sh, 0, 0, sw, sh);
  }
  deClipboard = tmp;
  const btn = document.getElementById('dePasteBtn');
  if (btn) { btn.disabled = false; btn.style.background = '#2a2a6a'; }
  hideCopyMenu();
  clearSelCanvas();
}

function showCopyMenu(e) {
  const r = deGetSelRect();
  if (!r) return;
  const menu = document.getElementById('deCopyMenu');
  if (!menu) return;
  const sc = document.getElementById('selCanvas');
  if (!sc) return;
  const rect = sc.getBoundingClientRect();
  const scaleX = rect.width / sc.width;
  const scaleY = rect.height / sc.height;
  let mx = rect.left + (r.x + r.w) * scaleX + 4;
  let my = rect.top + r.y * scaleY;
  if (mx + 140 > window.innerWidth) mx = rect.left + r.x * scaleX - 140;
  if (my + 100 > window.innerHeight) my = window.innerHeight - 110;
  menu.style.left = mx + 'px';
  menu.style.top = my + 'px';
  menu.style.display = 'block';
}

function hideCopyMenu() {
  const menu = document.getElementById('deCopyMenu');
  if (menu) menu.style.display = 'none';
}

function clearSelCanvas() {
  const c = document.getElementById('selCanvas');
  if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
  deSelStart = deSelEnd = null;
}

function startPaste() {
  if (!deClipboard || dePasteMode) return;
  hideCopyMenu();
  clearSelCanvas();
  dePasteMode = true;
  dePasteW = deClipboard.width;
  dePasteH = deClipboard.height;
  const dc = document.getElementById('drawCanvas');
  if (dc) {
    dePasteX = Math.round((dc.width - dePasteW) / 2);
    dePasteY = Math.round((dc.height - dePasteH) / 2);
  }
  const sc = document.getElementById('selCanvas');
  if (sc) { sc.classList.remove('active'); sc.classList.add('pasting'); }
  renderPastePreview();
  const bar = document.getElementById('dePasteBar');
  if (bar) bar.classList.add('active');
}

function renderPastePreview() {
  const c = document.getElementById('selCanvas');
  if (!c || !deClipboard) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(deClipboard, dePasteX, dePasteY, dePasteW, dePasteH);
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = '#00aaff';
  ctx.lineWidth = 2;
  ctx.strokeRect(dePasteX, dePasteY, dePasteW, dePasteH);
  ctx.setLineDash([]);
  const hs = Math.max(8, Math.min(16, Math.min(dePasteW, dePasteH) * 0.1));
  ctx.fillStyle = '#00aaff';
  const corners = [
    [dePasteX, dePasteY], [dePasteX + dePasteW, dePasteY],
    [dePasteX, dePasteY + dePasteH], [dePasteX + dePasteW, dePasteY + dePasteH]
  ];
  for (const [cx, cy] of corners) ctx.fillRect(cx - hs/2, cy - hs/2, hs, hs);
  ctx.fillStyle = '#00aaff';
  ctx.font = '14px monospace';
  ctx.fillText(Math.round(dePasteW) + '\u00d7' + Math.round(dePasteH), dePasteX, dePasteY - 6);
  ctx.restore();
}

function confirmPaste() {
  if (!deClipboard || !dePasteMode) return;
  const dc = document.getElementById('drawCanvas');
  if (!dc) return;
  deDrawSaveState();
  const ctx = dc.getContext('2d');
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(deClipboard, dePasteX, dePasteY, dePasteW, dePasteH);
  cancelPaste();
}

function cancelPaste() {
  dePasteMode = false;
  dePasteDrag = null;
  dePasteResize = null;
  const c = document.getElementById('selCanvas');
  if (c) {
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
    c.classList.remove('pasting');
    if (deCurrentTool === 'select') c.classList.add('active');
  }
  const bar = document.getElementById('dePasteBar');
  if (bar) bar.classList.remove('active');
}

function deHitPasteCorner(p) {
  const sc = document.getElementById('selCanvas');
  if (!sc) return null;
  const r = sc.getBoundingClientRect();
  const hitR = 14 * (sc.width / r.width);
  const corners = [
    {x: dePasteX, y: dePasteY, c: 'tl'},
    {x: dePasteX + dePasteW, y: dePasteY, c: 'tr'},
    {x: dePasteX, y: dePasteY + dePasteH, c: 'bl'},
    {x: dePasteX + dePasteW, y: dePasteY + dePasteH, c: 'br'}
  ];
  for (const corner of corners) {
    if (Math.abs(p.x - corner.x) < hitR && Math.abs(p.y - corner.y) < hitR) return corner.c;
  }
  return null;
}

function deHitPasteRect(p) {
  return p.x >= dePasteX && p.x <= dePasteX + dePasteW &&
         p.y >= dePasteY && p.y <= dePasteY + dePasteH;
}

// === Pointer events for drawing, selection, and paste ===
(function() {
  // Drawing events on drawCanvas
  document.addEventListener('pointerdown', function(e) {
    const c = document.getElementById('drawCanvas');
    if (!c || e.target !== c) return;
    if (dePasteMode || deCurrentTool === 'select') return;
    e.preventDefault();
    deIsDrawing = true;
    deDrawSaveState();
    const ctx = c.getContext('2d');
    const p = deGetCoords(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = deLineWidth;
    if (deCurrentTool === 'pen') {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = deCurrentColor;
    } else if (deCurrentTool === 'eraser') {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.strokeStyle = 'rgba(0,0,0,1)';
    } else if (deCurrentTool === 'overwrite') {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = '#ffffff';
    }
  });

  document.addEventListener('pointermove', function(e) {
    if (!deIsDrawing) return;
    const c = document.getElementById('drawCanvas');
    if (!c) return;
    e.preventDefault();
    const ctx = c.getContext('2d');
    const p = deGetCoords(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  });

  document.addEventListener('pointerup', function() {
    deIsDrawing = false;
  });

  // Selection & paste events on selCanvas
  document.addEventListener('pointerdown', function(e) {
    const sc = document.getElementById('selCanvas');
    if (!sc || e.target !== sc) return;

    if (dePasteMode) {
      e.preventDefault();
      const p = deGetSelCoords(e);
      const corner = deHitPasteCorner(p);
      if (corner) {
        dePasteResize = {startX: p.x, startY: p.y, origW: dePasteW, origH: dePasteH, origX: dePasteX, origY: dePasteY, corner: corner};
      } else if (deHitPasteRect(p)) {
        dePasteDrag = {startX: p.x, startY: p.y, origX: dePasteX, origY: dePasteY};
      }
      return;
    }

    if (deCurrentTool === 'select') {
      e.preventDefault();
      hideCopyMenu();
      deSelecting = true;
      deSelStart = deGetSelCoords(e);
      deSelEnd = {...deSelStart};
    }
  });

  document.addEventListener('pointermove', function(e) {
    const sc = document.getElementById('selCanvas');
    if (!sc) return;

    if (dePasteMode) {
      if (dePasteResize) {
        e.preventDefault();
        const p = deGetSelCoords(e);
        const dx = p.x - dePasteResize.startX;
        const dy = p.y - dePasteResize.startY;
        const cn = dePasteResize.corner;
        if (cn === 'br') {
          dePasteW = Math.max(10, dePasteResize.origW + dx);
          dePasteH = Math.max(10, dePasteResize.origH + dy);
        } else if (cn === 'bl') {
          const nw = Math.max(10, dePasteResize.origW - dx);
          dePasteX = dePasteResize.origX + dePasteResize.origW - nw;
          dePasteW = nw;
          dePasteH = Math.max(10, dePasteResize.origH + dy);
        } else if (cn === 'tr') {
          dePasteW = Math.max(10, dePasteResize.origW + dx);
          const nh = Math.max(10, dePasteResize.origH - dy);
          dePasteY = dePasteResize.origY + dePasteResize.origH - nh;
          dePasteH = nh;
        } else if (cn === 'tl') {
          const nw = Math.max(10, dePasteResize.origW - dx);
          const nh = Math.max(10, dePasteResize.origH - dy);
          dePasteX = dePasteResize.origX + dePasteResize.origW - nw;
          dePasteY = dePasteResize.origY + dePasteResize.origH - nh;
          dePasteW = nw;
          dePasteH = nh;
        }
        renderPastePreview();
      } else if (dePasteDrag) {
        e.preventDefault();
        const p = deGetSelCoords(e);
        dePasteX = dePasteDrag.origX + (p.x - dePasteDrag.startX);
        dePasteY = dePasteDrag.origY + (p.y - dePasteDrag.startY);
        renderPastePreview();
      } else if (e.target === sc) {
        const p = deGetSelCoords(e);
        const corner = deHitPasteCorner(p);
        if (corner) {
          sc.style.cursor = (corner === 'tl' || corner === 'br') ? 'nwse-resize' : 'nesw-resize';
        } else if (deHitPasteRect(p)) {
          sc.style.cursor = 'move';
        } else {
          sc.style.cursor = 'default';
        }
      }
      return;
    }

    if (deSelecting) {
      e.preventDefault();
      deSelEnd = deGetSelCoords(e);
      deDrawSelRect();
    }
  });

  document.addEventListener('pointerup', function(e) {
    if (dePasteMode) {
      dePasteDrag = null;
      dePasteResize = null;
      return;
    }
    if (deSelecting) {
      deSelecting = false;
      const r = deGetSelRect();
      if (r && r.w > 4 && r.h > 4) {
        showCopyMenu(e);
      } else {
        clearSelCanvas();
      }
    }
  });
})();

async function saveDrawingToServer(type) {
  const bgCanvas = document.getElementById('bgCanvas');
  const drawCanvas = document.getElementById('drawCanvas');
  if (!bgCanvas || !drawCanvas) return;

  const tmp = document.createElement('canvas');
  tmp.width = bgCanvas.width;
  tmp.height = bgCanvas.height;
  const tmpCtx = tmp.getContext('2d');
  if (type === 'composite') {
    tmpCtx.drawImage(bgCanvas, 0, 0);
    tmpCtx.drawImage(drawCanvas, 0, 0);
  } else {
    tmpCtx.fillStyle = '#ffffff';
    tmpCtx.fillRect(0, 0, tmp.width, tmp.height);
    tmpCtx.drawImage(drawCanvas, 0, 0);
  }
  const dataUrl = tmp.toDataURL('image/png');

  try {
    const body = { image: dataUrl, type: type, source: deSourceUrl };
    if (sessionPassword) body.password = sessionPassword;
    const res = await fetch('/api/drawing/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    updateDrawings();
  } catch(e) { alert('Save error: ' + e.message); }
}

function updateDrawings() {
  const pw = sessionPassword ? '?password=' + encodeURIComponent(sessionPassword) : '';
  fetch('/api/drawings' + pw)
    .then(r => r.json())
    .then(data => {
      if (data.error) return;
      renderDrawings(data.items || []);
    }).catch(() => {});
}

function renderDrawings(items) {
  const container = document.getElementById('drawingsList');
  if (!container) return;
  const section = document.getElementById('drawingsSection');
  if (items.length === 0) {
    if (section) section.style.display = 'none';
    return;
  }
  if (section) section.style.display = 'block';

  const pw = sessionPassword ? '?password=' + encodeURIComponent(sessionPassword) : '';
  let html = '';
  for (const item of items) {
    const imgUrl = '/api/drawing/' + item.drawing_id + pw;
    const isDraft = item.type === 'draft';
    const date = new Date(item.created * 1000);
    const timeStr = date.toLocaleTimeString();

    if (isDraft) {
      html += '<div class="drawing-card draft">'
        + '<div class="drawing-draft-label">DRAFT</div>'
        + '<img class="drawing-thumb" src="' + imgUrl + '" loading="lazy" style="cursor:pointer;" onclick="resumeDrawing(\'' + item.drawing_id + '\')">'
        + '<div class="drawing-info">' + timeStr + '</div>'
        + '<div class="drawing-actions">'
        + '<button onclick="deleteDrawing(\'' + item.drawing_id + '\')">Del</button>'
        + '</div></div>';
    } else {
      const refVal = 'drawing:' + item.drawing_id;
      const typeLabel = item.type === 'composite' ? '+bg' : 'line';
      html += '<div class="drawing-card">'
        + '<img class="drawing-thumb" src="' + imgUrl + '" loading="lazy">'
        + '<div class="drawing-info">' + typeLabel + ' ' + timeStr + '</div>'
        + '<div class="drawing-actions">'
        + '<a href="' + imgUrl + '" download="drawing_' + item.drawing_id + '.png">DL</a>'
        + '<button onclick="deleteDrawing(\'' + item.drawing_id + '\')">Del</button>'
        + '</div>'
        + '<div class="drawing-radio-row">'
        + '<label><input type="radio" name="gallery_slot1" value="' + refVal + '" ' + getCheckedAttr(1, refVal) + '> Img1</label>'
        + '<label><input type="radio" name="gallery_slot2" value="' + refVal + '" ' + getCheckedAttr(2, refVal) + '> Img2</label>'
        + '</div></div>';
    }
  }
  container.innerHTML = html;

  function uncheckT2i() {
    const t2i = document.getElementById('t2i');
    if (t2i && t2i.checked) { t2i.checked = false; t2i.dispatchEvent(new Event('change')); }
  }
  container.querySelectorAll('input[name="gallery_slot1"]').forEach(r => {
    r.addEventListener('click', () => {
      if (selectedSlot1 === r.value) { selectedSlot1 = null; setTimeout(() => { r.checked = false; }, 0); }
      else { selectedSlot1 = r.value; uncheckT2i(); clearFileForSlot(1); }
      updateSlotIndicators();
    });
  });
  container.querySelectorAll('input[name="gallery_slot2"]').forEach(r => {
    r.addEventListener('click', () => {
      if (selectedSlot2 === r.value) { selectedSlot2 = null; setTimeout(() => { r.checked = false; }, 0); }
      else { selectedSlot2 = r.value; uncheckT2i(); clearFileForSlot(2); }
      updateSlotIndicators();
    });
  });
}

function deleteDrawing(drawingId) {
  if (!confirm('Delete this drawing?')) return;
  const pw = sessionPassword ? '?password=' + encodeURIComponent(sessionPassword) : '';
  fetch('/api/drawing/' + drawingId + pw, {method: 'DELETE'})
    .then(r => r.json())
    .then(data => { if (data.ok) updateDrawings(); else alert(data.error || 'Error'); })
    .catch(() => alert('Error'));
}

function clearFileForSlot(slot) {
  if (slot === 1) clearFileInput('image1', 'preview1');
  if (slot === 2) clearFileInput('image2', 'preview2');
}

function updateSlotIndicators() {
  const ind1 = document.getElementById('galleryInd1');
  const ind2 = document.getElementById('galleryInd2');
  if (ind1) ind1.textContent = selectedSlot1 ? 'Gallery: ' + selectedSlot1 : '';
  if (ind2) ind2.textContent = selectedSlot2 ? 'Gallery: ' + selectedSlot2 : '';
}

function getCheckedAttr(slot, value) {
  if (slot === 1 && selectedSlot1 === value) return 'checked';
  if (slot === 2 && selectedSlot2 === value) return 'checked';
  return '';
}

let myUserHash = '';

function deleteGalleryItem(jobId) {
  if (!confirm('この履歴を削除しますか？ / Delete this entry?')) return;
  fetch('/api/gallery/' + jobId + '?password=' + encodeURIComponent(sessionPassword), {method: 'DELETE'})
    .then(r => r.json())
    .then(data => { if (data.ok) updateGallery(); else alert(data.error || 'Error'); })
    .catch(() => alert('Error'));
}

function renderGalleryItem(item) {
  const date = new Date(item.created * 1000);
  const timeStr = date.toLocaleString();

  if (item.deleted) {
    let userInfo = item.user_hash ? ' | user: ' + item.user_hash : '';
    return '<div class="gallery-item gallery-deleted">'
      + '<div class="gallery-meta">' + timeStr + userInfo + '</div>'
      + '<div class="gallery-deleted-msg">このエントリーは削除されました / This entry has been deleted</div>'
      + '</div>';
  }

  let seedInfo = item.seed !== null && item.seed !== undefined ? ' | seed: ' + item.seed : '';
  let t2iInfo = item.t2i ? ' | t2i' : '';
  let userInfo = item.user_hash ? ' | user: ' + item.user_hash : '';

  const pwParam = sessionPassword ? '?password=' + encodeURIComponent(sessionPassword) : '';

  let inputThumbs = '';
  if (!item.t2i) {
    for (let i = 0; i < item.input_count; i++) {
      const imgUrl = '/api/input/' + item.job_id + '/' + i + pwParam;
      const refVal = item.job_id + ':input:' + i;
      inputThumbs += '<div class="gallery-img-box">'
        + '<div class="gallery-img-label">Input ' + (i+1) + '</div>'
        + '<img src="' + imgUrl + '" class="gallery-thumb" onclick="openLightbox(\'' + imgUrl + '\')" loading="lazy">'
        + '<div class="gallery-dl"><a href="' + imgUrl + '" download="input' + i + '_' + item.job_id + '.png">DL</a></div>'
        + '<div class="gallery-radio-row">'
        + '<label><input type="radio" name="gallery_slot1" value="' + refVal + '" ' + getCheckedAttr(1, refVal) + '> Img1</label>'
        + '<label><input type="radio" name="gallery_slot2" value="' + refVal + '" ' + getCheckedAttr(2, refVal) + '> Img2</label>'
        + '</div></div>';
    }
  }

  const resultUrl = '/api/result/' + item.job_id + pwParam;
  const resultRef = item.job_id + ':result';
  const resultThumb = '<div class="gallery-img-box">'
    + '<div class="gallery-img-label">Result</div>'
    + '<img src="' + resultUrl + '" class="gallery-thumb" onclick="openLightbox(\'' + resultUrl + '\')" loading="lazy">'
    + '<div class="gallery-dl"><a href="' + resultUrl + '" download="result_' + item.job_id + '.png">DL</a></div>'
    + '<div class="gallery-radio-row">'
    + '<label><input type="radio" name="gallery_slot1" value="' + resultRef + '" ' + getCheckedAttr(1, resultRef) + '> Img1</label>'
    + '<label><input type="radio" name="gallery_slot2" value="' + resultRef + '" ' + getCheckedAttr(2, resultRef) + '> Img2</label>'
    + '</div></div>';

  let deleteBtn = '';
  if (item.user_hash && item.user_hash === myUserHash) {
    deleteBtn = ' <button class="gallery-delete-btn" onclick="deleteGalleryItem(\'' + item.job_id + '\')">&times;</button>';
  }

  return '<div class="gallery-item" data-job-id="' + item.job_id + '">'
    + '<div class="gallery-meta">' + timeStr + seedInfo + t2iInfo + userInfo + deleteBtn + '</div>'
    + '<div class="gallery-prompt">' + escapeHtml(item.prompt) + '</div>'
    + '<div class="gallery-images">' + inputThumbs + resultThumb + '</div>'
    + '</div>';
}

function saveSelections() {
  const r1 = document.querySelector('input[name="gallery_slot1"]:checked');
  const r2 = document.querySelector('input[name="gallery_slot2"]:checked');
  selectedSlot1 = r1 ? r1.value : null;
  selectedSlot2 = r2 ? r2.value : null;
}

function renderGallery(items) {
  saveSelections();
  const container = document.getElementById('galleryList');
  if (!container) return;
  if (items.length === 0) {
    container.innerHTML = '<div style="color:#6b7280; font-size:0.85rem;">No completed jobs yet.</div>';
    updateSlotIndicators();
    return;
  }
  // Validate selections still exist
  const allRefs = new Set();
  for (const item of items) {
    if (item.deleted) continue;
    allRefs.add(item.job_id + ':result');
    for (let i = 0; i < item.input_count; i++) {
      allRefs.add(item.job_id + ':input:' + i);
    }
  }
  if (selectedSlot1 && !selectedSlot1.startsWith('drawing:') && !allRefs.has(selectedSlot1)) selectedSlot1 = null;
  if (selectedSlot2 && !selectedSlot2.startsWith('drawing:') && !allRefs.has(selectedSlot2)) selectedSlot2 = null;

  container.innerHTML = items.map(renderGalleryItem).join('');

  // Prompt expand
  container.querySelectorAll('.gallery-prompt').forEach(el => {
    el.addEventListener('click', () => el.classList.toggle('expanded'));
  });
  // Radio state sync + click-to-deselect + uncheck t2i
  function uncheckT2i() {
    const t2i = document.getElementById('t2i');
    if (t2i && t2i.checked) { t2i.checked = false; t2i.dispatchEvent(new Event('change')); }
  }
  container.querySelectorAll('input[name="gallery_slot1"]').forEach(r => {
    r.addEventListener('click', () => {
      if (selectedSlot1 === r.value) { selectedSlot1 = null; setTimeout(() => { r.checked = false; }, 0); }
      else { selectedSlot1 = r.value; uncheckT2i(); clearFileForSlot(1); }
      updateSlotIndicators();
    });
  });
  container.querySelectorAll('input[name="gallery_slot2"]').forEach(r => {
    r.addEventListener('click', () => {
      if (selectedSlot2 === r.value) { selectedSlot2 = null; setTimeout(() => { r.checked = false; }, 0); }
      else { selectedSlot2 = r.value; uncheckT2i(); clearFileForSlot(2); }
      updateSlotIndicators();
    });
  });
  updateSlotIndicators();
}

function updateGallery() {
  fetch('/api/gallery?password=' + encodeURIComponent(sessionPassword))
    .then(r => r.json())
    .then(data => {
      if (data.error) return;
      myUserHash = data.caller_hash || '';
      renderGallery(data.items || []);
    })
    .catch(() => {});
}

initDrawingEditor();
updateDrawings();

if (document.getElementById('gallerySection')) {
  setInterval(updateGallery, 30000);
  updateGallery();
}
</script>
</body>
</html>
"""


# =======================
# Entry point
# =======================
def main():
    global server_password

    ap = argparse.ArgumentParser(description="Qwen Image Edit Web Server (Rapid AIO V23)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    ap.add_argument("--password", default="password", help="Generation password (default: password)")
    ap.add_argument("--no-progress", action="store_true", help="Hide HF download progress")
    ap.add_argument("--no-offload", action="store_true",
                    help="Disable offloading, keep all on GPU (high VRAM)")
    ap.add_argument("--offload", action="store_true",
                    help="Sequential CPU offload (slow, low VRAM)")
    ap.add_argument("--lora", action="append", default=[], metavar="REPO_OR_PATH",
                    help="LoRA weights (repeatable). Format: path_or_repo or repo::weight_name")
    ap.add_argument("--gallery", action="store_true",
                    help="Enable gallery mode (show generation history)")
    ap.add_argument("--preset", action="append", default=[], metavar='"label::prompt"',
                    help="Prompt preset button (repeatable). Format: label::prompt or just prompt")
    args = ap.parse_args()

    server_password = args.password
    global gallery_enabled
    gallery_enabled = args.gallery

    # Parse presets
    global prompt_presets
    for i, raw in enumerate(args.preset, 1):
        if "::" in raw:
            label, prompt_text = raw.split("::", 1)
        else:
            label, prompt_text = f"preset{i}", raw
        prompt_presets.append({"label": label.strip(), "prompt": prompt_text.strip()})

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Parse and register LoRAs
    lora_registry.extend(parse_lora_args(args.lora))

    print("[info] loading model...", file=sys.stderr)
    load_pipeline(progress=not args.no_progress,
                  no_offload=args.no_offload, offload=args.offload)

    # Start worker thread
    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    # Start cleanup thread
    cleaner = threading.Thread(target=cleanup_loop, daemon=True)
    cleaner.start()

    print(f"[info] server starting at http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[info] password: {args.password}", file=sys.stderr)

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
