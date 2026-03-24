#!/usr/bin/env python3
"""
Web server for Nunchaku Qwen-Image-Edit-2509 Lightning.
Based on simple_image_edit_nunchaku_qwen.py.

Usage:
  python app_nunchaku.py
  python app_nunchaku.py --password mysecret --port 8080
  python app_nunchaku.py --host 0.0.0.0 --no-progress
  python app_nunchaku.py --no-offload       # full GPU (high VRAM)
  python app_nunchaku.py --rank 128         # higher quality

Note: Requires diffusers==0.36.x (0.37+ has breaking API changes with nunchaku).
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
NUM_INFERENCE_STEPS = 8  # default; overridden by --steps

MAX_W = 2048
MAX_H = 2048
W_MULT = 8
H_MULT = 16

RANK = 32
NUM_BLOCKS_ON_GPU = 1
GPU_MEM_THRESHOLD_GB = 18.0

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
jobs: dict[str, dict] = {}
processing_queue: deque[str] = deque()
current_processing: str | None = None
cancel_requests: set[str] = set()

pipeline_ref = {"pipe": None, "dtype": None}
pipeline_load_lock = threading.Lock()
model_info: dict[str, str] = {}

server_password = "password"
gallery_enabled = False


# =======================
# Image processing helpers
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


def build_scheduler_config():
    return {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }


# =======================
# Pipeline management
# =======================
def load_pipeline(progress: bool = True, no_offload: bool = False,
                  rank: int = RANK, num_blocks_on_gpu: int = NUM_BLOCKS_ON_GPU,
                  lora: str | None = None, lora_weight_name: str | None = None,
                  lora_scale: float = 1.0, steps: int = NUM_INFERENCE_STEPS):
    """Load Nunchaku pipeline once (thread-safe).

    Offload strategy (same as official nunchaku sample):
      no_offload=True:       pipeline.to("cuda")
      GPU > 18GB:            enable_model_cpu_offload()
      GPU <= 18GB:           transformer.set_offload() + enable_sequential_cpu_offload()
    """
    with pipeline_load_lock:
        if pipeline_ref["pipe"] is not None:
            return pipeline_ref["pipe"]

        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
        from nunchaku import NunchakuQwenImageTransformer2DModel
        from nunchaku.utils import get_gpu_memory, get_precision

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

        # Build model path
        precision = get_precision()
        model_path = (
            f"nunchaku-tech/nunchaku-qwen-image-edit-2509/lightning-251115/"
            f"svdq-{precision}_r{rank}-qwen-image-edit-2509-lightning-{steps}steps-251115.safetensors"
        )
        print(f"[info] model_path: {model_path}", file=sys.stderr)

        # Load transformer
        print("[info] loading transformer...", file=sys.stderr)
        t0 = time.time()
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)
        print(f"[info] transformer loaded ({time.time()-t0:.1f}s)", file=sys.stderr)

        # LoRA loading (via lora_qwen.py — ported from ComfyUI-QwenImageLoraLoader)
        if lora:
            print(f"[info] loading LoRA: {lora}", file=sys.stderr)
            lora_path = lora
            if lora_weight_name:
                from huggingface_hub import hf_hub_download
                lora_path = hf_hub_download(repo_id=lora, filename=lora_weight_name)
            elif not os.path.isfile(lora):
                # HF repo without weight_name — download default safetensors
                from huggingface_hub import hf_hub_download
                lora_path = hf_hub_download(repo_id=lora, filename="pytorch_lora_weights.safetensors")
            from nunchaku_lora_qwen import apply_lora
            n_applied = apply_lora(transformer, lora_path, strength=lora_scale)
            print(f"[info] LoRA loaded: {n_applied} layers (scale={lora_scale})", file=sys.stderr)

        # Load pipeline
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(build_scheduler_config())
        print("[info] loading pipeline: Qwen/Qwen-Image-Edit-2509", file=sys.stderr)
        t1 = time.time()
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=dtype,
        )
        print(f"[info] pipeline loaded ({time.time()-t1:.1f}s)", file=sys.stderr)

        # Offload strategy (same as official sample)
        gpu_mem_gb = get_gpu_memory()
        print(f"[info] GPU memory: {gpu_mem_gb:.1f} GB", file=sys.stderr)

        if no_offload:
            print("[info] --no-offload: pipeline.to('cuda')", file=sys.stderr)
            pipe.to("cuda")
        elif gpu_mem_gb > GPU_MEM_THRESHOLD_GB:
            print("[info] high VRAM: enable_model_cpu_offload()", file=sys.stderr)
            pipe.enable_model_cpu_offload()
        else:
            print(f"[info] low VRAM: transformer.set_offload + sequential_cpu_offload "
                  f"(blocks_on_gpu={num_blocks_on_gpu})", file=sys.stderr)
            transformer.set_offload(True, use_pin_memory=False,
                                    num_blocks_on_gpu=num_blocks_on_gpu)
            pipe._exclude_from_cpu_offload.append("transformer")
            pipe.enable_sequential_cpu_offload()

        pipeline_ref["pipe"] = pipe
        pipeline_ref["dtype"] = dtype

        # Update global NUM_INFERENCE_STEPS so run_inference() and job dict use it
        global NUM_INFERENCE_STEPS
        NUM_INFERENCE_STEPS = steps

        # Collect model info for UI display
        model_info["pipeline"] = "Qwen/Qwen-Image-Edit-2509"
        model_info["transformer"] = model_path
        model_info["dtype"] = str(dtype).replace("torch.", "")
        model_info["steps"] = str(steps)
        model_info["rank"] = str(rank)
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
        if lora:
            model_info["lora"] = lora
            if lora_weight_name:
                model_info["lora_weight_name"] = lora_weight_name
            model_info["lora_scale"] = str(lora_scale)

        print("[info] pipeline ready", file=sys.stderr)
        return pipe


def run_inference(pipe, images: list, prompt: str, seed: int | None, job_id: str):
    import torch

    gen = None
    if seed is not None:
        gen = torch.Generator("cuda").manual_seed(seed)

    def on_step_end(pipeline, step_index, timestep, callback_kwargs):
        with job_lock:
            jobs[job_id]["current_step"] = step_index + 1
        if job_id in cancel_requests:
            pipeline._interrupt = True
        return callback_kwargs

    output = pipe(
        image=images,
        prompt=prompt,
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

            result_img = run_inference(pipe, image_list, job["prompt"], job["seed"], job_id)

            if job_id in cancel_requests:
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
    """Parse 'job_id:input:0' or 'job_id:result' and copy file to new job's input slot."""
    import shutil
    parts = ref_str.split(":")
    if len(parts) < 2:
        return None
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
    return render_template_string(HTML_TEMPLATE, gallery_enabled=gallery_enabled)


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

        # Image 1: file upload > gallery ref
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

        # Image 2: file upload > gallery ref (optional)
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


@app.route("/api/queue_info")
def queue_info():
    with job_lock:
        return jsonify({
            "queue_size": len(processing_queue),
            "processing": current_processing is not None,
        })


# =======================
# HTML Template
# =======================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen Image Edit Server (Nunchaku)</title>
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
#lightbox {
  display: none; position: fixed; inset: 0; z-index: 1000;
  background: rgba(0,0,0,0.85); cursor: pointer;
  align-items: center; justify-content: center;
}
#lightbox img { max-width: 95vw; max-height: 95vh; border-radius: 8px; }
.gallery-dl { font-size: 0.7rem; }
.gallery-dl a { color: #a78bfa; text-decoration: none; }
.gallery-dl a:hover { text-decoration: underline; }
#loginGate {
  max-width: 360px; margin: 80px auto; text-align: center;
}
#loginGate h1 { margin-bottom: 24px; }
#loginGate .login-error { color: #f87171; font-size: 0.85rem; margin-top: 8px; display: none; }
.image-slot .preview-thumb {
  max-height: 120px; border-radius: 6px;
  border: 1px solid #3a3a4a; margin-top: 6px; display: none;
}
</style>
</head>
<body>
{% if gallery_enabled %}
<div id="loginGate" class="container">
  <h1 style="color:#a78bfa; font-size:1.4rem;">Qwen Image Edit Server (Nunchaku)</h1>
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
  <h1>Qwen Image Edit Server (Nunchaku)</h1>

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
    </div>

    <label for="prompt">Prompt (blank = default)</label>
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

  {% if gallery_enabled %}
  <div class="card" id="gallerySection">
    <h2 style="font-size:1.1rem; color:#a78bfa; margin:0 0 16px 0;">Gallery</h2>
    <div id="galleryList" style="display:flex; flex-direction:column; gap:16px;">
      <div style="color:#6b7280; font-size:0.85rem;">No completed jobs yet.</div>
    </div>
  </div>
  <div id="lightbox" onclick="this.style.display='none'">
    <img id="lightboxImg">
  </div>
  {% endif %}
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
    dtype: 'Dtype', steps: 'Steps', rank: 'Rank',
    lora: 'LoRA', lora_weight_name: 'LoRA File', lora_scale: 'LoRA Scale'
  };
  const order = ['pipeline','transformer','text_encoder_class','tokenizer','vae_class','dtype','steps','rank','lora','lora_weight_name','lora_scale'];
  for (const k of order) {
    if (d[k]) rows.push(`<div class="mi-row"><span class="mi-label">${labels[k] || k}</span><span class="mi-val">${d[k]}</span></div>`);
  }
  modelInfoEl.innerHTML = rows.join('') || 'No model info';
}).catch(() => { modelInfoEl.textContent = 'Failed to load model info'; });

t2iCheck.addEventListener('change', () => {
  imageSection.style.display = t2iCheck.checked ? 'none' : 'block';
});

function setupPreview(input, preview) {
  input.addEventListener('change', () => {
    if (input.files && input.files[0]) {
      preview.src = URL.createObjectURL(input.files[0]);
      preview.style.display = 'block';
    } else {
      preview.style.display = 'none';
    }
  });
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

cancelBtn.addEventListener('click', async () => {
  if (!activeJobId) return;
  cancelBtn.disabled = true;
  try {
    await fetch(`/api/cancel/${activeJobId}`, { method: 'POST' });
  } catch (err) {}
});

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
          resultArea.innerHTML = `
            <img src="/api/result/${jobId}${rPw}" alt="result">
            <br>
            <a href="/api/result/${jobId}${rPw}" download="result_${jobId}.png">Download</a>
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
      } catch (err) {}
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

function openLightbox(url) {
  const lb = document.getElementById('lightbox');
  if (!lb) return;
  document.getElementById('lightboxImg').src = url;
  lb.style.display = 'flex';
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    const lb = document.getElementById('lightbox');
    if (lb) lb.style.display = 'none';
  }
});

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
  if (selectedSlot1 && !allRefs.has(selectedSlot1)) selectedSlot1 = null;
  if (selectedSlot2 && !allRefs.has(selectedSlot2)) selectedSlot2 = null;

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
      else { selectedSlot1 = r.value; uncheckT2i(); }
      updateSlotIndicators();
    });
  });
  container.querySelectorAll('input[name="gallery_slot2"]').forEach(r => {
    r.addEventListener('click', () => {
      if (selectedSlot2 === r.value) { selectedSlot2 = null; setTimeout(() => { r.checked = false; }, 0); }
      else { selectedSlot2 = r.value; uncheckT2i(); }
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

    ap = argparse.ArgumentParser(description="Qwen Image Edit Web Server (Nunchaku)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    ap.add_argument("--password", default="password", help="Generation password (default: password)")
    ap.add_argument("--no-progress", action="store_true", help="Hide HF download progress")
    ap.add_argument("--no-offload", action="store_true",
                    help="Disable offloading, keep all on GPU (high VRAM)")
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS,
                    help="Inference steps: 4 or 8 (default: 8)")
    ap.add_argument("--rank", type=int, default=RANK,
                    help="Nunchaku rank: 32 or 128 (default: 32)")
    ap.add_argument("--num-blocks-on-gpu", type=int, default=NUM_BLOCKS_ON_GPU,
                    help="Blocks to keep on GPU in low-VRAM mode (default: 1)")
    ap.add_argument("--lora", default=None, metavar="REPO_OR_PATH",
                    help="LoRA weights: HF repo ID or local .safetensors path")
    ap.add_argument("--lora-weight-name", default=None, metavar="FILE",
                    help="Weight file name within HF repo (optional)")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="LoRA strength (default: 1.0)")
    ap.add_argument("--gallery", action="store_true",
                    help="Enable gallery mode (show generation history)")
    args = ap.parse_args()

    server_password = args.password
    global gallery_enabled
    gallery_enabled = args.gallery

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("[info] loading model...", file=sys.stderr)
    load_pipeline(progress=not args.no_progress, no_offload=args.no_offload,
                  rank=args.rank, num_blocks_on_gpu=args.num_blocks_on_gpu,
                  lora=args.lora, lora_weight_name=args.lora_weight_name,
                  lora_scale=args.lora_scale, steps=args.steps)

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
