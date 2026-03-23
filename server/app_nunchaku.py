#!/usr/bin/env python3
"""
Web server for Nunchaku Qwen-Image-Edit-2509 Lightning.
Based on simple_image_edit_nunchaku_qwen.py.

Usage:
  python app_nunchaku.py
  python app_nunchaku.py --password mysecret --port 8080
  python app_nunchaku.py --host 0.0.0.0 --progress
  python app_nunchaku.py --no-offload       # full GPU (high VRAM)
  python app_nunchaku.py --rank 128         # higher quality

Note: Requires diffusers==0.36.x (0.37+ has breaking API changes with nunchaku).
"""

from __future__ import annotations

import argparse
import gc
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
NUM_INFERENCE_STEPS = 8

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

server_password = "password"


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
def load_pipeline(progress: bool = False, no_offload: bool = False,
                  rank: int = RANK, num_blocks_on_gpu: int = NUM_BLOCKS_ON_GPU,
                  lora: str | None = None, lora_weight_name: str | None = None,
                  lora_scale: float = 1.0):
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
            f"svdq-{precision}_r{rank}-qwen-image-edit-2509-lightning-{NUM_INFERENCE_STEPS}steps-251115.safetensors"
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
# Routes
# =======================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/submit", methods=["POST"])
def submit():
    pw = request.form.get("password", "")
    if pw != server_password:
        return jsonify({"error": "パスワードが正しくありません"}), 403

    with job_lock:
        waiting_count = len(processing_queue)
        if waiting_count >= 1 + MAX_QUEUE_WAITING:
            return jsonify({"error": "サーバーがビジー状態です。しばらく待ってから再試行してください。",
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
        files = request.files.getlist("images")
        if not files or len(files) < 1:
            return jsonify({"error": "画像ファイルを1つ以上アップロードしてください"}), 400
        if len(files) > 2:
            return jsonify({"error": "画像は最大2ファイルまでです"}), 400

        for i, f in enumerate(files):
            if not f.filename:
                continue
            ext = Path(f.filename).suffix.lower() or ".png"
            save_path = TMP_DIR / f"{job_id}_in{i}{ext}"
            f.save(save_path)
            input_paths.append(str(save_path))

        if not input_paths:
            return jsonify({"error": "有効な画像ファイルがありません"}), 400
    else:
        if not prompt or prompt == PROMPT_DEFAULT:
            return jsonify({"error": "t2iモードではプロンプトの入力が必要です"}), 400

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
        }
        processing_queue.append(job_id)
        queue_pos = len(processing_queue)

    return jsonify({"job_id": job_id, "queue_position": queue_pos})


@app.route("/api/status/<job_id>")
def status(job_id):
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "ジョブが見つかりません"}), 404

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
            return jsonify({"error": "ジョブが見つかりません"}), 404
        if job["status"] in ("done", "error", "cancelled"):
            return jsonify({"error": "このジョブは既に終了しています"}), 400

        cancel_requests.add(job_id)

        if job["status"] == "queued":
            try:
                processing_queue.remove(job_id)
            except ValueError:
                pass
            job["status"] = "cancelled"
            cancel_requests.discard(job_id)

    return jsonify({"ok": True, "message": "キャンセルを要求しました"})


@app.route("/api/result/<job_id>")
def result(job_id):
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "ジョブが見つかりません"}), 404
        if job["status"] != "done":
            return jsonify({"error": "まだ処理中です"}), 400
        rp = job["result_path"]

    if not rp or not os.path.exists(rp):
        return jsonify({"error": "結果ファイルが見つかりません"}), 404

    return send_file(rp, mimetype="image/png", as_attachment=True,
                     download_name=f"result_{job_id}.png")


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
.preview-row {
  display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap;
}
.preview-row img {
  max-height: 120px; border-radius: 6px;
  border: 1px solid #3a3a4a;
}
</style>
</head>
<body>
<div class="container">
  <h1>Qwen Image Edit Server (Nunchaku)</h1>

  <div class="queue-badge" id="queueBadge">Queue: --</div>

  <form id="editForm" class="card" enctype="multipart/form-data">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" required>

    <div class="checkbox-row">
      <input type="checkbox" id="t2i" name="t2i">
      <label for="t2i">Text-to-Image (t2i) mode (1024x1024)</label>
    </div>

    <div id="imageSection">
      <label for="images">Image (1-2 files)</label>
      <input type="file" id="images" name="images" accept="image/*" multiple>
      <div class="preview-row" id="previewRow"></div>
    </div>

    <label for="prompt">Prompt (blank = default)</label>
    <textarea id="prompt" name="prompt" placeholder="Fix visible seams and misalignment..."></textarea>

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
</div>

<script>
const form = document.getElementById('editForm');
const statusBox = document.getElementById('statusBox');
const statusText = document.getElementById('statusText');
const resultArea = document.getElementById('resultArea');
const submitBtn = document.getElementById('submitBtn');
const cancelBtn = document.getElementById('cancelBtn');
const t2iCheck = document.getElementById('t2i');
const imageSection = document.getElementById('imageSection');
const imagesInput = document.getElementById('images');
const previewRow = document.getElementById('previewRow');
const queueBadge = document.getElementById('queueBadge');
const progressBarContainer = document.getElementById('progressBarContainer');
const progressBar = document.getElementById('progressBar');

let pollTimer = null;
let queueTimer = null;
let activeJobId = null;

t2iCheck.addEventListener('change', () => {
  imageSection.style.display = t2iCheck.checked ? 'none' : 'block';
});

imagesInput.addEventListener('change', () => {
  previewRow.innerHTML = '';
  const files = imagesInput.files;
  for (let i = 0; i < Math.min(files.length, 2); i++) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(files[i]);
    previewRow.appendChild(img);
  }
});

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
    const files = imagesInput.files;
    if (!files || files.length === 0) {
      alert('Please select image file(s)');
      return;
    }
    if (files.length > 2) {
      alert('Max 2 files');
      return;
    }
    for (let i = 0; i < files.length; i++) {
      fd.append('images', files[i]);
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
          resultArea.innerHTML = `
            <img src="/api/result/${jobId}" alt="result">
            <br>
            <a href="/api/result/${jobId}" download="result_${jobId}.png">Download</a>
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
    ap.add_argument("--progress", action="store_true", help="Show HF download progress")
    ap.add_argument("--no-offload", action="store_true",
                    help="Disable offloading, keep all on GPU (high VRAM)")
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
    args = ap.parse_args()

    server_password = args.password

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("[info] loading model...", file=sys.stderr)
    load_pipeline(progress=args.progress, no_offload=args.no_offload,
                  rank=args.rank, num_blocks_on_gpu=args.num_blocks_on_gpu,
                  lora=args.lora, lora_weight_name=args.lora_weight_name,
                  lora_scale=args.lora_scale)

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
