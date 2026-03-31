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
import sys
import threading
import time

import lib.server_common as common
from lib.server_routes import register_routes
from lib.image_utils import preprocess_image, round_up, parse_lora_args

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


# =======================
# Pipeline management
# =======================
pipeline_ref = {"pipe": None, "dtype": None}
pipeline_load_lock = threading.Lock()


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
        for entry in common.lora_registry:
            print(f"[info] loading LoRA adapter: {entry['name']}", file=sys.stderr)
            try:
                pipe.load_lora_weights(entry["path"], adapter_name=entry["name"])
                print(f"[info] LoRA adapter loaded: {entry['name']}", file=sys.stderr)
            except Exception as ex:
                print(f"[warn] LoRA adapter failed: {entry['name']}: {ex}", file=sys.stderr)
                failed_loras.append(entry["name"])
        if failed_loras:
            common.lora_registry[:] = [e for e in common.lora_registry if e["name"] not in failed_loras]
        if common.lora_registry:
            pipe.set_adapters([e["name"] for e in common.lora_registry],
                              [e["default_scale"] for e in common.lora_registry])

        pipeline_ref["pipe"] = pipe
        pipeline_ref["dtype"] = dtype

        # Collect model info for UI display
        common.model_info["pipeline"] = BASE_MODEL_ID
        common.model_info["transformer"] = TRANSFORMER_ID
        common.model_info["dtype"] = str(dtype).replace("torch.", "")
        common.model_info["steps"] = str(NUM_INFERENCE_STEPS)
        try:
            common.model_info["text_encoder_class"] = pipe.text_encoder.__class__.__name__
        except Exception:
            pass
        try:
            common.model_info["tokenizer"] = pipe.tokenizer.__class__.__name__
        except Exception:
            pass
        try:
            common.model_info["vae_class"] = pipe.vae.__class__.__name__
        except Exception:
            pass
        if common.lora_registry:
            common.model_info["loras"] = ", ".join(e["name"] for e in common.lora_registry)

        print("[info] pipeline ready", file=sys.stderr)
        return pipe


def apply_job_loras(lora_selection: list[dict]):
    """Apply selected LoRAs for this job (diffusers set_adapters)."""
    pipe = pipeline_ref["pipe"]
    if pipe is None or not common.lora_registry:
        return

    valid_names = {e["name"] for e in common.lora_registry}
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
        with common.job_lock:
            common.jobs[job_id]["current_step"] = step_index + 1
        # Check cancel request
        if job_id in common.cancel_requests:
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
    while True:
        job_id = None
        with common.job_lock:
            # Remove queued jobs that were cancelled before processing
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

        cancelled = False
        try:
            job = common.jobs[job_id]
            pipe = pipeline_ref["pipe"]

            from PIL import Image

            image_list = []
            for img_path in job["input_paths"]:
                img = Image.open(img_path).convert("RGB")
                img = preprocess_image(img, job["pre_resize"],
                                       MAX_W, MAX_H, W_MULT, H_MULT)
                image_list.append(img)

            if job["t2i"]:
                tw, th = T2I_SIZE
                tw, th = round_up(tw, W_MULT), round_up(th, H_MULT)
                image_list = [Image.new("RGB", (tw, th), (255, 255, 255))]

            # Apply LoRA selection for this job
            if common.lora_registry:
                apply_job_loras(job.get("loras", []))

            result_img = run_inference(pipe, image_list, job["prompt"], job["seed"], job_id)

            # Check if was cancelled during inference
            if job_id in common.cancel_requests:
                cancelled = True
                with common.job_lock:
                    common.cancel_requests.discard(job_id)
                    common.jobs[job_id]["status"] = "cancelled"
            else:
                out_path = common.TMP_DIR / f"{job_id}_out.png"
                result_img.save(out_path)
                with common.job_lock:
                    common.jobs[job_id]["status"] = "done"
                    common.jobs[job_id]["result_path"] = str(out_path)

                # Persist to gallery DB
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
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


# =======================
# Register shared routes & load template
# =======================
HTML_TEMPLATE = common.load_html_template()

register_routes(
    server_title="AIO",
    pre_resize_options=[{"value": "0.3m", "label": "0.3M pixels"}, {"value": "1m", "label": "1M pixels"}],
    pre_resize_map={"0.3m": 300_000, "1m": 1_000_000},
    default_pre_resize="0.3m",
    has_preview=False,
    get_total_steps=lambda: NUM_INFERENCE_STEPS,
    prompt_default=PROMPT_DEFAULT,
    html_template=HTML_TEMPLATE,
)


# =======================
# Entry point
# =======================
def main():
    ap = argparse.ArgumentParser(description="Qwen Image Edit Web Server (Rapid AIO V23)")
    common.add_common_args(ap)
    ap.add_argument("--no-progress", action="store_true", help="Hide HF download progress")
    ap.add_argument("--no-offload", action="store_true",
                    help="Disable offloading, keep all on GPU (high VRAM)")
    ap.add_argument("--offload", action="store_true",
                    help="Sequential CPU offload (slow, low VRAM)")
    ap.add_argument("--lora", action="append", default=[], metavar="REPO_OR_PATH",
                    help="LoRA weights (repeatable). Format: path_or_repo or repo::weight_name")
    args = ap.parse_args()

    common.apply_common_args(args)

    # Parse and register LoRAs
    common.lora_registry.extend(parse_lora_args(args.lora))

    print("[info] loading model...", file=sys.stderr)
    load_pipeline(progress=not args.no_progress,
                  no_offload=args.no_offload, offload=args.offload)

    # Start worker and cleanup threads
    common.start_server_threads(worker_loop)

    print(f"[info] server starting at http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[info] password: {args.password}", file=sys.stderr)

    common.app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
