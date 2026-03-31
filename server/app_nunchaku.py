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
import math
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
NUM_INFERENCE_STEPS = 8  # default; overridden by --steps

MAX_W = 2048
MAX_H = 2048
W_MULT = 8
H_MULT = 16

RANK = 32
NUM_BLOCKS_ON_GPU = 1
GPU_MEM_THRESHOLD_GB = 18.0

T2I_SIZE = (1024, 1024)

# =======================
# Nunchaku-specific state
# =======================
pipeline_ref = {"pipe": None, "dtype": None}
pipeline_load_lock = threading.Lock()
_pipeline_args: dict = {}  # saved load_pipeline kwargs for reload
_current_lora_key: tuple | None = None  # (name, scale) tuples for change detection


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
                  steps: int = NUM_INFERENCE_STEPS,
                  lora_configs: list[tuple] | None = None):
    """Load Nunchaku pipeline (thread-safe, supports reload).

    Args:
        lora_configs: [(path, scale), ...] LoRAs to apply before offloading.
                      If None, applies all registered LoRAs at default scale.
    """
    with pipeline_load_lock:
        import torch

        # Destroy existing pipeline if reloading
        if pipeline_ref["pipe"] is not None:
            print("[info] destroying existing pipeline for reload...", file=sys.stderr)
            old_pipe = pipeline_ref["pipe"]
            pipeline_ref["pipe"] = None
            pipeline_ref["dtype"] = None
            del old_pipe
            gc.collect()
            torch.cuda.empty_cache()

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

        # Pre-apply LoRAs BEFORE pipeline construction and offloading
        if lora_configs is None:
            lora_configs = [(e["path"], e["default_scale"]) for e in common.lora_registry]
        active_configs = [(p, s) for p, s in lora_configs if abs(s) > 1e-5]
        if active_configs:
            from lib.nunchaku_lora_qwen import compose_loras
            safe_configs = []
            for path, scale in active_configs:
                try:
                    n = compose_loras(transformer, [(path, scale)])
                    safe_configs.append((path, scale))
                    print(f"[info] LoRA OK: {path} (scale={scale}, {n} layers)", file=sys.stderr)
                except Exception as ex:
                    print(f"[warn] LoRA skipped (incompatible): {path}: {ex}", file=sys.stderr)
                    common.lora_registry[:] = [e for e in common.lora_registry if e["path"] != path]
            if safe_configs:
                compose_loras(transformer, safe_configs)
                print(f"[info] LoRA applied: {len(safe_configs)} LoRAs", file=sys.stderr)
            else:
                print("[info] all LoRAs failed, continuing without LoRA", file=sys.stderr)
        else:
            print("[info] no LoRA applied", file=sys.stderr)

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

        # Offload strategy
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

        # Update global NUM_INFERENCE_STEPS
        global NUM_INFERENCE_STEPS
        NUM_INFERENCE_STEPS = steps

        # Collect model info for UI display
        common.model_info["pipeline"] = "Qwen/Qwen-Image-Edit-2509"
        common.model_info["transformer"] = model_path
        common.model_info["dtype"] = str(dtype).replace("torch.", "")
        common.model_info["steps"] = str(steps)
        common.model_info["rank"] = str(rank)
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
        active_names = []
        for p, s in (active_configs if active_configs else []):
            for e in common.lora_registry:
                if e["path"] == p:
                    active_names.append(f"{e['name']}({s})")
                    break
        common.model_info["loras"] = ", ".join(active_names) if active_names else ""

        print("[info] pipeline ready", file=sys.stderr)
        return pipe


def apply_job_loras(lora_selection: list[dict]):
    """Reload pipeline if LoRA configuration changed since last run."""
    global _current_lora_key

    lora_configs = []
    selected_names = {s["name"]: s["scale"] for s in lora_selection}
    for entry in common.lora_registry:
        name = entry["name"]
        if name in selected_names:
            lora_configs.append((entry["path"], selected_names[name]))
        else:
            lora_configs.append((entry["path"], 0.0))

    new_key = tuple(
        (path, scale) for path, scale in lora_configs if abs(scale) > 1e-5
    )

    if new_key == _current_lora_key:
        return

    print(f"[info] LoRA config changed, reloading pipeline...", file=sys.stderr)
    print(f"[info]   old: {_current_lora_key}", file=sys.stderr)
    print(f"[info]   new: {new_key}", file=sys.stderr)

    try:
        load_pipeline(lora_configs=lora_configs, **_pipeline_args)
        _current_lora_key = new_key
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"[error] pipeline reload failed, restoring without LoRA...", file=sys.stderr)
        try:
            load_pipeline(lora_configs=[], **_pipeline_args)
            _current_lora_key = ()
            print(f"[info] recovered: pipeline loaded without LoRA", file=sys.stderr)
        except Exception:
            traceback.print_exc()
            print(f"[error] recovery also failed", file=sys.stderr)
        raise RuntimeError(f"LoRA reload failed: {ex}")


def run_inference(pipe, images: list, prompt: str, seed: int | None, job_id: str):
    import torch

    gen = None
    if seed is not None:
        gen = torch.Generator("cuda").manual_seed(seed)

    def on_step_end(pipeline, step_index, timestep, callback_kwargs):
        with common.job_lock:
            common.jobs[job_id]["current_step"] = step_index + 1
        if job_id in common.cancel_requests:
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

            # Apply LoRA selection (may reload pipeline)
            if common.lora_registry:
                apply_job_loras(job.get("loras", []))
                pipe = pipeline_ref["pipe"]

            if pipe is None:
                raise RuntimeError("Pipeline is not loaded / パイプラインが読み込まれていません")

            result_img = run_inference(pipe, image_list, job["prompt"], job["seed"], job_id)

            if job_id in common.cancel_requests:
                with common.job_lock:
                    common.cancel_requests.discard(job_id)
                    common.jobs[job_id]["status"] = "cancelled"
            else:
                out_path = common.TMP_DIR / f"{job_id}_out.png"
                result_img.save(out_path)
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
    server_title="Nunchaku",
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
    global _pipeline_args, _current_lora_key

    ap = argparse.ArgumentParser(description="Qwen Image Edit Web Server (Nunchaku)")
    common.add_common_args(ap)
    ap.add_argument("--no-progress", action="store_true", help="Hide HF download progress")
    ap.add_argument("--no-offload", action="store_true",
                    help="Disable offloading, keep all on GPU (high VRAM)")
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS,
                    help="Inference steps: 4 or 8 (default: 8)")
    ap.add_argument("--rank", type=int, default=RANK,
                    help="Nunchaku rank: 32 or 128 (default: 32)")
    ap.add_argument("--num-blocks-on-gpu", type=int, default=NUM_BLOCKS_ON_GPU,
                    help="Blocks to keep on GPU in low-VRAM mode (default: 1)")
    ap.add_argument("--lora", action="append", default=[], metavar="REPO_OR_PATH",
                    help="LoRA weights (repeatable). Format: path_or_repo or repo::weight_name")
    args = ap.parse_args()

    common.apply_common_args(args)

    # Parse and register LoRAs
    common.lora_registry.extend(parse_lora_args(args.lora))

    # Save pipeline args for future reloads (LoRA switching)
    _pipeline_args = dict(
        progress=not args.no_progress,
        no_offload=args.no_offload,
        rank=args.rank,
        num_blocks_on_gpu=args.num_blocks_on_gpu,
        steps=args.steps,
    )

    print("[info] loading model...", file=sys.stderr)
    load_pipeline(**_pipeline_args)

    # Initialize current LoRA key from what was loaded
    _current_lora_key = tuple(
        (e["path"], e["default_scale"]) for e in common.lora_registry
        if abs(e["default_scale"]) > 1e-5
    )

    common.start_server_threads(worker_loop)

    print(f"[info] server starting at http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[info] password: {args.password}", file=sys.stderr)

    common.app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
