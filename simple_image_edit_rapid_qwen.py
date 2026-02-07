# simple_image_edit_rapid_qwen.py
#
# Single-image editor using Qwen-Image-Edit-Rapid-AIO-V23 (4-step accelerated).
# Model: prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23
# Base:  Qwen/Qwen-Image-Edit-2511
#
# Features:
# - Input: 1 image file path
# - Output: input stem + suffix + .png
# - Size cap: fit within 2048x2048 (keep aspect; only scales DOWN)
# - Align: height to multiple of 16, width to multiple of 8 (by padding)
# - Optional pre-resize by total pixels (1m/2m)
# - Optional HF download progress bars (--progress)
# - Optional memory log (--mem-log, requires psutil)
#
# No nunchaku dependency. Requires recent diffusers with QwenImageEditPlusPipeline.
#
# Install:
#   pip install -U pip pillow huggingface_hub psutil transformers accelerate safetensors
#   pip install -U diffusers
#
# Usage:
#   py .\simple_image_edit_rapid_qwen.py .\sample.png
#   py .\simple_image_edit_rapid_qwen.py .\sample.png --prompt "Enhance quality." --seed 42
#   py .\simple_image_edit_rapid_qwen.py .\sample.png --pre-resize 2m --progress

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path


# =======================
# Fixed parameters (EDIT HERE)
# =======================
PROMPT = "Correct the unnatural line breaks at the page boundaries. Remove text-like elements. Remove MPEG noise and low-resolution noise. Maintain identical camera angle, framing, and perspective. Maintain character's posture and body orientation. Make sure to leave character's facial features identical and costume."
TRUE_CFG_SCALE = 1.0
GUIDANCE_SCALE = 1.0
NEGATIVE_PROMPT = " "

NUM_INFERENCE_STEPS = 4

OUT_SUFFIX = "_filtered_rapid"
OUT_EXT = ".png"

MAX_W = 2048
MAX_H = 2048
W_MULT = 8
H_MULT = 16

TRANSFORMER_ID = "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23"
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
# =======================


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    eprint(msg)
    sys.exit(code)


def require_imports():
    missing = []
    for name in ["torch", "diffusers", "pillow", "huggingface_hub",
                 "accelerate", "transformers", "safetensors"]:
        try:
            if name == "pillow":
                from PIL import Image  # noqa
            else:
                __import__(name)
        except ModuleNotFoundError:
            missing.append(name)

    if missing:
        die(
            "エラー: 必要なパッケージが見つかりません: " + ", ".join(missing) + "\n"
            "例:\n"
            f"  pip install -U {' '.join(missing)}\n"
        )


def set_hf_progress(enabled: bool):
    try:
        from huggingface_hub.utils import enable_progress_bars, disable_progress_bars
        (enable_progress_bars if enabled else disable_progress_bars)()
    except Exception:
        pass


def make_mem_logger(enabled: bool):
    if not enabled:
        return lambda tag, torch=None: None

    try:
        import psutil
    except ModuleNotFoundError:
        eprint("[warn] --mem-log を使うには psutil が必要です")
        return lambda tag, torch=None: None

    proc = psutil.Process(os.getpid())

    def mem(tag: str, torch=None):
        rss = proc.memory_info().rss / (1024**3)
        vm = psutil.virtual_memory()
        eprint(f"[mem] {tag} | RSS={rss:.1f}GB, SYS={vm.used/1024**3:.1f}/{vm.total/1024**3:.1f}GB")
        if torch is not None and torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                eprint(f"[mem] {tag} | VRAM free={free/1024**3:.1f}GB / total={total/1024**3:.1f}GB")
            except Exception:
                pass

    return mem


def parse_pre_resize(s: str | None) -> int | None:
    if s is None:
        return None
    t = s.strip().lower()
    if t in ("none", "off", "0", ""):
        return None
    if t in ("1m", "1mp"):
        return 1_000_000
    if t in ("2m", "2mp"):
        return 2_000_000
    try:
        v = int(t)
        return v if v > 0 else None
    except Exception:
        raise argparse.ArgumentTypeError("pre-resize は '1m'/'2m'/正の整数/none を指定してください")


def open_image(path: Path):
    from PIL import Image
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError:
        die(f"エラー: 入力ファイルが見つかりません: {path}")
    except Exception as ex:
        die(f"エラー: 画像を開けませんでした: {path}\n詳細: {ex}")


def pre_resize_to_total_pixels(img, target_pixels: int):
    w, h = img.size
    if w * h <= target_pixels:
        return img, False

    scale = math.sqrt(target_pixels / (w * h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    eprint(f"[info] pre-resize: {w}x{h} -> {nw}x{nh}")
    return img.resize((nw, nh)), True


def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def fit_and_align(img, max_w: int, max_h: int, w_mult: int, h_mult: int):
    from PIL import Image

    w, h = img.size

    # Scale down to fit max bounds
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))
        eprint(f"[info] scaled to fit: {w}x{h}")

    # Pad to alignment
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
        eprint(f"[info] padded: {w}x{h} -> {aw}x{ah}")
        return canvas, (aw, ah)

    return img, (w, h)


def pick_dtype(torch):
    """Pick best dtype: bf16 if supported, otherwise fp16."""
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    return dtype, bf16_ok


def main():
    ap = argparse.ArgumentParser(description="Qwen-Image-Edit-Rapid-AIO-V23 (single image)")
    ap.add_argument("input", help="入力画像ファイル")
    ap.add_argument("--suffix", default=OUT_SUFFIX, help="出力suffix")
    ap.add_argument("--pre-resize", type=parse_pre_resize, default=None, metavar="1m|2m|NUM",
                    help="総ピクセル数を縮小")
    ap.add_argument("--progress", action="store_true", help="HFダウンロード進捗を表示")
    ap.add_argument("--mem-log", action="store_true", help="メモリログ")
    ap.add_argument("--no-offload", action="store_true", help="オフロード無効（高VRAM向け）")
    ap.add_argument("--offload", action="store_true",
                    help="sequential CPU offload（低VRAM向け、遅いがVRAM節約）")
    ap.add_argument("--layerwise-fp8", action="store_true",
                    help="transformerの重みをFP8で保持（VRAM約半減、品質低下の可能性あり）")
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS,
                    help="推論ステップ数 (default: 4)")
    ap.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE,
                    help="ガイダンススケール (default: 1.0)")
    ap.add_argument("--true-cfg-scale", type=float, default=TRUE_CFG_SCALE,
                    help="true CFGスケール (default: 1.0)")
    ap.add_argument("--prompt", default=None, help="プロンプト上書き")
    ap.add_argument("--negative-prompt", default=None, help="ネガティブプロンプト上書き")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（省略時はランダム）")
    args = ap.parse_args()

    if args.no_offload and args.offload:
        die("エラー: --no-offload と --offload は同時に指定できません。")

    require_imports()
    set_hf_progress(args.progress)

    import torch

    try:
        from diffusers.models import QwenImageTransformer2DModel
        from diffusers import QwenImageEditPlusPipeline
    except ImportError:
        die(
            "エラー: QwenImageEditPlusPipeline / QwenImageTransformer2DModel が見つかりません。\n"
            "diffusers のバージョンが古い可能性があります。\n"
            "  pip install -U diffusers"
        )

    if not torch.cuda.is_available():
        die("エラー: CUDA GPU が見つかりません。")

    mem = make_mem_logger(args.mem_log)
    mem("start", torch)

    # Load and preprocess image
    in_path = Path(args.input)
    img = open_image(in_path)

    if args.pre_resize:
        img, _ = pre_resize_to_total_pixels(img, args.pre_resize)

    img, (out_w, out_h) = fit_and_align(img, MAX_W, MAX_H, W_MULT, H_MULT)
    eprint(f"[info] final image size: {out_w}x{out_h}")

    # Pick dtype (bf16 preferred, fp16 fallback)
    dtype, bf16_ok = pick_dtype(torch)
    eprint(f"[info] dtype={dtype} (bf16_supported={bf16_ok})")

    # Load transformer from rapid model
    eprint(f"[info] loading transformer: {TRANSFORMER_ID}")
    t0 = time.time()
    try:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            TRANSFORMER_ID,
            torch_dtype=dtype,
        )
    except Exception as ex:
        die(
            f"エラー: transformerのロードに失敗しました。\n"
            f"  model: {TRANSFORMER_ID}\n"
            f"  詳細: {ex}"
        )
    eprint(f"[info] transformer loaded ({time.time()-t0:.1f}s)")

    # Layerwise FP8 casting: store weights in float8, compute in bf16/fp16
    if args.layerwise_fp8:
        try:
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=dtype,
            )
            eprint("[info] layerwise FP8 casting enabled")
        except Exception as ex:
            eprint(f"[warn] layerwise FP8 casting failed (diffusers が古い可能性): {ex}")

    # Load pipeline with base model + rapid transformer
    eprint(f"[info] loading pipeline: {BASE_MODEL_ID}")
    t1 = time.time()
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_ID,
            transformer=transformer,
            torch_dtype=dtype,
        )
    except Exception as ex:
        die(
            f"エラー: パイプラインのロードに失敗しました。\n"
            f"  base model: {BASE_MODEL_ID}\n"
            f"  詳細: {ex}"
        )
    eprint(f"[info] pipeline loaded ({time.time()-t1:.1f}s)")
    mem("after load", torch)

    # Offload strategy
    if args.no_offload:
        eprint("[info] --no-offload: pipeline.to('cuda')")
        pipeline.to("cuda")
    elif args.offload:
        eprint("[info] --offload: enable_sequential_cpu_offload()")
        pipeline.enable_sequential_cpu_offload()
    else:
        eprint("[info] enable_model_cpu_offload()")
        pipeline.enable_model_cpu_offload()

    mem("after offload setup", torch)

    # Run inference
    prompt = args.prompt if args.prompt else PROMPT
    negative_prompt = args.negative_prompt if args.negative_prompt else NEGATIVE_PROMPT
    gen = None
    if args.seed is not None:
        gen = torch.Generator("cuda").manual_seed(args.seed)
        eprint(f"[info] seed={args.seed}")

    eprint("[info] starting inference...")
    t2 = time.time()

    try:
        output = pipeline(
            image=[img],
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=args.guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
            num_inference_steps=args.steps,
            generator=gen,
        )
    except torch.cuda.OutOfMemoryError:
        die(
            "エラー: CUDA OOM\n"
            "対策:\n"
            "  - --pre-resize 1m を使う\n"
            "  - --offload を使う（sequential CPU offload、遅いがVRAM節約）\n"
            "  - --layerwise-fp8 を使う（重みFP8化でVRAM約半減）\n"
        )

    eprint(f"[info] inference done ({time.time()-t2:.1f}s)")

    # Save output
    out_img = output.images[0]
    out_path = in_path.with_name(in_path.stem + args.suffix + OUT_EXT)
    out_img.save(out_path)
    print(str(out_path))

    # Cleanup
    del pipeline, transformer, output
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
