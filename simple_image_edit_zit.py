# simple_image_edit_zit.py
#
# Z-Image Turbo img2img runner (single image)
# Default model: unsloth/Z-Image-Turbo-unsloth-bnb-4bit (4bit quantized)
#
# Features:
# - Input: 1 image file path
# - Output: input stem + suffix + .png
# - Size cap: fit within 2048x2048 (keep aspect; only scales DOWN)
# - Align: height to multiple of 16, width to multiple of 8 (by padding)
# - Optional pre-resize by total pixels (1m/2m)
# - Prefers bf16 over fp16 to avoid NaN/black output issues

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
PROMPT = "Fix visible seams and misalignment at image boundaries. Remove all overlaid text, subtitles, and credits. Remove compression artifacts and upscaling noise. Preserve the original composition, character identity, pose, and costume exactly."
NEGATIVE_PROMPT = ""

NUM_STEPS = 9
GUIDANCE_SCALE = 0.0
STRENGTH = 0.5

OUT_SUFFIX = "_filtered_zit"
OUT_EXT = ".png"

MAX_W = 2048
MAX_H = 2048
W_MULT = 8
H_MULT = 16

MODEL_DEFAULT = "unsloth/Z-Image-Turbo-unsloth-bnb-4bit"
MODEL_OFFICIAL = "Tongyi-MAI/Z-Image-Turbo"

T2I_DEFAULT_SIZE = (1024, 1024)
T2I_OUTPUT_NAME = "t2i"
# =======================


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    eprint(msg)
    sys.exit(code)


def require_imports(need_bitsandbytes: bool = True):
    missing = []
    for name in ["torch", "diffusers", "pillow", "huggingface_hub", "accelerate", "transformers", "safetensors"]:
        try:
            if name == "pillow":
                from PIL import Image  # noqa
            else:
                __import__(name)
        except ModuleNotFoundError:
            missing.append(name)

    if need_bitsandbytes:
        try:
            import bitsandbytes  # noqa
        except ModuleNotFoundError:
            missing.append("bitsandbytes")

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
    if t in ("0.3m", "0.3mp"):
        return 300_000
    if t in ("1m", "1mp"):
        return 1_000_000
    if t in ("2m", "2mp"):
        return 2_000_000
    try:
        v = int(t)
        return v if v > 0 else None
    except Exception:
        raise argparse.ArgumentTypeError("pre-resize は '0.3m'/'1m'/'2m'/正の整数/none を指定してください")


def parse_size(s: str) -> tuple[int, int]:
    """Parse 'WxH' size string."""
    try:
        parts = s.lower().split("x")
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError
        return (w, h)
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError("--size は 'WxH' 形式で指定 (例: 1024x1024)")


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
    # Prefer bf16 when supported (fixes NaN/black output in many cases)
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    return dtype, bf16_ok


def main():
    ap = argparse.ArgumentParser(description="Z-Image Turbo img2img (default: unsloth-4bit)")
    ap.add_argument("input", nargs="?", default=None,
                    help="入力画像ファイル（--t2i 時は省略可）")
    ap.add_argument("--t2i", action="store_true",
                    help="text-to-imageモード（入力画像不要、白画像から生成）")
    ap.add_argument("--size", type=parse_size, default=None, metavar="WxH",
                    help="t2iモード時の出力サイズ (default: 1024x1024)")
    ap.add_argument("--suffix", default=OUT_SUFFIX, help="出力suffix")
    ap.add_argument("--pre-resize", type=parse_pre_resize, default=None, metavar="1m|2m|NUM",
                    help="総ピクセル数を縮小")
    ap.add_argument("--progress", action="store_true", help="HFダウンロード進捗を表示")
    ap.add_argument("--mem-log", action="store_true", help="メモリログ")
    ap.add_argument("--offload", action="store_true", help="VRAM節約（遅いが安定）")
    ap.add_argument("--official", action="store_true",
                    help="公式モデル Tongyi-MAI/Z-Image-Turbo を使用（bitsandbytes不要）")
    ap.add_argument("--prompt", default=None, help="プロンプト上書き")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（省略時はランダム）")
    ap.add_argument("--ref", action="append", default=[], metavar="FILE",
                    help="参照画像（ZImageでは未対応、無視されます）")
    ap.add_argument("--lora", default=None, metavar="REPO_OR_PATH",
                    help="LoRA重み（ZImageでは未対応、無視されます）")
    ap.add_argument("--lora-weight-name", default=None, metavar="FILE",
                    help="LoRA重みファイル名（ZImageでは未対応、無視されます）")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="LoRA適用強度（ZImageでは未対応、無視されます）")
    args = ap.parse_args()

    if args.t2i:
        if args.ref:
            die("エラー: --t2i モードでは --ref は使用できません。")
        if args.pre_resize:
            die("エラー: --t2i モードでは --pre-resize は使用できません。")
        if not args.prompt:
            die("エラー: --t2i モードでは --prompt の指定が必要です。")
    else:
        if args.input is None:
            die("エラー: 入力画像ファイルを指定してください（text-to-imageモードは --t2i を使用）。")
        if args.size is not None:
            die("エラー: --size は --t2i モードでのみ使用できます。")

    if args.ref:
        eprint("[warn] --ref は ZImageImg2ImgPipeline では未対応のため無視されます。")
    if args.lora:
        eprint("[warn] --lora は ZImageImg2ImgPipeline では未対応のため無視されます。")

    # bitsandbytes is only needed for unsloth-4bit model
    require_imports(need_bitsandbytes=not args.official)
    set_hf_progress(args.progress)

    import torch

    if not torch.cuda.is_available():
        die("エラー: CUDA GPU が見つかりません。")

    mem = make_mem_logger(args.mem_log)
    mem("start", torch)

    # Load and preprocess image
    if args.t2i:
        from PIL import Image as _PILImage
        t2i_w, t2i_h = args.size or T2I_DEFAULT_SIZE
        t2i_w, t2i_h = round_up(t2i_w, W_MULT), round_up(t2i_h, H_MULT)
        if t2i_w > MAX_W or t2i_h > MAX_H:
            die(f"エラー: --size が最大サイズ {MAX_W}x{MAX_H} を超えています。")
        img = _PILImage.new("RGB", (t2i_w, t2i_h), (255, 255, 255))
        out_w, out_h = t2i_w, t2i_h
        in_path = Path(T2I_OUTPUT_NAME)
        eprint(f"[info] text-to-image mode: {out_w}x{out_h}")
    else:
        in_path = Path(args.input)
        img = open_image(in_path)

        if args.pre_resize:
            img, _ = pre_resize_to_total_pixels(img, args.pre_resize)

        img, (out_w, out_h) = fit_and_align(img, MAX_W, MAX_H, W_MULT, H_MULT)
        eprint(f"[info] final image size: {out_w}x{out_h}")

    # Pick dtype
    dtype, bf16_ok = pick_dtype(torch)
    eprint(f"[info] dtype={dtype} (bf16_supported={bf16_ok})")

    # Load pipeline
    model_id = MODEL_OFFICIAL if args.official else MODEL_DEFAULT

    try:
        from diffusers import ZImageImg2ImgPipeline
    except ImportError:
        die(
            "エラー: diffusers に ZImageImg2ImgPipeline が見つかりません。\n"
            "diffusers を更新してください:\n"
            "  pip install -U diffusers\n"
        )

    eprint(f"[info] loading pipeline: {model_id}")
    t0 = time.time()
    pipe = ZImageImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    eprint(f"[info] pipeline loaded ({time.time()-t0:.1f}s)")

    # Move to GPU
    pipe = pipe.to("cuda")

    # VRAM optimizations
    try:
        pipe.enable_attention_slicing("max")
    except Exception:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    if args.offload:
        eprint("[info] enable_model_cpu_offload()")
        pipe.enable_model_cpu_offload()

    mem("after pipeline ready", torch)

    # Inference
    prompt = args.prompt if args.prompt else PROMPT
    gen = None
    if args.seed is not None:
        gen = torch.Generator("cuda").manual_seed(args.seed)
        eprint(f"[info] seed={args.seed}")

    strength = 1.0 if args.t2i else STRENGTH

    eprint("[info] starting inference...")
    t1 = time.time()

    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=img,
            strength=strength,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            width=out_w,
            height=out_h,
            generator=gen,
        )
    except torch.cuda.OutOfMemoryError:
        die(
            "エラー: CUDA OOM\n"
            "対策:\n"
            "  - --pre-resize 1m を使う\n"
            "  - --offload を使う\n"
        )

    eprint(f"[info] inference done ({time.time()-t1:.1f}s)")

    # Save output
    out_img = out.images[0]
    out_path = in_path.with_name(in_path.stem + args.suffix + OUT_EXT)
    out_img.save(out_path)
    print(str(out_path))

    # Cleanup
    del pipe, out
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
