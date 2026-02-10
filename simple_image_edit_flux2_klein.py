# simple_image_edit_flux2_klein.py
#
# Single-image editor using FLUX.2 Klein 4B.
# Model: black-forest-labs/FLUX.2-klein-4B
#
# Features:
# - Input: 1 image file path
# - Output: input stem + suffix + .png
# - Size cap: fit within 2048x2048 (keep aspect; only scales DOWN)
# - Align: width and height to multiple of 16 (by padding)
# - Optional pre-resize by total pixels (1m/2m)
# - Optional HF download progress bars (--progress)
# - Optional memory log (--mem-log, requires psutil)

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
GUIDANCE_SCALE = 1.0
NUM_INFERENCE_STEPS = 20

OUT_SUFFIX = "_filtered_klein"
OUT_EXT = ".png"

MAX_W = 2048
MAX_H = 2048
W_MULT = 16
H_MULT = 16

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

T2I_DEFAULT_SIZE = (1024, 1024)
T2I_OUTPUT_NAME = "t2i"
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


def preprocess_image(path, pre_resize_target, max_w, max_h, w_mult, h_mult, label="image"):
    img = open_image(Path(path))
    if pre_resize_target:
        img, _ = pre_resize_to_total_pixels(img, pre_resize_target)
    img, (w, h) = fit_and_align(img, max_w, max_h, w_mult, h_mult)
    eprint(f"[info] {label} final size: {w}x{h}")
    return img, (w, h)


def pick_dtype(torch):
    """Pick best dtype: bf16 if supported, otherwise fp16."""
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    return dtype, bf16_ok


def main():
    ap = argparse.ArgumentParser(description="FLUX.2 Klein 4B image editor (single image)")
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
    ap.add_argument("--no-offload", action="store_true", help="オフロード無効（高VRAM向け）")
    ap.add_argument("--offload", action="store_true",
                    help="sequential CPU offload（低VRAM向け、遅いがVRAM節約）")
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS,
                    help="推論ステップ数 (default: 4)")
    ap.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE,
                    help="ガイダンススケール (default: 1.0)")
    ap.add_argument("--prompt", default=None, help="プロンプト上書き")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（省略時はランダム）")
    ap.add_argument("--ref", action="append", default=[], metavar="FILE",
                    help="参照画像を追加（最大3回指定可能、合計4画像まで）")
    ap.add_argument("--lora", default=None, metavar="REPO_OR_PATH",
                    help="LoRA重みのHFリポジトリIDまたはローカルパス")
    ap.add_argument("--lora-weight-name", default=None, metavar="FILE",
                    help="HFリポジトリ内のLoRA重みファイル名")
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="LoRA適用強度 (default: 1.0)")
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

    if len(args.ref) > 3:
        die("エラー: --ref は最大3回まで指定できます（入力画像含め合計4画像）。")

    if args.no_offload and args.offload:
        die("エラー: --no-offload と --offload は同時に指定できません。")

    require_imports()
    set_hf_progress(args.progress)

    import torch
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        die(
            "エラー: Flux2KleinPipeline が見つかりません。\n"
            "diffusers のバージョンが古い可能性があります。\n"
            "  pip install -U diffusers"
        )

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
        ref_images = []
        eprint(f"[info] text-to-image mode: {out_w}x{out_h}")
    else:
        in_path = Path(args.input)
        img, (out_w, out_h) = preprocess_image(
            in_path, args.pre_resize, MAX_W, MAX_H, W_MULT, H_MULT, label="input")

        ref_images = []
        for i, ref_path in enumerate(args.ref):
            ref_img, _ = preprocess_image(
                ref_path, args.pre_resize, MAX_W, MAX_H, W_MULT, H_MULT, label=f"ref[{i+1}]")
            ref_images.append(ref_img)

    # Pick dtype (bf16 preferred, fp16 fallback)
    dtype, bf16_ok = pick_dtype(torch)
    eprint(f"[info] dtype={dtype} (bf16_supported={bf16_ok})")

    # Load pipeline
    eprint(f"[info] loading model: {MODEL_ID}")
    t0 = time.time()
    try:
        pipeline = Flux2KleinPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
    except Exception as ex:
        die(
            f"エラー: パイプラインのロードに失敗しました。\n"
            f"  model: {MODEL_ID}\n"
            f"  詳細: {ex}"
        )
    eprint(f"[info] pipeline loaded ({time.time()-t0:.1f}s)")
    mem("after load", torch)

    # LoRA loading (diffusers API)
    if args.lora:
        eprint(f"[info] loading LoRA: {args.lora}")
        try:
            lora_kwargs = {}
            if args.lora_weight_name:
                lora_kwargs["weight_name"] = args.lora_weight_name
            pipeline.load_lora_weights(args.lora, **lora_kwargs)
            if args.lora_scale != 1.0:
                pipeline.fuse_lora(lora_scale=args.lora_scale)
                eprint(f"[info] LoRA fused (scale={args.lora_scale})")
            eprint("[info] LoRA loaded")
        except Exception as ex:
            die(f"エラー: LoRAの読み込みに失敗しました。\n  詳細: {ex}")

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
    gen = None
    if args.seed is not None:
        gen = torch.Generator("cuda").manual_seed(args.seed)
        eprint(f"[info] seed={args.seed}")

    eprint("[info] starting inference...")
    t1 = time.time()

    if ref_images:
        image_arg = [img] + ref_images
        eprint(f"[info] image count: {len(image_arg)}")
    else:
        image_arg = img

    try:
        output = pipeline(
            image=image_arg,
            prompt=prompt,
            height=out_h,
            width=out_w,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=gen,
        )
    except torch.cuda.OutOfMemoryError:
        die(
            "エラー: CUDA OOM\n"
            "対策:\n"
            "  - --pre-resize 1m を使う\n"
            "  - --offload を使う（sequential CPU offload）\n"
            "  - --ref の数を減らす\n"
        )

    eprint(f"[info] inference done ({time.time()-t1:.1f}s)")

    # Save output
    out_img = output.images[0]
    out_path = in_path.with_name(in_path.stem + args.suffix + OUT_EXT)
    out_img.save(out_path)
    print(str(out_path))

    # Cleanup
    del pipeline, output
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
