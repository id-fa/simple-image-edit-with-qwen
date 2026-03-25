# simple_i2v_ltx2_native.py
#
# Video generator using LTX-2 19B Distilled (native ltx-pipelines version).
# Checkpoint: ltx-2-19b-distilled-fp8.safetensors (direct fp8)
#
# Modes:
# - i2v (default): Image-to-video (first frame conditioning)
# - flf2v (--ref): First+last frame to video
# - t2v (--t2i): Text-to-video (no input image)
#
# Features:
# - Two-stage distilled pipeline (8 steps + upsample + 3 steps)
# - Native fp8 checkpoint (no on-the-fly quantization overhead)
# - Local model files (no HF from_pretrained)
# - Optional --no-audio to skip audio output
#
# Requirements:
#   pip install ltx-core ltx-pipelines pillow
#
# Model files (download from https://huggingface.co/Lightricks/LTX-2):
#   - ltx-2-19b-distilled-fp8.safetensors (or ltx-2-19b-distilled.safetensors)
#   - ltx-2-spatial-upscaler-x2-1.0.safetensors
#   - Gemma 3 text encoder directory
#
# Download example
#    mkdir -p /tmp/models/{checkpoints,latent_upscale_models,loras,text_encoders}
#    hf auth login
#    hf download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir /tmp/models/checkpoints
#    hf download Lightricks/LTX-2 ltx-2-spatial-upscaler-x2-1.0.safetensors --local-dir /tmp/models/latent_upscale_models
#    hf download google/gemma-3-12b-it --local-dir /tmp/models/text_encoders
#

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path


# =======================
# Fixed parameters (EDIT HERE)
# =======================
PROMPT = "The scene comes to life with natural, fluid motion."
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

NUM_FRAMES = 97
FRAME_RATE = 25.0

OUT_SUFFIX = "_ltx2n"
OUT_EXT = ".mp4"

MAX_W = 1280
MAX_H = 768
W_MULT = 64
H_MULT = 64

DEFAULT_W = 768
DEFAULT_H = 512

# Default model paths (override via CLI args)
DEFAULT_CHECKPOINT = None       # e.g. "models/ltx-2-19b-distilled-fp8.safetensors"
DEFAULT_UPSAMPLER = None        # e.g. "models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_GEMMA_ROOT = None       # e.g. "models/gemma-3-4b-it"

T2V_OUTPUT_NAME = "t2v"
# =======================


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    eprint(msg)
    sys.exit(code)


def require_imports():
    missing = []
    for name, import_name in [("ltx-core", "ltx_core"), ("ltx-pipelines", "ltx_pipelines"),
                               ("pillow", "PIL"), ("torch", "torch")]:
        try:
            __import__(import_name)
        except ModuleNotFoundError:
            missing.append(name)

    if missing:
        die(
            "エラー: 必要なパッケージが見つかりません: " + ", ".join(missing) + "\n"
            "例:\n"
            "  pip install ltx-core ltx-pipelines pillow\n"
        )


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
        raise argparse.ArgumentTypeError("--size は 'WxH' 形式で指定 (例: 768x512)")


def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def fit_to_bounds(w: int, h: int, max_w: int, max_h: int, mult: int) -> tuple[int, int]:
    """Scale down to fit within max bounds and align to multiples."""
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        w, h = int(w * scale), int(h * scale)

    # Round down to alignment (not up, to stay within bounds)
    w = (w // mult) * mult
    h = (h // mult) * mult
    return max(mult, w), max(mult, h)


def main():
    ap = argparse.ArgumentParser(
        description="LTX-2 19B Distilled video generator [native] (i2v / flf2v / t2v)")
    ap.add_argument("input", nargs="?", default=None,
                    help="入力画像ファイル（--t2i 時は省略可）")
    ap.add_argument("--t2i", action="store_true",
                    help="text-to-videoモード（入力画像不要）")
    ap.add_argument("--ref", default=None, metavar="FILE",
                    help="最終フレーム画像（first+last frame to video）")

    # Model paths
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, metavar="PATH",
                    help="distilledチェックポイント (.safetensors)")
    ap.add_argument("--upsampler", default=DEFAULT_UPSAMPLER, metavar="PATH",
                    help="spatial upsampler (.safetensors)")
    ap.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT, metavar="DIR",
                    help="Gemma 3 テキストエンコーダのディレクトリ")

    ap.add_argument("--suffix", default=OUT_SUFFIX, help="出力suffix")
    ap.add_argument("--size", type=parse_size, default=None, metavar="WxH",
                    help=f"出力サイズ (default: 入力画像から自動 / t2v時 {DEFAULT_W}x{DEFAULT_H})")
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES,
                    help=f"フレーム数 (default: {NUM_FRAMES})")
    ap.add_argument("--frame-rate", type=float, default=FRAME_RATE,
                    help=f"フレームレート (default: {FRAME_RATE})")
    ap.add_argument("--prompt", default=None, help="プロンプト上書き")
    ap.add_argument("--seed", type=int, default=42, help="乱数シード (default: 42)")
    ap.add_argument("--no-fp8", action="store_true",
                    help="fp8量子化を無効にする")
    ap.add_argument("--no-audio", action="store_true", help="音声なしで出力")
    ap.add_argument("--enhance-prompt", action="store_true",
                    help="Gemma によるプロンプト拡張を有効にする")
    ap.add_argument("--mem-log", action="store_true", help="メモリログ")
    args = ap.parse_args()

    # Validate arguments
    if args.t2i:
        if args.ref:
            die("エラー: --t2i モードでは --ref は使用できません。")
        if not args.prompt:
            die("エラー: --t2i モードでは --prompt の指定が必要です。")
    else:
        if args.input is None:
            die("エラー: 入力画像ファイルを指定してください（text-to-videoモードは --t2i を使用）。")

    # Validate model paths
    if not args.checkpoint:
        die("エラー: --checkpoint でチェックポイントファイルを指定してください。\n"
            "  例: --checkpoint models/ltx-2-19b-distilled-fp8.safetensors\n"
            "  ダウンロード: https://huggingface.co/Lightricks/LTX-2")
    if not args.upsampler:
        die("エラー: --upsampler で spatial upsampler を指定してください。\n"
            "  例: --upsampler models/ltx-2-spatial-upscaler-x2-1.0.safetensors")
    if not args.gemma_root:
        die("エラー: --gemma-root で Gemma 3 テキストエンコーダのパスを指定してください。\n"
            "  例: --gemma-root models/gemma-3-4b-it")

    for label, path in [("checkpoint", args.checkpoint), ("upsampler", args.upsampler),
                        ("gemma-root", args.gemma_root)]:
        if not Path(path).exists():
            die(f"エラー: {label} が見つかりません: {path}")

    require_imports()

    import torch
    from ltx_pipelines import DistilledPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_core.quantization import QuantizationPolicy

    if not torch.cuda.is_available():
        die("エラー: CUDA GPU が見つかりません。")

    mem = make_mem_logger(args.mem_log)
    mem("start", torch)

    # Determine output size
    if args.t2i:
        out_w, out_h = args.size or (DEFAULT_W, DEFAULT_H)
        out_w, out_h = fit_to_bounds(out_w, out_h, MAX_W, MAX_H, W_MULT)
        in_path = Path(T2V_OUTPUT_NAME)
        eprint(f"[info] text-to-video mode: {out_w}x{out_h}")
    else:
        in_path = Path(args.input)
        if args.size:
            out_w, out_h = args.size
        else:
            from PIL import Image
            with Image.open(in_path) as im:
                out_w, out_h = im.size
        out_w, out_h = fit_to_bounds(out_w, out_h, MAX_W, MAX_H, W_MULT)
        eprint(f"[info] output size: {out_w}x{out_h}")

    # Build image conditions
    images = []
    if not args.t2i:
        images.append(ImageConditioningInput(
            path=str(in_path), frame_idx=0, strength=1.0, crf=33))
        eprint(f"[info] first frame: {in_path}")

        if args.ref:
            ref_path = Path(args.ref)
            if not ref_path.exists():
                die(f"エラー: ref ファイルが見つかりません: {ref_path}")
            # Use last frame index (num_frames - 1)
            last_idx = args.num_frames - 1
            images.append(ImageConditioningInput(
                path=str(ref_path), frame_idx=last_idx, strength=1.0, crf=33))
            eprint(f"[info] last frame (idx={last_idx}): {ref_path}")

    # Quantization
    quant = None
    if not args.no_fp8:
        quant = QuantizationPolicy.fp8_cast()
        eprint("[info] fp8_cast quantization enabled")

    # Load pipeline
    eprint(f"[info] loading pipeline...")
    eprint(f"  checkpoint: {args.checkpoint}")
    eprint(f"  upsampler:  {args.upsampler}")
    eprint(f"  gemma:      {args.gemma_root}")
    t0 = time.time()

    try:
        pipeline = DistilledPipeline(
            distilled_checkpoint_path=args.checkpoint,
            gemma_root=args.gemma_root,
            spatial_upsampler_path=args.upsampler,
            loras=[],
            quantization=quant,
        )
    except Exception as ex:
        die(f"エラー: パイプラインのロードに失敗しました。\n  詳細: {ex}")

    eprint(f"[info] pipeline loaded ({time.time()-t0:.1f}s)")
    mem("after load", torch)

    # Run inference
    prompt = args.prompt if args.prompt else PROMPT
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    mode_str = "t2v" if args.t2i else ("flf2v" if args.ref else "i2v")
    eprint(f"[info] generating ({mode_str}): {out_w}x{out_h}, "
           f"{args.num_frames} frames @ {args.frame_rate}fps, seed={args.seed}")
    t1 = time.time()

    try:
        video, audio = pipeline(
            prompt=prompt,
            seed=args.seed,
            height=out_h,
            width=out_w,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=args.enhance_prompt,
        )
    except torch.cuda.OutOfMemoryError:
        die(
            "エラー: CUDA OOM\n"
            "対策:\n"
            "  - --size 512x320 など小さいサイズを指定\n"
            "  - --num-frames を減らす (例: 49)\n"
            "  - fp8チェックポイントを使用する\n"
        )

    eprint(f"[info] generation done ({time.time()-t1:.1f}s)")
    mem("after generation", torch)

    # Save output
    out_path = in_path.with_name(in_path.stem + args.suffix + OUT_EXT)

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=None if args.no_audio else audio,
        output_path=str(out_path),
        video_chunks_number=video_chunks_number,
    )
    print(str(out_path))
    eprint(f"[info] total time: {time.time()-t0:.1f}s")

    if args.no_audio:
        eprint("[info] --no-audio: 音声なしで保存")


if __name__ == "__main__":
    main()
