# simple_i2v_ltx2_distilled.py
#
# Video generator using LTX-2 19B Distilled (diffusers version).
# Model: rootonchair/LTX-2-19b-distilled
#
# Modes:
# - i2v (default): Image-to-video (first frame conditioning)
# - flf2v (--ref): First+last frame to video
# - t2v (--t2i): Text-to-video (no input image)
#
# Features:
# - Two-stage pipeline: Stage1 (8 steps) -> latent upsample -> Stage2 (3 steps)
# - fp8 layerwise casting by default (--no-fp8 to disable)
# - Optional --no-stage2 to skip upsampling/refinement
# - Size cap: fit within 1280x720 (keep aspect; only scales DOWN)
# - Align: width and height to multiple of 32
# - Optional pre-resize by total pixels (0.3m/1m/2m)
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
PROMPT = "The scene comes to life with natural, fluid motion."
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
GUIDANCE_SCALE = 1.0
NUM_INFERENCE_STEPS_S1 = 8
NUM_INFERENCE_STEPS_S2 = 3

NUM_FRAMES = 121
FRAME_RATE = 24.0

OUT_SUFFIX = "_ltx2"
OUT_EXT = ".mp4"

MAX_W = 1280
MAX_H = 720
W_MULT = 32
H_MULT = 32

DEFAULT_W = 768
DEFAULT_H = 512

MODEL_ID = "rootonchair/LTX-2-19b-distilled"

T2V_DEFAULT_SIZE = (768, 512)
T2V_OUTPUT_NAME = "t2v"

LORA = None             # LoRA重み: HFリポジトリID or ローカルパス (None=無効)
LORA_WEIGHT_NAME = None # HFリポジトリ内のLoRA重みファイル名 (None=自動)
LORA_SCALE = 1.0        # LoRA適用強度
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
        raise argparse.ArgumentTypeError("--size は 'WxH' 形式で指定 (例: 768x512)")


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
    ap = argparse.ArgumentParser(description="LTX-2 19B Distilled video generator [diffusers] (i2v / flf2v / t2v)")
    ap.add_argument("input", nargs="?", default=None,
                    help="入力画像ファイル（--t2i 時は省略可）")
    ap.add_argument("--t2i", action="store_true",
                    help="text-to-videoモード（入力画像不要）")
    ap.add_argument("--ref", default=None, metavar="FILE",
                    help="最終フレーム画像（first+last frame to video）")
    ap.add_argument("--suffix", default=OUT_SUFFIX, help="出力suffix")
    ap.add_argument("--size", type=parse_size, default=None, metavar="WxH",
                    help=f"出力サイズ (default: 入力画像から自動 / t2v時 {DEFAULT_W}x{DEFAULT_H})")
    ap.add_argument("--pre-resize", type=parse_pre_resize, default=None, metavar="1m|2m|NUM",
                    help="総ピクセル数を縮小")
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES,
                    help=f"フレーム数 (default: {NUM_FRAMES})")
    ap.add_argument("--frame-rate", type=float, default=FRAME_RATE,
                    help=f"フレームレート (default: {FRAME_RATE})")
    ap.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS_S1,
                    help=f"Stage1推論ステップ数 (default: {NUM_INFERENCE_STEPS_S1})")
    ap.add_argument("--steps-s2", type=int, default=NUM_INFERENCE_STEPS_S2,
                    help=f"Stage2推論ステップ数 (default: {NUM_INFERENCE_STEPS_S2})")
    ap.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE,
                    help=f"ガイダンススケール (default: {GUIDANCE_SCALE})")
    ap.add_argument("--prompt", default=None, help="プロンプト上書き")
    ap.add_argument("--negative-prompt", default=None, help="ネガティブプロンプト上書き")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（省略時はランダム）")
    ap.add_argument("--no-stage2", action="store_true",
                    help="Stage2（latent upsample + refinement）をスキップ")
    ap.add_argument("--no-fp8", action="store_true",
                    help="fp8 layerwise castingを無効にする（bf16でロード）")
    ap.add_argument("--no-offload", action="store_true", help="オフロード無効（高VRAM向け）")
    ap.add_argument("--offload", action="store_true",
                    help="sequential CPU offload（低VRAM向け、遅いがVRAM節約）")
    ap.add_argument("--lora", default=LORA, metavar="REPO_OR_PATH",
                    help="LoRA重みのHFリポジトリIDまたはローカルパス")
    ap.add_argument("--lora-weight-name", default=LORA_WEIGHT_NAME, metavar="FILE",
                    help="HFリポジトリ内のLoRA重みファイル名")
    ap.add_argument("--lora-scale", type=float, default=LORA_SCALE,
                    help="LoRA適用強度 (default: 1.0)")
    ap.add_argument("--no-audio", action="store_true", help="音声なしで出力")
    ap.add_argument("--progress", action="store_true", help="HFダウンロード進捗を表示")
    ap.add_argument("--mem-log", action="store_true", help="メモリログ")
    args = ap.parse_args()

    # Validate arguments
    if args.t2i:
        if args.ref:
            die("エラー: --t2i モードでは --ref は使用できません。")
        if args.pre_resize:
            die("エラー: --t2i モードでは --pre-resize は使用できません。")
        if not args.prompt:
            die("エラー: --t2i モードでは --prompt の指定が必要です。")
    else:
        if args.input is None:
            die("エラー: 入力画像ファイルを指定してください（text-to-videoモードは --t2i を使用）。")

    if args.no_offload and args.offload:
        die("エラー: --no-offload と --offload は同時に指定できません。")

    require_imports()
    set_hf_progress(args.progress)

    import torch
    try:
        from diffusers import LTX2ConditionPipeline, AutoModel
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.pipelines.ltx2.export_utils import encode_video
    except ImportError as ex:
        die(
            f"エラー: LTX2ConditionPipeline が見つかりません。\n"
            "diffusers のバージョンが古い可能性があります。\n"
            "  pip install -U git+https://github.com/huggingface/diffusers\n"
            f"  詳細: {ex}"
        )

    if not torch.cuda.is_available():
        die("エラー: CUDA GPU が見つかりません。")

    mem = make_mem_logger(args.mem_log)
    mem("start", torch)

    # Determine mode and prepare images / conditions
    conditions = []

    if args.t2i:
        # Text-to-video: no conditions
        t2v_w, t2v_h = args.size or T2V_DEFAULT_SIZE
        t2v_w, t2v_h = round_up(t2v_w, W_MULT), round_up(t2v_h, H_MULT)
        if t2v_w > MAX_W or t2v_h > MAX_H:
            die(f"エラー: --size が最大サイズ {MAX_W}x{MAX_H} を超えています。")
        out_w, out_h = t2v_w, t2v_h
        in_path = Path(T2V_OUTPUT_NAME)
        eprint(f"[info] text-to-video mode: {out_w}x{out_h}")
    else:
        in_path = Path(args.input)

        if args.size:
            target_w, target_h = args.size
            target_w, target_h = round_up(target_w, W_MULT), round_up(target_h, H_MULT)
            if target_w > MAX_W or target_h > MAX_H:
                die(f"エラー: --size が最大サイズ {MAX_W}x{MAX_H} を超えています。")
            img = open_image(in_path)
            img = img.resize((target_w, target_h))
            out_w, out_h = target_w, target_h
            eprint(f"[info] resized to --size: {out_w}x{out_h}")
        else:
            img, (out_w, out_h) = preprocess_image(
                in_path, args.pre_resize, MAX_W, MAX_H, W_MULT, H_MULT, label="input")

        # First frame condition
        conditions.append(LTX2VideoCondition(frames=img, index=0, strength=1.0))
        eprint("[info] first frame condition set")

        # Last frame condition (--ref)
        if args.ref:
            if args.size:
                ref_img = open_image(Path(args.ref))
                ref_img = ref_img.resize((out_w, out_h))
                eprint(f"[info] ref resized to --size: {out_w}x{out_h}")
            else:
                ref_img, _ = preprocess_image(
                    args.ref, args.pre_resize, MAX_W, MAX_H, W_MULT, H_MULT, label="ref(last)")
                if ref_img.size != (out_w, out_h):
                    ref_img = ref_img.resize((out_w, out_h))
                    eprint(f"[info] ref resized to match input: {out_w}x{out_h}")

            conditions.append(LTX2VideoCondition(frames=ref_img, index=-1, strength=1.0))
            eprint("[info] last frame condition set (flf2v mode)")

    # Pick dtype (bf16 preferred)
    dtype, bf16_ok = pick_dtype(torch)
    eprint(f"[info] dtype={dtype} (bf16_supported={bf16_ok})")

    # Load pipeline with optional fp8 layerwise casting
    use_fp8 = not args.no_fp8
    eprint(f"[info] loading model: {MODEL_ID} (fp8={use_fp8})")
    t0 = time.time()
    try:
        if use_fp8:
            # Load transformer separately for layerwise casting
            transformer = AutoModel.from_pretrained(
                MODEL_ID, subfolder="transformer", torch_dtype=dtype)
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
            eprint("[info] fp8 layerwise casting enabled (float8_e4m3fn)")
            pipeline = LTX2ConditionPipeline.from_pretrained(
                MODEL_ID, transformer=transformer, torch_dtype=dtype)
        else:
            pipeline = LTX2ConditionPipeline.from_pretrained(
                MODEL_ID, torch_dtype=dtype)
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

    # Run inference - Stage 1
    prompt = args.prompt if args.prompt else PROMPT
    neg_prompt = args.negative_prompt if args.negative_prompt else NEGATIVE_PROMPT
    gen = None
    if args.seed is not None:
        gen = torch.Generator("cuda").manual_seed(args.seed)
        eprint(f"[info] seed={args.seed}")

    mode_str = "t2v" if args.t2i else ("flf2v" if args.ref else "i2v")
    eprint(f"[info] Stage 1 ({mode_str}): {args.steps} steps, {out_w}x{out_h}, "
           f"{args.num_frames} frames @ {args.frame_rate}fps")
    t1 = time.time()

    output_type_s1 = "latent" if not args.no_stage2 else "np"

    pipe_kwargs = dict(
        conditions=conditions if conditions else None,
        prompt=prompt,
        negative_prompt=neg_prompt,
        height=out_h,
        width=out_w,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        sigmas=DISTILLED_SIGMA_VALUES,
        generator=gen,
        output_type=output_type_s1,
        return_dict=False,
    )

    try:
        video_out, audio_out = pipeline(**pipe_kwargs)
    except torch.cuda.OutOfMemoryError:
        die(
            "エラー: CUDA OOM\n"
            "対策:\n"
            "  - --pre-resize 0.3m を使う\n"
            "  - --size 512x320 など小さいサイズを指定\n"
            "  - --num-frames を減らす (例: 61)\n"
            "  - --offload を使う（sequential CPU offload）\n"
            "  - --no-stage2 を使う\n"
        )

    eprint(f"[info] Stage 1 done ({time.time()-t1:.1f}s)")
    mem("after stage1", torch)

    # Stage 2: latent upsample + refinement
    if not args.no_stage2:
        eprint("[info] Stage 2: latent upsample + refinement...")
        t2 = time.time()

        try:
            from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        except ImportError as ex:
            die(
                f"エラー: LTX2LatentUpsamplePipeline が見つかりません。\n"
                f"  詳細: {ex}"
            )

        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            MODEL_ID, subfolder="latent_upsampler", torch_dtype=dtype)

        upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=pipeline.vae, latent_upsampler=latent_upsampler)
        upsample_pipe.enable_model_cpu_offload()

        upscaled_latent = upsample_pipe(
            latents=video_out,
            output_type="latent",
            return_dict=False,
        )[0]

        del upsample_pipe, latent_upsampler
        gc.collect()
        torch.cuda.empty_cache()
        mem("after upsample", torch)

        # Enable VAE tiling for Stage 2 (higher resolution)
        pipeline.vae.enable_tiling()

        eprint(f"[info] Stage 2 refinement: {args.steps_s2} steps")
        try:
            video_out, audio_out = pipeline(
                latents=upscaled_latent,
                audio_latents=audio_out,
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=args.steps_s2,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                guidance_scale=args.guidance_scale,
                output_type="np",
                return_dict=False,
            )
        except torch.cuda.OutOfMemoryError:
            die(
                "エラー: Stage 2 で CUDA OOM\n"
                "対策:\n"
                "  - --no-stage2 を使ってStage2をスキップ\n"
                "  - --offload を使う\n"
            )

        eprint(f"[info] Stage 2 done ({time.time()-t2:.1f}s)")
        mem("after stage2", torch)

    # Save output
    out_path = in_path.with_name(in_path.stem + args.suffix + OUT_EXT)

    audio_sr = None
    audio_data = None
    if not args.no_audio and audio_out is not None:
        try:
            audio_sr = pipeline.vocoder.config.output_sampling_rate
            audio_data = audio_out[0].float().cpu()
        except Exception:
            eprint("[warn] 音声データの取得に失敗しました。映像のみ保存します。")

    if args.no_audio:
        eprint("[info] --no-audio: 音声なしで保存")

    encode_video(
        video_out[0],
        fps=args.frame_rate,
        audio=audio_data,
        audio_sample_rate=audio_sr,
        output_path=str(out_path),
    )
    print(str(out_path))
    eprint(f"[info] total time: {time.time()-t0:.1f}s")

    # Cleanup
    del pipeline, video_out, audio_out
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
