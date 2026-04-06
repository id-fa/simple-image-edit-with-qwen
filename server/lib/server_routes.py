"""
Shared Flask route handlers for all web server variants.

Usage:
    from server_routes import register_routes
    register_routes(
        server_title="AIO",
        pre_resize_options=[{"value": "0.3m", "label": "0.3M pixels"}, ...],
        pre_resize_map={"0.3m": 300_000, "1m": 1_000_000},
        default_pre_resize="0.3m",
        has_preview=False,
        get_total_steps=lambda: NUM_INFERENCE_STEPS,
        prompt_default=PROMPT_DEFAULT,
    )
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from pathlib import Path

from flask import request, jsonify, send_file, render_template_string

import lib.server_common as common


def register_routes(
    *,
    server_title: str,
    pre_resize_options: list[dict],
    pre_resize_map: dict[str, int],
    default_pre_resize: str,
    has_preview: bool = False,
    has_enhance: bool = False,
    enhance_fn: callable = None,
    get_total_steps: callable,
    prompt_default: str,
    html_template: str,
):
    """Register all shared routes on server_common.app."""
    app = common.app

    @app.route("/")
    def index():
        return render_template_string(
            html_template,
            gallery_enabled=common.gallery_enabled,
            prompt_presets=common.prompt_presets,
            server_title=server_title,
            pre_resize_options=pre_resize_options,
            has_preview=has_preview,
            has_enhance=has_enhance,
        )

    @app.route("/api/login", methods=["POST"])
    def login():
        """Verify plaintext password (POST body) and return hashed token for subsequent GET requests."""
        pw = request.form.get("password", "") or (request.json or {}).get("password", "")
        if pw != common.server_password:
            return jsonify({"error": "パスワードが正しくありません / Invalid password"}), 403
        return jsonify({"ok": True, "token": common.server_password_hash})

    @app.route("/api/submit", methods=["POST"])
    def submit():
        pw = request.form.get("password", "")
        if pw != common.server_password:
            return jsonify({"error": "パスワードが正しくありません / Invalid password"}), 403

        with common.job_lock:
            waiting_count = len(common.processing_queue)
            if waiting_count >= 1 + common.MAX_QUEUE_WAITING:
                return jsonify({"error": "サーバーがビジー状態です / Server is busy. Please try again later.",
                                "busy": True, "queue_size": waiting_count}), 503

        room = request.form.get("room", "")
        t2i = request.form.get("t2i") == "1"
        prompt = request.form.get("prompt", "").strip() or prompt_default
        original_prompt = request.form.get("original_prompt", "").strip() or None
        seed_str = request.form.get("seed", "").strip()
        seed = int(seed_str) if seed_str else None
        pre_resize_str = request.form.get("pre_resize", default_pre_resize)
        pre_resize = pre_resize_map.get(pre_resize_str, list(pre_resize_map.values())[0])

        job_id = uuid.uuid4().hex[:12]
        input_paths = []
        input_names = []

        if not t2i:
            f1 = request.files.get("image1")
            f2 = request.files.get("image2")
            g1 = request.form.get("gallery_image1", "").strip()
            g2 = request.form.get("gallery_image2", "").strip()

            if f1 and f1.filename:
                ext1 = Path(f1.filename).suffix.lower() or ".png"
                save1 = common.TMP_DIR / f"{job_id}_in0{ext1}"
                f1.save(save1)
                input_paths.append(str(save1))
                input_names.append(Path(f1.filename).stem)
            elif g1:
                resolved = common.resolve_gallery_ref(g1, job_id, 0, room)
                if not resolved:
                    return jsonify({"error": "ギャラリー画像1の参照が無効または期限切れです / Gallery image 1 reference is invalid or expired"}), 400
                input_paths.append(resolved)
            else:
                return jsonify({"error": "Image 1 を選択してください / Please select Image 1"}), 400

            if f2 and f2.filename:
                ext2 = Path(f2.filename).suffix.lower() or ".png"
                save2 = common.TMP_DIR / f"{job_id}_in1{ext2}"
                f2.save(save2)
                input_paths.append(str(save2))
                input_names.append(Path(f2.filename).stem)
            elif g2:
                resolved = common.resolve_gallery_ref(g2, job_id, 1, room)
                if resolved:
                    input_paths.append(resolved)
        else:
            if not prompt or prompt == prompt_default:
                return jsonify({"error": "t2iモードではプロンプトの入力が必要です / Prompt is required in t2i mode"}), 400

        # Parse LoRA selection
        lora_selection = []
        loras_raw = request.form.get("loras", "").strip()
        if loras_raw:
            try:
                lora_selection = json.loads(loras_raw)
            except Exception:
                pass

        user_hash = common.get_user_hash()

        total_steps = get_total_steps()
        with common.job_lock:
            common.jobs[job_id] = {
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
                "total_steps": total_steps,
                "user_hash": user_hash,
                "loras": lora_selection,
                "room": room,
                "original_prompt": original_prompt,
                "input_names": input_names or None,
            }
            common.processing_queue.append(job_id)
            queue_pos = len(common.processing_queue)

        return jsonify({"job_id": job_id, "queue_position": queue_pos,
                        "token": common.server_password_hash})

    @app.route("/api/status/<job_id>")
    def status(job_id):
        with common.job_lock:
            job = common.jobs.get(job_id)
            if not job:
                return jsonify({"error": "ジョブが見つかりません / Job not found"}), 404

            queue_pos = 0
            for i, qid in enumerate(common.processing_queue):
                if qid == job_id:
                    queue_pos = i + 1
                    break

            resp = {
                "status": job["status"],
                "queue_position": queue_pos,
                "queue_total": len(common.processing_queue),
                "error": job.get("error"),
                "current_step": job.get("current_step", 0),
                "total_steps": job.get("total_steps", get_total_steps()),
            }
            if has_preview:
                resp["has_preview"] = "preview" in job

            return jsonify(resp)

    if has_preview:
        @app.route("/api/preview/<job_id>")
        def preview(job_id):
            with common.job_lock:
                job = common.jobs.get(job_id)
                if not job or "preview" not in job:
                    return "", 204
                preview_data = job["preview"]
            content_type = "image/jpeg"
            if preview_data[:4] == b'\x89PNG':
                content_type = "image/png"
            return send_file(io.BytesIO(preview_data), mimetype=content_type)

    if has_enhance and enhance_fn:
        @app.route("/api/enhance", methods=["POST"])
        def enhance_prompt():
            prompt_text = request.form.get("prompt", "").strip()
            if not prompt_text:
                return jsonify({"error": "プロンプトが空です / Prompt is empty"}), 400
            image_bytes = None
            f = request.files.get("image")
            if f and f.filename:
                image_bytes = f.read()
            try:
                enhanced = enhance_fn(prompt_text, image_bytes)
                return jsonify({"enhanced": enhanced})
            except Exception as ex:
                return jsonify({"error": f"プロンプト拡張に失敗しました / Enhance failed: {ex}"}), 500

    @app.route("/api/cancel/<job_id>", methods=["POST"])
    def cancel(job_id):
        with common.job_lock:
            job = common.jobs.get(job_id)
            if not job:
                return jsonify({"error": "ジョブが見つかりません / Job not found"}), 404
            if job["status"] in ("done", "error", "cancelled"):
                return jsonify({"error": "このジョブは既に終了しています / This job has already finished"}), 400
            common.cancel_requests.add(job_id)
            if job["status"] == "queued":
                try:
                    common.processing_queue.remove(job_id)
                except ValueError:
                    pass
                job["status"] = "cancelled"
                common.cancel_requests.discard(job_id)

        return jsonify({"ok": True, "message": "キャンセルを要求しました / Cancel requested"})

    @app.route("/api/result/<job_id>")
    def result(job_id):
        if common.gallery_enabled:
            pw = request.args.get("password", "")
            if pw != common.server_password_hash:
                return jsonify({"error": "Unauthorized"}), 403
        rp = None
        input_names = None
        job_user_hash = None
        with common.job_lock:
            job = common.jobs.get(job_id)
            if job:
                if job["status"] != "done":
                    return jsonify({"error": "まだ処理中です / Still processing"}), 400
                rp = job["result_path"]
                input_names = job.get("input_names")
                job_user_hash = job.get("user_hash")
        if not rp and common.gallery_enabled:
            room = request.args.get("room", "")
            from lib.gallery_db import get_room_db
            rdb = get_room_db(common.db_dir, room)
            db_job = rdb.get_job(job_id)
            if db_job:
                rp = db_job.get("result_path")
                if not input_names and db_job.get("input_names"):
                    input_names = db_job["input_names"]
                    job_user_hash = db_job.get("user_hash")

        if not rp or not os.path.exists(rp):
            return jsonify({"error": "結果ファイルが見つかりません / Result file not found"}), 404
        dl_name = f"result_{job_id}.png"
        if input_names and job_user_hash == common.get_user_hash():
            dl_name = f"{input_names[0]}_result_{job_id}.png"
        return send_file(rp, mimetype="image/png", as_attachment=True,
                         download_name=dl_name)

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
        if not common.gallery_enabled:
            return jsonify({"error": "Gallery is disabled"}), 404
        pw = request.args.get("password", "")
        if pw != common.server_password_hash:
            return jsonify({"error": "Unauthorized"}), 403
        room = request.args.get("room", "")
        from lib.gallery_db import get_room_db
        rdb = get_room_db(common.db_dir, room)
        items = rdb.get_gallery()
        db_ids = {it["job_id"] for it in items}
        with common.job_lock:
            for jid, job in common.jobs.items():
                if jid in db_ids:
                    continue
                if job.get("room", "") != room:
                    continue
                if job["status"] == "done":
                    entry = {
                        "job_id": jid,
                        "created": job["created"],
                        "prompt": job.get("prompt", ""),
                        "seed": job.get("seed"),
                        "t2i": job.get("t2i", False),
                        "input_count": len(job.get("input_paths", [])),
                        "user_hash": job.get("user_hash", ""),
                        "deleted": False,
                    }
                    if job.get("original_prompt"):
                        entry["original_prompt"] = job["original_prompt"]
                    if job.get("input_names"):
                        entry["input_names"] = job["input_names"]
                    items.append(entry)
        items.sort(key=lambda x: x["created"], reverse=True)
        caller_hash = common.get_user_hash()
        # Only expose input_names to the owner
        for it in items:
            if it.get("input_names") and it.get("user_hash") != caller_hash:
                del it["input_names"]
        return jsonify({"items": items, "caller_hash": caller_hash})

    @app.route("/api/gallery/<job_id>", methods=["DELETE"])
    def gallery_delete(job_id):
        if not common.gallery_enabled:
            return jsonify({"error": "Gallery is disabled"}), 404
        pw = request.args.get("password", "")
        if pw != common.server_password_hash:
            return jsonify({"error": "Unauthorized"}), 403
        room = request.args.get("room", "")
        caller_hash = common.get_user_hash()
        with common.job_lock:
            job = common.jobs.get(job_id)
            if job:
                if job.get("user_hash", "") != caller_hash:
                    return jsonify({"error": "他のユーザーの履歴は削除できません / Cannot delete another user's entry"}), 403
                job["status"] = "hidden"
        from lib.gallery_db import get_room_db
        rdb = get_room_db(common.db_dir, room)
        rdb.hide_job(job_id, caller_hash)
        return jsonify({"ok": True})

    @app.route("/api/input/<job_id>/<int:index>")
    def serve_input(job_id, index):
        if common.gallery_enabled:
            pw = request.args.get("password", "")
            if pw != common.server_password_hash:
                return jsonify({"error": "Unauthorized"}), 403
        path = None
        with common.job_lock:
            job = common.jobs.get(job_id)
            if job:
                paths = job.get("input_paths", [])
                if 0 <= index < len(paths):
                    path = paths[index]
        if not path and common.gallery_enabled:
            room = request.args.get("room", "")
            from lib.gallery_db import get_room_db
            rdb = get_room_db(common.db_dir, room)
            db_job = rdb.get_job(job_id)
            if db_job:
                paths = db_job.get("input_paths", [])
                if 0 <= index < len(paths):
                    path = paths[index]
        if not path or not os.path.exists(path):
            return jsonify({"error": "File not found"}), 404
        ext = Path(path).suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "bmp": "image/bmp"}.get(ext, "image/png")
        return send_file(path, mimetype=mime)

    @app.route("/api/model_info")
    def get_model_info():
        return jsonify(common.model_info)

    @app.route("/api/loras")
    def get_loras():
        return jsonify([{"name": e["name"], "default_scale": e.get("default_scale", 1.0)}
                         for e in common.lora_registry])

    @app.route("/api/queue_info")
    def queue_info():
        with common.job_lock:
            return jsonify({
                "queue_size": len(common.processing_queue),
                "processing": common.current_processing is not None,
            })

    # =======================
    # Drawing API
    # =======================
    @app.route("/api/drawing/save", methods=["POST"])
    def save_drawing():
        if common.gallery_enabled:
            pw = request.json.get("password", "")
            if pw != common.server_password_hash:
                return jsonify({"error": "Unauthorized"}), 403

        import base64
        data = request.json
        img_data = data.get("image", "")
        draw_type = data.get("type", "composite")
        source_info = data.get("source", "")
        room = data.get("room", "")

        if not img_data:
            return jsonify({"error": "No image data"}), 400

        if "," in img_data:
            img_data = img_data.split(",", 1)[1]

        drawing_id = uuid.uuid4().hex[:12]
        path = common.TMP_DIR / f"draw_{drawing_id}.png"

        with open(path, "wb") as f:
            f.write(base64.b64decode(img_data))

        bg_path = None
        overlay_path = None
        if draw_type == "draft":
            bg_data = data.get("bg", "")
            overlay_data = data.get("overlay", "")
            if bg_data:
                if "," in bg_data:
                    bg_data = bg_data.split(",", 1)[1]
                bg_path = str(common.TMP_DIR / f"draw_{drawing_id}_bg.png")
                with open(bg_path, "wb") as f:
                    f.write(base64.b64decode(bg_data))
            if overlay_data:
                if "," in overlay_data:
                    overlay_data = overlay_data.split(",", 1)[1]
                overlay_path = str(common.TMP_DIR / f"draw_{drawing_id}_ov.png")
                with open(overlay_path, "wb") as f:
                    f.write(base64.b64decode(overlay_data))

        user_hash = common.get_user_hash()
        from lib.gallery_db import get_room_db
        rdb = get_room_db(common.db_dir, room)
        rdb.add_drawing(
            drawing_id=drawing_id,
            user_hash=user_hash,
            created=time.time(),
            draw_type=draw_type,
            source=source_info,
            path=str(path),
            bg_path=bg_path,
            overlay_path=overlay_path,
        )

        return jsonify({"drawing_id": drawing_id})

    @app.route("/api/drawing/<drawing_id>")
    def serve_drawing(drawing_id):
        if common.gallery_enabled:
            pw = request.args.get("password", "")
            if pw != common.server_password_hash:
                return jsonify({"error": "Unauthorized"}), 403
        room = request.args.get("room", "")
        from lib.gallery_db import get_room_db
        rdb = get_room_db(common.db_dir, room)
        d = rdb.get_drawing(drawing_id)
        if not d:
            return jsonify({"error": "Drawing not found"}), 404
        if d["user_hash"] != common.get_user_hash():
            return jsonify({"error": "Unauthorized"}), 403
        if not os.path.exists(d["path"]):
            return jsonify({"error": "File not found"}), 404
        return send_file(d["path"], mimetype="image/png")

    @app.route("/api/drawing/<drawing_id>/bg")
    def serve_drawing_bg(drawing_id):
        if common.gallery_enabled:
            pw = request.args.get("password", "")
            if pw != common.server_password_hash:
                return jsonify({"error": "Unauthorized"}), 403
        room = request.args.get("room", "")
        from lib.gallery_db import get_room_db
        rdb = get_room_db(common.db_dir, room)
        d = rdb.get_drawing(drawing_id)
        if not d:
            return jsonify({"error": "Drawing not found"}), 404
        if d["user_hash"] != common.get_user_hash():
            return jsonify({"error": "Unauthorized"}), 403
        bg = d.get("bg_path")
        if not bg or not os.path.exists(bg):
            return jsonify({"error": "BG not found"}), 404
        return send_file(bg, mimetype="image/png")

    @app.route("/api/drawing/<drawing_id>/overlay")
    def serve_drawing_overlay(drawing_id):
        if common.gallery_enabled:
            pw = request.args.get("password", "")
            if pw != common.server_password_hash:
                return jsonify({"error": "Unauthorized"}), 403
        room = request.args.get("room", "")
        from lib.gallery_db import get_room_db
        rdb = get_room_db(common.db_dir, room)
        d = rdb.get_drawing(drawing_id)
        if not d:
            return jsonify({"error": "Drawing not found"}), 404
        if d["user_hash"] != common.get_user_hash():
            return jsonify({"error": "Unauthorized"}), 403
        ov = d.get("overlay_path")
        if not ov or not os.path.exists(ov):
            return jsonify({"error": "Overlay not found"}), 404
        return send_file(ov, mimetype="image/png")

    @app.route("/api/drawings")
    def list_drawings():
        if common.gallery_enabled:
            pw = request.args.get("password", "")
            if pw != common.server_password_hash:
                return jsonify({"error": "Unauthorized"}), 403
        room = request.args.get("room", "")
        user_hash = common.get_user_hash()
        from lib.gallery_db import get_room_db
        rdb = get_room_db(common.db_dir, room)
        items = rdb.get_drawings(user_hash)
        return jsonify({"items": items})

    @app.route("/api/drawing/<drawing_id>", methods=["DELETE"])
    def delete_drawing(drawing_id):
        if common.gallery_enabled:
            pw = request.args.get("password", "")
            if pw != common.server_password_hash:
                return jsonify({"error": "Unauthorized"}), 403
        room = request.args.get("room", "")
        user_hash = common.get_user_hash()
        from lib.gallery_db import get_room_db
        rdb = get_room_db(common.db_dir, room)
        paths_to_delete = rdb.delete_drawing(drawing_id, user_hash)
        if not paths_to_delete:
            return jsonify({"error": "Drawing not found or unauthorized"}), 404
        for p in paths_to_delete:
            try:
                os.remove(p)
            except OSError:
                pass
        return jsonify({"ok": True})
