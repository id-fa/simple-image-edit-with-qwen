"""
Shared utilities for all web server variants.

Provides:
- Global state management (job queue, locks, config)
- User identification (get_user_hash)
- Gallery reference resolution (resolve_gallery_ref)
- Cleanup loop (cleanup_loop)
- HTML template loading (load_html_template)
- Common argparse setup (add_common_args, apply_common_args)
- Preset parsing (parse_presets)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from flask import Flask, request, jsonify
from lib.gallery_db import GalleryDB, get_room_db, cleanup_old_room_dbs


# =======================
# Global state (shared by all servers)
# =======================
app = Flask(__name__)

job_lock = threading.Lock()
jobs: dict[str, dict] = {}
processing_queue: deque[str] = deque()
current_processing: str | None = None
cancel_requests: set[str] = set()

model_info: dict[str, str] = {}

server_password = "password"
server_password_hash = hashlib.sha256(b"password").hexdigest()
gallery_enabled = False
prompt_presets: list[dict[str, str]] = []
lora_registry: list[dict] = []
db_dir = "./db"

CLEANUP_INTERVAL_SEC = 300
MAX_AGE_SEC = 3600
MAX_QUEUE_WAITING = 2

SERVER_DIR = Path(__file__).parent.parent  # server/
TMP_DIR = SERVER_DIR / "tmp"


# =======================
# User identification
# =======================
def get_user_hash() -> str:
    """Generate a short hash from client IP (X-Forwarded-For preferred) + User-Agent."""
    ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not ip:
        ip = request.remote_addr or ""
    ua = request.headers.get("User-Agent", "")
    raw = f"{ip}:{ua}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


# =======================
# Gallery reference resolution
# =======================
def resolve_gallery_ref(ref_str: str, new_job_id: str, slot: int, room: str = "") -> str | None:
    """Parse 'job_id:input:0', 'job_id:result', or 'drawing:id' and copy file to new job's input slot."""
    parts = ref_str.split(":")
    if len(parts) < 2:
        return None

    # Handle drawing references
    if parts[0] == "drawing":
        drawing_id = parts[1]
        rdb = get_room_db(db_dir, room) if room else None
        d = rdb.get_drawing(drawing_id) if rdb else None
        if not d:
            return None
        src_path = d.get("path")
        if not src_path or not os.path.exists(src_path):
            return None
        dest_path = TMP_DIR / f"{new_job_id}_in{slot}.png"
        shutil.copy2(src_path, dest_path)
        return str(dest_path)

    src_job_id, ref_type = parts[0], parts[1]
    # Check in-memory jobs first, then DB
    src_path = None
    with job_lock:
        src_job = jobs.get(src_job_id)
        if src_job:
            if ref_type == "result":
                src_path = src_job.get("result_path")
            elif ref_type == "input" and len(parts) >= 3:
                idx = int(parts[2])
                paths = src_job.get("input_paths", [])
                src_path = paths[idx] if 0 <= idx < len(paths) else None
    if not src_path and room:
        rdb = get_room_db(db_dir, room)
        db_job = rdb.get_job(src_job_id)
        if db_job:
            if ref_type == "result":
                src_path = db_job.get("result_path")
            elif ref_type == "input" and len(parts) >= 3:
                idx = int(parts[2])
                paths = db_job.get("input_paths", [])
                src_path = paths[idx] if 0 <= idx < len(paths) else None
    if not src_path or not os.path.exists(src_path):
        return None
    ext = Path(src_path).suffix.lower() or ".png"
    dest_path = TMP_DIR / f"{new_job_id}_in{slot}{ext}"
    shutil.copy2(src_path, dest_path)
    return str(dest_path)


# =======================
# Cleanup loop
# =======================
def cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL_SEC)
        now = time.time()
        cutoff = now - MAX_AGE_SEC

        # Clean old in-memory job entries
        to_remove = []
        with job_lock:
            for jid, job in jobs.items():
                if job["created"] < cutoff:
                    to_remove.append(jid)

        for jid in to_remove:
            with job_lock:
                jobs.pop(jid, None)
                cancel_requests.discard(jid)

        # Clean old DB records (all room DBs)
        db_path = Path(db_dir)
        if db_path.exists():
            for db_file in db_path.glob("room_*.db"):
                try:
                    rdb = GalleryDB(db_file)
                    files = rdb.cleanup(MAX_AGE_SEC)
                    for p in files:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                except Exception:
                    pass
            # Remove stale room DB files (no activity for 24h)
            cleanup_old_room_dbs(db_dir)

        # Also clean orphaned files in tmp
        for f in TMP_DIR.iterdir():
            if f.name == ".gitkeep":
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass


# =======================
# HTML template loading
# =======================
def load_html_template() -> str:
    """Load shared app_template.html from the same directory."""
    template_path = Path(__file__).parent / "app_template.html"
    try:
        html = template_path.read_text(encoding="utf-8")
        print(f"[info] HTML template loaded: {template_path}", file=sys.stderr)
        return html
    except FileNotFoundError:
        print(f"[error] HTML template not found: {template_path}", file=sys.stderr)
        return (
            "<!DOCTYPE html><html><body>"
            "<h1>Error: app_template.html not found</h1>"
            "</body></html>"
        )


# =======================
# Argparse helpers
# =======================
def add_common_args(ap, *, description: str = "Web Server"):
    """Add common arguments shared by all server variants."""
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    ap.add_argument("--password", default="password", help="Generation password (default: password)")
    ap.add_argument("--gallery", action="store_true",
                    help="Enable gallery mode (show generation history)")
    ap.add_argument("--db-dir", default="./db",
                    help="Gallery database directory (default: ./db)")
    ap.add_argument("--preset", action="append", default=[], metavar='"label::prompt"',
                    help="Prompt preset button (repeatable). Format: label::prompt or just prompt")


def apply_common_args(args):
    """Apply parsed common arguments to global state."""
    global server_password, server_password_hash, gallery_enabled, db_dir, prompt_presets
    server_password = args.password
    server_password_hash = hashlib.sha256(args.password.encode()).hexdigest()
    gallery_enabled = args.gallery
    db_dir = args.db_dir

    # Parse presets
    for i, raw in enumerate(args.preset, 1):
        if "::" in raw:
            label, prompt_text = raw.split("::", 1)
        else:
            label, prompt_text = f"preset{i}", raw
        prompt_presets.append({"label": label.strip(), "prompt": prompt_text.strip()})

    TMP_DIR.mkdir(parents=True, exist_ok=True)


# =======================
# Worker helpers
# =======================
def persist_job_to_db(job_id: str, job: dict, result_path: str):
    """Write completed job to gallery DB (called from worker after success)."""
    if not gallery_enabled:
        return
    try:
        rdb = get_room_db(db_dir, job.get("room", ""))
        rdb.add_job(
            job_id=job_id,
            created=job["created"],
            prompt=job["prompt"],
            seed=job["seed"],
            t2i=job["t2i"],
            input_count=len(job["input_paths"]),
            user_hash=job["user_hash"],
            input_paths=job["input_paths"],
            result_path=result_path,
            original_prompt=job.get("original_prompt"),
            input_names=job.get("input_names"),
        )
    except Exception as ex:
        print(f"[warn] DB write failed: {ex}", file=sys.stderr)


def start_server_threads(worker_fn):
    """Start worker and cleanup daemon threads."""
    worker = threading.Thread(target=worker_fn, daemon=True)
    worker.start()
    cleaner = threading.Thread(target=cleanup_loop, daemon=True)
    cleaner.start()
