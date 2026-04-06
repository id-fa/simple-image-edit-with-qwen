"""
gallery_db.py — SQLite-based gallery persistence module.

Provides room-based gallery storage for all web server variants.
Each room gets its own SQLite database file (WAL mode).

Usage:
    from gallery_db import get_room_db, cleanup_old_rooms

    db = get_room_db(db_dir="./db", room_name="room1")
    db.add_job(job_id, created, prompt, seed, t2i, input_count,
               user_hash, input_paths, result_path)
    items = db.get_gallery()
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path


def room_hash(room_name: str) -> str:
    """SHA-256 hash of room name, truncated to 16 hex chars."""
    return hashlib.sha256(room_name.encode()).hexdigest()[:16]


def get_room_db_path(db_dir: str | Path, room_name: str) -> Path:
    """Return the DB file path for a given room."""
    d = Path(db_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"room_{room_hash(room_name)}.db"


class GalleryDB:
    """Thread-safe SQLite gallery database (one connection per call via context manager)."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        conn = self._conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS gallery_jobs (
                    job_id       TEXT PRIMARY KEY,
                    created      REAL NOT NULL,
                    prompt       TEXT,
                    seed         INTEGER,
                    t2i          INTEGER NOT NULL DEFAULT 0,
                    input_count  INTEGER NOT NULL DEFAULT 0,
                    user_hash    TEXT,
                    status       TEXT NOT NULL DEFAULT 'done',
                    input_paths  TEXT,
                    result_path  TEXT,
                    original_prompt TEXT,
                    input_names  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON gallery_jobs(created);

                CREATE TABLE IF NOT EXISTS drawings (
                    drawing_id    TEXT PRIMARY KEY,
                    user_hash     TEXT NOT NULL,
                    created       REAL NOT NULL,
                    type          TEXT NOT NULL,
                    source        TEXT,
                    path          TEXT NOT NULL,
                    bg_path       TEXT,
                    overlay_path  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_drawings_user ON drawings(user_hash);
                CREATE INDEX IF NOT EXISTS idx_drawings_created ON drawings(created);
            """)
            conn.commit()
            # Migrate: add original_prompt column if missing (existing DBs)
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(gallery_jobs)").fetchall()}
            if "original_prompt" not in cols:
                conn.execute("ALTER TABLE gallery_jobs ADD COLUMN original_prompt TEXT")
                conn.commit()
            if "input_names" not in cols:
                conn.execute("ALTER TABLE gallery_jobs ADD COLUMN input_names TEXT")
                conn.commit()
        finally:
            conn.close()

    # ---- Job operations ----

    def add_job(self, job_id: str, created: float, prompt: str,
                seed: int | None, t2i: bool, input_count: int,
                user_hash: str, input_paths: list[str],
                result_path: str, original_prompt: str | None = None,
                input_names: list[str] | None = None):
        """Insert a completed job into the gallery."""
        conn = self._conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO gallery_jobs
                   (job_id, created, prompt, seed, t2i, input_count,
                    user_hash, status, input_paths, result_path, original_prompt,
                    input_names)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'done', ?, ?, ?, ?)""",
                (job_id, created, prompt, seed, 1 if t2i else 0,
                 input_count, user_hash,
                 json.dumps(input_paths), result_path, original_prompt,
                 json.dumps(input_names) if input_names else None)
            )
            conn.commit()
        finally:
            conn.close()

    def get_gallery(self) -> list[dict]:
        """Return all gallery items (done + hidden), newest first."""
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT job_id, created, prompt, seed, t2i, input_count,
                          user_hash, status, original_prompt, input_names
                   FROM gallery_jobs
                   ORDER BY created DESC"""
            ).fetchall()
            items = []
            for r in rows:
                if r["status"] == "hidden":
                    items.append({
                        "job_id": r["job_id"],
                        "created": r["created"],
                        "user_hash": r["user_hash"],
                        "deleted": True,
                    })
                else:
                    item = {
                        "job_id": r["job_id"],
                        "created": r["created"],
                        "prompt": r["prompt"],
                        "seed": r["seed"],
                        "t2i": bool(r["t2i"]),
                        "input_count": r["input_count"],
                        "user_hash": r["user_hash"],
                        "deleted": False,
                    }
                    if r["original_prompt"]:
                        item["original_prompt"] = r["original_prompt"]
                    if r["input_names"]:
                        try:
                            item["input_names"] = json.loads(r["input_names"])
                        except Exception:
                            pass
                    items.append(item)
            return items
        finally:
            conn.close()

    def get_job(self, job_id: str) -> dict | None:
        """Return a single job record or None."""
        conn = self._conn()
        try:
            row = conn.execute(
                """SELECT job_id, created, prompt, seed, t2i, input_count,
                          user_hash, status, input_paths, result_path,
                          input_names
                   FROM gallery_jobs WHERE job_id = ?""",
                (job_id,)
            ).fetchone()
            if not row:
                return None
            job = {
                "job_id": row["job_id"],
                "created": row["created"],
                "prompt": row["prompt"],
                "seed": row["seed"],
                "t2i": bool(row["t2i"]),
                "input_count": row["input_count"],
                "user_hash": row["user_hash"],
                "status": row["status"],
                "input_paths": json.loads(row["input_paths"]) if row["input_paths"] else [],
                "result_path": row["result_path"],
            }
            if row["input_names"]:
                try:
                    job["input_names"] = json.loads(row["input_names"])
                except Exception:
                    pass
            return job
        finally:
            conn.close()

    def hide_job(self, job_id: str, user_hash: str) -> bool:
        """Soft-delete a gallery item (set status=hidden). Returns True if successful."""
        conn = self._conn()
        try:
            cur = conn.execute(
                """UPDATE gallery_jobs SET status = 'hidden'
                   WHERE job_id = ? AND user_hash = ? AND status = 'done'""",
                (job_id, user_hash)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # ---- Drawing operations ----

    def add_drawing(self, drawing_id: str, user_hash: str, created: float,
                    draw_type: str, source: str, path: str,
                    bg_path: str | None = None,
                    overlay_path: str | None = None):
        """Insert a drawing record."""
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO drawings
                   (drawing_id, user_hash, created, type, source, path, bg_path, overlay_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (drawing_id, user_hash, created, draw_type, source,
                 path, bg_path, overlay_path)
            )
            conn.commit()
        finally:
            conn.close()

    def get_drawings(self, user_hash: str) -> list[dict]:
        """Return all drawings for a user, newest first."""
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT drawing_id, created, type, source
                   FROM drawings WHERE user_hash = ?
                   ORDER BY created DESC""",
                (user_hash,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_drawing(self, drawing_id: str) -> dict | None:
        """Return a single drawing record or None."""
        conn = self._conn()
        try:
            row = conn.execute(
                """SELECT drawing_id, user_hash, created, type, source,
                          path, bg_path, overlay_path
                   FROM drawings WHERE drawing_id = ?""",
                (drawing_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def delete_drawing(self, drawing_id: str, user_hash: str) -> list[str]:
        """Delete a drawing. Returns list of file paths to clean up."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT path, bg_path, overlay_path FROM drawings WHERE drawing_id = ? AND user_hash = ?",
                (drawing_id, user_hash)
            ).fetchone()
            if not row:
                return []
            paths = [p for p in [row["path"], row["bg_path"], row["overlay_path"]] if p]
            conn.execute("DELETE FROM drawings WHERE drawing_id = ?", (drawing_id,))
            conn.commit()
            return paths
        finally:
            conn.close()

    # ---- Cleanup ----

    def cleanup(self, max_age_sec: float) -> list[str]:
        """Remove old records and return file paths to delete from disk."""
        cutoff = time.time() - max_age_sec
        files_to_delete: list[str] = []
        conn = self._conn()
        try:
            # Collect file paths from old jobs
            rows = conn.execute(
                "SELECT input_paths, result_path FROM gallery_jobs WHERE created < ?",
                (cutoff,)
            ).fetchall()
            for r in rows:
                if r["result_path"]:
                    files_to_delete.append(r["result_path"])
                if r["input_paths"]:
                    for p in json.loads(r["input_paths"]):
                        files_to_delete.append(p)

            # Collect file paths from old drawings
            rows = conn.execute(
                "SELECT path, bg_path, overlay_path FROM drawings WHERE created < ?",
                (cutoff,)
            ).fetchall()
            for r in rows:
                for col in ["path", "bg_path", "overlay_path"]:
                    if r[col]:
                        files_to_delete.append(r[col])

            # Delete old records
            conn.execute("DELETE FROM gallery_jobs WHERE created < ?", (cutoff,))
            conn.execute("DELETE FROM drawings WHERE created < ?", (cutoff,))
            conn.commit()

            return files_to_delete
        finally:
            conn.close()


# ---- Room DB cache ----

_db_cache: dict[str, GalleryDB] = {}


def get_room_db(db_dir: str | Path, room_name: str) -> GalleryDB:
    """Get or create a GalleryDB for a room (cached)."""
    db_path = get_room_db_path(db_dir, room_name)
    key = str(db_path)
    if key not in _db_cache:
        _db_cache[key] = GalleryDB(db_path)
    return _db_cache[key]


def cleanup_old_room_dbs(db_dir: str | Path, max_age_sec: float = 86400):
    """Remove room DB files that haven't been modified for max_age_sec (default 1 day)."""
    d = Path(db_dir)
    if not d.exists():
        return
    cutoff = time.time() - max_age_sec
    for f in d.glob("room_*.db"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                # Also remove WAL and SHM files
                f.with_suffix(".db-wal").unlink(missing_ok=True)
                f.with_suffix(".db-shm").unlink(missing_ok=True)
        except OSError:
            pass
