from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = os.getenv("JOB_DB_PATH", "/tmp/step-analysis-jobs/jobs.db")
JOB_FILES_DIR = Path(os.getenv("JOB_FILES_DIR", "/tmp/step-analysis-jobs/files"))

_db_lock = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    JOB_FILES_DIR.mkdir(parents=True, exist_ok=True)
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    filename TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_details TEXT,
                    consumed_at TEXT,
                    phase TEXT,
                    progress_percent INTEGER,
                    eta_seconds INTEGER
                )
                """
            )

            existing_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }

            for col_name, col_type in [
                ("phase", "TEXT"),
                ("progress_percent", "INTEGER"),
                ("eta_seconds", "INTEGER"),
            ]:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type}")

            conn.commit()
        finally:
            conn.close()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_job(filename: str, request_payload: dict[str, Any], file_bytes: bytes) -> str:
    job_id = str(uuid.uuid4())
    created_at = utc_now_iso()

    job_dir = JOB_FILES_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.bin"
    filename_path = job_dir / "input_name.txt"

    input_path.write_bytes(file_bytes)
    filename_path.write_text(filename, encoding="utf-8")

    with _db_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, created_at, filename, request_json, phase, progress_percent, eta_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    "queued",
                    created_at,
                    filename,
                    json.dumps(request_payload, ensure_ascii=False),
                    "queued",
                    0,
                    None,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _db_lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()

    if not row:
        return None

    result = dict(row)
    if result.get("request_json"):
        result["request_json"] = json.loads(result["request_json"])
    if result.get("result_json"):
        result["result_json"] = json.loads(result["result_json"])
    return result


def claim_next_job() -> dict[str, Any] | None:
    """Atomically claim the oldest queued job.

    BEGIN IMMEDIATE serializes claimers at SQLite level, so this remains safe if
    the service is later started with multiple worker processes. Each worker processes one job at a time; multiple configured workers may claim jobs concurrently.
    """
    with _db_lock:
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()

            if not row:
                conn.commit()
                return None

            job_id = row["job_id"]
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = ?, phase = ?, progress_percent = ?, eta_seconds = ?
                WHERE job_id = ? AND status = 'queued'
                """,
                ("processing", utc_now_iso(), "queued", 1, None, job_id),
            )
            if cur.rowcount != 1:
                conn.rollback()
                return None

            updated = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    if not updated:
        return None

    result = dict(updated)
    result["request_json"] = json.loads(result["request_json"])
    return result

def mark_done(job_id: str, result_payload: dict[str, Any]) -> None:
    with _db_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, finished_at = ?, result_json = ?, error_code = NULL, error_details = NULL,
                    phase = ?, progress_percent = ?, eta_seconds = ?
                WHERE job_id = ?
                """,
                ("done", utc_now_iso(), json.dumps(result_payload, ensure_ascii=False), "done", 100, 0, job_id),
            )
            conn.commit()
        finally:
            conn.close()


def mark_error(job_id: str, error_code: str, details: str) -> None:
    with _db_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, finished_at = ?, error_code = ?, error_details = ?,
                    phase = ?, eta_seconds = ?
                WHERE job_id = ?
                """,
                ("error", utc_now_iso(), error_code, details, "error", 0, job_id),
            )
            conn.commit()
        finally:
            conn.close()


def mark_consumed(job_id: str) -> None:
    with _db_lock:
        conn = _connect()
        try:
            conn.execute(
                """
                UPDATE jobs
                SET consumed_at = ?
                WHERE job_id = ?
                """,
                (utc_now_iso(), job_id),
            )
            conn.commit()
        finally:
            conn.close()


def delete_job(job_id: str) -> bool:
    job_dir = JOB_FILES_DIR / job_id

    with _db_lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "DELETE FROM jobs WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()
            deleted = cur.rowcount > 0
        finally:
            conn.close()

    if job_dir.exists():
        for path in sorted(job_dir.glob("**/*"), reverse=True):
            try:
                if path.is_file():
                    path.unlink()
            except Exception:
                pass
        try:
            for path in sorted(job_dir.glob("**/*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
            job_dir.rmdir()
        except Exception:
            pass

    return deleted


def get_queue_position(job_id: str) -> int | None:
    job = get_job(job_id)
    if not job or job["status"] != "queued":
        return None

    created_at = job["created_at"]

    with _db_lock:
        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) + 1 AS pos
                FROM jobs
                WHERE status = 'queued'
                  AND created_at < ?
                """,
                (created_at,),
            ).fetchone()
        finally:
            conn.close()

    return int(row["pos"]) if row else None


def get_job_dir(job_id: str) -> Path:
    return JOB_FILES_DIR / job_id


def update_job_progress(
    job_id: str,
    *,
    phase: str | None = None,
    progress_percent: int | None = None,
    eta_seconds: int | None = None,
) -> None:
    fields = []
    values = []

    if phase is not None:
        fields.append("phase = ?")
        values.append(phase)

    if progress_percent is not None:
        fields.append("progress_percent = ?")
        values.append(int(progress_percent))

    if eta_seconds is not None:
        fields.append("eta_seconds = ?")
        values.append(int(eta_seconds))
    elif eta_seconds is None and phase is not None:
        fields.append("eta_seconds = ?")
        values.append(None)

    if not fields:
        return

    values.append(job_id)

    with _db_lock:
        conn = _connect()
        try:
            conn.execute(
                f"""
                UPDATE jobs
                SET {", ".join(fields)}
                WHERE job_id = ?
                """,
                tuple(values),
            )
            conn.commit()
        finally:
            conn.close()