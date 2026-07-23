"""SQLite storage for Verba: CRM write-back rows + conversation turns.

Stdlib sqlite3 only. One shared connection guarded by a lock so it's safe under
FastAPI's async loop and uvicorn --reload on Windows.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

# On Vercel the project filesystem is read-only except /tmp (ephemeral per instance).
_DB_BASE = Path("/tmp") if os.getenv("VERCEL") else Path(__file__).parent
DB_PATH = _DB_BASE / "app.db"
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS crm (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  scenario    TEXT    NOT NULL,          -- lead | collections | clinic
  kind        TEXT    NOT NULL,          -- callback | collection | appointment
  name        TEXT    DEFAULT '',
  phone       TEXT    DEFAULT '',
  summary     TEXT    DEFAULT '',
  details     TEXT    DEFAULT '{}',      -- full tool args as JSON
  status      TEXT    DEFAULT 'new',
  created_at  TEXT    DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS conversations (
  session_id  TEXT    PRIMARY KEY,
  created_at  TEXT    DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS turns (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT    NOT NULL,
  role        TEXT    NOT NULL,          -- user | assistant | tool
  text        TEXT,
  audio_ref   TEXT,
  created_at  TEXT    DEFAULT (datetime('now','localtime'))
);
"""


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def init_db() -> None:
    try:
        with _lock:
            conn = _connect()
            conn.executescript(SCHEMA)
            conn.commit()
    except Exception:
        pass


def ensure_conversation(session_id: str) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT OR IGNORE INTO conversations (session_id) VALUES (?)", (session_id,)
        )
        conn.commit()


def log_turn(session_id: str, role: str, text: str, audio_ref: str | None = None) -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO turns (session_id, role, text, audio_ref) VALUES (?,?,?,?)",
            (session_id, role, text, audio_ref),
        )
        conn.commit()


def insert_crm(r: dict) -> dict:
    """Insert a CRM row and return it as saved."""
    with _lock:
        conn = _connect()
        cur = conn.execute(
            "INSERT INTO crm (scenario, kind, name, phone, summary, details, status) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                str(r.get("scenario", "")).strip(),
                str(r.get("kind", "")).strip(),
                str(r.get("name", "") or "").strip(),
                str(r.get("phone", "") or "").strip(),
                str(r.get("summary", "") or "").strip(),
                str(r.get("details", "") or "{}"),
                str(r.get("status", "new") or "new").strip(),
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM crm WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)


def recent_crm(limit: int = 100) -> list[dict]:
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM crm ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# Ensure tables exist at import time too — Vercel serverless may not run the ASGI lifespan.
init_db()
