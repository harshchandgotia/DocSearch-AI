import sqlite3
import json
from uuid import uuid4
from datetime import datetime, timezone

from config import load_config

_config = load_config()
_db_path = _config["database"]["path"]


def _get_connection():
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db():
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                pinned_pdf_ids TEXT NOT NULL,
                title TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                is_supported TEXT,
                is_useful TEXT,
                revision_count INTEGER,
                rewrite_count INTEGER,
                sources TEXT,
                retrieval_used INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        conn.commit()
    finally:
        conn.close()


_init_db()


def create_session(pinned_pdf_ids: list[str], title: str = "") -> str:
    session_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, pinned_pdf_ids, title) VALUES (?, ?, ?, ?)",
            (session_id, now, json.dumps(pinned_pdf_ids), title),
        )
        conn.commit()
    finally:
        conn.close()
    return session_id


def get_session(session_id: str) -> dict | None:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["pinned_pdf_ids"] = json.loads(result["pinned_pdf_ids"])
        return result
    finally:
        conn.close()


def list_sessions() -> list[dict]:
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT s.*, COUNT(m.message_id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.created_at DESC
        """).fetchall()
        sessions = []
        for row in rows:
            session = dict(row)
            session["pinned_pdf_ids"] = json.loads(session["pinned_pdf_ids"])
            sessions.append(session)
        return sessions
    finally:
        conn.close()


def delete_session(session_id: str):
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


def add_message(
    session_id: str, role: str, content: str, metadata: dict | None = None
) -> str:
    message_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    meta = metadata or {}
    sources_json = json.dumps(meta.get("sources")) if meta.get("sources") else None
    retrieval_used = (
        1 if meta.get("retrieval_used") else 0
    ) if "retrieval_used" in meta else None

    conn = _get_connection()
    try:
        conn.execute(
            """INSERT INTO messages
            (message_id, session_id, timestamp, role, content,
             is_supported, is_useful, revision_count, rewrite_count, sources, retrieval_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message_id,
                session_id,
                now,
                role,
                content,
                meta.get("is_supported"),
                meta.get("is_useful"),
                meta.get("revision_count"),
                meta.get("rewrite_count"),
                sources_json,
                retrieval_used,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return message_id


def get_messages(session_id: str) -> list[dict]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
        messages = []
        for row in rows:
            msg = dict(row)
            if msg["sources"]:
                msg["sources"] = json.loads(msg["sources"])
            msg["retrieval_used"] = bool(msg["retrieval_used"]) if msg["retrieval_used"] is not None else None
            messages.append(msg)
        return messages
    finally:
        conn.close()


def update_session_title(session_id: str, title: str):
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE session_id = ?", (title, session_id)
        )
        conn.commit()
    finally:
        conn.close()


def update_session_pdf_ids(session_id: str, pinned_pdf_ids: list[str]):
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE sessions SET pinned_pdf_ids = ? WHERE session_id = ?",
            (json.dumps(pinned_pdf_ids), session_id),
        )
        conn.commit()
    finally:
        conn.close()
