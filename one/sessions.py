"""Session storage for conversation history.

Stores and retrieves conversation sessions in SQLite, using the same
shared database (~/.one/one.db) as the memory store. Each session tracks
metadata (project, model, cost, timing) and an ordered sequence of messages.
"""

import os
import sqlite3
import uuid
import threading
from datetime import datetime, timezone
from typing import Optional

DB_DIR = os.path.expanduser("~/.one")
DB_PATH = os.path.join(DB_DIR, "one.db")

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with WAL mode enabled."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        _init_schema(conn)
    return _local.conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            project TEXT,
            model TEXT,
            start_time TEXT,
            end_time TEXT,
            turn_count INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES sessions(id),
            role TEXT,
            content TEXT,
            timestamp TEXT,
            turn_number INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_project
            ON sessions(project);
        CREATE INDEX IF NOT EXISTS idx_sessions_start_time
            ON sessions(start_time);
        CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
            ON session_messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_session_messages_turn_number
            ON session_messages(session_id, turn_number);
    """)
    conn.commit()


# ── Session lifecycle ─────────────────────────────────────────────


def create_session(project: str, model: str) -> str:
    """Start a new conversation session. Returns the session ID."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_conn()
    conn.execute(
        "INSERT INTO sessions (id, project, model, start_time) VALUES (?, ?, ?, ?)",
        (session_id, project, model, now),
    )
    conn.commit()
    return session_id


def end_session(session_id: str, total_cost: float = 0.0) -> None:
    """Finalize a session with its end time and total cost."""
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_conn()
    conn.execute(
        "UPDATE sessions SET end_time = ?, total_cost = ? WHERE id = ?",
        (now, total_cost, session_id),
    )
    conn.commit()


# ── Message operations ────────────────────────────────────────────


def add_message(
    session_id: str,
    role: str,
    content: str,
    turn_number: int,
) -> int:
    """Append a message to a session. Returns the message row ID.

    Automatically increments the session's turn_count to reflect the
    highest turn number recorded.
    """
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO session_messages (session_id, role, content, timestamp, turn_number) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, now, turn_number),
    )
    conn.execute(
        "UPDATE sessions SET turn_count = MAX(turn_count, ?) WHERE id = ?",
        (turn_number, session_id),
    )
    conn.commit()
    return cur.lastrowid


# ── Query operations ──────────────────────────────────────────────


def list_sessions(
    project: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """List sessions ordered by start time, optionally filtered by project."""
    conn = _get_conn()

    if project:
        rows = conn.execute(
            "SELECT id, project, model, start_time, end_time, turn_count, total_cost "
            "FROM sessions WHERE project = ? ORDER BY start_time DESC LIMIT ?",
            (project, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, project, model, start_time, end_time, turn_count, total_cost "
            "FROM sessions ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [dict(r) for r in rows]


def get_session_messages(
    session_id: str,
    limit: int = 100,
) -> list[dict]:
    """Retrieve messages for a session, ordered by turn number and timestamp."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, session_id, role, content, timestamp, turn_number "
        "FROM session_messages WHERE session_id = ? "
        "ORDER BY turn_number ASC, timestamp ASC LIMIT ?",
        (session_id, limit),
    ).fetchall()

    return [dict(r) for r in rows]


def get_latest_session(project: str) -> Optional[dict]:
    """Return the most recent session for a project, or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, project, model, start_time, end_time, turn_count, total_cost "
        "FROM sessions WHERE project = ? ORDER BY start_time DESC LIMIT 1",
        (project,),
    ).fetchone()

    return dict(row) if row else None


# ── Export ────────────────────────────────────────────────────────


def export_session_markdown(session_id: str) -> str:
    """Export a full session as a Markdown-formatted conversation transcript.

    Returns a string containing session metadata as a header followed by
    each message formatted with role labels and turn separators.
    """
    conn = _get_conn()

    session = conn.execute(
        "SELECT id, project, model, start_time, end_time, turn_count, total_cost "
        "FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()

    if session is None:
        return f"Session {session_id} not found."

    session = dict(session)
    messages = get_session_messages(session_id, limit=10000)

    lines = [
        f"# Session: {session['id'][:8]}",
        "",
        f"- **Project**: {session['project']}",
        f"- **Model**: {session['model']}",
        f"- **Started**: {session['start_time']}",
        f"- **Ended**: {session['end_time'] or 'in progress'}",
        f"- **Turns**: {session['turn_count']}",
        f"- **Cost**: ${session['total_cost']:.4f}",
        "",
        "---",
        "",
    ]

    _ROLE_LABELS = {
        "user": "User",
        "assistant": "Assistant",
        "tool_use": "Tool Use",
        "tool_result": "Tool Result",
        "thinking": "Thinking",
    }

    prev_turn = None
    for msg in messages:
        if msg["turn_number"] != prev_turn:
            if prev_turn is not None:
                lines.append("")
                lines.append("---")
                lines.append("")
            prev_turn = msg["turn_number"]

        label = _ROLE_LABELS.get(msg["role"], msg["role"])
        lines.append(f"### {label} (turn {msg['turn_number']})")
        lines.append("")
        lines.append(msg["content"] or "")
        lines.append("")

    return "\n".join(lines)
