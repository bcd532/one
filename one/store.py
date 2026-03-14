"""Local SQLite memory store.

Default storage backend requiring no external services. Provides vector
similarity search via numpy cosine over stored binary blobs, with the same
schema as the Foundry ontology backend.
"""

import os
import sqlite3
import uuid
import struct
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np

DB_DIR = os.path.expanduser("~/.one")
DB_PATH = os.path.join(DB_DIR, "one.db")
VECTOR_DIM = 4096

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
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            raw_text TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            hdc_vector BLOB,
            tm_label TEXT DEFAULT 'unclassified',
            regime_tag TEXT DEFAULT 'default',
            aif_confidence REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL,
            observation_count INTEGER DEFAULT 1,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_entities (
            memory_id TEXT REFERENCES memories(id),
            entity_id INTEGER REFERENCES entities(id),
            PRIMARY KEY (memory_id, entity_id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
        CREATE INDEX IF NOT EXISTS idx_memories_label ON memories(tm_label);
        CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
    """)
    conn.commit()


# ── Vector encoding/decoding ───────────────────────────────────────

def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


# ── Memory operations ──────────────────────────────────────────────

def push_memory(
    raw_text: str,
    source: str,
    tm_label: str = "unclassified",
    regime_tag: str = "default",
    aif_confidence: float = 0.0,
    hdc_vector: Optional[list[float]] = None,
) -> str:
    """Store a memory entry. Returns the generated memory ID."""
    if hdc_vector is None:
        from .hdc import encode_tagged
        vec = encode_tagged(raw_text, role=source)
        hdc_vector = vec.tolist()

    mid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    blob = _vec_to_blob(hdc_vector)

    conn = _get_conn()
    conn.execute(
        "INSERT INTO memories (id, raw_text, source, timestamp, hdc_vector, tm_label, regime_tag, aif_confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (mid, raw_text, source, now, blob, tm_label, regime_tag, aif_confidence),
    )
    conn.commit()
    return mid


def recall(
    query: str,
    n: int = 10,
) -> list[dict]:
    """Vector similarity search. Returns top N memories ranked by cosine similarity."""
    from .hdc import encode_text

    query_vec = encode_text(query).astype(np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm < 1e-10:
        return []

    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, raw_text, source, timestamp, hdc_vector, tm_label, regime_tag, aif_confidence FROM memories WHERE hdc_vector IS NOT NULL"
    ).fetchall()

    scored = []
    for row in rows:
        vec = _blob_to_vec(row["hdc_vector"])
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-10:
            continue
        sim = float(np.dot(query_vec, vec) / (query_norm * vec_norm))
        scored.append((sim, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim, row in scored[:n]:
        results.append({
            "id": row["id"],
            "raw_text": row["raw_text"],
            "source": row["source"],
            "timestamp": row["timestamp"],
            "tm_label": row["tm_label"],
            "regime_tag": row["regime_tag"],
            "aif_confidence": row["aif_confidence"],
            "similarity": sim,
        })
    return results


def get_memory_by_time(
    since: Optional[str] = None,
    until: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Query memories by time range and/or source."""
    conn = _get_conn()
    clauses = []
    params = []

    if since:
        clauses.append("timestamp >= ?")
        params.append(since)
    if until:
        clauses.append("timestamp <= ?")
        params.append(until)
    if source:
        clauses.append("source = ?")
        params.append(source)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT id, raw_text, source, timestamp, tm_label, regime_tag, aif_confidence FROM memories {where} ORDER BY timestamp DESC LIMIT ?",
        params + [limit],
    ).fetchall()

    return [dict(r) for r in rows]


# ── Entity operations ──────────────────────────────────────────────

def ensure_entity(entity: dict) -> int:
    """Create or update an entity. Returns the entity row ID."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    name = entity["name"]
    etype = entity["type"]

    row = conn.execute("SELECT id, observation_count FROM entities WHERE name = ?", (name,)).fetchone()
    if row:
        conn.execute(
            "UPDATE entities SET observation_count = ?, last_seen = ? WHERE id = ?",
            (row["observation_count"] + 1, now, row["id"]),
        )
        conn.commit()
        return row["id"]
    else:
        cur = conn.execute(
            "INSERT INTO entities (name, type, first_seen, last_seen) VALUES (?, ?, ?, ?)",
            (name, etype, now, now),
        )
        conn.commit()
        return cur.lastrowid


def link_memory_entity(memory_id: str, entity_id: int) -> None:
    """Create a link between a memory and an entity."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (memory_id, entity_id),
    )
    conn.commit()


def get_entities(entity_type: Optional[str] = None, limit: int = 50) -> list[dict]:
    """List entities, optionally filtered by type, ordered by observation count."""
    conn = _get_conn()
    if entity_type:
        rows = conn.execute(
            "SELECT * FROM entities WHERE type = ? ORDER BY observation_count DESC LIMIT ?",
            (entity_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM entities ORDER BY observation_count DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_memories_for_entity(entity_name: str, limit: int = 20) -> list[dict]:
    """Retrieve all memories linked to a given entity by name."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT m.id, m.raw_text, m.source, m.timestamp, m.tm_label, m.regime_tag, m.aif_confidence
        FROM memories m
        JOIN memory_entities me ON m.id = me.memory_id
        JOIN entities e ON me.entity_id = e.id
        WHERE e.name = ?
        ORDER BY m.timestamp DESC
        LIMIT ?
    """, (entity_name, limit)).fetchall()
    return [dict(r) for r in rows]


def get_related_entities(entity_name: str, limit: int = 10) -> list[dict]:
    """Find entities that co-occur with a given entity across shared memories."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT e2.name, e2.type, COUNT(*) as shared_memories
        FROM memory_entities me1
        JOIN entities e1 ON me1.entity_id = e1.id
        JOIN memory_entities me2 ON me1.memory_id = me2.memory_id
        JOIN entities e2 ON me2.entity_id = e2.id
        WHERE e1.name = ? AND e2.name != ?
        GROUP BY e2.id
        ORDER BY shared_memories DESC
        LIMIT ?
    """, (entity_name, entity_name, limit)).fetchall()
    return [dict(r) for r in rows]


# ── Stats ──────────────────────────────────────────────────────────

def stats() -> dict:
    """Return aggregate statistics about the memory store."""
    conn = _get_conn()
    mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    ent_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    link_count = conn.execute("SELECT COUNT(*) FROM memory_entities").fetchone()[0]

    top_concepts = conn.execute("""
        SELECT name, observation_count FROM entities
        WHERE type = 'concept' ORDER BY observation_count DESC LIMIT 5
    """).fetchall()

    top_files = conn.execute("""
        SELECT name, observation_count FROM entities
        WHERE type = 'file' ORDER BY observation_count DESC LIMIT 5
    """).fetchall()

    return {
        "memories": mem_count,
        "entities": ent_count,
        "links": link_count,
        "top_concepts": [dict(r) for r in top_concepts],
        "top_files": [dict(r) for r in top_files],
    }
