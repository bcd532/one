"""Synthesis engine — automatic insight generation from the knowledge graph.

Scans the entity graph for cross-domain connections between unrelated
concepts. When entities from different contexts co-occur in shared memories,
generates hypotheses about the underlying patterns using Gemma. Supports
recursive deep synthesis where existing insights seed further connections,
forming a directed acyclic graph of progressively deeper understanding.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from .store import _get_conn as _store_conn, push_memory, DB_DIR, DB_PATH


_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local connection with the syntheses table initialized."""
    if not hasattr(_local, "conn") or _local.conn is None:
        import os
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        _init_synthesis_schema(conn)
    return _local.conn


def _init_synthesis_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS syntheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT,
            entity_a TEXT,
            entity_b TEXT,
            hypothesis TEXT,
            confidence REAL,
            parent_id INTEGER REFERENCES syntheses(id),
            depth INTEGER DEFAULT 0,
            created TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_syntheses_project ON syntheses(project);
        CREATE INDEX IF NOT EXISTS idx_syntheses_depth ON syntheses(depth);
        CREATE INDEX IF NOT EXISTS idx_syntheses_parent ON syntheses(parent_id);
    """)
    conn.commit()


def init_schema() -> None:
    """Public entry point to ensure the syntheses table exists."""
    _init_synthesis_schema(_get_conn())


# ── Cross-domain entity pair discovery ─────────────────────────────


def _find_cross_domain_pairs(
    project: str,
    min_shared: int = 1,
    limit: int = 50,
) -> list[dict]:
    """Find entity pairs that share memories but belong to different label domains.

    Returns pairs ordered by number of shared memories descending.
    """
    conn = _store_conn()
    rows = conn.execute("""
        SELECT
            e1.name AS entity_a,
            e1.type AS type_a,
            e2.name AS entity_b,
            e2.type AS type_b,
            COUNT(DISTINCT me1.memory_id) AS shared_count
        FROM memory_entities me1
        JOIN memory_entities me2
            ON me1.memory_id = me2.memory_id AND me1.entity_id < me2.entity_id
        JOIN entities e1 ON me1.entity_id = e1.id
        JOIN entities e2 ON me2.entity_id = e2.id
        JOIN memories m ON me1.memory_id = m.id
        WHERE m.project = ? OR m.project = 'global'
        GROUP BY e1.id, e2.id
        HAVING shared_count >= ?
        ORDER BY shared_count DESC
        LIMIT ?
    """, (project, min_shared, limit)).fetchall()

    return [dict(r) for r in rows]


def _get_shared_memories(entity_a: str, entity_b: str, limit: int = 10) -> list[dict]:
    """Retrieve memories shared between two entities."""
    conn = _store_conn()
    rows = conn.execute("""
        SELECT DISTINCT m.id, m.raw_text, m.source, m.tm_label, m.aif_confidence
        FROM memories m
        JOIN memory_entities me1 ON m.id = me1.memory_id
        JOIN entities e1 ON me1.entity_id = e1.id
        JOIN memory_entities me2 ON m.id = me2.memory_id
        JOIN entities e2 ON me2.entity_id = e2.id
        WHERE e1.name = ? AND e2.name = ?
        ORDER BY m.timestamp DESC
        LIMIT ?
    """, (entity_a, entity_b, limit)).fetchall()

    return [dict(r) for r in rows]


def _already_synthesized(project: str, entity_a: str, entity_b: str) -> bool:
    """Check whether this entity pair has already been synthesized."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM syntheses WHERE project = ? AND entity_a = ? AND entity_b = ?",
        (project, entity_a, entity_b),
    ).fetchone()
    return row[0] > 0


# ── Hypothesis generation ──────────────────────────────────────────


SYNTHESIS_PROMPT = """You are a knowledge synthesis engine. Given two entities connected through shared memories, generate a concise hypothesis about the pattern or relationship connecting them.

Entity A: {entity_a} (type: {type_a})
Entity B: {entity_b} (type: {type_b})
Shared memories ({shared_count}):
{memory_context}

Generate a single precise hypothesis (2-3 sentences) about what pattern connects these two concepts. Focus on actionable insight, not restatement.

HYPOTHESIS:"""


DEEP_SYNTHESIS_PROMPT = """You are a deep knowledge synthesis engine operating at depth {depth}. Previous synthesis insights are provided below. Generate a higher-order hypothesis that connects or extends these existing insights.

Previous insights:
{prior_insights}

Generate a single precise meta-hypothesis (2-3 sentences) that identifies a deeper pattern across the prior insights. This should reveal something not obvious from any individual insight.

META-HYPOTHESIS:"""


def _generate_hypothesis(
    entity_a: str,
    type_a: str,
    entity_b: str,
    type_b: str,
    shared_memories: list[dict],
) -> Optional[str]:
    """Use Gemma to generate a synthesis hypothesis from shared context."""
    from .gemma import _call_ollama, is_available

    if not is_available():
        return _fallback_hypothesis(entity_a, entity_b, shared_memories)

    memory_lines = []
    for m in shared_memories[:8]:
        src = m.get("source", "?")
        label = m.get("tm_label", "?")
        text = m.get("raw_text", "")[:200]
        memory_lines.append(f"  [{src}|{label}] {text}")

    prompt = SYNTHESIS_PROMPT.format(
        entity_a=entity_a,
        type_a=type_a,
        entity_b=entity_b,
        type_b=type_b,
        shared_count=len(shared_memories),
        memory_context="\n".join(memory_lines),
    )

    result = _call_ollama(prompt, timeout=60)
    if result:
        return result.strip()

    return _fallback_hypothesis(entity_a, entity_b, shared_memories)


def _fallback_hypothesis(
    entity_a: str,
    entity_b: str,
    shared_memories: list[dict],
) -> str:
    """Generate a basic hypothesis without LLM when Gemma is unavailable."""
    labels = set()
    for m in shared_memories:
        label = m.get("tm_label", "")
        if label and label != "unclassified":
            labels.add(label)

    label_str = ", ".join(labels) if labels else "shared context"
    return (
        f"{entity_a} and {entity_b} co-occur across {len(shared_memories)} memories "
        f"in domains: {label_str}. This co-occurrence suggests a structural or "
        f"functional relationship worth investigating."
    )


def _generate_deep_hypothesis(prior_insights: list[str], depth: int) -> Optional[str]:
    """Generate a meta-hypothesis from existing synthesis insights."""
    from .gemma import _call_ollama, is_available

    if not is_available() or not prior_insights:
        return None

    insight_text = "\n".join(f"  - {h}" for h in prior_insights[:10])
    prompt = DEEP_SYNTHESIS_PROMPT.format(
        depth=depth,
        prior_insights=insight_text,
    )

    result = _call_ollama(prompt, timeout=60)
    return result.strip() if result else None


# ── Core synthesis operations ──────────────────────────────────────


def run_synthesis(
    project: str,
    min_shared: int = 1,
    max_pairs: int = 20,
) -> list[dict]:
    """Scan the entity graph for cross-connections and generate hypotheses.

    Returns a list of newly created synthesis entries.
    """
    init_schema()

    pairs = _find_cross_domain_pairs(project, min_shared=min_shared, limit=max_pairs)
    results = []

    for pair in pairs:
        entity_a = pair["entity_a"]
        entity_b = pair["entity_b"]

        if _already_synthesized(project, entity_a, entity_b):
            continue

        shared = _get_shared_memories(entity_a, entity_b)
        if not shared:
            continue

        hypothesis = _generate_hypothesis(
            entity_a, pair["type_a"],
            entity_b, pair["type_b"],
            shared,
        )
        if not hypothesis:
            continue

        # Compute confidence from shared memory count and average AIF confidence
        avg_conf = sum(m.get("aif_confidence", 0) for m in shared) / max(len(shared), 1)
        confidence = min(1.0, 0.5 + avg_conf * 0.3 + min(len(shared), 5) * 0.05)

        # Store the synthesis record
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """INSERT INTO syntheses
               (project, entity_a, entity_b, hypothesis, confidence, parent_id, depth, created)
               VALUES (?, ?, ?, ?, ?, NULL, 0, ?)""",
            (project, entity_a, entity_b, hypothesis, confidence, now),
        )
        conn.commit()
        synthesis_id = cur.lastrowid

        # Store as a memory for retrieval and context injection
        push_memory(
            raw_text=hypothesis,
            source="synthesis",
            tm_label="hypothesis",
            regime_tag="synthesis",
            aif_confidence=confidence,
            project=project,
        )

        results.append({
            "id": synthesis_id,
            "entity_a": entity_a,
            "entity_b": entity_b,
            "hypothesis": hypothesis,
            "confidence": confidence,
            "depth": 0,
        })

    return results


def run_deep_synthesis(
    project: str,
    depth: int = 3,
    max_per_level: int = 10,
) -> list[dict]:
    """Recursively synthesize: existing insights seed deeper connections.

    At each depth level, takes the prior level's hypotheses and generates
    meta-hypotheses that identify patterns across patterns. Each level
    builds on the previous, forming a DAG of progressively deeper insight.
    """
    init_schema()
    all_results = []

    # Level 0: run base synthesis if not already done
    base = run_synthesis(project)
    all_results.extend(base)

    for current_depth in range(1, depth + 1):
        # Gather hypotheses from the previous depth level
        conn = _get_conn()
        prior_rows = conn.execute(
            """SELECT id, hypothesis, confidence FROM syntheses
               WHERE project = ? AND depth = ?
               ORDER BY confidence DESC
               LIMIT ?""",
            (project, current_depth - 1, max_per_level * 2),
        ).fetchall()

        if len(prior_rows) < 2:
            break

        prior_insights = [r["hypothesis"] for r in prior_rows]
        prior_ids = [r["id"] for r in prior_rows]

        meta_hypothesis = _generate_deep_hypothesis(prior_insights, current_depth)
        if not meta_hypothesis:
            break

        # Average confidence of parent insights, slightly decayed
        avg_conf = sum(r["confidence"] for r in prior_rows) / len(prior_rows)
        confidence = min(1.0, avg_conf * 0.9)

        # Store with parent reference to the highest-confidence prior insight
        parent_id = prior_ids[0] if prior_ids else None
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """INSERT INTO syntheses
               (project, entity_a, entity_b, hypothesis, confidence, parent_id, depth, created)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (project, "meta", "meta", meta_hypothesis, confidence, parent_id, current_depth, now),
        )
        conn.commit()
        synthesis_id = cur.lastrowid

        push_memory(
            raw_text=meta_hypothesis,
            source="synthesis",
            tm_label=f"hypothesis:depth_{current_depth}",
            regime_tag="synthesis",
            aif_confidence=confidence,
            project=project,
        )

        result = {
            "id": synthesis_id,
            "entity_a": "meta",
            "entity_b": "meta",
            "hypothesis": meta_hypothesis,
            "confidence": confidence,
            "depth": current_depth,
            "parent_id": parent_id,
        }
        all_results.append(result)

    return all_results


# ── Synthesis chain retrieval ──────────────────────────────────────


def get_synthesis_chain(project: str) -> list[dict]:
    """Return the synthesis DAG as a list of connected insights.

    Each entry includes its parent reference, forming a traversable chain
    from base-level cross-domain observations to deep meta-hypotheses.
    """
    init_schema()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, entity_a, entity_b, hypothesis, confidence, parent_id, depth, created
           FROM syntheses
           WHERE project = ?
           ORDER BY depth ASC, confidence DESC""",
        (project,),
    ).fetchall()

    return [dict(r) for r in rows]


def get_syntheses_count(project: str) -> int:
    """Return the total number of synthesis entries for a project."""
    init_schema()
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM syntheses WHERE project = ?",
        (project,),
    ).fetchone()
    return row[0]
