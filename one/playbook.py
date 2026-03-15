"""Playbook system — distilled reusable strategies from completed tasks.

When an auto loop or research session completes, the playbook system
analyzes what worked and encodes the successful strategy as a structured,
retrievable memory. Playbooks are recalled by vector similarity during
future tasks, injecting proven approaches into the agent's context.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from .store import _get_conn as _store_conn, push_memory, recall, DB_DIR, DB_PATH


_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local connection with the playbooks table initialized."""
    if not hasattr(_local, "conn") or _local.conn is None:
        import os
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        _init_playbook_schema(conn)
    return _local.conn


def _init_playbook_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS playbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            task_description TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            key_decisions TEXT,
            reusable_patterns TEXT,
            pitfalls TEXT,
            full_playbook TEXT NOT NULL,
            confidence REAL DEFAULT 0.9,
            times_recalled INTEGER DEFAULT 0,
            created TEXT NOT NULL,
            updated TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_playbooks_project ON playbooks(project);
        CREATE INDEX IF NOT EXISTS idx_playbooks_category ON playbooks(category);
    """)
    conn.commit()


def init_schema() -> None:
    """Public entry point to ensure the playbooks table exists."""
    _init_playbook_schema(_get_conn())


# ── Playbook generation prompts ────────────────────────────────────


ANALYSIS_PROMPT = """Analyze this completed task and distill it into a reusable playbook.

TASK: {task_description}

STEPS TAKEN:
{steps_taken}

OUTCOME: {outcome}

Respond in exactly this format:

CATEGORY: <one-word category like: debug, feature, refactor, research, deploy, test>

KEY DECISIONS:
- <decision 1>
- <decision 2>
...

REUSABLE PATTERNS:
- <pattern 1>
- <pattern 2>
...

PITFALLS TO AVOID:
- <pitfall 1>
- <pitfall 2>
...

PLAYBOOK SUMMARY:
<2-4 sentence summary of the strategy that worked>"""


CATEGORY_FALLBACK = {
    "fix": "debug",
    "bug": "debug",
    "error": "debug",
    "test": "test",
    "deploy": "deploy",
    "ship": "deploy",
    "refactor": "refactor",
    "clean": "refactor",
    "research": "research",
    "investigate": "research",
    "build": "feature",
    "implement": "feature",
    "add": "feature",
    "create": "feature",
}


# ── Playbook generation ───────────────────────────────────────────


def _parse_analysis(text: str) -> dict:
    """Parse the structured analysis response into components."""
    sections = {
        "category": "general",
        "key_decisions": "",
        "reusable_patterns": "",
        "pitfalls": "",
        "playbook_summary": "",
    }

    current_section = None
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("CATEGORY:"):
            if current_section and current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "category"
            sections["category"] = stripped[9:].strip().lower()
            current_lines = []
        elif upper.startswith("KEY DECISIONS:"):
            if current_section and current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "key_decisions"
            current_lines = []
        elif upper.startswith("REUSABLE PATTERNS:"):
            if current_section and current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "reusable_patterns"
            current_lines = []
        elif upper.startswith("PITFALLS TO AVOID:") or upper.startswith("PITFALLS:"):
            if current_section and current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "pitfalls"
            current_lines = []
        elif upper.startswith("PLAYBOOK SUMMARY:") or upper.startswith("PLAYBOOK:"):
            if current_section and current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "playbook_summary"
            current_lines = []
        elif current_section:
            current_lines.append(stripped)

    if current_section and current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def _infer_category(task_description: str) -> str:
    """Infer a playbook category from the task description when LLM is unavailable."""
    lower = task_description.lower()
    for keyword, category in CATEGORY_FALLBACK.items():
        if keyword in lower:
            return category
    return "general"


def create_playbook(
    project: str,
    task_description: str,
    steps_taken: str,
    outcome: str,
) -> Optional[dict]:
    """Generate a playbook from a completed task.

    Uses Gemma to analyze the task, extract key decisions and reusable
    patterns, and produce a structured playbook. Falls back to a simpler
    format when Gemma is unavailable.

    Returns the playbook dict, or None if generation fails entirely.
    """
    init_schema()

    from .gemma import _call_ollama, is_available

    if is_available():
        prompt = ANALYSIS_PROMPT.format(
            task_description=task_description,
            steps_taken=steps_taken[:3000],
            outcome=outcome,
        )
        result = _call_ollama(prompt, timeout=90)

        if result:
            sections = _parse_analysis(result)
        else:
            sections = _build_fallback(task_description, steps_taken, outcome)
    else:
        sections = _build_fallback(task_description, steps_taken, outcome)

    # Build the full playbook text for storage and retrieval
    full_playbook = (
        f"Task: {task_description}\n"
        f"Category: {sections['category']}\n\n"
        f"Key Decisions:\n{sections['key_decisions']}\n\n"
        f"Reusable Patterns:\n{sections['reusable_patterns']}\n\n"
        f"Pitfalls:\n{sections['pitfalls']}\n\n"
        f"Summary: {sections['playbook_summary']}"
    )

    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO playbooks
           (project, task_description, category, key_decisions, reusable_patterns,
            pitfalls, full_playbook, confidence, created, updated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (project, task_description, sections["category"],
         sections["key_decisions"], sections["reusable_patterns"],
         sections["pitfalls"], full_playbook, 0.9, now, now),
    )
    conn.commit()
    playbook_id = cur.lastrowid

    # Store as a memory for vector retrieval
    push_memory(
        raw_text=full_playbook,
        source="playbook",
        tm_label=f"playbook:{sections['category']}",
        regime_tag="playbook",
        aif_confidence=0.9,
        project=project,
    )

    return {
        "id": playbook_id,
        "task_description": task_description,
        "category": sections["category"],
        "key_decisions": sections["key_decisions"],
        "reusable_patterns": sections["reusable_patterns"],
        "pitfalls": sections["pitfalls"],
        "full_playbook": full_playbook,
        "confidence": 0.9,
    }


def _build_fallback(
    task_description: str,
    steps_taken: str,
    outcome: str,
) -> dict:
    """Build a basic playbook structure without LLM assistance."""
    category = _infer_category(task_description)

    # Extract steps as key decisions
    steps = []
    for line in steps_taken.split("\n"):
        line = line.strip()
        if line and len(line) > 10:
            steps.append(f"- {line[:200]}")
    key_decisions = "\n".join(steps[:10]) if steps else "- (no steps recorded)"

    return {
        "category": category,
        "key_decisions": key_decisions,
        "reusable_patterns": f"- Applied {category} strategy to: {task_description[:100]}",
        "pitfalls": "- (analyze manually for pitfalls)",
        "playbook_summary": f"Completed {category} task: {task_description[:150]}. Outcome: {outcome[:150]}",
    }


# ── Playbook recall ───────────────────────────────────────────────


def recall_playbook(project: str, task_description: str, n: int = 3) -> list[dict]:
    """Find relevant playbooks by vector similarity to a task description.

    Searches both the playbooks table directly and the memory store
    for playbook-type memories, returning the most relevant matches.
    """
    init_schema()

    from .hdc import encode_text, similarity
    import numpy as np

    query_vec = encode_text(task_description)
    query_norm = np.linalg.norm(query_vec)
    if query_norm < 1e-10:
        return []

    # Search memory store for playbook memories
    memories = recall(task_description, n=n * 2, project=project)
    playbook_memories = [m for m in memories if m.get("source") == "playbook"]

    # Also search the playbooks table directly
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM playbooks WHERE project = ? ORDER BY created DESC LIMIT 50",
        (project,),
    ).fetchall()

    scored = []
    seen_ids = set()

    for row in rows:
        pb = dict(row)
        pb_vec = encode_text(pb["full_playbook"])
        pb_norm = np.linalg.norm(pb_vec)
        if pb_norm < 1e-10:
            continue
        sim = float(np.dot(query_vec, pb_vec) / (query_norm * pb_norm))
        scored.append((sim, pb))
        seen_ids.add(pb["id"])

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim, pb in scored[:n]:
        pb["similarity"] = sim
        results.append(pb)

        # Increment recall counter
        conn.execute(
            "UPDATE playbooks SET times_recalled = times_recalled + 1, updated = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), pb["id"]),
        )

    conn.commit()
    return results


def recall_playbook_context(project: str, task_description: str) -> str:
    """Format recalled playbooks as a context block for injection into auto mode."""
    playbooks = recall_playbook(project, task_description, n=2)
    if not playbooks:
        return ""

    lines = ["<prior-playbooks source=\"one\">"]
    for pb in playbooks:
        lines.append(f"[{pb['category']}] {pb['task_description'][:80]}")
        if pb.get("reusable_patterns"):
            lines.append(f"  Patterns: {pb['reusable_patterns'][:300]}")
        if pb.get("pitfalls"):
            lines.append(f"  Avoid: {pb['pitfalls'][:200]}")
        lines.append("")
    lines.append("</prior-playbooks>")

    return "\n".join(lines)


# ── Playbook listing ──────────────────────────────────────────────


def list_playbooks(project: str) -> list[dict]:
    """Return all playbooks for a project, ordered by recency."""
    init_schema()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, task_description, category, key_decisions, reusable_patterns,
                  pitfalls, confidence, times_recalled, created
           FROM playbooks
           WHERE project = ?
           ORDER BY created DESC""",
        (project,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_playbook_count(project: str) -> int:
    """Return the total number of playbooks for a project."""
    init_schema()
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM playbooks WHERE project = ?",
        (project,),
    ).fetchone()
    return row[0]
