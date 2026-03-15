"""Deep research mode — autonomous topic investigation with citation graphs.

Formulates search queries from a topic and existing knowledge, sends
structured research prompts to Gemma, extracts entities and findings,
builds a citation-like graph of supporting/contradicting relationships,
and iteratively identifies knowledge gaps to drive further investigation.

The research loop cycles through: research -> synthesize -> identify gaps ->
research more -> synthesize deeper, stopping when no gaps remain or the
turn budget is exhausted.
"""

import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import _get_conn as _store_conn, push_memory, DB_DIR, DB_PATH


_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local connection with the research tables initialized."""
    if not hasattr(_local, "conn") or _local.conn is None:
        import os
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        _init_research_schema(conn)
    return _local.conn


def _init_research_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS research_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            topic TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            turns_used INTEGER DEFAULT 0,
            turn_budget INTEGER DEFAULT 10,
            findings_count INTEGER DEFAULT 0,
            gaps_count INTEGER DEFAULT 0,
            synthesis_depth INTEGER DEFAULT 0,
            created TEXT NOT NULL,
            updated TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS research_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER REFERENCES research_topics(id),
            content TEXT NOT NULL,
            finding_type TEXT DEFAULT 'finding',
            source_query TEXT,
            confidence REAL DEFAULT 0.5,
            created TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS research_citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            finding_a INTEGER REFERENCES research_findings(id),
            finding_b INTEGER REFERENCES research_findings(id),
            relation TEXT DEFAULT 'supports',
            created TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS research_gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER REFERENCES research_topics(id),
            question TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            resolved_by INTEGER REFERENCES research_findings(id),
            created TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_research_topics_project ON research_topics(project);
        CREATE INDEX IF NOT EXISTS idx_research_findings_topic ON research_findings(topic_id);
        CREATE INDEX IF NOT EXISTS idx_research_gaps_topic ON research_gaps(topic_id);
        CREATE INDEX IF NOT EXISTS idx_research_gaps_status ON research_gaps(status);
    """)
    conn.commit()


def init_schema() -> None:
    """Public entry point to ensure research tables exist."""
    _init_research_schema(_get_conn())


# ── Research prompts ───────────────────────────────────────────────


RESEARCH_PROMPTS = [
    "Search for recent papers and findings on {topic}. What are the most significant results from the last 3 years?",
    "What are the key open problems in {topic}? What remains unsolved or poorly understood?",
    "What approaches from other fields could apply to {topic}? Look for cross-disciplinary connections.",
    "What are the contrarian or unconventional perspectives on {topic}? Where does the mainstream consensus have weaknesses?",
]

GAP_IDENTIFICATION_PROMPT = """Given these research findings on "{topic}", identify knowledge gaps — areas where we have partial information but lack connections or depth.

Existing findings:
{findings}

For each gap, state it as a specific research question. Output one question per line, prefixed with "GAP: ".

GAPS:"""

FOLLOWUP_PROMPT = """Based on existing research on "{topic}" and this specific gap:
{gap}

Provide a focused answer drawing on relevant literature, methods, and empirical results. Be specific and cite concrete examples where possible.

ANSWER:"""


# ── Core research operations ───────────────────────────────────────


def _store_finding(
    topic_id: int,
    content: str,
    finding_type: str = "finding",
    source_query: str = "",
    confidence: float = 0.5,
) -> int:
    """Store a research finding and return its ID."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO research_findings
           (topic_id, content, finding_type, source_query, confidence, created)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (topic_id, content, finding_type, source_query, confidence, now),
    )
    conn.commit()
    return cur.lastrowid


def _store_citation(finding_a: int, finding_b: int, relation: str = "supports") -> None:
    """Record a citation relationship between two findings."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO research_citations (finding_a, finding_b, relation, created) VALUES (?, ?, ?, ?)",
        (finding_a, finding_b, relation, now),
    )
    conn.commit()


def _store_gap(topic_id: int, question: str) -> int:
    """Store a knowledge gap question and return its ID."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO research_gaps (topic_id, question, created) VALUES (?, ?, ?)",
        (topic_id, question, now),
    )
    conn.commit()
    return cur.lastrowid


def _resolve_gap(gap_id: int, finding_id: int) -> None:
    """Mark a gap as resolved by a finding."""
    conn = _get_conn()
    conn.execute(
        "UPDATE research_gaps SET status = 'resolved', resolved_by = ? WHERE id = ?",
        (finding_id, gap_id),
    )
    conn.commit()


def _get_open_gaps(topic_id: int) -> list[dict]:
    """Return all open gaps for a research topic."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM research_gaps WHERE topic_id = ? AND status = 'open' ORDER BY created",
        (topic_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_findings(topic_id: int, limit: int = 50) -> list[dict]:
    """Return all findings for a research topic."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM research_findings WHERE topic_id = ? ORDER BY confidence DESC LIMIT ?",
        (topic_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _already_researched(topic_id: int, query: str) -> bool:
    """Check whether a query has already been used for this topic."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM research_findings WHERE topic_id = ? AND source_query = ?",
        (topic_id, query),
    ).fetchone()
    return row[0] > 0


def _build_context_from_knowledge(topic: str, project: str) -> str:
    """Pull existing knowledge about the topic from the memory store."""
    from .store import recall
    memories = recall(topic, n=5, project=project)
    if not memories:
        return ""

    lines = []
    for m in memories:
        lines.append(f"  [{m['source']}] {m['raw_text'][:200]}")
    return "\n".join(lines)


# ── Research execution ─────────────────────────────────────────────


def _execute_research_prompt(
    prompt: str,
    topic_id: int,
    project: str,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[int]:
    """Send a prompt to Gemma, store findings and entities. Returns finding IDs."""
    from .gemma import _call_ollama, is_available

    if not is_available():
        if on_log:
            on_log("gemma unavailable — skipping prompt")
        return []

    result = _call_ollama(prompt, timeout=120)
    if not result:
        return []

    # Split response into individual findings (by paragraph or bullet)
    findings = _split_findings(result)
    finding_ids = []

    for finding_text in findings:
        if len(finding_text.strip()) < 20:
            continue

        finding_id = _store_finding(
            topic_id=topic_id,
            content=finding_text.strip(),
            finding_type="finding",
            source_query=prompt[:200],
            confidence=0.6,
        )
        finding_ids.append(finding_id)

        # Store as memory for future retrieval
        push_memory(
            raw_text=finding_text.strip(),
            source="research",
            tm_label="research_finding",
            regime_tag="research",
            aif_confidence=0.6,
            project=project,
        )

        # Extract and link entities
        try:
            from .entities import extract_entities
            from .store import ensure_entity, link_memory_entity
            entities = extract_entities(finding_text, source="research")
            for ent in entities:
                eid = ensure_entity(ent)
                # Entity linking happens through the memory store
        except Exception:
            pass

    if on_log and finding_ids:
        on_log(f"extracted {len(finding_ids)} findings")

    return finding_ids


def _split_findings(text: str) -> list[str]:
    """Split research output into individual finding segments."""
    import re

    # Split on bullet points, numbered lists, or double newlines
    segments = re.split(r'\n\s*(?:[-*•]\s+|\d+[.)]\s+|\n)', text)
    results = []
    for seg in segments:
        seg = seg.strip()
        if len(seg) >= 20:
            results.append(seg)

    # If no splits found, return the whole text as one finding
    if not results and len(text.strip()) >= 20:
        results.append(text.strip())

    return results


def _identify_gaps(topic_id: int, topic: str) -> list[str]:
    """Analyze findings and identify knowledge gaps."""
    from .gemma import _call_ollama, is_available

    findings = _get_findings(topic_id)
    if not findings or not is_available():
        return []

    findings_text = "\n".join(
        f"  - {f['content'][:200]}" for f in findings[:15]
    )

    prompt = GAP_IDENTIFICATION_PROMPT.format(
        topic=topic,
        findings=findings_text,
    )

    result = _call_ollama(prompt, timeout=60)
    if not result:
        return []

    gaps = []
    for line in result.split("\n"):
        line = line.strip()
        if line.upper().startswith("GAP:"):
            question = line[4:].strip()
            if len(question) >= 10:
                gaps.append(question)
        elif line.startswith("- ") and "?" in line:
            question = line[2:].strip()
            if len(question) >= 10:
                gaps.append(question)

    return gaps


def _build_citation_links(topic_id: int, new_finding_ids: list[int]) -> None:
    """Build citation relationships between new findings and existing ones.

    Uses vector similarity to detect when findings support or relate to
    each other.
    """
    from .hdc import encode_text, similarity

    findings = _get_findings(topic_id)
    if len(findings) < 2:
        return

    # Index existing findings by ID
    finding_map = {f["id"]: f for f in findings}

    for new_id in new_finding_ids:
        if new_id not in finding_map:
            continue

        new_text = finding_map[new_id]["content"]
        new_vec = encode_text(new_text)

        for other in findings:
            if other["id"] == new_id:
                continue

            other_vec = encode_text(other["content"])
            sim = similarity(new_vec, other_vec)

            if sim > 0.5:
                # Determine relationship type
                relation = "supports" if sim > 0.65 else "related"
                _store_citation(new_id, other["id"], relation)


# ── Main research loop ─────────────────────────────────────────────


def start_research(
    topic: str,
    project: str,
    turn_budget: int = 10,
    on_log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Execute a full research loop on a topic.

    Phases:
    1. Initial broad research with structured prompts
    2. Gap identification from gathered findings
    3. Targeted research to fill gaps
    4. Synthesis across findings
    5. Repeat until no gaps or budget exhausted

    Returns a summary dict with topic_id, findings count, gaps found, and status.
    """
    init_schema()
    log = on_log or (lambda s: None)

    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO research_topics
           (project, topic, status, turn_budget, created, updated)
           VALUES (?, ?, 'active', ?, ?, ?)""",
        (project, topic, turn_budget, now, now),
    )
    conn.commit()
    topic_id = cur.lastrowid

    log(f"research started: {topic}")

    # Gather existing context
    existing_context = _build_context_from_knowledge(topic, project)
    context_prefix = ""
    if existing_context:
        context_prefix = f"Existing knowledge:\n{existing_context}\n\n"
        log("loaded existing knowledge context")

    turns_used = 0
    total_finding_ids = []

    # Phase 1: Initial broad research
    for prompt_template in RESEARCH_PROMPTS:
        if turns_used >= turn_budget:
            break

        prompt = context_prefix + prompt_template.format(topic=topic)
        if _already_researched(topic_id, prompt[:200]):
            continue

        log(f"researching: {prompt_template[:60].format(topic=topic[:30])}")
        finding_ids = _execute_research_prompt(prompt, topic_id, project, on_log=log)
        total_finding_ids.extend(finding_ids)

        # Build citation links for new findings
        if finding_ids:
            _build_citation_links(topic_id, finding_ids)

        turns_used += 1

        # Update topic metadata
        conn.execute(
            "UPDATE research_topics SET turns_used = ?, findings_count = ?, updated = ? WHERE id = ?",
            (turns_used, len(total_finding_ids), datetime.now(timezone.utc).isoformat(), topic_id),
        )
        conn.commit()

    # Phase 2: Gap identification and targeted research
    iteration = 0
    max_iterations = 3

    while turns_used < turn_budget and iteration < max_iterations:
        iteration += 1
        log(f"gap analysis iteration {iteration}")

        gaps = _identify_gaps(topic_id, topic)
        if not gaps:
            log("no gaps identified — research complete")
            break

        # Store gaps
        gap_records = []
        for question in gaps:
            gap_id = _store_gap(topic_id, question)
            gap_records.append({"id": gap_id, "question": question})

        conn.execute(
            "UPDATE research_topics SET gaps_count = ? WHERE id = ?",
            (len(gaps), topic_id),
        )
        conn.commit()

        log(f"identified {len(gaps)} gaps")

        # Research each gap
        for gap in gap_records:
            if turns_used >= turn_budget:
                break

            prompt = FOLLOWUP_PROMPT.format(topic=topic, gap=gap["question"])
            log(f"filling gap: {gap['question'][:60]}")
            finding_ids = _execute_research_prompt(prompt, topic_id, project, on_log=log)
            total_finding_ids.extend(finding_ids)

            if finding_ids:
                _build_citation_links(topic_id, finding_ids)
                _resolve_gap(gap["id"], finding_ids[0])

            turns_used += 1

        # Run synthesis on accumulated findings
        try:
            from .synthesis import run_synthesis
            syntheses = run_synthesis(project, min_shared=1)
            if syntheses:
                conn.execute(
                    "UPDATE research_topics SET synthesis_depth = synthesis_depth + 1 WHERE id = ?",
                    (topic_id,),
                )
                conn.commit()
                log(f"synthesized {len(syntheses)} cross-domain insights")
        except Exception:
            pass

    # Finalize
    status = "complete" if turns_used < turn_budget else "budget_exhausted"
    conn.execute(
        "UPDATE research_topics SET status = ?, turns_used = ?, findings_count = ?, updated = ? WHERE id = ?",
        (status, turns_used, len(total_finding_ids), datetime.now(timezone.utc).isoformat(), topic_id),
    )
    conn.commit()

    log(f"research {status}: {len(total_finding_ids)} findings, {turns_used} turns")

    return {
        "topic_id": topic_id,
        "topic": topic,
        "status": status,
        "findings": len(total_finding_ids),
        "turns_used": turns_used,
        "gaps_remaining": len(_get_open_gaps(topic_id)),
    }


# ── Status and frontier ───────────────────────────────────────────


def research_status(project: str) -> list[dict]:
    """Return status of all research topics for a project."""
    init_schema()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, topic, status, turns_used, turn_budget,
                  findings_count, gaps_count, synthesis_depth, created, updated
           FROM research_topics
           WHERE project = ?
           ORDER BY updated DESC""",
        (project,),
    ).fetchall()
    return [dict(r) for r in rows]


def research_frontier(project: str) -> dict:
    """Return the current research frontier: open gaps and suggested next questions.

    The frontier represents the boundary of what is known — the set of
    questions most worth investigating next.
    """
    init_schema()
    conn = _get_conn()

    # Get all active topics
    topics = conn.execute(
        "SELECT id, topic FROM research_topics WHERE project = ? AND status IN ('active', 'complete')",
        (project,),
    ).fetchall()

    open_gaps = []
    for topic in topics:
        gaps = _get_open_gaps(topic["id"])
        for gap in gaps:
            open_gaps.append({
                "topic": topic["topic"],
                "question": gap["question"],
                "gap_id": gap["id"],
            })

    # Get recent findings to suggest next directions
    recent_findings = conn.execute(
        """SELECT rf.content, rt.topic
           FROM research_findings rf
           JOIN research_topics rt ON rf.topic_id = rt.id
           WHERE rt.project = ?
           ORDER BY rf.created DESC
           LIMIT 10""",
        (project,),
    ).fetchall()

    return {
        "open_gaps": open_gaps,
        "active_topics": [dict(t) for t in topics],
        "recent_findings": [dict(f) for f in recent_findings],
        "total_gaps": len(open_gaps),
    }
