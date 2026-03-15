"""Deep research mode — autonomous topic investigation with citation graphs.

Formulates search queries from a topic and existing knowledge, sends
structured research prompts to Gemma, extracts entities and findings,
builds a citation-like graph of supporting/contradicting relationships,
and iteratively identifies knowledge gaps to drive further investigation.

The research loop cycles through: research -> synthesize -> identify gaps ->
research more -> synthesize deeper, stopping when no gaps remain or the
turn budget is exhausted.

v2: Iterative deepening, adversarial research, quantitative extraction,
    source quality scoring, temporal awareness.
"""

import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import push_memory, DB_DIR, DB_PATH


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
            depth_level INTEGER DEFAULT 0,
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
            source_quality REAL DEFAULT 0.5,
            quantitative_data TEXT,
            published_date TEXT,
            contradicts_finding_id INTEGER,
            depth_level INTEGER DEFAULT 0,
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
            priority REAL DEFAULT 0.5,
            resolved_by INTEGER REFERENCES research_findings(id),
            created TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_research_topics_project ON research_topics(project);
        CREATE INDEX IF NOT EXISTS idx_research_findings_topic ON research_findings(topic_id);
        CREATE INDEX IF NOT EXISTS idx_research_findings_type ON research_findings(finding_type);
        CREATE INDEX IF NOT EXISTS idx_research_gaps_topic ON research_gaps(topic_id);
        CREATE INDEX IF NOT EXISTS idx_research_gaps_status ON research_gaps(status);
        CREATE INDEX IF NOT EXISTS idx_research_gaps_priority ON research_gaps(priority);
    """)
    conn.commit()


def init_schema() -> None:
    """Public entry point to ensure research tables exist."""
    _init_research_schema(_get_conn())


# ── Research prompts — multi-depth ─────────────────────────────────


# Depth 0: Broad survey — what exists?
DEPTH_0_PROMPTS = [
    "Provide a comprehensive survey of {topic}. What are the major areas, key findings, and the current state of knowledge? Focus on breadth.",
    "What are the key open problems in {topic}? What remains unsolved or poorly understood? What are the most active areas of investigation?",
    "What approaches from other fields could apply to {topic}? Look for cross-disciplinary connections, analogies, and transferable methods.",
    "What are the contrarian or unconventional perspectives on {topic}? Where does the mainstream consensus have weaknesses? What do skeptics argue?",
]

# Depth 1: Mechanistic — how does it work?
DEPTH_1_PROMPTS = [
    "What are the fundamental mechanisms underlying {topic}? Describe the causal chain, not just correlations. What drives what?",
    "What are the competing theories about {topic}? For each theory: what evidence supports it, what evidence contradicts it, and what is its explanatory power?",
    "What are the mathematical models or quantitative frameworks used in {topic}? Include specific equations, parameters, and their empirical validation.",
    "What experimental results are most critical to understanding {topic}? Include sample sizes, effect sizes, p-values, and replication status where available.",
]

# Depth 2: Adversarial — what's wrong with what we think we know?
DEPTH_2_PROMPTS = [
    "For the strongest claims about {topic}: what are the counter-arguments? What studies contradict the consensus? What are the failure modes?",
    "What biases might affect research on {topic}? Publication bias, funding bias, methodological bias, confirmation bias. How might these distort findings?",
    "What are the boundary conditions of the main findings on {topic}? Under what conditions do they break down? What populations or contexts are underrepresented?",
    "What retractions, failed replications, or major corrections have occurred in {topic}? What did we previously believe that turned out to be wrong?",
]

# Depth 3: Synthesis — what does it all mean together?
DEPTH_3_PROMPTS = [
    "Given everything known about {topic}, what are the most promising unexplored combinations of approaches? What synthesis across sub-fields hasn't been tried?",
    "What would a unified theory of {topic} look like? What are the key tensions between sub-fields that a unified theory would need to resolve?",
    "What are the most impactful practical applications that follow from current understanding of {topic}? What's blocking implementation?",
]

ALL_DEPTH_PROMPTS = [DEPTH_0_PROMPTS, DEPTH_1_PROMPTS, DEPTH_2_PROMPTS, DEPTH_3_PROMPTS]

GAP_IDENTIFICATION_PROMPT = """Given these research findings on "{topic}" at depth level {depth}, identify knowledge gaps — areas where we have partial information but lack connections or depth.

Existing findings:
{findings}

For each gap, state it as a specific research question. Rate its priority (high/medium/low) based on how much answering it would advance understanding.
Output one question per line, prefixed with "GAP [priority]: ".

GAPS:"""

FOLLOWUP_PROMPT = """Based on existing research on "{topic}" and this specific gap:
{gap}

Provide a focused answer drawing on relevant literature, methods, and empirical results. Be specific:
- Cite concrete examples, not generalities
- Include quantitative data where available (sample sizes, effect sizes, percentages)
- Note the publication date/recency of findings
- Rate source quality (meta-analysis > RCT > observational > case study > expert opinion > blog)
- Flag any contradictions with existing findings

ANSWER:"""

ADVERSARIAL_PROMPT = """You are a rigorous scientific critic. For this finding about "{topic}":

FINDING: {finding}

Provide a thorough adversarial analysis:
1. What evidence would CONTRADICT this finding?
2. What alternative explanations exist?
3. What methodological weaknesses could invalidate it?
4. What's the replication status of the underlying research?
5. Who disagrees and why?

Be specific and evidence-based, not just skeptical for the sake of it.

CRITIQUE:"""

QUANTITATIVE_EXTRACTION_PROMPT = """Extract ALL quantitative data from this research text. For each data point, format as:
STAT: [metric] = [value] (source: [where], date: [when], quality: [high/medium/low])

Text:
{text}

STATISTICS:"""


# ── Source quality scoring ─────────────────────────────────────────


SOURCE_QUALITY_KEYWORDS = {
    # High quality (0.8-1.0)
    "meta-analysis": 0.95, "systematic review": 0.93, "cochrane": 0.95,
    "randomized controlled trial": 0.90, "rct": 0.90, "phase iii": 0.88,
    "nature": 0.90, "science": 0.90, "lancet": 0.90, "nejm": 0.92,
    "peer-reviewed": 0.80, "replicated": 0.85,
    # Medium quality (0.5-0.8)
    "observational study": 0.65, "cohort study": 0.70, "case-control": 0.65,
    "survey": 0.55, "preprint": 0.50, "arxiv": 0.55, "conference paper": 0.60,
    # Low quality (0.1-0.5)
    "blog": 0.25, "opinion": 0.30, "anecdotal": 0.15, "case report": 0.40,
    "press release": 0.20, "news article": 0.25, "wikipedia": 0.35,
    "social media": 0.10, "reddit": 0.15, "forum": 0.15,
}


def _score_source_quality(text: str) -> float:
    """Score the quality of a finding based on source indicators in the text."""
    lower = text.lower()
    best_score = 0.5  # default

    for keyword, score in SOURCE_QUALITY_KEYWORDS.items():
        if keyword in lower:
            best_score = max(best_score, score)

    # Quantitative data boosts quality
    if re.search(r'\b\d+\.?\d*\s*%', text) or re.search(r'p\s*[<>=]\s*0\.\d+', text):
        best_score = min(1.0, best_score + 0.1)

    # Sample size mentions boost quality
    if re.search(r'n\s*=\s*\d+', text, re.I):
        best_score = min(1.0, best_score + 0.05)

    return best_score


def _extract_quantitative_data(text: str) -> str:
    """Extract quantitative claims from finding text."""
    patterns = [
        r'\d+\.?\d*\s*%',                          # percentages
        r'p\s*[<>=]\s*0\.\d+',                     # p-values
        r'n\s*=\s*\d+',                             # sample sizes
        r'\d+\.?\d*\s*(?:mg|kg|ml|μg|ng|mmol)',    # dosages
        r'(?:OR|RR|HR|CI)\s*[=:]\s*\d+\.?\d*',    # odds/risk ratios
        r'\d+\.?\d*\s*(?:fold|x)\s',              # fold changes
        r'effect size\s*[=:]\s*\d+\.?\d*',         # effect sizes
    ]

    found = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.I)
        found.extend(matches)

    return "; ".join(found[:10]) if found else ""


def _extract_temporal_info(text: str) -> str:
    """Extract publication dates or temporal references from text."""
    patterns = [
        r'(?:published|reported|found|showed|demonstrated)\s+in\s+(\d{4})',
        r'\((\d{4})\)',
        r'(?:20[12]\d|199\d)',
    ]

    years = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            if 1990 <= int(m) <= 2030:
                years.add(m)

    if years:
        return max(years)  # most recent year
    return ""


# ── Core research operations ───────────────────────────────────────


def _store_finding(
    topic_id: int,
    content: str,
    finding_type: str = "finding",
    source_query: str = "",
    confidence: float = 0.5,
    source_quality: float = 0.5,
    quantitative_data: str = "",
    published_date: str = "",
    contradicts_finding_id: int | None = None,
    depth_level: int = 0,
) -> int:
    """Store a research finding and return its ID."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO research_findings
           (topic_id, content, finding_type, source_query, confidence,
            source_quality, quantitative_data, published_date,
            contradicts_finding_id, depth_level, created)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (topic_id, content, finding_type, source_query, confidence,
         source_quality, quantitative_data, published_date,
         contradicts_finding_id, depth_level, now),
    )
    conn.commit()
    return cur.lastrowid or 0


def _store_citation(finding_a: int, finding_b: int, relation: str = "supports") -> None:
    """Record a citation relationship between two findings."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO research_citations (finding_a, finding_b, relation, created) VALUES (?, ?, ?, ?)",
        (finding_a, finding_b, relation, now),
    )
    conn.commit()


def _store_gap(topic_id: int, question: str, priority: float = 0.5) -> int:
    """Store a knowledge gap question and return its ID."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO research_gaps (topic_id, question, priority, created) VALUES (?, ?, ?, ?)",
        (topic_id, question, priority, now),
    )
    conn.commit()
    return cur.lastrowid or 0


def _resolve_gap(gap_id: int, finding_id: int) -> None:
    """Mark a gap as resolved by a finding."""
    conn = _get_conn()
    conn.execute(
        "UPDATE research_gaps SET status = 'resolved', resolved_by = ? WHERE id = ?",
        (finding_id, gap_id),
    )
    conn.commit()


def _get_open_gaps(topic_id: int) -> list[dict]:
    """Return all open gaps for a research topic, ordered by priority."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM research_gaps WHERE topic_id = ? AND status = 'open' ORDER BY priority DESC, created",
        (topic_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_findings(topic_id: int, limit: int = 50, depth_level: int | None = None) -> list[dict]:
    """Return findings for a research topic, optionally filtered by depth."""
    conn = _get_conn()
    if depth_level is not None:
        rows = conn.execute(
            "SELECT * FROM research_findings WHERE topic_id = ? AND depth_level = ? ORDER BY confidence DESC LIMIT ?",
            (topic_id, depth_level, limit),
        ).fetchall()
    else:
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
    depth_level: int = 0,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[int]:
    """Send a prompt to Gemma, store findings with quality scoring. Returns finding IDs."""
    from .gemma import _call_ollama, is_available

    if not is_available():
        if on_log:
            on_log("gemma unavailable — skipping prompt")
        return []

    result = _call_ollama(prompt, timeout=120)
    if not result:
        return []

    # Split response into individual findings
    findings = _split_findings(result)
    finding_ids = []
    quality = 0.0

    for finding_text in findings:
        if len(finding_text.strip()) < 20:
            continue

        # Score source quality
        quality = _score_source_quality(finding_text)

        # Extract quantitative data
        quant_data = _extract_quantitative_data(finding_text)

        # Extract temporal info
        pub_date = _extract_temporal_info(finding_text)

        # Confidence is quality-weighted
        confidence = min(1.0, 0.4 + quality * 0.4 + (0.1 if quant_data else 0.0))

        finding_id = _store_finding(
            topic_id=topic_id,
            content=finding_text.strip(),
            finding_type="finding",
            source_query=prompt[:200],
            confidence=confidence,
            source_quality=quality,
            quantitative_data=quant_data,
            published_date=pub_date,
            depth_level=depth_level,
        )
        finding_ids.append(finding_id)

        # Store as memory for future retrieval
        push_memory(
            raw_text=finding_text.strip(),
            source="research",
            tm_label="research_finding",
            regime_tag="research",
            aif_confidence=confidence,
            project=project,
        )

        # Extract and link entities
        try:
            from .entities import extract_entities
            from .store import ensure_entity
            entities = extract_entities(finding_text, source="research")
            for ent in entities:
                ensure_entity(ent)
        except Exception:
            pass

    if on_log and finding_ids:
        avg_q = quality if finding_ids else 0.0
        on_log(f"extracted {len(finding_ids)} findings (depth {depth_level}, avg quality {avg_q:.2f})")

    return finding_ids


def _split_findings(text: str) -> list[str]:
    """Split research output into individual finding segments."""
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


def _identify_gaps(topic_id: int, topic: str, depth_level: int = 0) -> list[tuple[str, float]]:
    """Analyze findings and identify knowledge gaps with priorities.

    Returns list of (question, priority) tuples.
    """
    from .gemma import _call_ollama, is_available

    findings = _get_findings(topic_id)
    if not findings or not is_available():
        return []

    findings_text = "\n".join(
        f"  - [{f.get('source_quality', 0.5):.1f}] {f['content'][:200]}" for f in findings[:15]
    )

    prompt = GAP_IDENTIFICATION_PROMPT.format(
        topic=topic,
        depth=depth_level,
        findings=findings_text,
    )

    result = _call_ollama(prompt, timeout=60)
    if not result:
        return []

    gaps = []
    for line in result.split("\n"):
        line = line.strip()
        if line.upper().startswith("GAP"):
            # Parse priority
            priority = 0.5
            if "[HIGH]" in line.upper() or "HIGH" in line.upper().split(":")[0]:
                priority = 0.9
            elif "[LOW]" in line.upper() or "LOW" in line.upper().split(":")[0]:
                priority = 0.3

            # Extract the question
            # Remove "GAP [priority]: " prefix
            question = re.sub(r'^GAP\s*\[?\w*\]?\s*:\s*', '', line, flags=re.I).strip()
            if len(question) >= 10:
                gaps.append((question, priority))
        elif line.startswith("- ") and "?" in line:
            question = line[2:].strip()
            if len(question) >= 10:
                gaps.append((question, 0.5))

    return gaps


def _run_adversarial_check(
    topic_id: int,
    topic: str,
    project: str,
    on_log: Optional[Callable[[str], None]] = None,
) -> list[int]:
    """Run adversarial analysis on the highest-confidence findings.

    For each top finding, searches for counter-arguments and contradictions.
    Returns IDs of new adversarial findings.
    """
    from .gemma import _call_ollama, is_available

    if not is_available():
        return []

    findings = _get_findings(topic_id, limit=5)
    if not findings:
        return []

    log = on_log or (lambda s: None)
    adversarial_ids = []

    for finding in findings[:3]:  # Top 3 findings
        prompt = ADVERSARIAL_PROMPT.format(
            topic=topic,
            finding=finding["content"][:500],
        )

        result = _call_ollama(prompt, timeout=90)
        if not result or len(result.strip()) < 30:
            continue

        # Store the adversarial finding
        adv_id = _store_finding(
            topic_id=topic_id,
            content=result.strip(),
            finding_type="adversarial",
            source_query=f"adversarial check on finding {finding['id']}",
            confidence=0.6,
            source_quality=0.6,
            contradicts_finding_id=finding["id"],
            depth_level=2,
        )
        adversarial_ids.append(adv_id)

        # Create a contradiction citation
        _store_citation(adv_id, finding["id"], "contradicts")

        push_memory(
            raw_text=f"[ADVERSARIAL] {result.strip()[:500]}",
            source="research",
            tm_label="adversarial_finding",
            regime_tag="research",
            aif_confidence=0.6,
            project=project,
        )

    if adversarial_ids:
        log(f"adversarial analysis: {len(adversarial_ids)} counter-arguments found")

    return adversarial_ids


def _build_citation_links(topic_id: int, new_finding_ids: list[int]) -> None:
    """Build citation relationships between new findings and existing ones."""
    from .hdc import encode_text, similarity

    findings = _get_findings(topic_id)
    if len(findings) < 2:
        return

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
                # Check for contradiction signals
                has_contradiction = any(
                    word in new_text.lower()
                    for word in ["however", "contradicts", "contrary", "unlike",
                                 "fails to", "disproves", "refutes", "incorrect"]
                )

                if has_contradiction and sim > 0.4:
                    relation = "contradicts"
                elif sim > 0.65:
                    relation = "supports"
                else:
                    relation = "related"

                _store_citation(new_id, other["id"], relation)


# ── Main research loop with iterative deepening ───────────────────


def start_research(
    topic: str,
    project: str,
    turn_budget: int = 10,
    max_depth: int = 3,
    adversarial: bool = True,
    on_log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Execute a full research loop on a topic with iterative deepening.

    Phases per depth level:
    1. Structured prompts at current depth
    2. Gap identification from gathered findings
    3. Targeted research to fill gaps (prioritized)
    4. Adversarial analysis (optional)
    5. Synthesis across findings
    6. Deepen if gaps remain and budget allows

    Returns a summary dict with topic_id, findings count, gaps found, and status.
    """
    init_schema()
    log = on_log or (lambda s: None)

    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO research_topics
           (project, topic, status, turn_budget, depth_level, created, updated)
           VALUES (?, ?, 'active', ?, 0, ?, ?)""",
        (project, topic, turn_budget, now, now),
    )
    conn.commit()
    topic_id = cur.lastrowid
    assert topic_id is not None, "INSERT failed to produce a row ID"

    log(f"research started: {topic}")

    # Gather existing context
    existing_context = _build_context_from_knowledge(topic, project)
    context_prefix = ""
    if existing_context:
        context_prefix = f"Existing knowledge:\n{existing_context}\n\n"
        log("loaded existing knowledge context")

    turns_used = 0
    total_finding_ids = []

    # ── Iterative deepening across depth levels ──
    effective_max_depth = min(max_depth, len(ALL_DEPTH_PROMPTS) - 1)
    depth = 0

    for depth in range(effective_max_depth + 1):
        if turns_used >= turn_budget:
            break

        log(f"=== depth {depth}: {['survey', 'mechanistic', 'adversarial', 'synthesis'][min(depth, 3)]} ===")

        # Update topic depth
        conn.execute(
            "UPDATE research_topics SET depth_level = ? WHERE id = ?",
            (depth, topic_id),
        )
        conn.commit()

        # Phase 1: Run prompts at this depth level
        prompts = ALL_DEPTH_PROMPTS[depth] if depth < len(ALL_DEPTH_PROMPTS) else DEPTH_3_PROMPTS

        for prompt_template in prompts:
            if turns_used >= turn_budget:
                break

            prompt = context_prefix + prompt_template.format(topic=topic)
            if _already_researched(topic_id, prompt[:200]):
                continue

            log(f"researching: {prompt_template[:60].format(topic=topic[:30])}")
            finding_ids = _execute_research_prompt(
                prompt, topic_id, project,
                depth_level=depth, on_log=log,
            )
            total_finding_ids.extend(finding_ids)

            if finding_ids:
                _build_citation_links(topic_id, finding_ids)

            turns_used += 1
            _update_topic_meta(conn, topic_id, turns_used, len(total_finding_ids))

        # Phase 2: Gap identification and targeted research
        gaps_with_priority = _identify_gaps(topic_id, topic, depth_level=depth)
        if gaps_with_priority:
            # Sort by priority — fill high-priority gaps first
            gaps_with_priority.sort(key=lambda x: x[1], reverse=True)

            gap_records = []
            for question, priority in gaps_with_priority:
                gap_id = _store_gap(topic_id, question, priority=priority)
                gap_records.append({"id": gap_id, "question": question, "priority": priority})

            conn.execute(
                "UPDATE research_topics SET gaps_count = ? WHERE id = ?",
                (len(gaps_with_priority), topic_id),
            )
            conn.commit()

            log(f"identified {len(gaps_with_priority)} gaps ({sum(1 for _, p in gaps_with_priority if p > 0.7)} high priority)")

            # Fill gaps in priority order
            for gap in gap_records:
                if turns_used >= turn_budget:
                    break

                prompt = FOLLOWUP_PROMPT.format(topic=topic, gap=gap["question"])
                log(f"filling gap [{gap['priority']:.1f}]: {gap['question'][:60]}")
                finding_ids = _execute_research_prompt(
                    prompt, topic_id, project,
                    depth_level=depth, on_log=log,
                )
                total_finding_ids.extend(finding_ids)

                if finding_ids:
                    _build_citation_links(topic_id, finding_ids)
                    _resolve_gap(gap["id"], finding_ids[0])

                turns_used += 1

        # Phase 3: Adversarial check (at depth >= 1)
        if adversarial and depth >= 1 and turns_used < turn_budget:
            log("running adversarial analysis")
            adv_ids = _run_adversarial_check(topic_id, topic, project, on_log=log)
            total_finding_ids.extend(adv_ids)
            turns_used += 1

        # Phase 4: Synthesis
        if turns_used < turn_budget:
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

        # Check if more depth is needed
        remaining_gaps = _get_open_gaps(topic_id)
        if not remaining_gaps:
            log(f"no remaining gaps at depth {depth} — research converged")
            break

    # ── Finalize ──
    status = "complete" if turns_used < turn_budget else "budget_exhausted"

    # Compute research quality metrics
    all_findings = _get_findings(topic_id)
    avg_quality = sum(f.get("source_quality", 0.5) for f in all_findings) / max(len(all_findings), 1)
    quant_count = sum(1 for f in all_findings if f.get("quantitative_data"))
    adversarial_count = sum(1 for f in all_findings if f.get("finding_type") == "adversarial")

    conn.execute(
        "UPDATE research_topics SET status = ?, turns_used = ?, findings_count = ?, updated = ? WHERE id = ?",
        (status, turns_used, len(total_finding_ids), datetime.now(timezone.utc).isoformat(), topic_id),
    )
    conn.commit()

    log(f"research {status}: {len(total_finding_ids)} findings, {turns_used} turns, avg quality {avg_quality:.2f}")

    return {
        "topic_id": topic_id,
        "topic": topic,
        "status": status,
        "findings": len(total_finding_ids),
        "turns_used": turns_used,
        "max_depth_reached": min(effective_max_depth, depth if turns_used > 0 else 0),
        "gaps_remaining": len(_get_open_gaps(topic_id)),
        "avg_source_quality": round(avg_quality, 3),
        "quantitative_findings": quant_count,
        "adversarial_findings": adversarial_count,
    }


def _update_topic_meta(conn: sqlite3.Connection, topic_id: int, turns_used: int, findings_count: int) -> None:
    """Update topic metadata counters."""
    conn.execute(
        "UPDATE research_topics SET turns_used = ?, findings_count = ?, updated = ? WHERE id = ?",
        (turns_used, findings_count, datetime.now(timezone.utc).isoformat(), topic_id),
    )
    conn.commit()


# ── Status and frontier ───────────────────────────────────────────


def research_status(project: str) -> list[dict]:
    """Return status of all research topics for a project."""
    init_schema()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, topic, status, turns_used, turn_budget,
                  findings_count, gaps_count, synthesis_depth, depth_level, created, updated
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
                "priority": gap.get("priority", 0.5),
                "gap_id": gap["id"],
            })

    # Sort by priority
    open_gaps.sort(key=lambda g: g.get("priority", 0.5), reverse=True)

    # Get recent findings to suggest next directions
    recent_findings = conn.execute(
        """SELECT rf.content, rf.source_quality, rf.finding_type, rt.topic
           FROM research_findings rf
           JOIN research_topics rt ON rf.topic_id = rt.id
           WHERE rt.project = ?
           ORDER BY rf.created DESC
           LIMIT 10""",
        (project,),
    ).fetchall()

    # Get contradiction pairs
    contradictions = conn.execute(
        """SELECT
               rf1.content AS finding_1,
               rf2.content AS finding_2,
               rc.relation
           FROM research_citations rc
           JOIN research_findings rf1 ON rc.finding_a = rf1.id
           JOIN research_findings rf2 ON rc.finding_b = rf2.id
           JOIN research_topics rt ON rf1.topic_id = rt.id
           WHERE rt.project = ? AND rc.relation = 'contradicts'
           ORDER BY rc.created DESC
           LIMIT 5""",
        (project,),
    ).fetchall()

    return {
        "open_gaps": open_gaps,
        "active_topics": [dict(t) for t in topics],
        "recent_findings": [dict(f) for f in recent_findings],
        "contradictions": [dict(c) for c in contradictions],
        "total_gaps": len(open_gaps),
        "high_priority_gaps": sum(1 for g in open_gaps if g.get("priority", 0) > 0.7),
    }
