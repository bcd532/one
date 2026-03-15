"""Synthesis engine — automatic insight generation from the knowledge graph.

Scans the entity graph for cross-domain connections between unrelated
concepts. When entities from different contexts co-occur in shared memories,
generates hypotheses about the underlying patterns using Gemma. Supports
recursive deep synthesis where existing insights seed further connections,
forming a directed acyclic graph of progressively deeper understanding.

v2: Non-obvious connection detection, hypothesis testing, confidence
    calibration, contradiction detection, novelty scoring.
"""

import re
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
            novelty_score REAL DEFAULT 0.5,
            tested INTEGER DEFAULT 0,
            test_result TEXT,
            parent_id INTEGER REFERENCES syntheses(id),
            depth INTEGER DEFAULT 0,
            created TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_syntheses_project ON syntheses(project);
        CREATE INDEX IF NOT EXISTS idx_syntheses_depth ON syntheses(depth);
        CREATE INDEX IF NOT EXISTS idx_syntheses_parent ON syntheses(parent_id);
        CREATE INDEX IF NOT EXISTS idx_syntheses_novelty ON syntheses(novelty_score);
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

    Prioritizes pairs from DIFFERENT types (cross-domain) over same-type pairs.
    Returns pairs ordered by cross-domain score descending.
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

    pairs = [dict(r) for r in rows]

    # Score pairs: cross-domain pairs get a bonus
    for pair in pairs:
        cross_domain_bonus = 2.0 if pair["type_a"] != pair["type_b"] else 1.0
        pair["cross_score"] = pair["shared_count"] * cross_domain_bonus

    pairs.sort(key=lambda p: p["cross_score"], reverse=True)
    return pairs


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


SYNTHESIS_PROMPT = """You are a knowledge synthesis engine. Given two entities connected through shared memories, generate a hypothesis about the UNDERLYING pattern connecting them.

CRITICAL: Do NOT simply restate that they co-occur. Find the CAUSAL or STRUCTURAL relationship.

Bad example: "A and B both appear in coding contexts" (trivial restatement)
Good example: "A's mechanism of action on pathway X could explain B's unexpected efficacy when combined with Y, suggesting a shared regulatory target" (actual insight)

Entity A: {entity_a} (type: {type_a})
Entity B: {entity_b} (type: {type_b})
Shared memories ({shared_count}):
{memory_context}

Generate a single precise hypothesis (2-3 sentences) about what CAUSAL or STRUCTURAL pattern connects these two concepts. Focus on mechanisms, not correlations. The hypothesis should be TESTABLE.

HYPOTHESIS:"""


DEEP_SYNTHESIS_PROMPT = """You are a deep knowledge synthesis engine operating at depth {depth}. Previous synthesis insights are provided below. Generate a higher-order hypothesis that connects or extends these existing insights.

CRITICAL: Your meta-hypothesis must reveal something NOT OBVIOUS from any individual insight below. Look for:
- Hidden unifying principles
- Paradoxes that resolve at a higher level of abstraction
- Emergent properties that only appear when insights are combined
- Transferable structural patterns across domains

Previous insights:
{prior_insights}

Generate a single precise meta-hypothesis (2-3 sentences) that identifies a deeper pattern across the prior insights. This should be surprising and testable.

META-HYPOTHESIS:"""


HYPOTHESIS_TEST_PROMPT = """Given this hypothesis:
{hypothesis}

Search for evidence that either SUPPORTS or CONTRADICTS this hypothesis. Consider:
1. Has this exact combination been studied?
2. Are there related findings that partially test this?
3. What would we expect to observe if the hypothesis is true?
4. What evidence would falsify it?

Provide a verdict: SUPPORTED (evidence exists), CONTRADICTED (counter-evidence exists), UNTESTED (novel hypothesis), or PARTIALLY_SUPPORTED (mixed evidence).

Include specific evidence for your verdict.

VERDICT:"""


CONTRADICTION_DETECTION_PROMPT = """Analyze these two findings for contradiction:

Finding 1: {finding_1}
Finding 2: {finding_2}

Are these findings contradictory? If so, explain the specific point of contradiction and which finding has stronger supporting evidence.

If not contradictory, explain how they can be reconciled.

ANALYSIS:"""


def _score_novelty(
    hypothesis: str,
    entity_a: str,
    entity_b: str,
    shared_memories: list[dict],
) -> float:
    """Score how novel/non-obvious a synthesis hypothesis is.

    High novelty = the hypothesis reveals something that couldn't be
    trivially inferred from just knowing the two entities co-occur.
    Low novelty = mere restatement of co-occurrence.
    """
    lower = hypothesis.lower()

    # Penalize trivial restatements
    trivial_phrases = [
        "co-occur", "both appear", "both relate to", "connected through",
        "associated with", "linked to", "related to each other",
        "both involve", "commonly found together", "frequently mentioned",
    ]
    trivial_count = sum(1 for phrase in trivial_phrases if phrase in lower)
    if trivial_count >= 2:
        return 0.1

    # Reward mechanistic language
    mechanistic_phrases = [
        "because", "mechanism", "causes", "enables", "prevents",
        "pathway", "suggests that", "explains why", "predicts",
        "implies", "if.*then", "structural", "causal", "emergent",
        "underlying", "drives", "mediates", "modulates",
    ]
    mechanism_count = sum(1 for phrase in mechanistic_phrases if re.search(phrase, lower))

    # Reward testability
    testable_phrases = [
        "could be tested", "would predict", "should show",
        "experiment", "measurable", "observable", "verify",
        "if this is true", "we would expect",
    ]
    testability = sum(1 for phrase in testable_phrases if phrase in lower)

    # Reward specificity (specific terms, numbers, names)
    specificity = min(0.3, len(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', hypothesis)) * 0.1)

    novelty = min(1.0, max(0.1,
        0.3 + mechanism_count * 0.15 + testability * 0.1 + specificity
        - trivial_count * 0.2
    ))

    return novelty


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


def _test_hypothesis(hypothesis: str, project: str) -> tuple[str, str]:
    """Test a hypothesis against existing knowledge.

    Returns (verdict, evidence) where verdict is one of:
    SUPPORTED, CONTRADICTED, UNTESTED, PARTIALLY_SUPPORTED
    """
    from .gemma import _call_ollama, is_available

    if not is_available():
        return "UNTESTED", "LLM unavailable for hypothesis testing"

    prompt = HYPOTHESIS_TEST_PROMPT.format(hypothesis=hypothesis)
    result = _call_ollama(prompt, timeout=60)

    if not result:
        return "UNTESTED", "No response from LLM"

    # Parse verdict
    upper = result.upper()
    if "SUPPORTED" in upper and "PARTIALLY" not in upper and "CONTRADICTED" not in upper:
        verdict = "SUPPORTED"
    elif "CONTRADICTED" in upper:
        verdict = "CONTRADICTED"
    elif "PARTIALLY" in upper:
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "UNTESTED"

    return verdict, result.strip()


def _detect_contradictions(project: str) -> list[dict]:
    """Scan synthesis results for contradictory hypotheses.

    Contradictions are where breakthroughs hide — two plausible-sounding
    hypotheses that can't both be true indicate a deeper truth.
    """
    from .hdc import encode_text, similarity

    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, hypothesis, entity_a, entity_b, confidence
           FROM syntheses
           WHERE project = ? AND depth = 0
           ORDER BY confidence DESC
           LIMIT 30""",
        (project,),
    ).fetchall()

    if len(rows) < 2:
        return []

    syntheses = [dict(r) for r in rows]
    contradictions = []

    for i in range(len(syntheses)):
        for j in range(i + 1, len(syntheses)):
            h1 = syntheses[i]["hypothesis"]
            h2 = syntheses[j]["hypothesis"]

            # Check for contradiction signals
            has_negation_overlap = _has_contradiction_signals(h1, h2)

            if has_negation_overlap:
                # Verify with vector similarity (contradictions often have high similarity with negation)
                vec1 = encode_text(h1)
                vec2 = encode_text(h2)
                sim = similarity(vec1, vec2)

                if sim > 0.3:  # Related enough to be contradictory
                    contradictions.append({
                        "synthesis_1": syntheses[i],
                        "synthesis_2": syntheses[j],
                        "similarity": sim,
                        "type": "potential_contradiction",
                    })

    return contradictions


def _has_contradiction_signals(text_a: str, text_b: str) -> bool:
    """Check if two texts contain contradiction signals."""
    negation_words = {"not", "no", "never", "without", "lack", "absence",
                      "prevents", "inhibits", "blocks", "contradicts",
                      "unlike", "contrary", "opposite", "instead"}

    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    # Check if one text contains negation of concepts in the other
    shared_content = words_a & words_b - {"the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "in", "to"}
    negation_in_a = negation_words & words_a
    negation_in_b = negation_words & words_b

    # Contradiction: shared concepts + asymmetric negation
    if shared_content and (negation_in_a ^ negation_in_b):  # XOR — one has negation, other doesn't
        return True

    return False


# ── Core synthesis operations ──────────────────────────────────────


def run_synthesis(
    project: str,
    min_shared: int = 1,
    max_pairs: int = 20,
    test_hypotheses: bool = True,
) -> list[dict]:
    """Scan the entity graph for cross-connections and generate hypotheses.

    Prioritizes cross-domain pairs (different entity types) for maximum
    insight novelty. Optionally tests hypotheses against existing knowledge.

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

        # Score novelty
        novelty = _score_novelty(hypothesis, entity_a, entity_b, shared)

        # Skip trivial syntheses
        if novelty < 0.2:
            continue

        # Compute confidence from shared memory count, AIF confidence, and novelty
        avg_conf = sum(m.get("aif_confidence", 0) for m in shared) / max(len(shared), 1)
        confidence = min(1.0, 0.3 + avg_conf * 0.2 + min(len(shared), 5) * 0.05 + novelty * 0.3)

        # Test hypothesis if enabled
        test_result = None
        tested = 0
        if test_hypotheses:
            verdict, evidence = _test_hypothesis(hypothesis, project)
            test_result = f"{verdict}: {evidence[:500]}"
            tested = 1

            # Adjust confidence based on test result
            if verdict == "SUPPORTED":
                confidence = min(1.0, confidence + 0.2)
            elif verdict == "CONTRADICTED":
                confidence = max(0.1, confidence - 0.3)
            elif verdict == "PARTIALLY_SUPPORTED":
                confidence = min(1.0, confidence + 0.05)

        # Store the synthesis record
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """INSERT INTO syntheses
               (project, entity_a, entity_b, hypothesis, confidence,
                novelty_score, tested, test_result, parent_id, depth, created)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?)""",
            (project, entity_a, entity_b, hypothesis, confidence,
             novelty, tested, test_result, now),
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
            "novelty": novelty,
            "tested": tested,
            "test_result": test_result,
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
        conn = _get_conn()
        prior_rows = conn.execute(
            """SELECT id, hypothesis, confidence, novelty_score FROM syntheses
               WHERE project = ? AND depth = ?
               ORDER BY confidence DESC
               LIMIT ?""",
            (project, current_depth - 1, max_per_level * 2),
        ).fetchall()

        if len(prior_rows) < 2:
            break

        # Filter to high-novelty insights for deeper synthesis
        prior_rows = [r for r in prior_rows if (r["novelty_score"] or 0) > 0.3]
        if len(prior_rows) < 2:
            break

        prior_insights = [r["hypothesis"] for r in prior_rows]
        prior_ids = [r["id"] for r in prior_rows]

        meta_hypothesis = _generate_deep_hypothesis(prior_insights, current_depth)
        if not meta_hypothesis:
            break

        # Score novelty of the meta-hypothesis
        novelty = _score_novelty(meta_hypothesis, "meta", "meta", [])

        # Average confidence of parent insights, decayed with novelty bonus
        avg_conf = sum(r["confidence"] for r in prior_rows) / len(prior_rows)
        confidence = min(1.0, avg_conf * 0.85 + novelty * 0.15)

        parent_id = prior_ids[0] if prior_ids else None
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """INSERT INTO syntheses
               (project, entity_a, entity_b, hypothesis, confidence,
                novelty_score, tested, test_result, parent_id, depth, created)
               VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?)""",
            (project, "meta", "meta", meta_hypothesis, confidence,
             novelty, parent_id, current_depth, now),
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
            "novelty": novelty,
            "depth": current_depth,
            "parent_id": parent_id,
        }
        all_results.append(result)

    # Detect contradictions across all syntheses
    contradictions = _detect_contradictions(project)
    if contradictions:
        for c in contradictions:
            push_memory(
                raw_text=f"[CONTRADICTION] {c['synthesis_1']['hypothesis'][:200]} VS {c['synthesis_2']['hypothesis'][:200]}",
                source="synthesis",
                tm_label="contradiction",
                regime_tag="synthesis",
                aif_confidence=0.8,
                project=project,
            )

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
        """SELECT id, entity_a, entity_b, hypothesis, confidence,
                  novelty_score, tested, test_result, parent_id, depth, created
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


def get_contradictions(project: str) -> list[dict]:
    """Return detected contradictions for a project."""
    return _detect_contradictions(project)
