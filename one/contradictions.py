"""Contradiction Mining Engine — Active contradiction detection and resolution.

Contradictions are not bugs. They're signals that our understanding is
incomplete. Resolving contradictions produces the highest-value knowledge.

Contradiction severity levels:
- MINOR: different measurements
- MODERATE: different conclusions from similar data
- CRITICAL: directly opposing claims from credible sources
- PARADIGM: challenges a foundational assumption
"""

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import push_memory, recall, get_recent, DB_PATH, DB_DIR
from .hdc import encode_text, similarity
from .gemma import _call_ollama

import numpy as np


_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        import os
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
    return _local.conn


def init_schema() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            finding_a TEXT NOT NULL,
            finding_b TEXT NOT NULL,
            finding_a_source TEXT,
            finding_b_source TEXT,
            severity TEXT DEFAULT 'moderate',
            status TEXT DEFAULT 'active',
            resolution TEXT,
            resolution_type TEXT,
            confidence_a REAL DEFAULT 0.5,
            confidence_b REAL DEFAULT 0.5,
            created TEXT NOT NULL,
            resolved_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_contradictions_project ON contradictions(project);
        CREATE INDEX IF NOT EXISTS idx_contradictions_severity ON contradictions(severity);
        CREATE INDEX IF NOT EXISTS idx_contradictions_status ON contradictions(status);
    """)
    conn.commit()


SEVERITY_LEVELS = ["minor", "moderate", "critical", "paradigm"]

SCORE_PROMPT = """Assess the severity of this contradiction:

FINDING A:
{finding_a}

FINDING B:
{finding_b}

Severity levels:
- MINOR: Different measurements or minor disagreements (5% vs 7%)
- MODERATE: Different conclusions from similar data
- CRITICAL: Directly opposing claims from credible sources
- PARADIGM: Challenges a foundational assumption of the field

Respond with:
SEVERITY: [MINOR|MODERATE|CRITICAL|PARADIGM]
EXPLANATION: [why this is that severity level]
HIDDEN_VARIABLE: [what hidden variable might explain the disagreement, if any]"""

RESOLVE_PROMPT = """Systematically resolve this contradiction:

FINDING A:
{finding_a}

FINDING B:
{finding_b}

Determine which of these is true:
1. One side is WRONG (find the methodological flaw)
2. Both are RIGHT in different contexts (find the hidden variable)
3. Neither is right (both are approximations of a deeper truth)
4. This reveals a NEW PHENOMENON (breakthrough)

Respond with:
RESOLUTION_TYPE: [ONE_WRONG|CONTEXT_DEPENDENT|DEEPER_TRUTH|NEW_PHENOMENON]
RESOLUTION: [your resolution]
WRONG_SIDE: [A or B, if one is wrong, otherwise N/A]
HIDDEN_VARIABLE: [the variable that explains the disagreement, if context-dependent]
CONFIDENCE: [0.0-1.0]"""


class ContradictionMiner:
    """Actively mines and resolves contradictions in the knowledge graph."""

    def __init__(self, project: str, on_log: Optional[Callable] = None):
        self.project = project
        self._log = on_log or (lambda m: None)
        init_schema()

    def mine_contradictions(self, limit: int = 50) -> list[dict]:
        """Find all contradictions in the knowledge graph.

        Compares high-confidence findings pairwise using HDC similarity
        and contradiction signal detection.
        """
        self._log("mining contradictions...")

        # Get recent findings (use direct SQL, not vector search)
        findings = get_recent(n=limit, project=self.project)
        if len(findings) < 2:
            return []

        contradictions = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, f1 in enumerate(findings):
            for f2 in findings[i+1:]:
                # Skip if same source
                if f1.get("source") == f2.get("source"):
                    continue

                pair_key = tuple(sorted([f1["id"], f2["id"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Check for contradiction signals
                if self._has_contradiction_signals(f1["raw_text"], f2["raw_text"]):
                    contradiction = {
                        "finding_a": f1["raw_text"],
                        "finding_b": f2["raw_text"],
                        "finding_a_source": f1.get("source", ""),
                        "finding_b_source": f2.get("source", ""),
                    }
                    contradictions.append(contradiction)

        self._log(f"found {len(contradictions)} potential contradictions")
        return contradictions

    def _has_contradiction_signals(self, text_a: str, text_b: str) -> bool:
        """Detect if two texts might contradict each other."""
        a_lower = text_a.lower()
        b_lower = text_b.lower()

        # Extract significant words (skip stopwords)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                      "at", "to", "for", "of", "with", "by", "from", "and", "or",
                      "but", "not", "this", "that", "it", "as"}
        words_a = set(w for w in a_lower.split() if w not in stopwords and len(w) > 2)
        words_b = set(w for w in b_lower.split() if w not in stopwords and len(w) > 2)

        # Need some topical overlap
        overlap = words_a & words_b
        if len(overlap) < 2:
            return False

        # Check for negation asymmetry
        negation_words = {"not", "no", "never", "doesn't", "don't", "isn't",
                          "aren't", "won't", "can't", "cannot", "fails",
                          "failed", "ineffective", "inhibits", "prevents",
                          "blocks", "reduces", "decreases", "lacks"}

        neg_a = sum(1 for w in a_lower.split() if w in negation_words)
        neg_b = sum(1 for w in b_lower.split() if w in negation_words)

        # Asymmetric negation = potential contradiction
        if abs(neg_a - neg_b) >= 1 and (neg_a > 0 or neg_b > 0):
            return True

        # Check for opposing quantitative claims
        import re
        nums_a = set(re.findall(r'\d+(?:\.\d+)?%', a_lower))
        nums_b = set(re.findall(r'\d+(?:\.\d+)?%', b_lower))
        if nums_a and nums_b and nums_a != nums_b and overlap:
            return True

        return False

    def score_contradiction(self, finding_a: str, finding_b: str) -> dict:
        """Assess contradiction severity."""
        prompt = SCORE_PROMPT.format(finding_a=finding_a, finding_b=finding_b)
        result = _call_ollama(prompt, timeout=90)

        severity = "moderate"
        explanation = ""
        hidden_variable = ""

        if result:
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("SEVERITY:"):
                    s = line.split(":", 1)[1].strip().lower()
                    if s in SEVERITY_LEVELS:
                        severity = s
                elif line.startswith("EXPLANATION:"):
                    explanation = line.split(":", 1)[1].strip()
                elif line.startswith("HIDDEN_VARIABLE:"):
                    hidden_variable = line.split(":", 1)[1].strip()

        # Store the contradiction
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """INSERT INTO contradictions
               (project, finding_a, finding_b, severity, status, created)
               VALUES (?, ?, ?, ?, 'active', ?)""",
            (self.project, finding_a[:1000], finding_b[:1000], severity, now),
        )
        conn.commit()

        return {
            "id": cur.lastrowid or 0,
            "severity": severity,
            "explanation": explanation,
            "hidden_variable": hidden_variable,
        }

    def resolve_contradiction(self, finding_a: str, finding_b: str, contradiction_id: int = 0) -> dict:
        """Systematically resolve a contradiction."""
        self._log("resolving contradiction...")

        prompt = RESOLVE_PROMPT.format(finding_a=finding_a, finding_b=finding_b)
        result = _call_ollama(prompt, timeout=120)

        resolution_type = "CONTEXT_DEPENDENT"
        resolution = ""
        confidence = 0.5

        if result:
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("RESOLUTION_TYPE:"):
                    rt = line.split(":", 1)[1].strip().upper()
                    if rt in ("ONE_WRONG", "CONTEXT_DEPENDENT", "DEEPER_TRUTH", "NEW_PHENOMENON"):
                        resolution_type = rt
                elif line.startswith("RESOLUTION:"):
                    resolution = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

        # Update database
        if contradiction_id:
            conn = _get_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE contradictions SET status = 'resolved', resolution = ?, resolution_type = ?, resolved_at = ? WHERE id = ?",
                (resolution, resolution_type, now, contradiction_id),
            )
            conn.commit()

        # Store resolution as memory
        push_memory(
            f"[CONTRADICTION RESOLVED] {resolution_type}: {resolution}",
            source="contradiction",
            tm_label="contradiction_resolution",
            project=self.project,
            aif_confidence=confidence,
        )

        # Paradigm contradictions are especially high-value
        if resolution_type == "NEW_PHENOMENON":
            push_memory(
                f"[NEW PHENOMENON] {resolution}",
                source="contradiction",
                tm_label="breakthrough",
                project=self.project,
                aif_confidence=max(confidence, 0.8),
            )
            self._log(f"NEW PHENOMENON DISCOVERED: {resolution[:100]}")

        return {
            "resolution_type": resolution_type,
            "resolution": resolution,
            "confidence": confidence,
        }

    def get_paradigm_contradictions(self) -> list[dict]:
        """Get the big ones — paradigm-level contradictions."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM contradictions WHERE project = ? AND severity = 'paradigm' AND status = 'active' ORDER BY created DESC",
            (self.project,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_active(self, severity: str = "") -> list[dict]:
        """Get all active contradictions, optionally filtered by severity."""
        conn = _get_conn()
        if severity:
            rows = conn.execute(
                "SELECT * FROM contradictions WHERE project = ? AND status = 'active' AND severity = ? ORDER BY created DESC",
                (self.project, severity),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM contradictions WHERE project = ? AND status = 'active' ORDER BY CASE severity WHEN 'paradigm' THEN 0 WHEN 'critical' THEN 1 WHEN 'moderate' THEN 2 WHEN 'minor' THEN 3 END",
                (self.project,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_resolved(self, limit: int = 20) -> list[dict]:
        """Get resolved contradictions."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM contradictions WHERE project = ? AND status = 'resolved' ORDER BY resolved_at DESC LIMIT ?",
            (self.project, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def contradiction_dashboard(self) -> dict:
        """Summary stats for TUI display."""
        conn = _get_conn()

        total = conn.execute(
            "SELECT COUNT(*) FROM contradictions WHERE project = ?",
            (self.project,),
        ).fetchone()[0]

        active = conn.execute(
            "SELECT COUNT(*) FROM contradictions WHERE project = ? AND status = 'active'",
            (self.project,),
        ).fetchone()[0]

        resolved = conn.execute(
            "SELECT COUNT(*) FROM contradictions WHERE project = ? AND status = 'resolved'",
            (self.project,),
        ).fetchone()[0]

        paradigm = conn.execute(
            "SELECT COUNT(*) FROM contradictions WHERE project = ? AND severity = 'paradigm' AND status = 'active'",
            (self.project,),
        ).fetchone()[0]

        critical = conn.execute(
            "SELECT COUNT(*) FROM contradictions WHERE project = ? AND severity = 'critical' AND status = 'active'",
            (self.project,),
        ).fetchone()[0]

        return {
            "total": total,
            "active": active,
            "resolved": resolved,
            "paradigm_active": paradigm,
            "critical_active": critical,
            "resolution_rate": round(resolved / max(total, 1), 2),
        }
