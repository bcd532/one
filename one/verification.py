"""Self-Verifying Knowledge Engine + Active Question Generation.

System 5: Confidence lifecycle, verification sweeps, source quality.
System 6: Information value scoring, knowledge frontier mapping.

Every finding goes through:
NEW → CORROBORATED → CHALLENGED → VERIFIED → STALE → DEPRECATED
"""

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import push_memory, recall, get_recent, DB_PATH, DB_DIR
from .hdc import encode_text, similarity
from .gemma import _call_ollama


_local = threading.local()

# ── Source Quality Model ────────────────────────────────────────

SOURCE_QUALITY = {
    "meta-analysis": 0.95,
    "systematic review": 0.93,
    "cochrane": 0.95,
    "randomized controlled trial": 0.85,
    "rct": 0.85,
    "peer-reviewed": 0.80,
    "nature": 0.88,
    "science": 0.88,
    "cell": 0.88,
    "lancet": 0.88,
    "nejm": 0.88,
    "preprint": 0.60,
    "arxiv": 0.60,
    "biorxiv": 0.58,
    "expert blog": 0.40,
    "blog post": 0.35,
    "news article": 0.25,
    "press release": 0.20,
    "webpage": 0.15,
    "ai-generated": 0.10,
    "unverified": 0.10,
}

# ── Confidence Lifecycle States ─────────────────────────────────

CONFIDENCE_STATES = [
    "new",           # Initial from source quality
    "corroborated",  # Boosted by independent support
    "challenged",    # Reduced by counter-evidence
    "verified",      # Locked after re-verification
    "stale",         # Decayed after 30 days
    "deprecated",    # Below threshold, flagged for removal
]

STALE_DAYS = 30
DEPRECATION_THRESHOLD = 0.2


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
        CREATE TABLE IF NOT EXISTS verification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            memory_id TEXT,
            finding_id INTEGER,
            previous_confidence REAL,
            new_confidence REAL,
            verification_type TEXT,
            evidence TEXT,
            created TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_verification_project ON verification_log(project);
        CREATE INDEX IF NOT EXISTS idx_verification_memory ON verification_log(memory_id);

        CREATE TABLE IF NOT EXISTS knowledge_frontier (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            goal TEXT NOT NULL,
            question TEXT NOT NULL,
            information_value REAL DEFAULT 0.5,
            status TEXT DEFAULT 'unexplored',
            category TEXT DEFAULT 'general',
            created TEXT NOT NULL,
            explored_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_frontier_project ON knowledge_frontier(project);
        CREATE INDEX IF NOT EXISTS idx_frontier_value ON knowledge_frontier(information_value);
        CREATE INDEX IF NOT EXISTS idx_frontier_status ON knowledge_frontier(status);
    """)
    conn.commit()


# ── System 5: Self-Verification ─────────────────────────────────

class VerificationEngine:
    """Keeps the knowledge graph honest through active verification."""

    def __init__(self, project: str, on_log: Optional[Callable] = None):
        self.project = project
        self._log = on_log or (lambda _m: None)
        init_schema()

    def score_source(self, text: str) -> float:
        """Score source quality based on text content and source markers."""
        text_lower = text.lower()
        best_score = 0.15  # default: webpage quality

        for source, score in SOURCE_QUALITY.items():
            if source in text_lower:
                best_score = max(best_score, score)

        # Boost for quantitative data
        import re
        if re.search(r'\b(?:p\s*[<>=]\s*0\.\d+|n\s*=\s*\d{2,}|CI\s*[\[:(])', text):
            best_score = min(1.0, best_score + 0.1)

        return best_score

    def verify_finding(self, finding_text: str, current_confidence: float = 0.5) -> dict:
        """Verify a finding and return updated confidence."""
        self._log(f"verifying: {finding_text[:60]}...")

        prompt = f"""Verify this research finding. Search for replication, retraction, or new evidence.

FINDING: {finding_text}

Respond with:
STATUS: [VERIFIED|CORROBORATED|CHALLENGED|UNVERIFIABLE]
NEW_EVIDENCE: [any new evidence found]
CONFIDENCE_ADJUSTMENT: [+0.1, -0.2, etc.]
REASONING: [brief explanation]"""

        result = _call_ollama(prompt, timeout=120)

        status = "unverifiable"
        adjustment = 0.0
        evidence = ""
        reasoning = ""

        if result:
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("STATUS:"):
                    s = line.split(":", 1)[1].strip().lower()
                    if s in ("verified", "corroborated", "challenged", "unverifiable"):
                        status = s
                elif line.startswith("CONFIDENCE_ADJUSTMENT:"):
                    try:
                        adjustment = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("NEW_EVIDENCE:"):
                    evidence = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

        new_confidence = max(0.0, min(1.0, current_confidence + adjustment))

        # Log the verification
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO verification_log
               (project, previous_confidence, new_confidence, verification_type, evidence, created)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (self.project, current_confidence, new_confidence, status, evidence, now),
        )
        conn.commit()

        self._log(f"verification: {status} ({current_confidence:.2f} → {new_confidence:.2f})")

        return {
            "status": status,
            "previous_confidence": current_confidence,
            "new_confidence": new_confidence,
            "adjustment": adjustment,
            "evidence": evidence,
            "reasoning": reasoning,
        }

    def run_verification_sweep(self, n: int = 20) -> list[dict]:
        """Pick N highest-confidence stale findings and re-verify."""
        self._log(f"running verification sweep (n={n})...")

        # Get findings that haven't been verified recently
        findings = get_recent(n=n * 2, project=self.project)

        results = []
        verified = 0
        for finding in findings[:n]:
            result = self.verify_finding(
                finding["raw_text"],
                finding.get("aif_confidence", 0.5),
            )
            result["memory_id"] = finding["id"]
            results.append(result)
            verified += 1

        self._log(f"sweep complete: {verified} findings verified")
        return results

    def get_confidence_distribution(self) -> dict:
        """Histogram of confidence levels in the knowledge graph."""
        findings = get_recent(n=500, project=self.project)

        buckets = {
            "very_high (0.8-1.0)": 0,
            "high (0.6-0.8)": 0,
            "medium (0.4-0.6)": 0,
            "low (0.2-0.4)": 0,
            "very_low (0.0-0.2)": 0,
        }

        for f in findings:
            c = f.get("aif_confidence", 0.5)
            if c >= 0.8:
                buckets["very_high (0.8-1.0)"] += 1
            elif c >= 0.6:
                buckets["high (0.6-0.8)"] += 1
            elif c >= 0.4:
                buckets["medium (0.4-0.6)"] += 1
            elif c >= 0.2:
                buckets["low (0.2-0.4)"] += 1
            else:
                buckets["very_low (0.0-0.2)"] += 1

        return {
            "distribution": buckets,
            "total": len(findings),
            "avg_confidence": sum(f.get("aif_confidence", 0.5) for f in findings) / max(len(findings), 1),
        }

    def archive_deprecated(self, threshold: float = DEPRECATION_THRESHOLD) -> int:
        """Archive memories below confidence threshold. Returns count archived."""
        findings = get_recent(n=500, project=self.project)
        archived = 0

        for f in findings:
            if f.get("aif_confidence", 0.5) < threshold:
                # Mark as deprecated in memory
                push_memory(
                    f"[DEPRECATED] {f['raw_text'][:200]}",
                    source="verification",
                    tm_label="deprecated",
                    project=self.project,
                    aif_confidence=0.0,
                )
                archived += 1

        self._log(f"archived {archived} deprecated findings")
        return archived


# ── System 6: Active Question Generation ────────────────────────

class FrontierMapper:
    """Maps the knowledge frontier and generates the highest-value questions."""

    def __init__(self, project: str, on_log: Optional[Callable] = None):
        self.project = project
        self._log = on_log or (lambda _m: None)
        init_schema()

    def score_information_value(
        self,
        question: str,  # noqa: ARG002
        goal: str,  # noqa: ARG002
        unknowns_resolved: int = 1,
        contradictions_clarified: int = 0,
        goal_centrality: float = 0.5,
        novelty: float = 0.5,
    ) -> float:
        """Score the information value of a question."""
        return (
            min(1.0, unknowns_resolved * 0.1) * 0.3 +
            min(1.0, contradictions_clarified * 0.2) * 0.3 +
            goal_centrality * 0.25 +
            novelty * 0.15
        )

    def map_frontier(self, goal: str) -> dict:
        """Map the full knowledge frontier for a goal."""
        self._log(f"mapping frontier for: {goal[:60]}...")

        # Get existing knowledge
        findings = recall(goal, n=50, project=self.project)

        # Categorize by confidence
        explored = [f for f in findings if f.get("aif_confidence", 0) >= 0.7]
        partial = [f for f in findings if 0.3 <= f.get("aif_confidence", 0) < 0.7]
        weak = [f for f in findings if f.get("aif_confidence", 0) < 0.3]

        # Generate frontier questions
        questions = self._generate_frontier_questions(goal, findings)

        # Store questions
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        for q in questions:
            conn.execute(
                "INSERT INTO knowledge_frontier (project, goal, question, information_value, status, created) VALUES (?, ?, ?, ?, 'unexplored', ?)",
                (self.project, goal[:500], q["question"], q["information_value"], now),
            )
        conn.commit()

        # Coverage estimate
        total_aspects = max(len(explored) + len(partial) + len(weak) + len(questions), 1)
        coverage = len(explored) / total_aspects

        frontier = {
            "goal": goal,
            "explored": [
                {"text": f["raw_text"][:200], "confidence": f.get("aif_confidence", 0)}
                for f in explored[:10]
            ],
            "partially_explored": [
                {"text": f["raw_text"][:200], "confidence": f.get("aif_confidence", 0)}
                for f in partial[:10]
            ],
            "unexplored": questions[:10],
            "coverage": round(coverage, 2),
            "total_findings": len(findings),
        }

        self._log(f"frontier mapped: {len(explored)} explored, {len(partial)} partial, {len(questions)} questions")
        return frontier

    def _generate_frontier_questions(self, goal: str, existing_findings: list[dict]) -> list[dict]:
        """Generate high-value questions at the knowledge boundary."""
        existing_summary = "\n".join(
            f"- {f['raw_text'][:150]}" for f in existing_findings[:15]
        )

        prompt = f"""Given this research goal and existing findings, generate the 5 most valuable
questions that would advance our understanding the most.

GOAL: {goal}

EXISTING KNOWLEDGE:
{existing_summary if existing_summary else "(No findings yet)"}

For each question, consider:
- How many other unknowns would answering this resolve?
- Does this address a known contradiction?
- How central is this to the goal?
- Has anyone investigated this before?

Respond with exactly 5 questions, one per line, numbered 1-5.
Each should be specific and researchable, not vague."""

        result = _call_ollama(prompt, timeout=90)
        if not result:
            return []

        questions = []
        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip numbering
            for prefix in ["1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)", "5)", "-", "*"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break

            if len(line) > 20:
                # Score information value
                iv = self._estimate_information_value(line, goal, existing_findings)
                questions.append({
                    "question": line,
                    "information_value": iv,
                })

        questions.sort(key=lambda q: q["information_value"], reverse=True)
        return questions

    def _estimate_information_value(self, question: str, goal: str, findings: list[dict]) -> float:
        """Estimate information value without LLM call."""
        # Goal centrality: how similar is the question to the goal?
        q_vec = encode_text(question)
        g_vec = encode_text(goal)
        goal_centrality = max(0.0, float(similarity(q_vec, g_vec)))

        # Novelty: how dissimilar is it from existing findings?
        novelty = 1.0
        for f in findings[:10]:
            f_vec = encode_text(f["raw_text"])
            sim = float(similarity(q_vec, f_vec))
            novelty = min(novelty, 1.0 - max(0.0, sim))

        return self.score_information_value(
            question=question,
            goal=goal,
            unknowns_resolved=1,
            goal_centrality=goal_centrality,
            novelty=novelty,
        )

    def best_question(self, goal: str) -> Optional[dict]:
        """Return the single highest-value unexplored question."""
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM knowledge_frontier WHERE project = ? AND goal = ? AND status = 'unexplored' ORDER BY information_value DESC LIMIT 1",
            (self.project, goal[:500]),
        ).fetchone()

        if row:
            return dict(row)

        # Generate new questions if none exist
        frontier = self.map_frontier(goal)
        if frontier["unexplored"]:
            return frontier["unexplored"][0]
        return None

    def mark_explored(self, question_id: int) -> None:
        """Mark a frontier question as explored."""
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE knowledge_frontier SET status = 'explored', explored_at = ? WHERE id = ?",
            (now, question_id),
        )
        conn.commit()

    def frontier_coverage(self, goal: str) -> float:
        """Percentage of frontier questions that have been explored."""
        conn = _get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM knowledge_frontier WHERE project = ? AND goal = ?",
            (self.project, goal[:500]),
        ).fetchone()[0]

        if total == 0:
            return 0.0

        explored = conn.execute(
            "SELECT COUNT(*) FROM knowledge_frontier WHERE project = ? AND goal = ? AND status = 'explored'",
            (self.project, goal[:500]),
        ).fetchone()[0]

        return explored / total

    def update_frontier(self, goal: str, new_findings: list[str]) -> None:  # noqa: ARG002
        """Recompute frontier after new data arrives."""
        # Re-map considering new findings
        self.map_frontier(goal)
