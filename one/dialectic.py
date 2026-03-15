"""Adversarial Dialectic Engine — Knowledge forged through argument.

Every piece of knowledge is challenged through a thesis→antithesis→
synthesis→verification→meta-synthesis chain. Dialectic chains are
stored as linked structures in the knowledge graph.

Usage:
    engine = DialecticEngine(project="cancer")
    chain = engine.challenge(finding)
    synthesis = engine.synthesize(chain.thesis, chain.antithesis)
    verified = engine.verify(synthesis)
    pattern = engine.meta_synthesize([synthesis1, synthesis2, ...])
"""

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import push_memory, recall, DB_PATH, DB_DIR
from .hdc import encode_text, similarity
from .gemma import _call_ollama


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
        CREATE TABLE IF NOT EXISTS dialectic_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            topic TEXT,
            status TEXT DEFAULT 'active',
            created TEXT NOT NULL,
            updated TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_dialectic_project ON dialectic_chains(project);

        CREATE TABLE IF NOT EXISTS dialectic_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id INTEGER REFERENCES dialectic_chains(id),
            node_type TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            source TEXT,
            evidence TEXT,
            parent_node_id INTEGER REFERENCES dialectic_nodes(id),
            created TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_dialectic_nodes_chain ON dialectic_nodes(chain_id);
        CREATE INDEX IF NOT EXISTS idx_dialectic_nodes_type ON dialectic_nodes(node_type);
    """)
    conn.commit()


# ── Node Types ──────────────────────────────────────────────────

NODE_TYPES = [
    "thesis",
    "antithesis",
    "synthesis",
    "verification",
    "meta_synthesis",
]


# ── Prompts ─────────────────────────────────────────────────────

CHALLENGE_PROMPT = """You are an adversarial fact-checker. Given this finding, generate the STRONGEST possible counter-argument.

FINDING:
{finding}

Your counter-argument must be SPECIFIC and EVIDENCE-BASED. Not "this seems weak" but concrete flaws:
- Failed replications
- Statistical errors (sample size, p-hacking, confounders)
- Logical flaws
- Missing controls
- Alternative explanations
- Retracted sources
- Contradicting evidence from credible sources

If the finding is truly bulletproof, say so and explain why no counter-argument holds up.

Respond with ONLY the counter-argument. Be specific and cite evidence where possible."""

SYNTHESIZE_PROMPT = """Given a thesis and its antithesis, produce a SYNTHESIS that resolves the contradiction.

THESIS:
{thesis}

ANTITHESIS:
{antithesis}

Your synthesis must:
1. Acknowledge what's valid in BOTH positions
2. Identify the hidden variable or context that explains the disagreement
3. Produce a HIGHER-LEVEL understanding that encompasses both
4. Be testable — what prediction does the synthesis make?

Do NOT just split the difference. Find the DEEPER truth.

Respond with ONLY the synthesis."""

VERIFY_PROMPT = """Verify this synthesis against available evidence.

SYNTHESIS:
{synthesis}

ORIGINAL THESIS:
{thesis}

ORIGINAL ANTITHESIS:
{antithesis}

Search for:
1. Evidence that SUPPORTS the synthesis
2. Evidence that CONTRADICTS the synthesis
3. Whether the synthesis makes testable predictions
4. Whether similar resolutions exist in related fields

Respond with:
VERDICT: [SUPPORTED | CONTRADICTED | PARTIALLY_SUPPORTED | UNTESTED]
EVIDENCE: [specific evidence found]
CONFIDENCE: [0.0-1.0]
REASONING: [why you reached this verdict]"""

META_SYNTHESIS_PROMPT = """You have multiple dialectic syntheses. Find the UNIVERSAL PATTERN.

SYNTHESES:
{syntheses}

Look for:
1. Structural similarities across these resolutions
2. A DOMAIN-INDEPENDENT pattern that unifies them
3. Predictive power — if this pattern is real, what else should be true?
4. Historical precedents — has this pattern been observed before?

A meta-synthesis is NOT a summary. It is the discovery of a UNIVERSAL PRINCIPLE
that explains why all these individual syntheses resolve the way they do.

Respond with:
PATTERN NAME: [concise name]
PATTERN: [domain-independent description]
INSTANCES: [how each synthesis is an instance of this pattern]
PREDICTIONS: [what else should be true if this pattern holds]
CONFIDENCE: [0.0-1.0]"""


# ── Dialectic Engine ───────────────────────────────────────────

class DialecticEngine:
    """Produces higher-quality knowledge through structured argument."""

    def __init__(self, project: str, on_log: Optional[Callable] = None):
        self.project = project
        self._log = on_log or (lambda m: None)
        init_schema()

    def challenge(self, finding: str, source: str = "") -> dict:
        """Generate the strongest antithesis for a finding.

        Returns dict with chain_id, thesis, antithesis, confidence.
        """
        self._log(f"challenging: {finding[:80]}...")

        # Create chain
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO dialectic_chains (project, topic, created, updated) VALUES (?, ?, ?, ?)",
            (self.project, finding[:200], now, now),
        )
        conn.commit()
        chain_id = cur.lastrowid or 0

        # Store thesis
        cur = conn.execute(
            "INSERT INTO dialectic_nodes (chain_id, node_type, content, confidence, source, created) VALUES (?, ?, ?, ?, ?, ?)",
            (chain_id, "thesis", finding, 0.5, source, now),
        )
        conn.commit()
        thesis_id = cur.lastrowid or 0

        # Generate antithesis via Gemma
        prompt = CHALLENGE_PROMPT.format(finding=finding)
        antithesis_text = _call_ollama(prompt, timeout=120)

        if not antithesis_text:
            antithesis_text = "(No counter-argument generated — finding may be robust)"

        # Store antithesis
        cur = conn.execute(
            "INSERT INTO dialectic_nodes (chain_id, node_type, content, confidence, parent_node_id, created) VALUES (?, ?, ?, ?, ?, ?)",
            (chain_id, "antithesis", antithesis_text, 0.5, thesis_id, now),
        )
        conn.commit()
        antithesis_id = cur.lastrowid or 0

        # Store both as memories
        push_memory(
            f"[DIALECTIC THESIS] {finding}",
            source="dialectic",
            tm_label="dialectic_thesis",
            project=self.project,
        )
        push_memory(
            f"[DIALECTIC ANTITHESIS] {antithesis_text}",
            source="dialectic",
            tm_label="dialectic_antithesis",
            project=self.project,
        )

        self._log(f"antithesis generated for chain {chain_id}")

        return {
            "chain_id": chain_id,
            "thesis_id": thesis_id,
            "antithesis_id": antithesis_id,
            "thesis": finding,
            "antithesis": antithesis_text,
        }

    def synthesize(self, thesis: str, antithesis: str, chain_id: int = 0) -> dict:
        """Resolve thesis and antithesis into a synthesis."""
        self._log("synthesizing...")

        prompt = SYNTHESIZE_PROMPT.format(thesis=thesis, antithesis=antithesis)
        synthesis_text = _call_ollama(prompt, timeout=120)

        if not synthesis_text:
            synthesis_text = "(Synthesis generation failed)"

        # Store
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()

        if chain_id:
            cur = conn.execute(
                "INSERT INTO dialectic_nodes (chain_id, node_type, content, confidence, created) VALUES (?, ?, ?, ?, ?)",
                (chain_id, "synthesis", synthesis_text, 0.6, now),
            )
            conn.commit()
            conn.execute(
                "UPDATE dialectic_chains SET updated = ? WHERE id = ?",
                (now, chain_id),
            )
            conn.commit()

        push_memory(
            f"[DIALECTIC SYNTHESIS] {synthesis_text}",
            source="dialectic",
            tm_label="dialectic_synthesis",
            project=self.project,
        )

        self._log(f"synthesis complete for chain {chain_id}")

        return {
            "chain_id": chain_id,
            "synthesis": synthesis_text,
            "thesis": thesis,
            "antithesis": antithesis,
        }

    def verify(self, synthesis: str, thesis: str = "", antithesis: str = "", chain_id: int = 0) -> dict:
        """Verify a synthesis against available evidence."""
        self._log("verifying synthesis...")

        prompt = VERIFY_PROMPT.format(
            synthesis=synthesis,
            thesis=thesis,
            antithesis=antithesis,
        )
        result = _call_ollama(prompt, timeout=120)

        if not result:
            return {"verdict": "UNTESTED", "confidence": 0.0, "evidence": "", "reasoning": ""}

        # Parse verdict
        verdict = "UNTESTED"
        confidence = 0.5
        evidence = ""
        reasoning = ""

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                if v in ("SUPPORTED", "CONTRADICTED", "PARTIALLY_SUPPORTED", "UNTESTED"):
                    verdict = v
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("EVIDENCE:"):
                evidence = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Store verification
        if chain_id:
            conn = _get_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO dialectic_nodes (chain_id, node_type, content, confidence, evidence, created) VALUES (?, ?, ?, ?, ?, ?)",
                (chain_id, "verification", f"VERDICT: {verdict}\n{reasoning}", confidence, evidence, now),
            )
            conn.commit()
            conn.execute(
                "UPDATE dialectic_chains SET updated = ? WHERE id = ?",
                (now, chain_id),
            )
            conn.commit()

        push_memory(
            f"[DIALECTIC VERIFICATION] {verdict}: {reasoning}",
            source="dialectic",
            tm_label="dialectic_verification",
            project=self.project,
        )

        self._log(f"verification: {verdict} (conf: {confidence})")

        return {
            "chain_id": chain_id,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            "reasoning": reasoning,
        }

    def meta_synthesize(self, syntheses: list[str], chain_id: int = 0) -> dict:
        """Find universal patterns across multiple syntheses."""
        if len(syntheses) < 2:
            return {"pattern_name": "", "pattern": "", "confidence": 0.0}

        self._log(f"meta-synthesizing {len(syntheses)} syntheses...")

        syntheses_text = "\n\n".join(
            f"Synthesis {i+1}: {s}" for i, s in enumerate(syntheses)
        )
        prompt = META_SYNTHESIS_PROMPT.format(syntheses=syntheses_text)
        result = _call_ollama(prompt, timeout=120)

        if not result:
            return {"pattern_name": "", "pattern": "", "confidence": 0.0}

        # Parse result
        pattern_name = ""
        pattern = ""
        predictions = ""
        confidence = 0.5

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("PATTERN NAME:"):
                pattern_name = line.split(":", 1)[1].strip()
            elif line.startswith("PATTERN:"):
                pattern = line.split(":", 1)[1].strip()
            elif line.startswith("PREDICTIONS:"):
                predictions = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

        # Store meta-synthesis
        if chain_id:
            conn = _get_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO dialectic_nodes (chain_id, node_type, content, confidence, evidence, created) VALUES (?, ?, ?, ?, ?, ?)",
                (chain_id, "meta_synthesis", f"{pattern_name}: {pattern}", confidence, predictions, now),
            )
            conn.commit()

        # Store as high-value memory
        if pattern_name:
            push_memory(
                f"[UNIVERSAL PATTERN] {pattern_name}: {pattern}",
                source="dialectic",
                tm_label="universal_pattern",
                project=self.project,
                aif_confidence=confidence,
            )

        self._log(f"meta-synthesis: {pattern_name} (conf: {confidence})")

        return {
            "chain_id": chain_id,
            "pattern_name": pattern_name,
            "pattern": pattern,
            "predictions": predictions,
            "confidence": confidence,
        }

    def run_full_dialectic(self, finding: str, source: str = "") -> dict:
        """Run the full thesis→antithesis→synthesis→verification chain."""
        # Step 1: Challenge
        chain = self.challenge(finding, source)

        # Step 2: Synthesize
        synth = self.synthesize(
            chain["thesis"],
            chain["antithesis"],
            chain["chain_id"],
        )

        # Step 3: Verify
        verification = self.verify(
            synth["synthesis"],
            chain["thesis"],
            chain["antithesis"],
            chain["chain_id"],
        )

        # Update chain status
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        status = "verified" if verification["verdict"] in ("SUPPORTED", "PARTIALLY_SUPPORTED") else "challenged"
        conn.execute(
            "UPDATE dialectic_chains SET status = ?, updated = ? WHERE id = ?",
            (status, now, chain["chain_id"]),
        )
        conn.commit()

        return {
            "chain_id": chain["chain_id"],
            "thesis": chain["thesis"],
            "antithesis": chain["antithesis"],
            "synthesis": synth["synthesis"],
            "verdict": verification["verdict"],
            "confidence": verification["confidence"],
            "status": status,
        }

    def store_chain(self, chain: dict) -> int:
        """Persist a complete dialectic chain. Returns chain_id."""
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()

        cur = conn.execute(
            "INSERT INTO dialectic_chains (project, topic, status, created, updated) VALUES (?, ?, ?, ?, ?)",
            (self.project, chain.get("thesis", "")[:200], chain.get("status", "active"), now, now),
        )
        conn.commit()
        chain_id = cur.lastrowid or 0

        parent_id = None
        for node_type in NODE_TYPES:
            content = chain.get(node_type, "")
            if content:
                cur = conn.execute(
                    "INSERT INTO dialectic_nodes (chain_id, node_type, content, confidence, parent_node_id, created) VALUES (?, ?, ?, ?, ?, ?)",
                    (chain_id, node_type, content, chain.get("confidence", 0.5), parent_id, now),
                )
                conn.commit()
                parent_id = cur.lastrowid

        return chain_id

    def get_chains(self, topic: str = "", limit: int = 20) -> list[dict]:
        """Retrieve dialectic chains, optionally filtered by topic."""
        conn = _get_conn()

        if topic:
            rows = conn.execute(
                "SELECT * FROM dialectic_chains WHERE project = ? AND topic LIKE ? ORDER BY updated DESC LIMIT ?",
                (self.project, f"%{topic}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM dialectic_chains WHERE project = ? ORDER BY updated DESC LIMIT ?",
                (self.project, limit),
            ).fetchall()

        chains = []
        for row in rows:
            chain = dict(row)
            nodes = conn.execute(
                "SELECT * FROM dialectic_nodes WHERE chain_id = ? ORDER BY id",
                (chain["id"],),
            ).fetchall()
            chain["nodes"] = [dict(n) for n in nodes]
            chains.append(chain)

        return chains

    def get_chain(self, chain_id: int) -> Optional[dict]:
        """Get a single dialectic chain with all nodes."""
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM dialectic_chains WHERE id = ?",
            (chain_id,),
        ).fetchone()
        if not row:
            return None

        chain = dict(row)
        nodes = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ? ORDER BY id",
            (chain_id,),
        ).fetchall()
        chain["nodes"] = [dict(n) for n in nodes]
        return chain
