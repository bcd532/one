"""Analogical Transfer Engine — Cross-domain structural isomorphism.

Extracts domain-independent templates from findings and matches them
across domains. When the same structure appears in 3+ domains, it
becomes a UNIVERSAL PATTERN with predictive power.

Usage:
    engine = AnalogyEngine(project="cancer")
    template = engine.extract_template("checkpoint inhibitors block PD-1...")
    matches = engine.match_templates(template)
    patterns = engine.find_universal_patterns(min_domains=3)
"""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import push_memory, DB_PATH, DB_DIR
from .hdc import encode_text, similarity, normalize, DIM
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
        CREATE TABLE IF NOT EXISTS analogy_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            source_finding TEXT NOT NULL,
            domain TEXT NOT NULL,
            mechanism TEXT,
            target TEXT,
            location TEXT,
            effect TEXT,
            outcome TEXT,
            hdc_vector BLOB,
            confidence REAL DEFAULT 0.5,
            created TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_templates_project ON analogy_templates(project);
        CREATE INDEX IF NOT EXISTS idx_templates_domain ON analogy_templates(domain);

        CREATE TABLE IF NOT EXISTS universal_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            pattern_name TEXT NOT NULL,
            pattern_description TEXT NOT NULL,
            domains TEXT NOT NULL,
            domain_count INTEGER DEFAULT 0,
            instances TEXT,
            predictions TEXT,
            confidence REAL DEFAULT 0.5,
            hdc_vector BLOB,
            created TEXT NOT NULL,
            updated TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_patterns_project ON universal_patterns(project);
        CREATE INDEX IF NOT EXISTS idx_patterns_domains ON universal_patterns(domain_count);
    """)
    conn.commit()


# ── Prompts ─────────────────────────────────────────────────────

EXTRACT_TEMPLATE_PROMPT = """Extract the domain-independent STRUCTURAL TEMPLATE from this finding.

FINDING:
{finding}

Decompose into these structural elements:
- MECHANISM: What action/process is happening? (e.g., "blocking", "removing", "amplifying")
- TARGET: What is being acted on? (e.g., "inhibitory_receptor", "rate_limiter")
- LOCATION: Where in the system? (e.g., "cell_surface", "api_gateway", "market")
- EFFECT: What immediate result? (e.g., "releases_suppressed_function", "increases_throughput")
- OUTCOME: What final outcome? (e.g., "target_elimination", "performance_improvement")
- DOMAIN: What field is this from? (e.g., "immunology", "software", "economics")

Respond in this exact format:
MECHANISM: ...
TARGET: ...
LOCATION: ...
EFFECT: ...
OUTCOME: ...
DOMAIN: ..."""

PREDICT_PROMPT = """Given this universal pattern observed across multiple domains, generate a prediction for a new domain.

PATTERN: {pattern_name}
DESCRIPTION: {pattern_description}

KNOWN INSTANCES:
{instances}

NEW DOMAIN: {new_domain}

Generate a specific, testable prediction for how this pattern would manifest in {new_domain}.
Be concrete — not "it might apply" but "specifically, X in {new_domain} should behave like Y because Z."

Respond with:
PREDICTION: [specific prediction]
TESTABLE: [how to verify this prediction]
CONFIDENCE: [0.0-1.0]"""


# ── Template Dataclass ──────────────────────────────────────────

@dataclass
class StructuralTemplate:
    mechanism: str = ""
    target: str = ""
    location: str = ""
    effect: str = ""
    outcome: str = ""
    domain: str = ""
    source_finding: str = ""
    confidence: float = 0.5


# ── Analogy Engine ──────────────────────────────────────────────

class AnalogyEngine:
    """Cross-domain structural isomorphism detector."""

    def __init__(self, project: str, on_log: Optional[Callable] = None):
        self.project = project
        self._log = on_log or (lambda m: None)
        init_schema()

    def extract_template(self, finding: str) -> StructuralTemplate:
        """Extract a domain-independent structural template from a finding."""
        self._log(f"extracting template: {finding[:60]}...")

        prompt = EXTRACT_TEMPLATE_PROMPT.format(finding=finding)
        result = _call_ollama(prompt, timeout=90)

        template = StructuralTemplate(source_finding=finding)

        if result:
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("MECHANISM:"):
                    template.mechanism = line.split(":", 1)[1].strip()
                elif line.startswith("TARGET:"):
                    template.target = line.split(":", 1)[1].strip()
                elif line.startswith("LOCATION:"):
                    template.location = line.split(":", 1)[1].strip()
                elif line.startswith("EFFECT:"):
                    template.effect = line.split(":", 1)[1].strip()
                elif line.startswith("OUTCOME:"):
                    template.outcome = line.split(":", 1)[1].strip()
                elif line.startswith("DOMAIN:"):
                    template.domain = line.split(":", 1)[1].strip()

        # Store template
        self._store_template(template)
        self._log(f"template: {template.mechanism} → {template.effect} ({template.domain})")

        return template

    def _store_template(self, template: StructuralTemplate) -> int:
        """Persist a template to the database."""
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()

        # Encode template as HDC vector for similarity matching
        template_text = f"{template.mechanism} {template.target} {template.effect} {template.outcome}"
        vec = encode_text(template_text).astype(np.float32)
        blob = vec.tobytes()

        cur = conn.execute(
            """INSERT INTO analogy_templates
               (project, source_finding, domain, mechanism, target, location, effect, outcome, hdc_vector, confidence, created)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.project, template.source_finding[:500], template.domain,
             template.mechanism, template.target, template.location,
             template.effect, template.outcome, blob, template.confidence, now),
        )
        conn.commit()
        return cur.lastrowid or 0

    def match_templates(self, template: StructuralTemplate, min_similarity: float = 0.3) -> list[dict]:
        """Find structurally similar templates across domains."""
        conn = _get_conn()

        # Get all templates from OTHER domains
        rows = conn.execute(
            "SELECT * FROM analogy_templates WHERE project = ? AND domain != ?",
            (self.project, template.domain),
        ).fetchall()

        if not rows:
            return []

        # Encode query template
        query_text = f"{template.mechanism} {template.target} {template.effect} {template.outcome}"
        query_vec = encode_text(query_text)

        matches = []
        for row in rows:
            row_dict = dict(row)
            if row_dict.get("hdc_vector"):
                row_vec = np.frombuffer(row_dict["hdc_vector"], dtype=np.float32)
                if len(row_vec) == DIM:
                    sim = float(similarity(query_vec, row_vec))
                    if sim >= min_similarity:
                        row_dict["structural_similarity"] = sim
                        del row_dict["hdc_vector"]  # don't return blobs
                        matches.append(row_dict)

        matches.sort(key=lambda x: x["structural_similarity"], reverse=True)
        return matches[:20]

    def find_universal_patterns(self, min_domains: int = 3) -> list[dict]:
        """Find patterns that span min_domains or more domains."""
        conn = _get_conn()
        self._log(f"searching for patterns across {min_domains}+ domains...")

        # Get all templates grouped by structural similarity
        rows = conn.execute(
            "SELECT * FROM analogy_templates WHERE project = ?",
            (self.project,),
        ).fetchall()

        if len(rows) < min_domains:
            return []

        templates = []
        for row in rows:
            t = dict(row)
            if t.get("hdc_vector"):
                t["_vec"] = np.frombuffer(t["hdc_vector"], dtype=np.float32)
            templates.append(t)

        # Cluster templates by structural similarity
        clusters: list[list[dict]] = []
        used = set()

        for i, t1 in enumerate(templates):
            if i in used or "_vec" not in t1:
                continue

            cluster = [t1]
            used.add(i)

            for j, t2 in enumerate(templates):
                if j in used or j <= i or "_vec" not in t2:
                    continue
                sim = float(similarity(t1["_vec"], t2["_vec"]))
                if sim > 0.3:
                    cluster.append(t2)
                    used.add(j)

            if len(cluster) >= min_domains:
                # Check domain diversity
                domains = set(t.get("domain", "") for t in cluster)
                if len(domains) >= min_domains:
                    clusters.append(cluster)

        # Generate pattern descriptions for qualifying clusters
        patterns = []
        for cluster in clusters:
            domains = list(set(t.get("domain", "") for t in cluster))
            instances = "\n".join(
                f"- [{t.get('domain', '')}] {t.get('mechanism', '')} → {t.get('effect', '')}"
                for t in cluster
            )

            # Use Gemma to name the pattern
            pattern_prompt = f"""Name this cross-domain pattern:

{instances}

Respond with:
NAME: [concise pattern name like INHIBITOR_RELEASE or CASCADE_AMPLIFICATION]
DESCRIPTION: [one sentence describing the domain-independent principle]"""

            result = _call_ollama(pattern_prompt, timeout=60)
            name = "unnamed_pattern"
            description = ""
            if result:
                for line in result.split("\n"):
                    if line.strip().startswith("NAME:"):
                        name = line.split(":", 1)[1].strip()
                    elif line.strip().startswith("DESCRIPTION:"):
                        description = line.split(":", 1)[1].strip()

            pattern = {
                "pattern_name": name,
                "pattern_description": description,
                "domains": domains,
                "domain_count": len(domains),
                "instances": instances,
                "confidence": min(0.9, 0.5 + len(domains) * 0.1),
            }
            patterns.append(pattern)

            # Store the pattern
            self._store_pattern(pattern)

            # Store as high-value memory
            push_memory(
                f"[UNIVERSAL PATTERN] {name}: {description} (across {', '.join(domains)})",
                source="analogy",
                tm_label="universal_pattern",
                project=self.project,
                aif_confidence=pattern["confidence"],
            )

            self._log(f"UNIVERSAL PATTERN: {name} ({len(domains)} domains)")

        return patterns

    def _store_pattern(self, pattern: dict) -> int:
        """Persist a universal pattern."""
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()

        vec = encode_text(f"{pattern['pattern_name']} {pattern['pattern_description']}").astype(np.float32)
        blob = vec.tobytes()

        cur = conn.execute(
            """INSERT INTO universal_patterns
               (project, pattern_name, pattern_description, domains, domain_count,
                instances, predictions, confidence, hdc_vector, created, updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.project, pattern["pattern_name"], pattern.get("pattern_description", ""),
             ",".join(pattern.get("domains", [])), pattern.get("domain_count", 0),
             pattern.get("instances", ""), pattern.get("predictions", ""),
             pattern.get("confidence", 0.5), blob, now, now),
        )
        conn.commit()
        return cur.lastrowid or 0

    def predict_from_pattern(self, pattern_name: str, new_domain: str) -> dict:
        """Generate a hypothesis by applying a pattern to a new domain."""
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM universal_patterns WHERE project = ? AND pattern_name = ?",
            (self.project, pattern_name),
        ).fetchone()

        if not row:
            return {"prediction": "", "confidence": 0.0}

        pattern = dict(row)
        prompt = PREDICT_PROMPT.format(
            pattern_name=pattern["pattern_name"],
            pattern_description=pattern["pattern_description"],
            instances=pattern.get("instances", ""),
            new_domain=new_domain,
        )

        result = _call_ollama(prompt, timeout=90)
        if not result:
            return {"prediction": "", "confidence": 0.0}

        prediction = ""
        testable = ""
        confidence = 0.5

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("PREDICTION:"):
                prediction = line.split(":", 1)[1].strip()
            elif line.startswith("TESTABLE:"):
                testable = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

        if prediction:
            push_memory(
                f"[ANALOGY PREDICTION] {prediction} (from pattern {pattern_name} applied to {new_domain})",
                source="analogy",
                tm_label="analogy_prediction",
                project=self.project,
            )

        return {
            "pattern_name": pattern_name,
            "new_domain": new_domain,
            "prediction": prediction,
            "testable": testable,
            "confidence": confidence,
        }

    def get_patterns(self, min_domains: int = 2) -> list[dict]:
        """Get all stored universal patterns."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, project, pattern_name, pattern_description, domains, domain_count, instances, predictions, confidence, created, updated FROM universal_patterns WHERE project = ? AND domain_count >= ? ORDER BY confidence DESC",
            (self.project, min_domains),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_templates(self, domain: str = "") -> list[dict]:
        """Get stored templates, optionally filtered by domain."""
        conn = _get_conn()
        if domain:
            rows = conn.execute(
                "SELECT id, project, source_finding, domain, mechanism, target, location, effect, outcome, confidence, created FROM analogy_templates WHERE project = ? AND domain = ?",
                (self.project, domain),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, project, source_finding, domain, mechanism, target, location, effect, outcome, confidence, created FROM analogy_templates WHERE project = ?",
                (self.project,),
            ).fetchall()
        return [dict(r) for r in rows]
