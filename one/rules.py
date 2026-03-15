"""Rule tree — live contextual rules that activate based on what you're doing.

Rules are organized as a tree: root rules are always active, child rules
activate when their context matches the current conversation. The tree
updates mid-session as patterns emerge.

Structure:
    project: kim-red
    ├── * (always active)
    │   ├── "no floating point in core modules"
    │   └── "all tests must pass"
    ├── hdc.py, encoding, vector (context trigger)
    │   ├── "vector dim is 4096"
    │   └── trigram, codebook (sub-context)
    │       └── "seed must stay 0xDEAD"
    └── deploy, ssh, server (context trigger)
        └── "use tailscale SSH to dedi"
"""

import os
import re
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .hdc import encode_text, similarity

DB_DIR = os.path.expanduser("~/.one")
DB_PATH = os.path.join(DB_DIR, "one.db")

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def init_rules_schema() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rule_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            parent_id INTEGER REFERENCES rule_nodes(id),
            rule_text TEXT NOT NULL,
            activation_keywords TEXT NOT NULL DEFAULT '*',
            hdc_vector BLOB,
            confidence REAL DEFAULT 0.5,
            source_count INTEGER DEFAULT 1,
            active INTEGER DEFAULT 1,
            superseded_by INTEGER REFERENCES rule_nodes(id),
            created TEXT NOT NULL,
            updated TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_rules_project ON rule_nodes(project);
        CREATE INDEX IF NOT EXISTS idx_rules_parent ON rule_nodes(parent_id);
        CREATE INDEX IF NOT EXISTS idx_rules_active ON rule_nodes(active);
    """)
    conn.commit()


# ── Rule node operations ────────────────────────────────────────────

def add_rule(
    project: str,
    rule_text: str,
    activation_keywords: str = "*",
    parent_id: Optional[int] = None,
    confidence: float = 0.5,
    source_count: int = 1,
) -> int:
    """Add a rule node to the tree. Returns the node ID."""
    init_rules_schema()
    vec = encode_text(rule_text)
    blob = vec.astype(np.float32).tobytes()
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO rule_nodes
           (project, parent_id, rule_text, activation_keywords, hdc_vector,
            confidence, source_count, created, updated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (project, parent_id, rule_text, activation_keywords, blob,
         confidence, source_count, now, now),
    )
    conn.commit()
    return cur.lastrowid


def update_rule(rule_id: int, confidence: float = None, source_count: int = None) -> None:
    """Bump a rule's confidence or source count."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    if confidence is not None:
        conn.execute("UPDATE rule_nodes SET confidence=?, updated=? WHERE id=?", (confidence, now, rule_id))
    if source_count is not None:
        conn.execute("UPDATE rule_nodes SET source_count=?, updated=? WHERE id=?", (source_count, now, rule_id))
    conn.commit()


def supersede_rule(old_id: int, new_text: str, project: str) -> int:
    """Create a new version of a rule, marking the old one superseded."""
    conn = _get_conn()
    old = conn.execute("SELECT * FROM rule_nodes WHERE id=?", (old_id,)).fetchone()
    if not old:
        return add_rule(project, new_text)

    new_id = add_rule(
        project=project,
        rule_text=new_text,
        activation_keywords=old["activation_keywords"],
        parent_id=old["parent_id"],
        confidence=0.6,
        source_count=1,
    )
    conn.execute("UPDATE rule_nodes SET active=0, superseded_by=? WHERE id=?", (new_id, old_id))
    conn.commit()
    return new_id


def get_all_rules(project: str) -> list[dict]:
    """Get all active rules for a project as a flat list."""
    init_rules_schema()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM rule_nodes WHERE project=? AND active=1 ORDER BY parent_id NULLS FIRST, confidence DESC",
        (project,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Context matching ────────────────────────────────────────────────

def _matches_context(keywords: str, text: str, files: list[str], tools: list[str]) -> bool:
    """Check if a rule's activation keywords match the current context."""
    if keywords == "*":
        return True

    kw_list = [k.strip().lower() for k in keywords.split(",")]
    text_lower = text.lower()
    files_lower = " ".join(f.lower() for f in files)
    tools_lower = " ".join(t.lower() for t in tools)
    combined = f"{text_lower} {files_lower} {tools_lower}"

    # Any keyword present = activated
    for kw in kw_list:
        if kw in combined:
            return True
    return False


def get_active_rules(
    project: str,
    current_text: str = "",
    recent_files: Optional[list[str]] = None,
    recent_tools: Optional[list[str]] = None,
) -> list[dict]:
    """Get rules that are active given the current context.

    Walks the tree: starts from roots, activates branches whose
    keywords match, recursively activates their children.
    """
    all_rules = get_all_rules(project)
    if not all_rules:
        return []

    files = recent_files or []
    tools = recent_tools or []

    # Build parent→children map
    children_of: dict[Optional[int], list[dict]] = {}
    for r in all_rules:
        pid = r["parent_id"]
        children_of.setdefault(pid, []).append(r)

    # Walk from roots
    active = []

    def _walk(parent_id: Optional[int]):
        for node in children_of.get(parent_id, []):
            if _matches_context(node["activation_keywords"], current_text, files, tools):
                active.append(node)
                _walk(node["id"])

    _walk(None)
    return active


def format_rules_for_injection(rules: list[dict], project: str) -> str:
    """Format active rules as a context block for Claude."""
    if not rules:
        return ""

    lines = [f"<project-rules source=\"one\" project=\"{project}\">"]

    # Group by parent for visual hierarchy
    roots = [r for r in rules if r["parent_id"] is None]
    children_of: dict[int, list[dict]] = {}
    for r in rules:
        if r["parent_id"] is not None:
            children_of.setdefault(r["parent_id"], []).append(r)

    def _fmt(node: dict, indent: int = 0):
        prefix = "  " * indent + "- "
        conf = node["confidence"]
        lines.append(f"{prefix}{node['rule_text']} [{conf:.0%}]")
        for child in children_of.get(node["id"], []):
            _fmt(child, indent + 1)

    for root in roots:
        _fmt(root)

    lines.append("</project-rules>")
    return "\n".join(lines)


# ── Rule learning ───────────────────────────────────────────────────

def find_matching_rule(project: str, text: str, threshold: float = 0.6) -> Optional[dict]:
    """Find an existing rule that matches this text by vector similarity."""
    all_rules = get_all_rules(project)
    if not all_rules:
        return None

    text_vec = encode_text(text)
    best_sim = 0.0
    best_rule = None

    for r in all_rules:
        if r["hdc_vector"]:
            rule_vec = np.frombuffer(r["hdc_vector"], dtype=np.float32)
            if rule_vec.shape[0] == text_vec.shape[0]:
                sim = similarity(text_vec, rule_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_rule = r

    if best_sim >= threshold and best_rule:
        return best_rule
    return None


def learn_rule_from_memory(
    project: str,
    text: str,
    confidence: float = 0.5,
    activation_keywords: Optional[str] = None,
) -> Optional[int]:
    """Attempt to learn a new rule or reinforce an existing one.

    Called when the AIF gate detects a decision or preference signal.
    """
    # Check if a similar rule exists
    existing = find_matching_rule(project, text, threshold=0.5)

    if existing:
        # Reinforce — bump confidence and source count
        new_conf = min(1.0, existing["confidence"] + 0.05)
        new_count = existing["source_count"] + 1
        update_rule(existing["id"], confidence=new_conf, source_count=new_count)
        return existing["id"]

    # Determine activation context from entity extraction
    if activation_keywords is None:
        from .entities import extract_entities
        ents = extract_entities(text, source="user")
        concepts = [e["name"].lower() for e in ents if e["type"] == "concept"]
        files = [e["name"].split("/")[-1].lower() for e in ents if e["type"] == "file"]
        keywords = concepts + files
        activation_keywords = ", ".join(keywords) if keywords else "*"

    # Find best parent — the most specific active branch that matches
    all_rules = get_all_rules(project)
    best_parent = None
    best_overlap = 0

    if activation_keywords != "*":
        my_kws = set(k.strip() for k in activation_keywords.split(","))
        for r in all_rules:
            if r["activation_keywords"] == "*":
                continue
            their_kws = set(k.strip() for k in r["activation_keywords"].split(","))
            overlap = len(my_kws & their_kws)
            if overlap > best_overlap:
                best_overlap = overlap
                best_parent = r["id"]

    return add_rule(
        project=project,
        rule_text=text,
        activation_keywords=activation_keywords,
        parent_id=best_parent,
        confidence=confidence,
    )


# ── Batch rule extraction ───────────────────────────────────────────

def extract_rules_from_memories(project: str, min_frequency: int = 3) -> list[str]:
    """Scan memory store for repeated patterns that should become rules.

    Finds high-confidence memories that cluster together, indicating
    the user has expressed the same preference/decision multiple times.
    """
    from . import store

    conn = store._get_conn()
    rows = conn.execute(
        """SELECT raw_text, source, aif_confidence, hdc_vector
           FROM memories
           WHERE project = ? AND source = 'user' AND aif_confidence > 0.5 AND hdc_vector IS NOT NULL
           ORDER BY timestamp DESC LIMIT 500""",
        (project,),
    ).fetchall()

    if len(rows) < min_frequency:
        return []

    import struct

    # Cluster by vector similarity
    vecs = []
    texts = []
    for r in rows:
        blob = r["hdc_vector"]
        n = len(blob) // 4
        vec = np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)
        vecs.append(vec)
        texts.append(r["raw_text"])

    # Greedy clustering — find groups of similar messages
    used = set()
    clusters = []

    for i in range(len(vecs)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(vecs)):
            if j in used:
                continue
            norm_i = np.linalg.norm(vecs[i])
            norm_j = np.linalg.norm(vecs[j])
            if norm_i > 1e-10 and norm_j > 1e-10:
                sim = float(np.dot(vecs[i], vecs[j]) / (norm_i * norm_j))
                if sim > 0.4:
                    cluster.append(j)
                    used.add(j)

        if len(cluster) >= min_frequency:
            clusters.append([texts[k] for k in cluster])

    # Each cluster = a potential rule
    proposed = []
    for cluster in clusters:
        # Use the shortest message as the rule (most distilled)
        rule = min(cluster, key=len)
        if not find_matching_rule(project, rule, threshold=0.5):
            proposed.append(rule)

    return proposed
