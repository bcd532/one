"""Foundry Audit — Knowledge Quality Enforcement.

System 11: Every piece of data must EARN its place.

Checks:
- MEMORY QUALITY: intelligibility, usefulness, redundancy, vector integrity
- ENTITY QUALITY: real concepts, linked, no duplicates, correct types
- RULE QUALITY: actionable, specific, no contradictions
- RESEARCH QUALITY: sourced, calibrated, current, quantitative
- SYNTHESIS QUALITY: novel, justified, testable
- PLAYBOOK QUALITY: useful decisions, reusable patterns

Auto-fix mode deletes garbage, re-encodes vectors, merges duplicates.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import DB_PATH, DB_DIR
from .hdc import encode_text, similarity, DIM
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


QUALITY_PROMPT = """Score this memory on a scale of 0-10 for quality and usefulness.

MEMORY TEXT:
{text}

SOURCE: {source}
CONFIDENCE: {confidence}

Scoring criteria:
- Is the text intelligible? (not garbled, not raw JSON)
- Is it actually useful? Would recalling this help in a future session?
- Is it specific enough to be actionable?
- Does the confidence level make sense for the content?

Respond with ONLY a number 0-10 and one sentence explanation:
SCORE: [0-10]
REASON: [one sentence]"""


class AuditEngine:
    """Knowledge quality enforcement engine."""

    def __init__(self, project: str, on_log: Optional[Callable] = None):
        self.project = project
        self._log = on_log or (lambda _m: None)

    # ── Memory Quality ─────────────────────────────────────────

    def score_memory(self, memory: dict) -> dict:
        """Score a single memory's quality 0-10."""
        text = memory.get("raw_text", "")
        source = memory.get("source", "unknown")
        confidence = memory.get("aif_confidence", 0.5)

        score = 5.0  # default
        reason = ""

        # Quick heuristic checks (avoid LLM call for obvious cases)
        if not text or len(text.strip()) < 10:
            return {"score": 0, "reason": "empty or too short"}

        if text.startswith("{") or text.startswith("["):
            score -= 3  # raw JSON dump

        if len(text) > 5000:
            score -= 1  # too long to be useful

        # Check for garbled text (high punctuation ratio)
        alpha = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and alpha / len(text) < 0.3:
            return {"score": 1, "reason": "garbled or non-text content"}

        # Check for usefulness markers
        useful_markers = ["because", "means", "therefore", "found that",
                          "shows", "indicates", "result", "conclusion"]
        has_useful = any(m in text.lower() for m in useful_markers)
        if has_useful:
            score += 1

        # Confidence sanity check
        if confidence > 0.8 and source in ("unverified", "ai-generated"):
            score -= 2  # high confidence from low-quality source

        score = max(0, min(10, score))
        return {"score": round(score, 1), "reason": reason or "heuristic score"}

    def score_memory_llm(self, memory: dict) -> dict:
        """Score a memory using LLM (slower, more accurate)."""
        text = memory.get("raw_text", "")
        source = memory.get("source", "unknown")
        confidence = memory.get("aif_confidence", 0.5)

        prompt = QUALITY_PROMPT.format(
            text=text[:500], source=source, confidence=confidence,
        )
        result = _call_ollama(prompt, timeout=30)

        score = 5.0
        reason = ""

        if result:
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

        return {"score": round(max(0, min(10, score)), 1), "reason": reason}

    # ── Entity Quality ─────────────────────────────────────────

    def score_entity(self, entity: dict) -> dict:
        """Assess entity quality."""
        name = entity.get("name", "")
        entity_type = entity.get("entity_type", "")
        obs_count = entity.get("observation_count", 0)

        issues: list[str] = []
        score = 7.0  # start optimistic

        # Is it a real concept?
        stopwords = {"the", "a", "an", "it", "is", "was", "this", "that", "and", "or", "but"}
        if name.lower() in stopwords or len(name) < 2:
            issues.append("likely not a real concept")
            score -= 5

        # Has observations?
        if obs_count == 0:
            issues.append("no observations — orphan entity")
            score -= 2

        # Reasonable type?
        valid_types = {"concept", "file", "method", "person", "organization",
                       "tool", "technology", "paper", "chemical", "gene", "protein"}
        if entity_type and entity_type not in valid_types:
            issues.append(f"unusual entity type: {entity_type}")
            score -= 1

        return {
            "score": round(max(0, min(10, score)), 1),
            "issues": issues,
        }

    # ── Rule Quality ───────────────────────────────────────────

    def score_rule(self, rule: dict) -> dict:
        """Assess rule actionability."""
        text = rule.get("rule_text", "")
        source_count = rule.get("source_count", 0)
        confidence = rule.get("confidence", 0.5)

        issues: list[str] = []
        score = 5.0

        if not text or len(text) < 15:
            issues.append("too short to be actionable")
            score -= 3

        # Vague rule detection
        vague_patterns = ["always use best practices", "be careful",
                          "do good things", "try harder", "be better"]
        if any(v in text.lower() for v in vague_patterns):
            issues.append("too vague to be useful")
            score -= 3

        # Reinforcement check
        if source_count >= 3:
            score += 2  # well-reinforced
        elif source_count == 0:
            issues.append("never reinforced — possibly noise")
            score -= 1

        if confidence > 0.7:
            score += 1

        return {
            "score": round(max(0, min(10, score)), 1),
            "issues": issues,
        }

    # ── Duplicate Detection ────────────────────────────────────

    def find_duplicates(self, threshold: float = 0.85) -> dict:
        """Find duplicate memories and entities."""
        conn = _get_conn()
        self._log("scanning for duplicates...")

        duplicate_memories: list[dict] = []
        duplicate_entities: list[dict] = []

        # Memory duplicates via HDC similarity
        try:
            rows = conn.execute(
                "SELECT id, raw_text, hdc_vector FROM memories WHERE project = ? LIMIT 500",
                (self.project,),
            ).fetchall()

            vectors: list[tuple[int, str, np.ndarray]] = []
            for row in rows:
                r = dict(row)
                if r.get("hdc_vector"):
                    vec = np.frombuffer(r["hdc_vector"], dtype=np.float32)
                    if len(vec) == DIM:
                        vectors.append((r["id"], r["raw_text"][:100], vec))

            seen: set[int] = set()
            for i, (id_a, text_a, vec_a) in enumerate(vectors):
                if id_a in seen:
                    continue
                for id_b, text_b, vec_b in vectors[i + 1:]:
                    if id_b in seen:
                        continue
                    sim = float(similarity(vec_a, vec_b))
                    if sim >= threshold:
                        duplicate_memories.append({
                            "id_a": id_a, "text_a": text_a,
                            "id_b": id_b, "text_b": text_b,
                            "similarity": round(sim, 3),
                        })
                        seen.add(id_b)

        except sqlite3.OperationalError:
            pass

        # Entity duplicates (case-insensitive name match)
        try:
            rows = conn.execute(
                "SELECT id, name, entity_type, observation_count FROM entities WHERE project = ?",
                (self.project,),
            ).fetchall()

            name_map: dict[str, list[dict]] = {}
            for row in rows:
                r = dict(row)
                key = r["name"].lower().strip()
                if key not in name_map:
                    name_map[key] = []
                name_map[key].append(r)

            for name_lower, entities in name_map.items():
                if len(entities) > 1:
                    duplicate_entities.append({
                        "name": name_lower,
                        "entities": entities,
                        "count": len(entities),
                    })

        except sqlite3.OperationalError:
            pass

        self._log(f"found {len(duplicate_memories)} duplicate memories, {len(duplicate_entities)} duplicate entities")
        return {
            "duplicate_memories": duplicate_memories,
            "duplicate_entities": duplicate_entities,
        }

    # ── Garbage Detection ──────────────────────────────────────

    def find_garbage(self, score_threshold: float = 3.0) -> dict:
        """Find memories and entities below quality threshold."""
        conn = _get_conn()
        self._log("scanning for garbage...")

        garbage_memories: list[dict] = []
        garbage_entities: list[dict] = []

        # Score all memories
        try:
            rows = conn.execute(
                "SELECT id, raw_text, source, aif_confidence, hdc_vector FROM memories WHERE project = ? LIMIT 500",
                (self.project,),
            ).fetchall()

            for row in rows:
                r = dict(row)
                result = self.score_memory(r)
                if result["score"] < score_threshold:
                    garbage_memories.append({
                        "id": r["id"],
                        "text": r["raw_text"][:100],
                        "score": result["score"],
                        "reason": result["reason"],
                    })

        except sqlite3.OperationalError:
            pass

        # Score all entities
        try:
            rows = conn.execute(
                "SELECT id, name, entity_type, observation_count FROM entities WHERE project = ?",
                (self.project,),
            ).fetchall()

            for row in rows:
                r = dict(row)
                result = self.score_entity(r)
                if result["score"] < score_threshold:
                    garbage_entities.append({
                        "id": r["id"],
                        "name": r["name"],
                        "score": result["score"],
                        "issues": result["issues"],
                    })

        except sqlite3.OperationalError:
            pass

        self._log(f"found {len(garbage_memories)} garbage memories, {len(garbage_entities)} garbage entities")
        return {
            "garbage_memories": garbage_memories,
            "garbage_entities": garbage_entities,
        }

    # ── Auto Fix ───────────────────────────────────────────────

    def auto_fix(self) -> dict:
        """Auto-fix knowledge quality issues.

        1. Delete garbage memories (score < 3)
        2. Re-encode memories with zero/corrupted vectors
        3. Merge duplicate entities
        4. Delete orphaned entities
        5. Add UNVERIFIED tag to unsourced findings
        """
        conn = _get_conn()
        self._log("running auto-fix...")

        stats: dict[str, int] = {
            "deleted_memories": 0,
            "re_encoded": 0,
            "merged_entities": 0,
            "deleted_orphans": 0,
            "tagged_unverified": 0,
        }

        # 1. Delete garbage memories
        garbage = self.find_garbage(score_threshold=3.0)
        for g in garbage["garbage_memories"]:
            try:
                conn.execute("DELETE FROM memories WHERE id = ?", (g["id"],))
                stats["deleted_memories"] += 1
            except sqlite3.OperationalError:
                pass
        conn.commit()

        # 2. Re-encode memories with zero/corrupted vectors
        try:
            rows = conn.execute(
                "SELECT id, raw_text, hdc_vector FROM memories WHERE project = ?",
                (self.project,),
            ).fetchall()

            for row in rows:
                r = dict(row)
                needs_reencoding = False

                if not r.get("hdc_vector"):
                    needs_reencoding = True
                else:
                    vec = np.frombuffer(r["hdc_vector"], dtype=np.float32)
                    if len(vec) != DIM or np.all(vec == 0):
                        needs_reencoding = True

                if needs_reencoding and r.get("raw_text"):
                    new_vec = encode_text(r["raw_text"])
                    conn.execute(
                        "UPDATE memories SET hdc_vector = ? WHERE id = ?",
                        (new_vec.tobytes(), r["id"]),
                    )
                    stats["re_encoded"] += 1

            conn.commit()
        except sqlite3.OperationalError:
            pass

        # 3. Merge duplicate entities
        duplicates = self.find_duplicates()
        for dup in duplicates["duplicate_entities"]:
            entities = dup["entities"]
            if len(entities) < 2:
                continue
            # Keep the one with highest observation count
            entities.sort(key=lambda e: e.get("observation_count", 0), reverse=True)
            keep = entities[0]
            for remove in entities[1:]:
                try:
                    # Transfer observations
                    conn.execute(
                        "UPDATE entities SET observation_count = observation_count + ? WHERE id = ?",
                        (remove.get("observation_count", 0), keep["id"]),
                    )
                    conn.execute("DELETE FROM entities WHERE id = ?", (remove["id"],))
                    stats["merged_entities"] += 1
                except sqlite3.OperationalError:
                    pass
        conn.commit()

        # 4. Delete orphaned entities
        try:
            orphans = conn.execute(
                "SELECT id FROM entities WHERE project = ? AND observation_count = 0",
                (self.project,),
            ).fetchall()
            for orphan in orphans:
                conn.execute("DELETE FROM entities WHERE id = ?", (orphan["id"],))
                stats["deleted_orphans"] += 1
            conn.commit()
        except sqlite3.OperationalError:
            pass

        # 5. Tag unverified findings
        try:
            rows = conn.execute(
                "SELECT id, raw_text, source FROM memories WHERE project = ? AND source IN ('unverified', 'ai-generated', '') AND raw_text NOT LIKE '[UNVERIFIED]%'",
                (self.project,),
            ).fetchall()
            for row in rows:
                r = dict(row)
                conn.execute(
                    "UPDATE memories SET raw_text = ? WHERE id = ?",
                    (f"[UNVERIFIED] {r['raw_text']}", r["id"]),
                )
                stats["tagged_unverified"] += 1
            conn.commit()
        except sqlite3.OperationalError:
            pass

        self._log(f"auto-fix complete: {stats}")
        return stats

    # ── Rebuild Pipeline ───────────────────────────────────────

    def rebuild_pipeline(self) -> dict:
        """Re-process all memories through current HDC encoder."""
        conn = _get_conn()
        self._log("rebuilding pipeline — re-encoding all memories...")

        count = 0
        try:
            rows = conn.execute(
                "SELECT id, raw_text FROM memories WHERE project = ?",
                (self.project,),
            ).fetchall()

            for row in rows:
                r = dict(row)
                if r.get("raw_text"):
                    new_vec = encode_text(r["raw_text"])
                    conn.execute(
                        "UPDATE memories SET hdc_vector = ? WHERE id = ?",
                        (new_vec.tobytes(), r["id"]),
                    )
                    count += 1

            conn.commit()
        except sqlite3.OperationalError:
            pass

        self._log(f"rebuilt {count} memory vectors")
        return {"re_encoded": count}

    # ── Sync Audit (Foundry) ───────────────────────────────────

    def sync_audit(self) -> dict:
        """Check SQLite <-> Foundry consistency."""
        conn = _get_conn()
        self._log("running sync audit...")

        local_count = 0
        try:
            local_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project = ?",
                (self.project,),
            ).fetchone()[0]
        except sqlite3.OperationalError:
            pass

        # Note: actual Foundry comparison requires remote connection
        # This provides the local side of the audit
        return {
            "local_count": local_count,
            "foundry_connected": False,
            "sync_needed": False,
            "message": "Foundry sync requires active connection — use /foundry to connect",
        }

    # ── Continuous Audit Check ─────────────────────────────────

    def continuous_audit_check(self) -> dict:
        """Lightweight check for audit triggers.

        Should run after /auto completion, after Morgoth phase transitions,
        after every 100 new memories, on session start.
        """
        conn = _get_conn()

        total = 0
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project = ?",
                (self.project,),
            ).fetchone()[0]
        except sqlite3.OperationalError:
            pass

        # Quick garbage scan (sample 50)
        garbage_count = 0
        sampled = 0
        try:
            rows = conn.execute(
                "SELECT id, raw_text, source, aif_confidence FROM memories WHERE project = ? ORDER BY RANDOM() LIMIT 50",
                (self.project,),
            ).fetchall()
            sampled = len(rows)
            for row in rows:
                result = self.score_memory(dict(row))
                if result["score"] < 3:
                    garbage_count += 1
        except sqlite3.OperationalError:
            pass

        garbage_rate = garbage_count / max(sampled, 1)
        needs_audit = garbage_rate > 0.3

        return {
            "total_memories": total,
            "sample_garbage_rate": round(garbage_rate, 2),
            "needs_full_audit": needs_audit,
            "needs_nuclear": garbage_rate > 0.3,
        }

    # ── Full Audit Report ──────────────────────────────────────

    def run_full_audit(self) -> dict:
        """Run a complete audit of the knowledge base."""
        self._log("running full audit...")

        garbage = self.find_garbage()
        duplicates = self.find_duplicates()
        continuous = self.continuous_audit_check()

        # Count by category
        total_issues = (
            len(garbage["garbage_memories"])
            + len(garbage["garbage_entities"])
            + len(duplicates["duplicate_memories"])
            + len(duplicates["duplicate_entities"])
        )

        report = {
            "project": self.project,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_issues": total_issues,
            "garbage": garbage,
            "duplicates": duplicates,
            "continuous_check": continuous,
            "health": "healthy" if total_issues < 10 else "needs_attention" if total_issues < 50 else "critical",
        }

        self._log(f"audit complete: {total_issues} issues found ({report['health']})")
        return report

    def format_report(self, report: Optional[dict] = None) -> str:
        """Format audit report as readable string."""
        if report is None:
            report = self.run_full_audit()

        lines = []
        lines.append(f"═══ AUDIT REPORT: {self.project} ═══")
        lines.append(f"Status: {report['health'].upper()}")
        lines.append(f"Total issues: {report['total_issues']}")
        lines.append("")

        g = report["garbage"]
        lines.append(f"GARBAGE: {len(g['garbage_memories'])} memories, {len(g['garbage_entities'])} entities")
        for m in g["garbage_memories"][:5]:
            lines.append(f"  [{m['score']:.0f}] {m['text'][:60]}... ({m['reason']})")

        lines.append("")
        d = report["duplicates"]
        lines.append(f"DUPLICATES: {len(d['duplicate_memories'])} memories, {len(d['duplicate_entities'])} entities")
        for dup in d["duplicate_memories"][:5]:
            lines.append(f"  {dup['text_a'][:40]} ≈ {dup['text_b'][:40]} ({dup['similarity']:.2f})")

        lines.append("")
        c = report["continuous_check"]
        lines.append(f"CONTINUOUS: garbage_rate={c['sample_garbage_rate']:.0%}")
        if c["needs_full_audit"]:
            lines.append("  ⚠ Full audit recommended")
        if c["needs_nuclear"]:
            lines.append("  🔴 Nuclear option may be needed (/audit --nuke)")

        return "\n".join(lines)
