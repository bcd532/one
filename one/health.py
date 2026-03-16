"""Knowledge Health Metrics — The immune system of the knowledge graph.

System 9: Comprehensive health dashboard for the knowledge base.

Sections:
- VOLUME: total memories, confidence breakdown
- ENTITIES: concepts, files, methods, people
- INTELLIGENCE: syntheses, patterns, dialectics, contradictions, etc.
- QUALITY: coverage, avg confidence, freshness, contradiction rate
- WARNINGS: stale findings, unresolved contradictions, gaps
"""

import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable

from .store import DB_PATH, DB_DIR


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


def _safe_count(query: str, params: tuple[str, ...] = ()) -> int:
    """Execute a COUNT query, returning 0 if table doesn't exist."""
    try:
        return _get_conn().execute(query, params).fetchone()[0]
    except sqlite3.OperationalError:
        return 0


class HealthDashboard:
    """Generates comprehensive knowledge health reports."""

    def __init__(self, project: str, on_log: Optional[Callable] = None):
        self.project = project
        self._log = on_log or (lambda _m: None)

    def volume(self) -> dict:
        """Memory volume with confidence breakdown."""
        total = _safe_count(
            "SELECT COUNT(*) FROM memories WHERE project = ?",
            (self.project,),
        )

        high = _safe_count(
            "SELECT COUNT(*) FROM memories WHERE project = ? AND aif_confidence >= 0.7",
            (self.project,),
        )
        medium = _safe_count(
            "SELECT COUNT(*) FROM memories WHERE project = ? AND aif_confidence >= 0.4 AND aif_confidence < 0.7",
            (self.project,),
        )
        low = _safe_count(
            "SELECT COUNT(*) FROM memories WHERE project = ? AND aif_confidence < 0.4",
            (self.project,),
        )

        return {
            "total": total,
            "high_confidence": high,
            "medium_confidence": medium,
            "low_confidence": low,
            "high_pct": round(high / max(total, 1) * 100, 1),
            "medium_pct": round(medium / max(total, 1) * 100, 1),
            "low_pct": round(low / max(total, 1) * 100, 1),
        }

    def entities(self) -> dict:
        """Entity breakdown by type."""
        conn = _get_conn()
        result: dict[str, int] = {}

        try:
            rows = conn.execute(
                "SELECT type, COUNT(*) as cnt FROM entities GROUP BY type",
            ).fetchall()
            for row in rows:
                result[row["type"]] = row["cnt"]
        except sqlite3.OperationalError:
            pass

        total = sum(result.values())
        breakdown: dict[str, dict[str, object]] = {}
        for etype, count in result.items():
            breakdown[etype] = {
                "count": count,
                "pct": round(count / max(total, 1) * 100, 1),
            }

        return {
            "total": total,
            "breakdown": breakdown,
        }

    def intelligence(self) -> dict:
        """Intelligence metrics: syntheses, patterns, dialectics, etc."""
        return {
            "syntheses": _safe_count(
                "SELECT COUNT(*) FROM memories WHERE project = ? AND tm_label = 'synthesis'",
                (self.project,),
            ),
            "universal_patterns": _safe_count(
                "SELECT COUNT(*) FROM universal_patterns WHERE project = ?",
                (self.project,),
            ),
            "analogy_templates": _safe_count(
                "SELECT COUNT(*) FROM analogy_templates WHERE project = ?",
                (self.project,),
            ),
            "dialectic_chains": _safe_count(
                "SELECT COUNT(*) FROM dialectic_chains WHERE project = ?",
                (self.project,),
            ),
            "dialectic_complete": _safe_count(
                "SELECT COUNT(*) FROM dialectic_chains WHERE project = ? AND status = 'verified'",
                (self.project,),
            ),
            "contradictions_active": _safe_count(
                "SELECT COUNT(*) FROM contradictions WHERE project = ? AND status = 'active'",
                (self.project,),
            ),
            "contradictions_resolved": _safe_count(
                "SELECT COUNT(*) FROM contradictions WHERE project = ? AND status = 'resolved'",
                (self.project,),
            ),
            "experiments_passed": _safe_count(
                "SELECT COUNT(*) FROM experiments WHERE project = ? AND status = 'passed'",
                (self.project,),
            ),
            "experiments_failed": _safe_count(
                "SELECT COUNT(*) FROM experiments WHERE project = ? AND status = 'failed'",
                (self.project,),
            ),
            "playbooks": _safe_count(
                "SELECT COUNT(*) FROM playbooks WHERE project = ?",
                (self.project,),
            ),
            "rules_core": _safe_count(
                "SELECT COUNT(*) FROM rule_nodes WHERE project = ? AND source_count >= 3",
                (self.project,),
            ),
            "rules_contextual": _safe_count(
                "SELECT COUNT(*) FROM rule_nodes WHERE project = ? AND source_count < 3",
                (self.project,),
            ),
        }

    def quality(self) -> dict:
        """Quality metrics: coverage, confidence, freshness, contradiction rate."""
        conn = _get_conn()

        # Average confidence
        avg_conf = 0.5
        try:
            row = conn.execute(
                "SELECT AVG(aif_confidence) FROM memories WHERE project = ?",
                (self.project,),
            ).fetchone()
            if row and row[0] is not None:
                avg_conf = round(float(row[0]), 3)
        except sqlite3.OperationalError:
            pass

        # Freshness: % verified within 7 days
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        recent_verified = _safe_count(
            "SELECT COUNT(*) FROM verification_log WHERE project = ? AND created > ?",
            (self.project, seven_days_ago),
        )
        total_memories = _safe_count(
            "SELECT COUNT(*) FROM memories WHERE project = ?",
            (self.project,),
        )
        freshness = round(recent_verified / max(total_memories, 1) * 100, 1)

        # Contradiction rate
        total_findings = max(total_memories, 1)
        active_contradictions = _safe_count(
            "SELECT COUNT(*) FROM contradictions WHERE project = ? AND status = 'active'",
            (self.project,),
        )
        contradiction_rate = round(active_contradictions / total_findings * 100, 2)

        # Source quality average
        avg_source = 0.5
        try:
            row = conn.execute(
                "SELECT AVG(aif_confidence) FROM memories WHERE project = ? AND source != 'user'",
                (self.project,),
            ).fetchone()
            if row and row[0] is not None:
                avg_source = round(float(row[0]), 3)
        except sqlite3.OperationalError:
            pass

        # Frontier coverage
        coverage = 0.0
        try:
            total_frontier = conn.execute(
                "SELECT COUNT(*) FROM knowledge_frontier WHERE project = ?",
                (self.project,),
            ).fetchone()[0]
            explored_frontier = conn.execute(
                "SELECT COUNT(*) FROM knowledge_frontier WHERE project = ? AND status = 'explored'",
                (self.project,),
            ).fetchone()[0]
            if total_frontier > 0:
                coverage = round(explored_frontier / total_frontier * 100, 1)
        except sqlite3.OperationalError:
            pass

        return {
            "coverage": coverage,
            "avg_confidence": avg_conf,
            "freshness_pct": freshness,
            "contradiction_rate": contradiction_rate,
            "contradiction_healthy": contradiction_rate < 5.0,
            "avg_source_quality": avg_source,
        }

    def warnings(self) -> list[dict[str, str]]:
        """Generate warnings for knowledge health issues."""
        warnings: list[dict[str, str]] = []

        # Stale findings (>30 days without verification)
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        stale = _safe_count(
            "SELECT COUNT(*) FROM memories WHERE project = ? AND timestamp < ? AND aif_confidence > 0.3",
            (self.project, thirty_days_ago),
        )
        if stale > 0:
            warnings.append({
                "level": "warning",
                "message": f"{stale} findings older than 30 days need re-verification",
            })

        # Critical contradictions unresolved >72 hours
        three_days_ago = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        critical_old = _safe_count(
            "SELECT COUNT(*) FROM contradictions WHERE project = ? AND status = 'active' AND severity IN ('critical', 'paradigm') AND created < ?",
            (self.project, three_days_ago),
        )
        if critical_old > 0:
            warnings.append({
                "level": "critical",
                "message": f"{critical_old} critical/paradigm contradictions unresolved for 72+ hours",
            })

        # Failed experiments needing revision
        failed_experiments = _safe_count(
            "SELECT COUNT(*) FROM experiments WHERE project = ? AND status = 'failed'",
            (self.project,),
        )
        if failed_experiments > 0:
            warnings.append({
                "level": "info",
                "message": f"{failed_experiments} failed experiments need hypothesis revision",
            })

        # Low overall confidence
        quality = self.quality()
        if quality["avg_confidence"] < 0.4:
            warnings.append({
                "level": "warning",
                "message": f"Average confidence is low ({quality['avg_confidence']:.2f})",
            })

        # High contradiction rate
        if not quality["contradiction_healthy"]:
            warnings.append({
                "level": "warning",
                "message": f"Contradiction rate is {quality['contradiction_rate']}% (healthy: <5%)",
            })

        return warnings

    def full_report(self) -> dict:
        """Generate the complete health dashboard."""
        self._log("generating health report...")

        report = {
            "project": self.project,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "volume": self.volume(),
            "entities": self.entities(),
            "intelligence": self.intelligence(),
            "quality": self.quality(),
            "warnings": self.warnings(),
        }

        self._log(f"health report complete: {report['volume']['total']} memories")
        return report

    def format_report(self, report: Optional[dict] = None) -> str:
        """Format a health report as a readable string."""
        if report is None:
            report = self.full_report()

        lines = []
        lines.append(f"═══ KNOWLEDGE HEALTH: {self.project} ═══")
        lines.append("")

        # Volume
        vol = report["volume"]
        lines.append(f"VOLUME: {vol['total']} memories")
        lines.append(f"  High confidence:   {vol['high_confidence']:>4} ({vol['high_pct']}%)")
        lines.append(f"  Medium confidence: {vol['medium_confidence']:>4} ({vol['medium_pct']}%)")
        lines.append(f"  Low confidence:    {vol['low_confidence']:>4} ({vol['low_pct']}%)")
        lines.append("")

        # Entities
        ent = report["entities"]
        lines.append(f"ENTITIES: {ent['total']} total")
        for etype, info in ent.get("breakdown", {}).items():
            lines.append(f"  {etype:<15} {info['count']:>4} ({info['pct']}%)")
        lines.append("")

        # Intelligence
        intel = report["intelligence"]
        lines.append("INTELLIGENCE:")
        lines.append(f"  Syntheses:         {intel.get('syntheses', 0)}")
        lines.append(f"  Universal patterns:{intel.get('universal_patterns', 0)}")
        lines.append(f"  Dialectics:        {intel.get('dialectic_chains', 0)} ({intel.get('dialectic_complete', 0)} complete)")
        lines.append(f"  Contradictions:    {intel.get('contradictions_resolved', 0)} resolved / {intel.get('contradictions_active', 0)} active")
        lines.append(f"  Experiments:       {intel.get('experiments_passed', 0)} passed / {intel.get('experiments_failed', 0)} failed")
        lines.append(f"  Rules:             {intel.get('rules_core', 0)} core / {intel.get('rules_contextual', 0)} contextual")
        lines.append("")

        # Quality
        qual = report["quality"]
        lines.append("QUALITY:")
        lines.append(f"  Coverage:          {qual['coverage']}%")
        lines.append(f"  Avg confidence:    {qual['avg_confidence']}")
        lines.append(f"  Freshness:         {qual['freshness_pct']}% verified <7d")
        lines.append(f"  Contradiction rate:{qual['contradiction_rate']}% {'✓' if qual['contradiction_healthy'] else '✗'}")
        lines.append("")

        # Warnings
        warns = report["warnings"]
        if warns:
            lines.append("WARNINGS:")
            for w in warns:
                icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(w["level"], "·")
                lines.append(f"  {icon} {w['message']}")
        else:
            lines.append("WARNINGS: none — knowledge base is healthy")

        return "\n".join(lines)
