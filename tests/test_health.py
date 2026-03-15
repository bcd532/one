"""Tests for the Knowledge Health Metrics dashboard (health module)."""

import sqlite3
import threading
from datetime import datetime, timezone, timedelta

import pytest

from one import health as mod
from one.health import HealthDashboard


def _setup_tables(conn):
    """Create all tables referenced by the health module."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            raw_text TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            project TEXT DEFAULT 'global',
            hdc_vector BLOB,
            tm_label TEXT DEFAULT 'unclassified',
            regime_tag TEXT DEFAULT 'default',
            aif_confidence REAL DEFAULT 0.0,
            created TEXT
        );

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            observation_count INTEGER DEFAULT 1,
            project TEXT DEFAULT 'global'
        );

        CREATE TABLE IF NOT EXISTS universal_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT DEFAULT 'global',
            pattern TEXT
        );

        CREATE TABLE IF NOT EXISTS analogy_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT DEFAULT 'global',
            template TEXT
        );

        CREATE TABLE IF NOT EXISTS dialectic_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT DEFAULT 'global',
            status TEXT DEFAULT 'active'
        );

        CREATE TABLE IF NOT EXISTS contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT DEFAULT 'global',
            status TEXT DEFAULT 'active',
            severity TEXT DEFAULT 'minor',
            created TEXT
        );

        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT DEFAULT 'global',
            status TEXT DEFAULT 'designed'
        );

        CREATE TABLE IF NOT EXISTS verification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT DEFAULT 'global',
            created TEXT
        );

        CREATE TABLE IF NOT EXISTS knowledge_frontier (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT DEFAULT 'global',
            status TEXT DEFAULT 'unexplored'
        );

        CREATE TABLE IF NOT EXISTS playbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT DEFAULT 'global'
        );

        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT DEFAULT 'global',
            source_count INTEGER DEFAULT 0
        );
    """)
    conn.commit()


def _insert_memory(conn, project, raw_text="test memory", source="user",
                    aif_confidence=0.5, tm_label="unclassified", created=None):
    """Insert a memory row for testing."""
    import uuid
    mid = str(uuid.uuid4())
    now = created or datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memories (id, raw_text, source, timestamp, project, aif_confidence, tm_label, created) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (mid, raw_text, source, now, project, aif_confidence, tm_label, now),
    )
    conn.commit()
    return mid


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("one.health.DB_PATH", db_path)
    monkeypatch.setattr("one.health.DB_DIR", str(tmp_path))
    mod._local = threading.local()
    # Pre-create the schema
    conn = mod._get_conn()
    _setup_tables(conn)
    yield db_path


@pytest.fixture
def dash():
    return HealthDashboard("test_project")


# ── Volume ────────────────────────────────────────────────────


class TestVolume:
    def test_volume_empty(self, dash):
        v = dash.volume()
        assert v["total"] == 0
        assert v["high_confidence"] == 0
        assert v["medium_confidence"] == 0
        assert v["low_confidence"] == 0

    def test_volume_with_memories(self, dash):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project", aif_confidence=0.9)
        _insert_memory(conn, "test_project", aif_confidence=0.8)
        _insert_memory(conn, "test_project", aif_confidence=0.5)
        _insert_memory(conn, "test_project", aif_confidence=0.2)
        v = dash.volume()
        assert v["total"] == 4
        assert v["high_confidence"] == 2  # >= 0.7
        assert v["medium_confidence"] == 1  # >= 0.4 and < 0.7
        assert v["low_confidence"] == 1  # < 0.4

    def test_volume_percentages(self, dash):
        conn = mod._get_conn()
        for _ in range(3):
            _insert_memory(conn, "test_project", aif_confidence=0.9)
        _insert_memory(conn, "test_project", aif_confidence=0.1)
        v = dash.volume()
        assert v["high_pct"] == 75.0
        assert v["low_pct"] == 25.0

    def test_volume_ignores_other_project(self, dash):
        conn = mod._get_conn()
        _insert_memory(conn, "other_project", aif_confidence=0.9)
        _insert_memory(conn, "test_project", aif_confidence=0.5)
        v = dash.volume()
        assert v["total"] == 1


# ── Entities ──────────────────────────────────────────────────


class TestEntities:
    def test_entities_empty(self, dash):
        e = dash.entities()
        assert e["total"] == 0
        assert e["breakdown"] == {}

    def test_entities_breakdown(self, dash):
        conn = mod._get_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, project) VALUES (?, ?, ?)",
            ("Python", "concept", "test_project"),
        )
        conn.execute(
            "INSERT INTO entities (name, entity_type, project) VALUES (?, ?, ?)",
            ("main.py", "file", "test_project"),
        )
        conn.execute(
            "INSERT INTO entities (name, entity_type, project) VALUES (?, ?, ?)",
            ("FastAPI", "concept", "test_project"),
        )
        conn.commit()
        e = dash.entities()
        assert e["total"] == 3
        assert "concept" in e["breakdown"]
        assert e["breakdown"]["concept"]["count"] == 2
        assert "file" in e["breakdown"]
        assert e["breakdown"]["file"]["count"] == 1

    def test_entities_percentages(self, dash):
        conn = mod._get_conn()
        conn.execute(
            "INSERT INTO entities (name, entity_type, project) VALUES (?, ?, ?)",
            ("A", "concept", "test_project"),
        )
        conn.execute(
            "INSERT INTO entities (name, entity_type, project) VALUES (?, ?, ?)",
            ("B", "concept", "test_project"),
        )
        conn.execute(
            "INSERT INTO entities (name, entity_type, project) VALUES (?, ?, ?)",
            ("C", "file", "test_project"),
        )
        conn.commit()
        e = dash.entities()
        # concept: 2/3 = 66.7%, file: 1/3 = 33.3%
        assert e["breakdown"]["concept"]["pct"] == pytest.approx(66.7, abs=0.1)
        assert e["breakdown"]["file"]["pct"] == pytest.approx(33.3, abs=0.1)


# ── Intelligence ──────────────────────────────────────────────


class TestIntelligence:
    def test_intelligence_empty(self, dash):
        intel = dash.intelligence()
        assert intel["syntheses"] == 0
        assert intel["universal_patterns"] == 0
        assert intel["dialectic_chains"] == 0
        assert intel["experiments_passed"] == 0

    def test_intelligence_counts(self, dash):
        conn = mod._get_conn()
        # Add some synthesis memories
        _insert_memory(conn, "test_project", tm_label="synthesis")
        _insert_memory(conn, "test_project", tm_label="synthesis")
        # Add a universal pattern
        conn.execute(
            "INSERT INTO universal_patterns (project, pattern) VALUES (?, ?)",
            ("test_project", "pattern1"),
        )
        # Add experiments
        conn.execute(
            "INSERT INTO experiments (project, status) VALUES (?, ?)",
            ("test_project", "passed"),
        )
        conn.execute(
            "INSERT INTO experiments (project, status) VALUES (?, ?)",
            ("test_project", "failed"),
        )
        # Add dialectic chains
        conn.execute(
            "INSERT INTO dialectic_chains (project, status) VALUES (?, ?)",
            ("test_project", "verified"),
        )
        conn.execute(
            "INSERT INTO dialectic_chains (project, status) VALUES (?, ?)",
            ("test_project", "active"),
        )
        # Add rules
        conn.execute(
            "INSERT INTO rules (project, source_count) VALUES (?, ?)",
            ("test_project", 5),
        )
        conn.execute(
            "INSERT INTO rules (project, source_count) VALUES (?, ?)",
            ("test_project", 1),
        )
        conn.commit()
        intel = dash.intelligence()
        assert intel["syntheses"] == 2
        assert intel["universal_patterns"] == 1
        assert intel["experiments_passed"] == 1
        assert intel["experiments_failed"] == 1
        assert intel["dialectic_chains"] == 2
        assert intel["dialectic_complete"] == 1
        assert intel["rules_core"] == 1
        assert intel["rules_contextual"] == 1


# ── Quality ───────────────────────────────────────────────────


class TestQuality:
    def test_quality_empty(self, dash):
        q = dash.quality()
        assert q["coverage"] == 0.0
        assert q["freshness_pct"] == 0.0
        assert q["contradiction_rate"] == 0.0
        assert q["contradiction_healthy"] is True

    def test_quality_avg_confidence(self, dash):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project", aif_confidence=0.8)
        _insert_memory(conn, "test_project", aif_confidence=0.6)
        q = dash.quality()
        assert q["avg_confidence"] == pytest.approx(0.7, abs=0.01)

    def test_quality_freshness(self, dash):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project")
        # Add recent verification
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO verification_log (project, created) VALUES (?, ?)",
            ("test_project", now),
        )
        conn.commit()
        q = dash.quality()
        assert q["freshness_pct"] == 100.0

    def test_quality_contradiction_unhealthy(self, dash):
        conn = mod._get_conn()
        # Create memories so we have a total
        for _ in range(10):
            _insert_memory(conn, "test_project")
        # Create enough active contradictions to exceed 5%
        for _ in range(2):
            conn.execute(
                "INSERT INTO contradictions (project, status) VALUES (?, ?)",
                ("test_project", "active"),
            )
        conn.commit()
        q = dash.quality()
        assert q["contradiction_rate"] == 20.0
        assert q["contradiction_healthy"] is False

    def test_quality_coverage(self, dash):
        conn = mod._get_conn()
        conn.execute(
            "INSERT INTO knowledge_frontier (project, status) VALUES (?, ?)",
            ("test_project", "explored"),
        )
        conn.execute(
            "INSERT INTO knowledge_frontier (project, status) VALUES (?, ?)",
            ("test_project", "unexplored"),
        )
        conn.commit()
        q = dash.quality()
        assert q["coverage"] == 50.0


# ── Warnings ──────────────────────────────────────────────────


class TestWarnings:
    def test_no_warnings_on_empty(self, dash):
        w = dash.warnings()
        assert w == []

    def test_stale_findings_warning(self, dash):
        conn = mod._get_conn()
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        _insert_memory(conn, "test_project", aif_confidence=0.5, created=old_date)
        w = dash.warnings()
        stale_warnings = [x for x in w if "older than 30 days" in x["message"]]
        assert len(stale_warnings) == 1
        assert stale_warnings[0]["level"] == "warning"

    def test_critical_contradiction_warning(self, dash):
        conn = mod._get_conn()
        old_date = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
        conn.execute(
            "INSERT INTO contradictions (project, status, severity, created) VALUES (?, ?, ?, ?)",
            ("test_project", "active", "critical", old_date),
        )
        conn.commit()
        w = dash.warnings()
        crit_warnings = [x for x in w if "critical" in x["level"]]
        assert len(crit_warnings) == 1

    def test_failed_experiments_warning(self, dash):
        conn = mod._get_conn()
        conn.execute(
            "INSERT INTO experiments (project, status) VALUES (?, ?)",
            ("test_project", "failed"),
        )
        conn.commit()
        w = dash.warnings()
        exp_warnings = [x for x in w if "failed experiments" in x["message"]]
        assert len(exp_warnings) == 1
        assert exp_warnings[0]["level"] == "info"

    def test_low_confidence_warning(self, dash):
        conn = mod._get_conn()
        for _ in range(5):
            _insert_memory(conn, "test_project", aif_confidence=0.2)
        w = dash.warnings()
        conf_warnings = [x for x in w if "confidence is low" in x["message"]]
        assert len(conf_warnings) == 1


# ── full_report ───────────────────────────────────────────────


class TestFullReport:
    def test_full_report_structure(self, dash):
        report = dash.full_report()
        assert "project" in report
        assert report["project"] == "test_project"
        assert "timestamp" in report
        assert "volume" in report
        assert "entities" in report
        assert "intelligence" in report
        assert "quality" in report
        assert "warnings" in report

    def test_full_report_with_data(self, dash):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project", aif_confidence=0.9)
        report = dash.full_report()
        assert report["volume"]["total"] == 1


# ── format_report ─────────────────────────────────────────────


class TestFormatReport:
    def test_format_report_produces_string(self, dash):
        output = dash.format_report()
        assert isinstance(output, str)
        assert "KNOWLEDGE HEALTH" in output

    def test_format_report_contains_sections(self, dash):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project", aif_confidence=0.9)
        output = dash.format_report()
        assert "VOLUME:" in output
        assert "ENTITIES:" in output
        assert "INTELLIGENCE:" in output
        assert "QUALITY:" in output

    def test_format_report_healthy_no_warnings(self, dash):
        output = dash.format_report()
        assert "knowledge base is healthy" in output

    def test_format_report_with_warnings(self, dash):
        conn = mod._get_conn()
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        _insert_memory(conn, "test_project", aif_confidence=0.5, created=old_date)
        output = dash.format_report()
        assert "WARNINGS:" in output
        assert "older than 30 days" in output

    def test_format_report_from_provided_report(self, dash):
        """format_report can accept a pre-built report dict."""
        report = dash.full_report()
        output = dash.format_report(report)
        assert "KNOWLEDGE HEALTH" in output
