"""Tests for the Foundry Audit — Knowledge Quality Enforcement (audit module)."""

import sqlite3
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from one import audit as mod
from one.audit import AuditEngine
from one.hdc import DIM, encode_text


def _setup_tables(conn):
    """Create all tables referenced by the audit module."""
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
            aif_confidence REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            observation_count INTEGER DEFAULT 1
        );
    """)
    conn.commit()


def _insert_memory(conn, project, raw_text="This is a good test memory that shows results",
                    source="user", aif_confidence=0.5, hdc_vector=None):
    """Insert a memory row for testing."""
    mid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    blob = None
    if hdc_vector is not None:
        blob = hdc_vector.astype(np.float32).tobytes()
    conn.execute(
        "INSERT INTO memories (id, raw_text, source, timestamp, project, aif_confidence, hdc_vector) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (mid, raw_text, source, now, project, aif_confidence, blob),
    )
    conn.commit()
    return mid


def _insert_entity(conn, project, name="TestEntity", entity_type="concept",
                    observation_count=1):
    """Insert an entity row for testing."""
    conn.execute(
        "INSERT INTO entities (name, type, observation_count) "
        "VALUES (?, ?, ?)",
        (name, entity_type, observation_count),
    )
    conn.commit()


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("one.audit.DB_PATH", db_path)
    monkeypatch.setattr("one.audit.DB_DIR", str(tmp_path))
    mod._local = threading.local()
    # Pre-create tables
    conn = mod._get_conn()
    _setup_tables(conn)
    yield db_path


@pytest.fixture
def engine():
    return AuditEngine("test_project")


# ── score_memory ──────────────────────────────────────────────


class TestScoreMemory:
    def test_empty_text(self, engine):
        result = engine.score_memory({"raw_text": "", "source": "user"})
        assert result["score"] == 0
        assert "empty" in result["reason"]

    def test_short_text(self, engine):
        result = engine.score_memory({"raw_text": "hi", "source": "user"})
        assert result["score"] == 0

    def test_good_text(self, engine):
        result = engine.score_memory({
            "raw_text": "The experiment shows that caching reduces latency because of locality",
            "source": "user",
            "aif_confidence": 0.5,
        })
        # Should score above 5 (has "shows" and "because" markers)
        assert result["score"] >= 5

    def test_json_text_penalized(self, engine):
        result = engine.score_memory({
            "raw_text": '{"key": "value", "data": [1,2,3], "some extra text to make it long enough"}',
            "source": "user",
            "aif_confidence": 0.5,
        })
        assert result["score"] < 5

    def test_garbled_text(self, engine):
        result = engine.score_memory({
            "raw_text": "$$$$####@@@@!!!!****&&&&%%%%^^^^",
            "source": "user",
            "aif_confidence": 0.5,
        })
        assert result["score"] <= 1
        assert "garbled" in result["reason"]

    def test_long_text_penalty(self, engine):
        long_text = "This is a valid sentence that shows results. " * 200  # > 5000 chars
        result = engine.score_memory({
            "raw_text": long_text,
            "source": "user",
            "aif_confidence": 0.5,
        })
        # Should still score reasonably but get -1 for length
        assert result["score"] <= 5

    def test_high_confidence_low_source(self, engine):
        result = engine.score_memory({
            "raw_text": "This memory indicates something useful for future reference and analysis",
            "source": "unverified",
            "aif_confidence": 0.9,
        })
        # Gets -2 for high confidence from unverified source
        assert result["score"] <= 5

    def test_useful_markers_boost(self, engine):
        base = engine.score_memory({
            "raw_text": "The sky is blue and the grass is green and we like it",
            "source": "user",
            "aif_confidence": 0.5,
        })
        with_markers = engine.score_memory({
            "raw_text": "The result indicates that caching therefore improves performance",
            "source": "user",
            "aif_confidence": 0.5,
        })
        assert with_markers["score"] >= base["score"]

    def test_score_clamps_to_0_10(self, engine):
        result = engine.score_memory({
            "raw_text": '{"json": true, "makes it worse and also very long"}' + "x " * 3000,
            "source": "ai-generated",
            "aif_confidence": 0.95,
        })
        assert 0 <= result["score"] <= 10


# ── score_entity ──────────────────────────────────────────────


class TestScoreEntity:
    def test_good_entity(self, engine):
        result = engine.score_entity({
            "name": "Python",
            "entity_type": "concept",
            "observation_count": 5,
        })
        assert result["score"] >= 5
        assert result["issues"] == []

    def test_stopword_entity(self, engine):
        result = engine.score_entity({
            "name": "the",
            "entity_type": "concept",
            "observation_count": 1,
        })
        assert result["score"] < 5
        assert any("not a real concept" in i for i in result["issues"])

    def test_single_char_entity(self, engine):
        result = engine.score_entity({
            "name": "a",
            "entity_type": "concept",
            "observation_count": 1,
        })
        assert result["score"] < 5

    def test_orphan_entity(self, engine):
        result = engine.score_entity({
            "name": "OrphanConcept",
            "entity_type": "concept",
            "observation_count": 0,
        })
        assert any("orphan" in i for i in result["issues"])

    def test_unusual_type(self, engine):
        result = engine.score_entity({
            "name": "Widget",
            "entity_type": "gadget",
            "observation_count": 3,
        })
        assert any("unusual" in i for i in result["issues"])

    def test_valid_type_no_issue(self, engine):
        for valid_type in ["concept", "file", "method", "person", "tool"]:
            result = engine.score_entity({
                "name": "ValidEntity",
                "entity_type": valid_type,
                "observation_count": 2,
            })
            type_issues = [i for i in result["issues"] if "unusual" in i]
            assert type_issues == []


# ── score_rule ────────────────────────────────────────────────


class TestScoreRule:
    def test_good_rule(self, engine):
        result = engine.score_rule({
            "rule_text": "Always validate user input before database insertion to prevent SQL injection",
            "source_count": 5,
            "confidence": 0.9,
        })
        assert result["score"] >= 7
        assert result["issues"] == []

    def test_short_rule(self, engine):
        result = engine.score_rule({
            "rule_text": "be good",
            "source_count": 1,
            "confidence": 0.5,
        })
        assert result["score"] < 5
        assert any("too short" in i for i in result["issues"])

    def test_vague_rule(self, engine):
        result = engine.score_rule({
            "rule_text": "always use best practices when writing code and deploying things",
            "source_count": 1,
            "confidence": 0.5,
        })
        assert any("too vague" in i for i in result["issues"])

    def test_unreinforced_rule(self, engine):
        result = engine.score_rule({
            "rule_text": "Use dependency injection for testable service constructors",
            "source_count": 0,
            "confidence": 0.5,
        })
        assert any("never reinforced" in i for i in result["issues"])

    def test_well_reinforced_rule(self, engine):
        result = engine.score_rule({
            "rule_text": "Use dependency injection for testable service constructors",
            "source_count": 5,
            "confidence": 0.8,
        })
        # source_count >= 3 gives +2, confidence > 0.7 gives +1
        assert result["score"] >= 8

    def test_empty_rule(self, engine):
        result = engine.score_rule({"rule_text": "", "source_count": 0, "confidence": 0.5})
        assert result["score"] < 3


# ── find_duplicates ───────────────────────────────────────────


class TestFindDuplicates:
    def test_no_duplicates_empty(self, engine):
        result = engine.find_duplicates()
        assert result["duplicate_memories"] == []
        assert result["duplicate_entities"] == []

    def test_memory_duplicates_by_vector(self, engine):
        conn = mod._get_conn()
        vec = encode_text("caching improves performance significantly")
        # Insert same vector twice
        _insert_memory(conn, "test_project", raw_text="caching improves performance significantly", hdc_vector=vec)
        _insert_memory(conn, "test_project", raw_text="caching improves performance significantly duplicate", hdc_vector=vec)
        result = engine.find_duplicates(threshold=0.9)
        assert len(result["duplicate_memories"]) >= 1

    def test_entity_duplicates_case_insensitive(self, engine):
        conn = mod._get_conn()
        _insert_entity(conn, "test_project", name="Python", entity_type="concept")
        _insert_entity(conn, "test_project", name="python", entity_type="concept")
        result = engine.find_duplicates()
        assert len(result["duplicate_entities"]) >= 1
        assert result["duplicate_entities"][0]["count"] == 2

    def test_no_false_positives(self, engine):
        conn = mod._get_conn()
        vec1 = encode_text("machine learning algorithms for classification")
        vec2 = encode_text("cooking recipes for italian pasta dishes")
        _insert_memory(conn, "test_project", raw_text="machine learning algorithms", hdc_vector=vec1)
        _insert_memory(conn, "test_project", raw_text="cooking recipes", hdc_vector=vec2)
        result = engine.find_duplicates(threshold=0.95)
        assert len(result["duplicate_memories"]) == 0


# ── find_garbage ──────────────────────────────────────────────


class TestFindGarbage:
    def test_no_garbage_empty(self, engine):
        result = engine.find_garbage()
        assert result["garbage_memories"] == []
        assert result["garbage_entities"] == []

    def test_finds_garbage_memory(self, engine):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project", raw_text="hi")  # too short
        result = engine.find_garbage(score_threshold=3.0)
        assert len(result["garbage_memories"]) >= 1

    def test_finds_garbage_entity(self, engine):
        conn = mod._get_conn()
        _insert_entity(conn, "test_project", name="the", entity_type="concept",
                        observation_count=0)
        result = engine.find_garbage(score_threshold=3.0)
        assert len(result["garbage_entities"]) >= 1

    def test_good_content_not_garbage(self, engine):
        conn = mod._get_conn()
        _insert_memory(
            conn, "test_project",
            raw_text="The conclusion shows that distributed caching therefore reduces latency in microservices",
        )
        _insert_entity(conn, "test_project", name="Redis", entity_type="tool",
                        observation_count=5)
        result = engine.find_garbage(score_threshold=3.0)
        assert len(result["garbage_memories"]) == 0
        assert len(result["garbage_entities"]) == 0

    def test_custom_threshold(self, engine):
        conn = mod._get_conn()
        _insert_memory(
            conn, "test_project",
            raw_text="A normal memory about software engineering that seems adequate overall",
        )
        # With a very high threshold, even decent memories become "garbage"
        result = engine.find_garbage(score_threshold=9.0)
        assert len(result["garbage_memories"]) >= 1


# ── auto_fix ──────────────────────────────────────────────────


class TestAutoFix:
    def test_auto_fix_deletes_garbage(self, engine):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project", raw_text="x")  # garbage: too short
        stats = engine.auto_fix()
        assert stats["deleted_memories"] >= 1
        # Verify actually deleted
        count = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE project = ?", ("test_project",)
        ).fetchone()[0]
        assert count == 0

    def test_auto_fix_reencodes_missing_vectors(self, engine):
        conn = mod._get_conn()
        _insert_memory(
            conn, "test_project",
            raw_text="This is a valid sentence that shows important results for the analysis",
            hdc_vector=None,
        )
        stats = engine.auto_fix()
        assert stats["re_encoded"] >= 1

    def test_auto_fix_reencodes_zero_vectors(self, engine):
        conn = mod._get_conn()
        zero_vec = np.zeros(DIM, dtype=np.float32)
        _insert_memory(
            conn, "test_project",
            raw_text="This is a valid sentence that shows important results for the analysis",
            hdc_vector=zero_vec,
        )
        stats = engine.auto_fix()
        assert stats["re_encoded"] >= 1

    def test_auto_fix_merges_duplicate_entities(self, engine):
        conn = mod._get_conn()
        _insert_entity(conn, "test_project", name="Python", entity_type="concept",
                        observation_count=5)
        _insert_entity(conn, "test_project", name="python", entity_type="concept",
                        observation_count=3)
        stats = engine.auto_fix()
        assert stats["merged_entities"] >= 1
        # Only one should remain
        rows = conn.execute(
            "SELECT * FROM entities WHERE name IN ('Python', 'python')"
        ).fetchall()
        remaining = [dict(r) for r in rows]
        if len(remaining) > 0:
            # The kept entity should have combined observations
            assert remaining[0]["observation_count"] >= 5

    def test_auto_fix_deletes_orphans(self, engine):
        conn = mod._get_conn()
        _insert_entity(conn, "test_project", name="OrphanEntity",
                        entity_type="concept", observation_count=0)
        stats = engine.auto_fix()
        assert stats["deleted_orphans"] >= 1

    def test_auto_fix_tags_unverified(self, engine):
        conn = mod._get_conn()
        _insert_memory(
            conn, "test_project",
            raw_text="This finding indicates that something useful was found and should be noted",
            source="unverified",
            aif_confidence=0.5,
        )
        stats = engine.auto_fix()
        assert stats["tagged_unverified"] >= 1
        # Check the text was prefixed
        row = conn.execute(
            "SELECT raw_text FROM memories WHERE project = ? AND source = 'unverified'",
            ("test_project",),
        ).fetchone()
        if row:
            assert row[0].startswith("[UNVERIFIED]")

    def test_auto_fix_no_double_tag(self, engine):
        conn = mod._get_conn()
        _insert_memory(
            conn, "test_project",
            raw_text="[UNVERIFIED] Already tagged memory that indicates valid results worth keeping",
            source="unverified",
            aif_confidence=0.5,
        )
        stats = engine.auto_fix()
        assert stats["tagged_unverified"] == 0

    def test_auto_fix_returns_all_stat_keys(self, engine):
        stats = engine.auto_fix()
        assert "deleted_memories" in stats
        assert "re_encoded" in stats
        assert "merged_entities" in stats
        assert "deleted_orphans" in stats
        assert "tagged_unverified" in stats

    def test_auto_fix_on_empty_db(self, engine):
        stats = engine.auto_fix()
        assert stats["deleted_memories"] == 0
        assert stats["re_encoded"] == 0
        assert stats["merged_entities"] == 0
        assert stats["deleted_orphans"] == 0
        assert stats["tagged_unverified"] == 0


# ── rebuild_pipeline ──────────────────────────────────────────


class TestRebuildPipeline:
    def test_rebuild_empty(self, engine):
        result = engine.rebuild_pipeline()
        assert result["re_encoded"] == 0

    def test_rebuild_reencodes_all(self, engine):
        conn = mod._get_conn()
        _insert_memory(
            conn, "test_project",
            raw_text="Memory one about interesting analysis results and conclusions",
        )
        _insert_memory(
            conn, "test_project",
            raw_text="Memory two about different experimental findings and outcomes",
        )
        result = engine.rebuild_pipeline()
        assert result["re_encoded"] == 2

    def test_rebuild_updates_vectors(self, engine):
        conn = mod._get_conn()
        _insert_memory(
            conn, "test_project",
            raw_text="Memory to rebuild with new vectors showing results",
            hdc_vector=None,
        )
        engine.rebuild_pipeline()
        # After rebuild, the memory should have a vector
        row = conn.execute(
            "SELECT hdc_vector FROM memories WHERE project = ?", ("test_project",)
        ).fetchone()
        assert row["hdc_vector"] is not None
        # encode_text returns float64 vectors, so tobytes() stores them as float64
        vec = np.frombuffer(row["hdc_vector"], dtype=np.float64)
        assert len(vec) == DIM


# ── continuous_audit_check ────────────────────────────────────


class TestContinuousAuditCheck:
    def test_empty_db(self, engine):
        result = engine.continuous_audit_check()
        assert result["total_memories"] == 0
        assert result["sample_garbage_rate"] == 0.0
        assert result["needs_full_audit"] is False

    def test_clean_data_no_audit_needed(self, engine):
        conn = mod._get_conn()
        for i in range(10):
            _insert_memory(
                conn, "test_project",
                raw_text=f"Valid finding number {i} that shows useful results and conclusions",
            )
        result = engine.continuous_audit_check()
        assert result["total_memories"] == 10
        assert result["needs_full_audit"] is False

    def test_mostly_garbage_triggers_audit(self, engine):
        conn = mod._get_conn()
        for i in range(20):
            _insert_memory(conn, "test_project", raw_text=f"x{i}")  # all too short
        result = engine.continuous_audit_check()
        assert result["sample_garbage_rate"] > 0.3
        assert result["needs_full_audit"] is True
        assert result["needs_nuclear"] is True

    def test_result_structure(self, engine):
        result = engine.continuous_audit_check()
        assert "total_memories" in result
        assert "sample_garbage_rate" in result
        assert "needs_full_audit" in result
        assert "needs_nuclear" in result


# ── run_full_audit ────────────────────────────────────────────


class TestRunFullAudit:
    def test_full_audit_empty(self, engine):
        report = engine.run_full_audit()
        assert report["project"] == "test_project"
        assert report["total_issues"] == 0
        assert report["health"] == "healthy"

    def test_full_audit_with_issues(self, engine):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project", raw_text="x")  # garbage
        _insert_entity(conn, "test_project", name="the",
                        entity_type="concept", observation_count=0)
        report = engine.run_full_audit()
        assert report["total_issues"] >= 2

    def test_full_audit_healthy_threshold(self, engine):
        conn = mod._get_conn()
        for i in range(5):
            _insert_memory(
                conn, "test_project",
                raw_text=f"This is a good quality memory number {i} that shows useful results and conclusions",
            )
        report = engine.run_full_audit()
        assert report["health"] == "healthy"

    def test_full_audit_structure(self, engine):
        report = engine.run_full_audit()
        assert "garbage" in report
        assert "duplicates" in report
        assert "continuous_check" in report
        assert "timestamp" in report

    def test_full_audit_health_levels(self, engine):
        """Test the health classification thresholds."""
        conn = mod._get_conn()
        # Add lots of garbage to push past thresholds
        for i in range(60):
            _insert_memory(conn, "test_project", raw_text=f"z{i}")
        report = engine.run_full_audit()
        # With 60 garbage memories, total_issues >= 50 -> critical
        assert report["health"] == "critical"


# ── format_report ─────────────────────────────────────────────


class TestFormatReport:
    def test_format_produces_string(self, engine):
        output = engine.format_report()
        assert isinstance(output, str)
        assert "AUDIT REPORT" in output

    def test_format_contains_sections(self, engine):
        output = engine.format_report()
        assert "GARBAGE:" in output
        assert "DUPLICATES:" in output
        assert "CONTINUOUS:" in output
        assert "Status:" in output

    def test_format_shows_health_status(self, engine):
        report = engine.run_full_audit()
        output = engine.format_report(report)
        assert report["health"].upper() in output

    def test_format_from_provided_report(self, engine):
        report = engine.run_full_audit()
        output = engine.format_report(report)
        assert "AUDIT REPORT" in output

    def test_format_with_garbage_details(self, engine):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project", raw_text="x")  # garbage
        output = engine.format_report()
        assert "GARBAGE:" in output


# ── sync_audit ────────────────────────────────────────────────


class TestSyncAudit:
    def test_sync_audit_basic(self, engine):
        result = engine.sync_audit()
        assert result["local_count"] == 0
        assert result["foundry_connected"] is False

    def test_sync_audit_with_memories(self, engine):
        conn = mod._get_conn()
        _insert_memory(conn, "test_project", raw_text="Memory for sync audit testing that shows results")
        result = engine.sync_audit()
        assert result["local_count"] == 1


# ── Edge Cases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_on_log_callback(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "log_test.db")
        monkeypatch.setattr("one.audit.DB_PATH", db_path)
        monkeypatch.setattr("one.audit.DB_DIR", str(tmp_path))
        mod._local = threading.local()
        conn = mod._get_conn()
        _setup_tables(conn)

        logs = []
        eng = AuditEngine("test_project", on_log=logs.append)
        eng.find_garbage()
        assert any("scanning for garbage" in msg for msg in logs)

    def test_score_memory_llm(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.audit._call_ollama",
            lambda *a, **kw: "SCORE: 8\nREASON: very useful memory",
        )
        result = engine.score_memory_llm({
            "raw_text": "This memory is about testing",
            "source": "user",
            "aif_confidence": 0.5,
        })
        assert result["score"] == 8.0
        assert result["reason"] == "very useful memory"

    def test_score_memory_llm_returns_none(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.audit._call_ollama",
            lambda *a, **kw: None,
        )
        result = engine.score_memory_llm({
            "raw_text": "test",
            "source": "user",
            "aif_confidence": 0.5,
        })
        assert result["score"] == 5.0  # default

    def test_score_memory_llm_clamps(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.audit._call_ollama",
            lambda *a, **kw: "SCORE: 99\nREASON: overrated",
        )
        result = engine.score_memory_llm({
            "raw_text": "test",
            "source": "user",
            "aif_confidence": 0.5,
        })
        assert result["score"] == 10.0
