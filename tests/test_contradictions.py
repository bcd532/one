"""Tests for the Contradiction Mining Engine."""

import sqlite3
import threading
import pytest

from one import contradictions as cm
from one.contradictions import ContradictionMiner, SEVERITY_LEVELS, init_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a temporary database for each test, fully isolated."""
    db_path = str(tmp_path / "test_contradictions.db")
    db_dir = str(tmp_path)

    # Patch DB_PATH/DB_DIR on the contradictions module itself so that
    # _get_conn() picks up the temp path instead of the real one.
    monkeypatch.setattr("one.contradictions.DB_PATH", db_path)
    monkeypatch.setattr("one.contradictions.DB_DIR", db_dir)

    # Also patch store so that push_memory / recall use the same temp DB.
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", db_dir)

    # Reset thread-local connections in both modules.
    cm._local = threading.local()
    import one.store as store_mod
    store_mod._local = threading.local()

    yield db_path


@pytest.fixture()
def miner(monkeypatch):
    """Return a ContradictionMiner for the 'test_project' project.

    _call_ollama is patched to a no-op by default; individual tests can
    override it.
    """
    monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: None)
    monkeypatch.setattr("one.contradictions.push_memory", lambda *a, **kw: "mock-id")
    return ContradictionMiner("test_project")


def _insert_contradiction(miner, severity="moderate", status="active",
                          finding_a="A claims X", finding_b="B claims Y",
                          resolution=None, resolution_type=None,
                          resolved_at=None):
    """Helper: insert a contradiction row directly via the module connection."""
    from datetime import datetime, timezone
    conn = cm._get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO contradictions
           (project, finding_a, finding_b, severity, status, resolution,
            resolution_type, created, resolved_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (miner.project, finding_a, finding_b, severity, status,
         resolution, resolution_type, now, resolved_at),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 1. Schema creation
# ---------------------------------------------------------------------------

class TestSchema:
    def test_init_schema_creates_table(self):
        """init_schema should create the contradictions table."""
        init_schema()
        conn = cm._get_conn()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='contradictions'"
        ).fetchall()
        assert len(rows) == 1

    def test_init_schema_creates_indexes(self):
        """Three indexes should exist after init_schema."""
        init_schema()
        conn = cm._get_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_contradictions_%'"
        ).fetchall()
        index_names = {r["name"] for r in indexes}
        assert "idx_contradictions_project" in index_names
        assert "idx_contradictions_severity" in index_names
        assert "idx_contradictions_status" in index_names

    def test_init_schema_idempotent(self):
        """Calling init_schema twice should not raise."""
        init_schema()
        init_schema()  # second call is a no-op


# ---------------------------------------------------------------------------
# 2. mine_contradictions
# ---------------------------------------------------------------------------

class TestMineContradictions:
    def test_returns_empty_for_fewer_than_two_findings(self, monkeypatch, miner):
        """If recall returns < 2 findings, there's nothing to compare."""
        monkeypatch.setattr("one.contradictions.recall", lambda *a, **kw: [])
        assert miner.mine_contradictions() == []

    def test_returns_empty_for_one_finding(self, monkeypatch, miner):
        monkeypatch.setattr("one.contradictions.recall", lambda *a, **kw: [
            {"id": "1", "raw_text": "X increases performance", "source": "paper1"},
        ])
        assert miner.mine_contradictions() == []

    def test_detects_negation_contradiction(self, monkeypatch, miner):
        """Two findings with negation asymmetry and topical overlap."""
        monkeypatch.setattr("one.contradictions.recall", lambda *a, **kw: [
            {"id": "1", "raw_text": "The drug increases cancer survival rates significantly",
             "source": "paper1"},
            {"id": "2", "raw_text": "The drug does not increase cancer survival rates",
             "source": "paper2"},
        ])
        results = miner.mine_contradictions()
        assert len(results) == 1
        assert results[0]["finding_a_source"] == "paper1"

    def test_skips_same_source(self, monkeypatch, miner):
        """Findings from the same source are not compared."""
        monkeypatch.setattr("one.contradictions.recall", lambda *a, **kw: [
            {"id": "1", "raw_text": "The drug increases cancer survival rates",
             "source": "paper1"},
            {"id": "2", "raw_text": "The drug does not increase cancer survival rates",
             "source": "paper1"},
        ])
        assert miner.mine_contradictions() == []

    def test_quantitative_contradiction(self, monkeypatch, miner):
        """Different percentages on the same topic should be flagged."""
        monkeypatch.setattr("one.contradictions.recall", lambda *a, **kw: [
            {"id": "1", "raw_text": "Model accuracy reached 95% training data",
             "source": "a"},
            {"id": "2", "raw_text": "Model accuracy reached 72% training data",
             "source": "b"},
        ])
        results = miner.mine_contradictions()
        assert len(results) == 1

    def test_no_contradiction_without_overlap(self, monkeypatch, miner):
        """Two unrelated texts should not be flagged."""
        monkeypatch.setattr("one.contradictions.recall", lambda *a, **kw: [
            {"id": "1", "raw_text": "Python has dynamic typing", "source": "a"},
            {"id": "2", "raw_text": "The sunset was beautiful tonight", "source": "b"},
        ])
        results = miner.mine_contradictions()
        assert len(results) == 0

    def test_respects_limit(self, monkeypatch, miner):
        """limit parameter is forwarded to recall."""
        captured = {}

        def fake_recall(*args, **kwargs):
            captured["n"] = kwargs.get("n")
            return []

        monkeypatch.setattr("one.contradictions.recall", fake_recall)
        miner.mine_contradictions(limit=7)
        assert captured["n"] == 7

    def test_deduplicates_pairs(self, monkeypatch, miner):
        """Each pair should only appear once regardless of ordering."""
        monkeypatch.setattr("one.contradictions.recall", lambda *a, **kw: [
            {"id": "1", "raw_text": "The treatment improves infection rates significantly",
             "source": "a"},
            {"id": "2", "raw_text": "The treatment does not improve infection rates",
             "source": "b"},
        ])
        results = miner.mine_contradictions()
        assert len(results) == 1


# ---------------------------------------------------------------------------
# 3. _has_contradiction_signals
# ---------------------------------------------------------------------------

class TestHasContradictionSignals:
    def test_negation_asymmetry_true(self, miner):
        assert miner._has_contradiction_signals(
            "The enzyme promotes cell growth significantly",
            "The enzyme inhibits cell growth significantly",
        )

    def test_no_overlap_returns_false(self, miner):
        assert not miner._has_contradiction_signals(
            "Apples are red",
            "Quantum computing uses qubits",
        )

    def test_both_negative_no_asymmetry(self, miner):
        """Both texts contain negation -- no asymmetry, should still detect
        because abs(neg_a - neg_b) >= 1 is not satisfied when both equal."""
        # Both have exactly one negation word -> abs(1-1) = 0 -> False
        result = miner._has_contradiction_signals(
            "The model does not converge quickly with data",
            "The model cannot converge quickly with data",
        )
        assert result is False

    def test_quantitative_difference_detected(self, miner):
        assert miner._has_contradiction_signals(
            "Accuracy improved from baseline reaching 90% overall",
            "Accuracy improved from baseline reaching 40% overall",
        )

    def test_same_percentages_no_contradiction(self, miner):
        """Identical percentages should not flag a contradiction."""
        assert not miner._has_contradiction_signals(
            "Accuracy improved from baseline reaching 90% overall",
            "Accuracy improved from baseline reaching 90% overall",
        )

    def test_stopwords_filtered(self, miner):
        """Short / stopword-only overlap should not count as topical overlap."""
        assert not miner._has_contradiction_signals(
            "not the answer",
            "the question is clear",
        )

    def test_needs_at_least_two_overlapping_words(self, miner):
        """Only one overlapping non-stopword is not enough."""
        assert not miner._has_contradiction_signals(
            "The car stops suddenly",
            "The car is not blue at night",
        )


# ---------------------------------------------------------------------------
# 4. score_contradiction (mock LLM)
# ---------------------------------------------------------------------------

class TestScoreContradiction:
    def test_score_parses_severity(self, monkeypatch, miner):
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "SEVERITY: critical\n"
            "EXPLANATION: Directly opposing clinical results\n"
            "HIDDEN_VARIABLE: dosage amount"
        ))
        result = miner.score_contradiction("A works", "A does not work")
        assert result["severity"] == "critical"
        assert result["explanation"] == "Directly opposing clinical results"
        assert result["hidden_variable"] == "dosage amount"
        assert result["id"] > 0

    def test_score_defaults_to_moderate_on_none(self, miner):
        """When LLM returns None, defaults to moderate severity."""
        result = miner.score_contradiction("A", "B")
        assert result["severity"] == "moderate"

    def test_score_defaults_on_invalid_severity(self, monkeypatch, miner):
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "SEVERITY: catastrophic\nEXPLANATION: very bad"
        ))
        result = miner.score_contradiction("A", "B")
        assert result["severity"] == "moderate"  # fallback

    def test_score_stores_in_db(self, monkeypatch, miner):
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "SEVERITY: minor\nEXPLANATION: small difference"
        ))
        result = miner.score_contradiction("Finding A", "Finding B")
        conn = cm._get_conn()
        row = conn.execute(
            "SELECT * FROM contradictions WHERE id = ?", (result["id"],)
        ).fetchone()
        assert row is not None
        assert dict(row)["severity"] == "minor"
        assert dict(row)["status"] == "active"

    def test_score_truncates_long_findings(self, monkeypatch, miner):
        """Findings longer than 1000 chars should be truncated before storage."""
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: None)
        long_text = "x" * 2000
        result = miner.score_contradiction(long_text, long_text)
        conn = cm._get_conn()
        row = conn.execute(
            "SELECT finding_a, finding_b FROM contradictions WHERE id = ?",
            (result["id"],),
        ).fetchone()
        assert len(row["finding_a"]) == 1000
        assert len(row["finding_b"]) == 1000


# ---------------------------------------------------------------------------
# 5. resolve_contradiction (all resolution types)
# ---------------------------------------------------------------------------

class TestResolveContradiction:
    def test_resolve_one_wrong(self, monkeypatch, miner):
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "RESOLUTION_TYPE: ONE_WRONG\n"
            "RESOLUTION: Study A had flawed methodology\n"
            "WRONG_SIDE: A\n"
            "CONFIDENCE: 0.85"
        ))
        result = miner.resolve_contradiction("A", "B")
        assert result["resolution_type"] == "ONE_WRONG"
        assert "flawed methodology" in result["resolution"]
        assert result["confidence"] == 0.85

    def test_resolve_context_dependent(self, monkeypatch, miner):
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "RESOLUTION_TYPE: CONTEXT_DEPENDENT\n"
            "RESOLUTION: Depends on temperature\n"
            "HIDDEN_VARIABLE: temperature\n"
            "CONFIDENCE: 0.7"
        ))
        result = miner.resolve_contradiction("A", "B")
        assert result["resolution_type"] == "CONTEXT_DEPENDENT"

    def test_resolve_deeper_truth(self, monkeypatch, miner):
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "RESOLUTION_TYPE: DEEPER_TRUTH\n"
            "RESOLUTION: Both are approximations of underlying process\n"
            "CONFIDENCE: 0.6"
        ))
        result = miner.resolve_contradiction("A", "B")
        assert result["resolution_type"] == "DEEPER_TRUTH"

    def test_resolve_new_phenomenon(self, monkeypatch, miner):
        pushed = []
        monkeypatch.setattr("one.contradictions.push_memory",
                            lambda *a, **kw: pushed.append((a, kw)) or "mock-id")
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "RESOLUTION_TYPE: NEW_PHENOMENON\n"
            "RESOLUTION: Previously unknown interaction detected\n"
            "CONFIDENCE: 0.9"
        ))
        result = miner.resolve_contradiction("A", "B")
        assert result["resolution_type"] == "NEW_PHENOMENON"
        # Should have stored TWO memories: the resolution + the breakthrough
        assert len(pushed) == 2
        # Second push should be tagged as breakthrough
        assert pushed[1][1]["tm_label"] == "breakthrough"

    def test_resolve_defaults_on_none(self, miner):
        """When LLM returns None, defaults apply."""
        result = miner.resolve_contradiction("A", "B")
        assert result["resolution_type"] == "CONTEXT_DEPENDENT"
        assert result["confidence"] == 0.5

    def test_resolve_updates_db_row(self, monkeypatch, miner):
        """If contradiction_id is given, the DB row is updated."""
        _insert_contradiction(miner, severity="critical")
        cid = cm._get_conn().execute(
            "SELECT id FROM contradictions WHERE project = ?", (miner.project,)
        ).fetchone()["id"]

        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "RESOLUTION_TYPE: ONE_WRONG\n"
            "RESOLUTION: B was wrong\n"
            "CONFIDENCE: 0.9"
        ))
        miner.resolve_contradiction("A", "B", contradiction_id=cid)

        row = cm._get_conn().execute(
            "SELECT * FROM contradictions WHERE id = ?", (cid,)
        ).fetchone()
        assert dict(row)["status"] == "resolved"
        assert dict(row)["resolution_type"] == "ONE_WRONG"
        assert dict(row)["resolved_at"] is not None

    def test_resolve_no_db_update_without_id(self, miner):
        """If contradiction_id is 0 (default), no DB update occurs."""
        _insert_contradiction(miner, severity="moderate")
        miner.resolve_contradiction("A", "B", contradiction_id=0)
        row = cm._get_conn().execute(
            "SELECT * FROM contradictions WHERE project = ?", (miner.project,)
        ).fetchone()
        assert dict(row)["status"] == "active"

    def test_resolve_invalid_confidence_keeps_default(self, monkeypatch, miner):
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "RESOLUTION_TYPE: DEEPER_TRUTH\n"
            "RESOLUTION: something\n"
            "CONFIDENCE: not_a_number"
        ))
        result = miner.resolve_contradiction("A", "B")
        assert result["confidence"] == 0.5  # default

    def test_resolve_invalid_resolution_type_keeps_default(self, monkeypatch, miner):
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "RESOLUTION_TYPE: MAGIC\nRESOLUTION: wizard did it\nCONFIDENCE: 0.3"
        ))
        result = miner.resolve_contradiction("A", "B")
        assert result["resolution_type"] == "CONTEXT_DEPENDENT"  # default


# ---------------------------------------------------------------------------
# 6. get_paradigm_contradictions
# ---------------------------------------------------------------------------

class TestGetParadigmContradictions:
    def test_returns_only_paradigm_active(self, miner):
        _insert_contradiction(miner, severity="paradigm", status="active")
        _insert_contradiction(miner, severity="critical", status="active")
        _insert_contradiction(miner, severity="paradigm", status="resolved")
        results = miner.get_paradigm_contradictions()
        assert len(results) == 1
        assert results[0]["severity"] == "paradigm"
        assert results[0]["status"] == "active"

    def test_returns_empty_when_none(self, miner):
        assert miner.get_paradigm_contradictions() == []


# ---------------------------------------------------------------------------
# 7. get_active (with and without severity filter)
# ---------------------------------------------------------------------------

class TestGetActive:
    def test_returns_all_active_no_filter(self, miner):
        _insert_contradiction(miner, severity="minor")
        _insert_contradiction(miner, severity="critical")
        _insert_contradiction(miner, severity="moderate", status="resolved")
        results = miner.get_active()
        assert len(results) == 2

    def test_filters_by_severity(self, miner):
        _insert_contradiction(miner, severity="minor")
        _insert_contradiction(miner, severity="critical")
        results = miner.get_active(severity="critical")
        assert len(results) == 1
        assert results[0]["severity"] == "critical"

    def test_ordering_without_filter(self, miner):
        """Without filter, results should be ordered by severity importance:
        paradigm > critical > moderate > minor."""
        _insert_contradiction(miner, severity="minor")
        _insert_contradiction(miner, severity="paradigm")
        _insert_contradiction(miner, severity="critical")
        _insert_contradiction(miner, severity="moderate")
        results = miner.get_active()
        severities = [r["severity"] for r in results]
        assert severities == ["paradigm", "critical", "moderate", "minor"]

    def test_empty_when_none_active(self, miner):
        _insert_contradiction(miner, severity="minor", status="resolved")
        assert miner.get_active() == []


# ---------------------------------------------------------------------------
# 8. get_resolved
# ---------------------------------------------------------------------------

class TestGetResolved:
    def test_returns_only_resolved(self, miner):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        _insert_contradiction(miner, severity="minor", status="resolved",
                              resolution="fixed", resolved_at=now)
        _insert_contradiction(miner, severity="critical", status="active")
        results = miner.get_resolved()
        assert len(results) == 1
        assert results[0]["status"] == "resolved"

    def test_respects_limit(self, miner):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for _ in range(10):
            _insert_contradiction(miner, status="resolved", resolved_at=now)
        results = miner.get_resolved(limit=3)
        assert len(results) == 3

    def test_empty_when_none_resolved(self, miner):
        _insert_contradiction(miner, severity="minor", status="active")
        assert miner.get_resolved() == []


# ---------------------------------------------------------------------------
# 9. contradiction_dashboard
# ---------------------------------------------------------------------------

class TestContradictionDashboard:
    def test_dashboard_empty(self, miner):
        dash = miner.contradiction_dashboard()
        assert dash["total"] == 0
        assert dash["active"] == 0
        assert dash["resolved"] == 0
        assert dash["paradigm_active"] == 0
        assert dash["critical_active"] == 0
        assert dash["resolution_rate"] == 0.0

    def test_dashboard_counts(self, miner):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        _insert_contradiction(miner, severity="paradigm", status="active")
        _insert_contradiction(miner, severity="critical", status="active")
        _insert_contradiction(miner, severity="moderate", status="active")
        _insert_contradiction(miner, severity="minor", status="resolved",
                              resolved_at=now)
        dash = miner.contradiction_dashboard()
        assert dash["total"] == 4
        assert dash["active"] == 3
        assert dash["resolved"] == 1
        assert dash["paradigm_active"] == 1
        assert dash["critical_active"] == 1
        assert dash["resolution_rate"] == 0.25

    def test_dashboard_resolution_rate(self, miner):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        _insert_contradiction(miner, status="resolved", resolved_at=now)
        _insert_contradiction(miner, status="resolved", resolved_at=now)
        dash = miner.contradiction_dashboard()
        assert dash["resolution_rate"] == 1.0


# ---------------------------------------------------------------------------
# 10. Edge cases & misc
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_on_log_callback_called(self, monkeypatch):
        """on_log callback receives messages during mining."""
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: None)
        monkeypatch.setattr("one.contradictions.push_memory", lambda *a, **kw: "m")
        monkeypatch.setattr("one.contradictions.recall", lambda *a, **kw: [])
        logs = []
        m = ContradictionMiner("test_project", on_log=logs.append)
        m.mine_contradictions()
        assert any("mining" in msg for msg in logs)

    def test_project_isolation(self, monkeypatch):
        """Contradictions from one project are invisible to another."""
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: None)
        monkeypatch.setattr("one.contradictions.push_memory", lambda *a, **kw: "m")
        m_a = ContradictionMiner("project_a")
        m_b = ContradictionMiner("project_b")
        _insert_contradiction(m_a, severity="critical")
        _insert_contradiction(m_a, severity="minor")
        _insert_contradiction(m_b, severity="paradigm")

        assert len(m_a.get_active()) == 2
        assert len(m_b.get_active()) == 1
        assert m_b.get_active()[0]["severity"] == "paradigm"

    def test_severity_levels_constant(self):
        assert SEVERITY_LEVELS == ["minor", "moderate", "critical", "paradigm"]

    def test_new_phenomenon_pushes_with_min_confidence_0_8(self, monkeypatch, miner):
        """NEW_PHENOMENON resolution should push breakthrough memory with
        confidence of at least 0.8 even when LLM confidence is lower."""
        pushed = []
        monkeypatch.setattr("one.contradictions.push_memory",
                            lambda *a, **kw: pushed.append(kw) or "m")
        monkeypatch.setattr("one.contradictions._call_ollama", lambda *a, **kw: (
            "RESOLUTION_TYPE: NEW_PHENOMENON\n"
            "RESOLUTION: something new\n"
            "CONFIDENCE: 0.3"
        ))
        miner.resolve_contradiction("A", "B")
        # The breakthrough push should have confidence >= 0.8
        breakthrough_push = [p for p in pushed if p.get("tm_label") == "breakthrough"]
        assert len(breakthrough_push) == 1
        assert breakthrough_push[0]["aif_confidence"] == 0.8

    def test_mine_multiple_contradictions(self, monkeypatch, miner):
        """Multiple contradictions should be found from a set of findings."""
        monkeypatch.setattr("one.contradictions.recall", lambda *a, **kw: [
            {"id": "1", "raw_text": "Treatment increases patient recovery time significantly",
             "source": "a"},
            {"id": "2", "raw_text": "Treatment does not increase patient recovery time at all",
             "source": "b"},
            {"id": "3", "raw_text": "Treatment reduces patient recovery time significantly",
             "source": "c"},
        ])
        results = miner.mine_contradictions()
        # At least one contradiction should be detected
        assert len(results) >= 1
