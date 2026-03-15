"""Tests for the Executable Verification Engine (experiments module)."""

import json
import sqlite3
import threading

import pytest

from one import experiments as mod
from one.experiments import ExperimentEngine, EXPERIMENT_TYPES, EXPERIMENT_STATUSES


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("one.experiments.DB_PATH", db_path)
    monkeypatch.setattr("one.experiments.DB_DIR", str(tmp_path))
    # Also patch the store module so push_memory works
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", str(tmp_path))
    # Reset thread-local connections
    mod._local = threading.local()
    import one.store as store_mod
    store_mod._local = threading.local()
    store_mod.set_project("test_project")
    yield db_path


@pytest.fixture
def engine():
    """Create an ExperimentEngine for a test project."""
    return ExperimentEngine("test_project")


# ── Schema ────────────────────────────────────────────────────


class TestSchema:
    def test_init_schema_creates_table(self, engine):
        """init_schema is called in __init__; verify table exists."""
        conn = mod._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'"
        ).fetchone()
        assert row is not None

    def test_init_schema_creates_indexes(self, engine):
        conn = mod._get_conn()
        indexes = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        ]
        assert "idx_experiments_project" in indexes
        assert "idx_experiments_status" in indexes
        assert "idx_experiments_type" in indexes

    def test_init_schema_idempotent(self, engine):
        """Calling init_schema twice should not fail."""
        mod.init_schema()
        mod.init_schema()


# ── is_testable ───────────────────────────────────────────────


class TestIsTestable:
    def test_testable_yes(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "TESTABLE: YES\nTYPE: code\nREASONING: it is measurable\nMEASUREMENT: exit code",
        )
        result = engine.is_testable("Adding caching improves speed by 2x")
        assert result["testable"] is True
        assert result["experiment_type"] == "code"
        assert result["reasoning"] == "it is measurable"
        assert result["measurement"] == "exit code"

    def test_testable_no(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "TESTABLE: NO\nTYPE: code\nREASONING: too vague\nMEASUREMENT: none",
        )
        result = engine.is_testable("Life is beautiful")
        assert result["testable"] is False

    def test_testable_data_type(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "TESTABLE: YES\nTYPE: data\nREASONING: stats\nMEASUREMENT: mean",
        )
        result = engine.is_testable("Average salary is above 50k")
        assert result["experiment_type"] == "data"

    def test_testable_llm_returns_none(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: None,
        )
        result = engine.is_testable("some hypothesis")
        assert result["testable"] is False
        assert result["experiment_type"] == "code"

    def test_testable_invalid_type_falls_back(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "TESTABLE: YES\nTYPE: invalid_type\nREASONING: r\nMEASUREMENT: m",
        )
        result = engine.is_testable("hypothesis")
        assert result["experiment_type"] == "code"  # default fallback


# ── design_experiment ─────────────────────────────────────────


class TestDesignExperiment:
    def test_design_stores_in_db(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: (
                "INDEPENDENT_VARIABLE: cache size\n"
                "DEPENDENT_VARIABLE: latency\n"
                "BASELINE: no cache\n"
                "PROCEDURE: run benchmark\n"
                "PASS_CRITERION: latency < 100ms\n"
                "FAIL_CRITERION: latency >= 100ms\n"
                "COMMAND: python bench.py"
            ),
        )
        result = engine.design_experiment("Caching reduces latency", "code")
        assert result["experiment_id"] > 0
        assert result["plan"]["command"] == "python bench.py"
        assert result["plan"]["baseline"] == "no cache"

        # Verify stored in DB
        conn = mod._get_conn()
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (result["experiment_id"],)
        ).fetchone()
        assert row is not None
        assert dict(row)["status"] == "designed"

    def test_design_llm_returns_none(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: None,
        )
        result = engine.design_experiment("hypothesis", "code")
        # All plan fields should be empty strings
        for v in result["plan"].values():
            assert v == ""

    def test_design_experiment_type_stored(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama", lambda *a, **kw: ""
        )
        result = engine.design_experiment("hypothesis", "mathematical")
        assert result["experiment_type"] == "mathematical"


# ── run_experiment ────────────────────────────────────────────


class TestRunExperiment:
    def test_run_code_experiment_success(self, monkeypatch, engine):
        """Code experiment with a passing shell command."""
        monkeypatch.setattr(
            "one.experiments._call_ollama", lambda *a, **kw: ""
        )
        # First design to get an experiment_id
        design = engine.design_experiment("echo test passes", "code")

        # Override the _run_code_experiment to simulate success
        monkeypatch.setattr(
            engine,
            "_run_code_experiment",
            lambda cmd: {
                "output": "ok",
                "exit_code": 0,
                "measurement": "exit_code=0",
                "passed": True,
            },
        )
        plan = {
            "experiment_id": design["experiment_id"],
            "experiment_type": "code",
            "hypothesis": "echo test passes",
            "plan": {"command": "echo hello"},
        }
        result = engine.run_experiment(plan)
        assert result["passed"] is True
        assert result["status"] == "passed"

    def test_run_code_experiment_failure(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama", lambda *a, **kw: ""
        )
        design = engine.design_experiment("this will fail", "code")

        monkeypatch.setattr(
            engine,
            "_run_code_experiment",
            lambda cmd: {
                "output": "error",
                "exit_code": 1,
                "measurement": "exit_code=1",
                "passed": False,
            },
        )
        plan = {
            "experiment_id": design["experiment_id"],
            "experiment_type": "code",
            "hypothesis": "this will fail",
            "plan": {"command": "false"},
        }
        result = engine.run_experiment(plan)
        assert result["passed"] is False
        assert result["status"] == "failed"

    def test_run_llm_experiment(self, monkeypatch, engine):
        """Non-code experiment delegates to LLM reasoning."""
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "OUTCOME: PASSED\nMEASUREMENT: 42\nREASONING: it works",
        )
        plan = {
            "experiment_id": 0,
            "experiment_type": "mathematical",
            "hypothesis": "2+2=4",
            "plan": {"procedure": "compute"},
        }
        result = engine.run_experiment(plan)
        assert result["passed"] is True

    def test_run_experiment_error_handling(self, monkeypatch, engine):
        """Exception during experiment sets error status."""
        monkeypatch.setattr(
            "one.experiments._call_ollama", lambda *a, **kw: ""
        )
        design = engine.design_experiment("error hypothesis", "code")

        monkeypatch.setattr(
            engine,
            "_run_code_experiment",
            lambda cmd: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        plan = {
            "experiment_id": design["experiment_id"],
            "experiment_type": "code",
            "hypothesis": "error hypothesis",
            "plan": {"command": "kaboom"},
        }
        result = engine.run_experiment(plan)
        assert result["status"] == "error"
        assert result["passed"] is False

    def test_run_code_no_command_falls_to_llm(self, monkeypatch, engine):
        """Code experiment without a command falls back to LLM."""
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "OUTCOME: FAILED\nMEASUREMENT: none\nREASONING: no command",
        )
        plan = {
            "experiment_id": 0,
            "experiment_type": "code",
            "hypothesis": "no command",
            "plan": {"command": ""},
        }
        result = engine.run_experiment(plan)
        assert result["passed"] is False

    def test_run_code_na_command_falls_to_llm(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "OUTCOME: INCONCLUSIVE\nMEASUREMENT: ?\nREASONING: unsure",
        )
        plan = {
            "experiment_id": 0,
            "experiment_type": "code",
            "hypothesis": "N/A command",
            "plan": {"command": "N/A"},
        }
        result = engine.run_experiment(plan)
        # INCONCLUSIVE maps to not passed
        assert result["passed"] is False


# ── compare_to_baseline ───────────────────────────────────────


class TestCompareToBaseline:
    def test_compare_parses_llm_response(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "DELTA: +10ms\nSIGNIFICANT: YES\nINTERPRETATION: improvement",
        )
        results = {
            "experiment_id": 0,
            "result": {"measurement": "90ms"},
        }
        comparison = engine.compare_to_baseline(results, "100ms")
        assert comparison["delta"] == "+10ms"
        assert comparison["significant"] is True
        assert comparison["interpretation"] == "improvement"

    def test_compare_not_significant(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "DELTA: 0\nSIGNIFICANT: NO\nINTERPRETATION: no change",
        )
        results = {"experiment_id": 0, "result": {"measurement": "100ms"}}
        comparison = engine.compare_to_baseline(results, "100ms")
        assert comparison["significant"] is False

    def test_compare_updates_db(self, monkeypatch, engine):
        """When experiment_id > 0, delta should be stored in DB."""
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "DELTA: big\nSIGNIFICANT: YES\nINTERPRETATION: wow",
        )
        # Create an experiment first
        design_response = "INDEPENDENT_VARIABLE: x\nDEPENDENT_VARIABLE: y\nBASELINE: 0\nPROCEDURE: test\nPASS_CRITERION: y>0\nFAIL_CRITERION: y<=0\nCOMMAND: echo hi"
        monkeypatch.setattr(
            "one.experiments._call_ollama", lambda *a, **kw: design_response
        )
        design = engine.design_experiment("test", "code")
        eid = design["experiment_id"]

        # Now compare
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "DELTA: big\nSIGNIFICANT: YES\nINTERPRETATION: wow",
        )
        engine.compare_to_baseline(
            {"experiment_id": eid, "result": {"measurement": "42"}}, "0"
        )
        conn = mod._get_conn()
        row = conn.execute("SELECT delta FROM experiments WHERE id = ?", (eid,)).fetchone()
        assert row is not None
        delta_data = json.loads(row["delta"])
        assert delta_data["significant"] is True


# ── store_experiment ──────────────────────────────────────────


class TestStoreExperiment:
    def test_store_passed(self, engine):
        eid = engine.store_experiment(
            "hypothesis A",
            {"experiment_type": "code", "baseline": "none"},
            {"passed": True, "measurement": "ok"},
        )
        assert eid > 0
        exp = engine.get_experiment(eid)
        assert exp["status"] == "passed"
        assert exp["confidence"] == 0.8

    def test_store_failed(self, engine):
        eid = engine.store_experiment(
            "hypothesis B",
            {"experiment_type": "data", "baseline": "50"},
            {"passed": False},
        )
        exp = engine.get_experiment(eid)
        assert exp["status"] == "failed"
        assert exp["confidence"] == 0.3

    def test_store_error(self, engine):
        eid = engine.store_experiment(
            "hypothesis C",
            {"experiment_type": "simulation"},
            {"passed": False, "error": "timeout"},
        )
        exp = engine.get_experiment(eid)
        assert exp["status"] == "error"


# ── list_experiments ──────────────────────────────────────────


class TestListExperiments:
    def test_list_empty(self, engine):
        result = engine.list_experiments()
        assert result == []

    def test_list_all(self, engine):
        engine.store_experiment("h1", {"experiment_type": "code"}, {"passed": True})
        engine.store_experiment("h2", {"experiment_type": "code"}, {"passed": False})
        result = engine.list_experiments()
        assert len(result) == 2

    def test_list_with_status_filter(self, engine):
        engine.store_experiment("h1", {"experiment_type": "code"}, {"passed": True})
        engine.store_experiment("h2", {"experiment_type": "code"}, {"passed": False})
        passed = engine.list_experiments(status="passed")
        assert len(passed) == 1
        assert passed[0]["status"] == "passed"
        failed = engine.list_experiments(status="failed")
        assert len(failed) == 1

    def test_list_nonexistent_status(self, engine):
        engine.store_experiment("h1", {"experiment_type": "code"}, {"passed": True})
        result = engine.list_experiments(status="inconclusive")
        assert result == []


# ── experiment_dashboard ──────────────────────────────────────


class TestExperimentDashboard:
    def test_dashboard_empty(self, engine):
        d = engine.experiment_dashboard()
        assert d["total"] == 0
        assert d["passed"] == 0
        assert d["failed"] == 0
        assert d["errors"] == 0
        assert d["designed"] == 0
        assert d["pass_rate"] == 0.0

    def test_dashboard_with_experiments(self, engine):
        engine.store_experiment("h1", {"experiment_type": "code"}, {"passed": True})
        engine.store_experiment("h2", {"experiment_type": "code"}, {"passed": True})
        engine.store_experiment("h3", {"experiment_type": "code"}, {"passed": False})
        engine.store_experiment(
            "h4", {"experiment_type": "code"}, {"passed": False, "error": "oops"}
        )
        d = engine.experiment_dashboard()
        assert d["total"] == 4
        assert d["passed"] == 2
        assert d["failed"] == 1
        assert d["errors"] == 1
        assert d["pass_rate"] == 0.5  # 2 passed out of 4 (no designed)


# ── run_full_experiment ───────────────────────────────────────


class TestRunFullExperiment:
    def test_full_not_testable(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "TESTABLE: NO\nTYPE: code\nREASONING: too abstract\nMEASUREMENT: none",
        )
        result = engine.run_full_experiment("Beauty is truth")
        assert result["testable"] is False
        assert "reasoning" in result

    def test_full_testable_passed(self, monkeypatch, engine):
        call_count = {"n": 0}

        def mock_ollama(*a, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "TESTABLE: YES\nTYPE: simulation\nREASONING: can simulate\nMEASUREMENT: accuracy"
            elif call_count["n"] == 2:
                return "INDEPENDENT_VARIABLE: x\nDEPENDENT_VARIABLE: y\nBASELINE: 0\nPROCEDURE: sim\nPASS_CRITERION: y>0\nFAIL_CRITERION: y<=0\nCOMMAND: N/A"
            elif call_count["n"] == 3:
                return "OUTCOME: PASSED\nMEASUREMENT: 0.95\nREASONING: high accuracy"
            elif call_count["n"] == 4:
                return "DELTA: +0.95\nSIGNIFICANT: YES\nINTERPRETATION: big improvement"
            return ""

        monkeypatch.setattr("one.experiments._call_ollama", mock_ollama)
        result = engine.run_full_experiment("Simulation converges")
        assert result["testable"] is True
        assert result["passed"] is True
        assert "comparison" in result


# ── Edge Cases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_get_experiment_nonexistent(self, engine):
        assert engine.get_experiment(9999) is None

    def test_experiment_types_constant(self):
        assert "code" in EXPERIMENT_TYPES
        assert "data" in EXPERIMENT_TYPES
        assert "mathematical" in EXPERIMENT_TYPES
        assert "simulation" in EXPERIMENT_TYPES

    def test_experiment_statuses_constant(self):
        assert "designed" in EXPERIMENT_STATUSES
        assert "passed" in EXPERIMENT_STATUSES
        assert "failed" in EXPERIMENT_STATUSES
        assert "error" in EXPERIMENT_STATUSES

    def test_on_log_callback(self, tmp_path, monkeypatch):
        logs = []
        eng = ExperimentEngine("test_project", on_log=logs.append)
        monkeypatch.setattr(
            "one.experiments._call_ollama",
            lambda *a, **kw: "TESTABLE: NO\nTYPE: code\nREASONING: no\nMEASUREMENT: no",
        )
        eng.is_testable("test")
        assert any("assessing testability" in msg for msg in logs)

    def test_hypothesis_truncated_to_1000(self, monkeypatch, engine):
        monkeypatch.setattr(
            "one.experiments._call_ollama", lambda *a, **kw: ""
        )
        long_hyp = "x" * 2000
        result = engine.design_experiment(long_hyp, "code")
        exp = engine.get_experiment(result["experiment_id"])
        assert len(exp["hypothesis"]) == 1000
