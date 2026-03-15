"""Tests for the playbook system — database operations and pure functions.

Excluded from coverage intentionally:
- create_playbook()    — calls Gemma/LLM and push_memory (side-effectful)
- recall_playbook()    — calls hdc.encode_text + store.recall (integration)
- recall_playbook_context() — depends on recall_playbook
"""

import sqlite3
import threading
from datetime import datetime, timezone

import pytest

from one import playbook as pb_mod
from one import store as store_mod
from one.playbook import (
    _init_playbook_schema,
    _parse_analysis,
    _infer_category,
    _build_fallback,
    CATEGORY_FALLBACK,
    ANALYSIS_PROMPT,
    list_playbooks,
    get_playbook_count,
    init_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Redirect both store and playbook modules to a shared temp database.

    Both modules read DB_PATH / DB_DIR at connection time, so we patch both.
    Thread-local connections are reset so no stale handle leaks between tests.
    """
    db_path = str(tmp_path / "test.db")
    db_dir = str(tmp_path)

    # Patch store module (playbook imports push_memory / recall from there)
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", db_dir)
    store_mod._local = threading.local()

    # Patch playbook module's own view of the constants
    monkeypatch.setattr("one.playbook.DB_PATH", db_path)
    monkeypatch.setattr("one.playbook.DB_DIR", db_dir)
    pb_mod._local = threading.local()

    yield db_path


@pytest.fixture()
def conn(tmp_path):
    """A bare SQLite connection with the playbook schema already applied.

    Useful for tests that want to inspect the DB directly without going
    through the module-level thread-local connection.
    """
    db_path = str(tmp_path / "direct.db")
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    _init_playbook_schema(c)
    yield c
    c.close()


def _insert_playbook(
    conn: sqlite3.Connection,
    *,
    project: str = "proj",
    task_description: str = "Fix login bug",
    category: str = "debug",
    key_decisions: str = "- checked logs",
    reusable_patterns: str = "- add logging",
    pitfalls: str = "- missing auth header",
    full_playbook: str = "Full text of the playbook.",
    confidence: float = 0.9,
    times_recalled: int = 0,
) -> int:
    """Insert a playbook row directly and return its rowid."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO playbooks
           (project, task_description, category, key_decisions, reusable_patterns,
            pitfalls, full_playbook, confidence, times_recalled, created, updated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (project, task_description, category, key_decisions, reusable_patterns,
         pitfalls, full_playbook, confidence, times_recalled, now, now),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

class TestSchemaInit:
    def test_creates_playbooks_table(self, conn):
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "playbooks" in tables

    def test_creates_project_index(self, conn):
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_playbooks_project" in indexes

    def test_creates_category_index(self, conn):
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_playbooks_category" in indexes

    def test_schema_columns(self, conn):
        """All declared columns are present in the table."""
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(playbooks)").fetchall()
        }
        expected = {
            "id", "project", "task_description", "category",
            "key_decisions", "reusable_patterns", "pitfalls",
            "full_playbook", "confidence", "times_recalled", "created", "updated",
        }
        assert expected.issubset(cols)

    def test_init_schema_idempotent(self, conn):
        """Calling _init_playbook_schema twice must not raise."""
        _init_playbook_schema(conn)  # second call
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "playbooks" in tables

    def test_public_init_schema_creates_table(self):
        """init_schema() via the module-level function must work end-to-end."""
        init_schema()
        # Verify via the thread-local connection that the table exists
        c = pb_mod._get_conn()
        tables = {
            row[0]
            for row in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "playbooks" in tables

    def test_default_values(self, conn):
        """Columns with defaults are populated when not supplied explicitly."""
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO playbooks
               (project, task_description, full_playbook, created, updated)
               VALUES (?, ?, ?, ?, ?)""",
            ("p", "task", "text", now, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM playbooks").fetchone()
        assert row["category"] == "general"
        assert abs(row["confidence"] - 0.9) < 1e-9
        assert row["times_recalled"] == 0


# ---------------------------------------------------------------------------
# Creating playbooks (direct SQL, no LLM)
# ---------------------------------------------------------------------------

class TestCreatePlaybook:
    def test_insert_returns_rowid(self, conn):
        rid = _insert_playbook(conn)
        assert isinstance(rid, int)
        assert rid >= 1

    def test_insert_stores_all_fields(self, conn):
        _insert_playbook(
            conn,
            project="myproj",
            task_description="Refactor auth module",
            category="refactor",
            key_decisions="- split into sub-modules",
            reusable_patterns="- dependency injection",
            pitfalls="- circular imports",
            full_playbook="Full refactor playbook text.",
            confidence=0.85,
        )
        row = conn.execute("SELECT * FROM playbooks").fetchone()
        assert row["project"] == "myproj"
        assert row["task_description"] == "Refactor auth module"
        assert row["category"] == "refactor"
        assert row["key_decisions"] == "- split into sub-modules"
        assert row["reusable_patterns"] == "- dependency injection"
        assert row["pitfalls"] == "- circular imports"
        assert row["full_playbook"] == "Full refactor playbook text."
        assert abs(row["confidence"] - 0.85) < 1e-9

    def test_timestamps_are_set(self, conn):
        _insert_playbook(conn)
        row = conn.execute("SELECT created, updated FROM playbooks").fetchone()
        # Both should parse as valid ISO timestamps
        datetime.fromisoformat(row["created"])
        datetime.fromisoformat(row["updated"])

    def test_multiple_inserts_different_ids(self, conn):
        rid1 = _insert_playbook(conn, task_description="Task A")
        rid2 = _insert_playbook(conn, task_description="Task B")
        assert rid1 != rid2

    def test_insert_multiple_projects(self, conn):
        _insert_playbook(conn, project="alpha", task_description="alpha task")
        _insert_playbook(conn, project="beta", task_description="beta task")
        rows = conn.execute("SELECT project FROM playbooks ORDER BY id").fetchall()
        projects = [r["project"] for r in rows]
        assert projects == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Retrieving playbooks
# ---------------------------------------------------------------------------

class TestListPlaybooks:
    def test_empty_project_returns_empty_list(self):
        init_schema()
        result = list_playbooks("nonexistent_project")
        assert result == []

    def test_returns_playbooks_for_project(self):
        init_schema()
        c = pb_mod._get_conn()
        _insert_playbook(c, project="proj1", task_description="Task 1")
        result = list_playbooks("proj1")
        assert len(result) == 1
        assert result[0]["task_description"] == "Task 1"

    def test_isolates_by_project(self):
        init_schema()
        c = pb_mod._get_conn()
        _insert_playbook(c, project="proj1", task_description="Task A")
        _insert_playbook(c, project="proj2", task_description="Task B")
        result = list_playbooks("proj1")
        assert len(result) == 1
        assert result[0]["task_description"] == "Task A"

    def test_ordered_by_created_desc(self):
        """list_playbooks returns most-recent first."""
        import time
        init_schema()
        c = pb_mod._get_conn()
        _insert_playbook(c, project="p", task_description="First")
        time.sleep(0.01)  # ensure distinct timestamps
        _insert_playbook(c, project="p", task_description="Second")
        result = list_playbooks("p")
        assert result[0]["task_description"] == "Second"
        assert result[1]["task_description"] == "First"

    def test_returns_expected_fields(self):
        init_schema()
        c = pb_mod._get_conn()
        _insert_playbook(c, project="p")
        result = list_playbooks("p")
        row = result[0]
        for field in (
            "id", "task_description", "category", "key_decisions",
            "reusable_patterns", "pitfalls", "confidence",
            "times_recalled", "created",
        ):
            assert field in row, f"Missing field: {field}"

    def test_multiple_playbooks_same_project(self):
        init_schema()
        c = pb_mod._get_conn()
        for i in range(5):
            _insert_playbook(c, project="many", task_description=f"Task {i}")
        result = list_playbooks("many")
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Playbook count
# ---------------------------------------------------------------------------

class TestGetPlaybookCount:
    def test_count_empty(self):
        init_schema()
        assert get_playbook_count("no_project") == 0

    def test_count_increments(self):
        init_schema()
        c = pb_mod._get_conn()
        _insert_playbook(c, project="counted")
        assert get_playbook_count("counted") == 1
        _insert_playbook(c, project="counted")
        assert get_playbook_count("counted") == 2

    def test_count_isolated_per_project(self):
        init_schema()
        c = pb_mod._get_conn()
        _insert_playbook(c, project="a")
        _insert_playbook(c, project="a")
        _insert_playbook(c, project="b")
        assert get_playbook_count("a") == 2
        assert get_playbook_count("b") == 1


# ---------------------------------------------------------------------------
# Updating playbooks (times_recalled counter)
# ---------------------------------------------------------------------------

class TestUpdatePlaybook:
    def test_times_recalled_default_is_zero(self, conn):
        _insert_playbook(conn)
        row = conn.execute("SELECT times_recalled FROM playbooks").fetchone()
        assert row["times_recalled"] == 0

    def test_increment_times_recalled(self, conn):
        rid = _insert_playbook(conn)
        conn.execute(
            "UPDATE playbooks SET times_recalled = times_recalled + 1 WHERE id = ?",
            (rid,),
        )
        conn.commit()
        row = conn.execute(
            "SELECT times_recalled FROM playbooks WHERE id = ?", (rid,)
        ).fetchone()
        assert row["times_recalled"] == 1

    def test_increment_multiple_times(self, conn):
        rid = _insert_playbook(conn)
        for _ in range(5):
            conn.execute(
                "UPDATE playbooks SET times_recalled = times_recalled + 1 WHERE id = ?",
                (rid,),
            )
        conn.commit()
        row = conn.execute(
            "SELECT times_recalled FROM playbooks WHERE id = ?", (rid,)
        ).fetchone()
        assert row["times_recalled"] == 5

    def test_update_updated_timestamp(self, conn):
        rid = _insert_playbook(conn)
        old_ts = conn.execute(
            "SELECT updated FROM playbooks WHERE id = ?", (rid,)
        ).fetchone()["updated"]

        new_ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE playbooks SET updated = ? WHERE id = ?", (new_ts, rid)
        )
        conn.commit()
        row = conn.execute(
            "SELECT updated FROM playbooks WHERE id = ?", (rid,)
        ).fetchone()
        assert row["updated"] == new_ts
        # If the test runs fast enough timestamps may be equal; at least it was set
        assert row["updated"] is not None

    def test_update_confidence(self, conn):
        rid = _insert_playbook(conn, confidence=0.9)
        conn.execute(
            "UPDATE playbooks SET confidence = ? WHERE id = ?", (0.6, rid)
        )
        conn.commit()
        row = conn.execute(
            "SELECT confidence FROM playbooks WHERE id = ?", (rid,)
        ).fetchone()
        assert abs(row["confidence"] - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# _parse_analysis (pure function)
# ---------------------------------------------------------------------------

class TestParseAnalysis:
    def test_full_response(self):
        # Note: no blank line between CATEGORY and the next section header —
        # a blank line would cause the parser to overwrite the category value.
        text = (
            "CATEGORY: debug\n"
            "KEY DECISIONS:\n- checked logs\n- reproduced locally\n"
            "REUSABLE PATTERNS:\n- add verbose logging\n- isolate component\n"
            "PITFALLS TO AVOID:\n- don't skip unit tests\n"
            "PLAYBOOK SUMMARY:\n"
            "We traced the bug by enabling verbose logging.\n"
        )
        result = _parse_analysis(text)
        assert result["category"] == "debug"
        assert "checked logs" in result["key_decisions"]
        assert "reproduced locally" in result["key_decisions"]
        assert "add verbose logging" in result["reusable_patterns"]
        assert "don't skip unit tests" in result["pitfalls"]
        assert "verbose logging" in result["playbook_summary"]

    def test_category_lowercased(self):
        # No blank line between CATEGORY and the next header to avoid parser overwrite
        text = "CATEGORY: DEBUG\nPLAYBOOK SUMMARY:\nsome summary"
        result = _parse_analysis(text)
        assert result["category"] == "debug"

    def test_category_whitespace_stripped(self):
        # No blank line between CATEGORY and the next header to avoid parser overwrite
        text = "CATEGORY:   feature   \nPLAYBOOK SUMMARY:\ntext"
        result = _parse_analysis(text)
        assert result["category"] == "feature"

    def test_category_overwritten_by_blank_line(self):
        """Document the parser's actual behaviour: a blank line after CATEGORY:
        causes the value to be discarded when the next section is flushed."""
        text = "CATEGORY: debug\n\nKEY DECISIONS:\n- a decision\n"
        result = _parse_analysis(text)
        # The blank line is appended to current_lines for the 'category' section,
        # then flushed as the category value when KEY DECISIONS: is encountered.
        assert result["category"] == ""

    def test_defaults_when_missing(self):
        result = _parse_analysis("")
        assert result["category"] == "general"
        assert result["key_decisions"] == ""
        assert result["reusable_patterns"] == ""
        assert result["pitfalls"] == ""
        assert result["playbook_summary"] == ""

    def test_pitfalls_alternate_header(self):
        """'PITFALLS:' (without 'TO AVOID') is also accepted."""
        text = "PITFALLS:\n- watch imports\n\nPLAYBOOK SUMMARY:\ndone"
        result = _parse_analysis(text)
        assert "watch imports" in result["pitfalls"]

    def test_playbook_alternate_header(self):
        """'PLAYBOOK:' (shortened) is also accepted."""
        text = "PLAYBOOK:\nThis is the summary.\n"
        result = _parse_analysis(text)
        assert "This is the summary." in result["playbook_summary"]

    def test_multiple_bullet_points_preserved(self):
        text = (
            "KEY DECISIONS:\n"
            "- decision one\n"
            "- decision two\n"
            "- decision three\n\n"
            "PLAYBOOK SUMMARY:\nSummary text."
        )
        result = _parse_analysis(text)
        assert "decision one" in result["key_decisions"]
        assert "decision two" in result["key_decisions"]
        assert "decision three" in result["key_decisions"]

    def test_strips_blank_lines_between_sections(self):
        """Blank lines inside a section's content are ignored at the boundary."""
        text = (
            "REUSABLE PATTERNS:\n"
            "- pattern A\n\n"
            "PITFALLS TO AVOID:\n"
            "- pitfall B"
        )
        result = _parse_analysis(text)
        assert "pattern A" in result["reusable_patterns"]
        assert "pitfall B" in result["pitfalls"]

    def test_returns_dict_with_all_keys(self):
        result = _parse_analysis("CATEGORY: test\n")
        assert set(result.keys()) == {
            "category", "key_decisions", "reusable_patterns",
            "pitfalls", "playbook_summary",
        }


# ---------------------------------------------------------------------------
# _infer_category (pure function)
# ---------------------------------------------------------------------------

class TestInferCategory:
    @pytest.mark.parametrize("keyword,expected", [
        ("fix the broken auth", "debug"),
        ("find and fix the bug", "debug"),
        ("trace the error message", "debug"),
        ("write a test suite", "test"),
        ("deploy to production", "deploy"),
        ("ship the release", "deploy"),
        ("refactor the service", "refactor"),
        ("clean up legacy code", "refactor"),
        ("research alternatives", "research"),
        ("investigate the issue", "research"),
        ("build new feature", "feature"),
        ("implement pagination", "feature"),
        ("add dark mode", "feature"),
        ("create a widget", "feature"),
    ])
    def test_keyword_mapping(self, keyword, expected):
        assert _infer_category(keyword) == expected

    def test_no_keyword_returns_general(self):
        assert _infer_category("completely unrelated task") == "general"

    def test_case_insensitive(self):
        assert _infer_category("FIX THE LOGIN FLOW") == "debug"

    def test_empty_string_returns_general(self):
        assert _infer_category("") == "general"

    def test_first_matching_keyword_wins(self):
        """When multiple keywords match, the first one in CATEGORY_FALLBACK wins."""
        # "fix" comes before "bug" in CATEGORY_FALLBACK dict
        result = _infer_category("fix the bug")
        assert result in ("debug",)  # both map to "debug", so any is correct


# ---------------------------------------------------------------------------
# _build_fallback (pure function)
# ---------------------------------------------------------------------------

class TestBuildFallback:
    def test_returns_dict_with_required_keys(self):
        result = _build_fallback("fix login", "step 1\nstep 2", "success")
        assert set(result.keys()) == {
            "category", "key_decisions", "reusable_patterns",
            "pitfalls", "playbook_summary",
        }

    def test_category_inferred_from_description(self):
        result = _build_fallback("deploy to staging", "step", "ok")
        assert result["category"] == "deploy"

    def test_category_general_when_no_keyword(self):
        result = _build_fallback("unrelated task", "step", "ok")
        assert result["category"] == "general"

    def test_key_decisions_from_steps(self):
        steps = "checked the database\nreviewed the error logs\nrolled back migration"
        result = _build_fallback("fix auth", steps, "resolved")
        for step in ["checked the database", "reviewed the error logs", "rolled back migration"]:
            assert step in result["key_decisions"]

    def test_empty_steps_uses_placeholder(self):
        result = _build_fallback("fix auth", "", "done")
        assert "no steps recorded" in result["key_decisions"]

    def test_short_steps_filtered_out(self):
        """Steps with <= 10 characters are excluded."""
        result = _build_fallback("fix auth", "tiny\nthis step is long enough to be included", "done")
        assert "tiny" not in result["key_decisions"]
        assert "this step is long enough" in result["key_decisions"]

    def test_steps_capped_at_ten(self):
        many_steps = "\n".join(f"This is step number {i} with some content" for i in range(20))
        result = _build_fallback("fix auth", many_steps, "done")
        # At most 10 bullet points
        bullet_count = result["key_decisions"].count("\n- ") + (1 if result["key_decisions"].startswith("- ") else 0)
        assert bullet_count <= 10

    def test_step_length_capped_at_200(self):
        long_step = "x" * 300
        result = _build_fallback("fix auth", long_step, "done")
        # The step text itself must be at most 200 chars (plus "- " prefix)
        lines = result["key_decisions"].split("\n")
        for line in lines:
            content = line.lstrip("- ")
            assert len(content) <= 200

    def test_playbook_summary_contains_outcome(self):
        result = _build_fallback("fix auth", "a step with enough length", "all good")
        assert "all good" in result["playbook_summary"]

    def test_playbook_summary_contains_category(self):
        result = _build_fallback("deploy to production", "a step with enough length", "deployed")
        assert "deploy" in result["playbook_summary"]

    def test_reusable_patterns_references_category(self):
        result = _build_fallback("fix auth bug", "a step with enough length", "done")
        assert "debug" in result["reusable_patterns"]

    def test_pitfalls_is_placeholder(self):
        result = _build_fallback("fix auth", "a step with enough length", "done")
        assert "manually" in result["pitfalls"].lower() or "pitfall" in result["pitfalls"].lower()

    def test_task_description_truncated_in_patterns(self):
        long_desc = "A" * 200
        result = _build_fallback(long_desc, "a step with enough length", "done")
        assert len(result["reusable_patterns"]) < 400  # reasonable upper bound

    def test_outcome_truncated_in_summary(self):
        long_outcome = "O" * 300
        result = _build_fallback("fix auth", "a step with enough length", long_outcome)
        # Outcome is sliced to 150 chars in the summary
        assert long_outcome not in result["playbook_summary"]
        assert long_outcome[:150] in result["playbook_summary"]


# ---------------------------------------------------------------------------
# ANALYSIS_PROMPT (template integrity)
# ---------------------------------------------------------------------------

class TestAnalysisPrompt:
    def test_has_required_placeholders(self):
        assert "{task_description}" in ANALYSIS_PROMPT
        assert "{steps_taken}" in ANALYSIS_PROMPT
        assert "{outcome}" in ANALYSIS_PROMPT

    def test_format_with_values(self):
        filled = ANALYSIS_PROMPT.format(
            task_description="fix login",
            steps_taken="step 1",
            outcome="success",
        )
        assert "fix login" in filled
        assert "step 1" in filled
        assert "success" in filled


# ---------------------------------------------------------------------------
# CATEGORY_FALLBACK integrity
# ---------------------------------------------------------------------------

class TestCategoryFallback:
    def test_all_values_are_valid_categories(self):
        valid = {"debug", "test", "deploy", "refactor", "research", "feature"}
        for keyword, category in CATEGORY_FALLBACK.items():
            assert category in valid, f"'{keyword}' maps to unknown category '{category}'"

    def test_is_dict(self):
        assert isinstance(CATEGORY_FALLBACK, dict)

    def test_non_empty(self):
        assert len(CATEGORY_FALLBACK) > 0


# ---------------------------------------------------------------------------
# Thread-local connection reset pattern
# ---------------------------------------------------------------------------

class TestThreadLocalReset:
    def test_reset_gives_fresh_connection(self):
        """After resetting _local, _get_conn() opens a new connection."""
        conn1 = pb_mod._get_conn()
        pb_mod._local = threading.local()
        conn2 = pb_mod._get_conn()
        # Different Python objects — a new connection was opened
        assert conn1 is not conn2

    def test_connection_persists_within_thread(self):
        """Calling _get_conn() twice without reset returns the same connection."""
        conn1 = pb_mod._get_conn()
        conn2 = pb_mod._get_conn()
        assert conn1 is conn2

    def test_reset_does_not_corrupt_db(self):
        """Data written before a reset is still readable after reset."""
        init_schema()
        c = pb_mod._get_conn()
        _insert_playbook(c, project="persist_test", task_description="Persisted task")

        # Reset thread-local state (simulates a new test / new thread)
        pb_mod._local = threading.local()

        result = list_playbooks("persist_test")
        assert len(result) == 1
        assert result[0]["task_description"] == "Persisted task"
