"""Tests for the synthesis engine."""

import sqlite3
import threading
import pytest

import one.store
import one.synthesis
from one.synthesis import (
    _init_synthesis_schema,
    _score_novelty,
    _has_contradiction_signals,
    _already_synthesized,
    get_synthesis_chain,
    get_syntheses_count,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Redirect both store and synthesis to an isolated temp SQLite database."""
    db_path = str(tmp_path / "test.db")

    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", str(tmp_path))
    monkeypatch.setattr("one.synthesis.DB_PATH", db_path)
    monkeypatch.setattr("one.synthesis.DB_DIR", str(tmp_path))

    # Reset thread-local connections so the new paths are picked up.
    one.store._local = threading.local()
    one.synthesis._local = threading.local()

    one.store.set_project("test")
    yield db_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn(db_path: str) -> sqlite3.Connection:
    """Open a plain connection to the temp database for inspection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_synthesis(
    db_path: str,
    project: str = "test",
    entity_a: str = "A",
    entity_b: str = "B",
    hypothesis: str = "A causes B via pathway X.",
    confidence: float = 0.7,
    novelty_score: float = 0.6,
    depth: int = 0,
    parent_id=None,
) -> int:
    """Insert a synthesis row directly and return its rowid."""
    conn = _make_conn(db_path)
    cur = conn.execute(
        """INSERT INTO syntheses
           (project, entity_a, entity_b, hypothesis, confidence,
            novelty_score, tested, test_result, parent_id, depth, created)
           VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, datetime('now'))""",
        (project, entity_a, entity_b, hypothesis, confidence,
         novelty_score, parent_id, depth),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

class TestSchemaInit:
    def test_tables_created(self, tmp_path):
        db_path = str(tmp_path / "schema_test.db")
        conn = sqlite3.connect(db_path)
        _init_synthesis_schema(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        conn.close()
        assert "syntheses" in tables

    def test_columns_present(self, tmp_path):
        db_path = str(tmp_path / "cols_test.db")
        conn = sqlite3.connect(db_path)
        _init_synthesis_schema(conn)

        info = conn.execute("PRAGMA table_info(syntheses)").fetchall()
        col_names = {row[1] for row in info}
        conn.close()

        expected = {
            "id", "project", "entity_a", "entity_b", "hypothesis",
            "confidence", "novelty_score", "tested", "test_result",
            "parent_id", "depth", "created",
        }
        assert expected.issubset(col_names)

    def test_indexes_created(self, tmp_path):
        db_path = str(tmp_path / "idx_test.db")
        conn = sqlite3.connect(db_path)
        _init_synthesis_schema(conn)

        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        conn.close()

        assert "idx_syntheses_project" in indexes
        assert "idx_syntheses_depth" in indexes
        assert "idx_syntheses_parent" in indexes
        assert "idx_syntheses_novelty" in indexes

    def test_idempotent(self, tmp_path):
        """Calling _init_synthesis_schema twice must not raise."""
        db_path = str(tmp_path / "idem_test.db")
        conn = sqlite3.connect(db_path)
        _init_synthesis_schema(conn)
        _init_synthesis_schema(conn)  # second call — must be a no-op
        conn.close()

    def test_novelty_default(self, tmp_path):
        """novelty_score should default to 0.5."""
        db_path = str(tmp_path / "default_test.db")
        conn = sqlite3.connect(db_path)
        _init_synthesis_schema(conn)
        conn.execute(
            "INSERT INTO syntheses (project, entity_a, entity_b, hypothesis, confidence, created) "
            "VALUES ('p','A','B','h',0.5,datetime('now'))"
        )
        conn.commit()
        row = conn.execute("SELECT novelty_score, tested FROM syntheses").fetchone()
        assert row[0] == 0.5   # DEFAULT 0.5
        assert row[1] == 0     # DEFAULT 0
        conn.close()


# ---------------------------------------------------------------------------
# _score_novelty
# ---------------------------------------------------------------------------

class TestScoreNovelty:
    def _score(self, text: str) -> float:
        return _score_novelty(text, "A", "B", [])

    def test_returns_float_in_range(self):
        score = self._score("A and B interact.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_trivial_restatement_low_score(self):
        trivial = (
            "A and B co-occur frequently. They are associated with each other."
        )
        assert self._score(trivial) <= 0.2

    def test_mechanistic_hypothesis_higher_score(self):
        mechanistic = (
            "A activates the X pathway because it modulates receptor Y, "
            "which directly causes downstream inhibition of B's signalling cascade."
        )
        trivial = "A and B co-occur and are associated with each other."
        assert self._score(mechanistic) > self._score(trivial)

    def test_testable_phrase_boosts_score(self):
        base = "A enables B through mechanism X."
        testable = base + " We would expect to see increased co-expression if this is true."
        assert self._score(testable) >= self._score(base)

    def test_two_or_more_trivial_phrases_caps_at_low(self):
        text = (
            "A and B both appear in tests. They are related to each other "
            "and commonly found together."
        )
        assert self._score(text) <= 0.2

    def test_single_trivial_phrase_not_capped(self):
        text = "A and B co-occur in pathways X and Y, causing downstream inhibition."
        score = self._score(text)
        # One trivial phrase (-0.2) but mechanistic language offsets — must be > 0.1
        assert score > 0.1

    def test_camelcase_specificity_bonus(self):
        # CamelCase proper names raise specificity
        with_names = "AlphaProtein activates BetaReceptor causing GammaInhibition."
        without_names = "the protein activates the receptor causing inhibition."
        assert self._score(with_names) >= self._score(without_names)

    def test_minimum_floor(self):
        # Worst-case text should still be >= 0.1 (the clamped minimum)
        assert self._score("co-occur associated with related to each other") >= 0.1

    def test_maximum_ceiling(self):
        # Even the richest text must not exceed 1.0
        rich = (
            "AlphaProtein causes BetaReceptor inhibition because mechanism X "
            "enables pathway Y which predicts structural causal emergent effects. "
            "This could be tested in experiment and we would expect measurable "
            "observable results. If this is true, verify with causal drivers."
        )
        assert self._score(rich) <= 1.0


# ---------------------------------------------------------------------------
# _has_contradiction_signals
# ---------------------------------------------------------------------------

class TestHasContradictionSignals:
    def test_no_shared_content_returns_false(self):
        assert _has_contradiction_signals("cats sleep all day", "cars run on petrol") is False

    def test_shared_content_no_negation_returns_false(self):
        # Same topic words, both positive — no asymmetric negation
        assert _has_contradiction_signals(
            "A activates pathway B",
            "A stimulates pathway B",
        ) is False

    def test_shared_content_symmetric_negation_returns_false(self):
        # Both contain "not" — the XOR is False, so no contradiction signal
        assert _has_contradiction_signals(
            "A does not activate pathway B",
            "X does not inhibit pathway B",
        ) is False

    def test_asymmetric_negation_returns_true(self):
        # One sentence contains "not", the other does not.
        assert _has_contradiction_signals(
            "A activates pathway B",
            "A does not activate pathway B",
        ) is True

    def test_inhibits_as_negation_word(self):
        # "inhibits" is a negation word; the other text has shared words but no negation
        assert _has_contradiction_signals(
            "A inhibits the growth of B",
            "A promotes the growth of B",
        ) is True

    def test_prevents_as_negation_word(self):
        assert _has_contradiction_signals(
            "X prevents B formation",
            "X encourages B formation through a causal pathway",
        ) is True

    def test_empty_strings_returns_false(self):
        assert _has_contradiction_signals("", "") is False

    def test_stopwords_only_not_sufficient(self):
        # Shared stopwords after filtering should not be enough
        assert _has_contradiction_signals("the is a", "the not is") is False


# ---------------------------------------------------------------------------
# _already_synthesized
# ---------------------------------------------------------------------------

class TestAlreadySynthesized:
    def test_false_when_empty(self):
        # Initialise schema via public API so the table exists
        one.synthesis.init_schema()
        assert _already_synthesized("test", "A", "B") is False

    def test_true_after_insertion(self, tmp_path):
        one.synthesis.init_schema()
        db_path = str(tmp_path / "test.db")
        _insert_synthesis(db_path, project="test", entity_a="X", entity_b="Y")
        assert _already_synthesized("test", "X", "Y") is True

    def test_false_for_different_project(self, tmp_path):
        one.synthesis.init_schema()
        db_path = str(tmp_path / "test.db")
        _insert_synthesis(db_path, project="other", entity_a="X", entity_b="Y")
        assert _already_synthesized("test", "X", "Y") is False

    def test_false_for_different_pair(self, tmp_path):
        one.synthesis.init_schema()
        db_path = str(tmp_path / "test.db")
        _insert_synthesis(db_path, project="test", entity_a="X", entity_b="Y")
        assert _already_synthesized("test", "X", "Z") is False


# ---------------------------------------------------------------------------
# Store and retrieve syntheses
# ---------------------------------------------------------------------------

class TestStoreSyntheses:
    def _seed(self, tmp_path, **kwargs):
        one.synthesis.init_schema()
        db_path = str(tmp_path / "test.db")
        return _insert_synthesis(db_path, **kwargs)

    def test_get_synthesis_chain_empty(self):
        one.synthesis.init_schema()
        assert get_synthesis_chain("test") == []

    def test_get_synthesis_chain_returns_inserted(self, tmp_path):
        self._seed(tmp_path, hypothesis="A causes B via pathway X.")
        chain = get_synthesis_chain("test")
        assert len(chain) == 1
        assert chain[0]["hypothesis"] == "A causes B via pathway X."
        assert chain[0]["entity_a"] == "A"
        assert chain[0]["entity_b"] == "B"

    def test_get_synthesis_chain_ordered_by_depth_then_confidence(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        one.synthesis.init_schema()
        _insert_synthesis(db_path, depth=0, confidence=0.5, hypothesis="depth0 low")
        _insert_synthesis(db_path, depth=0, confidence=0.9, hypothesis="depth0 high")
        pid = _insert_synthesis(db_path, depth=1, confidence=0.8, hypothesis="depth1")

        chain = get_synthesis_chain("test")
        assert len(chain) == 3
        # depth=0 rows come first
        depths = [r["depth"] for r in chain]
        assert depths[0] == 0
        assert depths[1] == 0
        assert depths[2] == 1
        # within depth=0, higher confidence first
        assert chain[0]["confidence"] == 0.9

    def test_get_synthesis_chain_project_isolation(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        one.synthesis.init_schema()
        _insert_synthesis(db_path, project="test", hypothesis="belongs to test")
        _insert_synthesis(db_path, project="other", hypothesis="belongs to other")
        chain = get_synthesis_chain("test")
        assert len(chain) == 1
        assert chain[0]["hypothesis"] == "belongs to test"

    def test_get_syntheses_count_empty(self):
        one.synthesis.init_schema()
        assert get_syntheses_count("test") == 0

    def test_get_syntheses_count_after_inserts(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        one.synthesis.init_schema()
        for i in range(3):
            _insert_synthesis(db_path, hypothesis=f"hypothesis {i}")
        assert get_syntheses_count("test") == 3

    def test_get_syntheses_count_project_isolation(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        one.synthesis.init_schema()
        _insert_synthesis(db_path, project="test", hypothesis="for test")
        _insert_synthesis(db_path, project="other", hypothesis="for other")
        assert get_syntheses_count("test") == 1
        assert get_syntheses_count("other") == 1

    def test_synthesis_row_fields(self, tmp_path):
        """Every field selected by get_synthesis_chain must be present.

        Note: get_synthesis_chain filters by project but does not include
        'project' in its SELECT list, so that column is intentionally absent.
        """
        self._seed(tmp_path, confidence=0.75, novelty_score=0.6, depth=0)
        chain = get_synthesis_chain("test")
        row = chain[0]
        for field in ("id", "entity_a", "entity_b", "hypothesis",
                      "confidence", "novelty_score", "tested", "test_result",
                      "parent_id", "depth", "created"):
            assert field in row, f"Missing field: {field}"

    def test_parent_id_preserved(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        one.synthesis.init_schema()
        parent_id = _insert_synthesis(db_path, depth=0, hypothesis="parent")
        _insert_synthesis(db_path, depth=1, hypothesis="child", parent_id=parent_id)
        chain = get_synthesis_chain("test")
        child = next(r for r in chain if r["hypothesis"] == "child")
        assert child["parent_id"] == parent_id


# ---------------------------------------------------------------------------
# _detect_contradictions (pure DB + HDC, no LLM)
# ---------------------------------------------------------------------------

class TestDetectContradictions:
    """_detect_contradictions uses real HDC vectors (no LLM).
    We test its observable behaviour: structure of results and boundary cases.
    """

    def test_returns_empty_when_no_syntheses(self):
        one.synthesis.init_schema()
        result = one.synthesis._detect_contradictions("test")
        assert result == []

    def test_returns_empty_with_single_synthesis(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        one.synthesis.init_schema()
        _insert_synthesis(db_path, hypothesis="A enables B.")
        result = one.synthesis._detect_contradictions("test")
        assert result == []

    def test_no_false_positives_for_unrelated_hypotheses(self, tmp_path):
        """Two hypotheses with completely disjoint vocabularies cannot form
        a contradiction signal, so the result should be empty."""
        db_path = str(tmp_path / "test.db")
        one.synthesis.init_schema()
        _insert_synthesis(db_path, entity_a="A", entity_b="B",
                          hypothesis="Photosynthesis converts sunlight to glucose.")
        _insert_synthesis(db_path, entity_a="C", entity_b="D",
                          hypothesis="Continental drift reshapes ocean floors.")
        result = one.synthesis._detect_contradictions("test")
        assert result == []

    def test_contradiction_result_structure(self, tmp_path):
        """When a contradiction is detected, the result dicts must have
        the expected keys."""
        db_path = str(tmp_path / "test.db")
        one.synthesis.init_schema()
        # A pair with high lexical overlap + asymmetric negation triggers a signal.
        _insert_synthesis(db_path, entity_a="A", entity_b="B",
                          hypothesis="A activates pathway B causing growth.",
                          confidence=0.8)
        _insert_synthesis(db_path, entity_a="C", entity_b="D",
                          hypothesis="A does not activate pathway B causing growth.",
                          confidence=0.8)
        result = one.synthesis._detect_contradictions("test")
        # If a contradiction is found, validate its structure.
        for item in result:
            assert "synthesis_1" in item
            assert "synthesis_2" in item
            assert "similarity" in item
            assert "type" in item
            assert item["type"] == "potential_contradiction"
            assert 0.0 <= item["similarity"] <= 1.0

    def test_project_isolation_in_contradiction_detection(self, tmp_path):
        """Syntheses from another project must not be compared."""
        db_path = str(tmp_path / "test.db")
        one.synthesis.init_schema()
        _insert_synthesis(db_path, project="other", entity_a="A", entity_b="B",
                          hypothesis="A activates pathway B causing growth.")
        _insert_synthesis(db_path, project="other", entity_a="C", entity_b="D",
                          hypothesis="A does not activate pathway B causing growth.")
        # "test" project has no syntheses
        result = one.synthesis._detect_contradictions("test")
        assert result == []


# ---------------------------------------------------------------------------
# get_contradictions public API
# ---------------------------------------------------------------------------

class TestGetContradictions:
    def test_returns_list(self):
        one.synthesis.init_schema()
        result = one.synthesis.get_contradictions("test")
        assert isinstance(result, list)

    def test_empty_for_empty_project(self):
        one.synthesis.init_schema()
        assert one.synthesis.get_contradictions("nonexistent") == []
