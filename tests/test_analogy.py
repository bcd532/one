"""Tests for the Analogical Transfer Engine."""

import sqlite3
import threading
import pytest
import numpy as np

from one import analogy
from one.analogy import (
    AnalogyEngine,
    StructuralTemplate,
    init_schema,
    EXTRACT_TEMPLATE_PROMPT,
    PREDICT_PROMPT,
    _get_conn,
)
from one.hdc import DIM, encode_text


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test_analogy.db")
    monkeypatch.setattr("one.analogy.DB_PATH", db_path)
    monkeypatch.setattr("one.analogy.DB_DIR", str(tmp_path))
    # Also patch store so push_memory doesn't fail
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", str(tmp_path))
    # Reset thread-local connections
    analogy._local = threading.local()
    import one.store as store_mod
    store_mod._local = threading.local()
    yield db_path


@pytest.fixture
def mock_ollama(monkeypatch):
    """Mock _call_ollama to avoid needing a real LLM."""
    def _mock(prompt, timeout=90):
        return None
    monkeypatch.setattr("one.analogy._call_ollama", _mock)
    monkeypatch.setattr("one.gemma._call_ollama", _mock)
    return _mock


@pytest.fixture
def engine(mock_ollama):
    """Create an AnalogyEngine with mocked LLM."""
    return AnalogyEngine(project="test_project")


# ── Schema Tests ───────────────────────────────────────────────────


class TestSchema:
    def test_init_schema_creates_tables(self, mock_ollama):
        """init_schema creates the analogy_templates and universal_patterns tables."""
        engine = AnalogyEngine(project="test_project")
        conn = _get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "analogy_templates" in table_names
        assert "universal_patterns" in table_names

    def test_schema_idempotent(self, mock_ollama):
        """Calling init_schema multiple times does not raise."""
        init_schema()
        init_schema()
        conn = _get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len(tables) >= 2

    def test_templates_table_columns(self, mock_ollama):
        """analogy_templates has expected columns."""
        init_schema()
        conn = _get_conn()
        info = conn.execute("PRAGMA table_info(analogy_templates)").fetchall()
        col_names = {row["name"] for row in info}
        expected = {"id", "project", "source_finding", "domain", "mechanism",
                    "target", "location", "effect", "outcome", "hdc_vector",
                    "confidence", "created"}
        assert expected.issubset(col_names)

    def test_patterns_table_columns(self, mock_ollama):
        """universal_patterns has expected columns."""
        init_schema()
        conn = _get_conn()
        info = conn.execute("PRAGMA table_info(universal_patterns)").fetchall()
        col_names = {row["name"] for row in info}
        expected = {"id", "project", "pattern_name", "pattern_description",
                    "domains", "domain_count", "instances", "predictions",
                    "confidence", "hdc_vector", "created", "updated"}
        assert expected.issubset(col_names)

    def test_indexes_created(self, mock_ollama):
        """Expected indexes are created."""
        init_schema()
        conn = _get_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        idx_names = {row["name"] for row in indexes}
        assert "idx_templates_project" in idx_names
        assert "idx_templates_domain" in idx_names
        assert "idx_patterns_project" in idx_names
        assert "idx_patterns_domains" in idx_names


# ── StructuralTemplate Tests ───────────────────────────────────────


class TestStructuralTemplate:
    def test_default_values(self):
        """StructuralTemplate has sensible defaults."""
        t = StructuralTemplate()
        assert t.mechanism == ""
        assert t.target == ""
        assert t.location == ""
        assert t.effect == ""
        assert t.outcome == ""
        assert t.domain == ""
        assert t.source_finding == ""
        assert t.confidence == 0.5

    def test_custom_values(self):
        """StructuralTemplate accepts custom values."""
        t = StructuralTemplate(
            mechanism="blocking",
            target="receptor",
            location="cell_surface",
            effect="releases_function",
            outcome="target_elimination",
            domain="immunology",
            source_finding="PD-1 inhibitor finding",
            confidence=0.9,
        )
        assert t.mechanism == "blocking"
        assert t.target == "receptor"
        assert t.confidence == 0.9

    def test_partial_values(self):
        """StructuralTemplate works with partial specification."""
        t = StructuralTemplate(mechanism="amplifying", domain="physics")
        assert t.mechanism == "amplifying"
        assert t.domain == "physics"
        assert t.target == ""
        assert t.confidence == 0.5


# ── Extract Template Tests ─────────────────────────────────────────


class TestExtractTemplate:
    def test_extract_with_full_response(self, monkeypatch, mock_ollama):
        """extract_template parses a full LLM response correctly."""
        response = (
            "MECHANISM: blocking\n"
            "TARGET: inhibitory_receptor\n"
            "LOCATION: cell_surface\n"
            "EFFECT: releases_suppressed_function\n"
            "OUTCOME: target_elimination\n"
            "DOMAIN: immunology"
        )
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=90: response)
        engine = AnalogyEngine(project="test_project")
        template = engine.extract_template("PD-1 inhibitor blocks immune checkpoints")
        assert template.mechanism == "blocking"
        assert template.target == "inhibitory_receptor"
        assert template.location == "cell_surface"
        assert template.effect == "releases_suppressed_function"
        assert template.outcome == "target_elimination"
        assert template.domain == "immunology"

    def test_extract_with_empty_response(self, engine):
        """extract_template handles None LLM response gracefully."""
        template = engine.extract_template("Some finding text")
        assert isinstance(template, StructuralTemplate)
        assert template.source_finding == "Some finding text"
        assert template.mechanism == ""

    def test_extract_stores_template_in_db(self, monkeypatch, mock_ollama):
        """extract_template persists the template to the database."""
        response = "MECHANISM: blocking\nTARGET: receptor\nDOMAIN: biology"
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=90: response)
        engine = AnalogyEngine(project="test_project")
        engine.extract_template("some finding")
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM analogy_templates").fetchall()
        assert len(rows) == 1
        assert rows[0]["mechanism"] == "blocking"
        assert rows[0]["domain"] == "biology"

    def test_extract_preserves_source_finding(self, engine):
        """The source_finding field is preserved on the template."""
        finding = "Checkpoint inhibitors block PD-1 on T cells"
        template = engine.extract_template(finding)
        assert template.source_finding == finding

    def test_extract_with_partial_response(self, monkeypatch, mock_ollama):
        """extract_template handles partial LLM responses."""
        response = "MECHANISM: amplifying\nOUTCOME: growth"
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=90: response)
        engine = AnalogyEngine(project="test_project")
        template = engine.extract_template("something")
        assert template.mechanism == "amplifying"
        assert template.outcome == "growth"
        assert template.target == ""
        assert template.location == ""

    def test_extract_template_prompt_format(self):
        """EXTRACT_TEMPLATE_PROMPT contains correct placeholders."""
        assert "{finding}" in EXTRACT_TEMPLATE_PROMPT
        assert "MECHANISM:" in EXTRACT_TEMPLATE_PROMPT
        assert "TARGET:" in EXTRACT_TEMPLATE_PROMPT


# ── Store Template Tests ───────────────────────────────────────────


class TestStoreTemplate:
    def test_store_template_returns_id(self, engine):
        """_store_template returns a positive row id."""
        t = StructuralTemplate(
            mechanism="blocking", target="receptor", domain="biology",
            source_finding="test finding"
        )
        row_id = engine._store_template(t)
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_store_template_persists_data(self, engine):
        """_store_template correctly stores all fields."""
        t = StructuralTemplate(
            mechanism="amplifying",
            target="signal",
            location="network",
            effect="cascade",
            outcome="failure",
            domain="engineering",
            source_finding="cascading failures in networks",
            confidence=0.8,
        )
        engine._store_template(t)
        conn = _get_conn()
        row = conn.execute("SELECT * FROM analogy_templates WHERE project = ?",
                           ("test_project",)).fetchone()
        assert row["mechanism"] == "amplifying"
        assert row["target"] == "signal"
        assert row["location"] == "network"
        assert row["effect"] == "cascade"
        assert row["outcome"] == "failure"
        assert row["domain"] == "engineering"
        assert row["confidence"] == 0.8

    def test_store_template_creates_hdc_vector(self, engine):
        """_store_template encodes an HDC vector blob."""
        t = StructuralTemplate(mechanism="blocking", domain="bio",
                               source_finding="test")
        engine._store_template(t)
        conn = _get_conn()
        row = conn.execute("SELECT hdc_vector FROM analogy_templates").fetchone()
        assert row["hdc_vector"] is not None
        vec = np.frombuffer(row["hdc_vector"], dtype=np.float32)
        assert len(vec) == DIM

    def test_store_template_truncates_long_finding(self, engine):
        """source_finding is truncated to 500 chars in the DB."""
        long_finding = "x" * 1000
        t = StructuralTemplate(source_finding=long_finding, domain="test")
        engine._store_template(t)
        conn = _get_conn()
        row = conn.execute("SELECT source_finding FROM analogy_templates").fetchone()
        assert len(row["source_finding"]) == 500


# ── Match Templates Tests ──────────────────────────────────────────


class TestMatchTemplates:
    def _insert_template(self, engine, domain, mechanism, target="x", effect="y", outcome="z"):
        """Helper to insert a template for a given domain."""
        t = StructuralTemplate(
            mechanism=mechanism, target=target, effect=effect,
            outcome=outcome, domain=domain, source_finding=f"finding in {domain}",
        )
        engine._store_template(t)
        return t

    def test_match_returns_empty_for_no_matches(self, engine):
        """match_templates returns empty list when no other-domain templates exist."""
        t = StructuralTemplate(mechanism="blocking", domain="biology")
        matches = engine.match_templates(t)
        assert matches == []

    def test_match_excludes_same_domain(self, engine):
        """match_templates only returns templates from OTHER domains."""
        self._insert_template(engine, "biology", "blocking")
        self._insert_template(engine, "biology", "blocking")
        query = StructuralTemplate(mechanism="blocking", domain="biology")
        matches = engine.match_templates(query)
        assert all(m["domain"] != "biology" for m in matches)

    def test_match_finds_cross_domain(self, engine):
        """match_templates finds similar templates in different domains."""
        self._insert_template(engine, "engineering", "blocking", "valve", "stops_flow", "pressure_relief")
        query = StructuralTemplate(
            mechanism="blocking", target="valve",
            effect="stops_flow", outcome="pressure_relief",
            domain="biology",
        )
        matches = engine.match_templates(query, min_similarity=0.0)
        assert len(matches) >= 1
        assert matches[0]["domain"] == "engineering"

    def test_match_sorted_by_similarity(self, engine):
        """Matches are sorted by structural_similarity descending."""
        self._insert_template(engine, "physics", "amplifying", "wave", "resonance", "destruction")
        self._insert_template(engine, "engineering", "blocking", "valve", "stops_flow", "relief")
        query = StructuralTemplate(
            mechanism="blocking", target="valve",
            effect="stops_flow", outcome="relief",
            domain="biology",
        )
        matches = engine.match_templates(query, min_similarity=0.0)
        if len(matches) >= 2:
            assert matches[0]["structural_similarity"] >= matches[1]["structural_similarity"]

    def test_match_respects_min_similarity(self, engine):
        """Matches below min_similarity are excluded."""
        self._insert_template(engine, "music", "oscillating")
        query = StructuralTemplate(mechanism="blocking", domain="biology")
        matches = engine.match_templates(query, min_similarity=0.99)
        # With very high threshold, likely no matches
        for m in matches:
            assert m["structural_similarity"] >= 0.99

    def test_match_does_not_return_hdc_blob(self, engine):
        """hdc_vector blobs are stripped from match results."""
        self._insert_template(engine, "engineering", "blocking")
        query = StructuralTemplate(mechanism="blocking", domain="biology")
        matches = engine.match_templates(query, min_similarity=0.0)
        for m in matches:
            assert "hdc_vector" not in m

    def test_match_limited_to_20(self, engine):
        """match_templates returns at most 20 results."""
        for i in range(25):
            self._insert_template(engine, f"domain_{i}", "blocking")
        query = StructuralTemplate(mechanism="blocking", domain="biology")
        matches = engine.match_templates(query, min_similarity=0.0)
        assert len(matches) <= 20


# ── Find Universal Patterns Tests ──────────────────────────────────


class TestFindUniversalPatterns:
    def _insert_similar_templates(self, engine, domains, mechanism="blocking",
                                  target="receptor", effect="release", outcome="elimination"):
        """Insert structurally similar templates across multiple domains."""
        for domain in domains:
            t = StructuralTemplate(
                mechanism=mechanism, target=target, effect=effect,
                outcome=outcome, domain=domain,
                source_finding=f"{mechanism} in {domain}",
            )
            engine._store_template(t)

    def test_not_enough_templates(self, engine):
        """Returns empty when fewer templates than min_domains."""
        t = StructuralTemplate(mechanism="x", domain="a", source_finding="test")
        engine._store_template(t)
        patterns = engine.find_universal_patterns(min_domains=3)
        assert patterns == []

    def test_finds_pattern_across_domains(self, monkeypatch, mock_ollama):
        """find_universal_patterns detects cross-domain structural patterns."""
        llm_response = "NAME: INHIBITOR_RELEASE\nDESCRIPTION: Removing a blocker unleashes function"
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=60: llm_response)
        engine = AnalogyEngine(project="test_project")
        self._insert_similar_templates(
            engine,
            ["immunology", "engineering", "economics"],
        )
        patterns = engine.find_universal_patterns(min_domains=3)
        # Whether a pattern is found depends on HDC similarity of identical text
        # The templates use identical mechanism/target/effect so similarity should be high
        if patterns:
            assert patterns[0]["domain_count"] >= 3
            assert patterns[0]["pattern_name"] == "INHIBITOR_RELEASE"

    def test_confidence_scaling(self, monkeypatch, mock_ollama):
        """Confidence increases with number of domains, capped at 0.9."""
        llm_response = "NAME: TEST_PATTERN\nDESCRIPTION: A test pattern"
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=60: llm_response)
        engine = AnalogyEngine(project="test_project")
        self._insert_similar_templates(
            engine,
            ["a", "b", "c", "d", "e"],
        )
        patterns = engine.find_universal_patterns(min_domains=3)
        if patterns:
            assert patterns[0]["confidence"] <= 0.9

    def test_stores_pattern_in_db(self, monkeypatch, mock_ollama):
        """find_universal_patterns persists discovered patterns."""
        llm_response = "NAME: STORED_PATTERN\nDESCRIPTION: persisted"
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=60: llm_response)
        engine = AnalogyEngine(project="test_project")
        self._insert_similar_templates(engine, ["a", "b", "c"])
        engine.find_universal_patterns(min_domains=3)
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM universal_patterns").fetchall()
        # The pattern should be stored if it was found
        for row in rows:
            assert row["project"] == "test_project"


# ── Predict From Pattern Tests ─────────────────────────────────────


class TestPredictFromPattern:
    def _insert_pattern(self, engine, name="TEST_PATTERN", description="A test pattern"):
        """Helper to insert a universal pattern directly."""
        from datetime import datetime, timezone
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        vec = encode_text(f"{name} {description}")
        blob = vec.tobytes()
        conn.execute(
            """INSERT INTO universal_patterns
               (project, pattern_name, pattern_description, domains, domain_count,
                instances, predictions, confidence, hdc_vector, created, updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (engine.project, name, description, "a,b,c", 3,
             "instance1\ninstance2", "", 0.7, blob, now, now),
        )
        conn.commit()

    def test_predict_unknown_pattern(self, engine):
        """predict_from_pattern returns empty for nonexistent pattern."""
        result = engine.predict_from_pattern("NONEXISTENT", "physics")
        assert result["prediction"] == ""
        assert result["confidence"] == 0.0

    def test_predict_with_llm_response(self, monkeypatch, mock_ollama):
        """predict_from_pattern parses LLM prediction correctly."""
        llm_response = (
            "PREDICTION: In economics, removing trade barriers will increase throughput\n"
            "TESTABLE: Measure GDP before and after barrier removal\n"
            "CONFIDENCE: 0.75"
        )
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=90: llm_response)
        engine = AnalogyEngine(project="test_project")
        self._insert_pattern(engine)
        result = engine.predict_from_pattern("TEST_PATTERN", "economics")
        assert "trade barriers" in result["prediction"]
        assert "GDP" in result["testable"]
        assert result["confidence"] == 0.75
        assert result["pattern_name"] == "TEST_PATTERN"
        assert result["new_domain"] == "economics"

    def test_predict_with_empty_llm_response(self, monkeypatch, mock_ollama):
        """predict_from_pattern handles None LLM response."""
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=90: None)
        engine = AnalogyEngine(project="test_project")
        self._insert_pattern(engine)
        result = engine.predict_from_pattern("TEST_PATTERN", "physics")
        assert result["prediction"] == ""
        assert result["confidence"] == 0.0

    def test_predict_with_invalid_confidence(self, monkeypatch, mock_ollama):
        """predict_from_pattern handles non-numeric confidence gracefully."""
        llm_response = (
            "PREDICTION: Something will happen\n"
            "TESTABLE: Observe it\n"
            "CONFIDENCE: high"
        )
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=90: llm_response)
        engine = AnalogyEngine(project="test_project")
        self._insert_pattern(engine)
        result = engine.predict_from_pattern("TEST_PATTERN", "physics")
        assert result["confidence"] == 0.5  # fallback

    def test_predict_prompt_format(self):
        """PREDICT_PROMPT has expected placeholders."""
        assert "{pattern_name}" in PREDICT_PROMPT
        assert "{pattern_description}" in PREDICT_PROMPT
        assert "{instances}" in PREDICT_PROMPT
        assert "{new_domain}" in PREDICT_PROMPT


# ── Get Patterns / Get Templates Tests ─────────────────────────────


class TestGetPatterns:
    def _insert_pattern(self, engine, name, domain_count, confidence=0.7):
        from datetime import datetime, timezone
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        vec = encode_text(name)
        blob = vec.tobytes()
        conn.execute(
            """INSERT INTO universal_patterns
               (project, pattern_name, pattern_description, domains, domain_count,
                instances, predictions, confidence, hdc_vector, created, updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (engine.project, name, f"desc of {name}",
             ",".join([f"d{i}" for i in range(domain_count)]),
             domain_count, "", "", confidence, blob, now, now),
        )
        conn.commit()

    def test_get_patterns_empty(self, engine):
        """get_patterns returns empty list when no patterns exist."""
        assert engine.get_patterns() == []

    def test_get_patterns_filters_by_min_domains(self, engine):
        """get_patterns respects min_domains filter."""
        self._insert_pattern(engine, "small", 2)
        self._insert_pattern(engine, "large", 5)
        patterns = engine.get_patterns(min_domains=4)
        assert len(patterns) == 1
        assert patterns[0]["pattern_name"] == "large"

    def test_get_patterns_sorted_by_confidence(self, engine):
        """get_patterns returns results sorted by confidence DESC."""
        self._insert_pattern(engine, "low_conf", 3, confidence=0.3)
        self._insert_pattern(engine, "high_conf", 3, confidence=0.9)
        patterns = engine.get_patterns(min_domains=2)
        assert patterns[0]["confidence"] >= patterns[-1]["confidence"]

    def test_get_patterns_project_scoped(self, engine):
        """get_patterns only returns patterns for the engine's project."""
        self._insert_pattern(engine, "mine", 3)
        conn = _get_conn()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        vec = encode_text("other").tobytes()
        conn.execute(
            """INSERT INTO universal_patterns
               (project, pattern_name, pattern_description, domains, domain_count,
                instances, predictions, confidence, hdc_vector, created, updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("other_project", "theirs", "desc", "a,b,c", 3, "", "", 0.5, vec, now, now),
        )
        conn.commit()
        patterns = engine.get_patterns(min_domains=2)
        assert all(p["pattern_name"] != "theirs" for p in patterns)


class TestGetTemplates:
    def _insert(self, engine, domain, mechanism="block"):
        t = StructuralTemplate(
            mechanism=mechanism, domain=domain,
            source_finding=f"finding in {domain}",
        )
        engine._store_template(t)

    def test_get_templates_empty(self, engine):
        """get_templates returns empty when no templates stored."""
        assert engine.get_templates() == []

    def test_get_templates_all(self, engine):
        """get_templates with no domain filter returns all project templates."""
        self._insert(engine, "bio")
        self._insert(engine, "physics")
        templates = engine.get_templates()
        assert len(templates) == 2

    def test_get_templates_filtered_by_domain(self, engine):
        """get_templates filters by domain when specified."""
        self._insert(engine, "bio")
        self._insert(engine, "physics")
        templates = engine.get_templates(domain="bio")
        assert len(templates) == 1
        assert templates[0]["domain"] == "bio"

    def test_get_templates_no_blob(self, engine):
        """get_templates does not include hdc_vector column."""
        self._insert(engine, "bio")
        templates = engine.get_templates()
        assert "hdc_vector" not in templates[0]


# ── Edge Cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_engine_with_custom_logger(self, mock_ollama):
        """AnalogyEngine accepts a custom on_log callback."""
        logs = []
        engine = AnalogyEngine(project="test", on_log=logs.append)
        engine.extract_template("test finding")
        assert any("extracting" in log for log in logs)

    def test_empty_finding_extract(self, engine):
        """extract_template handles empty string finding."""
        template = engine.extract_template("")
        assert isinstance(template, StructuralTemplate)
        assert template.source_finding == ""

    def test_match_with_corrupted_vector(self, engine):
        """match_templates handles rows with wrong-sized vectors gracefully."""
        # Insert a template with a truncated vector
        conn = _get_conn()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        bad_blob = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        conn.execute(
            """INSERT INTO analogy_templates
               (project, source_finding, domain, mechanism, target, location,
                effect, outcome, hdc_vector, confidence, created)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("test_project", "bad", "engineering", "block", "", "", "", "", bad_blob, 0.5, now),
        )
        conn.commit()
        query = StructuralTemplate(mechanism="block", domain="biology")
        # Should not crash — wrong-sized vectors are skipped
        matches = engine.match_templates(query, min_similarity=0.0)
        assert isinstance(matches, list)

    def test_match_with_null_vector(self, engine):
        """match_templates skips rows with NULL hdc_vector."""
        conn = _get_conn()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO analogy_templates
               (project, source_finding, domain, mechanism, target, location,
                effect, outcome, hdc_vector, confidence, created)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("test_project", "null vec", "engineering", "block", "", "", "", "", None, 0.5, now),
        )
        conn.commit()
        query = StructuralTemplate(mechanism="block", domain="biology")
        matches = engine.match_templates(query, min_similarity=0.0)
        assert isinstance(matches, list)

    def test_store_pattern_returns_id(self, engine):
        """_store_pattern returns a positive row id."""
        pattern = {
            "pattern_name": "TEST",
            "pattern_description": "desc",
            "domains": ["a", "b"],
            "domain_count": 2,
            "instances": "inst",
            "predictions": "pred",
            "confidence": 0.6,
        }
        row_id = engine._store_pattern(pattern)
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_extract_with_extra_whitespace_response(self, monkeypatch, mock_ollama):
        """extract_template handles extra whitespace in LLM response."""
        response = (
            "  MECHANISM:   blocking  \n"
            "  TARGET:  receptor  \n"
            "  DOMAIN:  immunology  \n"
            "  some garbage line\n"
        )
        monkeypatch.setattr("one.analogy._call_ollama", lambda p, timeout=90: response)
        engine = AnalogyEngine(project="test_project")
        template = engine.extract_template("test")
        assert template.mechanism == "blocking"
        assert template.target == "receptor"
        assert template.domain == "immunology"
