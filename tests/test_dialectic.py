"""Tests for the Adversarial Dialectic Engine."""

import sqlite3
import threading
import pytest
import numpy as np

from one import dialectic
from one import store
from one.dialectic import (
    DialecticEngine,
    NODE_TYPES,
    init_schema,
    CHALLENGE_PROMPT,
    SYNTHESIZE_PROMPT,
    VERIFY_PROMPT,
    META_SYNTHESIS_PROMPT,
    _get_conn,
)


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test_dialectic.db")
    monkeypatch.setattr("one.dialectic.DB_PATH", db_path)
    monkeypatch.setattr("one.dialectic.DB_DIR", str(tmp_path))
    # Also patch store so push_memory / recall don't fail
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", str(tmp_path))
    # Reset thread-local connections
    dialectic._local = threading.local()
    store._local = threading.local()
    store.set_project("test_project")
    yield db_path


@pytest.fixture
def mock_ollama(monkeypatch):
    """Mock _call_ollama to avoid needing a real LLM."""
    def _mock(prompt, timeout=90):
        return None
    monkeypatch.setattr("one.dialectic._call_ollama", _mock)
    monkeypatch.setattr("one.gemma._call_ollama", _mock)
    return _mock


@pytest.fixture
def engine(mock_ollama):
    """Create a DialecticEngine with mocked LLM."""
    return DialecticEngine(project="test_project")


# ── NODE_TYPES Constant Tests ─────────────────────────────────────


class TestNodeTypes:
    def test_node_types_is_list(self):
        """NODE_TYPES is a list."""
        assert isinstance(NODE_TYPES, list)

    def test_node_types_contains_all_stages(self):
        """NODE_TYPES contains all dialectic stages."""
        assert "thesis" in NODE_TYPES
        assert "antithesis" in NODE_TYPES
        assert "synthesis" in NODE_TYPES
        assert "verification" in NODE_TYPES
        assert "meta_synthesis" in NODE_TYPES

    def test_node_types_order(self):
        """NODE_TYPES are in the correct dialectic order."""
        assert NODE_TYPES.index("thesis") < NODE_TYPES.index("antithesis")
        assert NODE_TYPES.index("antithesis") < NODE_TYPES.index("synthesis")
        assert NODE_TYPES.index("synthesis") < NODE_TYPES.index("verification")
        assert NODE_TYPES.index("verification") < NODE_TYPES.index("meta_synthesis")

    def test_node_types_length(self):
        """NODE_TYPES has exactly 5 stages."""
        assert len(NODE_TYPES) == 5


# ── Schema Tests ───────────────────────────────────────────────────


class TestSchema:
    def test_init_schema_creates_tables(self, mock_ollama):
        """init_schema creates dialectic_chains and dialectic_nodes tables."""
        engine = DialecticEngine(project="test_project")
        conn = _get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "dialectic_chains" in table_names
        assert "dialectic_nodes" in table_names

    def test_schema_idempotent(self, mock_ollama):
        """Calling init_schema multiple times does not raise."""
        init_schema()
        init_schema()
        conn = _get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len(tables) >= 2

    def test_chains_table_columns(self, mock_ollama):
        """dialectic_chains has expected columns."""
        init_schema()
        conn = _get_conn()
        info = conn.execute("PRAGMA table_info(dialectic_chains)").fetchall()
        col_names = {row["name"] for row in info}
        expected = {"id", "project", "topic", "status", "created", "updated"}
        assert expected.issubset(col_names)

    def test_nodes_table_columns(self, mock_ollama):
        """dialectic_nodes has expected columns."""
        init_schema()
        conn = _get_conn()
        info = conn.execute("PRAGMA table_info(dialectic_nodes)").fetchall()
        col_names = {row["name"] for row in info}
        expected = {"id", "chain_id", "node_type", "content", "confidence",
                    "source", "evidence", "parent_node_id", "created"}
        assert expected.issubset(col_names)

    def test_indexes_created(self, mock_ollama):
        """Expected indexes are created."""
        init_schema()
        conn = _get_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        idx_names = {row["name"] for row in indexes}
        assert "idx_dialectic_project" in idx_names
        assert "idx_dialectic_nodes_chain" in idx_names
        assert "idx_dialectic_nodes_type" in idx_names

    def test_chains_default_status(self, mock_ollama):
        """dialectic_chains has default status of 'active'."""
        init_schema()
        conn = _get_conn()
        info = conn.execute("PRAGMA table_info(dialectic_chains)").fetchall()
        status_col = [r for r in info if r["name"] == "status"][0]
        assert status_col["dflt_value"] == "'active'"


# ── Challenge Tests ────────────────────────────────────────────────


class TestChallenge:
    def test_challenge_returns_dict_keys(self, engine):
        """challenge returns a dict with expected keys."""
        result = engine.challenge("The earth is round")
        assert "chain_id" in result
        assert "thesis_id" in result
        assert "antithesis_id" in result
        assert "thesis" in result
        assert "antithesis" in result

    def test_challenge_stores_thesis(self, engine):
        """challenge persists the thesis as a dialectic node."""
        finding = "Checkpoint inhibitors are effective"
        result = engine.challenge(finding)
        conn = _get_conn()
        node = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ? AND node_type = 'thesis'",
            (result["chain_id"],),
        ).fetchone()
        assert node is not None
        assert node["content"] == finding

    def test_challenge_stores_antithesis(self, engine):
        """challenge persists the antithesis node."""
        result = engine.challenge("Some finding")
        conn = _get_conn()
        node = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ? AND node_type = 'antithesis'",
            (result["chain_id"],),
        ).fetchone()
        assert node is not None

    def test_challenge_with_llm_response(self, monkeypatch, mock_ollama):
        """challenge uses LLM-generated antithesis text."""
        antithesis_text = "Study had only 12 participants, p-value was 0.049"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: antithesis_text)
        engine = DialecticEngine(project="test_project")
        result = engine.challenge("Drug X cures cancer")
        assert result["antithesis"] == antithesis_text

    def test_challenge_with_empty_llm_response(self, engine):
        """challenge handles None LLM response with fallback text."""
        result = engine.challenge("Some finding")
        assert "No counter-argument" in result["antithesis"]

    def test_challenge_creates_chain(self, engine):
        """challenge creates a dialectic_chains row."""
        result = engine.challenge("Test finding")
        conn = _get_conn()
        chain = conn.execute(
            "SELECT * FROM dialectic_chains WHERE id = ?",
            (result["chain_id"],),
        ).fetchone()
        assert chain is not None
        assert chain["project"] == "test_project"
        assert chain["topic"] == "Test finding"[:200]

    def test_challenge_preserves_thesis_text(self, engine):
        """The thesis in the result matches the original finding."""
        finding = "Specific finding about immunotherapy"
        result = engine.challenge(finding)
        assert result["thesis"] == finding

    def test_challenge_with_source(self, engine):
        """challenge stores the source on the thesis node."""
        result = engine.challenge("Finding", source="pubmed:12345")
        conn = _get_conn()
        node = conn.execute(
            "SELECT source FROM dialectic_nodes WHERE id = ?",
            (result["thesis_id"],),
        ).fetchone()
        assert node["source"] == "pubmed:12345"

    def test_challenge_antithesis_links_to_thesis(self, engine):
        """The antithesis node has parent_node_id pointing to the thesis."""
        result = engine.challenge("A finding")
        conn = _get_conn()
        antithesis_node = conn.execute(
            "SELECT parent_node_id FROM dialectic_nodes WHERE id = ?",
            (result["antithesis_id"],),
        ).fetchone()
        assert antithesis_node["parent_node_id"] == result["thesis_id"]


# ── Synthesize Tests ───────────────────────────────────────────────


class TestSynthesize:
    def test_synthesize_returns_dict_keys(self, engine):
        """synthesize returns a dict with expected keys."""
        result = engine.synthesize("thesis text", "antithesis text")
        assert "chain_id" in result
        assert "synthesis" in result
        assert "thesis" in result
        assert "antithesis" in result

    def test_synthesize_with_llm_response(self, monkeypatch, mock_ollama):
        """synthesize uses LLM-generated synthesis text."""
        synthesis_text = "Both positions are partially correct because..."
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: synthesis_text)
        engine = DialecticEngine(project="test_project")
        result = engine.synthesize("thesis", "antithesis")
        assert result["synthesis"] == synthesis_text

    def test_synthesize_fallback_on_empty_response(self, engine):
        """synthesize uses fallback when LLM returns None."""
        result = engine.synthesize("thesis", "antithesis")
        assert result["synthesis"] == "(Synthesis generation failed)"

    def test_synthesize_with_chain_id_stores_node(self, engine):
        """synthesize stores a synthesis node when chain_id is provided."""
        chain_result = engine.challenge("test finding")
        chain_id = chain_result["chain_id"]
        engine.synthesize("thesis", "antithesis", chain_id=chain_id)
        conn = _get_conn()
        node = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ? AND node_type = 'synthesis'",
            (chain_id,),
        ).fetchone()
        assert node is not None

    def test_synthesize_without_chain_id_no_store(self, engine):
        """synthesize does not store a node when chain_id is 0."""
        engine.synthesize("thesis", "antithesis", chain_id=0)
        conn = _get_conn()
        node = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE node_type = 'synthesis'"
        ).fetchone()
        assert node is None

    def test_synthesize_updates_chain_timestamp(self, monkeypatch, mock_ollama):
        """synthesize updates the chain's updated timestamp."""
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: "synthesis result")
        engine = DialecticEngine(project="test_project")
        chain_result = engine.challenge("test finding")
        chain_id = chain_result["chain_id"]
        conn = _get_conn()
        original = conn.execute(
            "SELECT updated FROM dialectic_chains WHERE id = ?", (chain_id,)
        ).fetchone()["updated"]
        engine.synthesize("thesis", "antithesis", chain_id=chain_id)
        updated = conn.execute(
            "SELECT updated FROM dialectic_chains WHERE id = ?", (chain_id,)
        ).fetchone()["updated"]
        assert updated >= original

    def test_synthesize_preserves_inputs(self, engine):
        """synthesize preserves thesis and antithesis in the result."""
        result = engine.synthesize("my thesis", "my antithesis")
        assert result["thesis"] == "my thesis"
        assert result["antithesis"] == "my antithesis"

    def test_synthesize_node_confidence(self, engine):
        """synthesize stores node with confidence 0.6."""
        chain_result = engine.challenge("test finding")
        chain_id = chain_result["chain_id"]
        engine.synthesize("t", "a", chain_id=chain_id)
        conn = _get_conn()
        node = conn.execute(
            "SELECT confidence FROM dialectic_nodes WHERE chain_id = ? AND node_type = 'synthesis'",
            (chain_id,),
        ).fetchone()
        assert node["confidence"] == 0.6


# ── Verify Tests ───────────────────────────────────────────────────


class TestVerify:
    def test_verify_returns_dict_keys(self, monkeypatch, mock_ollama):
        """verify returns a dict with expected keys."""
        response = "VERDICT: SUPPORTED\nEVIDENCE: data\nCONFIDENCE: 0.7\nREASONING: good"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        result = engine.verify("synthesis text")
        assert "verdict" in result
        assert "confidence" in result
        assert "evidence" in result
        assert "reasoning" in result
        assert "chain_id" in result

    def test_verify_with_empty_response(self, engine):
        """verify returns UNTESTED verdict when LLM returns None."""
        result = engine.verify("some synthesis")
        assert result["verdict"] == "UNTESTED"
        assert result["confidence"] == 0.0

    def test_verify_parses_supported(self, monkeypatch, mock_ollama):
        """verify correctly parses SUPPORTED verdict."""
        response = (
            "VERDICT: SUPPORTED\n"
            "EVIDENCE: Multiple RCTs confirm this\n"
            "CONFIDENCE: 0.85\n"
            "REASONING: Strong evidence from meta-analysis"
        )
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        result = engine.verify("test synthesis")
        assert result["verdict"] == "SUPPORTED"
        assert result["confidence"] == 0.85
        assert "RCTs" in result["evidence"]
        assert "meta-analysis" in result["reasoning"]

    def test_verify_parses_contradicted(self, monkeypatch, mock_ollama):
        """verify correctly parses CONTRADICTED verdict."""
        response = "VERDICT: CONTRADICTED\nCONFIDENCE: 0.3\nEVIDENCE: none\nREASONING: flawed"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        result = engine.verify("bad synthesis")
        assert result["verdict"] == "CONTRADICTED"
        assert result["confidence"] == 0.3

    def test_verify_parses_partially_supported(self, monkeypatch, mock_ollama):
        """verify correctly parses PARTIALLY_SUPPORTED verdict."""
        response = "VERDICT: PARTIALLY_SUPPORTED\nCONFIDENCE: 0.6\nEVIDENCE: some\nREASONING: mixed"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        result = engine.verify("partial synthesis")
        assert result["verdict"] == "PARTIALLY_SUPPORTED"

    def test_verify_rejects_invalid_verdict(self, monkeypatch, mock_ollama):
        """verify falls back to UNTESTED for unrecognized verdict strings."""
        response = "VERDICT: MAYBE\nCONFIDENCE: 0.5\nEVIDENCE: n/a\nREASONING: unsure"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        result = engine.verify("test")
        assert result["verdict"] == "UNTESTED"

    def test_verify_handles_invalid_confidence(self, monkeypatch, mock_ollama):
        """verify handles non-numeric confidence gracefully."""
        response = "VERDICT: SUPPORTED\nCONFIDENCE: very high\nEVIDENCE: x\nREASONING: y"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        result = engine.verify("test")
        assert result["confidence"] == 0.5  # fallback

    def test_verify_stores_node_with_chain_id(self, monkeypatch, mock_ollama):
        """verify stores a verification node when chain_id is provided."""
        response = "VERDICT: SUPPORTED\nEVIDENCE: data\nCONFIDENCE: 0.8\nREASONING: solid"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        chain_result = engine.challenge("test finding")
        chain_id = chain_result["chain_id"]
        engine.verify("synthesis", chain_id=chain_id)
        conn = _get_conn()
        node = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ? AND node_type = 'verification'",
            (chain_id,),
        ).fetchone()
        assert node is not None

    def test_verify_no_store_without_chain_id(self, engine):
        """verify does not store a node when chain_id is 0."""
        engine.verify("synthesis", chain_id=0)
        conn = _get_conn()
        node = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE node_type = 'verification'"
        ).fetchone()
        assert node is None

    def test_verify_stores_evidence_field(self, monkeypatch, mock_ollama):
        """verify stores evidence in the node's evidence column."""
        response = "VERDICT: SUPPORTED\nEVIDENCE: RCT data\nCONFIDENCE: 0.9\nREASONING: strong"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        chain_result = engine.challenge("test")
        engine.verify("synth", chain_id=chain_result["chain_id"])
        conn = _get_conn()
        node = conn.execute(
            "SELECT evidence FROM dialectic_nodes WHERE chain_id = ? AND node_type = 'verification'",
            (chain_result["chain_id"],),
        ).fetchone()
        assert node["evidence"] == "RCT data"


# ── Meta-Synthesize Tests ──────────────────────────────────────────


class TestMetaSynthesize:
    def test_meta_synthesize_too_few(self, engine):
        """meta_synthesize returns empty result with fewer than 2 syntheses."""
        result = engine.meta_synthesize(["only one"])
        assert result["pattern_name"] == ""
        assert result["pattern"] == ""
        assert result["confidence"] == 0.0

    def test_meta_synthesize_empty_list(self, engine):
        """meta_synthesize handles empty list."""
        result = engine.meta_synthesize([])
        assert result["pattern_name"] == ""

    def test_meta_synthesize_with_llm_response(self, monkeypatch, mock_ollama):
        """meta_synthesize parses LLM response correctly."""
        response = (
            "PATTERN NAME: THRESHOLD_EFFECT\n"
            "PATTERN: Systems exhibit sudden transitions at critical thresholds\n"
            "PREDICTIONS: Other phase-transition-like behaviors should exist\n"
            "CONFIDENCE: 0.8"
        )
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        result = engine.meta_synthesize(["synth1", "synth2", "synth3"])
        assert result["pattern_name"] == "THRESHOLD_EFFECT"
        assert "thresholds" in result["pattern"]
        assert result["confidence"] == 0.8
        assert "phase-transition" in result["predictions"]

    def test_meta_synthesize_with_empty_llm_response(self, monkeypatch, mock_ollama):
        """meta_synthesize handles None LLM response."""
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: None)
        engine = DialecticEngine(project="test_project")
        result = engine.meta_synthesize(["a", "b"])
        assert result["pattern_name"] == ""
        assert result["confidence"] == 0.0

    def test_meta_synthesize_stores_node_with_chain_id(self, monkeypatch, mock_ollama):
        """meta_synthesize stores a meta_synthesis node when chain_id is provided."""
        response = "PATTERN NAME: TEST\nPATTERN: test\nPREDICTIONS: none\nCONFIDENCE: 0.5"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        chain_result = engine.challenge("finding")
        chain_id = chain_result["chain_id"]
        engine.meta_synthesize(["s1", "s2"], chain_id=chain_id)
        conn = _get_conn()
        node = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ? AND node_type = 'meta_synthesis'",
            (chain_id,),
        ).fetchone()
        assert node is not None

    def test_meta_synthesize_no_store_without_chain_id(self, engine):
        """meta_synthesize does not store a node when chain_id is 0."""
        engine.meta_synthesize(["a", "b"], chain_id=0)
        conn = _get_conn()
        node = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE node_type = 'meta_synthesis'"
        ).fetchone()
        assert node is None

    def test_meta_synthesize_invalid_confidence(self, monkeypatch, mock_ollama):
        """meta_synthesize handles non-numeric confidence gracefully."""
        response = "PATTERN NAME: X\nPATTERN: y\nPREDICTIONS: z\nCONFIDENCE: high"
        monkeypatch.setattr("one.dialectic._call_ollama",
                            lambda p, timeout=120: response)
        engine = DialecticEngine(project="test_project")
        result = engine.meta_synthesize(["a", "b"])
        assert result["confidence"] == 0.5  # fallback


# ── Run Full Dialectic Tests ───────────────────────────────────────


class TestRunFullDialectic:
    def test_full_dialectic_returns_keys(self, monkeypatch, mock_ollama):
        """run_full_dialectic returns dict with all expected keys."""
        call_count = {"n": 0}

        def _fake_ollama(prompt, timeout=120):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "This finding is flawed because..."
            elif call_count["n"] == 2:
                return "Both positions have merit..."
            elif call_count["n"] == 3:
                return "VERDICT: SUPPORTED\nEVIDENCE: data\nCONFIDENCE: 0.7\nREASONING: solid"
            return None

        monkeypatch.setattr("one.dialectic._call_ollama", _fake_ollama)
        engine = DialecticEngine(project="test_project")
        result = engine.run_full_dialectic("Test finding")
        assert "chain_id" in result
        assert "thesis" in result
        assert "antithesis" in result
        assert "synthesis" in result
        assert "verdict" in result
        assert "confidence" in result
        assert "status" in result

    def test_full_dialectic_supported_status(self, monkeypatch, mock_ollama):
        """run_full_dialectic sets status='verified' for SUPPORTED verdict."""
        call_count = {"n": 0}

        def _fake_ollama(prompt, timeout=120):
            call_count["n"] += 1
            if call_count["n"] == 3:
                return "VERDICT: SUPPORTED\nEVIDENCE: data\nCONFIDENCE: 0.8\nREASONING: good"
            return "some response"

        monkeypatch.setattr("one.dialectic._call_ollama", _fake_ollama)
        engine = DialecticEngine(project="test_project")
        result = engine.run_full_dialectic("Test finding")
        assert result["status"] == "verified"

    def test_full_dialectic_challenged_status(self, monkeypatch, mock_ollama):
        """run_full_dialectic sets status='challenged' for CONTRADICTED verdict."""
        call_count = {"n": 0}

        def _fake_ollama(prompt, timeout=120):
            call_count["n"] += 1
            if call_count["n"] == 3:
                return "VERDICT: CONTRADICTED\nEVIDENCE: none\nCONFIDENCE: 0.2\nREASONING: bad"
            return "some response"

        monkeypatch.setattr("one.dialectic._call_ollama", _fake_ollama)
        engine = DialecticEngine(project="test_project")
        result = engine.run_full_dialectic("Test finding")
        assert result["status"] == "challenged"

    def test_full_dialectic_partially_supported_is_verified(self, monkeypatch, mock_ollama):
        """run_full_dialectic sets status='verified' for PARTIALLY_SUPPORTED."""
        call_count = {"n": 0}

        def _fake_ollama(prompt, timeout=120):
            call_count["n"] += 1
            if call_count["n"] == 3:
                return "VERDICT: PARTIALLY_SUPPORTED\nEVIDENCE: some\nCONFIDENCE: 0.6\nREASONING: mixed"
            return "some response"

        monkeypatch.setattr("one.dialectic._call_ollama", _fake_ollama)
        engine = DialecticEngine(project="test_project")
        result = engine.run_full_dialectic("Test finding")
        assert result["status"] == "verified"

    def test_full_dialectic_updates_chain_status_in_db(self, monkeypatch, mock_ollama):
        """run_full_dialectic persists the final status to the chain row."""
        call_count = {"n": 0}

        def _fake_ollama(prompt, timeout=120):
            call_count["n"] += 1
            if call_count["n"] == 3:
                return "VERDICT: SUPPORTED\nEVIDENCE: e\nCONFIDENCE: 0.9\nREASONING: r"
            return "response"

        monkeypatch.setattr("one.dialectic._call_ollama", _fake_ollama)
        engine = DialecticEngine(project="test_project")
        result = engine.run_full_dialectic("Finding")
        conn = _get_conn()
        chain = conn.execute(
            "SELECT status FROM dialectic_chains WHERE id = ?",
            (result["chain_id"],),
        ).fetchone()
        assert chain["status"] == "verified"

    def test_full_dialectic_creates_all_node_types(self, monkeypatch, mock_ollama):
        """run_full_dialectic creates thesis, antithesis, synthesis, and verification nodes."""
        call_count = {"n": 0}

        def _fake_ollama(prompt, timeout=120):
            call_count["n"] += 1
            if call_count["n"] == 3:
                return "VERDICT: SUPPORTED\nEVIDENCE: e\nCONFIDENCE: 0.8\nREASONING: r"
            return "response text"

        monkeypatch.setattr("one.dialectic._call_ollama", _fake_ollama)
        engine = DialecticEngine(project="test_project")
        result = engine.run_full_dialectic("Finding")
        chain = engine.get_chain(result["chain_id"])
        node_types = [n["node_type"] for n in chain["nodes"]]
        assert "thesis" in node_types
        assert "antithesis" in node_types
        assert "synthesis" in node_types
        assert "verification" in node_types


# ── Store Chain / Get Chain / Get Chains Tests ─────────────────────


class TestStoreChain:
    def test_store_chain_returns_id(self, engine):
        """store_chain returns a positive chain id."""
        chain = {"thesis": "T", "antithesis": "A", "synthesis": "S"}
        chain_id = engine.store_chain(chain)
        assert isinstance(chain_id, int)
        assert chain_id > 0

    def test_store_chain_creates_nodes(self, engine):
        """store_chain creates nodes for each provided node_type."""
        chain = {
            "thesis": "Thesis text",
            "antithesis": "Antithesis text",
            "synthesis": "Synthesis text",
            "verification": "Verification text",
            "meta_synthesis": "Meta synthesis text",
            "confidence": 0.8,
        }
        chain_id = engine.store_chain(chain)
        conn = _get_conn()
        nodes = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ? ORDER BY id",
            (chain_id,),
        ).fetchall()
        assert len(nodes) == 5
        assert nodes[0]["node_type"] == "thesis"
        assert nodes[1]["node_type"] == "antithesis"
        assert nodes[2]["node_type"] == "synthesis"

    def test_store_chain_links_parent_nodes(self, engine):
        """store_chain sets parent_node_id linking nodes sequentially."""
        chain = {"thesis": "T", "antithesis": "A", "synthesis": "S"}
        chain_id = engine.store_chain(chain)
        conn = _get_conn()
        nodes = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ? ORDER BY id",
            (chain_id,),
        ).fetchall()
        # First node has no parent
        assert nodes[0]["parent_node_id"] is None
        # Second node's parent is the first
        assert nodes[1]["parent_node_id"] == nodes[0]["id"]
        # Third node's parent is the second
        assert nodes[2]["parent_node_id"] == nodes[1]["id"]

    def test_store_chain_skips_empty_types(self, engine):
        """store_chain only creates nodes for non-empty content."""
        chain = {"thesis": "T", "antithesis": "A"}
        chain_id = engine.store_chain(chain)
        conn = _get_conn()
        nodes = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ?",
            (chain_id,),
        ).fetchall()
        assert len(nodes) == 2

    def test_store_chain_with_status(self, engine):
        """store_chain respects the status field."""
        chain = {"thesis": "T", "status": "verified"}
        chain_id = engine.store_chain(chain)
        conn = _get_conn()
        row = conn.execute(
            "SELECT status FROM dialectic_chains WHERE id = ?", (chain_id,)
        ).fetchone()
        assert row["status"] == "verified"

    def test_store_chain_default_status(self, engine):
        """store_chain defaults to 'active' status."""
        chain = {"thesis": "T"}
        chain_id = engine.store_chain(chain)
        conn = _get_conn()
        row = conn.execute(
            "SELECT status FROM dialectic_chains WHERE id = ?", (chain_id,)
        ).fetchone()
        assert row["status"] == "active"

    def test_store_chain_topic_from_thesis(self, engine):
        """store_chain uses the thesis (truncated to 200) as topic."""
        long_thesis = "x" * 500
        chain_id = engine.store_chain({"thesis": long_thesis})
        conn = _get_conn()
        row = conn.execute(
            "SELECT topic FROM dialectic_chains WHERE id = ?", (chain_id,)
        ).fetchone()
        assert len(row["topic"]) == 200


class TestGetChains:
    def test_get_chains_empty(self, engine):
        """get_chains returns empty list when no chains exist."""
        assert engine.get_chains() == []

    def test_get_chains_returns_all(self, engine):
        """get_chains returns all chains for the project."""
        engine.store_chain({"thesis": "T1"})
        engine.store_chain({"thesis": "T2"})
        chains = engine.get_chains()
        assert len(chains) == 2

    def test_get_chains_includes_nodes(self, engine):
        """get_chains includes nodes for each chain."""
        engine.store_chain({"thesis": "T", "antithesis": "A"})
        chains = engine.get_chains()
        assert len(chains) == 1
        assert "nodes" in chains[0]
        assert len(chains[0]["nodes"]) == 2

    def test_get_chains_filter_by_topic(self, engine):
        """get_chains filters by topic substring."""
        engine.store_chain({"thesis": "Cancer immunotherapy study"})
        engine.store_chain({"thesis": "Quantum computing advances"})
        chains = engine.get_chains(topic="Cancer")
        assert len(chains) == 1
        assert "Cancer" in chains[0]["topic"]

    def test_get_chains_respects_limit(self, engine):
        """get_chains respects the limit parameter."""
        for i in range(10):
            engine.store_chain({"thesis": f"Chain {i}"})
        chains = engine.get_chains(limit=3)
        assert len(chains) == 3

    def test_get_chains_project_scoped(self, engine):
        """get_chains only returns chains for the engine's project."""
        engine.store_chain({"thesis": "Mine"})
        # Insert a chain for another project directly
        conn = _get_conn()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO dialectic_chains (project, topic, created, updated) VALUES (?, ?, ?, ?)",
            ("other_project", "Theirs", now, now),
        )
        conn.commit()
        chains = engine.get_chains()
        assert all(c["project"] == "test_project" for c in chains)

    def test_get_chains_ordered_by_updated_desc(self, engine):
        """get_chains returns chains ordered by updated timestamp descending."""
        engine.store_chain({"thesis": "First"})
        engine.store_chain({"thesis": "Second"})
        chains = engine.get_chains()
        assert chains[0]["updated"] >= chains[1]["updated"]


class TestGetChain:
    def test_get_chain_exists(self, engine):
        """get_chain returns the chain with nodes when it exists."""
        chain_id = engine.store_chain({"thesis": "T", "antithesis": "A"})
        chain = engine.get_chain(chain_id)
        assert chain is not None
        assert chain["id"] == chain_id
        assert "nodes" in chain
        assert len(chain["nodes"]) == 2

    def test_get_chain_not_found(self, engine):
        """get_chain returns None for nonexistent chain_id."""
        result = engine.get_chain(9999)
        assert result is None

    def test_get_chain_nodes_ordered(self, engine):
        """get_chain returns nodes ordered by id."""
        chain_id = engine.store_chain({
            "thesis": "T",
            "antithesis": "A",
            "synthesis": "S",
        })
        chain = engine.get_chain(chain_id)
        node_ids = [n["id"] for n in chain["nodes"]]
        assert node_ids == sorted(node_ids)


# ── Edge Cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_engine_with_custom_logger(self, mock_ollama):
        """DialecticEngine accepts a custom on_log callback."""
        logs = []
        engine = DialecticEngine(project="test", on_log=logs.append)
        engine.challenge("test finding")
        assert any("challenging" in log for log in logs)

    def test_challenge_truncates_long_topic(self, engine):
        """challenge truncates the topic to 200 chars."""
        long_finding = "x" * 500
        result = engine.challenge(long_finding)
        conn = _get_conn()
        chain = conn.execute(
            "SELECT topic FROM dialectic_chains WHERE id = ?",
            (result["chain_id"],),
        ).fetchone()
        assert len(chain["topic"]) == 200

    def test_prompt_templates_have_placeholders(self):
        """All prompt templates contain their expected placeholders."""
        assert "{finding}" in CHALLENGE_PROMPT
        assert "{thesis}" in SYNTHESIZE_PROMPT
        assert "{antithesis}" in SYNTHESIZE_PROMPT
        assert "{synthesis}" in VERIFY_PROMPT
        assert "{syntheses}" in META_SYNTHESIS_PROMPT

    def test_multiple_chains_independent(self, engine):
        """Multiple challenge calls create independent chains."""
        r1 = engine.challenge("Finding 1")
        r2 = engine.challenge("Finding 2")
        assert r1["chain_id"] != r2["chain_id"]
        chain1 = engine.get_chain(r1["chain_id"])
        chain2 = engine.get_chain(r2["chain_id"])
        assert chain1["topic"] != chain2["topic"]

    def test_engine_default_logger_noop(self, mock_ollama):
        """DialecticEngine with no on_log does not raise."""
        engine = DialecticEngine(project="test")
        # Should not raise even though _log is a noop lambda
        engine.challenge("a finding")

    def test_store_empty_chain(self, engine):
        """store_chain handles a dict with no node content."""
        chain_id = engine.store_chain({})
        assert chain_id > 0
        conn = _get_conn()
        nodes = conn.execute(
            "SELECT * FROM dialectic_nodes WHERE chain_id = ?", (chain_id,)
        ).fetchall()
        assert len(nodes) == 0
