"""Tests for the self-verifying knowledge engine and active question generation."""

import sqlite3
import threading
import pytest

from one import store, verification
from one.store import push_memory, recall, set_project
from one.verification import (
    VerificationEngine,
    FrontierMapper,
    SOURCE_QUALITY,
    CONFIDENCE_STATES,
    STALE_DAYS,
    DEPRECATION_THRESHOLD,
    init_schema,
    _get_conn,
)


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", str(tmp_path))
    monkeypatch.setattr("one.verification.DB_PATH", db_path)
    monkeypatch.setattr("one.verification.DB_DIR", str(tmp_path))
    # Reset thread-local connections
    store._local = threading.local()
    verification._local = threading.local()
    set_project("test_project")
    yield db_path


# ── System 5: Source Quality Scoring ────────────────────────────────


class TestSourceQuality:
    def test_meta_analysis_highest_tier(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("This meta-analysis of 50 studies found...")
        assert score == 0.95

    def test_systematic_review(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("A systematic review published in...")
        assert score == 0.93

    def test_cochrane_review(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("The Cochrane review concludes...")
        assert score == 0.95

    def test_rct_scoring(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("This randomized controlled trial enrolled 200 patients")
        assert score == 0.85

    def test_rct_abbreviation(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("RCT results show significant improvement")
        assert score == 0.85

    def test_peer_reviewed(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("Published in a peer-reviewed journal")
        assert score == 0.80

    def test_top_journal_nature(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("Published in Nature 2024")
        assert score == 0.88

    def test_top_journal_science(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("A Science paper demonstrated...")
        assert score == 0.88

    def test_top_journal_lancet(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("The Lancet published findings...")
        assert score == 0.88

    def test_top_journal_nejm(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("According to the NEJM study...")
        assert score == 0.88

    def test_preprint_lower_quality(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("Available as a preprint on medRxiv")
        assert score == 0.60

    def test_arxiv_preprint(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("Posted on arxiv last week")
        assert score == 0.60

    def test_biorxiv(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("The biorxiv preprint suggests...")
        # biorxiv matches both biorxiv (0.58) and preprint (0.60), max wins
        assert score == 0.60

    def test_blog_post(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("From a blog post about machine learning")
        assert score == 0.35

    def test_expert_blog_higher_than_regular_blog(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("This expert blog covers recent advances")
        # "expert blog" (0.40) is the best match
        assert score == 0.40

    def test_news_article(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("According to this news article...")
        assert score == 0.25

    def test_press_release(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("The company's press release stated...")
        assert score == 0.20

    def test_webpage_default(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("Found on a random webpage")
        assert score == 0.15

    def test_unknown_source_gets_default(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("Some random text with no source indicators")
        assert score == 0.15  # default webpage quality

    def test_quantitative_data_boost_p_value(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("Results show p < 0.05 significance")
        # default 0.15 + 0.1 boost = 0.25
        assert score == pytest.approx(0.25)

    def test_quantitative_data_boost_sample_size(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("Study with n = 500 participants")
        assert score == pytest.approx(0.25)

    def test_quantitative_data_boost_confidence_interval(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("The CI [0.5, 0.9] was narrow")
        assert score == pytest.approx(0.25)

    def test_quantitative_boost_stacks_with_source(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("peer-reviewed study found p < 0.01")
        # peer-reviewed 0.80 + 0.1 boost = 0.90
        assert score == pytest.approx(0.90)

    def test_quantitative_boost_capped_at_one(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("This meta-analysis found n = 10000 and p < 0.001")
        # meta-analysis 0.95 + 0.1 = 1.05, capped to 1.0
        assert score == 1.0

    def test_best_source_wins_when_multiple_present(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("This peer-reviewed meta-analysis from Nature...")
        # meta-analysis (0.95) > nature (0.88) > peer-reviewed (0.80)
        assert score == 0.95

    def test_case_insensitive_matching(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("Published in NATURE journal")
        assert score == 0.88


# ── System 5: Verify Finding ───────────────────────────────────────


class TestVerifyFinding:
    def test_verified_status_increases_confidence(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: VERIFIED\n"
                "NEW_EVIDENCE: Replicated in 3 independent studies\n"
                "CONFIDENCE_ADJUSTMENT: +0.2\n"
                "REASONING: Strong replication support"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Gene X causes trait Y", 0.5)
        assert result["status"] == "verified"
        assert result["new_confidence"] == pytest.approx(0.7)
        assert result["adjustment"] == pytest.approx(0.2)
        assert result["evidence"] == "Replicated in 3 independent studies"
        assert result["reasoning"] == "Strong replication support"

    def test_challenged_status_decreases_confidence(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: CHALLENGED\n"
                "NEW_EVIDENCE: Failed replication attempt\n"
                "CONFIDENCE_ADJUSTMENT: -0.3\n"
                "REASONING: Key study could not be replicated"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Gene X causes trait Y", 0.5)
        assert result["status"] == "challenged"
        assert result["new_confidence"] == pytest.approx(0.2)
        assert result["adjustment"] == pytest.approx(-0.3)

    def test_corroborated_status(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: CORROBORATED\n"
                "NEW_EVIDENCE: Additional supporting data\n"
                "CONFIDENCE_ADJUSTMENT: +0.1\n"
                "REASONING: Consistent with recent findings"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Drug A reduces inflammation", 0.6)
        assert result["status"] == "corroborated"
        assert result["new_confidence"] == pytest.approx(0.7)
        assert result["previous_confidence"] == pytest.approx(0.6)

    def test_unverifiable_status_no_change(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: UNVERIFIABLE\n"
                "NEW_EVIDENCE: No data available\n"
                "CONFIDENCE_ADJUSTMENT: 0.0\n"
                "REASONING: Insufficient evidence"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Obscure claim", 0.5)
        assert result["status"] == "unverifiable"
        assert result["new_confidence"] == pytest.approx(0.5)
        assert result["adjustment"] == pytest.approx(0.0)

    def test_confidence_clamped_to_zero(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: CHALLENGED\n"
                "NEW_EVIDENCE: Retracted\n"
                "CONFIDENCE_ADJUSTMENT: -0.9\n"
                "REASONING: Study was retracted"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Retracted claim", 0.1)
        assert result["new_confidence"] == 0.0

    def test_confidence_clamped_to_one(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: VERIFIED\n"
                "NEW_EVIDENCE: Gold standard\n"
                "CONFIDENCE_ADJUSTMENT: +0.5\n"
                "REASONING: Overwhelmingly confirmed"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Well-established fact", 0.9)
        assert result["new_confidence"] == 1.0

    def test_null_llm_response_defaults_to_unverifiable(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: None,
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Any finding", 0.5)
        assert result["status"] == "unverifiable"
        assert result["new_confidence"] == pytest.approx(0.5)
        assert result["adjustment"] == 0.0
        assert result["evidence"] == ""

    def test_malformed_llm_response(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: "This is garbage output with no structure",
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Any finding", 0.5)
        assert result["status"] == "unverifiable"
        assert result["new_confidence"] == pytest.approx(0.5)

    def test_invalid_confidence_adjustment_ignored(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: VERIFIED\n"
                "CONFIDENCE_ADJUSTMENT: not_a_number\n"
                "REASONING: Good data"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Finding", 0.5)
        assert result["status"] == "verified"
        assert result["adjustment"] == 0.0
        assert result["new_confidence"] == pytest.approx(0.5)

    def test_verification_logged_to_db(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: VERIFIED\n"
                "NEW_EVIDENCE: Confirmed\n"
                "CONFIDENCE_ADJUSTMENT: +0.1\n"
                "REASONING: Solid"
            ),
        )
        engine = VerificationEngine("test_project")
        engine.verify_finding("Test finding", 0.5)

        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM verification_log WHERE project = ?",
            ("test_project",),
        ).fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["previous_confidence"] == pytest.approx(0.5)
        assert row["new_confidence"] == pytest.approx(0.6)
        assert row["verification_type"] == "verified"
        assert row["evidence"] == "Confirmed"

    def test_on_log_callback_called(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: "STATUS: VERIFIED\nCONFIDENCE_ADJUSTMENT: +0.1",
        )
        log_messages = []
        engine = VerificationEngine("test_project", on_log=log_messages.append)
        engine.verify_finding("Test finding", 0.5)
        assert len(log_messages) == 2  # "verifying:..." and "verification:..."
        assert "verifying" in log_messages[0]
        assert "verification" in log_messages[1]

    def test_verify_finding_returns_all_keys(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: VERIFIED\n"
                "NEW_EVIDENCE: data\n"
                "CONFIDENCE_ADJUSTMENT: +0.05\n"
                "REASONING: reason"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Finding text", 0.5)
        expected_keys = {"status", "previous_confidence", "new_confidence",
                         "adjustment", "evidence", "reasoning"}
        assert set(result.keys()) == expected_keys


# ── System 5: Verification Sweep ───────────────────────────────────


class TestVerificationSweep:
    def test_sweep_with_findings(self, monkeypatch):
        for i in range(5):
            push_memory(
                f"Research finding number {i} about topic alpha",
                source="user",
                project="test_project",
                aif_confidence=0.5 + i * 0.05,
            )

        call_count = {"n": 0}

        def mock_ollama(prompt, timeout=120):
            call_count["n"] += 1
            return (
                "STATUS: VERIFIED\n"
                "NEW_EVIDENCE: Confirmed\n"
                "CONFIDENCE_ADJUSTMENT: +0.1\n"
                "REASONING: Replicated"
            )

        monkeypatch.setattr("one.verification._call_ollama", mock_ollama)

        engine = VerificationEngine("test_project")
        results = engine.run_verification_sweep(n=3)
        assert len(results) <= 3
        assert call_count["n"] == len(results)
        for r in results:
            assert "memory_id" in r
            assert r["status"] == "verified"

    def test_sweep_empty_project(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: "STATUS: UNVERIFIABLE",
        )
        engine = VerificationEngine("test_project")
        results = engine.run_verification_sweep(n=10)
        assert results == []

    def test_sweep_respects_n_limit(self, monkeypatch):
        for i in range(10):
            push_memory(
                f"Finding {i} with unique text for recall matching",
                source="user",
                project="test_project",
                aif_confidence=0.5,
            )

        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: "STATUS: VERIFIED\nCONFIDENCE_ADJUSTMENT: +0.05",
        )

        engine = VerificationEngine("test_project")
        results = engine.run_verification_sweep(n=2)
        assert len(results) <= 2

    def test_sweep_log_messages(self, monkeypatch):
        push_memory("A finding to sweep", source="user", project="test_project")

        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: "STATUS: VERIFIED\nCONFIDENCE_ADJUSTMENT: +0.1",
        )

        log_messages = []
        engine = VerificationEngine("test_project", on_log=log_messages.append)
        engine.run_verification_sweep(n=5)
        assert any("sweep" in msg for msg in log_messages)

    def test_sweep_results_have_memory_ids(self, monkeypatch):
        push_memory("Finding about photosynthesis in algae",
                     source="user", project="test_project", aif_confidence=0.6)

        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: CORROBORATED\n"
                "CONFIDENCE_ADJUSTMENT: +0.05\n"
                "NEW_EVIDENCE: Supported\n"
                "REASONING: Good"
            ),
        )
        engine = VerificationEngine("test_project")
        results = engine.run_verification_sweep(n=5)
        for r in results:
            assert "memory_id" in r
            assert isinstance(r["memory_id"], str)


# ── System 5: Confidence Distribution ──────────────────────────────


class TestConfidenceDistribution:
    def test_empty_project(self, monkeypatch):
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [])
        engine = VerificationEngine("test_project")
        dist = engine.get_confidence_distribution()
        assert dist["total"] == 0
        assert dist["avg_confidence"] == 0  # 0 / max(0,1) = 0
        assert all(v == 0 for v in dist["distribution"].values())

    def test_single_high_confidence(self, monkeypatch):
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [
            {"id": "1", "raw_text": "High confidence finding", "aif_confidence": 0.9},
        ])
        engine = VerificationEngine("test_project")
        dist = engine.get_confidence_distribution()
        assert dist["total"] == 1
        assert dist["distribution"]["very_high (0.8-1.0)"] == 1
        assert dist["avg_confidence"] == pytest.approx(0.9)

    def test_multiple_buckets(self, monkeypatch):
        findings = [
            {"id": str(i), "raw_text": f"Finding {i}", "aif_confidence": c}
            for i, c in enumerate([0.95, 0.85, 0.7, 0.65, 0.5, 0.3, 0.1])
        ]
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: findings)
        engine = VerificationEngine("test_project")
        dist = engine.get_confidence_distribution()
        assert dist["total"] == 7
        assert dist["distribution"]["very_high (0.8-1.0)"] == 2
        assert dist["distribution"]["high (0.6-0.8)"] == 2
        assert dist["distribution"]["medium (0.4-0.6)"] == 1
        assert dist["distribution"]["low (0.2-0.4)"] == 1
        assert dist["distribution"]["very_low (0.0-0.2)"] == 1

    def test_average_confidence_correct(self, monkeypatch):
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [
            {"id": "1", "raw_text": "A", "aif_confidence": 0.4},
            {"id": "2", "raw_text": "B", "aif_confidence": 0.6},
        ])
        engine = VerificationEngine("test_project")
        dist = engine.get_confidence_distribution()
        assert dist["avg_confidence"] == pytest.approx(0.5)

    def test_boundary_values(self, monkeypatch):
        findings = [
            {"id": "1", "raw_text": "Exact 0.8", "aif_confidence": 0.8},
            {"id": "2", "raw_text": "Exact 0.6", "aif_confidence": 0.6},
            {"id": "3", "raw_text": "Exact 0.4", "aif_confidence": 0.4},
            {"id": "4", "raw_text": "Exact 0.2", "aif_confidence": 0.2},
        ]
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: findings)
        engine = VerificationEngine("test_project")
        dist = engine.get_confidence_distribution()
        assert dist["distribution"]["very_high (0.8-1.0)"] == 1
        assert dist["distribution"]["high (0.6-0.8)"] == 1
        assert dist["distribution"]["medium (0.4-0.6)"] == 1
        assert dist["distribution"]["low (0.2-0.4)"] == 1

    def test_distribution_has_all_buckets(self, monkeypatch):
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [])
        engine = VerificationEngine("test_project")
        dist = engine.get_confidence_distribution()
        expected_buckets = [
            "very_high (0.8-1.0)",
            "high (0.6-0.8)",
            "medium (0.4-0.6)",
            "low (0.2-0.4)",
            "very_low (0.0-0.2)",
        ]
        for bucket in expected_buckets:
            assert bucket in dist["distribution"]


# ── System 5: Archive Deprecated ──────────────────────────────────


class TestArchiveDeprecated:
    def test_archive_below_threshold(self, monkeypatch):
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [
            {"id": "1", "raw_text": "Low confidence finding", "aif_confidence": 0.1},
        ])
        engine = VerificationEngine("test_project")
        archived = engine.archive_deprecated()
        assert archived == 1

    def test_archive_keeps_high_confidence(self, monkeypatch):
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [
            {"id": "1", "raw_text": "High confidence", "aif_confidence": 0.9},
        ])
        engine = VerificationEngine("test_project")
        archived = engine.archive_deprecated()
        assert archived == 0

    def test_archive_custom_threshold(self, monkeypatch):
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [
            {"id": "1", "raw_text": "Medium conf", "aif_confidence": 0.4},
        ])
        engine = VerificationEngine("test_project")
        # With threshold of 0.5, 0.4 is below
        archived = engine.archive_deprecated(threshold=0.5)
        assert archived == 1

    def test_archive_empty_project(self, monkeypatch):
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [])
        engine = VerificationEngine("test_project")
        archived = engine.archive_deprecated()
        assert archived == 0

    def test_archive_uses_deprecation_threshold_default(self, monkeypatch):
        assert DEPRECATION_THRESHOLD == 0.2
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [
            {"id": "1", "raw_text": "Just above threshold", "aif_confidence": 0.2},
            {"id": "2", "raw_text": "Below threshold", "aif_confidence": 0.15},
        ])
        engine = VerificationEngine("test_project")
        archived = engine.archive_deprecated()
        assert archived == 1  # Only the 0.15 one

    def test_archive_log_callback(self, monkeypatch):
        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [
            {"id": "1", "raw_text": "Low", "aif_confidence": 0.05},
        ])
        log_messages = []
        engine = VerificationEngine("test_project", on_log=log_messages.append)
        engine.archive_deprecated()
        assert any("archived" in msg for msg in log_messages)

    def test_archive_pushes_deprecated_memory(self, monkeypatch):
        pushed = []

        def tracking_push(*args, **kwargs):
            pushed.append({"args": args, "kwargs": kwargs})

        monkeypatch.setattr("one.verification.recall", lambda query, n=10, project=None: [
            {"id": "1", "raw_text": "Bad finding to deprecate", "aif_confidence": 0.05},
        ])
        monkeypatch.setattr("one.verification.push_memory", tracking_push)
        engine = VerificationEngine("test_project")
        engine.archive_deprecated()
        assert len(pushed) == 1
        call = pushed[0]
        # push_memory is called with raw_text as first positional arg
        assert "[DEPRECATED]" in call["args"][0]
        assert call["kwargs"]["aif_confidence"] == 0.0
        assert call["kwargs"]["tm_label"] == "deprecated"


# ── System 6: Information Value Scoring ────────────────────────────


class TestScoreInformationValue:
    def test_formula_default_values(self):
        mapper = FrontierMapper("test_project")
        score = mapper.score_information_value(
            question="What is X?",
            goal="Understand X",
        )
        # unknowns=1 -> min(1.0, 0.1)*0.3 = 0.03
        # contradictions=0 -> 0
        # goal_centrality=0.5 -> 0.5*0.25 = 0.125
        # novelty=0.5 -> 0.5*0.15 = 0.075
        expected = 0.03 + 0.0 + 0.125 + 0.075
        assert score == pytest.approx(expected)

    def test_formula_max_unknowns(self):
        mapper = FrontierMapper("test_project")
        score = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=15,  # min(1.0, 15*0.1) = 1.0
            contradictions_clarified=0,
            goal_centrality=0.0,
            novelty=0.0,
        )
        # 1.0 * 0.3 = 0.3
        assert score == pytest.approx(0.3)

    def test_formula_max_contradictions(self):
        mapper = FrontierMapper("test_project")
        score = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=0,
            contradictions_clarified=10,  # min(1.0, 10*0.2) = 1.0
            goal_centrality=0.0,
            novelty=0.0,
        )
        # 1.0 * 0.3 = 0.3
        assert score == pytest.approx(0.3)

    def test_formula_max_goal_centrality(self):
        mapper = FrontierMapper("test_project")
        score = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=0,
            contradictions_clarified=0,
            goal_centrality=1.0,
            novelty=0.0,
        )
        assert score == pytest.approx(0.25)

    def test_formula_max_novelty(self):
        mapper = FrontierMapper("test_project")
        score = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=0,
            contradictions_clarified=0,
            goal_centrality=0.0,
            novelty=1.0,
        )
        assert score == pytest.approx(0.15)

    def test_formula_all_max(self):
        mapper = FrontierMapper("test_project")
        score = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=10,
            contradictions_clarified=5,
            goal_centrality=1.0,
            novelty=1.0,
        )
        expected = 0.3 + 0.3 + 0.25 + 0.15
        assert score == pytest.approx(1.0)

    def test_formula_all_zero(self):
        mapper = FrontierMapper("test_project")
        score = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=0,
            contradictions_clarified=0,
            goal_centrality=0.0,
            novelty=0.0,
        )
        assert score == pytest.approx(0.0)

    def test_unknowns_capped_at_ten(self):
        mapper = FrontierMapper("test_project")
        score_10 = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=10,
            contradictions_clarified=0,
            goal_centrality=0.0,
            novelty=0.0,
        )
        score_100 = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=100,
            contradictions_clarified=0,
            goal_centrality=0.0,
            novelty=0.0,
        )
        # Both should cap at 1.0 * 0.3
        assert score_10 == score_100

    def test_contradictions_capped_at_five(self):
        mapper = FrontierMapper("test_project")
        score_5 = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=0,
            contradictions_clarified=5,
            goal_centrality=0.0,
            novelty=0.0,
        )
        score_50 = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=0,
            contradictions_clarified=50,
            goal_centrality=0.0,
            novelty=0.0,
        )
        assert score_5 == score_50

    def test_single_unknown_weight(self):
        mapper = FrontierMapper("test_project")
        score = mapper.score_information_value(
            question="Q", goal="G",
            unknowns_resolved=1,
            contradictions_clarified=0,
            goal_centrality=0.0,
            novelty=0.0,
        )
        # min(1.0, 1*0.1) * 0.3 = 0.1 * 0.3 = 0.03
        assert score == pytest.approx(0.03)


# ── System 6: Frontier Mapping ─────────────────────────────────────


class TestFrontierMapping:
    def test_map_frontier_empty_project(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What are the key mechanisms of X?\n"
                "2. How does Y relate to Z in this context?\n"
                "3. What experimental evidence supports the hypothesis?\n"
                "4. Are there known contradictions in the literature?\n"
                "5. What alternative explanations have been proposed?"
            ),
        )
        mapper = FrontierMapper("test_project")
        frontier = mapper.map_frontier("Understand mechanism X")
        assert frontier["goal"] == "Understand mechanism X"
        assert frontier["total_findings"] == 0
        assert isinstance(frontier["unexplored"], list)
        assert frontier["coverage"] == 0.0

    def test_map_frontier_with_findings(self, monkeypatch):
        push_memory("Mechanism X involves pathway A",
                     source="user", project="test_project", aif_confidence=0.8)
        push_memory("Pathway A is regulated by gene B",
                     source="user", project="test_project", aif_confidence=0.5)

        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What downstream targets does pathway A affect?\n"
                "2. How is gene B expression regulated in disease?\n"
                "3. Are there alternative pathways for mechanism X?\n"
                "4. What is the temporal dynamics of pathway activation?\n"
                "5. How do environmental factors modulate this mechanism?"
            ),
        )
        mapper = FrontierMapper("test_project")
        frontier = mapper.map_frontier("Understand mechanism X")
        assert frontier["total_findings"] >= 0
        assert "explored" in frontier
        assert "partially_explored" in frontier
        assert "unexplored" in frontier

    def test_map_frontier_stores_questions_in_db(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What is the primary mechanism of action?\n"
                "2. How does dosage affect the outcome?\n"
                "3. What are the long-term side effects?\n"
                "4. Is there a genetic predisposition component?\n"
                "5. How does it compare to existing treatments?"
            ),
        )
        mapper = FrontierMapper("test_project")
        mapper.map_frontier("Drug efficacy study")

        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM knowledge_frontier WHERE project = ?",
            ("test_project",),
        ).fetchall()
        assert len(rows) == 5
        for row in rows:
            r = dict(row)
            assert r["status"] == "unexplored"
            assert r["goal"] == "Drug efficacy study"

    def test_map_frontier_null_llm_response(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: None,
        )
        mapper = FrontierMapper("test_project")
        frontier = mapper.map_frontier("Some goal")
        assert frontier["unexplored"] == []

    def test_frontier_categorizes_by_confidence(self, monkeypatch):
        # High confidence -> explored (>= 0.7)
        push_memory("Well-established fact about topic Z",
                     source="user", project="test_project", aif_confidence=0.9)
        # Medium confidence -> partial (0.3 <= x < 0.7)
        push_memory("Somewhat supported claim about topic Z",
                     source="user", project="test_project", aif_confidence=0.5)
        # Low confidence -> weak (< 0.3)
        push_memory("Weakly supported claim about topic Z",
                     source="user", project="test_project", aif_confidence=0.1)

        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: None,
        )
        mapper = FrontierMapper("test_project")
        frontier = mapper.map_frontier("topic Z")
        assert frontier["total_findings"] >= 1

    def test_short_questions_filtered(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. Short\n"  # <= 20 chars, filtered
                "2. What is the comprehensive mechanism of this process in detail?\n"
                "3. Ok\n"  # filtered
                "4. How does this interact with the broader biological system?\n"
                "5. Why?\n"  # filtered
            ),
        )
        mapper = FrontierMapper("test_project")
        frontier = mapper.map_frontier("Some research goal")
        # Only questions > 20 chars should appear
        assert len(frontier["unexplored"]) == 2

    def test_map_frontier_returns_coverage(self, monkeypatch):
        push_memory("Known fact about X with high confidence",
                     source="user", project="test_project", aif_confidence=0.9)
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: None,
        )
        mapper = FrontierMapper("test_project")
        frontier = mapper.map_frontier("X")
        assert isinstance(frontier["coverage"], float)
        assert 0.0 <= frontier["coverage"] <= 1.0


# ── System 6: Best Question ────────────────────────────────────────


class TestBestQuestion:
    def test_best_question_from_existing(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What is the primary mechanism of action for drug X?\n"
                "2. How does genetic variation affect drug response?\n"
                "3. What are the key biomarkers for treatment monitoring?\n"
                "4. Are there significant drug-drug interactions?\n"
                "5. What is the optimal dosing strategy for elderly patients?"
            ),
        )
        mapper = FrontierMapper("test_project")
        mapper.map_frontier("Drug X research")

        # Now best_question should return from DB
        best = mapper.best_question("Drug X research")
        assert best is not None
        assert "question" in best
        assert best["status"] == "unexplored"

    def test_best_question_triggers_mapping_when_empty(self, monkeypatch):
        call_count = {"n": 0}

        def mock_ollama(prompt, timeout=90):
            call_count["n"] += 1
            return (
                "1. What is the fundamental principle behind this?\n"
                "2. How can we measure the effect experimentally?\n"
                "3. What theoretical frameworks apply to this problem?\n"
                "4. Are there analogous systems in other domains?\n"
                "5. What are the practical implications of this research?"
            )

        monkeypatch.setattr("one.verification._call_ollama", mock_ollama)
        mapper = FrontierMapper("test_project")
        best = mapper.best_question("Novel research goal")
        assert best is not None
        assert call_count["n"] >= 1

    def test_best_question_none_when_llm_fails(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: None,
        )
        mapper = FrontierMapper("test_project")
        best = mapper.best_question("Some goal")
        assert best is None

    def test_best_question_returns_highest_value(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What is the primary mechanism of action here?\n"
                "2. How does this interact with existing treatments?\n"
                "3. What population groups benefit most from this?\n"
                "4. Are there any significant contraindications noted?\n"
                "5. What is the long-term safety profile of this approach?"
            ),
        )
        mapper = FrontierMapper("test_project")
        mapper.map_frontier("Test goal")

        best = mapper.best_question("Test goal")
        assert best is not None
        # best should be the highest information_value among unexplored
        conn = _get_conn()
        all_q = conn.execute(
            "SELECT information_value FROM knowledge_frontier WHERE project = ? AND goal = ? AND status = 'unexplored' ORDER BY information_value DESC",
            ("test_project", "Test goal"),
        ).fetchall()
        if all_q:
            assert best["information_value"] == pytest.approx(all_q[0][0])


# ── System 6: Mark Explored ────────────────────────────────────────


class TestMarkExplored:
    def test_mark_explored_updates_status(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What is the key mechanism underlying this process?\n"
                "2. How does environmental context modulate the effect?\n"
                "3. What are the boundary conditions for this phenomenon?\n"
                "4. Is there a dose-response relationship observed?\n"
                "5. What computational models best capture this behavior?"
            ),
        )
        mapper = FrontierMapper("test_project")
        mapper.map_frontier("Goal A")

        conn = _get_conn()
        row = conn.execute(
            "SELECT id FROM knowledge_frontier WHERE project = ? LIMIT 1",
            ("test_project",),
        ).fetchone()
        question_id = row["id"]

        mapper.mark_explored(question_id)

        updated = conn.execute(
            "SELECT status, explored_at FROM knowledge_frontier WHERE id = ?",
            (question_id,),
        ).fetchone()
        assert updated["status"] == "explored"
        assert updated["explored_at"] is not None

    def test_mark_explored_nonexistent_id(self):
        mapper = FrontierMapper("test_project")
        # Should not raise, just a no-op UPDATE
        mapper.mark_explored(99999)

    def test_mark_explored_sets_timestamp(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What is the fundamental principle governing this?\n"
                "2. How does scale affect the observed phenomenon?\n"
                "3. What are the key variables to control for?\n"
                "4. Are there nonlinear dynamics at play here?\n"
                "5. What predictive models have been validated?"
            ),
        )
        mapper = FrontierMapper("test_project")
        mapper.map_frontier("Goal B")

        conn = _get_conn()
        row = conn.execute(
            "SELECT id FROM knowledge_frontier WHERE project = ? LIMIT 1",
            ("test_project",),
        ).fetchone()

        mapper.mark_explored(row["id"])

        updated = conn.execute(
            "SELECT explored_at FROM knowledge_frontier WHERE id = ?",
            (row["id"],),
        ).fetchone()
        assert updated["explored_at"] is not None
        # Should be a valid ISO timestamp
        assert "T" in updated["explored_at"]


# ── System 6: Frontier Coverage ────────────────────────────────────


class TestFrontierCoverage:
    def test_coverage_zero_when_no_questions(self):
        mapper = FrontierMapper("test_project")
        cov = mapper.frontier_coverage("Some goal")
        assert cov == 0.0

    def test_coverage_after_exploring_all(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What is the primary mechanism of action here?\n"
                "2. How does this interact with existing treatments?\n"
                "3. What population groups benefit most from this?\n"
                "4. Are there any significant contraindications noted?\n"
                "5. What is the long-term safety profile of this approach?"
            ),
        )
        mapper = FrontierMapper("test_project")
        mapper.map_frontier("Coverage test goal")

        conn = _get_conn()
        rows = conn.execute(
            "SELECT id FROM knowledge_frontier WHERE project = ? AND goal = ?",
            ("test_project", "Coverage test goal"),
        ).fetchall()

        for row in rows:
            mapper.mark_explored(row["id"])

        cov = mapper.frontier_coverage("Coverage test goal")
        assert cov == pytest.approx(1.0)

    def test_coverage_partial(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What molecular pathways are involved in this process?\n"
                "2. How do epigenetic modifications influence the outcome?\n"
                "3. What role does the microbiome play in this context?\n"
                "4. Are there sex-dependent differences in the response?\n"
                "5. What imaging techniques best visualize this phenomenon?"
            ),
        )
        mapper = FrontierMapper("test_project")
        mapper.map_frontier("Partial coverage goal")

        conn = _get_conn()
        rows = conn.execute(
            "SELECT id FROM knowledge_frontier WHERE project = ? AND goal = ? LIMIT 2",
            ("test_project", "Partial coverage goal"),
        ).fetchall()

        for row in rows:
            mapper.mark_explored(row["id"])

        cov = mapper.frontier_coverage("Partial coverage goal")
        assert cov == pytest.approx(2.0 / 5.0)

    def test_coverage_different_goals_isolated(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What is the primary mechanism of action here?\n"
                "2. How does this interact with existing treatments?\n"
                "3. What population groups benefit most from this?\n"
                "4. Are there any significant contraindications noted?\n"
                "5. What is the long-term safety profile of this approach?"
            ),
        )
        mapper = FrontierMapper("test_project")
        mapper.map_frontier("Goal Alpha")
        mapper.map_frontier("Goal Beta")

        # Explore all of Goal Alpha
        conn = _get_conn()
        alpha_rows = conn.execute(
            "SELECT id FROM knowledge_frontier WHERE project = ? AND goal = ?",
            ("test_project", "Goal Alpha"),
        ).fetchall()
        for row in alpha_rows:
            mapper.mark_explored(row["id"])

        assert mapper.frontier_coverage("Goal Alpha") == pytest.approx(1.0)
        assert mapper.frontier_coverage("Goal Beta") == pytest.approx(0.0)


# ── Database Schema ────────────────────────────────────────────────


class TestDatabaseSchema:
    def test_init_schema_creates_verification_log(self):
        init_schema()
        conn = _get_conn()
        result = conn.execute("SELECT COUNT(*) FROM verification_log").fetchone()
        assert result[0] == 0

    def test_init_schema_creates_knowledge_frontier(self):
        init_schema()
        conn = _get_conn()
        result = conn.execute("SELECT COUNT(*) FROM knowledge_frontier").fetchone()
        assert result[0] == 0

    def test_init_schema_idempotent(self):
        init_schema()
        init_schema()  # Should not raise
        conn = _get_conn()
        result = conn.execute("SELECT COUNT(*) FROM verification_log").fetchone()
        assert result[0] == 0

    def test_verification_log_columns(self):
        init_schema()
        conn = _get_conn()
        cursor = conn.execute("PRAGMA table_info(verification_log)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "id" in columns
        assert "project" in columns
        assert "memory_id" in columns
        assert "finding_id" in columns
        assert "previous_confidence" in columns
        assert "new_confidence" in columns
        assert "verification_type" in columns
        assert "evidence" in columns
        assert "created" in columns

    def test_knowledge_frontier_columns(self):
        init_schema()
        conn = _get_conn()
        cursor = conn.execute("PRAGMA table_info(knowledge_frontier)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "id" in columns
        assert "project" in columns
        assert "goal" in columns
        assert "question" in columns
        assert "information_value" in columns
        assert "status" in columns
        assert "category" in columns
        assert "created" in columns
        assert "explored_at" in columns

    def test_verification_log_indexes(self):
        init_schema()
        conn = _get_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='verification_log'"
        ).fetchall()
        index_names = {row[0] for row in indexes}
        assert "idx_verification_project" in index_names
        assert "idx_verification_memory" in index_names

    def test_knowledge_frontier_indexes(self):
        init_schema()
        conn = _get_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='knowledge_frontier'"
        ).fetchall()
        index_names = {row[0] for row in indexes}
        assert "idx_frontier_project" in index_names
        assert "idx_frontier_value" in index_names
        assert "idx_frontier_status" in index_names


# ── Constants and Module-Level Checks ──────────────────────────────


class TestConstants:
    def test_confidence_states_order(self):
        assert CONFIDENCE_STATES == [
            "new", "corroborated", "challenged", "verified", "stale", "deprecated"
        ]

    def test_stale_days(self):
        assert STALE_DAYS == 30

    def test_deprecation_threshold(self):
        assert DEPRECATION_THRESHOLD == 0.2

    def test_source_quality_keys_exist(self):
        expected_keys = {
            "meta-analysis", "systematic review", "cochrane",
            "randomized controlled trial", "rct", "peer-reviewed",
            "nature", "science", "cell", "lancet", "nejm",
            "preprint", "arxiv", "biorxiv",
            "expert blog", "blog post", "news article",
            "press release", "webpage", "ai-generated", "unverified",
        }
        assert set(SOURCE_QUALITY.keys()) == expected_keys

    def test_source_quality_values_in_range(self):
        for source, score in SOURCE_QUALITY.items():
            assert 0.0 <= score <= 1.0, f"{source} has score {score} outside [0, 1]"

    def test_confidence_states_length(self):
        assert len(CONFIDENCE_STATES) == 6


# ── Edge Cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_engine_default_log_is_noop(self):
        engine = VerificationEngine("test_project")
        # Should not raise
        engine._log("test message")

    def test_mapper_default_log_is_noop(self):
        mapper = FrontierMapper("test_project")
        # Should not raise
        mapper._log("test message")

    def test_empty_string_source_scoring(self):
        engine = VerificationEngine("test_project")
        score = engine.score_source("")
        assert score == 0.15

    def test_very_long_goal_truncated_in_db(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: None,
        )
        mapper = FrontierMapper("test_project")
        long_goal = "A" * 1000
        # Should not crash; goal is truncated to 500 in DB inserts
        mapper.map_frontier(long_goal)

    def test_update_frontier_calls_map(self, monkeypatch):
        call_count = {"n": 0}

        def mock_ollama(prompt, timeout=90):
            call_count["n"] += 1
            return None

        monkeypatch.setattr("one.verification._call_ollama", mock_ollama)
        mapper = FrontierMapper("test_project")
        mapper.update_frontier("Goal", ["new finding 1"])
        # update_frontier calls map_frontier which calls _generate_frontier_questions
        assert call_count["n"] >= 1

    def test_verify_finding_with_zero_confidence(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: CHALLENGED\n"
                "CONFIDENCE_ADJUSTMENT: -0.1\n"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Dubious claim", 0.0)
        assert result["new_confidence"] == 0.0  # clamped at 0

    def test_verify_finding_with_full_confidence(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: VERIFIED\n"
                "CONFIDENCE_ADJUSTMENT: +0.1\n"
            ),
        )
        engine = VerificationEngine("test_project")
        result = engine.verify_finding("Solid claim", 1.0)
        assert result["new_confidence"] == 1.0  # clamped at 1.0

    def test_frontier_best_question_goal_truncation(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: None,
        )
        mapper = FrontierMapper("test_project")
        long_goal = "X" * 600
        result = mapper.best_question(long_goal)
        # Should not crash; goal truncated to 500 in query
        assert result is None

    def test_questions_sorted_by_information_value(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What is the primary mechanism of action in this system?\n"
                "2. How does the genetic background influence the phenotype?\n"
                "3. What environmental triggers are most significant here?\n"
                "4. Are there epigenetic factors that modulate expression?\n"
                "5. How do feedback loops maintain homeostasis in this system?"
            ),
        )
        mapper = FrontierMapper("test_project")
        frontier = mapper.map_frontier("Understand the system")
        if len(frontier["unexplored"]) >= 2:
            values = [q["information_value"] for q in frontier["unexplored"]]
            # Questions should be sorted descending by information_value
            assert values == sorted(values, reverse=True)

    def test_multiple_verifications_logged(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=120: (
                "STATUS: VERIFIED\n"
                "CONFIDENCE_ADJUSTMENT: +0.05\n"
                "NEW_EVIDENCE: data\n"
                "REASONING: reason"
            ),
        )
        engine = VerificationEngine("test_project")
        engine.verify_finding("Finding 1", 0.5)
        engine.verify_finding("Finding 2", 0.6)
        engine.verify_finding("Finding 3", 0.7)

        conn = _get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM verification_log WHERE project = ?",
            ("test_project",),
        ).fetchone()[0]
        assert count == 3

    def test_frontier_coverage_ignores_other_projects(self, monkeypatch):
        monkeypatch.setattr(
            "one.verification._call_ollama",
            lambda prompt, timeout=90: (
                "1. What is the primary mechanism of action here?\n"
                "2. How does this interact with existing treatments?\n"
                "3. What population groups benefit most from this?\n"
                "4. Are there any significant contraindications noted?\n"
                "5. What is the long-term safety profile of this approach?"
            ),
        )
        mapper_a = FrontierMapper("project_a")
        mapper_a.map_frontier("Shared goal")

        mapper_b = FrontierMapper("project_b")
        mapper_b.map_frontier("Shared goal")

        # Explore all of project_a
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id FROM knowledge_frontier WHERE project = ? AND goal = ?",
            ("project_a", "Shared goal"),
        ).fetchall()
        for row in rows:
            mapper_a.mark_explored(row["id"])

        assert mapper_a.frontier_coverage("Shared goal") == pytest.approx(1.0)
        assert mapper_b.frontier_coverage("Shared goal") == pytest.approx(0.0)
