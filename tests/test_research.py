"""Tests for the deep research module."""

import sqlite3
import threading
import pytest

import one.research as research_mod
import one.store as store_mod
from one.research import (
    _init_research_schema,
    _score_source_quality,
    _extract_quantitative_data,
    _extract_temporal_info,
    _split_findings,
    _store_finding,
    _store_citation,
    _store_gap,
    _resolve_gap,
    _get_open_gaps,
    _get_findings,
    _already_researched,
    research_status,
    research_frontier,
    DEPTH_0_PROMPTS,
    DEPTH_1_PROMPTS,
    DEPTH_2_PROMPTS,
    DEPTH_3_PROMPTS,
    ALL_DEPTH_PROMPTS,
    GAP_IDENTIFICATION_PROMPT,
    FOLLOWUP_PROMPT,
    ADVERSARIAL_PROMPT,
    SOURCE_QUALITY_KEYWORDS,
)
from one.store import set_project


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Redirect all DB access to a temporary path for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("one.research.DB_PATH", db_path)
    monkeypatch.setattr("one.research.DB_DIR", str(tmp_path))
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", str(tmp_path))
    # Reset thread-local connections so the new path is picked up
    research_mod._local = threading.local()
    store_mod._local = threading.local()
    set_project("test_project")
    yield db_path


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------


class TestInitResearchSchema:
    def test_creates_all_tables(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "schema_test.db"))
        _init_research_schema(conn)

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "research_topics" in tables
        assert "research_findings" in tables
        assert "research_citations" in tables
        assert "research_gaps" in tables

    def test_creates_indexes(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "idx_test.db"))
        _init_research_schema(conn)

        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_research_topics_project" in indexes
        assert "idx_research_findings_topic" in indexes
        assert "idx_research_gaps_topic" in indexes
        assert "idx_research_gaps_status" in indexes
        assert "idx_research_gaps_priority" in indexes

    def test_idempotent(self, tmp_path):
        """Calling _init_research_schema twice must not raise."""
        conn = sqlite3.connect(str(tmp_path / "idempotent_test.db"))
        _init_research_schema(conn)
        _init_research_schema(conn)  # should not raise

    def test_topics_default_columns(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "defaults_test.db"))
        conn.row_factory = sqlite3.Row
        _init_research_schema(conn)

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO research_topics (project, topic, created, updated) VALUES (?, ?, ?, ?)",
            ("proj", "test topic", now, now),
        )
        conn.commit()

        row = dict(conn.execute("SELECT * FROM research_topics").fetchone())
        assert row["status"] == "active"
        assert row["turns_used"] == 0
        assert row["turn_budget"] == 10
        assert row["findings_count"] == 0
        assert row["gaps_count"] == 0
        assert row["synthesis_depth"] == 0
        assert row["depth_level"] == 0


# ---------------------------------------------------------------------------
# Topic insertion via public helper (uses _get_conn → real temp DB)
# ---------------------------------------------------------------------------


class TestTopicCreation:
    def _insert_topic(self, project="proj", topic="test topic", turn_budget=5):
        from datetime import datetime, timezone
        from one.research import _get_conn
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """INSERT INTO research_topics
               (project, topic, status, turn_budget, depth_level, created, updated)
               VALUES (?, ?, 'active', ?, 0, ?, ?)""",
            (project, topic, turn_budget, now, now),
        )
        conn.commit()
        return cur.lastrowid

    def test_insert_and_query(self):
        tid = self._insert_topic()
        assert isinstance(tid, int)
        assert tid > 0

    def test_project_scoping(self):
        self._insert_topic(project="a", topic="topic A")
        self._insert_topic(project="b", topic="topic B")
        status_a = research_status("a")
        status_b = research_status("b")
        assert len(status_a) == 1
        assert len(status_b) == 1
        assert status_a[0]["topic"] == "topic A"
        assert status_b[0]["topic"] == "topic B"

    def test_research_status_empty(self):
        results = research_status("no_such_project")
        assert results == []

    def test_research_status_fields(self):
        self._insert_topic(project="proj", topic="my topic", turn_budget=7)
        rows = research_status("proj")
        assert len(rows) == 1
        r = rows[0]
        for field in ("id", "topic", "status", "turns_used", "turn_budget",
                      "findings_count", "gaps_count", "depth_level"):
            assert field in r
        assert r["topic"] == "my topic"
        assert r["turn_budget"] == 7


# ---------------------------------------------------------------------------
# Findings CRUD
# ---------------------------------------------------------------------------


class TestFindings:
    def _make_topic(self, project="proj", topic="test"):
        from datetime import datetime, timezone
        from one.research import _get_conn
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO research_topics (project, topic, status, turn_budget, depth_level, created, updated) VALUES (?, ?, 'active', 10, 0, ?, ?)",
            (project, topic, now, now),
        )
        conn.commit()
        return cur.lastrowid

    def test_store_finding_returns_id(self):
        tid = self._make_topic()
        fid = _store_finding(topic_id=tid, content="a finding about X")
        assert isinstance(fid, int)
        assert fid > 0

    def test_store_finding_persists_all_fields(self):
        tid = self._make_topic()
        fid = _store_finding(
            topic_id=tid,
            content="specific finding",
            finding_type="adversarial",
            source_query="what is X?",
            confidence=0.8,
            source_quality=0.9,
            quantitative_data="50%",
            published_date="2023",
            depth_level=2,
        )
        findings = _get_findings(tid)
        assert len(findings) == 1
        f = findings[0]
        assert f["id"] == fid
        assert f["content"] == "specific finding"
        assert f["finding_type"] == "adversarial"
        assert f["source_query"] == "what is X?"
        assert abs(f["confidence"] - 0.8) < 1e-6
        assert abs(f["source_quality"] - 0.9) < 1e-6
        assert f["quantitative_data"] == "50%"
        assert f["published_date"] == "2023"
        assert f["depth_level"] == 2

    def test_get_findings_empty(self):
        tid = self._make_topic()
        assert _get_findings(tid) == []

    def test_get_findings_filter_by_depth(self):
        tid = self._make_topic()
        _store_finding(tid, "depth 0 finding", depth_level=0)
        _store_finding(tid, "depth 1 finding", depth_level=1)
        _store_finding(tid, "another depth 0", depth_level=0)

        d0 = _get_findings(tid, depth_level=0)
        d1 = _get_findings(tid, depth_level=1)
        assert len(d0) == 2
        assert len(d1) == 1
        assert all(f["depth_level"] == 0 for f in d0)
        assert all(f["depth_level"] == 1 for f in d1)

    def test_get_findings_ordered_by_confidence(self):
        tid = self._make_topic()
        _store_finding(tid, "low confidence", confidence=0.2)
        _store_finding(tid, "high confidence", confidence=0.9)
        _store_finding(tid, "medium confidence", confidence=0.5)

        findings = _get_findings(tid)
        confidences = [f["confidence"] for f in findings]
        assert confidences == sorted(confidences, reverse=True)

    def test_get_findings_respects_limit(self):
        tid = self._make_topic()
        for i in range(10):
            _store_finding(tid, f"finding {i}")
        assert len(_get_findings(tid, limit=3)) == 3

    def test_already_researched_false_initially(self):
        tid = self._make_topic()
        assert _already_researched(tid, "some query") is False

    def test_already_researched_true_after_store(self):
        tid = self._make_topic()
        _store_finding(tid, "content", source_query="the exact query")
        assert _already_researched(tid, "the exact query") is True

    def test_already_researched_different_query(self):
        tid = self._make_topic()
        _store_finding(tid, "content", source_query="query A")
        assert _already_researched(tid, "query B") is False

    def test_contradicts_field_stored(self):
        tid = self._make_topic()
        fid1 = _store_finding(tid, "original finding")
        fid2 = _store_finding(tid, "contradicting finding", contradicts_finding_id=fid1)
        findings = _get_findings(tid)
        contra = next(f for f in findings if f["id"] == fid2)
        assert contra["contradicts_finding_id"] == fid1


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


class TestCitations:
    def _make_topic_with_findings(self):
        from datetime import datetime, timezone
        from one.research import _get_conn
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO research_topics (project, topic, status, turn_budget, depth_level, created, updated) VALUES ('p', 't', 'active', 10, 0, ?, ?)",
            (now, now),
        )
        conn.commit()
        tid = cur.lastrowid
        fid1 = _store_finding(tid, "finding one")
        fid2 = _store_finding(tid, "finding two")
        return tid, fid1, fid2

    def test_store_citation_supports(self):
        tid, fid1, fid2 = self._make_topic_with_findings()
        _store_citation(fid1, fid2, "supports")
        from one.research import _get_conn
        rows = _get_conn().execute(
            "SELECT * FROM research_citations WHERE finding_a = ? AND finding_b = ?",
            (fid1, fid2),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["relation"] == "supports"

    def test_store_citation_contradicts(self):
        tid, fid1, fid2 = self._make_topic_with_findings()
        _store_citation(fid1, fid2, "contradicts")
        from one.research import _get_conn
        rows = _get_conn().execute(
            "SELECT * FROM research_citations WHERE relation = 'contradicts'",
        ).fetchall()
        assert len(rows) == 1

    def test_store_citation_default_relation(self):
        tid, fid1, fid2 = self._make_topic_with_findings()
        _store_citation(fid1, fid2)
        from one.research import _get_conn
        row = _get_conn().execute(
            "SELECT relation FROM research_citations WHERE finding_a = ?", (fid1,)
        ).fetchone()
        assert row["relation"] == "supports"


# ---------------------------------------------------------------------------
# Gaps CRUD
# ---------------------------------------------------------------------------


class TestGaps:
    def _make_topic(self):
        from datetime import datetime, timezone
        from one.research import _get_conn
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO research_topics (project, topic, status, turn_budget, depth_level, created, updated) VALUES ('proj', 'topic', 'active', 10, 0, ?, ?)",
            (now, now),
        )
        conn.commit()
        return cur.lastrowid

    def test_store_gap_returns_id(self):
        tid = self._make_topic()
        gid = _store_gap(tid, "What causes X?")
        assert isinstance(gid, int)
        assert gid > 0

    def test_get_open_gaps_empty(self):
        tid = self._make_topic()
        assert _get_open_gaps(tid) == []

    def test_get_open_gaps_after_store(self):
        tid = self._make_topic()
        _store_gap(tid, "Why does X happen?", priority=0.8)
        _store_gap(tid, "What is the mechanism?", priority=0.3)
        gaps = _get_open_gaps(tid)
        assert len(gaps) == 2
        assert gaps[0]["priority"] >= gaps[1]["priority"]  # ordered by priority DESC

    def test_get_open_gaps_status_filter(self):
        tid = self._make_topic()
        _store_gap(tid, "Open question?")
        fid = _store_finding(tid, "resolved finding")
        gid2 = _store_gap(tid, "Another question?")
        _resolve_gap(gid2, fid)

        open_gaps = _get_open_gaps(tid)
        assert len(open_gaps) == 1
        assert open_gaps[0]["question"] == "Open question?"

    def test_resolve_gap(self):
        tid = self._make_topic()
        gid = _store_gap(tid, "What is X?")
        fid = _store_finding(tid, "X is Y")
        _resolve_gap(gid, fid)

        from one.research import _get_conn
        row = dict(
            _get_conn().execute(
                "SELECT status, resolved_by FROM research_gaps WHERE id = ?", (gid,)
            ).fetchone()
        )
        assert row["status"] == "resolved"
        assert row["resolved_by"] == fid

    def test_gap_priority_stored(self):
        tid = self._make_topic()
        gid = _store_gap(tid, "High priority gap?", priority=0.95)
        from one.research import _get_conn
        row = _get_conn().execute(
            "SELECT priority FROM research_gaps WHERE id = ?", (gid,)
        ).fetchone()
        assert abs(row["priority"] - 0.95) < 1e-6

    def test_gap_default_status_is_open(self):
        tid = self._make_topic()
        gid = _store_gap(tid, "Gap question?")
        from one.research import _get_conn
        row = _get_conn().execute(
            "SELECT status FROM research_gaps WHERE id = ?", (gid,)
        ).fetchone()
        assert row["status"] == "open"


# ---------------------------------------------------------------------------
# Source quality scoring
# ---------------------------------------------------------------------------


class TestScoreSourceQuality:
    def test_default_score_for_plain_text(self):
        score = _score_source_quality("some generic statement without any indicators")
        assert score == 0.5

    def test_meta_analysis_high_score(self):
        score = _score_source_quality("A meta-analysis of 50 studies found significant effects.")
        assert score >= 0.95

    def test_systematic_review_high_score(self):
        score = _score_source_quality("systematic review of the literature")
        assert score >= 0.93

    def test_rct_high_score(self):
        score = _score_source_quality("randomized controlled trial showed improvements")
        assert score >= 0.90

    def test_blog_score_does_not_exceed_default(self):
        # blog=0.25 is below the 0.5 floor, so the score stays at the default.
        # Low-quality keywords cannot lower the score — only the absence of
        # high-quality keywords keeps it at the minimum baseline.
        score = _score_source_quality("According to this blog post, things are changing.")
        assert score == 0.5

    def test_reddit_score_stays_at_default(self):
        # reddit=0.15 is below the 0.5 floor — score stays at default.
        score = _score_source_quality("A reddit thread discussed this topic.")
        assert score == 0.5

    def test_social_media_score_stays_at_default(self):
        # social media=0.10 is below the 0.5 floor — score stays at default.
        score = _score_source_quality("social media posts indicate widespread concern.")
        assert score == 0.5

    def test_low_quality_keywords_below_default_floor(self):
        # Verify that the scoring function never drops below the 0.5 default
        # even when only low-quality source indicators are present.
        low_quality_texts = [
            "According to this blog post",
            "A reddit thread discussed this",
            "social media posts indicate this",
            "anecdotal evidence suggests this",
            "a press release announced this",
            "a news article reported this",
            "someone on a forum wrote this",
        ]
        for text in low_quality_texts:
            score = _score_source_quality(text)
            assert score >= 0.5, f"Score unexpectedly below default for: {text}"

    def test_percentage_boosts_score(self):
        baseline = _score_source_quality("some generic statement")
        boosted = _score_source_quality("some generic statement showing 42% improvement")
        assert boosted > baseline

    def test_p_value_boosts_score(self):
        baseline = _score_source_quality("some generic statement")
        boosted = _score_source_quality("some generic statement p < 0.05 significance")
        assert boosted > baseline

    def test_sample_size_boosts_score(self):
        baseline = _score_source_quality("generic statement")
        boosted = _score_source_quality("generic statement n=500 participants")
        assert boosted > baseline

    def test_score_capped_at_one(self):
        # pile on multiple high-quality signals — should not exceed 1.0
        text = "meta-analysis cochrane randomized controlled trial 80% p < 0.001 n=10000"
        assert _score_source_quality(text) <= 1.0

    def test_score_above_zero(self):
        assert _score_source_quality("any text") >= 0.0

    def test_case_insensitive(self):
        lower = _score_source_quality("meta-analysis")
        upper = _score_source_quality("META-ANALYSIS")
        assert lower == upper

    def test_cochrane_score(self):
        score = _score_source_quality("Cochrane review concluded no effect.")
        assert score >= 0.95

    def test_nejm_score(self):
        score = _score_source_quality("Published in NEJM, the study showed results.")
        assert score >= 0.92

    def test_preprint_medium_score(self):
        score = _score_source_quality("preprint posted on bioRxiv")
        assert 0.45 <= score <= 0.65

    def test_best_score_wins(self):
        """When multiple keywords present, highest-scoring one wins."""
        text = "A meta-analysis referenced in a blog post"
        score = _score_source_quality(text)
        # meta-analysis=0.95 > blog=0.25, so should reflect meta-analysis
        assert score >= 0.95


# ---------------------------------------------------------------------------
# Quantitative data extraction
# ---------------------------------------------------------------------------


class TestExtractQuantitativeData:
    def test_empty_string(self):
        assert _extract_quantitative_data("") == ""

    def test_no_quantitative_data(self):
        result = _extract_quantitative_data("There were some changes over time.")
        assert result == ""

    def test_extracts_percentage(self):
        result = _extract_quantitative_data("A 42% reduction was observed.")
        assert "42%" in result

    def test_extracts_decimal_percentage(self):
        result = _extract_quantitative_data("Results showed 3.7% improvement.")
        assert "3.7%" in result

    def test_extracts_p_value(self):
        result = _extract_quantitative_data("Significant result (p < 0.05).")
        assert "p < 0.05" in result or "p<0.05" in result.replace(" ", "")

    def test_extracts_sample_size(self):
        result = _extract_quantitative_data("The study enrolled n=1200 participants.")
        assert "n=1200" in result.replace(" ", "")

    def test_extracts_dosage_mg(self):
        result = _extract_quantitative_data("Patients received 500mg daily.")
        assert "500mg" in result.replace(" ", "")

    def test_extracts_odds_ratio(self):
        result = _extract_quantitative_data("The OR = 1.5 suggests elevated risk.")
        assert "OR" in result or "1.5" in result

    def test_extracts_effect_size(self):
        result = _extract_quantitative_data("effect size = 0.8 was found.")
        assert "0.8" in result

    def test_extracts_fold_change(self):
        result = _extract_quantitative_data("Expression increased 3.2 fold in treated cells.")
        assert "3.2" in result or "fold" in result.lower()

    def test_multiple_values_extracted(self):
        text = "42% reduction (p < 0.001), n=500 subjects, OR = 2.1"
        result = _extract_quantitative_data(text)
        assert "42%" in result or "p" in result or "n=500" in result.replace(" ", "")
        # Should have multiple data points separated by semicolons
        assert ";" in result

    def test_limit_ten_results(self):
        # Generate text with 15 different percentages
        text = " ".join(f"{i}%" for i in range(1, 16))
        result = _extract_quantitative_data(text)
        # Should cap at 10 entries
        parts = [p.strip() for p in result.split(";") if p.strip()]
        assert len(parts) <= 10

    def test_case_insensitive_sample_size(self):
        result = _extract_quantitative_data("N=300 was the sample size.")
        assert result != ""

    def test_p_equals(self):
        result = _extract_quantitative_data("p = 0.03 was the significance level.")
        assert result != ""


# ---------------------------------------------------------------------------
# Temporal info extraction
# ---------------------------------------------------------------------------


class TestExtractTemporalInfo:
    def test_empty_string(self):
        assert _extract_temporal_info("") == ""

    def test_no_year(self):
        assert _extract_temporal_info("This has no dates in it.") == ""

    def test_parenthetical_year(self):
        result = _extract_temporal_info("Smith et al. (2021) found that X.")
        assert result == "2021"

    def test_published_in_pattern(self):
        result = _extract_temporal_info("Published in 2019, the study showed results.")
        assert result == "2019"

    def test_reported_in_pattern(self):
        result = _extract_temporal_info("Reported in 2022, the findings were replicated.")
        assert result == "2022"

    def test_bare_year_pattern(self):
        result = _extract_temporal_info("The 2015 study used a different methodology.")
        assert result == "2015"

    def test_returns_most_recent_year(self):
        result = _extract_temporal_info("Studies from 2010 and 2020 were compared (2018).")
        assert result == "2020"

    def test_year_out_of_range_ignored(self):
        # Year 1985 is below the threshold of 1990
        result = _extract_temporal_info("A 1985 study is referenced here.")
        assert result == ""

    def test_future_year_ignored(self):
        # 2031 is above the 2030 cutoff
        result = _extract_temporal_info("Projected for 2031 completion.")
        assert result == ""

    def test_multiple_years_returns_latest(self):
        result = _extract_temporal_info("(1995) and (2023) and showed in 2017")
        assert result == "2023"

    def test_2020s_year(self):
        result = _extract_temporal_info("The 2024 meta-analysis confirmed earlier findings.")
        assert result == "2024"

    def test_1990s_year(self):
        result = _extract_temporal_info("A 1998 Cochrane review established the baseline.")
        assert result == "1998"

    def test_demonstrated_in_pattern(self):
        result = _extract_temporal_info("Demonstrated in 2016 using fMRI.")
        assert result == "2016"


# ---------------------------------------------------------------------------
# Finding splitter
# ---------------------------------------------------------------------------


class TestSplitFindings:
    def test_empty_string(self):
        result = _split_findings("")
        assert result == []

    def test_short_string_ignored(self):
        result = _split_findings("too short")
        assert result == []

    def test_single_long_paragraph(self):
        text = "This is a sufficiently long finding that has no obvious split points anywhere."
        result = _split_findings(text)
        assert len(result) == 1
        assert result[0] == text

    def test_splits_on_bullet_dash(self):
        text = "- First finding about something important\n- Second finding about another thing"
        result = _split_findings(text)
        assert len(result) >= 1
        # Each segment should be meaningful
        assert all(len(s) >= 20 for s in result)

    def test_splits_on_numbered_list(self):
        text = "1. First important finding here\n2. Second important finding here"
        result = _split_findings(text)
        assert len(result) >= 1

    def test_splits_on_double_newline(self):
        text = "First paragraph is long enough to matter.\n\nSecond paragraph is also long enough."
        result = _split_findings(text)
        assert len(result) >= 1

    def test_short_segments_filtered(self):
        text = "- ok\n- This is a longer finding that should be kept in the results"
        result = _split_findings(text)
        # "ok" segment is < 20 chars, should be filtered
        assert all(len(s) >= 20 for s in result)

    def test_strips_whitespace(self):
        text = "  A finding with leading whitespace that is long enough to include  "
        result = _split_findings(text)
        if result:
            assert result[0] == result[0].strip()

    def test_bullet_star(self):
        text = "* First finding that is long enough to pass the filter\n* Second finding that also passes"
        result = _split_findings(text)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Depth prompt constants
# ---------------------------------------------------------------------------


class TestDepthPrompts:
    def test_depth_0_is_list(self):
        assert isinstance(DEPTH_0_PROMPTS, list)
        assert len(DEPTH_0_PROMPTS) > 0

    def test_depth_1_is_list(self):
        assert isinstance(DEPTH_1_PROMPTS, list)
        assert len(DEPTH_1_PROMPTS) > 0

    def test_depth_2_is_list(self):
        assert isinstance(DEPTH_2_PROMPTS, list)
        assert len(DEPTH_2_PROMPTS) > 0

    def test_depth_3_is_list(self):
        assert isinstance(DEPTH_3_PROMPTS, list)
        assert len(DEPTH_3_PROMPTS) > 0

    def test_all_depth_prompts_contains_four_levels(self):
        assert len(ALL_DEPTH_PROMPTS) == 4

    def test_all_depth_prompts_matches_individual_lists(self):
        assert ALL_DEPTH_PROMPTS[0] is DEPTH_0_PROMPTS
        assert ALL_DEPTH_PROMPTS[1] is DEPTH_1_PROMPTS
        assert ALL_DEPTH_PROMPTS[2] is DEPTH_2_PROMPTS
        assert ALL_DEPTH_PROMPTS[3] is DEPTH_3_PROMPTS

    def test_prompts_contain_topic_placeholder(self):
        for depth_list in ALL_DEPTH_PROMPTS:
            for prompt in depth_list:
                assert "{topic}" in prompt, f"Prompt missing {{topic}}: {prompt[:60]}"

    def test_prompts_are_strings(self):
        for depth_list in ALL_DEPTH_PROMPTS:
            for prompt in depth_list:
                assert isinstance(prompt, str)
                assert len(prompt) > 20

    def test_gap_identification_prompt_has_placeholders(self):
        assert "{topic}" in GAP_IDENTIFICATION_PROMPT
        assert "{depth}" in GAP_IDENTIFICATION_PROMPT
        assert "{findings}" in GAP_IDENTIFICATION_PROMPT

    def test_followup_prompt_has_placeholders(self):
        assert "{topic}" in FOLLOWUP_PROMPT
        assert "{gap}" in FOLLOWUP_PROMPT

    def test_adversarial_prompt_has_placeholders(self):
        assert "{topic}" in ADVERSARIAL_PROMPT
        assert "{finding}" in ADVERSARIAL_PROMPT

    def test_depth_0_covers_breadth(self):
        combined = " ".join(DEPTH_0_PROMPTS).lower()
        assert "survey" in combined or "comprehensive" in combined or "breadth" in combined

    def test_depth_1_covers_mechanisms(self):
        combined = " ".join(DEPTH_1_PROMPTS).lower()
        assert "mechanism" in combined or "causal" in combined or "theory" in combined

    def test_depth_2_covers_adversarial(self):
        combined = " ".join(DEPTH_2_PROMPTS).lower()
        assert "bias" in combined or "contradict" in combined or "counter" in combined

    def test_depth_3_covers_synthesis(self):
        combined = " ".join(DEPTH_3_PROMPTS).lower()
        assert "synthesis" in combined or "unified" in combined or "combination" in combined


# ---------------------------------------------------------------------------
# Source quality keyword coverage
# ---------------------------------------------------------------------------


class TestSourceQualityKeywords:
    def test_is_dict(self):
        assert isinstance(SOURCE_QUALITY_KEYWORDS, dict)

    def test_all_scores_in_range(self):
        for keyword, score in SOURCE_QUALITY_KEYWORDS.items():
            assert 0.0 <= score <= 1.0, f"Score for '{keyword}' out of range: {score}"

    def test_high_quality_sources(self):
        high_quality = {"meta-analysis", "systematic review", "cochrane", "nejm", "replicated"}
        for kw in high_quality:
            assert kw in SOURCE_QUALITY_KEYWORDS
            assert SOURCE_QUALITY_KEYWORDS[kw] >= 0.80

    def test_low_quality_sources(self):
        low_quality = {"blog", "anecdotal", "social media", "reddit"}
        for kw in low_quality:
            assert kw in SOURCE_QUALITY_KEYWORDS
            assert SOURCE_QUALITY_KEYWORDS[kw] <= 0.35


# ---------------------------------------------------------------------------
# Research frontier
# ---------------------------------------------------------------------------


class TestResearchFrontier:
    def _make_topic(self, project="proj", topic="topic"):
        from datetime import datetime, timezone
        from one.research import _get_conn
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO research_topics (project, topic, status, turn_budget, depth_level, created, updated) VALUES (?, ?, 'active', 10, 0, ?, ?)",
            (project, topic, now, now),
        )
        conn.commit()
        return cur.lastrowid

    def test_frontier_empty_project(self):
        result = research_frontier("no_such_project")
        assert result["open_gaps"] == []
        assert result["active_topics"] == []
        assert result["total_gaps"] == 0

    def test_frontier_keys_present(self):
        result = research_frontier("proj")
        for key in ("open_gaps", "active_topics", "recent_findings",
                    "contradictions", "total_gaps", "high_priority_gaps"):
            assert key in result

    def test_frontier_with_gaps(self):
        tid = self._make_topic(project="proj2", topic="quantum computing")
        _store_gap(tid, "What is the decoherence time limit?", priority=0.9)
        _store_gap(tid, "How does error correction scale?", priority=0.5)

        result = research_frontier("proj2")
        assert result["total_gaps"] == 2
        assert result["high_priority_gaps"] == 1
        # Gaps should be sorted by priority descending
        assert result["open_gaps"][0]["priority"] >= result["open_gaps"][1]["priority"]

    def test_frontier_gap_has_topic_field(self):
        tid = self._make_topic(project="proj3", topic="neuroscience")
        _store_gap(tid, "What are the mechanisms of long-term potentiation?")
        result = research_frontier("proj3")
        gap = result["open_gaps"][0]
        assert gap["topic"] == "neuroscience"
        assert "question" in gap
        assert "priority" in gap
        assert "gap_id" in gap

    def test_frontier_excludes_resolved_gaps(self):
        tid = self._make_topic(project="proj4", topic="biology")
        gid = _store_gap(tid, "An open question that remains unanswered?")
        fid = _store_finding(tid, "The answer to this question has been found.")
        _resolve_gap(gid, fid)

        result = research_frontier("proj4")
        assert result["total_gaps"] == 0

    def test_frontier_with_contradictions(self):
        tid = self._make_topic(project="proj5", topic="medicine")
        fid1 = _store_finding(tid, "Drug X reduces symptoms significantly in trials.")
        fid2 = _store_finding(tid, "Drug X showed no benefit in meta-analysis.")
        _store_citation(fid2, fid1, "contradicts")

        result = research_frontier("proj5")
        assert len(result["contradictions"]) == 1
        c = result["contradictions"][0]
        assert "finding_1" in c
        assert "finding_2" in c
        assert c["relation"] == "contradicts"
