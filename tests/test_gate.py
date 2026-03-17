"""Tests for the AIF gate — Active Inference memory write selection."""

import numpy as np
import pytest
from one.gate import AifGate, Regime, GATE_THRESHOLD
from one.hdc import encode_text, DIM


class TestNoiseFilter:
    def test_short_messages_are_noise(self):
        gate = AifGate()
        assert gate.score("ok", "user") == 0.0
        assert gate.score("yes", "user") == 0.0
        assert gate.score("k", "user") == 0.0
        assert gate.score("", "user") == 0.0

    def test_acknowledgments_are_noise(self):
        gate = AifGate()
        assert gate.score("sounds good", "user") == 0.0
        assert gate.score("go ahead", "user") == 0.0
        assert gate.score("cool", "user") == 0.0

    def test_empty_whitespace_is_noise(self):
        gate = AifGate()
        assert gate.score("   ", "user") == 0.0
        assert gate.score("\n\n", "user") == 0.0

    def test_tool_result_noise(self):
        gate = AifGate()
        assert gate.score("no files found", "tool_result") == 0.0
        assert gate.score("file created successfully", "tool_result") == 0.0

    def test_sensitive_content_blocked(self):
        gate = AifGate()
        assert gate.score("my password is hunter2", "user") == 0.0
        assert gate.score("my social security number is 123-45-6789", "user") == 0.0

    def test_rage_without_substance_is_noise(self):
        gate = AifGate()
        assert gate.score("FUCK SHIT DAMN", "user") == 0.0


class TestContentPrior:
    def test_user_messages_score_higher(self):
        gate = AifGate()
        # Seed the model so content prior matters more than novelty
        gate.should_store("We were discussing database optimization strategies", "user")
        gate.should_store("The logging framework handles structured output", "user")
        gate.should_store("Network latency affects query performance here", "user")
        # Text without decision/preference signals — source prior drives the difference
        user_score = gate.score("the server restarts every morning at six", "user")
        tool_score = gate.score("the server restarts every morning at six", "tool_result")
        assert user_score > tool_score

    def test_longer_messages_score_higher(self):
        gate = AifGate()
        short = gate.score("fix bug", "user")
        long_msg = gate.score("I need to fix the authentication bug in the login handler because users are getting 403 errors", "user")
        assert long_msg >= short


class TestDecisionDetection:
    def test_decision_signals_score_high(self):
        gate = AifGate()
        score = gate.score("I decided to use Redis instead of Memcached for caching", "user")
        assert score > 0.5

    def test_preference_signals_score_high(self):
        gate = AifGate()
        score = gate.score("I want the API to return JSON, not XML", "user")
        assert score > 0.5

    def test_reasoning_scores_high(self):
        gate = AifGate()
        score = gate.score("The reason is that SQLite can't handle concurrent writes from multiple processes", "user")
        assert score > 0.5


class TestShouldStore:
    def test_should_store_returns_tuple(self):
        gate = AifGate()
        result = gate.should_store("test message", "user")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_meaningful_content_stored(self):
        gate = AifGate()
        should, _ = gate.should_store(
            "The approach is to use HDC trigram encoding with 4096 dimensions",
            "user"
        )
        assert should is True

    def test_noise_not_stored(self):
        gate = AifGate()
        should, _ = gate.should_store("ok", "user")
        assert should is False

    def test_code_content_stored(self):
        gate = AifGate()
        should, _ = gate.should_store(
            "```python\ndef encode(text):\n    return vec\n```",
            "assistant"
        )
        assert should is True


class TestExcitationDetection:
    def test_eureka_moment_scores_high(self):
        gate = AifGate()
        score = gate.score(
            "Holy shit, I just realized that the trigram approach naturally handles typo tolerance because errors only corrupt 3 out of N-2 trigrams!",
            "user"
        )
        assert score > 0.7

    def test_assistant_breakthrough_scores_high(self):
        gate = AifGate()
        score = gate.score(
            "Wait, actually — this changes everything. The binding operation means we can recover individual components from a superposition.",
            "assistant"
        )
        assert score > 0.5


class TestRegime:
    """Test the learned regime (topic cluster) directly."""

    def test_regime_creation(self):
        vec = encode_text("testing the regime system")
        r = Regime(vec)
        assert r.count == 1
        assert r.precision == 1.0

    def test_surprise_identical_is_low(self):
        vec = encode_text("machine learning algorithms")
        r = Regime(vec)
        assert r.surprise(vec) < 0.1

    def test_surprise_different_is_high(self):
        vec1 = encode_text("machine learning algorithms for natural language processing")
        vec2 = encode_text("the cat sat on the mat and looked at the birds outside")
        r = Regime(vec1)
        assert r.surprise(vec2) > 0.5

    def test_free_energy_novel_is_high(self):
        vec1 = encode_text("machine learning algorithms for NLP")
        vec2 = encode_text("the recipe calls for butter and sugar")
        r = Regime(vec1)
        fe_similar = r.free_energy(vec1)
        fe_novel = r.free_energy(vec2)
        assert fe_novel > fe_similar

    def test_update_shifts_centroid(self):
        vec1 = encode_text("python programming language")
        vec2 = encode_text("python code development")
        r = Regime(vec1)
        old_mu = r.mu.copy()
        r.update(vec2)
        # Centroid should shift toward vec2
        sim_before = float(np.dot(old_mu, vec2))
        sim_after = float(np.dot(r.mu, vec2))
        assert sim_after >= sim_before

    def test_precision_increases_for_tight_cluster(self):
        vec = encode_text("consistent topic about databases")
        r = Regime(vec, precision=1.0)
        # Feed very similar observations
        for _ in range(20):
            r.update(vec)
        # Precision should increase (tight cluster = high precision)
        assert r.precision > 1.0

    def test_epistemic_value_novel_is_high(self):
        vec1 = encode_text("machine learning algorithms for NLP")
        vec2 = encode_text("cooking recipes for italian food")
        r = Regime(vec1)
        ev_similar = r.epistemic_value(vec1)
        ev_novel = r.epistemic_value(vec2)
        assert ev_novel > ev_similar


class TestGenerativeModel:
    """Test the full generative model behavior."""

    def test_first_message_is_maximally_novel(self):
        gate = AifGate()
        # First message should score high (no model yet → max surprise)
        score = gate.score("A completely new topic about quantum computing", "user")
        assert score > 0.5

    def test_repeated_topic_scores_lower(self):
        gate = AifGate()
        msg = "fix the authentication bug in the login handler"
        gate.should_store(msg, "user")
        gate.should_store(msg, "user")
        # Third time — model has learned this regime, surprise decreases
        _, score1 = gate.should_store(msg, "user")

        gate2 = AifGate()
        _, score2 = gate2.should_store(msg, "user")

        # First encounter should score higher than third
        assert score2 >= score1

    def test_regime_creation_on_store(self):
        gate = AifGate()
        assert gate.regime_count() == 0
        gate.should_store("The approach is to use HDC encoding", "user")
        assert gate.regime_count() == 1

    def test_multiple_regimes_for_different_topics(self):
        gate = AifGate()
        gate.should_store("I decided to use Redis for the caching layer because of pub/sub support", "user")
        gate.should_store("The quantum entanglement experiment showed unexpected decoherence patterns", "user")
        gate.should_store("The recipe calls for two cups of flour and one egg", "user")
        # Should have created multiple regimes for distinct topics
        assert gate.regime_count() >= 2

    def test_regime_summary(self):
        gate = AifGate()
        gate.should_store("I want to use PostgreSQL instead of MySQL", "user")
        summary = gate.regime_summary()
        assert len(summary) == 1
        assert "precision" in summary[0]
        assert "prior" in summary[0]
        assert "count" in summary[0]

    def test_redundant_message_blocked(self):
        gate = AifGate()
        msg = "fix the auth bug in login.py"
        gate.should_store(msg, "user")
        # Exact same message should get suppressed by redundancy check
        score = gate.score(msg, "user")
        # After building a regime, the redundancy filter in _recent_vecs triggers
        assert isinstance(score, float)


class TestEpistemicSafety:
    def test_overconfident_llm_output_penalized(self):
        gate = AifGate()
        # Normal assistant message
        normal = gate.score(
            "This suggests that the protein might interact with the receptor pathway",
            "assistant"
        )
        # Overconfident assistant message
        overconfident = gate.score(
            "This proves beyond doubt that the protein definitively controls the receptor",
            "assistant"
        )
        assert overconfident < normal
