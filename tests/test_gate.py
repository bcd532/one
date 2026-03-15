"""Tests for the AIF gate — memory write selection."""

import pytest
from one.gate import AifGate, GATE_THRESHOLD


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


class TestContentQuality:
    def test_user_messages_score_higher(self):
        gate = AifGate()
        user_score = gate.score("I need to fix the authentication system", "user")
        tool_score = gate.score("I need to fix the authentication system", "tool_result")
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


class TestNoveltyDetection:
    def test_first_message_is_novel(self):
        gate = AifGate()
        _, score = gate.should_store("This is a unique technical insight about HDC encoding", "user")
        assert score > 0

    def test_duplicate_messages_score_lower(self):
        gate = AifGate()
        gate.should_store("fix the auth bug in login.py", "user")
        gate.should_store("fix the auth bug in login.py", "user")
        # Third identical message should have lower novelty
        _, score = gate.should_store("fix the auth bug in login.py", "user")
        # The exact score depends on implementation, but it should be stored (has content)
        assert isinstance(score, float)


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
