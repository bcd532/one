"""Tests for excitation detection."""

from one.excitation import (
    score_user_excitation, score_assistant_excitation, score_excitation,
    _user_caps_ratio, _user_profanity_density, _user_emphasis,
    _user_length_anomaly, _has_substance, _is_rage,
)


class TestUserCapsRatio:
    def test_all_caps(self):
        assert _user_caps_ratio("HELLO WORLD") == 1.0

    def test_no_caps(self):
        assert _user_caps_ratio("hello world") == 0.0

    def test_mixed(self):
        ratio = _user_caps_ratio("Hello World")
        assert 0.0 < ratio < 1.0

    def test_no_alpha(self):
        assert _user_caps_ratio("123 !!! ???") == 0.0


class TestProfanityDensity:
    def test_no_profanity(self):
        assert _user_profanity_density("hello world") == 0.0

    def test_has_profanity(self):
        density = _user_profanity_density("holy shit this actually works")
        assert density > 0.0

    def test_empty(self):
        assert _user_profanity_density("") == 0.0


class TestEmphasis:
    def test_exclamation_marks(self):
        score = _user_emphasis("This is amazing!!!")
        assert score > 0.0

    def test_repeated_chars(self):
        score = _user_emphasis("yooooo that's cool")
        assert score > 0.0

    def test_no_emphasis(self):
        score = _user_emphasis("this is normal text")
        assert score == 0.0


class TestSubstance:
    def test_has_substance(self):
        assert _has_substance("the authentication system needs refactoring") is True

    def test_no_substance(self):
        assert _has_substance("FUCK SHIT DAMN") is False

    def test_mixed(self):
        assert _has_substance("holy shit the encoding actually works") is True


class TestRage:
    def test_rage_detected(self):
        assert _is_rage("FUCK THIS SHIT") is True

    def test_not_rage_with_content(self):
        assert _is_rage("HOLY SHIT the trigram encoding works perfectly now") is False

    def test_not_rage_normal(self):
        assert _is_rage("please fix the bug") is False


class TestUserExcitation:
    def test_calm_message(self):
        score = score_user_excitation("please fix the authentication bug")
        assert score < 0.5

    def test_eureka_moment(self):
        score = score_user_excitation(
            "Holy shit, I just realized that binding and unbinding gives us associative memory for free!"
        )
        assert score > 0.7

    def test_rage_scores_low(self):
        score = score_user_excitation("FUCK SHIT DAMN HELL")
        assert score < 0.2

    def test_eureka_with_profanity(self):
        score = score_user_excitation(
            "dude bro this encoding actually works for multilingual text!"
        )
        assert score > 0.5


class TestAssistantExcitation:
    def test_routine_response(self):
        score = score_assistant_excitation("Here is the file you requested.")
        assert score < 0.5

    def test_breakthrough_language(self):
        score = score_assistant_excitation(
            "Wait, actually — I just realized this changes everything. "
            "The binding operation is its own inverse."
        )
        assert score > 0.7

    def test_long_response_scores_higher(self):
        short = score_assistant_excitation("Done.")
        long_text = "This is significant because " + "x " * 300
        long_score = score_assistant_excitation(long_text)
        assert long_score > short


class TestCombinedScorer:
    def test_user_routing(self):
        score = score_excitation("test message", "user")
        assert isinstance(score, float)

    def test_assistant_routing(self):
        score = score_excitation("test message", "assistant")
        assert isinstance(score, float)

    def test_unknown_source(self):
        score = score_excitation("test", "unknown")
        assert score == 0.0
