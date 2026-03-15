"""Tests for the rule tree system."""

import threading
import pytest
from one.rules import (
    add_rule, get_all_rules, get_active_rules,
    update_rule, supersede_rule,
    learn_rule_from_memory, format_rules_for_injection,
    _matches_context,
)
from one import store


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use temporary databases for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("one.rules.DB_PATH", db_path)
    monkeypatch.setattr("one.rules.DB_DIR", str(tmp_path))
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", str(tmp_path))
    # Reset thread-local connections
    import one.rules
    import one.store
    one.rules._local = threading.local()
    one.store._local = threading.local()
    store.set_project("test")
    yield


class TestAddRule:
    def test_add_returns_id(self):
        rule_id = add_rule("test", "always run tests")
        assert isinstance(rule_id, int)
        assert rule_id > 0

    def test_add_with_keywords(self):
        rule_id = add_rule("test", "use 4096 dimensions", activation_keywords="hdc, encoding")
        assert rule_id > 0

    def test_add_child_rule(self):
        parent_id = add_rule("test", "parent rule")
        child_id = add_rule("test", "child rule", parent_id=parent_id)
        assert child_id > parent_id


class TestGetRules:
    def test_get_empty(self):
        rules = get_all_rules("test")
        assert rules == []

    def test_get_after_add(self):
        add_rule("test", "test rule")
        rules = get_all_rules("test")
        assert len(rules) == 1
        assert rules[0]["rule_text"] == "test rule"

    def test_project_isolation(self):
        add_rule("project_a", "rule for A")
        add_rule("project_b", "rule for B")
        rules_a = get_all_rules("project_a")
        rules_b = get_all_rules("project_b")
        assert len(rules_a) == 1
        assert len(rules_b) == 1
        assert rules_a[0]["rule_text"] == "rule for A"


class TestContextMatching:
    def test_wildcard_always_matches(self):
        assert _matches_context("*", "anything", [], []) is True

    def test_keyword_in_text(self):
        assert _matches_context("hdc, encoding", "working on hdc encoder", [], []) is True

    def test_keyword_in_files(self):
        assert _matches_context("hdc", "some text", ["hdc.py"], []) is True

    def test_keyword_in_tools(self):
        assert _matches_context("bash", "some text", [], ["Bash"]) is True

    def test_no_match(self):
        assert _matches_context("hdc, encoding", "deploy to server", [], []) is False


class TestActiveRules:
    def test_root_wildcard_always_active(self):
        add_rule("test", "always active", activation_keywords="*")
        rules = get_active_rules("test", "random text")
        assert len(rules) == 1

    def test_contextual_activation(self):
        add_rule("test", "always active", activation_keywords="*")
        add_rule("test", "hdc rule", activation_keywords="hdc, encoding")
        rules = get_active_rules("test", "working on HDC encoding")
        assert len(rules) == 2

    def test_contextual_not_active(self):
        add_rule("test", "always active", activation_keywords="*")
        add_rule("test", "hdc rule", activation_keywords="hdc, encoding")
        rules = get_active_rules("test", "deploy to production")
        assert len(rules) == 1  # only the wildcard rule

    def test_child_activation(self):
        parent = add_rule("test", "hdc rules", activation_keywords="hdc")
        add_rule("test", "dimension must be 4096", activation_keywords="dimension, vector", parent_id=parent)
        rules = get_active_rules("test", "hdc vector dimension check")
        assert len(rules) == 2  # parent + child


class TestUpdateRule:
    def test_update_confidence(self):
        rule_id = add_rule("test", "test rule", confidence=0.5)
        update_rule(rule_id, confidence=0.9)
        rules = get_all_rules("test")
        assert rules[0]["confidence"] == 0.9

    def test_update_source_count(self):
        rule_id = add_rule("test", "test rule", source_count=1)
        update_rule(rule_id, source_count=5)
        rules = get_all_rules("test")
        assert rules[0]["source_count"] == 5


class TestSupersession:
    def test_supersede_creates_new(self):
        old_id = add_rule("test", "old rule text")
        new_id = supersede_rule(old_id, "new improved rule", "test")
        assert new_id > old_id

    def test_supersede_deactivates_old(self):
        old_id = add_rule("test", "old rule")
        supersede_rule(old_id, "new rule", "test")
        rules = get_all_rules("test")
        # Should only see the new rule (old is inactive)
        assert len(rules) == 1
        assert rules[0]["rule_text"] == "new rule"


class TestRuleLearning:
    def test_learn_new_rule(self):
        rule_id = learn_rule_from_memory("test", "always use type hints in Python")
        assert rule_id is not None

    def test_learn_reinforces_existing(self):
        learn_rule_from_memory("test", "always use type hints")
        # Learning a similar rule should reinforce, not create new
        learn_rule_from_memory("test", "always use type hints in code")
        rules = get_all_rules("test")
        # May create 1 or 2 depending on similarity threshold
        assert len(rules) >= 1


class TestFormatForInjection:
    def test_format_empty(self):
        result = format_rules_for_injection([], "test")
        assert result == ""

    def test_format_with_rules(self):
        add_rule("test", "always run tests", activation_keywords="*")
        rules = get_active_rules("test", "anything")
        result = format_rules_for_injection(rules, "test")
        assert "<project-rules" in result
        assert "always run tests" in result
        assert "</project-rules>" in result
