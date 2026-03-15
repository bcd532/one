"""Tests for the Swarm TUI Dashboard widgets."""

import pytest
from one.swarm_tui import (
    _sparkline,
    AgentStatusWidget,
    StatsPanel,
    BreakthroughPanel,
    ContradictionPanel,
    DialecticPanel,
    SwarmView,
)


class TestSparkline:
    def test_empty_values(self):
        result = _sparkline([], width=8)
        assert result == "▁" * 8

    def test_single_value(self):
        result = _sparkline([1.0])
        assert len(result) == 1

    def test_uniform_values(self):
        result = _sparkline([5.0, 5.0, 5.0, 5.0])
        # All same value — should use lowest bar since range is 0
        assert len(result) == 4

    def test_ascending_values(self):
        result = _sparkline([0.0, 0.25, 0.5, 0.75, 1.0], width=10)
        # Should go from low to high bars
        assert len(result) == 5
        assert result[0] < result[-1] or result[0] == result[-1]

    def test_width_truncation(self):
        values = [float(i) for i in range(20)]
        result = _sparkline(values, width=5)
        assert len(result) == 5  # only last 5 values

    def test_negative_values(self):
        result = _sparkline([-1.0, 0.0, 1.0])
        assert len(result) == 3


class TestAgentStatusWidget:
    def test_init(self):
        w = AgentStatusWidget("agent-1", "surveyor")
        assert w.agent_id == "agent-1"
        assert w.role == "surveyor"
        assert w.turn == 0
        assert w.alive is True
        assert w.history == []

    def test_update_status(self):
        w = AgentStatusWidget("agent-1", "mechanist")
        w.update_status(5, "researching PD-1", 0.8)
        assert w.turn == 5
        assert w.activity == "researching PD-1"
        assert len(w.history) == 1
        assert w.history[0] == 0.8

    def test_history_cap(self):
        w = AgentStatusWidget("agent-1", "verifier")
        for i in range(100):
            w.update_status(i, f"step {i}", float(i))
        assert len(w.history) == 60  # capped at 60

    def test_render_alive(self):
        w = AgentStatusWidget("agent-1", "conductor")
        w.update_status(3, "orchestrating")
        text = w.render()
        plain = text.plain
        assert "conductor" in plain.lower()
        assert "T3" in plain

    def test_render_dead(self):
        w = AgentStatusWidget("agent-1", "contrarian")
        w.alive = False
        text = w.render()
        assert "○" in text.plain


class TestStatsPanel:
    def test_default_values(self):
        p = StatsPanel()
        assert p.findings == 0
        assert p.hypotheses == 0
        assert p.coverage == 0.0

    def test_render_returns_panel(self):
        p = StatsPanel()
        p.findings = 42
        result = p.render()
        assert result is not None


class TestBreakthroughPanel:
    def test_default_empty(self):
        p = BreakthroughPanel()
        assert p.latest == ""

    def test_render_with_text(self):
        p = BreakthroughPanel()
        p.latest = "PD-1 inhibitor breakthrough!"
        result = p.render()
        assert result is not None


class TestContradictionPanel:
    def test_empty_items(self):
        p = ContradictionPanel()
        result = p.render()
        assert result is not None

    def test_update_items(self):
        p = ContradictionPanel()
        items = [
            {"severity": "critical", "finding_a": "Drug X works", "status": "active"},
            {"severity": "paradigm", "finding_a": "Theory Y holds", "status": "active"},
        ]
        p.update_items(items)
        assert len(p._items) == 2

    def test_max_display(self):
        p = ContradictionPanel()
        items = [{"severity": "minor", "finding_a": f"item {i}", "status": "active"} for i in range(10)]
        p.update_items(items)
        # render only shows first 5
        result = p.render()
        assert result is not None


class TestDialecticPanel:
    def test_default_empty(self):
        p = DialecticPanel()
        assert p.thesis == ""
        assert p.antithesis == ""
        assert p.synthesis == ""

    def test_render_with_content(self):
        p = DialecticPanel()
        p.thesis = "Drug X is effective"
        p.antithesis = "Drug X has no effect in RCTs"
        p.synthesis = "Drug X effective only in BRCA-mutated populations"
        result = p.render()
        assert result is not None
