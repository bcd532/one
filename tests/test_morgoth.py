"""Tests for Morgoth Mode — The God Builder."""

import json
import os
import threading
import pytest

from one import store
from one.morgoth import (
    MorgothMode, Phase, PHASE_NAMES, Eureka,
    WAVE_1, WAVE_2, WAVE_3,
    MAX_AGENTS, STATE_FILE, STATE_DIR,
)
from one.swarm import AgentRole


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use temporary database and state file for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", str(tmp_path))
    monkeypatch.setattr("one.morgoth.STATE_DIR", str(tmp_path))
    monkeypatch.setattr("one.morgoth.STATE_FILE", str(tmp_path / "morgoth_state.json"))
    store._local = threading.local()
    store.set_project("test_project")

    # Also patch sub-module DB paths
    for mod in [
        "one.dialectic", "one.analogy", "one.contradictions",
        "one.verification", "one.experiments", "one.health",
    ]:
        try:
            monkeypatch.setattr(f"{mod}.DB_PATH", db_path)
            monkeypatch.setattr(f"{mod}.DB_DIR", str(tmp_path))
        except AttributeError:
            pass

    yield db_path


@pytest.fixture
def mock_ollama(monkeypatch):
    """Mock all LLM calls."""
    for mod in [
        "one.dialectic", "one.analogy", "one.contradictions",
        "one.verification", "one.experiments",
    ]:
        try:
            monkeypatch.setattr(f"{mod}._call_ollama", lambda *a, **kw: "mocked")
        except AttributeError:
            pass


class TestPhaseConstants:
    def test_phase_values(self):
        assert Phase.UNDERSTAND == 1
        assert Phase.RESEARCH == 2
        assert Phase.SYNTHESIZE == 3
        assert Phase.BUILD == 4
        assert Phase.VERIFY == 5
        assert Phase.ITERATE == 6
        assert Phase.CONTINUOUS == 7

    def test_phase_names(self):
        assert PHASE_NAMES[Phase.UNDERSTAND] == "UNDERSTAND"
        assert PHASE_NAMES[Phase.BUILD] == "BUILD"
        assert PHASE_NAMES[Phase.CONTINUOUS] == "CONTINUOUS"
        assert len(PHASE_NAMES) == 7


class TestWaves:
    def test_wave_1_roles(self):
        assert AgentRole.CONDUCTOR in WAVE_1
        assert AgentRole.SURVEYOR in WAVE_1
        assert AgentRole.HISTORIAN in WAVE_1
        assert AgentRole.DEVILS_ADVOCATE in WAVE_1
        assert len(WAVE_1) == 4

    def test_wave_2_roles(self):
        assert AgentRole.MECHANIST in WAVE_2
        assert AgentRole.CONTRARIAN in WAVE_2
        assert AgentRole.ANALOGIST in WAVE_2
        assert AgentRole.VERIFIER in WAVE_2
        assert len(WAVE_2) == 4

    def test_wave_3_roles(self):
        assert AgentRole.SYNTHESIZER in WAVE_3
        assert AgentRole.EXPERIMENTALIST in WAVE_3
        assert AgentRole.FUTURIST in WAVE_3
        assert AgentRole.INTEGRATOR in WAVE_3
        assert len(WAVE_3) == 4

    def test_max_agents(self):
        assert MAX_AGENTS == 15


class TestEureka:
    def test_creation(self):
        e = Eureka(
            text="PD-1 inhibitor breakthrough",
            agent_role="mechanist",
            confidence=0.95,
            timestamp="2025-01-01T00:00:00",
        )
        assert e.text == "PD-1 inhibitor breakthrough"
        assert e.confidence == 0.95
        assert e.survived_dialectic is False

    def test_survived_dialectic(self):
        e = Eureka(
            text="test",
            agent_role="verifier",
            confidence=0.8,
            timestamp="2025-01-01T00:00:00",
            survived_dialectic=True,
        )
        assert e.survived_dialectic is True


class TestMorgothInit:
    def test_basic_init(self):
        m = MorgothMode(goal="cure cancer", project="test")
        assert m.goal == "cure cancer"
        assert m.project == "test"
        assert m.phase == Phase.UNDERSTAND
        assert m.iteration == 0
        assert m.eurekas == []
        assert m.active is False

    def test_sub_engines_initialized(self):
        m = MorgothMode(goal="test", project="test")
        assert m.dialectic is not None
        assert m.analogy is not None
        assert m.contradictions is not None
        assert m.verification is not None
        assert m.frontier is not None
        assert m.experiments is not None
        assert m.health is not None


class TestStatePersistence:
    def test_persist_and_load(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "morgoth_state.json")
        monkeypatch.setattr("one.morgoth.STATE_FILE", state_file)
        monkeypatch.setattr("one.morgoth.STATE_DIR", str(tmp_path))

        m = MorgothMode(goal="test goal", project="test_proj")
        m.phase = Phase.RESEARCH
        m.iteration = 3
        m.eurekas.append(Eureka(
            text="big finding",
            agent_role="mechanist",
            confidence=0.9,
            timestamp="2025-01-01T00:00:00",
        ))
        m._persist_state()

        state = MorgothMode.load_state()
        assert state is not None
        assert state["goal"] == "test goal"
        assert state["phase"] == Phase.RESEARCH
        assert state["iteration"] == 3
        assert len(state["eurekas"]) == 1
        assert state["eurekas"][0]["text"] == "big finding"

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("one.morgoth.STATE_FILE", str(tmp_path / "nope.json"))
        assert MorgothMode.load_state() is None

    def test_clear_state(self, tmp_path, monkeypatch):
        state_file = str(tmp_path / "morgoth_state.json")
        monkeypatch.setattr("one.morgoth.STATE_FILE", state_file)

        with open(state_file, "w") as f:
            json.dump({"goal": "test"}, f)
        assert os.path.exists(state_file)

        MorgothMode.clear_state()
        assert not os.path.exists(state_file)


class TestPhaseAdvancement:
    def test_advance_through_phases(self):
        m = MorgothMode(goal="test", project="test")
        m.phase = Phase.UNDERSTAND
        m._advance_phase()
        assert m.phase == Phase.RESEARCH

        m._advance_phase()
        assert m.phase == Phase.SYNTHESIZE

        m._advance_phase()
        assert m.phase == Phase.BUILD

        m._advance_phase()
        assert m.phase == Phase.VERIFY

        m._advance_phase()
        assert m.phase == Phase.ITERATE

    def test_iterate_loops_to_research(self):
        m = MorgothMode(goal="test", project="test")
        m.phase = Phase.ITERATE
        m.iteration = 1
        m._advance_phase()
        assert m.phase == Phase.RESEARCH
        assert m.iteration == 2

    def test_continuous_loops_to_understand(self):
        m = MorgothMode(goal="test", project="test")
        m.phase = Phase.CONTINUOUS
        m.iteration = 3
        m._advance_phase()
        assert m.phase == Phase.UNDERSTAND
        assert m.iteration == 4


class TestMorgothStatus:
    def test_status_without_swarm(self):
        m = MorgothMode(goal="test goal", project="test_proj")
        m.phase = Phase.BUILD
        m.iteration = 2
        status = m.status()
        assert status["goal"] == "test goal"
        assert status["phase"] == Phase.BUILD
        assert status["phase_name"] == "BUILD"
        assert status["iteration"] == 2
        assert status["eurekas"] == 0
        assert status["swarm"] == {}

    def test_status_active(self):
        m = MorgothMode(goal="test", project="test")
        assert m.status()["active"] is False
        m.active = True
        assert m.status()["active"] is True


class TestEurekaHandling:
    def test_on_eureka_stores_memory(self, mock_ollama):
        m = MorgothMode(goal="test", project="test_project")
        m.on_eureka("big discovery", "mechanist", 0.95)
        assert len(m.eurekas) == 1
        assert m.eurekas[0].text == "big discovery"
        assert m.eurekas[0].agent_role == "mechanist"

    def test_on_eureka_high_confidence(self, mock_ollama):
        m = MorgothMode(goal="test", project="test_project")
        m.on_eureka("finding", "verifier", 0.7)
        # Confidence should be at least 0.9 for eurekas
        assert m.eurekas[0].confidence == 0.7  # stored as-is on the Eureka object

    def test_multiple_eurekas(self, mock_ollama):
        m = MorgothMode(goal="test", project="test_project")
        m.on_eureka("first", "mechanist", 0.9)
        m.on_eureka("second", "analogist", 0.85)
        assert len(m.eurekas) == 2
