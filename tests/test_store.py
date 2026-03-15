"""Tests for the SQLite memory store."""

import os
import sqlite3
import tempfile
import pytest
import numpy as np

from one import store
from one.store import (
    push_memory, recall, get_memory_by_time, stats,
    ensure_entity, link_memory_entity, get_entities,
    get_memories_for_entity, get_related_entities,
    set_project, get_project,
    _vec_to_blob, _blob_to_vec,
)


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a temporary database for each test."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("one.store.DB_PATH", db_path)
    monkeypatch.setattr("one.store.DB_DIR", str(tmp_path))
    # Reset thread-local connection
    import threading
    store._local = threading.local()
    set_project("test_project")
    yield db_path


class TestVectorBlob:
    def test_roundtrip(self):
        vec = [1.0, 2.0, 3.0, 4.0]
        blob = _vec_to_blob(vec)
        result = _blob_to_vec(blob)
        assert np.allclose(result, vec)

    def test_high_dim_roundtrip(self):
        vec = np.random.randn(4096).tolist()
        blob = _vec_to_blob(vec)
        result = _blob_to_vec(blob)
        assert np.allclose(result, vec, atol=1e-6)


class TestPushMemory:
    def test_push_returns_id(self):
        mid = push_memory("test memory", "user")
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_push_with_vector(self):
        vec = np.random.randn(4096).tolist()
        mid = push_memory("test", "user", hdc_vector=vec)
        assert mid is not None

    def test_push_auto_encodes_vector(self):
        mid = push_memory("hello world test", "user")
        # Should have auto-generated a vector
        assert mid is not None

    def test_push_with_all_fields(self):
        mid = push_memory(
            raw_text="decision: use Redis",
            source="user",
            tm_label="decision",
            regime_tag="architecture",
            aif_confidence=0.9,
            project="myproject",
        )
        assert mid is not None


class TestRecall:
    def test_recall_empty_store(self):
        results = recall("anything")
        assert results == []

    def test_recall_finds_similar(self):
        push_memory("fix the authentication bug in login.py", "user")
        push_memory("deploy to production server", "user")
        results = recall("auth bug")
        assert len(results) > 0
        # The auth-related memory should rank higher
        assert "auth" in results[0]["raw_text"].lower()

    def test_recall_respects_n(self):
        for i in range(20):
            push_memory(f"memory number {i} about different topics", "user")
        results = recall("memory", n=5)
        assert len(results) <= 5

    def test_recall_returns_correct_fields(self):
        push_memory("test recall fields", "user", tm_label="test_label")
        results = recall("test recall fields")
        assert len(results) > 0
        r = results[0]
        assert "id" in r
        assert "raw_text" in r
        assert "source" in r
        assert "timestamp" in r
        assert "similarity" in r


class TestProjectScoping:
    def test_set_get_project(self):
        set_project("myproject")
        assert get_project() == "myproject"

    def test_project_isolation(self):
        set_project("project_a")
        push_memory("secret project A data", "user", project="project_a")
        set_project("project_b")
        push_memory("project B data", "user", project="project_b")

        results = recall("secret project A", project="project_b")
        # Should not find project_a's data when scoped to project_b
        # (unless it falls back to global, which it doesn't in this case)
        for r in results:
            assert r["project"] in ("project_b", "global")


class TestTimeQuery:
    def test_get_by_time(self):
        push_memory("time test memory", "user")
        results = get_memory_by_time(limit=10)
        assert len(results) > 0

    def test_get_by_source(self):
        push_memory("user message", "user")
        push_memory("assistant message", "assistant")
        results = get_memory_by_time(source="user")
        assert all(r["source"] == "user" for r in results)


class TestEntityOperations:
    def test_ensure_entity_creates(self):
        eid = ensure_entity({"name": "TestEntity", "type": "concept"})
        assert isinstance(eid, int)
        assert eid > 0

    def test_ensure_entity_idempotent(self):
        eid1 = ensure_entity({"name": "TestEntity", "type": "concept"})
        eid2 = ensure_entity({"name": "TestEntity", "type": "concept"})
        assert eid1 == eid2

    def test_ensure_entity_increments_count(self):
        ensure_entity({"name": "CountEntity", "type": "concept"})
        ensure_entity({"name": "CountEntity", "type": "concept"})
        entities = get_entities(entity_type="concept")
        count_ent = next(e for e in entities if e["name"] == "CountEntity")
        assert count_ent["observation_count"] == 2

    def test_link_memory_entity(self):
        mid = push_memory("entity link test", "user")
        eid = ensure_entity({"name": "LinkedEntity", "type": "concept"})
        link_memory_entity(mid, eid)
        memories = get_memories_for_entity("LinkedEntity")
        assert len(memories) > 0

    def test_get_related_entities(self):
        mid = push_memory("shared memory", "user")
        eid1 = ensure_entity({"name": "EntityA", "type": "concept"})
        eid2 = ensure_entity({"name": "EntityB", "type": "concept"})
        link_memory_entity(mid, eid1)
        link_memory_entity(mid, eid2)
        related = get_related_entities("EntityA")
        assert any(r["name"] == "EntityB" for r in related)

    def test_get_entities_by_type(self):
        ensure_entity({"name": "FileEntity", "type": "file"})
        ensure_entity({"name": "ConceptEntity", "type": "concept"})
        files = get_entities(entity_type="file")
        assert all(e["type"] == "file" for e in files)


class TestStats:
    def test_stats_empty(self):
        s = stats()
        assert s["memories"] == 0
        assert s["entities"] == 0
        assert s["links"] == 0

    def test_stats_after_push(self):
        push_memory("test memory", "user")
        s = stats()
        assert s["memories"] == 1
