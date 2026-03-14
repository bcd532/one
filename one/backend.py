"""Storage backend abstraction layer.

Routes memory operations to either the local SQLite store (default) or
a remote Foundry ontology backend. Both implement the same interface.
"""

from typing import Optional, Protocol


class Backend(Protocol):
    def push_memory(self, raw_text: str, source: str, tm_label: str, regime_tag: str, aif_confidence: float, hdc_vector: Optional[list[float]]) -> str: ...
    def recall(self, query: str, n: int) -> list[dict]: ...
    def recall_context(self, query: str, n: int, max_chars: int, use_gemma: bool) -> str: ...
    def ensure_entity(self, entity: dict) -> None: ...
    def stats(self) -> dict: ...


class SqliteBackend:
    """Local SQLite storage backend. No external services required."""

    def push_memory(self, raw_text, source, tm_label="unclassified", regime_tag="default", aif_confidence=0.0, hdc_vector=None) -> str:
        from . import store
        mid = store.push_memory(raw_text, source, tm_label, regime_tag, aif_confidence, hdc_vector)

        from .entities import extract_entities
        ents = extract_entities(raw_text, source=source)
        for ent in ents:
            eid = store.ensure_entity(ent)
            store.link_memory_entity(mid, eid)

        return mid

    def recall(self, query, n=10):
        from . import store
        return store.recall(query, n)

    def recall_context(self, query, n=10, max_chars=3000, use_gemma=False):
        memories = self.recall(query, n)
        if not memories:
            return ""

        if use_gemma and len(memories) >= 3:
            try:
                from .gemma import condense_memories, is_available
                if is_available():
                    condensed = condense_memories(memories)
                    if condensed:
                        return f"<prior-context source=\"one/local\" condensed=\"true\">\n{condensed}\n</prior-context>"
            except Exception:
                pass

        lines = ["<prior-context source=\"one/local\">"]
        chars = 0
        for m in memories:
            line = f"[{m['source']}|{m['tm_label']}] {m['raw_text']}"
            if chars + len(line) > max_chars:
                break
            lines.append(line)
            chars += len(line)
        lines.append("</prior-context>")
        return "\n".join(lines)

    def ensure_entity(self, entity):
        from . import store
        store.ensure_entity(entity)

    def stats(self):
        from . import store
        return store.stats()


class FoundryBackend:
    """Remote Foundry ontology storage backend."""

    def __init__(self, client):
        self.client = client

    def push_memory(self, raw_text, source, tm_label="unclassified", regime_tag="default", aif_confidence=0.0, hdc_vector=None):
        from .client import push_memory
        push_memory(self.client, raw_text, source, tm_label, regime_tag, aif_confidence, hdc_vector)

        from .entities import extract_entities
        from .entities import ensure_entity as foundry_ensure_entity
        ents = extract_entities(raw_text, source=source)
        for ent in ents:
            foundry_ensure_entity(self.client, ent)

        return ""

    def recall(self, query, n=10):
        from .retrieve import recall
        return recall(self.client, query, n)

    def recall_context(self, query, n=10, max_chars=3000, use_gemma=False):
        from .retrieve import recall_context
        return recall_context(self.client, query, n, max_chars, use_gemma)

    def ensure_entity(self, entity):
        from .entities import ensure_entity as foundry_ensure_entity
        foundry_ensure_entity(self.client, entity)

    def stats(self):
        return {"backend": "foundry", "note": "check AIP console"}


def get_backend(foundry=None) -> Backend:
    """Return the appropriate backend. Uses SQLite by default, Foundry if a client is provided."""
    if foundry is not None:
        return FoundryBackend(foundry)
    return SqliteBackend()
