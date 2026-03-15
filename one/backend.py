"""Storage backend abstraction layer.

Routes memory and rule operations to either the local SQLite store (default)
or a remote Foundry ontology backend. Both implement the same interface.
"""

from typing import Optional, Protocol


class Backend(Protocol):
    def push_memory(self, raw_text: str, source: str, tm_label: str, regime_tag: str, aif_confidence: float, hdc_vector: Optional[list[float]]) -> str: ...
    def recall(self, query: str, n: int) -> list[dict]: ...
    def recall_context(self, query: str, n: int, max_chars: int, use_gemma: bool) -> str: ...
    def ensure_entity(self, entity: dict) -> None: ...
    def push_rule(self, project: str, rule_text: str, activation_keywords: str, parent_label: str, confidence: float, source_count: int) -> None: ...
    def get_rules(self, project: str) -> list[dict]: ...
    def stats(self) -> dict: ...


class SqliteBackend:
    """Local SQLite storage backend."""

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

    def push_rule(self, project, rule_text, activation_keywords="*", parent_label="", confidence=0.5, source_count=1):
        from .rules import learn_rule_from_memory
        learn_rule_from_memory(project, rule_text, confidence, activation_keywords or None)

    def get_rules(self, project):
        from .rules import get_all_rules
        return get_all_rules(project)

    def stats(self):
        from . import store
        return store.stats()


class FoundryBackend:
    """Remote Foundry ontology storage backend.

    Rules are stored as MemoryEntry objects with source='rule' and
    tree structure encoded in tm_label (prefix hierarchy) and
    activation keywords in regime_tag.
    """

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

    def push_rule(self, project, rule_text, activation_keywords="*", parent_label="", confidence=0.5, source_count=1):
        """Store a rule as a MemoryEntry with source='rule'.

        Tree hierarchy via tm_label prefix: rule:core, rule:hdc, rule:hdc:codebook
        Activation keywords in regime_tag field.
        Confidence in aif_confidence field.
        Rule entity linked via hasMemoryEntries.
        """
        from .client import push_memory
        from .hdc import encode_tagged

        # Build the tm_label for tree hierarchy
        label = f"rule:{parent_label}" if parent_label else "rule:core"

        vec = encode_tagged(rule_text, role="rule")

        # Store as MemoryEntry
        push_memory(
            self.client,
            raw_text=rule_text,
            source="rule",
            tm_label=label,
            regime_tag=activation_keywords,
            aif_confidence=confidence,
            hdc_vector=vec.tolist(),
        )

        # Create a rule entity and link it
        from .entities import ensure_entity as foundry_ensure_entity
        rule_entity = {
            "name": f"rule:{rule_text[:50]}",
            "type": "rule",
            "id": f"rule:{rule_text[:50]}",
        }
        foundry_ensure_entity(self.client, rule_entity)

        # Also store in local SQLite for fast per-turn activation
        try:
            from .rules import learn_rule_from_memory
            learn_rule_from_memory(project, rule_text, confidence, activation_keywords or None)
        except Exception:
            pass

    def get_rules(self, project):
        """Get rules from local SQLite (fast) for per-turn activation.
        Foundry is the persistent store, SQLite is the hot cache.
        """
        from .rules import get_all_rules
        return get_all_rules(project)

    def sync_rules_from_foundry(self, project):
        """Pull rules from Foundry into local SQLite cache.
        Called on session start to hydrate the rule tree.
        """
        try:
            from foundry_sdk_runtime import AllowBetaFeatures
            from orion_push_sdk.ontology.search._memory_entry_object_type import MemoryEntryObjectType

            mt = MemoryEntryObjectType()

            with AllowBetaFeatures():
                results = self.client.ontology.objects.MemoryEntry.where(
                    mt.source == "rule"
                ).take(200)

            from .rules import add_rule, find_matching_rule, init_rules_schema
            init_rules_schema()

            synced = 0
            for r in results:
                text = r.raw_text or ""
                label = r.tm_label or "rule:core"
                keywords = r.regime_tag or "*"
                conf = r.aif_confidence or 0.5

                # Skip if already in local cache
                if find_matching_rule(project, text, threshold=0.7):
                    continue

                # Parse parent from label: rule:hdc:codebook → parent is rule:hdc
                parts = label.split(":")
                parent_label = ":".join(parts[:-1]) if len(parts) > 2 else ""

                add_rule(
                    project=project,
                    rule_text=text,
                    activation_keywords=keywords,
                    confidence=conf,
                )
                synced += 1

            return synced
        except Exception:
            return 0

    def stats(self):
        return {"backend": "foundry"}


def get_backend(foundry=None) -> Backend:
    """Return the appropriate backend."""
    if foundry is not None:
        return FoundryBackend(foundry)
    return SqliteBackend()
