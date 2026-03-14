"""Entity extraction from conversation text.

Identifies structured entities (file paths, technical concepts, tools,
projects) and optionally persists them to Foundry with memory linkage.
"""

import re
from typing import Optional
from datetime import datetime, timezone

from orion_push_sdk import FoundryClient
from foundry_sdk_runtime.ontology_edit import ObjectLocator


# ── Extraction patterns ─────────────────────────────────────────────

FILE_PATTERN = re.compile(
    r'(?:^|[\s\'"(])(/[\w./-]{3,80}|\.{1,2}/[\w./-]{2,80}|~/[\w./-]{2,80})',
)

CONCEPT_KEYWORDS = {
    # HDC
    "hdc": "HDC", "hyperdimensional": "HDC", "hypervector": "HDC",
    "trigram": "HDC", "codebook": "HDC", "bind": "HDC", "bundle": "HDC",
    # TM
    "tsetlin": "Tsetlin Machine", "clause": "Tsetlin Machine",
    "boolean logic": "Tsetlin Machine", "tm ": "Tsetlin Machine",
    # AIF
    "active inference": "Active Inference", "free energy": "Active Inference",
    "aif": "Active Inference", "variational": "Active Inference",
    # Infrastructure
    "foundry": "Palantir Foundry", "ontology": "Palantir Ontology",
    "palantir": "Palantir AIP", "aip": "Palantir AIP",
    # General
    "vector": "Vector Encoding", "encoding": "Vector Encoding",
    "retrieval": "Memory Retrieval", "recall": "Memory Retrieval",
    "entity": "Entity Extraction",
}

TOOL_NAMES = {
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebSearch", "WebFetch", "Agent", "Skill",
}


def extract_entities(text: str, source: str = "user") -> list[dict]:
    """Extract entities from text. Returns list of {type, name, id}."""
    entities = []
    lower = text.lower()

    # File paths
    for match in FILE_PATTERN.finditer(text):
        path = match.group(1).rstrip(".,;:)")
        eid = f"file:{path}"
        entities.append({"type": "file", "name": path, "id": eid})

    # Concepts
    seen_concepts = set()
    for keyword, concept in CONCEPT_KEYWORDS.items():
        if keyword in lower and concept not in seen_concepts:
            seen_concepts.add(concept)
            eid = f"concept:{concept.lower().replace(' ', '_')}"
            entities.append({"type": "concept", "name": concept, "id": eid})

    # Tools (from structured tool_use messages)
    if source == "tool_use":
        try:
            import json
            d = json.loads(text)
            tool = d.get("tool", "")
            if tool in TOOL_NAMES:
                eid = f"tool:{tool.lower()}"
                entities.append({"type": "tool", "name": tool, "id": eid})
        except (json.JSONDecodeError, AttributeError):
            pass

    return entities


def extract_from_tool_call(tool_name: str, tool_input: dict) -> list[dict]:
    """Extract entities from a tool call's name and input parameters."""
    entities = []

    if tool_name in TOOL_NAMES:
        entities.append({
            "type": "tool",
            "name": tool_name,
            "id": f"tool:{tool_name.lower()}",
        })

    for key in ("file_path", "path"):
        if key in tool_input and isinstance(tool_input[key], str):
            path = tool_input[key]
            entities.append({
                "type": "file",
                "name": path,
                "id": f"file:{path}",
            })

    if tool_name in ("Glob", "Grep") and "pattern" in tool_input:
        pattern = tool_input["pattern"]
        entities.append({
            "type": "pattern",
            "name": pattern,
            "id": f"pattern:{pattern}",
        })

    return entities


# ── Foundry push ────────────────────────────────────────────────────

_entity_cache: set[str] = set()


def ensure_entity(client: FoundryClient, entity: dict) -> None:
    """Create an entity in Foundry if it does not already exist. Results are cached."""
    cache_key = f"{entity['type']}:{entity['name']}"
    if cache_key in _entity_cache:
        return

    now = datetime.now(timezone.utc).isoformat()

    try:
        from orion_push_sdk.ontology.search._entity_object_type import EntityObjectType
        et = EntityObjectType()
        existing = client.ontology.objects.Entity.where(
            et.name == entity["name"]
        ).take(1)
        if existing:
            _entity_cache.add(cache_key)
            return
    except Exception:
        pass

    try:
        client.ontology.actions.create_entity(
            name=entity["name"],
            type=entity["type"],
            first_seen=now,
            last_seen=now,
            observation_count="1",
            hdc_vector="",
        )
        _entity_cache.add(cache_key)
    except Exception:
        pass


def link_memory_to_entities(
    client: FoundryClient,
    memory_entry_id: str,
    entities: list[dict],
) -> None:
    """Link a MemoryEntry to its extracted entities via ontology edits."""
    if not entities:
        return

    try:
        edits = client.ontology.edits()
        for entity in entities:
            ensure_entity(client, entity)
            edits.add_link(
                "hasMemoryEntries",
                source=ObjectLocator(object_type="Entity", primary_key=entity["id"]),
                target=ObjectLocator(object_type="MemoryEntry", primary_key=memory_entry_id),
            )
        client.ontology.edits.apply(edits)
    except Exception:
        pass
