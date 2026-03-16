"""Entity extraction from conversation text.

Identifies structured entities (file paths, technical concepts, tools,
projects, people, organizations, methods, code patterns) and optionally
persists them to Foundry with memory linkage.

v2: Expanded entity types, smarter extraction, relationship inference.
"""

from __future__ import annotations

import re
from typing import Optional
from datetime import datetime, timezone


# ── Extraction patterns ─────────────────────────────────────────────

FILE_PATTERN = re.compile(
    r'(?:^|[\s\'"(])(/[\w./-]{3,80}|\.{1,2}/[\w./-]{2,80}|~/[\w./-]{2,80})',
)

URL_PATTERN = re.compile(
    r'https?://[^\s<>\'")\]]+',
)

# Technical concept keywords → canonical names
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
    # ML/AI
    "neural network": "Neural Networks", "deep learning": "Deep Learning",
    "transformer": "Transformer", "attention": "Attention Mechanism",
    "llm": "Large Language Model", "language model": "Large Language Model",
    "reinforcement learning": "Reinforcement Learning",
    "gradient descent": "Gradient Descent",
    # Biology/Chemistry
    "protein": "Protein", "gene": "Gene", "dna": "DNA", "rna": "RNA",
    "enzyme": "Enzyme", "receptor": "Receptor", "pathway": "Pathway",
    "mutation": "Mutation", "expression": "Gene Expression",
    # Programming
    "api": "API", "rest": "REST API", "graphql": "GraphQL",
    "database": "Database", "sql": "SQL", "nosql": "NoSQL",
    "microservice": "Microservices", "kubernetes": "Kubernetes",
    "docker": "Docker", "container": "Containerization",
}

# Code pattern detection
CODE_PATTERNS = {
    re.compile(r'\bclass\s+([A-Z]\w+)'): "class",
    re.compile(r'\bdef\s+(\w+)\s*\('): "function",
    re.compile(r'\bimport\s+(\w+)'): "module",
    re.compile(r'\bfrom\s+(\w+)\s+import'): "module",
}

TOOL_NAMES = {
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebSearch", "WebFetch", "Agent", "Skill",
}

# Method/technique patterns
METHOD_PATTERNS = [
    re.compile(r'\b((?:Monte Carlo|Bayesian|Markov chain|gradient|stochastic|genetic)\s+\w+)', re.I),
    re.compile(r'\b(k-means|random forest|decision tree|neural net\w*|SVM|logistic regression)\b', re.I),
    re.compile(r'\b([A-Z]{2,5}(?:-\d+)?)\s+(?:algorithm|method|approach|technique)\b'),
]

# Person detection (rough heuristic — proper names)
PERSON_PATTERN = re.compile(
    r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
)

# Organization detection
ORG_KEYWORDS = {
    "google": "Google", "openai": "OpenAI", "anthropic": "Anthropic",
    "meta": "Meta", "microsoft": "Microsoft", "nvidia": "NVIDIA",
    "deepmind": "DeepMind", "hugging face": "Hugging Face",
    "stanford": "Stanford", "mit": "MIT", "berkeley": "UC Berkeley",
}


def extract_entities(text: str, source: str = "user") -> list[dict]:
    """Extract entities from text. Returns list of {type, name, id}.

    v2: Extracts files, concepts, tools, code patterns, methods,
    URLs, organizations, and rough person mentions.
    """
    entities = []
    lower = text.lower()
    seen_ids = set()

    def _add(etype: str, name: str):
        eid = f"{etype}:{name.lower().replace(' ', '_')}"
        if eid not in seen_ids:
            seen_ids.add(eid)
            entities.append({"type": etype, "name": name, "id": eid})

    # File paths (filter out venvs, caches, and false positives)
    _GARBAGE_PATH_PARTS = {
        ".venv", "site-packages", "__pycache__", "node_modules",
        ".git", ".tox", ".eggs", ".mypy_cache", ".pytest_cache",
    }
    for match in FILE_PATTERN.finditer(text):
        path = match.group(1).rstrip(".,;:)")
        # Skip slash commands misidentified as paths
        if path.startswith("/") and "/" not in path[1:] and len(path) < 20:
            continue
        # Skip garbage paths
        if any(part in path for part in _GARBAGE_PATH_PARTS):
            continue
        _add("file", path)

    # URLs
    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(".,;:)")
        _add("url", url)

    # Concepts
    for keyword, concept in CONCEPT_KEYWORDS.items():
        if keyword in lower:
            _add("concept", concept)

    # Code patterns (classes, functions, modules)
    for pattern, code_type in CODE_PATTERNS.items():
        for match in pattern.finditer(text):
            name = match.group(1)
            if len(name) > 1 and name not in {"self", "cls", "os", "re", "io", "sys"}:
                _add(code_type, name)

    # Methods/techniques
    for pattern in METHOD_PATTERNS:
        for match in pattern.finditer(text):
            method_name = match.group(1).strip()
            if len(method_name) > 3:
                _add("method", method_name)

    # Organizations
    for keyword, org in ORG_KEYWORDS.items():
        if keyword in lower:
            _add("organization", org)

    # Tools (from structured tool_use messages)
    if source == "tool_use":
        try:
            import json as _json
            d = _json.loads(text)
            tool = d.get("tool", "")
            if tool in TOOL_NAMES:
                _add("tool", tool)
        except (ValueError, AttributeError):
            pass

    # Tool names mentioned in regular text
    for tool in TOOL_NAMES:
        if tool in text:
            _add("tool", tool)

    return entities


def extract_from_tool_call(tool_name: str, tool_input: dict) -> list[dict]:
    """Extract entities from a tool call's name and input parameters."""
    entities = []
    seen_ids = set()

    def _add(etype: str, name: str):
        eid = f"{etype}:{name.lower().replace(' ', '_')}"
        if eid not in seen_ids:
            seen_ids.add(eid)
            entities.append({"type": etype, "name": name, "id": eid})

    if tool_name in TOOL_NAMES:
        _add("tool", tool_name)

    for key in ("file_path", "path"):
        if key in tool_input and isinstance(tool_input[key], str):
            path = tool_input[key]
            _add("file", path)

    if tool_name in ("Glob", "Grep") and "pattern" in tool_input:
        pattern = tool_input["pattern"]
        _add("pattern", pattern)

    if tool_name == "WebSearch" and "query" in tool_input:
        query = tool_input["query"]
        # Extract concepts from search queries
        for ent in extract_entities(query, source="user"):
            if ent["id"] not in seen_ids:
                seen_ids.add(ent["id"])
                entities.append(ent)

    return entities


# ── Relationship inference ─────────────────────────────────────────


RELATIONSHIP_PATTERNS = [
    (re.compile(r'(\w+)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w+)', re.I), "causes"),
    (re.compile(r'(\w+)\s+(?:contradicts?|opposes?|conflicts?\s+with)\s+(\w+)', re.I), "contradicts"),
    (re.compile(r'(\w+)\s+(?:is\s+(?:a\s+)?(?:part|component|subset)\s+of)\s+(\w+)', re.I), "component_of"),
    (re.compile(r'(\w+)\s+(?:supersedes?|replaces?|obsoletes?)\s+(\w+)', re.I), "supersedes"),
    (re.compile(r'(\w+)\s+(?:is\s+analogous\s+to|is\s+similar\s+to|resembles?)\s+(\w+)', re.I), "analogous_to"),
    (re.compile(r'(\w+)\s+(?:depends?\s+on|requires?|needs?)\s+(\w+)', re.I), "depends_on"),
    (re.compile(r'(\w+)\s+(?:enables?|supports?|facilitates?)\s+(\w+)', re.I), "enables"),
]


def extract_relationships(text: str) -> list[dict]:
    """Extract typed relationships between entities from text.

    Returns list of {source, target, relation_type} dicts.
    """
    relationships = []

    for pattern, rel_type in RELATIONSHIP_PATTERNS:
        for match in pattern.finditer(text):
            source = match.group(1).strip()
            target = match.group(2).strip()
            if len(source) > 1 and len(target) > 1:
                relationships.append({
                    "source": source,
                    "target": target,
                    "relation_type": rel_type,
                })

    return relationships


# ── Foundry push ────────────────────────────────────────────────────

_entity_cache: set[str] = set()


def ensure_entity(client, entity: dict) -> None:
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
            hdc_vector="[]",
        )
        _entity_cache.add(cache_key)
    except Exception:
        pass


def link_memory_to_entities(
    client,
    memory_entry_id: str,
    entities: list[dict],
) -> None:
    """Link a MemoryEntry to its extracted entities via ontology edits."""
    if not entities:
        return

    try:
        from foundry_sdk_runtime.ontology_edit import ObjectLocator

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
