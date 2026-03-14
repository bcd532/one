"""Memory retrieval pipeline for Foundry-backed storage.

Encodes queries as HDC vectors, performs nearest-neighbor search on the
Foundry ontology, and optionally condenses results via a local LLM before
injecting them as tagged context blocks.
"""

from foundry_sdk_runtime import AllowBetaFeatures
from orion_push_sdk import FoundryClient
from orion_push_sdk.ontology.search._memory_entry_object_type import MemoryEntryObjectType

from .hdc import encode_text_to_list


def recall(
    client: FoundryClient,
    query: str,
    n: int = 10,
) -> list[dict]:
    """Find the N most relevant memories for a query via vector similarity."""
    vec = encode_text_to_list(query)
    mt = MemoryEntryObjectType()

    with AllowBetaFeatures():
        results = client.ontology.objects.MemoryEntry.nearest_neighbors(
            query=vec,
            vector_property=mt.hdc_vector,
            num_neighbors=n,
        ).take(n)

    memories = []
    for r in results:
        memories.append({
            "raw_text": r.raw_text or "",
            "source": r.source or "unknown",
            "tm_label": r.tm_label or "",
            "regime_tag": r.regime_tag or "",
            "aif_confidence": r.aif_confidence or 0.0,
            "timestamp": str(r.timestamp) if r.timestamp else "",
        })
    return memories


def format_raw(memories: list[dict], max_chars: int = 4000) -> str:
    """Format memories as a tagged context block without LLM condensation."""
    if not memories:
        return ""

    lines = ["<prior-context source=\"one/foundry\">"]
    chars = 0

    for m in memories:
        line = f"[{m['source']}|{m['tm_label']}] {m['raw_text']}"
        if chars + len(line) > max_chars:
            break
        lines.append(line)
        chars += len(line)

    lines.append("</prior-context>")
    return "\n".join(lines)


def format_condensed(condensed: str) -> str:
    """Wrap LLM-condensed context in tagged markup."""
    return f"<prior-context source=\"one/foundry\" condensed=\"true\">\n{condensed}\n</prior-context>"


def recall_context(
    client: FoundryClient,
    query: str,
    n: int = 10,
    max_chars: int = 4000,
    use_gemma: bool = True,
) -> str:
    """Full retrieval pipeline: query, recall, optionally condense, and format.

    When Gemma is available and 3+ memories are retrieved, condenses via
    the local LLM. Otherwise falls back to raw tagged injection.
    """
    memories = recall(client, query, n)

    if not memories:
        return ""

    if use_gemma and len(memories) >= 3:
        try:
            from .gemma import condense_memories, is_available

            if is_available():
                condensed = condense_memories(memories)
                if condensed:
                    return format_condensed(condensed)
        except Exception:
            pass

    return format_raw(memories, max_chars)
