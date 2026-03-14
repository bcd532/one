"""Foundry client for pushing and pulling ontology objects.

Manages MemoryEntry and Entity objects on a Palantir AIP Foundry instance.
Authentication is configured via environment variables or local config files.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from orion_push_sdk import FoundryClient, UserTokenAuth

VECTOR_DIM = 4096


def get_client() -> FoundryClient:
    """Create a Foundry client from environment variables or ~/.one/config."""
    hostname = os.environ.get("ONE_FOUNDRY_HOST", "")
    token = os.environ.get("FOUNDRY_TOKEN", "")

    config_file = os.path.expanduser("~/.one/config")
    token_file = os.path.expanduser("~/.one/token")

    if not hostname and os.path.exists(config_file):
        for line in open(config_file):
            line = line.strip()
            if line.startswith("host="):
                hostname = line.split("=", 1)[1].strip()

    if not token:
        if os.path.exists(token_file):
            with open(token_file) as f:
                token = f.read().strip()

    if not hostname:
        raise RuntimeError("no host — set ONE_FOUNDRY_HOST or add host= to ~/.one/config")
    if not token:
        raise RuntimeError("no token — set FOUNDRY_TOKEN or write to ~/.one/token")

    return FoundryClient(
        auth=UserTokenAuth(token=token),
        hostname=hostname,
    )


def push_memory(
    client: FoundryClient,
    raw_text: str,
    source: str,
    tm_label: str = "unclassified",
    regime_tag: str = "default",
    aif_confidence: float = 0.0,
    hdc_vector: Optional[list[float]] = None,
) -> None:
    """Push a MemoryEntry to Foundry.

    If no hdc_vector is provided, the raw text is encoded automatically
    using the HDC trigram encoder.
    """
    if hdc_vector is None:
        from .hdc import encode_tagged
        vec = encode_tagged(raw_text, role=source)
        hdc_vector = vec.tolist()

    client.ontology.actions.create_memory_entry(
        timestamp=datetime.now(timezone.utc),
        raw_text=raw_text,
        source=source,
        tm_label=tm_label,
        regime_tag=regime_tag,
        aif_confidence=aif_confidence,
        hdc_vector=hdc_vector,
    )


def push_entity(
    client: FoundryClient,
    entity_id: str,
    name: str,
    entity_type: str = "unknown",
) -> None:
    """Push an Entity to Foundry."""
    now = datetime.now(timezone.utc).isoformat()
    client.ontology.actions.create_entity(
        entity_id=entity_id,
        name=name,
        type=entity_type,
        first_seen=now,
        last_seen=now,
        observation_count="1",
    )
