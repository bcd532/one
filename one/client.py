"""Foundry client with automatic token lifecycle management.

Reads token from ~/.one/token, tracks expiry from the JWT, warns before
expiration, and prompts for refresh when needed. Falls back gracefully
when Foundry is unreachable.
"""

from __future__ import annotations

import os
import base64
import json
import time
import threading
from datetime import datetime, timezone
from typing import Any, Optional

VECTOR_DIM = 4096
TOKEN_FILE = os.path.expanduser("~/.one/token")
CONFIG_FILE = os.path.expanduser("~/.one/config")
TOKEN_WARNING_SECONDS = 3600  # warn 1 hour before expiry
TOKEN_CHECK_INTERVAL = 300    # check every 5 minutes

_client_cache = None
_token_expiry = 0
_token_monitor_started = False


def _decode_jwt_expiry(token: str) -> float:
    """Extract expiration timestamp from a JWT without verification."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return 0
        payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
        data = json.loads(base64.b64decode(payload))
        return float(data.get("exp", 0))
    except Exception:
        return 0


def _get_hostname() -> str:
    hostname = os.environ.get("ONE_FOUNDRY_HOST", "")
    if not hostname and os.path.exists(CONFIG_FILE):
        for line in open(CONFIG_FILE):
            line = line.strip()
            if line.startswith("host="):
                hostname = line.split("=", 1)[1].strip()
    return hostname


def _get_token() -> str:
    token = os.environ.get("FOUNDRY_TOKEN", "")
    if not token and os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            token = f.read().strip()
    return token


def token_time_remaining() -> float:
    """Seconds until the current token expires. Negative = already expired."""
    global _token_expiry
    if _token_expiry == 0:
        token = _get_token()
        if token:
            _token_expiry = _decode_jwt_expiry(token)
    if _token_expiry == 0:
        return -1
    return _token_expiry - time.time()


def token_status() -> dict:
    """Return token status: remaining time, expired flag, warning flag."""
    remaining = token_time_remaining()
    return {
        "remaining_seconds": remaining,
        "remaining_human": _format_duration(remaining),
        "expired": remaining < 0,
        "warning": 0 < remaining < TOKEN_WARNING_SECONDS,
        "ok": remaining > TOKEN_WARNING_SECONDS,
    }


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return "expired"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def refresh_token(new_token: str) -> None:
    """Write a new token to disk and reset the client cache."""
    global _client_cache, _token_expiry

    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        f.write(new_token.strip())
    os.chmod(TOKEN_FILE, 0o600)

    _token_expiry = _decode_jwt_expiry(new_token.strip())
    _client_cache = None  # force re-creation on next get_client()


def get_client() -> Any:
    """Create or return a cached Foundry client.

    Checks token validity before returning. Raises RuntimeError if
    token is expired or missing.
    """
    global _client_cache, _token_expiry, _token_monitor_started

    hostname = _get_hostname()
    token = _get_token()

    if not hostname:
        raise RuntimeError("no host — set ONE_FOUNDRY_HOST or add host= to ~/.one/config")
    if not token:
        raise RuntimeError("no token — set FOUNDRY_TOKEN or write to ~/.one/token")

    # Check expiry
    _token_expiry = _decode_jwt_expiry(token)
    remaining = _token_expiry - time.time() if _token_expiry > 0 else -1

    if remaining < 0 and _token_expiry > 0:
        _client_cache = None
        raise RuntimeError(f"token expired {_format_duration(-remaining)} ago — refresh ~/.one/token")

    # Return cached client if token hasn't changed
    if _client_cache is not None:
        return _client_cache

    from orion_push_sdk import FoundryClient, UserTokenAuth

    _client_cache = FoundryClient(
        auth=UserTokenAuth(token=token),
        hostname=hostname,
    )

    # Start background monitor
    if not _token_monitor_started:
        _token_monitor_started = True
        threading.Thread(target=_token_monitor, daemon=True).start()

    return _client_cache


def _token_monitor() -> None:
    """Background thread that monitors token expiry and logs warnings."""
    while True:
        time.sleep(TOKEN_CHECK_INTERVAL)
        status = token_status()

        if status["expired"]:
            # Invalidate cached client
            global _client_cache
            _client_cache = None

        # The app's status bar reads token_status() to show remaining time


def push_memory(
    client,
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
    client,
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
