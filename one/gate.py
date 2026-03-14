"""Active Inference (AIF) gating for memory write selection.

Determines whether a message is worth persisting by evaluating novelty,
content quality, and information type. Supports cross-session temporal
awareness when a Foundry client is available.
"""

import re
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from .hdc import encode_text, encode_text_to_list, similarity

# ── Config ──────────────────────────────────────────────────────────

GATE_THRESHOLD = 0.3
RECENT_BUFFER_SIZE = 20
REDUNDANCY_THRESHOLD = 0.7  # Cosine above this = too similar to recent buffer
STALE_MINUTES = 10  # Memory older than this is considered stale
FOUNDRY_CHECK_INTERVAL = 3  # Query Foundry every N messages

# ── Noise patterns ──────────────────────────────────────────────────

NOISE_PATTERNS = [
    re.compile(r'^(ok|okay|yes|no|yeah|yep|nah|sure|cool|thanks|ty|thx|k|lol|lmao|bet)[\.\!\?]?$', re.I),
    re.compile(r'^(go ahead|sounds good|do it|lets go|alright|got it)[\.\!\?]?$', re.I),
    re.compile(r'^\s*$'),
]

NOISE_TOOL_RESULTS = [
    "no files found",
    "bash completed with no output",
    "file created successfully",
    "file has been updated successfully",
    "command was manually backgrounded",
]

# Content that should NEVER be stored — privacy/sensitivity filter
REDACT_PATTERNS = [
    re.compile(r'don.?t talk about|do not talk about|don.?t mention|do not mention', re.I),
    re.compile(r'\bpatent\b|\bconfidential\b|\bproprietary\b|\bnda\b', re.I),
    re.compile(r'\bresume\b|\bsalary\b|\binterview\b|\bapplied to\b|\bjob application\b', re.I),
    re.compile(r'\bpassword\b|\bsecret key\b|\bapi.?key\b|\btoken\b.*\beyJ', re.I),
    re.compile(r'\bssn\b|\bsocial security\b|\bcredit card\b', re.I),
]

# ── High-value signals ──────────────────────────────────────────────

DECISION_SIGNALS = [
    re.compile(r'\b(decided?|choosing?|going with|lets? go with|switching to|pivot)', re.I),
    re.compile(r'\b(the plan is|approach is|strategy is|architecture is)', re.I),
    re.compile(r'\b(because|reason is|the issue is|problem is|root cause)', re.I),
    re.compile(r'\b(don\'?t|never|always|must|should not|avoid)\b', re.I),
    re.compile(r'\b(remember|important|critical|key thing|note that)', re.I),
]

PREFERENCE_SIGNALS = [
    re.compile(r'\b(i want|i need|i like|i hate|i prefer|make it|don\'?t make)', re.I),
    re.compile(r'\b(stop|quit|no more|enough|fuck|shit|ass)\b', re.I),
]


class AifGate:
    """Scores messages for storage worthiness using novelty, quality, and temporal signals.

    Optionally accepts a Foundry client for cross-session temporal awareness.
    Without Foundry, falls back to local-only novelty checking.
    """

    def __init__(self, threshold: float = GATE_THRESHOLD, foundry_client=None):
        self.threshold = threshold
        self.foundry = foundry_client
        self._recent_vecs: list[np.ndarray] = []
        self._msg_count = 0
        self._last_foundry_result: Optional[dict] = None

    def score(self, text: str, source: str = "user") -> float:
        """Score a message from 0.0 (noise) to 1.0 (high value)."""
        if self._is_noise(text, source):
            return 0.0

        content_score = self._content_quality(text, source)
        local_novelty = self._local_novelty(text)
        temporal_novelty = self._temporal_novelty(text)
        info_score = self._information_type(text, source)

        # Temporal novelty can boost past local redundancy:
        # a topic mentioned days ago is still worth re-storing.
        novelty = max(local_novelty, temporal_novelty)

        final = (
            0.25 * content_score +
            0.30 * novelty +
            0.45 * info_score
        )

        return min(1.0, max(0.0, final))

    def should_store(self, text: str, source: str = "user") -> tuple[bool, float]:
        """Returns (should_store, confidence)."""
        self._msg_count += 1
        s = self.score(text, source)
        store = s >= self.threshold

        if store:
            vec = encode_text(text)
            self._recent_vecs.append(vec)
            if len(self._recent_vecs) > RECENT_BUFFER_SIZE:
                self._recent_vecs = self._recent_vecs[-RECENT_BUFFER_SIZE:]

        return store, s

    # ── Hard noise filter ───────────────────────────────────────────

    def _is_noise(self, text: str, source: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 4:
            return True
        for pattern in NOISE_PATTERNS:
            if pattern.match(stripped):
                return True
        if source == "tool_result":
            lower = stripped.lower()
            for noise in NOISE_TOOL_RESULTS:
                if noise in lower:
                    return True
        # Redact sensitive content — never store
        for pattern in REDACT_PATTERNS:
            if pattern.search(stripped):
                return True
        return False

    # ── Content quality ─────────────────────────────────────────────

    def _content_quality(self, text: str, source: str) -> float:
        if source == "tool_result":
            if len(text) > 500:
                return 0.1
            return 0.3
        if source == "tool_use":
            return 0.4

        base = {"user": 0.7, "assistant": 0.6, "system": 0.5}.get(source, 0.3)
        length = len(text.strip())

        if length < 10:
            return base * 0.3
        elif length < 30:
            return base * 0.6
        elif length < 100:
            return base * 0.8
        return base

    # ── Local novelty (same-session buffer) ─────────────────────────

    def _local_novelty(self, text: str) -> float:
        if not self._recent_vecs:
            return 1.0

        vec = encode_text(text)
        max_sim = max(similarity(vec, rv) for rv in self._recent_vecs)

        if max_sim > REDUNDANCY_THRESHOLD:
            return 0.0
        return 1.0 - max_sim

    # ── Temporal novelty (cross-session via Foundry) ────────────────

    def _temporal_novelty(self, text: str) -> float:
        """Evaluate temporal novelty against stored memories.

        Returns high scores for truly novel content or content that matches
        old memories (indicating a recurring topic worth re-storing), and low
        scores for recently stored duplicates.
        """
        if not self.foundry:
            return 0.5

        if self._msg_count % FOUNDRY_CHECK_INTERVAL != 0:
            return 0.5

        try:
            match = self._query_foundry_nearest(text)
            if not match:
                return 1.0

            match_time = match.get("timestamp")
            if not match_time:
                return 0.5

            if isinstance(match_time, str):
                try:
                    match_dt = datetime.fromisoformat(match_time.replace("Z", "+00:00"))
                except ValueError:
                    return 0.5
            elif isinstance(match_time, datetime):
                match_dt = match_time
            else:
                return 0.5

            now = datetime.now(timezone.utc)
            age = now - match_dt

            if age < timedelta(minutes=STALE_MINUTES):
                return 0.1
            elif age < timedelta(hours=1):
                return 0.5
            elif age < timedelta(days=1):
                return 0.7
            else:
                return 0.9

        except Exception:
            return 0.5

    def _query_foundry_nearest(self, text: str) -> Optional[dict]:
        """Find the most similar existing memory in Foundry. Times out after 3 seconds."""
        import concurrent.futures

        def _query():
            from foundry_sdk_runtime import AllowBetaFeatures
            from orion_push_sdk.ontology.search._memory_entry_object_type import MemoryEntryObjectType

            vec = encode_text_to_list(text)
            mt = MemoryEntryObjectType()

            with AllowBetaFeatures():
                results = self.foundry.ontology.objects.MemoryEntry.nearest_neighbors(
                    query=vec,
                    vector_property=mt.hdc_vector,
                    num_neighbors=1,
                ).take(1)

            if results:
                r = results[0]
                return {
                    "raw_text": r.raw_text or "",
                    "timestamp": r.timestamp,
                    "source": r.source or "",
                    "tm_label": r.tm_label or "",
                }
            return None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(_query).result(timeout=3)
        except (concurrent.futures.TimeoutError, Exception):
            return None
        except Exception:
            return None

    # ── Information type scoring ────────────────────────────────────

    def _information_type(self, text: str, source: str) -> float:
        score = 0.0

        for pattern in DECISION_SIGNALS:
            if pattern.search(text):
                score = max(score, 0.9)
                break

        for pattern in PREFERENCE_SIGNALS:
            if pattern.search(text):
                score = max(score, 0.8)
                break

        from .entities import extract_entities
        entities = extract_entities(text, source=source)
        entity_bonus = min(0.5, len(entities) * 0.15)
        score = max(score, entity_bonus)

        if re.search(r'/[\w./-]{3,}', text):
            score = max(score, 0.5)

        if re.search(r'```|def |class |function |import |from ', text):
            score = max(score, 0.6)

        if source == "user" and text.strip().endswith("?"):
            score = max(score, 0.5)

        if score == 0.0 and len(text.strip()) > 20:
            score = 0.3

        return score
