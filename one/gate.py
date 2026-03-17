"""Active Inference (AIF) gating for memory write selection.

Implements a proper Active Inference gate using Karl Friston's free energy
framework. The gate maintains a generative model of expected observations
in HDC vector space and decides whether to store messages based on
variational free energy.

Key concepts:
- Generative model: mixture of learned regimes (topic clusters) in HDC space
- Surprise: prediction error — how far a message deviates from expected
- Precision: learned confidence per regime — inverse variance of cluster
- Free energy: F = precision * surprise - log_prior
  High free energy = genuinely informative = worth storing
- Expected free energy: epistemic value of storing (will it reduce future surprise?)
- Belief updating: regime parameters update online after each observation

The gate retains hard noise/redaction filters as pre-processing (these handle
classification that vector similarity cannot) and content heuristics as
likelihood priors (different source types have different base informativeness).
"""

import re
import time
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional

from .hdc import encode_text, encode_text_to_list, similarity, DIM

# ── Config ──────────────────────────────────────────────────────────

GATE_THRESHOLD = 0.3
RECENT_BUFFER_SIZE = 20
REDUNDANCY_THRESHOLD = 0.7  # Cosine above this = too similar to recent buffer
STALE_MINUTES = 10  # Memory older than this is considered stale
FOUNDRY_CHECK_INTERVAL = 3  # Query Foundry every N messages

# AIF parameters
MAX_REGIMES = 16           # Maximum number of learned topic clusters
REGIME_MERGE_THRESHOLD = 0.8  # Cosine similarity above this merges regimes
INITIAL_PRECISION = 1.0    # Starting precision for new regimes
PRECISION_LEARNING_RATE = 0.05  # How fast precision adapts
PRIOR_DECAY = 0.995        # Regime priors decay toward uniform over time
MIN_PRECISION = 0.1        # Floor — prevents division by zero
MAX_PRECISION = 20.0       # Ceiling — prevents over-concentration
SURPRISE_SCALE = 2.0       # Scales raw surprise into [0, 1] range
EPISTEMIC_WEIGHT = 0.3     # Weight of expected free energy (epistemic value)
CONTENT_PRIOR_WEIGHT = 0.25  # Weight of content-type prior in final score

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


# ── Regime (topic cluster) ──────────────────────────────────────────

class Regime:
    """A learned topic cluster in HDC vector space.

    Each regime is a Gaussian-like belief about expected observations:
    - mu: mean vector (centroid of the cluster)
    - precision: inverse variance (how tight the cluster is)
    - prior: probability of being in this regime (updated via observation counts)
    - count: number of observations assigned to this regime
    """

    __slots__ = ("mu", "precision", "prior", "count", "_sum_sq_error")

    def __init__(self, mu: np.ndarray, precision: float = INITIAL_PRECISION):
        self.mu = mu.copy()
        self.precision = precision
        self.prior = 1.0  # Will be normalized across all regimes
        self.count = 1
        self._sum_sq_error = 0.0  # Running sum of squared prediction errors

    def surprise(self, vec: np.ndarray) -> float:
        """Prediction error: how far is this observation from the regime mean?

        Uses (1 - cosine_similarity) as the distance metric, which ranges
        from 0 (identical) to 2 (opposite). For normalized HDC vectors,
        this is proportional to squared Euclidean distance.
        """
        return 1.0 - float(np.dot(self.mu, vec))

    def free_energy(self, vec: np.ndarray) -> float:
        """Variational free energy for this observation under this regime.

        F = precision * surprise - log(prior)

        High precision + high surprise = very informative (model expected
        something specific and got something different).
        Low precision + high surprise = uncertain regime, surprise is expected.
        """
        s = self.surprise(vec)
        return self.precision * s - math.log(max(self.prior, 1e-10))

    def update(self, vec: np.ndarray, learning_rate: float = 0.1):
        """Bayesian belief update: shift mean toward observation, update precision.

        The mean moves toward the observation proportional to learning rate.
        Precision is updated from running prediction error statistics:
        high variance → low precision, low variance → high precision.
        """
        error = self.surprise(vec)

        # Update running error statistics (exponential moving average)
        self._sum_sq_error = (1 - PRECISION_LEARNING_RATE) * self._sum_sq_error + \
                             PRECISION_LEARNING_RATE * (error ** 2)

        # Precision = 1 / variance, clamped to safe range
        variance = max(self._sum_sq_error, 1e-6)
        self.precision = min(MAX_PRECISION, max(MIN_PRECISION, 1.0 / variance))

        # Move centroid toward observation
        rate = learning_rate / (1 + math.log1p(self.count))  # Decay with experience
        self.mu = self.mu + rate * (vec - self.mu)
        norm = np.linalg.norm(self.mu)
        if norm > 1e-10:
            self.mu = self.mu / norm

        self.count += 1

    def epistemic_value(self, vec: np.ndarray) -> float:
        """Expected free energy: how much would storing this reduce future surprise?

        Messages far from the centroid have high epistemic value — they would
        shift the regime's beliefs significantly. Messages near the centroid
        are already well-modeled and have low epistemic value.

        This is the information gain: D_KL[q(s|o,a=store) || q(s|a=~store)]
        approximated as the magnitude of the belief update that would occur.
        """
        error = self.surprise(vec)
        # Magnitude of the centroid shift that would occur
        rate = 0.1 / (1 + math.log1p(self.count))
        shift_magnitude = rate * error
        # Weighted by precision — shifting a precise belief is more informative
        return shift_magnitude * self.precision


class AifGate:
    """Active Inference gate for memory write selection.

    Maintains a generative model of expected messages as a mixture of
    learned regimes (topic clusters) in HDC vector space. Computes
    variational free energy to determine whether a message is genuinely
    informative and worth persisting.

    The decision to store is based on:
    1. Free energy (surprise weighted by precision and prior)
    2. Expected free energy (epistemic value — will storing reduce future surprise?)
    3. Content prior (source type and information structure)
    4. Epistemic safety (LLM overconfidence detection)

    Hard filters (noise, redaction, rage) run first and override everything.
    """

    def __init__(self, threshold: float = GATE_THRESHOLD, foundry_client=None):
        self.threshold = threshold
        self.foundry = foundry_client
        self._regimes: list[Regime] = []
        self._recent_vecs: list[np.ndarray] = []
        self._msg_count = 0
        self._total_observations = 0
        self._last_foundry_result: Optional[dict] = None

    def score(self, text: str, source: str = "user") -> float:
        """Score a message from 0.0 (noise) to 1.0 (high value).

        Pipeline:
        1. Hard noise filter (regex, length, redaction) → 0.0
        2. Encode text to HDC vector
        3. Compute variational free energy against generative model
        4. Compute expected free energy (epistemic value)
        5. Compute content prior (source type, info structure)
        6. Combine via precision-weighted mixture
        7. Apply epistemic safety check for LLM sources
        """
        if self._is_noise(text, source):
            return 0.0

        vec = encode_text(text)

        # Variational free energy: how surprising is this under the model?
        vfe = self._variational_free_energy(vec)

        # Expected free energy: epistemic value of storing this
        efe = self._expected_free_energy(vec)

        # Content prior: structural informativeness heuristic
        content_prior = self._content_prior(text, source)

        # Temporal novelty (cross-session via Foundry)
        temporal = self._temporal_novelty(text)

        # Combine: free energy drives the score, epistemic value and
        # content prior modulate it
        score = (
            (1.0 - EPISTEMIC_WEIGHT - CONTENT_PRIOR_WEIGHT) * vfe +
            EPISTEMIC_WEIGHT * efe +
            CONTENT_PRIOR_WEIGHT * content_prior
        )

        # Temporal modulation: recently stored duplicates get suppressed
        score *= temporal

        # Epistemic safety: penalize overconfident LLM output
        if source in ("assistant", "research", "synthesis", "dialectic"):
            from .epistemic_safety import detect_false_certainty
            false_cert = detect_false_certainty(text)
            if false_cert:
                score *= 0.6

        return min(1.0, max(0.0, score))

    def should_store(self, text: str, source: str = "user") -> tuple[bool, float]:
        """Evaluate and update: score the message, then update the generative model.

        Returns (should_store, score). If stored, the generative model is
        updated with the observation — this is the "action" in active inference,
        where storing changes future predictions.
        """
        self._msg_count += 1
        self._total_observations += 1
        s = self.score(text, source)
        store = s >= self.threshold

        vec = encode_text(text)

        if store:
            # Action: store — update generative model with this observation
            self._update_generative_model(vec)
            self._recent_vecs.append(vec)
            if len(self._recent_vecs) > RECENT_BUFFER_SIZE:
                self._recent_vecs = self._recent_vecs[-RECENT_BUFFER_SIZE:]
        else:
            # Even non-stored observations update regime priors (we observed
            # this type of message, so it's more expected in the future)
            self._update_priors(vec)

        return store, s

    # ── Generative model ─────────────────────────────────────────────

    def _variational_free_energy(self, vec: np.ndarray) -> float:
        """Compute variational free energy of an observation.

        F = precision * surprise - log(prior)

        With no regimes (fresh model), everything is maximally surprising.
        With established regimes, surprise is relative to the best-matching
        regime, weighted by that regime's precision.

        Returns a score in [0, 1] where 1 = maximally informative.
        """
        if not self._regimes:
            return 1.0  # No model yet — everything is novel

        # Find the regime with lowest free energy (best explanation)
        min_fe = float('inf')
        for regime in self._regimes:
            fe = regime.free_energy(vec)
            min_fe = min(min_fe, fe)

        # Also check local buffer for redundancy
        if self._recent_vecs:
            max_sim = max(float(np.dot(vec, rv)) for rv in self._recent_vecs)
            if max_sim > REDUNDANCY_THRESHOLD:
                return 0.0  # Redundant — no information value

        # Scale free energy to [0, 1] range using sigmoid
        # Low free energy → 0 (expected, not worth storing)
        # High free energy → 1 (surprising, worth storing)
        scaled = 1.0 / (1.0 + math.exp(-SURPRISE_SCALE * (min_fe - 1.0)))
        return scaled

    def _expected_free_energy(self, vec: np.ndarray) -> float:
        """Expected free energy: epistemic value of storing this observation.

        G = ambiguity (how much would beliefs shift?) + novelty (is this a new regime?)

        High epistemic value means storing this would significantly update
        the generative model — either by refining an existing regime or by
        establishing a new one.
        """
        if not self._regimes:
            return 1.0  # First observation always has maximum epistemic value

        # Find closest regime
        best_regime = None
        best_sim = -1.0
        for regime in self._regimes:
            sim = float(np.dot(vec, regime.mu))
            if sim > best_sim:
                best_sim = sim
                best_regime = regime

        # Novelty: if nothing matches well, this could be a new regime
        if best_sim < 0.3:
            return 0.9  # Likely a new topic — high epistemic value

        # Ambiguity: how much would the best regime's beliefs change?
        epistemic = best_regime.epistemic_value(vec)

        # Scale to [0, 1] — epistemic value is typically small
        return min(1.0, epistemic * 10.0)

    def _update_generative_model(self, vec: np.ndarray):
        """Update the generative model after a store decision.

        Finds the closest matching regime and updates it, or creates a
        new regime if the observation doesn't fit any existing cluster.
        Merges regimes that converge. Decays priors toward uniform.
        """
        if not self._regimes:
            self._regimes.append(Regime(vec))
            return

        # Find closest regime
        best_regime = None
        best_sim = -1.0
        for regime in self._regimes:
            sim = float(np.dot(vec, regime.mu))
            if sim > best_sim:
                best_sim = sim
                best_regime = regime

        if best_sim > 0.3:
            # Update existing regime
            best_regime.update(vec)
            best_regime.prior += 1.0
        else:
            # New regime
            if len(self._regimes) < MAX_REGIMES:
                self._regimes.append(Regime(vec))
            else:
                # Replace the weakest regime (lowest prior * count)
                weakest = min(self._regimes, key=lambda r: r.prior * r.count)
                idx = self._regimes.index(weakest)
                self._regimes[idx] = Regime(vec)

        # Normalize priors
        self._normalize_priors()

        # Merge regimes that have converged
        self._merge_close_regimes()

    def _update_priors(self, vec: np.ndarray):
        """Update regime priors for non-stored observations.

        Even messages we don't store update our beliefs about what's
        expected. This prevents the model from becoming too narrow.
        """
        if not self._regimes:
            return

        # Soft assignment: regimes close to the observation get prior boost
        for regime in self._regimes:
            sim = float(np.dot(vec, regime.mu))
            if sim > 0.3:
                regime.prior += 0.1 * sim  # Small boost proportional to similarity

        # Decay all priors toward uniform
        for regime in self._regimes:
            regime.prior *= PRIOR_DECAY

        self._normalize_priors()

    def _normalize_priors(self):
        """Normalize regime priors to sum to 1."""
        total = sum(r.prior for r in self._regimes)
        if total > 0:
            for r in self._regimes:
                r.prior /= total

    def _merge_close_regimes(self):
        """Merge regimes whose centroids have converged."""
        if len(self._regimes) < 2:
            return

        merged = True
        while merged:
            merged = False
            for i in range(len(self._regimes)):
                for j in range(i + 1, len(self._regimes)):
                    sim = float(np.dot(self._regimes[i].mu, self._regimes[j].mu))
                    if sim > REGIME_MERGE_THRESHOLD:
                        ri, rj = self._regimes[i], self._regimes[j]
                        # Weighted merge: larger regime dominates
                        total = ri.count + rj.count
                        new_mu = (ri.mu * ri.count + rj.mu * rj.count) / total
                        norm = np.linalg.norm(new_mu)
                        if norm > 1e-10:
                            new_mu /= norm
                        ri.mu = new_mu
                        ri.count = total
                        ri.prior += rj.prior
                        ri.precision = (ri.precision * ri.count + rj.precision * rj.count) / total
                        self._regimes.pop(j)
                        merged = True
                        break
                if merged:
                    break

    # ── Content prior ────────────────────────────────────────────────

    def _content_prior(self, text: str, source: str) -> float:
        """Prior probability that this message type is informative.

        This is the likelihood term p(o|s) in the generative model —
        different source types and content patterns have different base
        rates of being worth storing, learned from the structure of
        conversation.
        """
        score = 0.0

        # Source prior: user decisions and preferences are inherently valuable
        for pattern in DECISION_SIGNALS:
            if pattern.search(text):
                score = max(score, 0.9)
                break

        for pattern in PREFERENCE_SIGNALS:
            if pattern.search(text):
                score = max(score, 0.8)
                break

        # Entity mentions increase informativeness
        from .entities import extract_entities
        entities = extract_entities(text, source=source)
        entity_bonus = min(0.5, len(entities) * 0.15)
        score = max(score, entity_bonus)

        # File paths and code are structural content
        if re.search(r'/[\w./-]{3,}', text):
            score = max(score, 0.5)
        if re.search(r'```|def |class |function |import |from ', text):
            score = max(score, 0.6)

        # Questions signal information-seeking
        if source == "user" and text.strip().endswith("?"):
            score = max(score, 0.5)

        # Source-type prior
        source_prior = {
            "user": 0.6, "assistant": 0.5, "system": 0.4,
            "tool_use": 0.4, "tool_result": 0.2,
        }.get(source, 0.3)

        # Length modulation on source prior
        length = len(text.strip())
        if length < 10:
            source_prior *= 0.3
        elif length < 30:
            source_prior *= 0.6
        elif length < 100:
            source_prior *= 0.8

        score = max(score, source_prior)

        # Excitation (breakthrough detection) boosts prior
        from .excitation import score_excitation
        excitation = score_excitation(text, source)
        if excitation > 0.8:
            score = max(score, 0.9)
        elif excitation > 0.5:
            score = max(score, excitation)

        if score == 0.0 and len(text.strip()) > 20:
            score = 0.3

        return score

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
        # Pure rage without substance — not worth storing
        from .excitation import _is_rage
        if source == "user" and _is_rage(stripped):
            return True
        return False

    # ── Temporal novelty (cross-session via Foundry) ────────────────

    def _temporal_novelty(self, text: str) -> float:
        """Temporal novelty modulator.

        Returns a multiplier in [0.1, 1.0]:
        - 1.0 for novel content or old recurring topics
        - 0.1 for recently stored duplicates
        - 0.5 when Foundry is unavailable (agnostic)
        """
        if not self.foundry:
            return 1.0  # No temporal info — don't penalize

        if self._msg_count % FOUNDRY_CHECK_INTERVAL != 0:
            return 1.0

        try:
            match = self._query_foundry_nearest(text)
            if not match:
                return 1.0

            match_time = match.get("timestamp")
            if not match_time:
                return 1.0

            if isinstance(match_time, str):
                try:
                    match_dt = datetime.fromisoformat(match_time.replace("Z", "+00:00"))
                except ValueError:
                    return 1.0
            elif isinstance(match_time, datetime):
                match_dt = match_time
            else:
                return 1.0

            now = datetime.now(timezone.utc)
            age = now - match_dt

            if age < timedelta(minutes=STALE_MINUTES):
                return 0.1
            elif age < timedelta(hours=1):
                return 0.5
            elif age < timedelta(days=1):
                return 0.8
            else:
                return 1.0

        except Exception:
            return 1.0

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
        except Exception:
            return None

    # ── Introspection ────────────────────────────────────────────────

    def regime_count(self) -> int:
        """Number of learned topic regimes."""
        return len(self._regimes)

    def regime_summary(self) -> list[dict]:
        """Summary of learned regimes for debugging/visualization."""
        return [
            {
                "index": i,
                "precision": r.precision,
                "prior": r.prior,
                "count": r.count,
            }
            for i, r in enumerate(self._regimes)
        ]
