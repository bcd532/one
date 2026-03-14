"""Hyperdimensional Computing (HDC) text encoder.

Encodes text into 4096-dimensional float vectors using character trigrams,
word-level encoding, and word bigrams with HDC algebra (bind=multiply,
bundle=sum, permute=rotate).
"""

import re
import numpy as np
from functools import lru_cache
from typing import Optional

DIM = 4096
SEED = 0xDEAD
CONTEXT_ALPHA = 0.7    # EMA blend factor — higher values favor recent messages
SHIFT_THRESHOLD = 0.05  # Jaccard overlap below this triggers a topic shift


# ── Codebook ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _byte_codebook() -> np.ndarray:
    """256 bipolar random vectors, one per byte value. Shape: (256, DIM)."""
    rng = np.random.default_rng(SEED)
    return rng.choice([-1.0, 1.0], size=(256, DIM))


@lru_cache(maxsize=4096)
def _word_vec(word: str) -> np.ndarray:
    """Deterministic bipolar random vector for a word, seeded by hash."""
    h = hash(word) & 0xFFFFFFFF
    rng = np.random.default_rng(SEED ^ h)
    return rng.choice([-1.0, 1.0], size=DIM)


# ── Primitives ──────────────────────────────────────────────────────

def permute(v: np.ndarray, n: int = 1) -> np.ndarray:
    return np.roll(v, n)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def bundle(*vectors: np.ndarray) -> np.ndarray:
    return np.sum(vectors, axis=0)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


# ── Text preprocessing ──────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalize text: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens."""
    return _clean(text).split()


# ── Encoding levels ─────────────────────────────────────────────────

def _encode_trigrams(raw: bytes) -> np.ndarray:
    """Character trigram encoding.

    Typo-tolerant: a single character error only corrupts 3 out of N-2 trigrams.
    Operates on raw bytes for natural Unicode handling.
    """
    cb = _byte_codebook()

    if len(raw) == 0:
        return np.zeros(DIM)
    if len(raw) == 1:
        return cb[raw[0]]
    if len(raw) == 2:
        return bind(cb[raw[0]], permute(cb[raw[1]], 1))

    acc = np.zeros(DIM)
    for i in range(len(raw) - 2):
        a = cb[raw[i]]
        b = np.roll(cb[raw[i + 1]], 1)
        c = np.roll(cb[raw[i + 2]], 2)
        acc += a * b * c

    return acc


def _encode_words(words: list[str]) -> np.ndarray:
    """Word-level bag encoding with positional permutation.

    Each word receives a deterministic random vector, permuted by its position
    in the sequence.
    """
    if not words:
        return np.zeros(DIM)

    acc = np.zeros(DIM)
    for i, word in enumerate(words):
        acc += permute(_word_vec(word), i % 64)
    return acc


def _encode_bigrams_words(words: list[str]) -> np.ndarray:
    """Word bigram encoding for phrase-level pattern capture.

    Order-sensitive: bind("check", "ssh") differs from bind("ssh", "check").
    """
    if len(words) < 2:
        return np.zeros(DIM)

    acc = np.zeros(DIM)
    for i in range(len(words) - 1):
        acc += bind(_word_vec(words[i]), permute(_word_vec(words[i + 1]), 1))
    return acc


# ── Main encoder ────────────────────────────────────────────────────

def encode_text(text: str) -> np.ndarray:
    """Multi-level HDC encoding of text.

    Combines character trigrams (0.3), word vectors (0.4), and word
    bigrams (0.3) into a single normalized unit vector.
    """
    cleaned = _clean(text)
    raw = cleaned.encode("utf-8", errors="replace")
    words = cleaned.split()

    v_tri = _encode_trigrams(raw)
    v_word = _encode_words(words)
    v_bigram = _encode_bigrams_words(words)

    combined = 0.3 * v_tri + 0.4 * v_word + 0.3 * v_bigram
    return normalize(combined)


def encode_text_to_list(text: str) -> list[float]:
    return encode_text(text).tolist()


# ── Tagged encoding ─────────────────────────────────────────────────

def encode_tagged(text: str, **tags: str) -> np.ndarray:
    """Encode text with metadata tags blended in.

    Tags are encoded as bind(key_vec, value_vec) and added at low weight
    so content dominates while metadata remains recoverable.
    """
    v = encode_text(text)
    for key, val in tags.items():
        tag_vec = bind(_word_vec(key), _word_vec(val))
        v = v + tag_vec * 0.1
    return normalize(v)


# ── Conversation context tracker ────────────────────────────────────

class ConversationContext:
    """Tracks conversation flow using an exponential moving average (EMA) context vector.

    Detects topic shifts via word overlap and encodes messages with context
    momentum so semantically related messages cluster together.
    """

    def __init__(self, alpha: float = CONTEXT_ALPHA):
        self.alpha = alpha
        self._context: Optional[np.ndarray] = None
        self._prev_vec: Optional[np.ndarray] = None
        self._turn_count = 0
        self._recent_words: Optional[set[str]] = None
        self._last_sim = 0.0
        self._shifted = False

    def encode(self, text: str, source: str = "user") -> np.ndarray:
        """Encode text with conversation context blended in.

        Topic shift detection uses Jaccard word overlap on a sliding window,
        which is more reliable than raw cosine similarity for random-indexed
        vectors. The HDC vector is still used for the output encoding.
        """
        raw_vec = encode_tagged(text, role=source)
        words = set(_clean(text).split())

        # Topic shift detection via word overlap
        self._last_sim = 0.0
        if self._recent_words:
            overlap = len(words & self._recent_words)
            union = len(words | self._recent_words)
            self._last_sim = overlap / union if union > 0 else 0.0
            self._shifted = self._last_sim < SHIFT_THRESHOLD
        else:
            self._shifted = False

        # Update sliding word window
        if self._shifted:
            self._recent_words = words.copy()
        else:
            self._recent_words = (self._recent_words | words) if self._recent_words else words.copy()
            if len(self._recent_words) > 80:
                self._recent_words = words | set(list(self._recent_words)[:40])

        # EMA context vector update
        if self._context is None or self._shifted:
            self._context = raw_vec.copy()
        else:
            self._context = normalize(
                (1 - self.alpha) * self._context + self.alpha * raw_vec
            )

        ctx_vec = normalize(raw_vec + 0.15 * self._context)

        self._prev_vec = raw_vec
        self._turn_count += 1

        return ctx_vec

    @property
    def last_similarity(self) -> float:
        return self._last_sim

    @property
    def shifted(self) -> bool:
        return self._shifted

    def get_regime(self, vec: Optional[np.ndarray] = None) -> str:
        """Return the current conversation regime tag. Call after encode()."""
        if self._turn_count <= 1:
            return "new"
        if self._shifted:
            return "shift"
        return "continue"

    @property
    def turn_count(self) -> int:
        return self._turn_count
