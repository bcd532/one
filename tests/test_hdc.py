"""Tests for the HDC encoder — the mathematical foundation of one."""

import numpy as np
import pytest

from one.hdc import (
    DIM, encode_text, encode_tagged, encode_text_to_list,
    similarity, normalize, permute, bind, bundle,
    _byte_codebook, _word_vec, _encode_trigrams, _encode_words,
    _encode_bigrams_words, _clean, _tokenize,
    ConversationContext,
)


class TestPrimitives:
    def test_permute_is_rotation(self):
        v = np.arange(DIM, dtype=float)
        p = permute(v, 1)
        assert p[0] == v[-1]
        assert p[1] == v[0]

    def test_permute_reversible(self):
        v = np.random.randn(DIM)
        assert np.allclose(permute(permute(v, 3), -3), v)

    def test_bind_is_elementwise_multiply(self):
        a = np.ones(DIM)
        b = np.ones(DIM) * -1
        result = bind(a, b)
        assert np.all(result == -1)

    def test_bind_self_inverse(self):
        """Binding a bipolar vector with itself gives all 1s."""
        v = np.random.choice([-1.0, 1.0], size=DIM)
        result = bind(v, v)
        assert np.all(result == 1.0)

    def test_bundle_is_sum(self):
        a = np.ones(DIM)
        b = np.ones(DIM) * 2
        result = bundle(a, b)
        assert np.all(result == 3.0)

    def test_normalize_unit_length(self):
        v = np.random.randn(DIM)
        n = normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_normalize_zero_vector(self):
        v = np.zeros(DIM)
        n = normalize(v)
        assert np.all(n == 0)

    def test_similarity_identical(self):
        v = np.random.randn(DIM)
        assert abs(similarity(v, v) - 1.0) < 1e-6

    def test_similarity_orthogonal(self):
        """Random high-dim vectors should be nearly orthogonal."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(DIM)
        b = rng.standard_normal(DIM)
        sim = similarity(a, b)
        assert abs(sim) < 0.1  # near zero for random vectors


class TestCodebook:
    def test_byte_codebook_shape(self):
        cb = _byte_codebook()
        assert cb.shape == (256, DIM)

    def test_byte_codebook_bipolar(self):
        cb = _byte_codebook()
        assert set(np.unique(cb)) == {-1.0, 1.0}

    def test_byte_codebook_deterministic(self):
        cb1 = _byte_codebook()
        cb2 = _byte_codebook()
        assert np.array_equal(cb1, cb2)

    def test_word_vec_deterministic(self):
        v1 = _word_vec("hello")
        v2 = _word_vec("hello")
        assert np.array_equal(v1, v2)

    def test_word_vec_different_words(self):
        v1 = _word_vec("hello")
        v2 = _word_vec("world")
        sim = similarity(v1, v2)
        assert abs(sim) < 0.2  # different words = nearly orthogonal


class TestTextPreprocessing:
    def test_clean_lowercase(self):
        assert _clean("Hello WORLD") == "hello world"

    def test_clean_strip_punctuation(self):
        assert _clean("hello, world!") == "hello world"

    def test_clean_collapse_whitespace(self):
        assert _clean("hello   world") == "hello world"

    def test_tokenize(self):
        tokens = _tokenize("Hello World Test")
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_empty(self):
        assert _tokenize("") == []


class TestEncodingLevels:
    def test_trigrams_empty(self):
        result = _encode_trigrams(b"")
        assert np.all(result == 0)

    def test_trigrams_single_byte(self):
        result = _encode_trigrams(b"a")
        assert result.shape == (DIM,)
        assert not np.all(result == 0)

    def test_trigrams_two_bytes(self):
        result = _encode_trigrams(b"ab")
        assert result.shape == (DIM,)

    def test_trigrams_normal(self):
        result = _encode_trigrams(b"hello world")
        assert result.shape == (DIM,)
        assert np.linalg.norm(result) > 0

    def test_words_empty(self):
        result = _encode_words([])
        assert np.all(result == 0)

    def test_words_single(self):
        result = _encode_words(["hello"])
        assert result.shape == (DIM,)
        assert not np.all(result == 0)

    def test_bigrams_empty(self):
        result = _encode_bigrams_words([])
        assert np.all(result == 0)

    def test_bigrams_single_word(self):
        result = _encode_bigrams_words(["hello"])
        assert np.all(result == 0)  # need >= 2 words

    def test_bigrams_order_sensitive(self):
        ab = _encode_bigrams_words(["check", "ssh"])
        ba = _encode_bigrams_words(["ssh", "check"])
        sim = similarity(ab, ba)
        assert sim < 0.9  # different order = different encoding


class TestMainEncoder:
    def test_encode_text_shape(self):
        v = encode_text("hello world")
        assert v.shape == (DIM,)

    def test_encode_text_normalized(self):
        v = encode_text("hello world")
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_encode_text_similar_content(self):
        v1 = encode_text("fix the authentication bug")
        v2 = encode_text("fix the auth bug")
        sim = similarity(v1, v2)
        assert sim > 0.5  # similar content

    def test_encode_text_different_content(self):
        v1 = encode_text("fix the authentication bug")
        v2 = encode_text("deploy to production server")
        sim = similarity(v1, v2)
        assert sim < 0.5  # different content

    def test_encode_text_empty(self):
        v = encode_text("")
        assert v.shape == (DIM,)

    def test_encode_text_unicode(self):
        v = encode_text("こんにちは世界")
        assert v.shape == (DIM,)
        assert np.linalg.norm(v) > 0

    def test_encode_text_very_long(self):
        """Test degradation with very long text."""
        v = encode_text("word " * 10000)
        assert v.shape == (DIM,)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_encode_text_to_list(self):
        result = encode_text_to_list("hello")
        assert isinstance(result, list)
        assert len(result) == DIM

    def test_encode_tagged(self):
        text = "the hyperdimensional computing encoder uses trigram representations for robust text encoding"
        v1 = encode_tagged(text, role="user")
        v2 = encode_tagged(text, role="assistant")
        sim = similarity(v1, v2)
        assert sim > 0.8  # same content dominates
        assert sim < 1.0  # tags create slight difference

    def test_encode_tagged_deterministic(self):
        v1 = encode_tagged("test", role="user")
        v2 = encode_tagged("test", role="user")
        assert np.allclose(v1, v2)


class TestConversationContext:
    def test_context_initialization(self):
        ctx = ConversationContext()
        assert ctx.turn_count == 0
        assert ctx.shifted is False

    def test_context_first_encode(self):
        ctx = ConversationContext()
        v = ctx.encode("hello world", "user")
        assert v.shape == (DIM,)
        assert ctx.turn_count == 1

    def test_context_regime_first_turn(self):
        ctx = ConversationContext()
        ctx.encode("hello", "user")
        assert ctx.get_regime() == "new"

    def test_context_regime_continue(self):
        ctx = ConversationContext()
        ctx.encode("hello world", "user")
        ctx.encode("hello world again", "user")
        assert ctx.get_regime() == "continue"

    def test_context_topic_shift(self):
        ctx = ConversationContext()
        ctx.encode("the HDC encoder uses trigrams and codebooks for vector encoding", "user")
        ctx.encode("the HDC encoder uses trigrams and codebooks for vector encoding", "user")
        # Completely different topic (no word overlap with above)
        ctx.encode("basketball game tomorrow evening at seven pm", "user")
        assert ctx.shifted is True
        assert ctx.get_regime() == "shift"

    def test_context_similarity_tracking(self):
        ctx = ConversationContext()
        ctx.encode("hello", "user")
        ctx.encode("hello again", "user")
        assert ctx.last_similarity >= 0.0
