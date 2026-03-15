"""Excitation detection — arousal scoring for breakthrough identification.

Detects when a speaker is genuinely excited or has discovered something
important, regardless of communication style. Uses both pattern-based
heuristics (fast) and optional LLM classification (accurate).

Two speaker profiles:
  - User: casual, profane, caps-heavy when excited
  - Assistant: formal, structured, breaks patterns when excited
"""

import re
from typing import Optional


# ── User excitation signals ─────────────────────────────────────────

def _user_caps_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _user_profanity_density(text: str) -> float:
    """Profanity as engagement signal, not negativity."""
    words = text.lower().split()
    if not words:
        return 0.0
    profanity = {"fuck", "shit", "damn", "hell", "holy", "omg", "wtf", "lmao",
                  "fucking", "shitty", "damned"}
    hits = sum(1 for w in words if w in profanity)
    return min(1.0, hits / max(len(words), 1) * 5)


def _user_emphasis(text: str) -> float:
    """Exclamation marks, repeated characters, caps words."""
    score = 0.0
    score += min(0.4, text.count("!") * 0.1)
    score += min(0.3, text.count("?!") * 0.15)

    # Repeated characters: "yooooo", "noooo", "yessss"
    repeats = len(re.findall(r'(.)\1{2,}', text))
    score += min(0.3, repeats * 0.1)

    return min(1.0, score)


def _user_length_anomaly(text: str, baseline_avg: float = 50.0) -> float:
    """Long messages from a usually-short speaker = something important."""
    ratio = len(text) / max(baseline_avg, 1)
    if ratio > 5:
        return 0.8
    if ratio > 3:
        return 0.5
    if ratio > 2:
        return 0.3
    return 0.0


def _has_substance(text: str) -> bool:
    """Check if a message contains actual technical/actionable content beyond emotion."""
    cleaned = re.sub(r'[^a-zA-Z\s/.]', '', text).lower().strip()
    words = [w for w in cleaned.split() if len(w) > 2]
    # Filter out pure profanity/emotion words
    noise_words = {
        "fuck", "shit", "damn", "hell", "omg", "wtf", "lmao", "bro", "dude",
        "the", "this", "that", "what", "why", "how", "not", "dont", "its",
        "fucking", "fuuuck", "fuckkkk", "shittt",
    }
    substance_words = [w for w in words if w not in noise_words and not re.match(r'^(.)\1{2,}$', w)]
    return len(substance_words) >= 2


def _is_rage(text: str) -> bool:
    """Detect pure rage/frustration without actionable content."""
    caps = _user_caps_ratio(text)
    profanity = _user_profanity_density(text)
    has_content = _has_substance(text)
    # High emotion + no substance = rage
    return (caps > 0.5 or profanity > 0.3) and not has_content


def score_user_excitation(text: str) -> float:
    """Score user message excitation 0.0 (calm) to 1.0 (breakthrough).

    Distinguishes rage (high arousal, no content) from eureka
    (high arousal, real content). Rage scores near zero.
    """
    # Rage filter — venting without substance
    if _is_rage(text):
        return 0.05

    caps = _user_caps_ratio(text)
    profanity = _user_profanity_density(text)
    emphasis = _user_emphasis(text)
    length = _user_length_anomaly(text)

    # Eureka phrases — only trigger if there's actual content
    eureka = 0.0
    eureka_patterns = [
        r'\b(eureka|holy shit|wait wait|oh my god|oh shit|no way)\b',
        r'\b(i just realized|it just hit me|that means|this changes everything)\b',
        r'\b(dude|bro|yo)\b.{0,20}(this|that|it)\b.{0,20}(works?|means?|could)',
        r'\b(we found|i found|figured out|cracked it|got it)\b',
    ]
    if _has_substance(text):
        for p in eureka_patterns:
            if re.search(p, text, re.I):
                eureka = 0.9
                break

    score = max(
        eureka,
        0.3 * caps + 0.2 * profanity + 0.3 * emphasis + 0.2 * length,
    )
    return min(1.0, score)


# ── Assistant excitation signals ────────────────────────────────────

def score_assistant_excitation(text: str) -> float:
    """Score assistant message excitation 0.0 (routine) to 1.0 (breakthrough)."""
    score = 0.0

    # Pattern breaks — assistant going off-script
    significance_phrases = [
        (r'\bthis is (significant|important|critical|key|notable)\b', 0.7),
        (r'\bactually,? (wait|hold on|I just|let me reconsider)\b', 0.8),
        (r'\bI (just )?realized\b', 0.85),
        (r'\bthis changes\b', 0.9),
        (r'\bthis means\b', 0.7),
        (r'\bcrucially\b|\bfundamentally\b|\bbreakthrough\b', 0.8),
        (r'\bthe (key|core|critical) (insight|thing|realization) is\b', 0.85),
        (r'\bwait.{0,30}that means\b', 0.9),
    ]
    for pattern, val in significance_phrases:
        if re.search(pattern, text, re.I):
            score = max(score, val)

    # Unprompted elaboration — long response to simple question
    # (caller should check input length vs output length)
    if len(text) > 500:
        score = max(score, 0.3)
    if len(text) > 1500:
        score = max(score, 0.5)

    # Deviation from bullet-point style into prose paragraphs
    lines = text.strip().split("\n")
    prose_lines = sum(1 for l in lines if len(l) > 100 and not l.strip().startswith(("-", "*", "#", "|")))
    if prose_lines > 3:
        score = max(score, 0.4)

    # Hedging reduction — confident language = found something
    confident = [
        r'\bthis is\b(?! just| only| merely)',
        r'\bthe answer is\b',
        r'\bdefinitely\b|\bclearly\b|\bwithout question\b',
    ]
    for p in confident:
        if re.search(p, text, re.I):
            score = max(score, 0.3)
            break

    return min(1.0, score)


# ── Combined scorer ─────────────────────────────────────────────────

def score_excitation(text: str, source: str = "user") -> float:
    """Score excitation for any speaker."""
    if source == "user":
        return score_user_excitation(text)
    elif source == "assistant":
        return score_assistant_excitation(text)
    return 0.0


# ── Gemma-based excitation (optional, more accurate) ────────────────

EXCITATION_PROMPT = """Rate the excitation level of this message from 0.0 (routine) to 1.0 (breakthrough/discovery).

Consider:
- Did the speaker discover something new?
- Are they breaking their normal communication pattern?
- Is this routine conversation or a genuine insight?

Speaker profile:
- user: casual, uses profanity for emphasis, short messages normally
- assistant: formal, structured, measured — excitement shows as pattern breaks

SPEAKER: {source}
MESSAGE: {text}

Reply with ONLY a number between 0.0 and 1.0:"""


def score_excitation_gemma(text: str, source: str = "user") -> Optional[float]:
    """Use Gemma for more accurate excitation scoring. Returns None on failure."""
    try:
        from .gemma import _call_ollama
        prompt = EXCITATION_PROMPT.format(source=source, text=text[:500])
        result = _call_ollama(prompt, timeout=15)
        if result:
            # Extract first float from response
            import re as _re
            match = _re.search(r'(\d\.\d+|\d)', result.strip())
            if match:
                return min(1.0, max(0.0, float(match.group(1))))
    except Exception:
        pass
    return None
