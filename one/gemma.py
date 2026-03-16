"""Local LLM orchestration for context condensation.

Uses Ollama with Gemma 3 (4B) to condense retrieved memories into compact
context blocks and extract key facts from research output. Falls back
gracefully when Ollama is unavailable.
"""

import subprocess
import json
from typing import Optional

OLLAMA_MODEL = "gemma3:4b"
OLLAMA_TIMEOUT = 90

CONDENSE_PROMPT = """You are a technical note condenser for a software project.
Given these retrieved memory fragments, produce a SINGLE concise context block.

Rules:
- Preserve exact technical terms (HDC, Tsetlin Machine, clause, bind, bundle, etc.)
- Keep file paths, function names, and numbers exact
- Decisions and their rationale are highest priority
- User preferences/rules are highest priority
- Drop greetings, acknowledgments, routine tool output
- Output ONLY the condensed context, no preamble

MEMORIES:
{memories}

CONDENSED CONTEXT:"""

RESEARCH_EXTRACT_PROMPT = """Extract the key facts from this research output.
Keep technical detail. Drop boilerplate. Bullet points.

RESEARCH:
{text}

KEY FACTS:"""


def _call_claude(prompt: str, timeout: int = 120) -> Optional[str]:
    """Send a prompt to Claude via CLI. Returns response text or None."""
    try:
        from .proxy import ClaudeProxy
        return ClaudeProxy.quick_ask(prompt, model="sonnet", timeout=timeout)
    except Exception:
        return None


def _call_gemma(prompt: str, timeout: int = OLLAMA_TIMEOUT) -> Optional[str]:
    """Send a prompt to local Gemma via Ollama. Returns response text or None."""
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, "--nowordwrap"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _call_ollama(prompt: str, timeout: int = OLLAMA_TIMEOUT) -> Optional[str]:
    """Send a prompt to Claude first, Gemma fallback. Returns response text or None.

    Claude is the primary reasoning engine. Gemma is only used when
    Claude is unavailable (no CLI, rate limited, etc).
    """
    # Claude first — real intelligence
    result = _call_claude(prompt, timeout=timeout)
    if result:
        return result

    # Gemma fallback
    return _call_gemma(prompt, timeout=timeout)


def is_available() -> bool:
    """Check whether Ollama is installed and the Gemma model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5,
        )
        return OLLAMA_MODEL.split(":")[0] in result.stdout
    except Exception:
        return False


def condense_memories(memories: list[dict], max_input_chars: int = 3000) -> Optional[str]:
    """Condense a list of retrieved memories into a compact context block.

    Args:
        memories: List of dicts with "raw_text", "source", "tm_label", "aif_confidence".
        max_input_chars: Maximum character budget for the prompt input.

    Returns:
        Condensed text string, or None if condensation fails.
    """
    if not memories:
        return None

    ranked = sorted(memories, key=lambda m: m.get("aif_confidence", 0), reverse=True)
    lines = []
    chars = 0
    for m in ranked:
        src = m.get("source", "?")
        label = m.get("tm_label", "?")
        text = m.get("raw_text", "")
        conf = m.get("aif_confidence", 0)
        line = f"[{src}|{label}|{conf:.1f}] {text}"
        if chars + len(line) > max_input_chars:
            break
        lines.append(line)
        chars += len(line)

    prompt = CONDENSE_PROMPT.format(memories="\n".join(lines))
    return _call_ollama(prompt)


def extract_research(text: str) -> Optional[str]:
    """Extract key facts from research output (e.g., web search results).

    Short texts (< 200 chars) are returned as-is without LLM processing.
    """
    if len(text) < 200:
        return text

    prompt = RESEARCH_EXTRACT_PROMPT.format(text=text[:4000])
    return _call_ollama(prompt, timeout=60)


def condense_topic_thread(messages: list[dict]) -> Optional[str]:
    """Condense a completed topic thread into a single summary.

    Called when a topic regime ends (shift detected). Requires at least
    3 messages to produce a meaningful summary.

    Args:
        messages: Raw messages from the topic thread.

    Returns:
        Condensed summary string, or None if condensation fails.
    """
    if len(messages) < 3:
        return None

    lines = []
    for m in messages:
        src = m.get("source", "?")
        text = m.get("raw_text", "")
        lines.append(f"{src}: {text}")

    convo = "\n".join(lines)
    if len(convo) > 4000:
        convo = convo[:4000] + "\n... (truncated)"

    prompt = f"""Summarize this conversation thread into a concise technical note.
Preserve all decisions, technical details, file paths, and action items.
Be specific. Use bullet points.

CONVERSATION:
{convo}

SUMMARY:"""

    return _call_ollama(prompt)
