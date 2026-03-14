"""ANSI terminal UI for One.

Lightweight alternative to the Textual-based app. Renders directly to
stdout using ANSI escape codes, with streaming output, spinner animation,
and live memory recall support.
"""

import sys
import json
import threading
import shutil
import time
import itertools
import textwrap
from queue import Queue, Empty
from typing import Optional

from .proxy import ClaudeProxy

# ── ANSI ────────────────────────────────────────────────────────────
B = "\033[1m"
D = "\033[2m"
R = "\033[0m"
I = "\033[3m"

C = "\033[36m"
G = "\033[32m"
Y = "\033[33m"
RE = "\033[31m"
M = "\033[35m"
BL = "\033[34m"
W = "\033[37m"
GR = "\033[90m"

CLR = "\033[2J\033[H"
CLINE = "\033[2K\r"
CUR_SHOW = "\033[?25h"
CUR_HIDE = "\033[?25l"
ALT_ON = "\033[?1049h"
ALT_OFF = "\033[?1049l"

MAX_W = 100


LOGO = f"""{B}{C}
     ██████╗ ███╗   ██╗███████╗
    ██╔═══██╗████╗  ██║██╔════╝
    ██║   ██║██╔██╗ ██║█████╗
    ██║   ██║██║╚██╗██║██╔══╝
    ╚██████╔╝██║ ╚████║███████╗
     ╚═════╝ ╚═╝  ╚═══╝╚══════╝{R}"""


def tw():
    return min(shutil.get_terminal_size((80, 24)).columns, MAX_W)


def hr(w=None):
    return f"{D}{'─' * ((w or tw()) - 4)}{R}"


RECALL_EVERY = 5


class OneTUI:
    """ANSI-based terminal interface with streaming output and memory recall."""

    def __init__(self, proxy: ClaudeProxy, foundry_client=None):
        self.proxy = proxy
        self.foundry = foundry_client
        self._pending_push = Queue()
        self._session_id = None
        self._model = None
        self._last_user_msg = ""

        self._in_thinking = False
        self._in_text = False
        self._in_tool_input = False
        self._current_tool = None
        self._tool_input_buf = ""
        self._thinking_col = 0
        self._spinner_active = False
        self._spinner_lock = threading.Lock()

        self._turn_complete = threading.Event()
        self._initialized = threading.Event()
        self._total_cost = 0.0
        self._total_duration = 0
        self._total_turns = 0

        self._turn_counter = 0
        self._recent_texts: list[str] = []
        self._ctx_tracker = None

    def start(self):
        self.proxy.on_event(self._handle_event)

        sys.stdout.write(ALT_ON + CLR)
        sys.stdout.flush()

        self._print_splash()
        self.proxy.start()

        if self.foundry:
            threading.Thread(target=self._push_loop, daemon=True).start()

        self._input_loop()

    def _print_splash(self):
        print(LOGO)
        print()
        f = f"{G}foundry:on{R}" if self.foundry else f"{GR}foundry:off{R}"
        print(f"    {D}v0.1{R}  {f}  {D}palantir aip{R}")
        print(f"    {hr()}")
        print()

    def _user_header(self, text: str):
        w = tw() - 6
        display = text if len(text) <= w else text[:w - 3] + "..."
        print(f"  {B}{G}▌{R} {B}{display}{R}")
        print(f"  {hr()}")

    # ── Input loop ──────────────────────────────────────────────────

    def _input_loop(self):
        try:
            while True:
                text = self._read_input()
                if text is None or text.strip() in ("/quit", "/exit", "/q"):
                    break

                stripped = text.strip()
                if not stripped:
                    continue

                if stripped == "/cost":
                    print(f"  {D}${self._total_cost:.4f} · {self._total_duration / 1000:.1f}s · {self._total_turns} turns{R}\n")
                    continue
                if stripped == "/clear":
                    sys.stdout.write(CLR)
                    sys.stdout.flush()
                    self._print_splash()
                    continue
                if stripped == "/session":
                    print(f"  {D}{self._session_id or 'initializing...'}{R}\n")
                    continue
                if stripped == "/recall":
                    self._force_recall()
                    continue

                self._last_user_msg = text
                self._capture("user", text)
                self._turn_counter += 1
                self._recent_texts.append(text)
                if len(self._recent_texts) > 5:
                    self._recent_texts = self._recent_texts[-5:]

                self._turn_complete.clear()

                print()
                self._user_header(text)
                self._start_spinner()

                enriched = self._maybe_recall(text)
                self.proxy.send(enriched)

                while not self._turn_complete.is_set():
                    self._turn_complete.wait(timeout=0.1)
                    if not self.proxy.alive:
                        break

                if not self.proxy.alive:
                    break

        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self._stop_spinner()
            sys.stdout.write(CUR_SHOW + ALT_OFF)
            sys.stdout.flush()
            self.proxy.stop()

    def _read_input(self) -> Optional[str]:
        """Read user input, supporting multi-line entry via trailing backslash."""
        lines = []
        try:
            while True:
                line = input(f"  {B}{C}>{R} " if not lines else f"  {D}·{R} ")
                if not lines and not line.endswith("\\"):
                    return line
                if line.endswith("\\"):
                    lines.append(line[:-1])
                    continue
                lines.append(line)
                return "\n".join(lines)
        except EOFError:
            return "\n".join(lines) if lines else None

    # ── Spinner ─────────────────────────────────────────────────────

    def _start_spinner(self):
        with self._spinner_lock:
            self._spinner_active = True
        threading.Thread(target=self._spin, daemon=True).start()

    def _stop_spinner(self):
        with self._spinner_lock:
            if not self._spinner_active:
                return
            self._spinner_active = False
        time.sleep(0.1)
        sys.stdout.write(CLINE)
        sys.stdout.flush()

    def _spin(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        for frame in itertools.cycle(frames):
            with self._spinner_lock:
                if not self._spinner_active:
                    sys.stdout.write(CLINE)
                    sys.stdout.flush()
                    return
            sys.stdout.write(f"{CLINE}  {C}{frame}{R} {D}working...{R}")
            sys.stdout.flush()
            time.sleep(0.08)

    # ── Events ──────────────────────────────────────────────────────

    def _handle_event(self, event: dict):
        t = event.get("type", "")
        s = event.get("subtype", "")

        if t == "system" and s == "init":
            self._session_id = event.get("session_id")
            self._model = event.get("model")
            self._initialized.set()
        elif t == "stream_event":
            self._handle_stream(event.get("event", {}))
        elif t == "assistant":
            self._handle_assistant(event)
        elif t == "user":
            self._handle_user_event(event)
        elif t == "result":
            self._handle_result(event)

    # ── Stream rendering ────────────────────────────────────────────

    def _handle_stream(self, ev: dict):
        t = ev.get("type", "")

        if t == "content_block_start":
            bt = ev.get("content_block", {}).get("type", "")

            if bt == "thinking":
                self._stop_spinner()
                self._in_thinking = True
                self._thinking_col = 0
                w = tw() - 6
                sys.stdout.write(f"\n  {D}┌ {I}thinking{R}{D} {'─' * max(0, w - 11)}┐\n  │ ")
                sys.stdout.flush()

            elif bt == "text":
                self._stop_spinner()
                self._in_text = True

            elif bt == "tool_use":
                self._stop_spinner()
                self._current_tool = ev.get("content_block", {}).get("name", "?")
                self._tool_input_buf = ""
                self._in_tool_input = True

        elif t == "content_block_delta":
            delta = ev.get("delta", {})
            dt = delta.get("type", "")

            if dt == "thinking_delta" and self._in_thinking:
                chunk = delta.get("thinking", "")
                if chunk:
                    self._render_thinking(chunk)
            elif dt == "text_delta" and self._in_text:
                chunk = delta.get("text", "")
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            elif dt == "input_json_delta" and self._in_tool_input:
                self._tool_input_buf += delta.get("partial_json", "")

        elif t == "content_block_stop":
            if self._in_thinking:
                self._in_thinking = False
                w = tw() - 6
                sys.stdout.write(f"\n  └{'─' * (w + 2)}┘{R}\n\n")
                sys.stdout.flush()
            elif self._in_text:
                self._in_text = False
            elif self._in_tool_input:
                self._in_tool_input = False
                self._render_tool()

    def _render_thinking(self, text: str):
        """Render thinking text character-by-character with line wrapping."""
        w = tw() - 7
        for ch in text:
            if ch == "\n" or self._thinking_col >= w:
                sys.stdout.write(f"\n  │ ")
                self._thinking_col = 0
                if ch == "\n":
                    continue
            sys.stdout.write(ch)
            self._thinking_col += 1
        sys.stdout.flush()

    # ── Tool rendering ──────────────────────────────────────────────

    def _render_tool(self):
        tool = self._current_tool
        try:
            inp = json.loads(self._tool_input_buf)
        except json.JSONDecodeError:
            inp = self._tool_input_buf

        d = inp if isinstance(inp, dict) else {}
        w = tw() - 4

        colors = {
            "Bash": Y, "Write": M, "Edit": M, "Read": BL,
            "Glob": BL, "Grep": BL, "WebSearch": C, "WebFetch": C,
            "Agent": C, "Skill": C,
        }
        col = colors.get(tool, Y)

        if tool == "Bash":
            cmd = d.get("command", str(inp))
            desc = d.get("description", "")
            print(f"\n  {col}{B}$ Bash{R}")
            lines = cmd.split("\n")
            bw = w - 2
            print(f"  {D}┌{'─' * bw}┐{R}")
            for ln in lines[:10]:
                print(f"  {D}│{R} {ln[:bw - 2]}")
            if len(lines) > 10:
                print(f"  {D}│ ...{len(lines) - 10} more{R}")
            print(f"  {D}└{'─' * bw}┘{R}")
            if desc:
                print(f"  {D}{desc}{R}")

        elif tool in ("Write", "Edit"):
            print(f"\n  {col}{B}{tool}{R} {D}{d.get('file_path', '?')}{R}")

        elif tool == "Read":
            print(f"\n  {col}{B}Read{R} {D}{d.get('file_path', '?')}{R}")

        elif tool in ("Glob", "Grep"):
            print(f"\n  {col}{B}{tool}{R} {D}{d.get('pattern', '?')}{R}")

        elif tool in ("WebSearch", "WebFetch"):
            print(f"\n  {col}{B}{tool}{R} {D}{d.get('query', d.get('url', ''))}{R}")

        elif tool == "Agent":
            at = d.get("subagent_type", "")
            desc = d.get("description", "")
            print(f"\n  {col}{B}Agent{R} {D}{at} — {desc}{R}")

        elif tool == "Skill":
            print(f"\n  {col}{B}Skill{R} {D}{d.get('skill', '')}{R}")

        else:
            print(f"\n  {col}{B}{tool}{R}")
            for k, v in list(d.items())[:3]:
                print(f"    {D}{k}: {str(v)[:w]}{R}")

        self._start_spinner()
        self._capture("tool_use", json.dumps({"tool": tool, "input": inp}))

    # ── Assistant message handling ──────────────────────────────────

    def _handle_assistant(self, event: dict):
        msg = event.get("message", {})
        parts = [b["text"] for b in msg.get("content", []) if b.get("type") == "text"]
        if parts:
            self._capture("assistant", "".join(parts))

    # ── Tool results ────────────────────────────────────────────────

    def _handle_user_event(self, event: dict):
        self._stop_spinner()
        msg = event.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            return

        w = tw() - 6

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            txt = block.get("content", "")
            if not isinstance(txt, str) or not txt:
                continue

            is_err = block.get("is_error", False)

            if "requested permissions" in txt.lower() or "haven't granted" in txt.lower():
                print(f"  {RE}{B}✗ denied{R} {D}{txt[:w]}{R}")
            elif is_err:
                lines = textwrap.wrap(txt, width=w)
                print(f"  {RE}✗{R} {RE}{lines[0] if lines else ''}{R}")
                for ln in lines[1:3]:
                    print(f"    {RE}{ln}{R}")
                if len(lines) > 3:
                    print(f"    {D}...{len(lines) - 3} more lines{R}")
            else:
                print(f"  {G}✓{R}")

            self._capture("tool_result", txt)

    # ── Result ──────────────────────────────────────────────────────

    def _handle_result(self, event: dict):
        self._stop_spinner()
        cost = event.get("total_cost_usd", 0)
        duration = event.get("duration_ms", 0)
        turns = event.get("num_turns", 0)
        self._total_cost += cost
        self._total_duration += duration
        self._total_turns += turns

        t = "turn" if turns == 1 else "turns"
        print(f"\n  {hr()}")
        print(f"  {D}${cost:.4f} · {duration / 1000:.1f}s · {turns} {t} · total: ${self._total_cost:.4f}{R}\n")
        self._turn_complete.set()

    # ── Live recall ──────────────────────────────────────────────────

    def _get_ctx_tracker(self):
        if self._ctx_tracker is None:
            from .hdc import ConversationContext
            self._ctx_tracker = ConversationContext()
        return self._ctx_tracker

    def _maybe_recall(self, text: str) -> str:
        """Inject recalled context on topic shift or periodic interval. Returns enriched message."""
        if not self.foundry:
            return text

        ctx = self._get_ctx_tracker()
        vec = ctx.encode(text, source="user")
        shifted = ctx.shifted
        periodic = (self._turn_counter % RECALL_EVERY == 0) and self._turn_counter > 0

        if not shifted and not periodic:
            return text

        query = " ".join(self._recent_texts[-3:])
        context_block = self._do_recall(query)

        if not context_block:
            return text

        reason = "topic shift" if shifted else f"turn {self._turn_counter}"
        print(f"  {C}{D}↻ recalling ({reason})...{R}")

        return f"{context_block}\n\n{text}"

    def _force_recall(self):
        """Handle the /recall command to display stored memories."""
        if not self.foundry:
            print(f"  {D}foundry offline{R}\n")
            return

        query = " ".join(self._recent_texts[-3:]) if self._recent_texts else "recent"
        memories = self._do_recall_raw(query)

        if not memories:
            print(f"  {D}no memories found{R}\n")
            return

        print(f"  {C}{B}recalled {len(memories)} memories:{R}")
        for m in memories:
            src = m["source"]
            label = m["tm_label"]
            text = m["raw_text"][:70]
            print(f"  {D}[{src}|{label}] {text}{R}")
        print()

    def _do_recall(self, query: str) -> str:
        """Query the backend and return a formatted context block."""
        try:
            from .retrieve import recall_context
            return recall_context(self.foundry, query=query, n=8, max_chars=3000)
        except Exception:
            return ""

    def _do_recall_raw(self, query: str) -> list[dict]:
        """Query the backend and return raw memory dictionaries."""
        try:
            from .retrieve import recall
            return recall(self.foundry, query=query, n=10)
        except Exception:
            return []

    # ── Background memory persistence ───────────────────────────────

    def _capture(self, role: str, content: str):
        if self.foundry and content and content.strip():
            self._pending_push.put((role, content))

    def _push_loop(self):
        """Background loop that encodes and persists captured messages."""
        from .client import push_memory
        from .hdc import ConversationContext

        ctx = ConversationContext()

        while True:
            try:
                role, content = self._pending_push.get(timeout=2)
            except Empty:
                continue
            try:
                vec = ctx.encode(content, source=role)
                regime = ctx.get_regime(vec)
                push_memory(
                    self.foundry,
                    raw_text=content,
                    source=role,
                    regime_tag=regime,
                    hdc_vector=vec.tolist(),
                )
            except Exception:
                pass
