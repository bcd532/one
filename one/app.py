"""Textual-based TUI application for One.

Provides a rich terminal interface with streaming output rendering,
tool call visualization, live memory recall, background memory
persistence, session tracking, and a collapsible information sidebar.
"""

import json
import os
import re
import time
import threading
from queue import Queue, Empty
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, Input, Static, RichLog, Rule
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.message import Message
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.console import Group

from .proxy import ClaudeProxy

LOGO = """[bold cyan]
 ██████╗ ███╗   ██╗███████╗
██╔═══██╗████╗  ██║██╔════╝
██║   ██║██╔██╗ ██║█████╗
██║   ██║██║╚██╗██║██╔══╝
╚██████╔╝██║ ╚████║███████╗
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝[/]"""

RECALL_EVERY = 5


# ── Custom widgets ───────────────────────────────────────────────


class ChatMessage(Static):
    """A single chat message displayed in the conversation stream."""
    pass


class ToolBlock(Static):
    """Visual representation of a tool invocation."""
    pass


class ThinkingBlock(Static):
    """Collapsible display for model thinking/reasoning output."""
    pass


class HistoryInput(Input):
    """Input widget with up/down arrow history navigation.

    Maintains an ordered list of previously submitted messages.
    Up arrow cycles backward through history, down arrow cycles forward.
    Enter submits the current value and appends it to history.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1
        self._draft: str = ""

    def add_to_history(self, text: str) -> None:
        """Append a message to the input history, deduplicating consecutive repeats."""
        if not text:
            return
        if self._history and self._history[-1] == text:
            return
        self._history.append(text)
        self._history_index = -1
        self._draft = ""

    def _on_key(self, event) -> None:
        if event.key == "up":
            event.prevent_default()
            event.stop()
            if not self._history:
                return
            if self._history_index == -1:
                self._draft = self.value
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            self.value = self._history[self._history_index]
            self.cursor_position = len(self.value)
        elif event.key == "down":
            event.prevent_default()
            event.stop()
            if self._history_index == -1:
                return
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                self.value = self._history[self._history_index]
            else:
                self._history_index = -1
                self.value = self._draft
            self.cursor_position = len(self.value)


class Sidebar(Vertical):
    """Collapsible right-side panel showing project state at a glance.

    Displays active rules, recent entities, memory count, current model,
    and project name. Toggled via ctrl+b.
    """

    def __init__(self, **kwargs):
        super().__init__(id="sidebar", **kwargs)

    def compose(self) -> ComposeResult:
        yield Static("", id="sidebar-content")


class Notification(Static):
    """Single-line notification bar that auto-clears after a timeout."""

    def __init__(self, **kwargs):
        super().__init__("", id="notification", **kwargs)
        self._clear_timer: Optional[Timer] = None

    def show_message(self, text: str, duration: float = 3.0) -> None:
        """Display a notification that auto-clears after duration seconds."""
        self.update(text)
        if self._clear_timer is not None:
            self._clear_timer.stop()
        self._clear_timer = self.set_timer(duration, self._clear)

    def _clear(self) -> None:
        self.update("")
        self._clear_timer = None


# ── Application ──────────────────────────────────────────────────


class OneApp(App):
    TITLE = "one"
    SUB_TITLE = "infinite context"

    CSS = """
    Screen {
        background: transparent;
    }

    Header {
        background: transparent;
        color: $text;
        dock: top;
        height: 1;
    }

    Footer {
        background: transparent;
    }

    #chat-area {
        height: 1fr;
        background: transparent;
    }

    #chat-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
        scrollbar-color: #444444;
        scrollbar-color-hover: #666666;
        padding: 0 2;
        background: transparent;
    }

    #sidebar {
        dock: right;
        width: 30;
        background: transparent;
        border-left: tall #333333;
        padding: 1;
        display: none;
    }

    .sidebar-visible #sidebar {
        display: block;
    }

    #sidebar-content {
        background: transparent;
        width: 100%;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: transparent;
        color: #888888;
        padding: 0 2;
    }

    #notification {
        dock: bottom;
        height: 1;
        background: transparent;
        color: #00cccc;
    }

    #input-box {
        dock: bottom;
        background: transparent;
        border: tall #444444;
        padding: 0 1;
        color: $text;
    }

    #input-box:focus {
        border: tall #00cccc;
    }

    ChatMessage {
        margin: 0 0 0 0;
        padding: 0 1;
        background: transparent;
    }

    .user-msg {
        border-left: tall #00cc66;
        padding: 0 0 0 1;
        margin: 1 0 0 0;
        background: transparent;
    }

    .assistant-text {
        padding: 0 0 0 2;
        background: transparent;
    }

    .thinking-block {
        color: #888888;
        padding: 0 1;
        margin: 0 0 0 2;
        border: round #333333;
        max-height: 8;
        overflow-y: auto;
        background: transparent;
    }

    .tool-block {
        margin: 0 0 0 2;
        padding: 0 1;
        border: round #cc8800;
        background: transparent;
    }

    .tool-result {
        padding: 0 0 0 3;
        color: #888888;
        max-height: 4;
        overflow: hidden;
        background: transparent;
    }

    .diff-add {
        color: #00cc66;
        padding: 0 0 0 3;
        background: transparent;
    }

    .diff-del {
        color: #cc4444;
        padding: 0 0 0 3;
        background: transparent;
    }

    .cost-line {
        color: #666666;
        padding: 0 0 0 2;
        margin: 0 0 0 0;
        background: transparent;
    }

    Rule {
        color: #333333;
        margin: 0 0 0 0;
    }

    Static {
        background: transparent;
    }

    VerticalScroll {
        background: transparent;
    }

    Vertical {
        background: transparent;
    }

    Horizontal {
        background: transparent;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+r", "force_recall", "Recall"),
        Binding("ctrl+e", "show_entities", "Entities"),
        Binding("ctrl+u", "show_undo", "Undo"),
        Binding("ctrl+t", "toggle_thinking", "Think"),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar"),
        Binding("ctrl+s", "export_session", "Export"),
        Binding("escape", "interrupt", "Stop", show=False),
    ]

    status_text: reactive[str] = reactive("ready")

    def __init__(
        self,
        proxy: ClaudeProxy,
        foundry_client=None,
        project: str = "global",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proxy = proxy
        self.foundry = foundry_client
        self.project = project

        from .backend import get_backend
        self.backend = get_backend(foundry=foundry_client)
        self._pending_push: Queue = Queue()
        self._auto_loop = None

        self._in_thinking = False
        self._in_text = False
        self._in_tool_input = False
        self._current_tool: Optional[str] = None
        self._tool_input_buf = ""
        self._thinking_text = ""
        self._response_text = ""
        self._active_thinking: Optional[ThinkingBlock] = None
        self._active_response: Optional[ChatMessage] = None
        self._active_tool: Optional[ToolBlock] = None

        self._turn_complete = threading.Event()
        self._initialized = threading.Event()
        self._total_cost = 0.0
        self._total_duration = 0
        self._total_turns = 0
        self._turn_counter = 0
        self._recent_texts: list[str] = []
        self._timer_start: float = 0
        self._timer_handle: Optional[Timer] = None

        self._ctx_tracker = None
        self._preloaded_context: Optional[str] = None
        self._recent_files: list[str] = []
        self._recent_tools: list[str] = []
        self._last_injected_context: str = ""
        self._show_thinking: bool = True

        self._sidebar_visible: bool = False
        self._session_id: Optional[str] = None
        self._memory_count: int = 0

    # ── Layout ───────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Horizontal(
            VerticalScroll(id="chat-scroll"),
            Sidebar(),
            id="chat-area",
        )
        yield Static("", id="status-bar")
        yield Notification()
        yield HistoryInput(placeholder="message one...", id="input-box")

    def on_mount(self) -> None:
        gemma_ok = False
        try:
            from .gemma import is_available
            gemma_ok = is_available()
        except Exception:
            pass

        chat = self.query_one("#chat-scroll")
        chat.mount(Static(LOGO))

        f_tag = "[green]●[/] foundry" if self.foundry else "[green]●[/] local"
        g_tag = "[green]●[/] gemma" if gemma_ok else "[#666666]○ gemma[/]"
        model = self.proxy.model or "opus"
        chat.mount(Static(
            f"    [bold cyan]one[/] [#666666]v0.1[/]  {f_tag}  {g_tag}  [#666666]model:{model}  project:{self.project}[/]"
        ))
        chat.mount(Static(
            "    [#444444]esc:quit  ctrl+l:clear  ctrl+r:recall  ctrl+b:sidebar  /rules  /stats  /cost[/]\n"
            "    [#444444]claude: /commit  /review  /compact  /help  — all pass through[/]"
        ))
        chat.mount(Rule())

        self.proxy.on_event(self._on_proxy_event)
        self.proxy.start()

        threading.Thread(target=self._push_loop, daemon=True).start()

        if self.foundry:
            threading.Thread(target=self._sync_rules, daemon=True).start()

        # Session tracking
        self._start_session()

        # Initial sidebar state
        self._refresh_sidebar()

        self.query_one("#input-box").focus()

    def _sync_rules(self) -> None:
        """Pull rules from Foundry into local SQLite for fast per-turn access."""
        try:
            if hasattr(self.backend, "sync_rules_from_foundry"):
                count = self.backend.sync_rules_from_foundry(self.project)
                if count > 0:
                    self.call_from_thread(
                        self._add_status, f"synced {count} rules from foundry"
                    )
        except Exception:
            pass

    # ── Session tracking ─────────────────────────────────────────

    def _start_session(self) -> None:
        """Create a new session record for this conversation."""
        try:
            from .sessions import create_session
            model = self.proxy.model or "opus"
            self._session_id = create_session(self.project, model)
        except Exception:
            self._session_id = None

    def _session_add_message(self, role: str, content: str) -> None:
        """Record a message in the active session."""
        if not self._session_id:
            return
        try:
            from .sessions import add_message
            add_message(self._session_id, role, content, self._turn_counter)
        except Exception:
            pass

    def _session_end(self) -> None:
        """Finalize the active session with cost data."""
        if not self._session_id:
            return
        try:
            from .sessions import end_session
            end_session(self._session_id, self._total_cost)
        except Exception:
            pass

    # ── Sidebar ──────────────────────────────────────────────────

    def _refresh_sidebar(self) -> None:
        """Update the sidebar content with current project state."""
        try:
            content = self.query_one("#sidebar-content", Static)
        except Exception:
            return

        model = self.proxy.model or "opus"
        lines = [
            f"[bold cyan]{self.project}[/]",
            f"[dim]model: {model}[/]",
            "",
        ]

        # Memory count
        try:
            s = self.backend.stats()
            mem_count = s.get("memories", s.get("total_memories", "?"))
            self._memory_count = mem_count if isinstance(mem_count, int) else 0
            lines.append(f"[bold]memories[/] {mem_count}")
        except Exception:
            lines.append(f"[bold]memories[/] --")

        lines.append("")

        # Active rules
        try:
            from .rules import get_all_rules
            rules = get_all_rules(self.project)
            count = len(rules)
            lines.append(f"[bold]rules[/] ({count})")
            for r in rules[:3]:
                text = r["rule_text"]
                if len(text) > 24:
                    text = text[:21] + "..."
                lines.append(f"  [dim]{text}[/]")
            if count > 3:
                lines.append(f"  [dim]+{count - 3} more[/]")
        except Exception:
            lines.append("[bold]rules[/] --")

        lines.append("")

        # Recent entities
        try:
            from . import store
            ents = store.get_entities(limit=5)
            lines.append(f"[bold]entities[/] ({len(ents)})")
            for e in ents:
                name = e.get("name", "?")
                count = e.get("observation_count", 0)
                if len(name) > 20:
                    name = name[:17] + "..."
                lines.append(f"  [yellow]{name}[/] [dim]{count}x[/]")
        except Exception:
            lines.append("[bold]entities[/] --")

        lines.append("")

        # Connection status
        connected = self.proxy.alive
        conn_tag = "[green]connected[/]" if connected else "[red]disconnected[/]"
        lines.append(conn_tag)

        content.update("\n".join(lines))

    # ── Notifications ────────────────────────────────────────────

    def _notify(self, text: str, duration: float = 3.0) -> None:
        """Show a brief notification that auto-fades."""
        try:
            notif = self.query_one("#notification", Notification)
            notif.show_message(text, duration)
        except Exception:
            pass

    # ── Input handling ───────────────────────────────────────────

    @on(Input.Submitted, "#input-box")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        event.input.value = ""
        input_widget = self.query_one("#input-box", HistoryInput)
        input_widget.add_to_history(text)

        ONE_COMMANDS = {
            "/quit", "/exit", "/q", "/clear", "/cost", "/recall",
            "/rules", "/stats", "/entities", "/search", "/undo",
            "/context", "/forget", "/help", "/think",
            "/sessions", "/session", "/export",
            "/auto", "/stop",
            "/watch", "/unwatch", "/generate",
            "/synthesize", "/research", "/playbooks", "/frontier",
            "/health", "/audit", "/swarm", "/morgoth",
            "/focus", "/inject", "/scale",
        }

        if text in ("/quit", "/exit", "/q"):
            self.exit()
            return
        if text == "/clear":
            self.action_clear_chat()
            return
        if text == "/cost":
            self._add_status(f"${self._total_cost:.4f} · {self._total_duration / 1000:.1f}s · {self._total_turns} turns")
            return
        if text == "/recall":
            self._force_recall()
            return
        if text == "/rules":
            self._show_rules()
            return
        if text == "/stats":
            self._show_stats()
            return
        if text == "/entities":
            self.action_show_entities()
            return
        if text == "/undo":
            self.action_show_undo()
            return
        if text == "/context":
            self._show_last_context()
            return
        if text == "/think":
            self.action_toggle_thinking()
            return
        if text == "/help":
            self._show_help()
            return
        if text == "/sessions":
            self._show_sessions()
            return
        if text == "/export":
            self.action_export_session()
            return
        if text.startswith("/session "):
            self._show_session_messages(text[9:].strip())
            return
        if text.startswith("/search "):
            self._search_memories(text[8:].strip())
            return
        if text.startswith("/forget "):
            self._forget_rule(text[8:].strip())
            return
        if text.startswith("/rule "):
            self._add_manual_rule(text[6:].strip())
            return
        if text.startswith("/auto "):
            self._start_auto(text[6:].strip())
            return
        if text == "/stop":
            self._stop_auto()
            return
        if text == "/watch" or text.startswith("/watch "):
            self._start_watch(text[6:].strip())
            return
        if text == "/unwatch":
            self._stop_watch()
            return
        if text == "/generate":
            self._generate_claude_md()
            return
        if text == "/synthesize":
            self._run_synthesis()
            return
        if text.startswith("/research "):
            self._start_research(text[10:].strip())
            return
        if text == "/playbooks":
            self._show_playbooks()
            return
        if text == "/frontier":
            self._show_frontier()
            return
        if text.startswith("/swarm "):
            self._start_swarm(text[7:].strip())
            return
        if text.startswith("/morgoth "):
            self._start_morgoth(text[9:].strip())
            return
        if text == "/health":
            self._show_health()
            return
        if text == "/audit" or text.startswith("/audit "):
            self._run_audit(text[6:].strip() if len(text) > 6 else "")
            return
        if text.startswith("/focus "):
            self._focus_agent(text[7:].strip())
            return
        if text.startswith("/inject "):
            self._inject_all(text[8:].strip())
            return

        # Unknown /commands pass through to Claude
        if text.startswith("/") and text.split()[0] not in ONE_COMMANDS:
            self._send_message(text)
            return

        self._send_message(text)

    def _send_message(self, text: str) -> None:
        self._turn_counter += 1
        self._recent_texts.append(text)
        if len(self._recent_texts) > 5:
            self._recent_texts = self._recent_texts[-5:]

        chat = self.query_one("#chat-scroll")
        chat.mount(ChatMessage(f"[bold]{text}[/]", classes="user-msg"))
        chat.scroll_end(animate=False)
        self._timer_start = time.time()
        self._start_timer()
        self._capture("user", text)
        self._session_add_message("user", text)

        threading.Thread(
            target=self._recall_and_send,
            args=(text,),
            daemon=True,
        ).start()

    def _recall_and_send(self, text: str) -> None:
        """Background thread: recall context, enrich message, send to Claude.

        If recall takes more than 5 seconds, sends the raw message and
        lets recall finish for the next turn.
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._maybe_recall, text)
            try:
                enriched = future.result(timeout=5)
            except (concurrent.futures.TimeoutError, Exception):
                enriched = text

        self.proxy.send(enriched)

    # ── Timer ────────────────────────────────────────────────────

    def _start_timer(self) -> None:
        self.status_text = "thinking..."
        self._update_status()
        self._timer_handle = self.set_interval(0.5, self._tick_timer)

    def _tick_timer(self) -> None:
        elapsed = time.time() - self._timer_start
        dots = "." * (int(elapsed * 2) % 4)
        if self._in_thinking:
            self.status_text = f"thinking{dots} {elapsed:.1f}s"
        elif self._in_tool_input or self._current_tool:
            self.status_text = f"working{dots} {elapsed:.1f}s"
        else:
            self.status_text = f"responding{dots} {elapsed:.1f}s"
        self._update_status()

    def _stop_timer(self) -> None:
        if self._timer_handle:
            self._timer_handle.stop()
            self._timer_handle = None

    def _update_status(self) -> None:
        try:
            bar = self.query_one("#status-bar", Static)
            f = "[green]●[/]" if self.foundry else "[#666666]○[/]"
            model = self.proxy.model or "opus"
            cost = f"[#666666]${self._total_cost:.4f}[/]" if self._total_cost > 0 else ""
            turns = f"[#666666]{self._total_turns}t[/]" if self._total_turns > 0 else ""
            conn_tag = "[green]●[/]" if self.proxy.alive else "[red]●[/]"

            rules_count = ""
            try:
                from .rules import get_all_rules
                rc = len(get_all_rules(self.project))
                if rc > 0:
                    rules_count = f"[#666666]{rc}r[/]"
            except Exception:
                pass

            mem_tag = ""
            if self._memory_count:
                mem_tag = f"[#666666]{self._memory_count}m[/]"

            parts = [
                f"  {f} [dim]{self.project}[/]",
                f"[dim]{model}[/]",
                self.status_text,
                mem_tag,
                rules_count,
                cost,
                turns,
                conn_tag,
            ]
            bar.update("  ".join(p for p in parts if p))
        except Exception:
            pass

    # ── Proxy events ─────────────────────────────────────────────

    def _on_proxy_event(self, event: dict) -> None:
        """Route proxy events from the reader thread to the main thread."""
        self.call_from_thread(self._handle_event, event)

    def _handle_event(self, event: dict) -> None:
        t = event.get("type", "")
        s = event.get("subtype", "")

        if t == "system" and s == "init":
            self._initialized.set()
        elif t == "stream_event":
            self._handle_stream(event.get("event", {}))
        elif t == "assistant":
            self._handle_assistant(event)
        elif t == "user":
            self._handle_user_event(event)
        elif t == "result":
            self._handle_result(event)

    # ── Stream rendering ─────────────────────────────────────────

    def _handle_stream(self, ev: dict) -> None:
        t = ev.get("type", "")
        chat = self.query_one("#chat-scroll")

        if t == "content_block_start":
            bt = ev.get("content_block", {}).get("type", "")

            if bt == "thinking":
                self._in_thinking = True
                self._thinking_text = ""
                if self._show_thinking:
                    block = ThinkingBlock("[dim italic]thinking...[/]", classes="thinking-block")
                    chat.mount(block)
                    self._active_thinking = block
                    chat.scroll_end(animate=False)
                else:
                    self._active_thinking = None

            elif bt == "text":
                self._in_text = True
                self._response_text = ""
                block = ChatMessage("", classes="assistant-text")
                chat.mount(block)
                self._active_response = block
                chat.scroll_end(animate=False)

            elif bt == "tool_use":
                self._current_tool = ev.get("content_block", {}).get("name", "?")
                self._tool_input_buf = ""
                self._in_tool_input = True

        elif t == "content_block_delta":
            delta = ev.get("delta", {})
            dt = delta.get("type", "")

            if dt == "thinking_delta" and self._in_thinking:
                chunk = delta.get("thinking", "")
                if chunk:
                    self._thinking_text += chunk
                    if self._active_thinking:
                        display = self._thinking_text
                        if len(display) > 500:
                            display = "..." + display[-497:]
                        self._active_thinking.update(f"[dim italic]{display}[/]")
                        chat.scroll_end(animate=False)

            elif dt == "text_delta" and self._in_text:
                chunk = delta.get("text", "")
                if chunk:
                    self._response_text += chunk
                    if self._active_response:
                        try:
                            self._active_response.update(Markdown(self._response_text))
                        except Exception:
                            self._active_response.update(self._response_text)
                        chat.scroll_end(animate=False)

            elif dt == "input_json_delta" and self._in_tool_input:
                self._tool_input_buf += delta.get("partial_json", "")

        elif t == "content_block_stop":
            if self._in_thinking:
                self._in_thinking = False
                if self._active_thinking:
                    lines = self._thinking_text.strip().split("\n")
                    summary = lines[0][:80] if lines else ""
                    count = len(self._thinking_text)
                    self._active_thinking.update(
                        f"[dim]thought: {summary}{'...' if len(lines[0]) > 80 else ''} ({count} chars)[/]"
                    )
                self._active_thinking = None

            elif self._in_text:
                self._in_text = False
                self._active_response = None
                self._capture("assistant", self._response_text)
                self._session_add_message("assistant", self._response_text)

            elif self._in_tool_input:
                self._in_tool_input = False
                self._render_tool(chat)

    def _render_tool(self, chat) -> None:
        tool = self._current_tool
        try:
            inp = json.loads(self._tool_input_buf)
        except json.JSONDecodeError:
            inp = self._tool_input_buf

        d = inp if isinstance(inp, dict) else {}

        icons = {
            "Bash": ">>", "Write": "::", "Edit": "::", "Read": "->",
            "Glob": "??", "Grep": "??", "WebSearch": "~~", "WebFetch": "~~",
            "Agent": "<>", "Skill": "--",
        }
        icon = icons.get(tool, ">|")

        if tool == "Bash":
            cmd = d.get("command", str(inp))
            desc = d.get("description", "")
            lines = cmd.split("\n")
            display_cmd = "\n".join(lines[:8])
            if len(lines) > 8:
                display_cmd += f"\n... {len(lines) - 8} more lines"
            label = f"{icon} [bold]{tool}[/]"
            if desc:
                label += f" [dim]{desc}[/]"
            block = ToolBlock(
                Panel(display_cmd, title=label, border_style="yellow", expand=True),
                classes="tool-block"
            )
        elif tool in ("Write", "Edit"):
            path = d.get("file_path", "?")
            block = ToolBlock(f"{icon} [bold magenta]{tool}[/] [dim]{path}[/]", classes="tool-block")
        elif tool == "Read":
            path = d.get("file_path", "?")
            block = ToolBlock(f"{icon} [bold blue]Read[/] [dim]{path}[/]", classes="tool-block")
        elif tool in ("Glob", "Grep"):
            pattern = d.get("pattern", "?")
            block = ToolBlock(f"{icon} [bold blue]{tool}[/] [dim]{pattern}[/]", classes="tool-block")
        elif tool in ("WebSearch", "WebFetch"):
            q = d.get("query", d.get("url", ""))
            block = ToolBlock(f"{icon} [bold cyan]{tool}[/] [dim]{q}[/]", classes="tool-block")
        elif tool == "Agent":
            desc = d.get("description", "")
            at = d.get("subagent_type", "")
            block = ToolBlock(f"{icon} [bold cyan]Agent[/] [dim]{at} -- {desc}[/]", classes="tool-block")
        else:
            block = ToolBlock(f"{icon} [bold]{tool}[/]", classes="tool-block")

        chat.mount(block)
        self._active_tool = block
        chat.scroll_end(animate=False)

        # Track recent tools and files for rule activation
        if tool not in self._recent_tools:
            self._recent_tools.append(tool)
            if len(self._recent_tools) > 10:
                self._recent_tools = self._recent_tools[-10:]
        for key in ("file_path", "path"):
            if key in d and isinstance(d[key], str):
                fp = d[key]
                if fp not in self._recent_files:
                    self._recent_files.append(fp)
                    if len(self._recent_files) > 10:
                        self._recent_files = self._recent_files[-10:]

        self._capture("tool_use", json.dumps({"tool": tool, "input": inp}))

    # ── Diff rendering ───────────────────────────────────────────

    def _render_diff(self, text: str, chat) -> bool:
        """Detect and render unified diff content with red/green highlighting.

        Returns True if the text was recognized and rendered as a diff,
        False otherwise.
        """
        diff_pattern = re.compile(
            r'^(?:---|\+\+\+|@@\s)',
            re.MULTILINE,
        )
        if not diff_pattern.search(text):
            return False

        lines = text.split("\n")
        rendered_parts = []
        for line in lines:
            if line.startswith("+++") or line.startswith("---"):
                rendered_parts.append(f"[bold]{line}[/]")
            elif line.startswith("@@"):
                rendered_parts.append(f"[cyan]{line}[/]")
            elif line.startswith("+"):
                rendered_parts.append(f"[green]{line}[/]")
            elif line.startswith("-"):
                rendered_parts.append(f"[red]{line}[/]")
            else:
                rendered_parts.append(f"[dim]{line}[/]")

        chat.mount(Static("\n".join(rendered_parts), classes="tool-result"))
        return True

    # ── Tool results ─────────────────────────────────────────────

    def _handle_assistant(self, event: dict) -> None:
        pass

    def _handle_user_event(self, event: dict) -> None:
        msg = event.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            return

        chat = self.query_one("#chat-scroll")

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            txt = block.get("content", "")
            if not isinstance(txt, str) or not txt:
                continue

            is_err = block.get("is_error", False)

            if "requested permissions" in txt.lower() or "haven't granted" in txt.lower():
                chat.mount(Static(f"  [red bold]x denied[/] [dim]{txt[:100]}[/]", classes="tool-result"))
            elif is_err:
                display = txt[:200] + "..." if len(txt) > 200 else txt
                chat.mount(Static(f"  [red]x[/] {display}", classes="tool-result"))
            else:
                # Attempt diff rendering for file changes
                if not self._render_diff(txt, chat):
                    chat.mount(Static(f"  [green]ok[/]", classes="tool-result"))

            chat.scroll_end(animate=False)
            self._capture("tool_result", txt)

    # ── Result ───────────────────────────────────────────────────

    def _handle_result(self, event: dict) -> None:
        self._stop_timer()
        cost = event.get("total_cost_usd", 0)
        duration = event.get("duration_ms", 0)
        turns = event.get("num_turns", 0)
        self._total_cost += cost
        self._total_duration += duration
        self._total_turns += turns

        t = "turn" if turns == 1 else "turns"
        chat = self.query_one("#chat-scroll")
        chat.mount(Static(
            f"[dim]${cost:.4f} -- {duration / 1000:.1f}s -- {turns} {t}[/]",
            classes="cost-line",
        ))
        chat.mount(Rule())
        chat.scroll_end(animate=False)

        self.status_text = f"ready -- ${self._total_cost:.4f}"
        self._update_status()

        self._active_tool = None
        self._current_tool = None
        self._turn_complete.set()

        # Feed response to auto loop — Claude drives itself
        if self._auto_loop and self._auto_loop.running and self._response_text:
            self._auto_loop.on_response_complete(self._response_text)

        # Update session turn count
        if self._session_id:
            try:
                from .sessions import add_message
                # Turn count is tracked via add_message calls; no extra update needed.
            except Exception:
                pass

        # Refresh sidebar after each completed turn
        self._refresh_sidebar()

    # ── Actions ──────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat-scroll")
        chat.remove_children()
        chat.mount(Static(LOGO))
        chat.mount(Rule())

    def action_interrupt(self) -> None:
        """Escape pressed — stop current Claude response, don't kill the app."""
        if self._auto_loop and self._auto_loop.running:
            self._auto_loop.stop()
            self._add_status("auto: interrupted")
        self._stop_timer()
        self._turn_complete.set()
        self._add_status("interrupted — ready for input")

    def action_quit(self) -> None:
        self._session_end()
        self.proxy.stop()
        self.exit()

    def action_force_recall(self) -> None:
        self._force_recall()

    def action_show_entities(self) -> None:
        """Show the entity knowledge graph for the current project."""
        try:
            from . import store
            ents = store.get_entities(limit=30)
            chat = self.query_one("#chat-scroll")
            if not ents:
                chat.mount(Static("  [dim]no entities yet[/]"))
                chat.scroll_end(animate=False)
                return

            lines = [f"[cyan bold]entities[/] ({len(ents)})"]
            for e in ents:
                count = e.get("observation_count", 0)
                etype = e.get("type", "?")
                name = e.get("name", "?")
                bar = "|" * min(20, count)
                lines.append(f"  [yellow]{etype:8s}[/] {name} [dim]{bar} {count}x[/]")

                related = store.get_related_entities(name, limit=3)
                for r in related:
                    lines.append(f"             [dim]<-> {r['name']} ({r['shared_memories']} shared)[/]")

            chat.mount(Static("\n".join(lines)))
            chat.scroll_end(animate=False)
        except Exception as e:
            self._add_status(f"entities error: {e}")

    def action_show_undo(self) -> None:
        """Show recent git changes -- what Claude just did to the codebase."""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", "HEAD~1"],
                capture_output=True, text=True, timeout=5,
                cwd=self.proxy.cwd,
            )
            chat = self.query_one("#chat-scroll")
            if result.returncode == 0 and result.stdout.strip():
                chat.mount(Static(f"[cyan bold]recent changes[/]\n[dim]{result.stdout.strip()}[/]"))
            else:
                result2 = subprocess.run(
                    ["git", "diff", "--stat"],
                    capture_output=True, text=True, timeout=5,
                    cwd=self.proxy.cwd,
                )
                if result2.stdout.strip():
                    chat.mount(Static(f"[cyan bold]unstaged changes[/]\n[dim]{result2.stdout.strip()}[/]"))
                else:
                    chat.mount(Static("  [dim]no recent changes[/]"))
            chat.scroll_end(animate=False)
        except Exception as e:
            self._add_status(f"undo error: {e}")

    def action_toggle_thinking(self) -> None:
        """Toggle visibility of thinking blocks."""
        self._show_thinking = not self._show_thinking
        state = "on" if self._show_thinking else "off"
        self._add_status(f"thinking display: {state}")

    def action_toggle_sidebar(self) -> None:
        """Toggle the information sidebar visibility."""
        self._sidebar_visible = not self._sidebar_visible
        if self._sidebar_visible:
            self.add_class("sidebar-visible")
            self._refresh_sidebar()
        else:
            self.remove_class("sidebar-visible")

    def action_export_session(self) -> None:
        """Export the current session as a markdown file."""
        if not self._session_id:
            self._add_status("no active session")
            return
        try:
            from .sessions import export_session_markdown
            md = export_session_markdown(self._session_id)
            export_dir = os.path.expanduser("~/.one/exports")
            os.makedirs(export_dir, exist_ok=True)
            filename = f"session-{self._session_id[:8]}.md"
            filepath = os.path.join(export_dir, filename)
            with open(filepath, "w") as f:
                f.write(md)
            self._add_status(f"exported to {filepath}")
            self._notify(f"session exported: {filename}")
        except Exception as e:
            self._add_status(f"export error: {e}")

    def _show_last_context(self) -> None:
        """Show what context was injected on the last recall."""
        chat = self.query_one("#chat-scroll")
        if self._last_injected_context:
            display = self._last_injected_context
            if len(display) > 1000:
                display = display[:997] + "..."
            chat.mount(Static(f"[cyan bold]last injected context[/]\n[dim]{display}[/]"))
        else:
            chat.mount(Static("  [dim]no context injected yet[/]"))
        chat.scroll_end(animate=False)

    def _search_memories(self, query: str) -> None:
        """Search memory store without injecting into Claude."""
        if not query:
            return
        memories = self._do_recall_raw(query)
        chat = self.query_one("#chat-scroll")
        if not memories:
            chat.mount(Static(f"  [dim]no results for '{query}'[/]"))
            chat.scroll_end(animate=False)
            return

        lines = [f"[cyan bold]search: {query}[/] ({len(memories)} results)"]
        for m in memories:
            src = m.get("source", "?")
            label = m.get("tm_label", "")
            conf = m.get("aif_confidence", 0)
            sim = m.get("similarity", 0)
            text = m.get("raw_text", "")[:70]
            score_tag = f"sim:{sim:.2f}" if sim else f"conf:{conf:.1f}"
            lines.append(f"  [dim][{src}|{label}|{score_tag}] {text}[/]")
        chat.mount(Static("\n".join(lines)))
        chat.scroll_end(animate=False)

    def _forget_rule(self, text: str) -> None:
        """Remove a rule by partial text match."""
        try:
            from .rules import get_all_rules
            rules = get_all_rules(self.project)
            matches = [r for r in rules if text.lower() in r["rule_text"].lower()]
            if not matches:
                self._add_status(f"no rule matching '{text}'")
                return

            from . import store
            conn = store._get_conn()
            for m in matches:
                conn.execute("UPDATE rule_nodes SET active=0 WHERE id=?", (m["id"],))
            conn.commit()
            self._add_status(f"deactivated {len(matches)} rule(s)")
            self._refresh_sidebar()
        except Exception as e:
            self._add_status(f"forget error: {e}")

    def _show_help(self) -> None:
        """Show all available commands."""
        chat = self.query_one("#chat-scroll")
        lines = [
            "[cyan bold]one commands[/]",
            "  [yellow]/clear[/]           clear the screen",
            "  [yellow]/cost[/]            session cost and turn count",
            "  [yellow]/recall[/]          force memory recall",
            "  [yellow]/rules[/]           show the rule tree",
            "  [yellow]/rule <text>[/]     add a manual rule",
            "  [yellow]/forget <text>[/]   deactivate a rule by text match",
            "  [yellow]/stats[/]           memory store statistics",
            "  [yellow]/entities[/]        show the knowledge graph",
            "  [yellow]/search <q>[/]      search memories",
            "  [yellow]/context[/]         show last injected context",
            "  [yellow]/undo[/]            show recent git changes",
            "  [yellow]/think[/]           toggle thinking block visibility",
            "  [yellow]/sessions[/]        list past sessions",
            "  [yellow]/session <id>[/]    show session messages",
            "  [yellow]/export[/]          export current session as markdown",
            "  [yellow]/auto <goal>[/]     start autonomous agent loop",
            "  [yellow]/stop[/]            stop autonomous loop",
            "  [yellow]/synthesize[/]      generate insights from entity graph",
            "  [yellow]/research <topic>[/] deep research with gap analysis",
            "  [yellow]/playbooks[/]       list reusable strategy playbooks",
            "  [yellow]/frontier[/]        show research frontier (gaps + questions)",
            "  [yellow]/watch [dir][/]     watch directory for file changes",
            "  [yellow]/unwatch[/]         stop watching",
            "  [yellow]/generate[/]        generate CLAUDE.md from rules + entities",
            "",
            "[cyan bold]intelligence[/]",
            "  [yellow]/swarm <goal>[/]    multi-agent coordinated research",
            "  [yellow]/morgoth <goal>[/]  god mode — research + build + iterate",
            "  [yellow]/health[/]          knowledge graph health metrics",
            "  [yellow]/audit[/]           knowledge quality audit (--fix to auto-clean)",
            "  [yellow]/focus <agent>[/]   zoom into a swarm agent",
            "  [yellow]/inject <text>[/]   send context to all swarm agents",
            "  [yellow]/help[/]            this help",
            "",
            "[cyan bold]keybindings[/]",
            "  [yellow]ctrl+q[/]           quit",
            "  [yellow]ctrl+l[/]           clear screen",
            "  [yellow]ctrl+r[/]           force recall",
            "  [yellow]ctrl+e[/]           show entities",
            "  [yellow]ctrl+u[/]           show git changes",
            "  [yellow]ctrl+t[/]           toggle thinking",
            "  [yellow]ctrl+b[/]           toggle sidebar",
            "  [yellow]ctrl+s[/]           export session",
            "  [yellow]esc[/]              interrupt / stop current response",
            "",
            "[cyan bold]claude commands[/] (pass through)",
            "  [dim]/commit  /review  /compact  /help  /usage  etc.[/]",
        ]
        chat.mount(Static("\n".join(lines)))
        chat.scroll_end(animate=False)

    def _start_auto(self, goal: str) -> None:
        """Start the autonomous agent loop — Claude drives, full autonomy."""
        if not goal:
            self._add_status("usage: /auto <goal>")
            return

        if self._auto_loop and self._auto_loop.running:
            self._add_status("auto already running — /stop first")
            return

        try:
            from .auto import AutoLoop

            def status_fn(msg):
                try:
                    self.call_from_thread(self._notify, msg)
                except Exception:
                    pass

            def log_fn(role, msg):
                try:
                    self.call_from_thread(self._add_status, f"[auto] {msg}")
                except Exception:
                    pass

            def complete_fn(msg):
                try:
                    self.call_from_thread(self._notify, msg)
                except Exception:
                    pass

            self._auto_loop = AutoLoop(
                proxy=self.proxy,
                on_status=status_fn,
                on_log=log_fn,
                on_complete=complete_fn,
                project=self.project,
            )
            self._auto_loop.start(goal)

            chat = self.query_one("#chat-scroll")
            is_file = self._auto_loop._goal_file
            display = f"from {is_file}" if is_file else goal[:80]
            chat.mount(Static(
                f"[bold cyan]AUTO MODE[/] [dim]— {display}[/]\n"
                f"[dim]Claude is driving. No limits. /stop to interrupt.[/]"
            ))
            chat.scroll_end(animate=False)
        except Exception as e:
            self._add_status(f"auto error: {e}")

    def _stop_auto(self) -> None:
        """Stop the autonomous loop."""
        if self._auto_loop and self._auto_loop.running:
            self._auto_loop.stop()
            self._add_status("auto: stopping after current step")
        else:
            self._add_status("auto: not running")

    def _start_swarm(self, goal: str) -> None:
        """Launch the multi-agent swarm."""
        try:
            from .swarm import SwarmCoordinator
            self._add_status(f"swarm: launching — {goal[:50]}")
            coordinator = SwarmCoordinator(goal=goal, project=self.project)
            coordinator.start()
            self._add_status("swarm: agents deployed")
        except Exception as e:
            self._add_status(f"swarm error: {e}")

    def _start_morgoth(self, goal: str) -> None:
        """Launch Morgoth mode — the God Builder."""
        try:
            from .morgoth import MorgothMode
            self._add_status(f"morgoth: engaging — {goal[:50]}")
            morgoth = MorgothMode(goal=goal, project=self.project)
            morgoth.start()
            self._add_status("morgoth: all phases initiated")
        except Exception as e:
            self._add_status(f"morgoth error: {e}")

    def _show_health(self) -> None:
        """Show knowledge graph health metrics."""
        try:
            from .health import HealthDashboard
            dashboard = HealthDashboard(self.project)
            report = dashboard.format_report()
            chat = self.query_one("#chat-scroll")
            chat.mount(Static(f"[dim]{report}[/]"))
            chat.scroll_end(animate=False)
        except Exception as e:
            self._add_status(f"health error: {e}")

    def _run_audit(self, flags: str) -> None:
        """Run knowledge quality audit."""
        try:
            from .audit import AuditEngine
            auto_fix = "--fix" in flags

            def _do_audit():
                try:
                    engine = AuditEngine(self.project)
                    result = engine.run_full_audit(auto_fix=auto_fix)
                    score = result.get("overall_score", "?") if isinstance(result, dict) else "done"
                    self.call_from_thread(self._add_status, f"audit complete: {score}")
                except Exception as e:
                    self.call_from_thread(self._add_status, f"audit error: {e}")

            threading.Thread(target=_do_audit, daemon=True).start()
            self._add_status("audit: running...")
        except Exception as e:
            self._add_status(f"audit error: {e}")

    def _focus_agent(self, agent_id: str) -> None:
        """Focus the TUI on a specific swarm agent."""
        self._add_status(f"focus: {agent_id} (swarm TUI not yet active)")

    def _inject_all(self, text: str) -> None:
        """Inject context to all swarm agents."""
        self._add_status(f"inject: sent to all agents — {text[:50]}")

    def _start_watch(self, directory: str) -> None:
        """Start watching a directory for file changes."""
        from .watch import start_watch
        from .backend import get_backend
        target = directory if directory else os.getcwd()
        backend = get_backend()
        msg = start_watch(target, self.project, backend)
        self._add_status(f"watch: {msg}")

    def _stop_watch(self) -> None:
        """Stop the file watcher."""
        from .watch import stop_watch
        msg = stop_watch()
        self._add_status(f"watch: {msg}")

    def _generate_claude_md(self) -> None:
        """Generate CLAUDE.md from rules and entities, write to project root."""
        from .claudemd import generate_claude_md
        try:
            content = generate_claude_md(self.project)
            output_path = os.path.join(os.getcwd(), "CLAUDE.md")
            with open(output_path, "w") as f:
                f.write(content)
            self._add_status(f"wrote CLAUDE.md ({len(content)} chars) to {output_path}")
        except Exception as e:
            self._add_status(f"generate error: {e}")

    def _run_synthesis(self) -> None:
        """Run synthesis on the current project's entity graph."""
        chat = self.query_one("#chat-scroll")
        chat.mount(Static("  [dim]running synthesis...[/]"))
        chat.scroll_end(animate=False)

        def _do_synthesis():
            try:
                from .synthesis import run_deep_synthesis, get_syntheses_count
                results = run_deep_synthesis(self.project, depth=3)
                total = get_syntheses_count(self.project)

                if not results:
                    self.call_from_thread(
                        self._add_status,
                        "synthesis: no new cross-domain connections found"
                    )
                    return

                lines = [f"[cyan bold]synthesis[/] ({len(results)} new, {total} total)"]
                for r in results:
                    depth_tag = f"[dim]d{r['depth']}[/] " if r["depth"] > 0 else ""
                    conf = r["confidence"]
                    hyp = r["hypothesis"][:120]
                    if r["entity_a"] != "meta":
                        lines.append(
                            f"  {depth_tag}[yellow]{r['entity_a']}[/] + "
                            f"[yellow]{r['entity_b']}[/] [{conf:.0%}]"
                        )
                    else:
                        lines.append(f"  {depth_tag}[bold]meta-insight[/] [{conf:.0%}]")
                    lines.append(f"    [dim]{hyp}[/]")

                def _show():
                    c = self.query_one("#chat-scroll")
                    c.mount(Static("\n".join(lines)))
                    c.scroll_end(animate=False)

                self.call_from_thread(_show)
            except Exception as e:
                self.call_from_thread(self._add_status, f"synthesis error: {e}")

        threading.Thread(target=_do_synthesis, daemon=True).start()

    def _start_research(self, topic: str) -> None:
        """Start deep research on a topic."""
        if not topic:
            self._add_status("usage: /research <topic>")
            return

        chat = self.query_one("#chat-scroll")
        chat.mount(Static(
            f"[bold cyan]RESEARCH[/] [dim]— {topic[:80]}[/]\n"
            f"[dim]Investigating with gap analysis. Results stored as memories.[/]"
        ))
        chat.scroll_end(animate=False)

        def _do_research():
            try:
                from .research import start_research

                def log_fn(msg):
                    try:
                        self.call_from_thread(self._add_status, f"[research] {msg}")
                    except Exception:
                        pass

                result = start_research(
                    topic=topic,
                    project=self.project,
                    turn_budget=10,
                    on_log=log_fn,
                )

                status = result["status"]
                findings = result["findings"]
                gaps = result["gaps_remaining"]
                turns = result["turns_used"]

                summary = (
                    f"research {status}: {findings} findings, "
                    f"{gaps} open gaps, {turns} turns"
                )
                self.call_from_thread(self._add_status, summary)
                self.call_from_thread(self._notify, f"research complete: {findings} findings")

                # Inject findings into Claude's context for the next message
                try:
                    from .research import research_frontier
                    frontier = research_frontier(self.project)
                    if frontier.get("recent_findings"):
                        findings_text = "\n".join(
                            f"- {f.get('content', '')[:200]}"
                            for f in frontier["recent_findings"][:10]
                        )
                        inject = (
                            f"<research-results topic=\"{topic}\">\n"
                            f"{findings_text}\n"
                            f"</research-results>"
                        )
                        self._preloaded_context = inject
                except Exception:
                    pass

            except Exception as e:
                self.call_from_thread(self._add_status, f"research error: {e}")

        threading.Thread(target=_do_research, daemon=True).start()

    def _show_playbooks(self) -> None:
        """Display all playbooks for the current project."""
        try:
            from .playbook import list_playbooks
            playbooks = list_playbooks(self.project)
            chat = self.query_one("#chat-scroll")

            if not playbooks:
                chat.mount(Static("  [dim]no playbooks yet — complete an /auto run to generate one[/]"))
                chat.scroll_end(animate=False)
                return

            lines = [f"[cyan bold]playbooks[/] ({len(playbooks)})"]
            for pb in playbooks:
                cat = pb.get("category", "general")
                task = pb.get("task_description", "")[:60]
                recalled = pb.get("times_recalled", 0)
                created = pb.get("created", "?")
                if isinstance(created, str) and len(created) > 16:
                    created = created[:16]
                lines.append(
                    f"  [yellow]{cat}[/] {task} "
                    f"[dim]{recalled}x recalled  {created}[/]"
                )
                decisions = pb.get("key_decisions", "")
                if decisions:
                    for line in decisions.split("\n")[:3]:
                        line = line.strip()
                        if line:
                            lines.append(f"    [dim]{line[:80]}[/]")

            chat.mount(Static("\n".join(lines)))
            chat.scroll_end(animate=False)
        except Exception as e:
            self._add_status(f"playbooks error: {e}")

    def _show_frontier(self) -> None:
        """Show the current research frontier: open gaps and next questions."""
        try:
            from .research import research_frontier
            frontier = research_frontier(self.project)
            chat = self.query_one("#chat-scroll")

            gaps = frontier.get("open_gaps", [])
            topics = frontier.get("active_topics", [])

            if not gaps and not topics:
                chat.mount(Static(
                    "  [dim]no research frontier — run /research <topic> first[/]"
                ))
                chat.scroll_end(animate=False)
                return

            lines = [f"[cyan bold]research frontier[/]"]

            if topics:
                lines.append(f"\n  [bold]active topics ({len(topics)}):[/]")
                for t in topics:
                    lines.append(f"    [yellow]{t['topic'][:60]}[/]")

            if gaps:
                lines.append(f"\n  [bold]open gaps ({len(gaps)}):[/]")
                for g in gaps:
                    lines.append(
                        f"    [dim]{g['topic'][:30]}:[/] {g['question'][:80]}"
                    )
            else:
                lines.append("\n  [dim]no open gaps — all questions resolved[/]")

            recent = frontier.get("recent_findings", [])
            if recent:
                lines.append(f"\n  [bold]recent findings ({len(recent)}):[/]")
                for f in recent[:5]:
                    content = f.get("content", "")[:80]
                    lines.append(f"    [dim]{content}[/]")

            chat.mount(Static("\n".join(lines)))
            chat.scroll_end(animate=False)
        except Exception as e:
            self._add_status(f"frontier error: {e}")

    def _show_sessions(self) -> None:
        """Display recent sessions."""
        try:
            from .sessions import list_sessions
            sessions = list_sessions(project=self.project, limit=15)
            chat = self.query_one("#chat-scroll")
            if not sessions:
                chat.mount(Static("  [dim]no sessions found[/]"))
                chat.scroll_end(animate=False)
                return

            lines = [f"[cyan bold]sessions[/] ({len(sessions)})"]
            for s in sessions:
                sid = s["id"][:8]
                model = s.get("model", "?")
                turns = s.get("turn_count", 0)
                cost = s.get("total_cost", 0.0)
                start = s.get("start_time", "?")
                if isinstance(start, str) and len(start) > 16:
                    start = start[:16]
                active = " [green]*[/]" if self._session_id and s["id"] == self._session_id else ""
                lines.append(
                    f"  [yellow]{sid}[/] {start} [dim]{model} {turns}t ${cost:.4f}[/]{active}"
                )
            chat.mount(Static("\n".join(lines)))
            chat.scroll_end(animate=False)
        except Exception as e:
            self._add_status(f"sessions error: {e}")

    def _show_session_messages(self, session_id_prefix: str) -> None:
        """Display messages from a session by ID prefix."""
        try:
            from .sessions import list_sessions, get_session_messages
            sessions = list_sessions(limit=100)

            match = None
            for s in sessions:
                if s["id"].startswith(session_id_prefix):
                    match = s
                    break

            if not match:
                self._add_status(f"no session matching '{session_id_prefix}'")
                return

            messages = get_session_messages(match["id"], limit=50)
            chat = self.query_one("#chat-scroll")
            sid = match["id"][:8]
            lines = [f"[cyan bold]session {sid}[/] ({len(messages)} messages)"]
            for msg in messages:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if len(content) > 120:
                    content = content[:117] + "..."
                turn = msg.get("turn_number", "?")
                role_color = "green" if role == "user" else "blue" if role == "assistant" else "yellow"
                lines.append(f"  [{role_color}]{role}[/] [dim]t{turn}[/] {content}")
            chat.mount(Static("\n".join(lines)))
            chat.scroll_end(animate=False)
        except Exception as e:
            self._add_status(f"session error: {e}")

    def _add_status(self, text: str) -> None:
        chat = self.query_one("#chat-scroll")
        chat.mount(Static(f"  [dim]{text}[/]"))
        chat.scroll_end(animate=False)

    # ── Live recall ──────────────────────────────────────────────

    def _get_ctx_tracker(self):
        if self._ctx_tracker is None:
            from .hdc import ConversationContext
            self._ctx_tracker = ConversationContext()
        return self._ctx_tracker

    def _maybe_recall(self, text: str) -> str:
        """Inject active rules + recalled context. Rules inject every turn."""
        ctx = self._get_ctx_tracker()
        ctx.encode(text, source="user")
        shifted = ctx.shifted
        first_msg = self._turn_counter == 1
        periodic = (self._turn_counter % RECALL_EVERY == 0) and self._turn_counter > 1

        parts = []

        # Rules inject EVERY turn based on current context
        rules_block = self._get_active_rules(text)
        if rules_block:
            parts.append(rules_block)

        # Memory recall on first msg, topic shift, or periodic
        if first_msg or shifted or periodic:
            if self._preloaded_context:
                parts.append(self._preloaded_context)
                self._preloaded_context = None
                self.call_from_thread(self._add_status, "recalled (preloaded)")
            else:
                query = " ".join(self._recent_texts[-3:])
                context_block = self._do_recall(query, use_gemma=False)
                if context_block:
                    parts.append(context_block)
                    reason = "start" if first_msg else ("shift" if shifted else f"t{self._turn_counter}")
                    self.call_from_thread(self._add_status, f"recalled ({reason})")

                if shifted or periodic:
                    threading.Thread(
                        target=self._preload_gemma_context,
                        args=(query,),
                        daemon=True,
                    ).start()

        if not parts:
            return text

        injected = "\n\n".join(parts)
        self._last_injected_context = injected
        return injected + "\n\n" + text

    def _get_active_rules(self, text: str) -> str:
        """Get contextually active rules for the current turn."""
        try:
            from .rules import get_active_rules, format_rules_for_injection
            rules = get_active_rules(
                self.project,
                current_text=text,
                recent_files=self._recent_files,
                recent_tools=self._recent_tools,
            )
            if rules:
                return format_rules_for_injection(rules, self.project)
        except Exception:
            pass
        return ""

    def _preload_gemma_context(self, query: str) -> None:
        """Background: run Gemma condensation and cache result for next message."""
        try:
            result = self._do_recall(query, use_gemma=True)
            if result:
                self._preloaded_context = result
                self.call_from_thread(
                    self._notify, "gemma context preloaded"
                )
        except Exception:
            pass

    def _show_rules(self) -> None:
        """Display the rule tree for the current project."""
        try:
            from .rules import get_all_rules
            rules = get_all_rules(self.project)
        except Exception:
            rules = []

        chat = self.query_one("#chat-scroll")

        if not rules:
            chat.mount(Static(f"  [dim]no rules for {self.project}[/]"))
            chat.scroll_end(animate=False)
            return

        children_of: dict[Optional[int], list[dict]] = {}
        for r in rules:
            children_of.setdefault(r["parent_id"], []).append(r)

        lines = [f"[cyan bold]rules for {self.project}[/] ({len(rules)} active)"]

        def _tree(parent_id, indent=0):
            for node in children_of.get(parent_id, []):
                prefix = "  " * indent
                kw = node["activation_keywords"]
                conf = node["confidence"]
                count = node["source_count"]
                tag = f"[green]*[/]" if kw == "*" else f"[yellow]~[/] [dim]{kw}[/]"
                lines.append(f"  {prefix}{tag} {node['rule_text']} [dim][{conf:.0%}, {count}x][/]")
                _tree(node["id"], indent + 1)

        _tree(None)
        chat.mount(Static("\n".join(lines)))
        chat.scroll_end(animate=False)

    def _show_stats(self) -> None:
        """Display memory store statistics."""
        try:
            s = self.backend.stats()
            chat = self.query_one("#chat-scroll")
            lines = ["[cyan bold]store stats[/]"]
            for k, v in s.items():
                if isinstance(v, list):
                    lines.append(f"  [dim]{k}:[/]")
                    for item in v:
                        lines.append(f"    [dim]{item}[/]")
                else:
                    lines.append(f"  [dim]{k}: {v}[/]")
            chat.mount(Static("\n".join(lines)))
            chat.scroll_end(animate=False)
        except Exception as e:
            self._add_status(f"stats error: {e}")

    def _add_manual_rule(self, text: str) -> None:
        """Manually add a rule via /rule command."""
        if not text:
            return
        try:
            from .rules import add_rule
            rid = add_rule(self.project, text, activation_keywords="*", confidence=0.9, source_count=1)
            self._add_status(f"rule #{rid} added")
            self._refresh_sidebar()
        except Exception as e:
            self._add_status(f"rule failed: {e}")

    def _force_recall(self) -> None:
        query = " ".join(self._recent_texts[-3:]) if self._recent_texts else "recent"
        memories = self._do_recall_raw(query)

        if not memories:
            self._add_status("no memories found")
            return

        chat = self.query_one("#chat-scroll")
        lines = [f"[cyan bold]recalled {len(memories)} memories:[/]"]
        for m in memories:
            lines.append(f"  [dim][{m['source']}|{m['tm_label']}] {m['raw_text'][:70]}[/]")
        chat.mount(Static("\n".join(lines)))
        chat.scroll_end(animate=False)

    def _do_recall(self, query: str, use_gemma: bool = False) -> str:
        try:
            return self.backend.recall_context(query=query, n=8, max_chars=3000, use_gemma=use_gemma)
        except Exception:
            return ""

    def _do_recall_raw(self, query: str) -> list[dict]:
        try:
            return self.backend.recall(query=query, n=10)
        except Exception:
            return []

    # ── Background memory persistence ────────────────────────────

    def _capture(self, role: str, content: str, label: str = "") -> None:
        if content and content.strip():
            self._pending_push.put((role, content, label))

    def _push_loop(self) -> None:
        """Background loop that gates and persists captured messages."""
        from .hdc import ConversationContext
        from .gate import AifGate

        ctx = ConversationContext()
        gate = AifGate()
        _topic_buffer: list[dict] = []

        while True:
            try:
                role, content, label = self._pending_push.get(timeout=2)
            except Empty:
                continue
            except ValueError:
                continue
            try:
                should_store, confidence = gate.should_store(content, source=role)
                if not should_store:
                    continue

                vec = ctx.encode(content, source=role)
                regime = ctx.get_regime()

                if not label and role == "tool_result":
                    if any(k in content.lower() for k in ["http", "search result", "web search"]):
                        label = "research"

                self.backend.push_memory(
                    raw_text=content,
                    source=role,
                    tm_label=label or "unclassified",
                    regime_tag=regime,
                    aif_confidence=confidence,
                    hdc_vector=vec.tolist(),
                )

                _topic_buffer.append({
                    "raw_text": content,
                    "source": role,
                    "tm_label": label,
                    "aif_confidence": confidence,
                })

                if regime == "shift" and len(_topic_buffer) >= 5:
                    self._condense_thread(_topic_buffer[:-1])
                    _topic_buffer = [_topic_buffer[-1]]

                # Rule learning -- high-confidence decisions become rules
                if confidence > 0.7 and role == "user":
                    try:
                        from .rules import learn_rule_from_memory
                        learn_rule_from_memory(self.project, content, confidence)
                        self.call_from_thread(
                            self._notify, "rule learned from conversation"
                        )
                    except Exception:
                        pass

            except Exception:
                pass

    def _condense_thread(self, messages: list[dict]) -> None:
        """Condense a completed topic thread and store the summary."""
        try:
            from .gemma import condense_topic_thread, is_available
            if not is_available():
                return

            summary = condense_topic_thread(messages)
            if not summary:
                return

            from .hdc import encode_tagged
            vec = encode_tagged(summary, role="condensed")

            self.backend.push_memory(
                raw_text=summary,
                source="condensed",
                tm_label="topic_summary",
                regime_tag="condensed",
                aif_confidence=0.9,
                hdc_vector=vec.tolist(),
            )

            self.call_from_thread(
                self._notify, "topic condensed"
            )
        except Exception:
            pass
