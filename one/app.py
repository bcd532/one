"""Textual-based TUI application for One.

Provides a rich terminal interface with streaming output rendering,
tool call visualization, live memory recall, and background memory
persistence via the configured storage backend.
"""

import json
import time
import threading
from queue import Queue, Empty
from typing import Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, RichLog, Rule
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Group
from rich.spinner import Spinner

from .proxy import ClaudeProxy

LOGO = """[bold cyan]
 ██████╗ ███╗   ██╗███████╗
██╔═══██╗████╗  ██║██╔════╝
██║   ██║██╔██╗ ██║█████╗
██║   ██║██║╚██╗██║██╔══╝
╚██████╔╝██║ ╚████║███████╗
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝[/]"""

RECALL_EVERY = 5


class ChatMessage(Static):
    pass


class ToolBlock(Static):
    pass


class ThinkingBlock(Static):
    pass


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

    #chat-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
        scrollbar-color: #444444;
        scrollbar-color-hover: #666666;
        padding: 0 2;
        background: transparent;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: transparent;
        color: #888888;
        padding: 0 2;
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
    """

    BINDINGS = [
        ("escape", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
        ("ctrl+r", "force_recall", "Recall"),
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

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-scroll")
        yield Static("", id="status-bar")
        yield Input(placeholder="message one...", id="input-box")

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
            "    [#444444]esc:quit  ctrl+l:clear  ctrl+r:recall  /rules  /stats  /cost[/]\n"
            "    [#444444]claude: /commit  /review  /compact  /help  — all pass through[/]"
        ))
        chat.mount(Rule())

        self.proxy.on_event(self._on_proxy_event)
        self.proxy.start()

        threading.Thread(target=self._push_loop, daemon=True).start()

        # Sync rules from Foundry to local cache (background, non-blocking)
        if self.foundry:
            threading.Thread(target=self._sync_rules, daemon=True).start()

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

    # ── Input handling ──────────────────────────────────────────────

    @on(Input.Submitted, "#input-box")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        event.input.value = ""

        # one commands
        ONE_COMMANDS = {"/quit", "/exit", "/q", "/clear", "/cost", "/recall", "/rules", "/stats"}

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
        if text.startswith("/rule "):
            self._add_manual_rule(text[6:].strip())
            return

        # Anything starting with / that isn't ours → pass to Claude as-is
        # This handles /commit, /review, /compact, /help, /usage, etc.
        if text.startswith("/") and text.split()[0] not in ONE_COMMANDS:
            self._send_message(text)
            return

        self._send_message(text)

    def _send_message(self, text: str) -> None:
        self._turn_counter += 1
        self._recent_texts.append(text)
        if len(self._recent_texts) > 5:
            self._recent_texts = self._recent_texts[-5:]

        # Show message and start timer IMMEDIATELY
        chat = self.query_one("#chat-scroll")
        chat.mount(ChatMessage(f"[bold]{text}[/]", classes="user-msg"))
        chat.scroll_end(animate=False)
        self._timer_start = time.time()
        self._start_timer()
        self._capture("user", text)

        # All recall + send happens in background — UI never blocks
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
                enriched = text  # skip recall, send raw

        self.proxy.send(enriched)

    # ── Timer ───────────────────────────────────────────────────────

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
            cost = f"[#666666]${self._total_cost:.4f}[/]" if self._total_cost > 0 else ""
            turns = f"[#666666]{self._total_turns}t[/]" if self._total_turns > 0 else ""
            parts = [f"  {f}", self.status_text, cost, turns]
            bar.update("  ".join(p for p in parts if p))
        except Exception:
            pass

    # ── Proxy events ────────────────────────────────────────────────

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

    # ── Stream rendering ────────────────────────────────────────────

    def _handle_stream(self, ev: dict) -> None:
        t = ev.get("type", "")
        chat = self.query_one("#chat-scroll")

        if t == "content_block_start":
            bt = ev.get("content_block", {}).get("type", "")

            if bt == "thinking":
                self._in_thinking = True
                self._thinking_text = ""
                block = ThinkingBlock("[dim italic]thinking...[/]", classes="thinking-block")
                chat.mount(block)
                self._active_thinking = block
                chat.scroll_end(animate=False)

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
                        f"[dim]💭 {summary}{'...' if len(lines[0]) > 80 else ''} ({count} chars)[/]"
                    )
                self._active_thinking = None

            elif self._in_text:
                self._in_text = False
                self._active_response = None
                self._capture("assistant", self._response_text)

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
            "Bash": "⚡", "Write": "✏", "Edit": "✏", "Read": "📄",
            "Glob": "🔍", "Grep": "🔍", "WebSearch": "🌐", "WebFetch": "🌐",
            "Agent": "🤖", "Skill": "⚙",
        }
        icon = icons.get(tool, "🔧")

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
            block = ToolBlock(f"{icon} [bold cyan]Agent[/] [dim]{at} — {desc}[/]", classes="tool-block")
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

    # ── Tool results ────────────────────────────────────────────────

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
                chat.mount(Static(f"  [red bold]✗ denied[/] [dim]{txt[:100]}[/]", classes="tool-result"))
            elif is_err:
                display = txt[:200] + "..." if len(txt) > 200 else txt
                chat.mount(Static(f"  [red]✗[/] {display}", classes="tool-result"))
            else:
                chat.mount(Static(f"  [green]✓[/]", classes="tool-result"))

            chat.scroll_end(animate=False)
            self._capture("tool_result", txt)

    # ── Result ──────────────────────────────────────────────────────

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
            f"[dim]${cost:.4f} · {duration / 1000:.1f}s · {turns} {t}[/]",
            classes="cost-line",
        ))
        chat.mount(Rule())
        chat.scroll_end(animate=False)

        self.status_text = f"ready · ${self._total_cost:.4f}"
        self._update_status()

        self._active_tool = None
        self._current_tool = None
        self._turn_complete.set()

    # ── Actions ─────────────────────────────────────────────────────

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat-scroll")
        chat.remove_children()
        chat.mount(Static(LOGO))
        chat.mount(Rule())

    def action_quit(self) -> None:
        self.proxy.stop()
        self.exit()

    def action_force_recall(self) -> None:
        self._force_recall()

    def _add_status(self, text: str) -> None:
        chat = self.query_one("#chat-scroll")
        chat.mount(Static(f"  [dim]{text}[/]"))
        chat.scroll_end(animate=False)

    # ── Live recall ─────────────────────────────────────────────────

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
                self._add_status("↻ recalled (preloaded)")
            else:
                query = " ".join(self._recent_texts[-3:])
                context_block = self._do_recall(query, use_gemma=False)
                if context_block:
                    parts.append(context_block)
                    reason = "start" if first_msg else ("shift" if shifted else f"t{self._turn_counter}")
                    self._add_status(f"↻ recalled ({reason})")

                if shifted or periodic:
                    threading.Thread(
                        target=self._preload_gemma_context,
                        args=(query,),
                        daemon=True,
                    ).start()

        if not parts:
            return text

        return "\n\n".join(parts) + "\n\n" + text

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

        # Build tree display
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
                tag = f"[green]●[/]" if kw == "*" else f"[yellow]◐[/] [dim]{kw}[/]"
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

    # ── Background memory persistence ───────────────────────────────

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

                # Rule learning — high-confidence decisions become rules
                if confidence > 0.7 and role == "user":
                    try:
                        from .rules import learn_rule_from_memory
                        learn_rule_from_memory(self.project, content, confidence)
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
        except Exception:
            pass
