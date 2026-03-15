"""Swarm TUI Dashboard — Watch the Palantír.

System 8: Real-time visualization of multi-agent swarm operations.

Provides:
- Per-agent status rows with sparkline activity trends
- Real-time breakthrough alerts
- Contradiction panel with resolution status
- Live dialectic display (thesis/antithesis/synthesis)
- Stats panel: findings, hypotheses, contradictions, etc.
- Commands: /focus, /inject, /stop, /scale, /health, /frontier
"""

from datetime import datetime, timezone
from typing import Optional, Callable, Any

from textual.app import ComposeResult
from textual.widgets import Static, RichLog
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual.timer import Timer
from rich.text import Text
from rich.panel import Panel
from rich.table import Table


# ── Sparkline Helper ───────────────────────────────────────────

def _sparkline(values: list[float], width: int = 12) -> str:
    """Render a compact sparkline from a list of values."""
    if not values:
        return "▁" * width
    bars = "▁▂▃▄▅▆▇█"
    mn = min(values)
    mx = max(values)
    rng = mx - mn if mx != mn else 1.0
    recent = values[-width:]
    return "".join(bars[min(len(bars) - 1, int((v - mn) / rng * (len(bars) - 1)))] for v in recent)


# ── Agent Status Row ───────────────────────────────────────────

class AgentStatusWidget(Static):
    """Displays a single agent's status with sparkline."""

    def __init__(self, agent_id: str, role: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.role = role
        self.turn = 0
        self.activity = ""
        self.history: list[float] = []
        self.alive = True

    def render(self) -> Text:
        spark = _sparkline(self.history)
        indicator = "●" if self.alive else "○"
        color = "green" if self.alive else "red"
        return Text.from_markup(
            f"[{color}]{indicator}[/] [{color} bold]{self.role:<14}[/] "
            f"{spark}  T{self.turn:<4} {self.activity[:40]}"
        )

    def update_status(self, turn: int, activity: str, metric: float = 1.0) -> None:
        self.turn = turn
        self.activity = activity
        self.history.append(metric)
        if len(self.history) > 60:
            self.history = self.history[-60:]
        self.refresh()


# ── Stats Panel ────────────────────────────────────────────────

class StatsPanel(Static):
    """Displays aggregate swarm statistics."""

    findings: reactive[int] = reactive(0)
    hypotheses: reactive[int] = reactive(0)
    contradictions: reactive[int] = reactive(0)
    syntheses: reactive[int] = reactive(0)
    experiments: reactive[int] = reactive(0)
    patterns: reactive[int] = reactive(0)
    coverage: reactive[float] = reactive(0.0)
    avg_confidence: reactive[float] = reactive(0.0)

    def render(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("metric", style="dim")
        table.add_column("value", style="bold cyan")
        table.add_row("Findings", str(self.findings))
        table.add_row("Hypotheses", str(self.hypotheses))
        table.add_row("Contradictions", str(self.contradictions))
        table.add_row("Syntheses", str(self.syntheses))
        table.add_row("Experiments", str(self.experiments))
        table.add_row("Patterns", str(self.patterns))
        table.add_row("Coverage", f"{self.coverage:.0%}")
        table.add_row("Avg Confidence", f"{self.avg_confidence:.2f}")
        return Panel(table, title="[bold]STATS[/]", border_style="cyan")


# ── Breakthrough Panel ─────────────────────────────────────────

class BreakthroughPanel(Static):
    """Displays the latest breakthrough finding."""

    latest: reactive[str] = reactive("")

    def render(self) -> Panel:
        text = self.latest if self.latest else "(waiting for breakthroughs...)"
        return Panel(
            Text(text[:200], style="bold yellow"),
            title="[bold yellow]LATEST BREAKTHROUGH[/]",
            border_style="yellow",
        )


# ── Contradiction Panel ────────────────────────────────────────

class ContradictionPanel(Static):
    """Displays active contradictions with severity."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._items: list[dict[str, str]] = []

    def render(self) -> Panel:
        if not self._items:
            text = Text("(no active contradictions)", style="dim")
        else:
            lines = []
            for c in self._items[:5]:
                severity = c.get("severity", "?").upper()
                color = {"PARADIGM": "red", "CRITICAL": "yellow", "MODERATE": "cyan"}.get(severity, "white")
                status = c.get("status", "active")
                lines.append(f"[{color}]{severity}[/] {c.get('finding_a', '')[:50]}... [{status}]")
            text = Text.from_markup("\n".join(lines))
        return Panel(text, title="[bold red]CONTRADICTIONS[/]", border_style="red")

    def update_items(self, items: list[dict[str, str]]) -> None:
        self._items = items
        self.refresh()


# ── Dialectic Panel ────────────────────────────────────────────

class DialecticPanel(Static):
    """Shows the current dialectic chain in progress."""

    thesis: reactive[str] = reactive("")
    antithesis: reactive[str] = reactive("")
    synthesis: reactive[str] = reactive("")

    def render(self) -> Panel:
        parts = []
        if self.thesis:
            parts.append(f"[bold green]THESIS:[/] {self.thesis[:100]}")
        if self.antithesis:
            parts.append(f"[bold red]ANTITHESIS:[/] {self.antithesis[:100]}")
        if self.synthesis:
            parts.append(f"[bold yellow]SYNTHESIS:[/] {self.synthesis[:100]}")
        text = "\n".join(parts) if parts else "(no dialectic in progress)"
        return Panel(
            Text.from_markup(text),
            title="[bold]DIALECTIC[/]",
            border_style="magenta",
        )


# ── Main Swarm View ────────────────────────────────────────────

class SwarmView(Vertical):
    """Full swarm dashboard view — replaces chat when swarm is active."""

    DEFAULT_CSS = """
    SwarmView {
        height: 100%;
        width: 100%;
    }

    SwarmView #swarm-header {
        height: 1;
        background: $primary-darken-2;
        color: $text;
        text-align: center;
    }

    SwarmView #agent-list {
        height: auto;
        max-height: 50%;
        border-bottom: solid $primary-darken-1;
    }

    SwarmView #panels-row {
        height: auto;
        min-height: 10;
    }

    SwarmView #log-panel {
        height: 1fr;
        border-top: solid $primary-darken-1;
    }

    SwarmView #swarm-footer {
        height: 1;
        background: $primary-darken-2;
        color: $text-muted;
        text-align: center;
    }
    """

    def __init__(
        self,
        goal: str = "",
        on_command: Optional[Callable[..., object]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.goal = goal
        self._on_command = on_command
        self._agent_widgets: dict[str, AgentStatusWidget] = {}
        self._start_time = datetime.now(timezone.utc)
        self._refresh_timer: Optional[Timer] = None

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), id="swarm-header")
        yield Vertical(id="agent-list")
        yield Horizontal(
            StatsPanel(id="stats-panel"),
            BreakthroughPanel(id="breakthrough-panel"),
            id="panels-row",
        )
        yield Horizontal(
            ContradictionPanel(id="contradiction-panel"),
            DialecticPanel(id="dialectic-panel"),
            id="panels-row-2",
        )
        yield RichLog(id="log-panel", highlight=True, markup=True)
        yield Static(
            "/focus <agent>  /inject <text>  /stop  /scale <n>  /health  /frontier",
            id="swarm-footer",
        )

    def on_mount(self) -> None:
        self._refresh_timer = self.set_interval(2.0, self._refresh_header)

    def _header_text(self) -> str:
        elapsed = datetime.now(timezone.utc) - self._start_time
        mins = int(elapsed.total_seconds() // 60)
        secs = int(elapsed.total_seconds() % 60)
        n = len(self._agent_widgets)
        return f" SWARM: \"{self.goal[:40]}\" — {n} agents — {mins:02d}:{secs:02d} "

    def _refresh_header(self) -> None:
        header = self.query_one("#swarm-header", Static)
        header.update(self._header_text())

    # ── Public API for SwarmCoordinator to call ─────────────────

    def add_agent(self, agent_id: str, role: str) -> None:
        """Register a new agent in the dashboard."""
        widget = AgentStatusWidget(agent_id, role)
        self._agent_widgets[agent_id] = widget
        agent_list = self.query_one("#agent-list", Vertical)
        agent_list.mount(widget)

    def remove_agent(self, agent_id: str) -> None:
        """Mark an agent as dead."""
        w = self._agent_widgets.get(agent_id)
        if w:
            w.alive = False
            w.refresh()

    def update_agent(self, agent_id: str, turn: int, activity: str, metric: float = 1.0) -> None:
        """Update an agent's status row."""
        w = self._agent_widgets.get(agent_id)
        if w:
            w.update_status(turn, activity, metric)

    def update_stats(
        self,
        findings: int = 0,
        hypotheses: int = 0,
        contradictions: int = 0,
        syntheses: int = 0,
        experiments: int = 0,
        patterns: int = 0,
        coverage: float = 0.0,
        avg_confidence: float = 0.0,
    ) -> None:
        """Update the stats panel."""
        panel = self.query_one("#stats-panel", StatsPanel)
        panel.findings = findings
        panel.hypotheses = hypotheses
        panel.contradictions = contradictions
        panel.syntheses = syntheses
        panel.experiments = experiments
        panel.patterns = patterns
        panel.coverage = coverage
        panel.avg_confidence = avg_confidence

    def show_breakthrough(self, text: str) -> None:
        """Flash a breakthrough in the breakthrough panel."""
        panel = self.query_one("#breakthrough-panel", BreakthroughPanel)
        panel.latest = text

    def update_contradictions(self, items: list[dict[str, str]]) -> None:
        """Update the contradiction panel."""
        panel = self.query_one("#contradiction-panel", ContradictionPanel)
        panel.update_items(items)

    def update_dialectic(
        self,
        thesis: str = "",
        antithesis: str = "",
        synthesis: str = "",
    ) -> None:
        """Update the dialectic panel."""
        panel = self.query_one("#dialectic-panel", DialecticPanel)
        panel.thesis = thesis
        panel.antithesis = antithesis
        panel.synthesis = synthesis

    def write_log(self, message: str) -> None:
        """Append a message to the scrolling log."""
        log_widget = self.query_one("#log-panel", RichLog)
        log_widget.write(Text.from_markup(message))

    def log_finding(self, agent_role: str, finding: str, confidence: float) -> None:
        """Log a finding with agent attribution."""
        color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
        self.write_log(
            f"[{color}]■[/] [{color} bold]{agent_role}[/]: {finding[:120]} "
            f"[dim](conf: {confidence:.2f})[/]"
        )
