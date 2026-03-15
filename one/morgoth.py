"""Morgoth Mode — The God Builder.

System 10: Research + Build + Verify + Iterate until done.

Morgoth = swarm + dialectic + analogy + contradiction mining + verification
+ experimentation + question generation ALL RUNNING SIMULTANEOUSLY.

The swarm doesn't just research — it IMPLEMENTS. It opens the codebase,
writes code, writes tests, runs them, fixes failures, commits, and moves
to the next discovery.

The 7 Phases:
1. UNDERSTAND — Map the problem space
2. RESEARCH — Deep dive every sub-problem
3. SYNTHESIZE — Connect findings, extract patterns
4. BUILD — Write actual production code + tests
5. VERIFY — Full adversarial review
6. ITERATE — Feed learnings back, repeat
7. THERE IS NO PHASE 7 — Morgoth does not stop

Agent waves:
- Wave 1: Conductor, Surveyor, Historian, Devil's Advocate
- Wave 2: Mechanist, Contrarian, Analogist, Verifier
- Wave 3: Synthesizer, Experimentalist, Builder, Futurist
- Dynamic: Specialist, Debugger, Integrator (as needed)
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable

from .swarm import SwarmCoordinator, AgentRole
from .dialectic import DialecticEngine
from .analogy import AnalogyEngine
from .contradictions import ContradictionMiner
from .verification import VerificationEngine, FrontierMapper
from .experiments import ExperimentEngine
from .health import HealthDashboard
from .store import push_memory


# ── Phases ─────────────────────────────────────────────────────

class Phase:
    UNDERSTAND = 1
    RESEARCH = 2
    SYNTHESIZE = 3
    BUILD = 4
    VERIFY = 5
    ITERATE = 6
    CONTINUOUS = 7  # "There is no Phase 7" — but we model it

PHASE_NAMES = {
    Phase.UNDERSTAND: "UNDERSTAND",
    Phase.RESEARCH: "RESEARCH",
    Phase.SYNTHESIZE: "SYNTHESIZE",
    Phase.BUILD: "BUILD",
    Phase.VERIFY: "VERIFY",
    Phase.ITERATE: "ITERATE",
    Phase.CONTINUOUS: "CONTINUOUS",
}

# ── Agent Waves ────────────────────────────────────────────────

WAVE_1 = [
    AgentRole.CONDUCTOR,
    AgentRole.SURVEYOR,
    AgentRole.HISTORIAN,
    AgentRole.DEVILS_ADVOCATE,
]

WAVE_2 = [
    AgentRole.MECHANIST,
    AgentRole.CONTRARIAN,
    AgentRole.ANALOGIST,
    AgentRole.VERIFIER,
]

WAVE_3 = [
    AgentRole.SYNTHESIZER,
    AgentRole.EXPERIMENTALIST,
    AgentRole.FUTURIST,
    AgentRole.INTEGRATOR,
]

MAX_AGENTS = 15
STATE_DIR = os.path.expanduser("~/.one")
STATE_FILE = os.path.join(STATE_DIR, "morgoth_state.json")
PERSIST_INTERVAL = 60  # seconds


# ── Eureka ─────────────────────────────────────────────────────

@dataclass
class Eureka:
    """A breakthrough finding with excitation > 0.8."""
    text: str
    agent_role: str
    confidence: float
    timestamp: str
    survived_dialectic: bool = False


# ── Morgoth Mode ───────────────────────────────────────────────

class MorgothMode:
    """The God Builder. Research + Build + Verify + Iterate until done."""

    def __init__(
        self,
        goal: str,
        project: str,
        proxy_factory: Optional[Callable] = None,
        on_log: Optional[Callable] = None,
    ):
        self.goal = goal
        self.project = project
        self.proxy_factory = proxy_factory
        self._log = on_log or (lambda _m: None)

        self.phase = Phase.UNDERSTAND
        self.iteration = 0
        self.eurekas: list[Eureka] = []
        self.active = False

        # Sub-engines
        self.swarm: Optional[SwarmCoordinator] = None
        self.dialectic = DialecticEngine(project, on_log=on_log)
        self.analogy = AnalogyEngine(project, on_log=on_log)
        self.contradictions = ContradictionMiner(project, on_log=on_log)
        self.verification = VerificationEngine(project, on_log=on_log)
        self.frontier = FrontierMapper(project, on_log=on_log)
        self.experiments = ExperimentEngine(project, on_log=on_log)
        self.health = HealthDashboard(project, on_log=on_log)

        # Threading
        self._lock = threading.Lock()
        self._persist_thread: Optional[threading.Thread] = None
        self._phase_thread: Optional[threading.Thread] = None

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the Morgoth loop from Phase 1."""
        self._log(f"MORGOTH ACTIVATED: {self.goal[:80]}")
        self.active = True
        self.phase = Phase.UNDERSTAND
        self.iteration = 1

        # Start state persistence
        self._persist_thread = threading.Thread(
            target=self._persist_loop, daemon=True,
        )
        self._persist_thread.start()

        # Start main phase loop
        self._phase_thread = threading.Thread(
            target=self._run_phases, daemon=True,
        )
        self._phase_thread.start()

    def stop(self) -> None:
        """Stop the Morgoth loop."""
        self._log("MORGOTH STOPPING")
        self.active = False
        if self.swarm:
            self.swarm.stop()
        self._persist_state()

    def resume(self) -> None:
        """Resume from persisted state."""
        state = self.load_state()
        if not state:
            self._log("no saved state found, starting fresh")
            self.start()
            return

        self.goal = state.get("goal", self.goal)
        self.project = state.get("project", self.project)
        self.phase = state.get("phase", Phase.UNDERSTAND)
        self.iteration = state.get("iteration", 1)
        self.eurekas = [
            Eureka(**e) for e in state.get("eurekas", [])
        ]

        self._log(f"MORGOTH RESUMED: phase={PHASE_NAMES.get(self.phase, '?')}, iteration={self.iteration}")
        self.active = True

        self._persist_thread = threading.Thread(
            target=self._persist_loop, daemon=True,
        )
        self._persist_thread.start()

        self._phase_thread = threading.Thread(
            target=self._run_phases, daemon=True,
        )
        self._phase_thread.start()

    # ── Phase Loop ─────────────────────────────────────────────

    def _run_phases(self) -> None:
        """Main loop: cycle through phases until stopped."""
        while self.active:
            phase_name = PHASE_NAMES.get(self.phase, "UNKNOWN")
            self._log(f"═══ PHASE {self.phase}: {phase_name} (iteration {self.iteration}) ═══")

            try:
                if self.phase == Phase.UNDERSTAND:
                    self._phase_understand()
                elif self.phase == Phase.RESEARCH:
                    self._phase_research()
                elif self.phase == Phase.SYNTHESIZE:
                    self._phase_synthesize()
                elif self.phase == Phase.BUILD:
                    self._phase_build()
                elif self.phase == Phase.VERIFY:
                    self._phase_verify()
                elif self.phase == Phase.ITERATE:
                    self._phase_iterate()
                elif self.phase == Phase.CONTINUOUS:
                    self._phase_continuous()
            except Exception as exc:
                self._log(f"phase error: {exc}")
                time.sleep(10)
                continue

            if not self.active:
                break

            # Advance phase
            self._advance_phase()

    def _advance_phase(self) -> None:
        """Move to the next phase."""
        if self.phase < Phase.ITERATE:
            self.phase += 1
        elif self.phase == Phase.ITERATE:
            # Loop back to RESEARCH for next iteration
            self.phase = Phase.RESEARCH
            self.iteration += 1
            self._log(f"starting iteration {self.iteration}")
        else:
            # Continuous mode: generate next goal
            self.phase = Phase.UNDERSTAND
            self.iteration += 1

    # ── Phase Implementations ──────────────────────────────────

    def _phase_understand(self) -> None:
        """Phase 1: Map the problem space."""
        self._log("mapping problem space...")

        # Launch Wave 1: Conductor, Surveyor, Historian, Devil's Advocate
        self._launch_wave(WAVE_1)

        # Map the knowledge frontier
        self.frontier.map_frontier(self.goal)

        # Wait for wave to produce initial findings
        self._wait_for_findings(min_findings=3, timeout=300)

    def _phase_research(self) -> None:
        """Phase 2: Deep dive every sub-problem."""
        self._log("deep research phase...")

        # Launch Wave 2: Mechanist, Contrarian, Analogist, Verifier
        self._launch_wave(WAVE_2)

        # Mine for contradictions
        contradictions = self.contradictions.mine_contradictions()
        for c in contradictions[:5]:
            self.contradictions.score_contradiction(
                c["finding_a"], c["finding_b"],
            )

        # Run dialectics on major findings
        findings = self._get_high_value_findings(5)
        for f in findings:
            self.dialectic.run_full_dialectic(f)

        self._wait_for_findings(min_findings=10, timeout=600)

    def _phase_synthesize(self) -> None:
        """Phase 3: Connect findings, extract patterns."""
        self._log("synthesis phase...")

        # Extract analogy templates from findings
        findings = self._get_high_value_findings(20)
        for f in findings[:10]:
            template = self.analogy.extract_template(f)
            self.analogy.match_templates(template)

        # Find universal patterns
        self.analogy.find_universal_patterns(min_domains=2)

        # Resolve active contradictions
        active = self.contradictions.get_active()
        for c in active[:3]:
            self.contradictions.resolve_contradiction(
                c["finding_a"], c["finding_b"], c["id"],
            )

        # Generate best questions
        best_q = self.frontier.best_question(self.goal)
        if best_q:
            self._log(f"best question: {best_q.get('question', '')[:80]}")

    def _phase_build(self) -> None:
        """Phase 4: Write actual production code + tests."""
        self._log("build phase...")

        # Launch Wave 3: Synthesizer, Experimentalist, Futurist, Integrator
        self._launch_wave(WAVE_3)

        # Run experiments on testable hypotheses
        findings = self._get_high_value_findings(10)
        for f in findings[:3]:
            testability = self.experiments.is_testable(f)
            if testability.get("testable"):
                self.experiments.run_full_experiment(f)

        self._wait_for_findings(min_findings=5, timeout=600)

    def _phase_verify(self) -> None:
        """Phase 5: Full adversarial review."""
        self._log("verification phase...")

        # Run verification sweep
        self.verification.run_verification_sweep(n=20)

        # Check health
        report = self.health.full_report()
        warnings = report.get("warnings", [])
        if warnings:
            for w in warnings:
                self._log(f"health warning: {w['message']}")

        # Archive deprecated findings
        self.verification.archive_deprecated()

    def _phase_iterate(self) -> None:
        """Phase 6: Feed learnings back, repeat."""
        self._log("iteration phase...")

        # What did we learn?
        confidence_dist = self.verification.get_confidence_distribution()
        self._log(f"confidence distribution: {confidence_dist}")

        # Update frontier with new findings
        self.frontier.update_frontier(self.goal, [])

        # Check coverage
        coverage = self.frontier.frontier_coverage(self.goal)
        self._log(f"frontier coverage: {coverage:.1%}")

        # If coverage > 80%, consider moving to continuous
        if coverage > 0.8:
            self._log("high coverage — moving to continuous mode")
            self.phase = Phase.CONTINUOUS - 1  # _advance_phase will increment

    def _phase_continuous(self) -> None:
        """Phase 7: Continuous improvement — Morgoth does not stop."""
        self._log("continuous mode — looking for improvements...")

        # Generate next questions
        best = self.frontier.best_question(self.goal)
        if best:
            question = best.get("question", "")
            self._log(f"next question: {question[:80]}")

            # Push as new research goal
            push_memory(
                f"[MORGOTH CONTINUOUS] Next question: {question}",
                source="morgoth",
                tm_label="morgoth_continuous",
                project=self.project,
            )

        # Wait before looping
        for _ in range(30):
            if not self.active:
                return
            time.sleep(1)

    # ── Wave Management ────────────────────────────────────────

    def _launch_wave(self, roles: list[AgentRole]) -> None:
        """Launch a wave of agents if swarm is active."""
        if not self.swarm:
            self._log("swarm not initialized, creating...")
            self.swarm = SwarmCoordinator(
                goal=self.goal,
                project=self.project,
            )

        swarm = self.swarm
        assert swarm is not None
        for role in roles:
            if not self.active:
                return
            try:
                swarm.spawn_agent(role)
            except Exception as exc:
                self._log(f"failed to spawn {role.value}: {exc}")

    def _wait_for_findings(self, min_findings: int, timeout: int) -> None:
        """Wait until we have enough findings or timeout."""
        start = time.monotonic()
        while self.active and (time.monotonic() - start) < timeout:
            if self.swarm:
                findings = self.swarm.get_findings()
                if len(findings) >= min_findings:
                    return
            time.sleep(5)

    def _get_high_value_findings(self, n: int) -> list[str]:
        """Get high-value finding texts from the swarm."""
        if not self.swarm:
            return []
        findings = list(self.swarm.get_findings())
        # Sort by confidence/importance
        findings.sort(key=lambda f: f.confidence, reverse=True)
        return [f.content for f in findings[:n] if f.content]

    # ── Eureka Handling ────────────────────────────────────────

    def on_eureka(self, finding_text: str, agent_role: str, confidence: float) -> None:
        """Handle a breakthrough finding (excitation > 0.8).

        1. Broadcast to all agents
        2. Store at maximum confidence
        3. Trigger dialectic challenge
        4. Flash TUI alert
        """
        self._log(f"EUREKA from {agent_role}: {finding_text[:80]}")

        eureka = Eureka(
            text=finding_text,
            agent_role=agent_role,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self.eurekas.append(eureka)

        # Store at max confidence
        push_memory(
            f"[EUREKA] {finding_text}",
            source="morgoth",
            tm_label="eureka",
            project=self.project,
            aif_confidence=max(confidence, 0.9),
        )

        # Broadcast to swarm
        if self.swarm:
            self.swarm.inject(f"EUREKA BROADCAST: {finding_text}")

        # Challenge via dialectic
        try:
            result = self.dialectic.run_full_dialectic(finding_text, source=agent_role)
            if result.get("verdict") in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                eureka.survived_dialectic = True
                self._log(f"EUREKA SURVIVED DIALECTIC: {result.get('verdict')}")
            else:
                self._log(f"EUREKA CHALLENGED: {result.get('verdict')}")
        except Exception as exc:
            self._log(f"dialectic challenge failed: {exc}")

    # ── State Persistence ──────────────────────────────────────

    def _persist_loop(self) -> None:
        """Serialize state to disk every PERSIST_INTERVAL seconds."""
        while self.active:
            time.sleep(PERSIST_INTERVAL)
            self._persist_state()

    def _persist_state(self) -> None:
        """Write current state to disk."""
        os.makedirs(STATE_DIR, exist_ok=True)

        state = {
            "goal": self.goal,
            "project": self.project,
            "phase": self.phase,
            "iteration": self.iteration,
            "active": self.active,
            "eurekas": [
                {
                    "text": e.text,
                    "agent_role": e.agent_role,
                    "confidence": e.confidence,
                    "timestamp": e.timestamp,
                    "survived_dialectic": e.survived_dialectic,
                }
                for e in self.eurekas
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except OSError as exc:
            self._log(f"state persist failed: {exc}")

    @staticmethod
    def load_state() -> Optional[dict]:
        """Load persisted state from disk."""
        if not os.path.exists(STATE_FILE):
            return None
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def clear_state() -> None:
        """Remove persisted state file."""
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)

    # ── Status ─────────────────────────────────────────────────

    def status(self) -> dict:
        """Current Morgoth status."""
        swarm_status = self.swarm.status() if self.swarm else {}
        return {
            "active": self.active,
            "goal": self.goal,
            "project": self.project,
            "phase": self.phase,
            "phase_name": PHASE_NAMES.get(self.phase, "UNKNOWN"),
            "iteration": self.iteration,
            "eurekas": len(self.eurekas),
            "swarm": swarm_status,
        }
