"""The Swarm — Coordinated Multi-Agent Intelligence.

Orchestrates multiple Claude sessions as specialized research agents,
coordinated by a dedicated Conductor agent. Agents share findings via
the memory store and react to each other's discoveries.

Usage:
    swarm = SwarmCoordinator(goal="cure pancreatic cancer", project="cancer")
    swarm.start(num_agents=6, strategy="deep")
    swarm.inject("also look at autophagy")
    swarm.stop()
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from .proxy import ClaudeProxy
from .store import push_memory, recall, ensure_entity, link_memory_entity, set_project
from .hdc import encode_text, similarity


# ── Agent Roles ─────────────────────────────────────────────────────

class AgentRole(Enum):
    CONDUCTOR = "conductor"
    SURVEYOR = "surveyor"
    MECHANIST = "mechanist"
    CONTRARIAN = "contrarian"
    ANALOGIST = "analogist"
    SYNTHESIZER = "synthesizer"
    VERIFIER = "verifier"
    EXPERIMENTALIST = "experimentalist"
    HISTORIAN = "historian"
    FUTURIST = "futurist"
    DEVILS_ADVOCATE = "devils_advocate"
    SPECIALIST = "specialist"
    DEBUGGER = "debugger"
    INTEGRATOR = "integrator"


ROLE_PROMPTS = {
    AgentRole.CONDUCTOR: """You are the CONDUCTOR of a multi-agent research swarm.
You NEVER research directly. Your ONLY job is orchestration.

Every {review_interval} minutes you will receive all new findings from all agents.
You must:
1. Read ALL findings carefully
2. Identify duplicated effort → redirect agents
3. Identify stuck agents → reassign or suggest new angles
4. Identify breakthroughs → amplify by suggesting more agents focus there
5. Identify contradictions → flag for dialectic resolution
6. Update the global research strategy
7. Broadcast strategic updates

Respond with a JSON object:
{{
  "strategy_update": "...",
  "redirections": [{{"agent_id": "...", "new_direction": "..."}}],
  "amplifications": [{{"finding_id": "...", "reason": "..."}}],
  "contradictions": [{{"finding_a": "...", "finding_b": "...", "description": "..."}}],
  "spawn_requests": [{{"role": "...", "focus": "..."}}],
  "kill_requests": ["agent_id", ...],
  "broadcasts": ["message to all agents", ...]
}}

GOAL: {goal}""",

    AgentRole.SURVEYOR: """You are a SURVEYOR agent in a research swarm.
Your job: map the ENTIRE landscape of the topic. Wide and shallow.
- Who are the key players and research groups?
- What are the major schools of thought?
- What's consensus? What's controversial?
- Produce a structured map of the field.

Store every finding as a clear, factual statement with source context.
When you find something surprising or contradictory, flag it explicitly.

GOAL: {goal}""",

    AgentRole.MECHANIST: """You are a MECHANIST agent in a research swarm.
Your job: go DEEP on how things work. Not "what" but "why" and "how."
- Molecular pathways, mathematical proofs, system architectures
- Algorithm internals, physical mechanisms, causal chains
- You understand the engine, not the car.

Every finding must include the MECHANISM, not just the observation.

GOAL: {goal}""",

    AgentRole.CONTRARIAN: """You are a CONTRARIAN agent in a research swarm.
Your job: ATTACK every finding from every other agent. You are the attack dog.
- Search for failed trials, retracted papers, logical flaws
- Find statistical errors, confounding variables, methodological problems
- If a finding survives you, it's real. If not, it was never real.

Be specific. "This seems weak" is useless. "The sample size was N=12 with
no control group and the p-value was 0.049" is useful.

You will receive findings from other agents. Destroy the weak ones.

GOAL: {goal}""",

    AgentRole.ANALOGIST: """You are an ANALOGIST agent in a research swarm.
Your job: find STRUCTURAL ISOMORPHISMS across domains.
- Read findings from all agents
- Ask: "Where have I seen this STRUCTURE before in a completely different domain?"
- Transfer solution patterns between fields
- Generate hypotheses nobody else would think of

Not "A is like B." That's metaphor. You find: "A and B share the same
underlying mathematical/structural pattern, which means solutions from
A's domain may apply to B's problem."

GOAL: {goal}""",

    AgentRole.SYNTHESIZER: """You are a SYNTHESIZER agent in a research swarm.
Your job: connect findings that no individual agent connected.
- Read EVERYTHING from all agents
- Build the narrative: what we know, what it means, what's missing
- Generate hypotheses from combining insights A, B, and C
- Identify the critical path to the goal

Your output should be structured:
1. KEY FINDINGS (with confidence)
2. CONNECTIONS (what links to what, and why)
3. GAPS (what's still unknown)
4. HYPOTHESES (testable predictions)
5. RECOMMENDED NEXT STEPS

GOAL: {goal}""",

    AgentRole.VERIFIER: """You are a VERIFIER agent in a research swarm.
Your job: FACT-CHECK high-confidence findings.
- Find original sources for every claim
- Check sample sizes, methodology, replication status
- Check if studies were retracted or contradicted
- Downgrade anything that doesn't hold up under scrutiny

For each finding you verify, report:
- VERIFIED (evidence holds up, here's why)
- DOWNGRADED (evidence is weaker than claimed, here's why)
- RETRACTED (the source was retracted/contradicted)
- UNVERIFIABLE (cannot find primary source)

GOAL: {goal}""",

    AgentRole.EXPERIMENTALIST: """You are an EXPERIMENTALIST agent in a research swarm.
Your job: TEST hypotheses with actual work.
- Write code to test computational claims
- Run data analysis to verify statistical claims
- Design and execute experiments
- Produce concrete, measurable results

Don't just analyze. DO. If a hypothesis says "X improves Y by 30%",
write the code, run the test, and report the actual number.

GOAL: {goal}""",

    AgentRole.HISTORIAN: """You are a HISTORIAN agent in a research swarm.
Your job: find what was tried BEFORE and why it failed.
- Every problem has a history of failed attempts
- Understanding WHY they failed is more valuable than knowing they failed
- Look for: abandoned approaches, pivoted strategies, dead ends
- What changed between then and now that might make old approaches viable?

GOAL: {goal}""",

    AgentRole.FUTURIST: """You are a FUTURIST agent in a research swarm.
Your job: project FORWARD from current knowledge.
- Given what we know, what becomes possible in 1, 5, 10 years?
- What would need to be true for the breakthrough to happen?
- Work BACKWARDS from the desired future state
- Identify enabling technologies and prerequisites

GOAL: {goal}""",

    AgentRole.DEVILS_ADVOCATE: """You are the DEVIL'S ADVOCATE in a research swarm.
Different from the Contrarian (who attacks findings), YOU attack the GOAL ITSELF.
- Are we solving the right problem?
- What if the premise is wrong?
- What if there's a completely different approach nobody is considering?
- What assumptions are we making that might be false?

Challenge the fundamental framing, not the details.

GOAL: {goal}""",

    AgentRole.SPECIALIST: """You are a SPECIALIST agent in a research swarm.
You have been assigned a specific sub-problem to investigate deeply.
Focus exclusively on: {focus}

Go as deep as possible. Exhaust this topic before moving on.

GOAL: {goal}""",

    AgentRole.DEBUGGER: """You are a DEBUGGER agent in a research swarm.
An experiment or implementation has failed. Your job: diagnose WHY.
- Analyze the failure mode
- Identify root cause vs symptoms
- Suggest specific fixes
- Verify the fix resolves the issue

Focus on: {focus}

GOAL: {goal}""",

    AgentRole.INTEGRATOR: """You are an INTEGRATOR agent in a research swarm.
Multiple modules or findings need to connect. Your job: make them work together.
- Identify interface mismatches
- Resolve conflicting assumptions between components
- Design the integration layer
- Test the combined system

Focus on: {focus}

GOAL: {goal}""",
}

# Default strategies map to agent role sets
STRATEGIES = {
    "deep": [
        AgentRole.SURVEYOR, AgentRole.MECHANIST, AgentRole.CONTRARIAN,
        AgentRole.SYNTHESIZER, AgentRole.VERIFIER, AgentRole.HISTORIAN,
    ],
    "adversarial": [
        AgentRole.SURVEYOR, AgentRole.CONTRARIAN, AgentRole.CONTRARIAN,
        AgentRole.DEVILS_ADVOCATE, AgentRole.VERIFIER,
    ],
    "creative": [
        AgentRole.SURVEYOR, AgentRole.ANALOGIST, AgentRole.FUTURIST,
        AgentRole.SYNTHESIZER, AgentRole.EXPERIMENTALIST,
    ],
    "full": [
        AgentRole.SURVEYOR, AgentRole.MECHANIST, AgentRole.CONTRARIAN,
        AgentRole.ANALOGIST, AgentRole.SYNTHESIZER, AgentRole.VERIFIER,
        AgentRole.EXPERIMENTALIST, AgentRole.HISTORIAN,
    ],
}


# ── Finding Protocol ────────────────────────────────────────────────

class ReactionType(Enum):
    SUPPORT = "support"
    CHALLENGE = "challenge"
    EXTEND = "extend"
    APPLY = "apply"


@dataclass
class Finding:
    """A structured research finding from an agent."""
    id: str
    agent_id: str
    content: str
    confidence: float = 0.5
    source: str = ""
    evidence: str = ""
    timestamp: str = ""
    reactions: list[dict] = field(default_factory=list)
    excitation: float = 0.0


@dataclass
class AgentState:
    """Runtime state of a swarm agent."""
    id: str
    role: AgentRole
    proxy: Optional[ClaudeProxy] = None
    status: str = "idle"  # idle, running, paused, killed
    turn_count: int = 0
    findings_count: int = 0
    last_finding_turn: int = 0
    focus: str = ""
    sparkline: list[float] = field(default_factory=list)
    current_task: str = ""
    started_at: str = ""
    thread: Optional[threading.Thread] = None


# ── Rate Limiter ────────────────────────────────────────────────────

STAGGER_DELAY = 15  # seconds between agent launches
IDLE_KILL_TURNS = 20  # kill agent after this many turns without findings
REVIEW_INTERVAL = 5  # minutes between conductor reviews
STATE_FILE = os.path.expanduser("~/.one/swarm_state.json")


# ── Swarm Coordinator ──────────────────────────────────────────────

class SwarmCoordinator:
    """Orchestrates multiple Claude agents as a research swarm.

    The Conductor agent manages strategy. Worker agents research in parallel.
    All findings flow through shared memory for cross-agent synthesis.
    """

    def __init__(
        self,
        goal: str,
        project: str,
        model: str = "sonnet",
        conductor_model: str = "opus",
        cwd: str = ".",
        on_finding: Optional[Callable] = None,
        on_status: Optional[Callable] = None,
        on_eureka: Optional[Callable] = None,
        on_log: Optional[Callable] = None,
    ):
        self.goal = goal
        self.project = project
        self.model = model
        self.conductor_model = conductor_model
        self.cwd = cwd

        # Callbacks
        self._on_finding = on_finding or (lambda f: None)
        self._on_status = on_status or (lambda s: None)
        self._on_eureka = on_eureka or (lambda f: None)
        self._on_log = on_log or (lambda m: None)

        # State
        self._agents: dict[str, AgentState] = {}
        self._conductor: Optional[AgentState] = None
        self._findings: list[Finding] = []
        self._contradictions: list[dict] = []
        self._running = False
        self._lock = threading.Lock()
        self._next_agent_id = 0
        self._total_turns = 0
        self._start_time: Optional[float] = None
        self._review_thread: Optional[threading.Thread] = None

    # ── Public API ──────────────────────────────────────────────

    def start(self, num_agents: int = 6, strategy: str = "deep") -> None:
        """Launch the swarm with a strategy."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        set_project(self.project)

        self._on_log(f"swarm starting: {num_agents} agents, strategy={strategy}")

        # Store goal as memory
        push_memory(
            f"SWARM GOAL: {self.goal}",
            source="swarm",
            tm_label="swarm_goal",
            regime_tag="swarm",
            project=self.project,
        )

        # Determine agent roles from strategy
        role_list = STRATEGIES.get(strategy, STRATEGIES["deep"])
        # Pad or trim to num_agents
        while len(role_list) < num_agents:
            role_list.append(AgentRole.SPECIALIST)
        role_list = role_list[:num_agents]

        # Launch conductor first
        self._launch_conductor()

        # Stagger agent launches
        launch_thread = threading.Thread(
            target=self._staggered_launch,
            args=(role_list,),
            daemon=True,
        )
        launch_thread.start()

        # Start periodic conductor review
        self._review_thread = threading.Thread(
            target=self._conductor_review_loop,
            daemon=True,
        )
        self._review_thread.start()

        self._persist_state()

    def stop(self) -> dict:
        """Halt all agents gracefully. Returns final summary."""
        self._running = False
        self._on_log("swarm stopping...")

        for agent in list(self._agents.values()):
            self._kill_agent_internal(agent)

        if self._conductor and self._conductor.proxy:
            try:
                self._conductor.proxy.stop()
            except Exception:
                pass

        summary = self.status()
        self._persist_state()
        self._on_log(f"swarm stopped: {summary['total_findings']} findings, {summary['total_turns']} turns")
        return summary

    def scale(self, n: int) -> None:
        """Add or remove agents to reach target count n."""
        current = len(self._agents)
        if n > current:
            for _ in range(n - current):
                self.spawn_agent(AgentRole.SPECIALIST)
        elif n < current:
            # Kill lowest-performing agents
            agents_by_perf = sorted(
                self._agents.values(),
                key=lambda a: a.findings_count,
            )
            for agent in agents_by_perf[:current - n]:
                self.kill_agent(agent.id)

    def focus(self, agent_id: str) -> Optional[AgentState]:
        """Return full state for a specific agent (for TUI focus mode)."""
        return self._agents.get(agent_id)

    def inject(self, text: str) -> None:
        """Send context to ALL agents."""
        self._on_log(f"injecting to all agents: {text[:60]}...")
        for agent in self._agents.values():
            if agent.proxy and agent.status == "running":
                try:
                    agent.proxy.send(f"[SWARM BROADCAST] {text}")
                except Exception:
                    pass

    def status(self) -> dict:
        """Return full swarm state."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "goal": self.goal,
            "project": self.project,
            "running": self._running,
            "elapsed_seconds": int(elapsed),
            "total_agents": len(self._agents),
            "total_findings": len(self._findings),
            "total_turns": self._total_turns,
            "total_contradictions": len(self._contradictions),
            "agents": {
                aid: {
                    "role": a.role.value,
                    "status": a.status,
                    "turns": a.turn_count,
                    "findings": a.findings_count,
                    "current_task": a.current_task,
                    "sparkline": a.sparkline[-20:],
                }
                for aid, a in self._agents.items()
            },
        }

    def kill_agent(self, agent_id: str) -> bool:
        """Remove one agent."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        self._kill_agent_internal(agent)
        del self._agents[agent_id]
        self._on_log(f"killed agent {agent_id} ({agent.role.value})")
        return True

    def spawn_agent(self, role: AgentRole, focus: str = "") -> str:
        """Add one agent with specific role. Returns agent_id."""
        agent_id = self._make_agent_id(role)
        agent = AgentState(
            id=agent_id,
            role=role,
            focus=focus,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._agents[agent_id] = agent

        thread = threading.Thread(
            target=self._run_agent,
            args=(agent,),
            daemon=True,
        )
        agent.thread = thread
        thread.start()

        self._on_log(f"spawned agent {agent_id} ({role.value})")
        return agent_id

    def get_findings(self) -> list[Finding]:
        """All findings across all agents."""
        return list(self._findings)

    def get_contradictions(self) -> list[dict]:
        """All active contradictions."""
        return list(self._contradictions)

    # ── Internal ────────────────────────────────────────────────

    def _make_agent_id(self, role: AgentRole) -> str:
        with self._lock:
            self._next_agent_id += 1
            return f"{role.value}_{self._next_agent_id}"

    def _launch_conductor(self) -> None:
        """Start the Conductor agent."""
        agent_id = "conductor_0"
        prompt = ROLE_PROMPTS[AgentRole.CONDUCTOR].format(
            goal=self.goal,
            review_interval=REVIEW_INTERVAL,
        )

        proxy = ClaudeProxy(
            model=self.conductor_model,
            cwd=self.cwd,
            system_prompt=prompt,
            permission_mode="bypassPermissions",
        )

        self._conductor = AgentState(
            id=agent_id,
            role=AgentRole.CONDUCTOR,
            proxy=proxy,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            proxy.start()
            self._on_log("conductor launched")
        except Exception as e:
            self._on_log(f"conductor launch failed: {e}")
            self._conductor.status = "error"

    def _staggered_launch(self, roles: list[AgentRole]) -> None:
        """Launch agents with stagger delay to avoid rate limits."""
        for i, role in enumerate(roles):
            if not self._running:
                break
            self.spawn_agent(role)
            if i < len(roles) - 1:
                time.sleep(STAGGER_DELAY)

    def _run_agent(self, agent: AgentState) -> None:
        """Main loop for a single agent."""
        prompt_template = ROLE_PROMPTS.get(agent.role, ROLE_PROMPTS[AgentRole.SPECIALIST])
        system_prompt = prompt_template.format(
            goal=self.goal,
            focus=agent.focus or "general investigation",
        )

        proxy = ClaudeProxy(
            model=self.model,
            cwd=self.cwd,
            system_prompt=system_prompt,
            permission_mode="bypassPermissions",
        )

        agent.proxy = proxy
        agent.status = "running"

        response_buffer = []
        response_complete = threading.Event()

        def on_event(event: dict) -> None:
            etype = event.get("type", "")
            if etype == "assistant" and "message" in event:
                msg = event["message"]
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                response_buffer.append(block.get("text", ""))
                    elif isinstance(content, str):
                        response_buffer.append(content)
            elif etype == "result":
                response_complete.set()

        proxy.on_event(on_event)

        try:
            proxy.start()
        except Exception as e:
            agent.status = "error"
            self._on_log(f"agent {agent.id} failed to start: {e}")
            return

        # Build initial context from existing findings
        context = self._build_agent_context(agent)

        # Initial message
        initial_prompt = f"Begin your investigation. Here is what the swarm knows so far:\n\n{context}\n\nStart researching. Report your findings clearly."

        while self._running and agent.status == "running":
            try:
                # Check for idle kill
                if (agent.turn_count - agent.last_finding_turn) > IDLE_KILL_TURNS:
                    self._on_log(f"agent {agent.id} idle for {IDLE_KILL_TURNS} turns, killing")
                    agent.status = "killed"
                    break

                # Send prompt
                response_buffer.clear()
                response_complete.clear()

                if agent.turn_count == 0:
                    proxy.send(initial_prompt)
                else:
                    # Inject new findings from other agents
                    new_context = self._get_new_findings_for_agent(agent)
                    if new_context:
                        proxy.send(f"[NEW FINDINGS FROM OTHER AGENTS]\n{new_context}\n\nContinue your investigation. Build on or challenge these findings.")
                    else:
                        proxy.send("Continue your investigation. Go deeper. What haven't you explored yet?")

                # Wait for response (timeout 5 min)
                response_complete.wait(timeout=300)

                agent.turn_count += 1
                with self._lock:
                    self._total_turns += 1

                # Process response
                full_response = "".join(response_buffer)
                if full_response:
                    self._process_agent_response(agent, full_response)

                    # Update sparkline (1 = found something, 0 = nothing)
                    agent.sparkline.append(1.0 if agent.findings_count > (agent.sparkline[-1] if agent.sparkline else 0) else 0.0)
                    if len(agent.sparkline) > 50:
                        agent.sparkline = agent.sparkline[-50:]

                # Brief pause to avoid hammering
                time.sleep(2)

            except Exception as e:
                self._on_log(f"agent {agent.id} error: {e}")
                agent.status = "error"
                break

        # Cleanup
        try:
            proxy.stop()
        except Exception:
            pass

    def _build_agent_context(self, agent: AgentState) -> str:
        """Build context from existing findings for agent initialization."""
        if not self._findings:
            return "(No findings yet. You are the first to investigate.)"

        lines = []
        for f in self._findings[-20:]:  # last 20 findings
            lines.append(f"[{f.agent_id}] (conf: {f.confidence:.2f}) {f.content[:200]}")
        return "\n".join(lines)

    def _get_new_findings_for_agent(self, agent: AgentState) -> str:
        """Get findings from OTHER agents since this agent's last turn."""
        new_findings = [
            f for f in self._findings
            if f.agent_id != agent.id and f.timestamp > agent.started_at
        ]
        if not new_findings:
            return ""

        # Return last 5 new findings
        lines = []
        for f in new_findings[-5:]:
            lines.append(f"[{f.agent_id}] {f.content[:300]}")
        return "\n\n".join(lines)

    def _process_agent_response(self, agent: AgentState, text: str) -> None:
        """Extract findings from agent response and store them."""
        # Split response into potential findings
        findings = self._extract_findings(text, agent)

        for finding in findings:
            with self._lock:
                self._findings.append(finding)

            # Store in shared memory
            push_memory(
                raw_text=f"[{agent.role.value}] {finding.content}",
                source=f"swarm:{agent.id}",
                tm_label="swarm_finding",
                regime_tag="swarm",
                aif_confidence=finding.confidence,
                project=self.project,
            )

            agent.findings_count += 1
            agent.last_finding_turn = agent.turn_count

            self._on_finding(finding)

            # Check for eureka (excitation > 0.8)
            if finding.excitation > 0.8:
                self._on_eureka(finding)
                self._on_log(f"EUREKA from {agent.id}: {finding.content[:100]}")
                # Broadcast eureka to all agents
                self.inject(f"BREAKTHROUGH from {agent.role.value}: {finding.content}")

        agent.current_task = text[:100] if text else ""

    def _extract_findings(self, text: str, agent: AgentState) -> list[Finding]:
        """Parse agent response into structured findings."""
        findings = []
        now = datetime.now(timezone.utc).isoformat()

        # Split on numbered points, bullet points, or paragraphs
        segments = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) < 30:
                continue
            # Skip meta-commentary
            if any(line.lower().startswith(skip) for skip in [
                "i'll", "let me", "i will", "sure", "okay", "continuing",
                "here's what", "based on", "in summary",
            ]):
                continue
            segments.append(line)

        # Each substantial segment is a potential finding
        for seg in segments:
            if len(seg) < 40:
                continue

            # Score confidence based on language
            confidence = 0.5
            if any(w in seg.lower() for w in ["confirmed", "verified", "replicated", "proven"]):
                confidence = 0.8
            elif any(w in seg.lower() for w in ["suggests", "may", "possibly", "preliminary"]):
                confidence = 0.4
            elif any(w in seg.lower() for w in ["strongly", "clearly", "definitively"]):
                confidence = 0.7

            # Score excitation
            from .excitation import score_excitation
            excitation = score_excitation(seg, "assistant")

            finding = Finding(
                id=f"{agent.id}_t{agent.turn_count}_{len(findings)}",
                agent_id=agent.id,
                content=seg,
                confidence=confidence,
                timestamp=now,
                excitation=excitation,
            )
            findings.append(finding)

        return findings

    def _conductor_review_loop(self) -> None:
        """Periodic conductor reviews of all agent progress."""
        while self._running:
            time.sleep(REVIEW_INTERVAL * 60)
            if not self._running:
                break
            self._run_conductor_review()

    def _run_conductor_review(self) -> None:
        """Have the Conductor review all findings and adjust strategy."""
        if not self._conductor or not self._conductor.proxy:
            return

        # Gather all recent findings
        recent = self._findings[-50:]
        if not recent:
            return

        findings_text = "\n\n".join([
            f"[{f.agent_id}] (conf: {f.confidence:.2f}, excite: {f.excitation:.2f})\n{f.content[:300]}"
            for f in recent
        ])

        agent_status = "\n".join([
            f"  {aid}: role={a.role.value}, turns={a.turn_count}, findings={a.findings_count}, status={a.status}"
            for aid, a in self._agents.items()
        ])

        review_prompt = f"""CONDUCTOR REVIEW — {len(self._findings)} total findings, {self._total_turns} total turns

AGENT STATUS:
{agent_status}

RECENT FINDINGS (last {len(recent)}):
{findings_text}

CONTRADICTIONS DETECTED: {len(self._contradictions)}

Analyze the swarm's progress. Respond with your strategic assessment and directives as JSON."""

        response_buffer = []
        response_complete = threading.Event()

        def on_event(event: dict) -> None:
            etype = event.get("type", "")
            if etype == "assistant" and "message" in event:
                msg = event["message"]
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                response_buffer.append(block.get("text", ""))
                    elif isinstance(content, str):
                        response_buffer.append(content)
            elif etype == "result":
                response_complete.set()

        # Re-register handler for this review
        self._conductor.proxy.on_event(on_event)

        try:
            self._conductor.proxy.send(review_prompt)
            response_complete.wait(timeout=300)

            response = "".join(response_buffer)
            if response:
                self._process_conductor_directives(response)
                self._conductor.turn_count += 1
        except Exception as e:
            self._on_log(f"conductor review failed: {e}")

    def _process_conductor_directives(self, response: str) -> None:
        """Process conductor's strategic directives."""
        # Try to parse JSON from response
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                directives = json.loads(response[start:end])
            else:
                self._on_log(f"conductor response (no JSON): {response[:200]}")
                return
        except json.JSONDecodeError:
            self._on_log(f"conductor response (invalid JSON): {response[:200]}")
            return

        # Process broadcasts
        for msg in directives.get("broadcasts", []):
            self.inject(msg)

        # Process spawn requests
        for req in directives.get("spawn_requests", []):
            role_str = req.get("role", "specialist")
            try:
                role = AgentRole(role_str)
            except ValueError:
                role = AgentRole.SPECIALIST
            self.spawn_agent(role, focus=req.get("focus", ""))

        # Process kill requests
        for agent_id in directives.get("kill_requests", []):
            self.kill_agent(agent_id)

        # Log strategy update
        if "strategy_update" in directives:
            self._on_log(f"CONDUCTOR: {directives['strategy_update'][:200]}")
            push_memory(
                raw_text=f"SWARM STRATEGY: {directives['strategy_update']}",
                source="swarm:conductor",
                tm_label="swarm_strategy",
                regime_tag="swarm",
                project=self.project,
            )

        # Track contradictions
        for c in directives.get("contradictions", []):
            self._contradictions.append(c)

        self._persist_state()

    def _kill_agent_internal(self, agent: AgentState) -> None:
        """Stop an agent's proxy."""
        agent.status = "killed"
        if agent.proxy:
            try:
                agent.proxy.stop()
            except Exception:
                pass

    # ── State Persistence ───────────────────────────────────────

    def _persist_state(self) -> None:
        """Save swarm state to disk for crash recovery."""
        state = {
            "goal": self.goal,
            "project": self.project,
            "model": self.model,
            "running": self._running,
            "start_time": self._start_time,
            "total_turns": self._total_turns,
            "agents": {
                aid: {
                    "role": a.role.value,
                    "status": a.status,
                    "turn_count": a.turn_count,
                    "findings_count": a.findings_count,
                    "focus": a.focus,
                }
                for aid, a in self._agents.items()
            },
            "findings": [
                {
                    "id": f.id,
                    "agent_id": f.agent_id,
                    "content": f.content[:500],
                    "confidence": f.confidence,
                    "timestamp": f.timestamp,
                    "excitation": f.excitation,
                }
                for f in self._findings[-100:]  # last 100
            ],
            "contradictions": self._contradictions[-20:],
            "timestamp": time.time(),
        }

        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    @classmethod
    def load_state(cls) -> Optional[dict]:
        """Load persisted swarm state."""
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    @classmethod
    def clear_state(cls) -> None:
        """Remove persisted state."""
        try:
            os.remove(STATE_FILE)
        except FileNotFoundError:
            pass
