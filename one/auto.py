"""Autonomous agent loop — Claude drives, Gemma observes, everything persists.

Claude operates as a fully autonomous CTO: plans, researches, implements,
tests with real strategies (not smoke tests), reviews its own work, and
iterates until the goal is genuinely complete. Gemma runs in the background
condensing, extracting rules, and enriching the knowledge graph.

The user watches Claude work in a dedicated auto view. They can observe
the stream, interrupt with /stop, or let it run to completion.

v2: Reflection checkpoints, milestone tracking, quality gates,
    full context recovery on compaction, state serialization.
"""

import json
import os
import threading
import time
from typing import Optional, Callable


AUTO_SYSTEM_PROMPT = """You are operating in AUTONOMOUS MODE. You are a CTO-level engineering demon.

You have been given a goal. You will accomplish it completely, with zero hand-holding. No limits. No breaks. End to end completion.

YOUR RULES:
1. PLAN FIRST. Before touching any code, read the relevant files, understand the architecture, and lay out your approach. State your plan explicitly with numbered steps.

2. RESEARCH BEFORE IMPLEMENTING. If the task involves something you're not 100% certain about, use WebSearch or read documentation FIRST. Do not guess. Do not approximate. Get the facts.

3. REAL TESTS, NOT SMOKE TESTS. Unit tests are NOT ENOUGH. You must:
   - Test edge cases, not just the happy path
   - Test failure modes — what happens when input is garbage?
   - Test INTEGRATION — does this work through the ACTUAL APPLICATION?
   - DO NOT just write pytest files and call it done
   - ACTUALLY RUN the feature as a user would:
     * If you built a CLI command, RUN that command and verify the output
     * If you built a web endpoint, CURL it and check the response
     * If you built a module, IMPORT it from the app entry point and call it
     * If you built a TUI feature, verify the app imports and handler wiring
   - Verify ALL imports resolve. Run: python -c "from module import thing"
   - Verify the function signatures match what the callers expect
   - If the app calls health.get_report(), make sure health HAS get_report()
   - Run the tests. If they fail, fix them. Do not move on with failing tests.
   - If there's no test framework, create one.
   - NEVER assume your code works because it looks right. PROVE it works.

4. REVIEW YOUR OWN WORK. After implementing, re-read what you wrote. Check for:
   - Off-by-one errors, type mismatches, unhandled exceptions
   - Security issues (injection, path traversal, etc.)
   - Performance problems (O(n²) where O(n) is possible)
   - Missing error handling at system boundaries
   - Consistency with the rest of the codebase

5. ITERATE. If something isn't right, fix it. Don't explain why it's broken — fix it. Loop until it's actually correct.

6. COMMIT WHEN DONE. When tests pass and code review is clean, commit with a descriptive message. Then move to the next step.

7. DO NOT ASK FOR PERMISSION. You have full autonomy. Edit files, run commands, create tests, restructure code — whatever the goal requires.

8. DO NOT STOP EARLY. "Good enough" is not done. Done is done. Complete every single aspect of the goal.

9. IF CONTEXT GETS COMPRESSED, re-read the critical files and your plan. Do not lose track of where you are. State "Resuming from step N" and continue.

10. IF YOU HIT AN ERROR YOU CANNOT SOLVE, document it clearly and move to the next part of the goal. Come back to it after completing other steps — fresh perspective helps.

REFLECTION PROTOCOL:
Every 10 turns, pause and reflect:
- What have I accomplished so far? (list completed milestones)
- What is my current understanding of the problem?
- What am I still confused about or uncertain of?
- What should I investigate next and why?
- Am I on the right track or should I pivot?

QUALITY GATES:
Before marking any step as complete, verify:
- Does the implementation actually solve the stated problem?
- Have I tested it with realistic inputs, not just toy examples?
- Could this break something elsewhere in the system?
- Is this the simplest correct solution?
- Did I ACTUALLY RUN IT end-to-end, not just write tests?
- Did I verify every import chain works? Every function name matches?
- Did I test it THE WAY A USER WOULD USE IT, not the way a developer would test it?
- If this is a website, did I actually make HTTP requests to verify?
- If this is a CLI tool, did I actually run the command?
- If this is an API, did I actually call every endpoint?
- SMOKE TESTS ARE FAILURE. If your "test" is just checking that a function returns something, that is NOT a test. A test verifies CORRECT BEHAVIOR under REAL CONDITIONS.

WHEN YOU HAVE COMPLETED THE ENTIRE GOAL with all tests passing, all code reviewed, and all commits made, output exactly: [AUTO_COMPLETE]

GOAL: {goal}

BEGIN."""

# ── Milestone tracking ─────────────────────────────────────────────

MILESTONE_STATE_FILE = os.path.expanduser("~/.one/auto_state.json")


def _save_state(state: dict) -> None:
    """Persist auto loop state to disk for crash recovery."""
    try:
        os.makedirs(os.path.dirname(MILESTONE_STATE_FILE), exist_ok=True)
        with open(MILESTONE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


def _load_state() -> Optional[dict]:
    """Load persisted auto loop state."""
    try:
        if os.path.exists(MILESTONE_STATE_FILE):
            with open(MILESTONE_STATE_FILE) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _clear_state() -> None:
    """Remove persisted state after completion."""
    try:
        if os.path.exists(MILESTONE_STATE_FILE):
            os.remove(MILESTONE_STATE_FILE)
    except OSError:
        pass


class AutoLoop:
    """Claude-driven autonomous execution with Gemma background enrichment.

    No turn limits. Handles context compaction, crash recovery, file-based
    goals, and continuous operation until the goal is genuinely complete.

    v2 additions: reflection checkpoints, milestone tracking, quality gates,
    full state serialization for crash recovery.
    """

    REFLECTION_INTERVAL = 10  # Reflect every N turns
    STALL_TIMEOUT = 300       # 5 minutes without response = stalled
    MAX_CONSECUTIVE_ERRORS = 3

    def __init__(
        self,
        proxy,
        on_status: Optional[Callable[[str], None]] = None,
        on_log: Optional[Callable[[str, str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        project: str = "global",
    ):
        self.proxy = proxy
        self.on_status = on_status or (lambda s: None)
        self.on_log = on_log or (lambda r, t: None)
        self.on_complete = on_complete or (lambda s: None)
        self.project = project

        self._running = False
        self._turn_count = 0
        self._goal = ""
        self._goal_file = ""
        self._accumulated_text = ""
        self._step_texts: list[str] = []
        self._last_response_time = 0.0
        self._consecutive_errors = 0
        self._context_compacted = False

        # v2: Milestone tracking
        self._milestones: list[dict] = []
        self._current_step = 0
        self._total_steps = 0
        self._last_reflection_turn = 0
        self._quality_checks_passed = 0
        self._quality_checks_failed = 0

        self._gemma_queue: list[str] = []
        self._gemma_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def progress(self) -> dict:
        """Return current progress metrics."""
        return {
            "turn": self._turn_count,
            "milestones": len(self._milestones),
            "current_step": self._current_step,
            "total_steps": self._total_steps,
            "quality_passed": self._quality_checks_passed,
            "quality_failed": self._quality_checks_failed,
            "goal": self._goal[:100],
        }

    def start(self, goal: str) -> None:
        """Launch autonomous mode. Accepts text goals or file paths."""
        if self._running:
            return

        self._running = True
        self._turn_count = 0
        self._accumulated_text = ""
        self._step_texts = []
        self._consecutive_errors = 0
        self._context_compacted = False
        self._milestones = []
        self._current_step = 0
        self._total_steps = 0
        self._last_reflection_turn = 0
        self._quality_checks_passed = 0
        self._quality_checks_failed = 0

        # Check if goal is a file path
        resolved_goal = self._resolve_goal(goal)
        self._goal = resolved_goal

        # Check for crash recovery
        saved_state = _load_state()
        if saved_state and saved_state.get("goal") == resolved_goal[:200]:
            self._recover_from_state(saved_state)
            return

        self.on_status(f"auto: engaging — {resolved_goal[:60]}")

        threading.Thread(target=self._gemma_background, daemon=True).start()

        auto_prompt = AUTO_SYSTEM_PROMPT.format(goal=resolved_goal)

        context = self._build_context()
        if context:
            full_prompt = f"{context}\n\n{auto_prompt}"
        else:
            full_prompt = auto_prompt

        self._last_response_time = time.time()
        self.proxy.send(full_prompt)
        self._turn_count += 1

        # Save initial state
        self._persist_state()

        # Start watchdog for crash/timeout detection
        threading.Thread(target=self._watchdog, daemon=True).start()

    def stop(self) -> None:
        self._running = False
        self._persist_state()
        self.on_status("auto: stopping after current response")

    def on_response_complete(self, text: str) -> None:
        """Called when Claude finishes a response."""
        if not self._running:
            return

        self._last_response_time = time.time()
        self._consecutive_errors = 0
        self._accumulated_text += text + "\n"
        self._step_texts.append(text)

        with self._gemma_lock:
            self._gemma_queue.append(text)

        # Track milestones from Claude's output
        self._extract_milestones(text)

        # Completion signal
        if "[AUTO_COMPLETE]" in text:
            self._running = False
            self.on_status(f"auto: complete — {self._turn_count} turns, {len(self._milestones)} milestones")
            self.on_complete(f"Completed: {self._goal[:60]}")
            self._store_auto_session()
            _clear_state()
            return

        # Detect context compaction
        compaction_phrases = [
            "context was compressed", "lost track", "can you remind me",
            "what were we working on", "i don't have access to previous",
            "resuming from", "context window",
        ]
        if any(phrase in text.lower() for phrase in compaction_phrases):
            self._context_compacted = True
            self.on_status("auto: context compacted — re-injecting full state")

        self._turn_count += 1
        self.on_status(f"auto: turn {self._turn_count} | {len(self._milestones)} milestones")

        # Persist state periodically
        if self._turn_count % 5 == 0:
            self._persist_state()

        # Build continuation prompt
        if self._context_compacted:
            self._context_compacted = False
            self._send_full_recovery()
        elif self._turn_count - self._last_reflection_turn >= self.REFLECTION_INTERVAL:
            self._send_reflection_prompt()
        else:
            self.proxy.send("Continue. If the entire goal is complete with all tests passing, output [AUTO_COMPLETE].")

    def _extract_milestones(self, text: str) -> None:
        """Extract milestone markers from Claude's output."""
        import re

        # Look for step completion patterns
        step_patterns = [
            r'(?:completed?|finished?|done with)\s+(?:step|phase|part)\s+(\d+)',
            r'step\s+(\d+)\s+(?:is\s+)?(?:complete|done|finished)',
            r'✓\s+step\s+(\d+)',
            r'moving to step\s+(\d+)',
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.I)
            for match in matches:
                step_num = int(match)
                self._current_step = max(self._current_step, step_num)
                self._milestones.append({
                    "step": step_num,
                    "turn": self._turn_count,
                    "summary": text[:200],
                    "timestamp": time.time(),
                })

        # Look for total step count
        total_match = re.search(r'(\d+)\s+(?:total\s+)?steps', text, re.I)
        if total_match:
            self._total_steps = max(self._total_steps, int(total_match.group(1)))

        # Look for quality gate results
        if any(phrase in text.lower() for phrase in ["tests pass", "all tests", "verified", "confirmed working"]):
            self._quality_checks_passed += 1
        if any(phrase in text.lower() for phrase in ["test failed", "tests fail", "broken", "error:"]):
            self._quality_checks_failed += 1

    def _send_reflection_prompt(self) -> None:
        """Send a reflection checkpoint prompt."""
        self._last_reflection_turn = self._turn_count

        milestone_summary = ""
        if self._milestones:
            recent = self._milestones[-5:]
            milestone_summary = "\n".join(
                f"  - Turn {m['turn']}: Step {m['step']}"
                for m in recent
            )
        else:
            milestone_summary = "  (no milestones recorded yet)"

        prompt = (
            f"REFLECTION CHECKPOINT (turn {self._turn_count}):\n\n"
            f"Goal: {self._goal[:500]}\n\n"
            f"Recent milestones:\n{milestone_summary}\n\n"
            f"Quality: {self._quality_checks_passed} passed, {self._quality_checks_failed} failed\n\n"
            f"Pause and reflect:\n"
            f"1. What have you accomplished so far?\n"
            f"2. What is your current understanding?\n"
            f"3. What are you uncertain about?\n"
            f"4. What should you investigate next?\n"
            f"5. Are you on track or should you pivot?\n\n"
            f"After reflecting, continue working. If complete, output [AUTO_COMPLETE]."
        )
        self.proxy.send(prompt)

    def _send_full_recovery(self) -> None:
        """Send a complete state recovery after context compaction."""
        context = self._build_context()

        # Build milestone history
        milestone_text = ""
        if self._milestones:
            milestone_text = "\n\nCOMPLETED MILESTONES:\n" + "\n".join(
                f"  Turn {m['turn']}: Step {m['step']} — {m['summary'][:100]}"
                for m in self._milestones
            )

        # Include recent step summaries
        recent_steps = ""
        if self._step_texts:
            last_n = self._step_texts[-3:]
            recent_steps = "\n\nLAST 3 RESPONSES (most recent):\n" + "\n---\n".join(
                text[:500] for text in last_n
            )

        resume = (
            f"CONTEXT WAS COMPACTED. Full state recovery:\n\n"
            f"GOAL: {self._goal[:2000]}\n\n"
            f"{context}\n"
            f"{milestone_text}\n"
            f"{recent_steps}\n\n"
            f"You were on turn {self._turn_count}. "
            f"Current progress: step {self._current_step}/{self._total_steps or '?'}. "
            f"Quality: {self._quality_checks_passed} passed, {self._quality_checks_failed} failed.\n\n"
            f"Review the current state and resume where you left off. "
            f"State which step you're resuming from."
        )
        self.proxy.send(resume)

    def _recover_from_state(self, state: dict) -> None:
        """Recover from a previously saved state (crash recovery)."""
        self._turn_count = state.get("turn_count", 0)
        self._milestones = state.get("milestones", [])
        self._current_step = state.get("current_step", 0)
        self._total_steps = state.get("total_steps", 0)
        self._quality_checks_passed = state.get("quality_passed", 0)
        self._quality_checks_failed = state.get("quality_failed", 0)

        self.on_status(f"auto: recovering from turn {self._turn_count}")

        threading.Thread(target=self._gemma_background, daemon=True).start()

        context = self._build_context()
        resume = (
            f"CRASH RECOVERY — resuming autonomous mode.\n\n"
            f"GOAL: {self._goal[:2000]}\n\n"
            f"{context}\n\n"
            f"You were on turn {self._turn_count}, step {self._current_step}/{self._total_steps or '?'}. "
            f"{len(self._milestones)} milestones completed. "
            f"Review the codebase state and resume."
        )

        self._last_response_time = time.time()
        self.proxy.send(resume)
        self._turn_count += 1

        threading.Thread(target=self._watchdog, daemon=True).start()

    def _persist_state(self) -> None:
        """Save current auto loop state to disk."""
        state = {
            "goal": self._goal[:200],
            "turn_count": self._turn_count,
            "milestones": self._milestones,
            "current_step": self._current_step,
            "total_steps": self._total_steps,
            "quality_passed": self._quality_checks_passed,
            "quality_failed": self._quality_checks_failed,
            "project": self.project,
            "timestamp": time.time(),
        }
        _save_state(state)

    def _resolve_goal(self, goal: str) -> str:
        """Resolve a goal — if it's a file path, read and use its contents."""
        stripped = goal.strip()

        for path in [stripped, os.path.expanduser(stripped)]:
            if os.path.isfile(path):
                try:
                    with open(path, "r") as f:
                        content = f.read()
                    self._goal_file = path
                    self.on_status(f"auto: loaded goal from {path}")
                    return content
                except (OSError, UnicodeDecodeError):
                    pass

        return stripped

    def _watchdog(self) -> None:
        """Monitor for crashes, timeouts, and connection drops."""
        while self._running:
            time.sleep(10)

            elapsed = time.time() - self._last_response_time

            if elapsed > self.STALL_TIMEOUT:
                self._consecutive_errors += 1
                self.on_status(f"auto: no response for {int(elapsed)}s — retry #{self._consecutive_errors}")

                if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    self.on_status("auto: 3 consecutive timeouts — stopping")
                    self._running = False
                    self._persist_state()
                    self._store_auto_session()
                    return

                # Re-send with state recovery
                self.proxy.send(
                    f"You appear to have stalled on turn {self._turn_count}. "
                    f"Goal: {self._goal[:300]}. "
                    f"Current step: {self._current_step}/{self._total_steps or '?'}. "
                    f"Review the codebase state and continue."
                )
                self._last_response_time = time.time()

    def _build_context(self) -> str:
        """Gather project context for the auto run, including relevant playbooks."""
        parts = []

        try:
            from .rules import get_active_rules, format_rules_for_injection
            rules = get_active_rules(self.project, self._goal)
            if rules:
                parts.append(format_rules_for_injection(rules, self.project))
        except Exception:
            pass

        try:
            from .backend import get_backend
            backend = get_backend()
            ctx = backend.recall_context(self._goal, n=10, max_chars=3000, use_gemma=False)
            if ctx:
                parts.append(ctx)
        except Exception:
            pass

        # Inject relevant playbooks from prior successful runs
        try:
            from .playbook import recall_playbook_context
            pb_ctx = recall_playbook_context(self.project, self._goal)
            if pb_ctx:
                parts.append(pb_ctx)
        except Exception:
            pass

        return "\n\n".join(parts) if parts else ""

    def _store_auto_session(self) -> None:
        """Store the auto run as a condensed memory and generate a playbook."""
        try:
            from . import store
            from .hdc import encode_tagged

            # Build a richer summary
            milestone_text = ""
            if self._milestones:
                milestone_text = " | Milestones: " + ", ".join(
                    f"step {m['step']}" for m in self._milestones[-10:]
                )

            summary = (
                f"Auto goal: {self._goal[:300]}\n"
                f"Turns: {self._turn_count} | Steps: {len(self._step_texts)}"
                f"{milestone_text}\n"
                f"Quality: {self._quality_checks_passed} passed, {self._quality_checks_failed} failed"
            )
            vec = encode_tagged(summary, role="auto")
            store.push_memory(
                summary, "auto", "auto_session", "condensed", 0.95, vec.tolist(),
                project=self.project,
            )
        except Exception:
            pass

        self._generate_playbook()

    def _generate_playbook(self) -> None:
        """Distill the completed auto session into a reusable playbook."""
        try:
            from .playbook import create_playbook

            steps_summary = "\n".join(
                f"Step {i + 1}: {text[:300]}"
                for i, text in enumerate(self._step_texts[-20:])
            )

            completed = "[AUTO_COMPLETE]" in (self._accumulated_text or "")
            outcome = "completed successfully" if completed else f"stopped after {self._turn_count} turns"

            # Include quality metrics in outcome
            outcome += f" (quality: {self._quality_checks_passed}/{self._quality_checks_passed + self._quality_checks_failed} gates passed)"

            create_playbook(
                project=self.project,
                task_description=self._goal[:500],
                steps_taken=steps_summary,
                outcome=outcome,
            )
            self.on_log("auto", "playbook generated from session")
        except Exception:
            pass

    def _gemma_background(self) -> None:
        """Background thread: Gemma processes completed steps for the knowledge graph."""
        try:
            from .gemma import is_available, condense_memories
            if not is_available():
                return
        except Exception:
            return

        while self._running:
            time.sleep(5)

            with self._gemma_lock:
                if not self._gemma_queue:
                    continue
                batch = self._gemma_queue[:]
                self._gemma_queue.clear()

            # Extract rules from Claude's decisions
            for text in batch:
                self._extract_auto_rules(text)

            # Condense if enough has accumulated
            if len(self._step_texts) >= 5 and len(self._step_texts) % 5 == 0:
                try:
                    memories = [{"raw_text": t, "source": "assistant", "tm_label": "auto", "aif_confidence": 0.7} for t in self._step_texts[-5:]]
                    condensed = condense_memories(memories)
                    if condensed:
                        from . import store
                        from .hdc import encode_tagged
                        vec = encode_tagged(condensed, role="condensed")
                        store.push_memory(
                            condensed, "condensed", "auto_condensed", "condensed", 0.9, vec.tolist(),
                            project=self.project,
                        )
                        self.on_log("auto", f"condensed {len(memories)} steps")
                except Exception:
                    pass

    def _extract_auto_rules(self, text: str) -> None:
        """Look for decision patterns in Claude's autonomous output and learn rules."""
        import re

        decision_patterns = [
            r"(?:I'm going to|I'll|Let's|I decided to|The approach is|I'm using) (.{20,100})",
            r"(?:because|the reason is|this is better because) (.{20,100})",
            r"(?:IMPORTANT|NOTE|WARNING|RULE): (.{10,100})",
        ]

        for pattern in decision_patterns:
            matches = re.findall(pattern, text, re.I)
            for match in matches[:2]:
                try:
                    from .rules import learn_rule_from_memory
                    learn_rule_from_memory(self.project, match.strip(), confidence=0.6)
                except Exception:
                    pass
