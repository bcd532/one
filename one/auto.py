"""Autonomous agent loop — Claude drives, Gemma observes, everything persists.

Claude operates as a fully autonomous CTO: plans, researches, implements,
tests with real strategies (not smoke tests), reviews its own work, and
iterates until the goal is genuinely complete. Gemma runs in the background
condensing, extracting rules, and enriching the knowledge graph.

The user watches Claude work in a dedicated auto view. They can observe
the stream, interrupt with /stop, or let it run to completion.
"""

import json
import threading
import time
from typing import Optional, Callable


AUTO_SYSTEM_PROMPT = """You are operating in AUTONOMOUS MODE. You are a CTO-level engineering demon.

You have been given a goal. You will accomplish it completely, with zero hand-holding.

YOUR RULES:
1. PLAN FIRST. Before touching any code, read the relevant files, understand the architecture, and lay out your approach. State your plan explicitly.

2. RESEARCH BEFORE IMPLEMENTING. If the task involves something you're not 100% certain about, use WebSearch or read documentation FIRST. Do not guess. Do not approximate. Get the facts.

3. REAL TESTS, NOT SMOKE TESTS. When you write tests:
   - Test edge cases, not just the happy path
   - Test failure modes — what happens when input is garbage?
   - Test integration — does this work with the rest of the system?
   - Run the tests. If they fail, fix them. Do not move on with failing tests.

4. REVIEW YOUR OWN WORK. After implementing, re-read what you wrote. Check for:
   - Off-by-one errors, type mismatches, unhandled exceptions
   - Security issues (injection, path traversal, etc.)
   - Performance problems (O(n²) where O(n) is possible)
   - Missing error handling at system boundaries

5. ITERATE. If something isn't right, fix it. Don't explain why it's broken — fix it. Loop until it's actually correct.

6. COMMIT WHEN DONE. When tests pass and code review is clean, commit with a descriptive message. Then move to the next step.

7. DO NOT ASK FOR PERMISSION. You have full autonomy. Edit files, run commands, create tests, restructure code — whatever the goal requires.

8. DO NOT STOP EARLY. "Good enough" is not done. Done is done.

WHEN YOU HAVE COMPLETED THE ENTIRE GOAL, output exactly: [AUTO_COMPLETE]

GOAL: {goal}

BEGIN."""


class AutoLoop:
    """Claude-driven autonomous execution with Gemma background enrichment."""

    def __init__(
        self,
        proxy,
        on_status: Optional[Callable[[str], None]] = None,
        on_log: Optional[Callable[[str, str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        max_turns: int = 100,
        project: str = "global",
    ):
        self.proxy = proxy
        self.on_status = on_status or (lambda s: None)
        self.on_log = on_log or (lambda r, t: None)
        self.on_complete = on_complete or (lambda s: None)
        self.max_turns = max_turns
        self.project = project

        self._running = False
        self._turn_count = 0
        self._goal = ""
        self._accumulated_text = ""
        self._step_texts: list[str] = []

        # Background Gemma thread for condensation
        self._gemma_queue: list[str] = []
        self._gemma_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._running

    def start(self, goal: str) -> None:
        """Launch autonomous mode. Sends the goal to Claude with the CTO prompt."""
        if self._running:
            return

        self._running = True
        self._goal = goal
        self._turn_count = 0
        self._accumulated_text = ""
        self._step_texts = []

        self.on_status(f"auto: engaging — {goal[:60]}")

        # Start Gemma background enrichment thread
        threading.Thread(target=self._gemma_background, daemon=True).start()

        # Build the auto system prompt with goal
        auto_prompt = AUTO_SYSTEM_PROMPT.format(goal=goal)

        # Inject recalled context + rules
        context = self._build_context()
        if context:
            full_prompt = f"{context}\n\n{auto_prompt}"
        else:
            full_prompt = auto_prompt

        # Send to Claude
        self.proxy.send(full_prompt)
        self._turn_count += 1

    def stop(self) -> None:
        self._running = False
        self.on_status("auto: stopping after current response")

    def on_response_complete(self, text: str) -> None:
        """Called when Claude finishes a response. Decides whether to continue."""
        if not self._running:
            return

        self._accumulated_text += text + "\n"
        self._step_texts.append(text)

        # Queue for Gemma background processing
        with self._gemma_lock:
            self._gemma_queue.append(text)

        # Check if Claude declared completion
        if "[AUTO_COMPLETE]" in text:
            self._running = False
            self.on_status(f"auto: goal complete in {self._turn_count} turns")
            self.on_complete(f"Completed: {self._goal}")
            self._store_auto_session()
            return

        # Check turn budget
        self._turn_count += 1
        if self._turn_count >= self.max_turns:
            self._running = False
            self.on_status(f"auto: hit {self.max_turns} turn limit")
            self.on_complete(f"Budget exhausted: {self._goal}")
            self._store_auto_session()
            return

        # Continue — Claude drives itself. Send a minimal continuation.
        # If Claude stopped at end_turn, it needs a nudge to keep going.
        self.on_status(f"auto: turn {self._turn_count}/{self.max_turns}")
        self.proxy.send("Continue. If you are done with the entire goal, output [AUTO_COMPLETE].")

    def _build_context(self) -> str:
        """Gather project context for the auto run."""
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

        return "\n\n".join(parts) if parts else ""

    def _store_auto_session(self) -> None:
        """Store the auto run as a condensed memory."""
        try:
            from . import store
            from .hdc import encode_tagged

            summary = f"Auto goal: {self._goal}\nTurns: {self._turn_count}\nSteps: {len(self._step_texts)}"
            vec = encode_tagged(summary, role="auto")
            store.push_memory(
                summary, "auto", "auto_session", "condensed", 0.95, vec.tolist(),
                project=self.project,
            )
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
