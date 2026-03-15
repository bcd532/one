"""Autonomous agent loop — Gemma plans, Claude executes, everything persists.

Gemma acts as project manager: reads codebase and memory, breaks goals into
steps, evaluates Claude's output, identifies knowledge gaps, and loops until
the goal is met or the turn budget is exhausted.

Usage:
    /auto build the HDC encoder with better trigram handling
    /auto research active inference and write a summary
    /auto review all files in one/ and fix any bugs
"""

import json
import time
import threading
import subprocess
from typing import Optional, Callable
from datetime import datetime, timezone

from .gemma import _call_ollama, is_available as gemma_available


PLAN_PROMPT = """You are a technical project manager. Given a goal and project context, break it into concrete steps.

Rules:
- Each step is ONE action: research, read a file, write code, run tests, review, or commit
- Steps must be specific: "read one/hdc.py" not "look at the code"
- Include research steps BEFORE implementation when the topic needs it
- Include a test step after any code change
- Include a review step before any commit
- Maximum 15 steps
- Output as a JSON array of objects: {{"step": 1, "action": "research|read|write|test|review|commit", "description": "what to do", "target": "file or topic"}}

GOAL: {goal}

CONTEXT:
{context}

STEPS (JSON array):"""

EVALUATE_PROMPT = """You are reviewing an AI coding agent's output for a specific task step.

The step was: {step_description}
The agent responded with:

---
{result}
---

Evaluate:
1. Did the agent complete the step correctly?
2. Is more research needed before continuing?
3. Are there issues that need fixing?

Reply with ONLY a JSON object:
{{"complete": true/false, "needs_research": true/false, "research_query": "what to research if needed", "feedback": "what to fix if incomplete", "approved": true/false}}"""

FORMULATE_PROMPT = """You are formulating a precise instruction for an AI coding agent.

The current step is: {step_description}
Action type: {action}
Target: {target}

Previous context:
{previous_results}

Write a clear, specific instruction for the agent. Be precise about file paths and what changes to make. Output ONLY the instruction, nothing else."""


class AutoLoop:
    """Autonomous goal execution loop."""

    def __init__(
        self,
        send_to_claude: Callable[[str], None],
        on_status: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        max_turns: int = 50,
        project: str = "global",
    ):
        self.send_to_claude = send_to_claude
        self.on_status = on_status or (lambda s: None)
        self.on_complete = on_complete or (lambda s: None)
        self.max_turns = max_turns
        self.project = project

        self._running = False
        self._turn_count = 0
        self._plan: list[dict] = []
        self._current_step = 0
        self._results: list[str] = []
        self._pending_response = threading.Event()
        self._last_response = ""
        self._goal = ""

    @property
    def running(self) -> bool:
        return self._running

    def start(self, goal: str) -> None:
        """Start the autonomous loop in a background thread."""
        if self._running:
            return
        self._goal = goal
        self._running = True
        self._turn_count = 0
        threading.Thread(target=self._run, args=(goal,), daemon=True).start()

    def stop(self) -> None:
        """Stop the loop after the current step completes."""
        self._running = False

    def feed_response(self, text: str) -> None:
        """Feed Claude's response back into the loop."""
        self._last_response = text
        self._pending_response.set()

    def _run(self, goal: str) -> None:
        """Main loop: plan → execute → evaluate → repeat."""
        if not gemma_available():
            self.on_status("auto: gemma not available")
            self._running = False
            return

        self.on_status(f"auto: planning — {goal[:60]}")

        # Build context from memory
        context = self._build_context()

        # Generate plan
        plan = self._generate_plan(goal, context)
        if not plan:
            self.on_status("auto: failed to generate plan")
            self._running = False
            return

        self._plan = plan
        self.on_status(f"auto: {len(plan)} steps planned")

        # Execute each step
        for i, step in enumerate(plan):
            if not self._running:
                self.on_status("auto: stopped by user")
                break

            if self._turn_count >= self.max_turns:
                self.on_status(f"auto: hit turn limit ({self.max_turns})")
                break

            self._current_step = i + 1
            action = step.get("action", "")
            desc = step.get("description", "")
            target = step.get("target", "")

            self.on_status(f"auto: step {i+1}/{len(plan)} — {desc[:50]}")

            # Formulate the instruction for Claude
            instruction = self._formulate_instruction(step)
            if not instruction:
                instruction = desc

            # Send to Claude and wait for response
            self._pending_response.clear()
            self._last_response = ""
            self.send_to_claude(f"[auto step {i+1}/{len(plan)}] {instruction}")
            self._turn_count += 1

            # Wait for Claude to respond (timeout: 5 minutes per step)
            if not self._pending_response.wait(timeout=300):
                self.on_status(f"auto: step {i+1} timed out")
                continue

            result = self._last_response
            self._results.append(result)

            # Evaluate the result
            evaluation = self._evaluate_result(step, result)

            if evaluation.get("needs_research"):
                query = evaluation.get("research_query", desc)
                self.on_status(f"auto: researching — {query[:50]}")
                self._pending_response.clear()
                self.send_to_claude(f"[auto research] {query}")
                self._turn_count += 1
                self._pending_response.wait(timeout=300)
                self._results.append(self._last_response)

            if not evaluation.get("complete", True) and evaluation.get("feedback"):
                feedback = evaluation["feedback"]
                self.on_status(f"auto: fixing — {feedback[:50]}")
                self._pending_response.clear()
                self.send_to_claude(f"[auto fix] {feedback}")
                self._turn_count += 1
                self._pending_response.wait(timeout=300)
                self._results.append(self._last_response)

        # Done
        self._running = False
        summary = f"auto: completed {self._current_step}/{len(plan)} steps in {self._turn_count} turns"
        self.on_status(summary)
        self.on_complete(summary)

        # Store the auto session as a condensed memory
        try:
            from .store import push_memory
            from .hdc import encode_tagged
            condensed = f"Auto goal: {goal}\nSteps completed: {self._current_step}/{len(plan)}\nTurns: {self._turn_count}"
            vec = encode_tagged(condensed, role="auto")
            push_memory(condensed, "auto", "auto_session", "condensed", 0.9, vec.tolist())
        except Exception:
            pass

    def _build_context(self) -> str:
        """Gather project context from memory, rules, and recent files."""
        parts = []

        try:
            from .rules import get_active_rules, format_rules_for_injection
            rules = get_active_rules(self.project, "")
            if rules:
                parts.append(format_rules_for_injection(rules, self.project))
        except Exception:
            pass

        try:
            from . import store
            store.set_project(self.project)
            s = store.stats()
            parts.append(f"Memory: {s['memories']} entries, {s['entities']} entities")
        except Exception:
            pass

        # Recent git status
        try:
            r = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                parts.append(f"Git status:\n{r.stdout.strip()}")
        except Exception:
            pass

        return "\n\n".join(parts) if parts else "No prior context available."

    def _generate_plan(self, goal: str, context: str) -> list[dict]:
        """Use Gemma to break the goal into steps."""
        prompt = PLAN_PROMPT.format(goal=goal, context=context[:2000])
        result = _call_ollama(prompt, timeout=60)
        if not result:
            return []

        try:
            # Extract JSON from response
            import re
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return []

    def _formulate_instruction(self, step: dict) -> str:
        """Use Gemma to write a precise instruction for Claude."""
        recent = "\n".join(self._results[-3:]) if self._results else "No previous results."
        if len(recent) > 1500:
            recent = recent[-1500:]

        prompt = FORMULATE_PROMPT.format(
            step_description=step.get("description", ""),
            action=step.get("action", ""),
            target=step.get("target", ""),
            previous_results=recent,
        )
        return _call_ollama(prompt, timeout=30)

    def _evaluate_result(self, step: dict, result: str) -> dict:
        """Use Gemma to evaluate Claude's output."""
        if len(result) > 2000:
            result = result[:2000] + "... (truncated)"

        prompt = EVALUATE_PROMPT.format(
            step_description=step.get("description", ""),
            result=result,
        )
        response = _call_ollama(prompt, timeout=30)
        if not response:
            return {"complete": True, "approved": True}

        try:
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return {"complete": True, "approved": True}
