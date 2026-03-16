"""Bidirectional JSON-stream proxy for the Claude CLI.

Wraps the `claude` command-line tool in a subprocess, communicating via
newline-delimited JSON on stdin/stdout. Supports session management,
permission modes, and event-driven output handling.
"""

import json
import subprocess
import threading
import os
from typing import Optional, Callable


class ClaudeProxy:
    def __init__(
        self,
        model: str = "opus",
        cwd: Optional[str] = None,
        session_id: Optional[str] = None,
        resume: bool = False,
        continue_last: bool = False,
        permission_mode: str = "acceptEdits",
        system_prompt: Optional[str] = None,
        append_system_prompt: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        disallowed_tools: Optional[list[str]] = None,
    ):
        self.model = model
        self.cwd = cwd or os.getcwd()
        self.session_id = session_id
        self.resume = resume
        self.continue_last = continue_last
        self.permission_mode = permission_mode
        self.system_prompt = system_prompt
        self.append_system_prompt = append_system_prompt
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
        self.process: Optional[subprocess.Popen] = None
        self._on_event: Optional[Callable] = None

    def on_event(self, callback: Callable):
        self._on_event = callback

    def start(self):
        cmd = [
            "claude", "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model", self.model,
            "--permission-mode", self.permission_mode,
            "--replay-user-messages",
        ]

        if self.continue_last:
            cmd.append("--continue")
        elif self.resume and self.session_id:
            cmd.extend(["--resume", self.session_id])
        elif self.session_id:
            cmd.extend(["--session-id", self.session_id])

        if self.system_prompt:
            cmd.extend(["--system-prompt", self.system_prompt])
        if self.append_system_prompt:
            cmd.extend(["--append-system-prompt", self.append_system_prompt])
        if self.allowed_tools:
            cmd.extend(["--allowedTools"] + self.allowed_tools)
        if self.disallowed_tools:
            cmd.extend(["--disallowedTools"] + self.disallowed_tools)

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            text=True,
            bufsize=1,
        )

        threading.Thread(target=self._read_output, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def send(self, text: str):
        msg = {"type": "user", "message": {"role": "user", "content": text}}
        try:
            self.process.stdin.write(json.dumps(msg) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def stop(self):
        if self.process:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _read_output(self):
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if self._on_event:
                self._on_event(event)

    def _read_stderr(self):
        for line in self.process.stderr:
            pass

    def ask(self, prompt: str, timeout: int = 120) -> Optional[str]:
        """Synchronous: send prompt, wait for full response, return text.

        Uses claude -p (print mode) for clean one-shot responses.
        No stream-json, no event parsing — just subprocess in, text out.
        """
        import subprocess as _sp

        cmd = ["claude", "-p", "--model", self.model, "--output-format", "text"]
        if self.system_prompt:
            cmd.extend(["--system-prompt", self.system_prompt])

        try:
            result = _sp.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (_sp.TimeoutExpired, OSError):
            pass
        return None

    @staticmethod
    def quick_ask(prompt: str, model: str = "sonnet", cwd: str = ".", timeout: int = 120) -> Optional[str]:
        """One-shot: ask Claude a question, get the answer. Simple subprocess call."""
        import subprocess as _sp

        cmd = [
            "claude", "-p",
            "--model", model,
            "--output-format", "text",
            "--system-prompt", "You are a research engine. Analyze thoroughly. Report findings as detailed bullet points. No preamble.",
        ]

        try:
            result = _sp.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (_sp.TimeoutExpired, OSError):
            pass
        return None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None
