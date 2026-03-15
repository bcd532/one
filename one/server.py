"""one server — universal API that any frontend can connect to.

Provides a REST API + WebSocket server that powers every integration:
Telegram, Discord, Slack, webhooks, or any custom frontend.

Architecture:
    one server (this) ← REST/WS → Telegram bot
                      ← REST/WS → Discord bot
                      ← REST/WS → Slack bot
                      ← REST/WS → Custom webhook
                      ← REST/WS → Terminal TUI (optional)

Each project gets its own session with independent Claude proxy,
memory scope, rules, and auto loop.
"""

import json
import uuid
import threading
import time
import os
from typing import Optional, Callable
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from .proxy import ClaudeProxy
from .backend import get_backend
from .auto import AutoLoop


class Session:
    """A single project session with its own Claude proxy and state."""

    def __init__(self, project: str, model: str = "opus", cwd: Optional[str] = None):
        self.id = str(uuid.uuid4())[:8]
        self.project = project
        self.model = model
        self.cwd = cwd or os.getcwd()
        self.proxy = None
        self.auto_loop = None
        self.backend = get_backend()
        self.turn_count = 0
        self.total_cost = 0.0
        self.status = "idle"
        self.response_buffer = ""
        self.response_complete = threading.Event()
        self._listeners: list[Callable] = []
        self._started = False

        # Set project scope
        from . import store
        store.set_project(project)

    def start(self):
        if self._started:
            return
        self.proxy = ClaudeProxy(
            model=self.model,
            cwd=self.cwd,
            permission_mode="bypassPermissions",
        )
        self.proxy.on_event(self._on_event)
        self.proxy.start()
        self._started = True
        self.status = "ready"

    def send(self, text: str) -> None:
        if not self._started:
            self.start()
        self.response_buffer = ""
        self.response_complete.clear()
        self.status = "thinking"
        self.turn_count += 1
        self.proxy.send(text)

        # Capture and store
        from .gate import AifGate
        gate = AifGate()
        should_store, conf = gate.should_store(text, source="user")
        if should_store:
            from .hdc import encode_tagged
            vec = encode_tagged(text, role="user")
            self.backend.push_memory(text, "user", "unclassified", "default", conf, vec.tolist())

    def start_auto(self, goal: str) -> None:
        if not self._started:
            self.start()

        self.auto_loop = AutoLoop(
            proxy=self.proxy,
            on_status=lambda s: self._emit("status", s),
            on_log=lambda r, t: self._emit("log", t),
            on_complete=lambda s: self._emit("complete", s),
            project=self.project,
        )
        self.auto_loop.start(goal)
        self.status = "auto"

    def stop_auto(self) -> None:
        if self.auto_loop and self.auto_loop.running:
            self.auto_loop.stop()

    def add_listener(self, callback: Callable) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable) -> None:
        self._listeners = [l for l in self._listeners if l != callback]

    def _emit(self, event_type: str, data: str) -> None:
        for listener in self._listeners:
            try:
                listener({"type": event_type, "session": self.id, "project": self.project, "data": data})
            except Exception:
                pass

    def _on_event(self, event: dict) -> None:
        etype = event.get("type", "")

        if etype == "stream_event":
            inner = event.get("event", {})
            delta = inner.get("delta", {})
            dt = delta.get("type", "")

            if dt == "text_delta":
                chunk = delta.get("text", "")
                self.response_buffer += chunk
                self._emit("text", chunk)

            elif dt == "thinking_delta":
                chunk = delta.get("thinking", "")
                self._emit("thinking", chunk)

        elif etype == "assistant":
            msg = event.get("message", {})
            parts = [b["text"] for b in msg.get("content", []) if b.get("type") == "text"]
            full_text = "".join(parts)
            if full_text:
                self._emit("message", full_text)

                # Store assistant response
                try:
                    from .gate import AifGate
                    gate = AifGate()
                    should_store, conf = gate.should_store(full_text, source="assistant")
                    if should_store:
                        from .hdc import encode_tagged
                        vec = encode_tagged(full_text, role="assistant")
                        self.backend.push_memory(full_text, "assistant", "unclassified", "default", conf, vec.tolist())
                except Exception:
                    pass

                # Feed to auto loop
                if self.auto_loop and self.auto_loop.running:
                    self.auto_loop.on_response_complete(full_text)

        elif etype == "result":
            cost = event.get("total_cost_usd", 0)
            self.total_cost += cost
            self.status = "auto" if (self.auto_loop and self.auto_loop.running) else "ready"
            self.response_complete.set()
            self._emit("result", json.dumps({
                "cost": cost,
                "total_cost": self.total_cost,
                "turns": self.turn_count,
            }))

    def stop(self):
        if self.auto_loop:
            self.auto_loop.stop()
        if self.proxy:
            self.proxy.stop()
        self.status = "stopped"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project": self.project,
            "model": self.model,
            "status": self.status,
            "turns": self.turn_count,
            "cost": self.total_cost,
            "auto_running": self.auto_loop.running if self.auto_loop else False,
        }


class SessionManager:
    """Manages multiple project sessions."""

    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, project: str, model: str = "opus", cwd: Optional[str] = None) -> Session:
        with self._lock:
            session = Session(project, model, cwd)
            self.sessions[session.id] = session
            return session

    def get(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def get_by_project(self, project: str) -> Optional[Session]:
        for s in self.sessions.values():
            if s.project == project and s.status != "stopped":
                return s
        return None

    def list_all(self) -> list[dict]:
        return [s.to_dict() for s in self.sessions.values() if s.status != "stopped"]

    def stop_all(self):
        for s in self.sessions.values():
            s.stop()


# Global session manager
manager = SessionManager()


class OneAPIHandler(BaseHTTPRequestHandler):
    """REST API for one server."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/sessions":
            self._json_response(manager.list_all())

        elif path == "/api/token":
            from .client import token_status
            self._json_response(token_status())

        elif path.startswith("/api/session/"):
            sid = path.split("/")[-1]
            session = manager.get(sid)
            if session:
                self._json_response(session.to_dict())
            else:
                self._error(404, "session not found")

        elif path == "/api/health":
            self._json_response({"status": "ok", "sessions": len(manager.sessions)})

        elif path == "/graph":
            from .graph import GRAPH_HTML
            self._html_response(GRAPH_HTML)

        elif path == "/api/graph":
            from .graph import get_graph_data
            self._json_response(get_graph_data())

        elif path.startswith("/api/entity/") and path.endswith("/memories"):
            # /api/entity/<name>/memories
            parts = path.split("/")
            # parts: ['', 'api', 'entity', '<name>', 'memories']
            if len(parts) >= 5:
                from urllib.parse import unquote
                entity_name = unquote(parts[3])
                from .graph import get_entity_memories
                self._json_response(get_entity_memories(entity_name))
            else:
                self._error(400, "invalid entity path")

        elif path == "/api/claudemd":
            from .claudemd import generate_claude_md
            from . import store
            project = parse_qs(parsed.query).get("project", [store.get_project()])[0]
            self._json_response({"project": project, "content": generate_claude_md(project)})

        else:
            self._error(404, "not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()

        if path == "/api/session/new":
            project = body.get("project", "default")
            model = body.get("model", "opus")
            cwd = body.get("cwd")
            session = manager.create(project, model, cwd)
            session.start()
            self._json_response(session.to_dict())

        elif path == "/api/send":
            sid = body.get("session_id")
            text = body.get("text", "")
            session = manager.get(sid)
            if not session:
                # Auto-create by project name
                project = body.get("project", "default")
                session = manager.get_by_project(project)
                if not session:
                    session = manager.create(project)
                    session.start()
            session.send(text)
            self._json_response({"status": "sent", "session": session.id})

        elif path == "/api/auto":
            sid = body.get("session_id")
            goal = body.get("goal", "")
            project = body.get("project", "default")
            session = manager.get(sid) if sid else manager.get_by_project(project)
            if not session:
                session = manager.create(project)
                session.start()

            # Check if goal is a file path
            if os.path.isfile(goal):
                with open(goal) as f:
                    goal = f.read()

            session.start_auto(goal)
            self._json_response({"status": "auto_started", "session": session.id})

        elif path == "/api/stop":
            sid = body.get("session_id")
            session = manager.get(sid)
            if session:
                session.stop_auto()
                self._json_response({"status": "stopped"})
            else:
                self._error(404, "session not found")

        elif path == "/api/token/refresh":
            new_token = body.get("token", "")
            if new_token:
                from .client import refresh_token
                refresh_token(new_token)
                self._json_response({"status": "refreshed"})
            else:
                self._error(400, "no token provided")

        elif path == "/api/webhook":
            # Generic webhook — route to appropriate session
            project = body.get("project", "default")
            text = body.get("text", body.get("message", ""))
            session = manager.get_by_project(project)
            if not session:
                session = manager.create(project)
                session.start()
            session.send(text)
            self._json_response({"status": "sent", "session": session.id})

        else:
            self._error(404, "not found")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _html_response(self, html, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode())

    def _error(self, status, message):
        self._json_response({"error": message}, status)

    def log_message(self, format, *args):
        pass  # silence request logs


def start_server(host: str = "0.0.0.0", port: int = 4111) -> HTTPServer:
    """Start the one API server."""
    server = HTTPServer((host, port), OneAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
