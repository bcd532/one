"""one server — universal API that any frontend can connect to.

Provides a REST API + WebSocket server that powers every integration:
Telegram, Discord, Slack, webhooks, or any custom frontend.

v2: API key authentication, rate limiting, CORS configuration,
    proper error responses with status codes, request logging.

Architecture:
    one server (this) <- REST/WS -> Telegram bot
                      <- REST/WS -> Discord bot
                      <- REST/WS -> Slack bot
                      <- REST/WS -> Custom webhook
                      <- REST/WS -> Terminal TUI (optional)

Each project gets its own session with independent Claude proxy,
memory scope, rules, and auto loop.
"""

import hashlib
import json
import logging
import os
import time
import threading
import uuid
from collections import defaultdict
from typing import Optional, Callable
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from .proxy import ClaudeProxy
from .backend import get_backend
from .auto import AutoLoop

# ── Configuration ─────────────────────────────────────────────────

CONFIG_DIR = os.path.expanduser("~/.one")
API_KEY_FILE = os.path.join(CONFIG_DIR, "api_key")
SERVER_CONFIG_FILE = os.path.join(CONFIG_DIR, "server.json")

# Default CORS origins (can be overridden via config)
DEFAULT_CORS_ORIGINS = {"*"}

# Rate limit defaults
RATE_LIMIT_REQUESTS = 60   # per window
RATE_LIMIT_WINDOW = 60     # seconds

logger = logging.getLogger("one.server")


def _load_server_config() -> dict:
    """Load server configuration from disk."""
    try:
        if os.path.exists(SERVER_CONFIG_FILE):
            with open(SERVER_CONFIG_FILE) as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _get_api_key() -> str:
    """Get or generate API key for authentication."""
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE) as f:
            return f.read().strip()

    # Auto-generate on first run
    key = f"one_{uuid.uuid4().hex[:24]}"
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(API_KEY_FILE, "w") as f:
        f.write(key)
    os.chmod(API_KEY_FILE, 0o600)
    logger.info(f"Generated API key: {key}")
    return key


# ── Rate limiter ──────────────────────────────────────────────────


class RateLimiter:
    """Simple sliding window rate limiter per client IP."""

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for this client."""
        now = time.time()
        with self._lock:
            # Clean old entries
            self._requests[client_id] = [
                t for t in self._requests[client_id]
                if now - t < self.window
            ]
            if len(self._requests[client_id]) >= self.max_requests:
                return False
            self._requests[client_id].append(now)
            return True

    def remaining(self, client_id: str) -> int:
        """Return remaining requests in current window."""
        now = time.time()
        with self._lock:
            active = [t for t in self._requests.get(client_id, []) if now - t < self.window]
            return max(0, self.max_requests - len(active))


# ── Session management ────────────────────────────────────────────


class Session:
    """A single project session with its own Claude proxy and state."""

    def __init__(self, project: str, model: str = "opus", cwd: Optional[str] = None):
        self.id = str(uuid.uuid4())[:8]
        self.project = project
        self.model = model
        self.cwd = cwd or os.getcwd()
        self.proxy: Optional[ClaudeProxy] = None
        self.auto_loop: Optional[AutoLoop] = None
        self.backend = get_backend()
        self.turn_count = 0
        self.total_cost = 0.0
        self.status = "idle"
        self.response_buffer = ""
        self.response_complete = threading.Event()
        self._listeners: list[Callable] = []
        self._started = False
        self.proxy: Optional[ClaudeProxy] = None

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
        assert self.proxy is not None
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
            on_log=lambda _r, t: self._emit("log", t),
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
        self._listeners = [listener for listener in self._listeners if listener != callback]

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
        result = {
            "id": self.id,
            "project": self.project,
            "model": self.model,
            "status": self.status,
            "turns": self.turn_count,
            "cost": self.total_cost,
            "auto_running": self.auto_loop.running if self.auto_loop else False,
        }
        if self.auto_loop and self.auto_loop.running:
            result["auto_progress"] = self.auto_loop.progress
        return result


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


# Global instances
manager = SessionManager()
rate_limiter = RateLimiter()
_api_key: Optional[str] = None
_cors_origins: set[str] = DEFAULT_CORS_ORIGINS
_auth_enabled = True


def _init_server_globals():
    """Initialize server globals from config."""
    global _api_key, _cors_origins, _auth_enabled

    config = _load_server_config()
    _api_key = _get_api_key()
    _cors_origins = set(config.get("cors_origins", ["*"]))
    _auth_enabled = config.get("auth_enabled", True)

    rate_config = config.get("rate_limit", {})
    if rate_config:
        rate_limiter.max_requests = rate_config.get("requests", RATE_LIMIT_REQUESTS)
        rate_limiter.window = rate_config.get("window", RATE_LIMIT_WINDOW)


# ── API Handler ───────────────────────────────────────────────────


class OneAPIHandler(BaseHTTPRequestHandler):
    """REST API for one server with authentication and rate limiting."""

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self._send_cors_headers()
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if not self._check_auth_and_rate():
            return

        parsed = urlparse(self.path)
        path = parsed.path

        try:
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
                self._json_response({
                    "status": "ok",
                    "sessions": len(manager.sessions),
                    "version": "2.0",
                })

            elif path == "/graph":
                from .graph import GRAPH_HTML
                self._html_response(GRAPH_HTML)

            elif path == "/api/graph":
                from .graph import get_graph_data
                self._json_response(get_graph_data())

            elif path.startswith("/api/entity/") and path.endswith("/memories"):
                parts = path.split("/")
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

            elif path == "/api/stats":
                from . import store
                self._json_response(store.stats())

            elif path == "/api/research":
                from .research import research_status
                from . import store
                project = parse_qs(parsed.query).get("project", [store.get_project()])[0]
                self._json_response(research_status(project))

            elif path == "/api/synthesis":
                from .synthesis import get_synthesis_chain
                from . import store
                project = parse_qs(parsed.query).get("project", [store.get_project()])[0]
                self._json_response(get_synthesis_chain(project))

            else:
                self._error(404, "not found")

        except Exception as e:
            logger.exception("GET error")
            self._error(500, f"internal error: {type(e).__name__}")

    def do_POST(self):
        if not self._check_auth_and_rate():
            return

        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()

        try:
            if path == "/api/session/new":
                project = body.get("project", "default")
                model = body.get("model", "opus")
                cwd = body.get("cwd")
                session = manager.create(project, model, cwd)
                session.start()
                self._json_response(session.to_dict(), 201)

            elif path == "/api/send":
                sid = body.get("session_id")
                text = body.get("text", "")
                if not text:
                    self._error(400, "text is required")
                    return

                session = manager.get(sid) if sid else None
                if not session:
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
                if not goal:
                    self._error(400, "goal is required")
                    return

                project = body.get("project", "default")
                session = manager.get(sid) if sid else manager.get_by_project(project)
                if not session:
                    session = manager.create(project)
                    session.start()

                # Check if goal is a file path
                if os.path.isfile(goal):
                    try:
                        with open(goal) as f:
                            goal = f.read()
                    except (OSError, UnicodeDecodeError) as e:
                        self._error(400, f"failed to read goal file: {e}")
                        return

                session.start_auto(goal)
                self._json_response({"status": "auto_started", "session": session.id})

            elif path == "/api/stop":
                sid = body.get("session_id")
                session = manager.get(sid) if sid else None
                if session:
                    session.stop_auto()
                    self._json_response({"status": "stopped"})
                else:
                    self._error(404, "session not found")

            elif path == "/api/token/refresh":
                new_token = body.get("token", "")
                if not new_token:
                    self._error(400, "token is required")
                    return
                from .client import refresh_token
                refresh_token(new_token)
                self._json_response({"status": "refreshed"})

            elif path == "/api/webhook":
                project = body.get("project", "default")
                text = body.get("text", body.get("message", ""))
                if not text:
                    self._error(400, "text or message is required")
                    return
                session = manager.get_by_project(project)
                if not session:
                    session = manager.create(project)
                    session.start()
                session.send(text)
                self._json_response({"status": "sent", "session": session.id})

            elif path == "/api/research/start":
                topic = body.get("topic", "")
                if not topic:
                    self._error(400, "topic is required")
                    return
                project = body.get("project", "default")
                budget = body.get("budget", 10)
                from .research import start_research
                # Run in background thread
                def _run():
                    result = start_research(topic, project, turn_budget=budget)
                    logger.info(f"Research complete: {result}")
                threading.Thread(target=_run, daemon=True).start()
                self._json_response({"status": "research_started", "topic": topic})

            else:
                self._error(404, "not found")

        except Exception as e:
            logger.exception("POST error")
            self._error(500, f"internal error: {type(e).__name__}")

    def _check_auth_and_rate(self) -> bool:
        """Verify authentication and rate limits. Returns True if allowed."""
        # Rate limiting (by client IP)
        client_ip = self.client_address[0]
        if not rate_limiter.is_allowed(client_ip):
            self._error(429, "rate limit exceeded")
            return False

        # Authentication (skip for health check and graph view)
        if _auth_enabled and self.path not in ("/api/health", "/graph"):
            auth_header = self.headers.get("Authorization", "")
            api_key_param = ""

            # Check query string for api_key
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if "api_key" in params:
                api_key_param = params["api_key"][0]

            token = ""
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
            elif api_key_param:
                token = api_key_param

            if _api_key and token != _api_key:
                self._error(401, "invalid or missing API key")
                return False

        return True

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            raw = self.rfile.read(length)
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _send_cors_headers(self):
        """Send CORS headers based on configuration."""
        origin = self.headers.get("Origin", "*")
        if "*" in _cors_origins or origin in _cors_origins:
            self.send_header("Access-Control-Allow-Origin", origin if "*" not in _cors_origins else "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        # Rate limit headers
        client_ip = self.client_address[0]
        self.send_header("X-RateLimit-Remaining", str(rate_limiter.remaining(client_ip)))
        self.send_header("X-RateLimit-Limit", str(rate_limiter.max_requests))
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _html_response(self, html, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(html.encode())

    def _error(self, status, message):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        body = json.dumps({"error": message, "status": status})
        self.wfile.write(body.encode())

    def log_message(self, format, *args):
        """Log requests at DEBUG level to avoid noise."""
        logger.debug(format, *args)


def start_server(host: str = "0.0.0.0", port: int = 4111) -> HTTPServer:
    """Start the one API server."""
    _init_server_globals()
    server = HTTPServer((host, port), OneAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"one server running on {host}:{port}")
    if _auth_enabled:
        logger.info(f"API key: {_api_key}")
    return server
