"""Tests for one/server.py — RateLimiter, API key helpers, CORS config loading,
and the _init_server_globals / _load_server_config / _get_api_key functions.

No HTTP server is started. All file I/O uses real temporary directories via
tmp_path; no mocks are used for the filesystem.
"""

import json
import os
import stat
import threading
import time
import uuid

import pytest

# ---------------------------------------------------------------------------
# Helpers — isolate module-level state before importing server symbols
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_server_globals(monkeypatch, tmp_path):
    """Redirect the module-level CONFIG_DIR and file paths to tmp_path so tests
    never touch ~/.one, and reset global state between tests."""
    import one.server as srv

    config_dir = str(tmp_path / "one_config")
    api_key_file = os.path.join(config_dir, "api_key")
    server_config_file = os.path.join(config_dir, "server.json")

    monkeypatch.setattr(srv, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(srv, "API_KEY_FILE", api_key_file)
    monkeypatch.setattr(srv, "SERVER_CONFIG_FILE", server_config_file)

    # Reset global server state
    monkeypatch.setattr(srv, "_api_key", None)
    monkeypatch.setattr(srv, "_cors_origins", srv.DEFAULT_CORS_ORIGINS.copy())
    monkeypatch.setattr(srv, "_auth_enabled", True)

    # Reset the module-level rate_limiter singleton so tests don't bleed into
    # each other via the shared object.
    fresh_rl = srv.RateLimiter(
        max_requests=srv.RATE_LIMIT_REQUESTS,
        window=srv.RATE_LIMIT_WINDOW,
    )
    monkeypatch.setattr(srv, "rate_limiter", fresh_rl)

    yield


# ---------------------------------------------------------------------------
# RateLimiter — sliding window per client
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_up_to_limit(self):
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=5, window=60)
        for _ in range(5):
            assert rl.is_allowed("client_a") is True

    def test_blocks_after_limit(self):
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=3, window=60)
        for _ in range(3):
            rl.is_allowed("client_a")
        assert rl.is_allowed("client_a") is False

    def test_different_clients_are_independent(self):
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=1, window=60)
        assert rl.is_allowed("alpha") is True
        assert rl.is_allowed("alpha") is False  # exhausted
        assert rl.is_allowed("beta") is True    # different client — fresh bucket

    def test_window_expiry_allows_new_requests(self):
        """Entries outside the window should be pruned and slots freed."""
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=2, window=1)  # 1-second window
        assert rl.is_allowed("client_x") is True
        assert rl.is_allowed("client_x") is True
        assert rl.is_allowed("client_x") is False  # full

        # Advance time past the window by backdating the stored timestamps
        with rl._lock:
            rl._requests["client_x"] = [t - 2 for t in rl._requests["client_x"]]

        # Old entries are now outside the window; slot should be free again
        assert rl.is_allowed("client_x") is True

    def test_remaining_starts_at_max(self):
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=10, window=60)
        assert rl.remaining("new_client") == 10

    def test_remaining_decrements_with_each_request(self):
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=5, window=60)
        rl.is_allowed("c")
        rl.is_allowed("c")
        assert rl.remaining("c") == 3

    def test_remaining_never_goes_below_zero(self):
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=2, window=60)
        for _ in range(5):
            rl.is_allowed("c")
        assert rl.remaining("c") == 0

    def test_remaining_excludes_expired_entries(self):
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=5, window=1)
        rl.is_allowed("c")
        rl.is_allowed("c")

        # Backdate both timestamps so they fall outside the window
        with rl._lock:
            rl._requests["c"] = [t - 2 for t in rl._requests["c"]]

        assert rl.remaining("c") == 5  # all expired → full quota again

    def test_thread_safety(self):
        """Many concurrent goroutines should not exceed max_requests."""
        from one.server import RateLimiter

        limit = 20
        rl = RateLimiter(max_requests=limit, window=60)
        allowed_count = 0
        lock = threading.Lock()

        def hit():
            nonlocal allowed_count
            result = rl.is_allowed("shared_client")
            if result:
                with lock:
                    allowed_count += 1

        threads = [threading.Thread(target=hit) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert allowed_count == limit

    def test_zero_max_requests_blocks_immediately(self):
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=0, window=60)
        assert rl.is_allowed("c") is False

    def test_large_burst_then_block(self):
        from one.server import RateLimiter

        rl = RateLimiter(max_requests=100, window=60)
        results = [rl.is_allowed("flood") for _ in range(150)]
        assert results[:100] == [True] * 100
        assert results[100:] == [False] * 50


# ---------------------------------------------------------------------------
# _get_api_key — generate on first call, reuse on subsequent calls
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_generates_key_when_file_missing(self, tmp_path):
        import one.server as srv

        key = srv._get_api_key()
        assert isinstance(key, str)
        assert key.startswith("one_")
        # 24 hex chars after the prefix
        suffix = key[len("one_"):]
        assert len(suffix) == 24
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_key_file_is_created(self, tmp_path):
        import one.server as srv

        srv._get_api_key()
        assert os.path.exists(srv.API_KEY_FILE)

    def test_key_file_permissions_are_restrictive(self, tmp_path):
        import one.server as srv

        srv._get_api_key()
        mode = os.stat(srv.API_KEY_FILE).st_mode
        # Owner-read and owner-write only (0o600)
        assert mode & 0o777 == 0o600

    def test_returns_same_key_on_repeated_calls(self, tmp_path):
        import one.server as srv

        key1 = srv._get_api_key()
        key2 = srv._get_api_key()
        assert key1 == key2

    def test_loads_existing_key_from_file(self, tmp_path):
        import one.server as srv

        # Pre-populate the key file
        os.makedirs(srv.CONFIG_DIR, exist_ok=True)
        expected = "one_" + "a" * 24
        with open(srv.API_KEY_FILE, "w") as f:
            f.write(expected + "\n")  # trailing newline stripped via .strip()

        key = srv._get_api_key()
        assert key == expected

    def test_strips_whitespace_from_stored_key(self, tmp_path):
        import one.server as srv

        os.makedirs(srv.CONFIG_DIR, exist_ok=True)
        stored = "  one_aabbccddeeff00112233\n\n  "
        with open(srv.API_KEY_FILE, "w") as f:
            f.write(stored)

        key = srv._get_api_key()
        assert key == stored.strip()

    def test_each_generation_produces_unique_key(self, tmp_path):
        """Two fresh config dirs → two different keys."""
        import one.server as srv

        key1 = srv._get_api_key()

        # Simulate a second fresh environment by deleting the file
        os.remove(srv.API_KEY_FILE)
        key2 = srv._get_api_key()

        assert key1 != key2


# ---------------------------------------------------------------------------
# _load_server_config — reads JSON from SERVER_CONFIG_FILE
# ---------------------------------------------------------------------------


class TestLoadServerConfig:
    def test_returns_empty_dict_when_file_missing(self):
        from one.server import _load_server_config

        cfg = _load_server_config()
        assert cfg == {}

    def test_returns_empty_dict_on_invalid_json(self, tmp_path):
        import one.server as srv

        os.makedirs(srv.CONFIG_DIR, exist_ok=True)
        with open(srv.SERVER_CONFIG_FILE, "w") as f:
            f.write("{ not valid json }")

        cfg = srv._load_server_config()
        assert cfg == {}

    def test_reads_valid_config(self, tmp_path):
        import one.server as srv

        payload = {"cors_origins": ["https://app.example.com"], "auth_enabled": False}
        os.makedirs(srv.CONFIG_DIR, exist_ok=True)
        with open(srv.SERVER_CONFIG_FILE, "w") as f:
            json.dump(payload, f)

        cfg = srv._load_server_config()
        assert cfg == payload

    def test_returns_empty_dict_on_os_error(self, tmp_path, monkeypatch):
        """If the file exists but cannot be read, return {}."""
        import one.server as srv

        os.makedirs(srv.CONFIG_DIR, exist_ok=True)
        with open(srv.SERVER_CONFIG_FILE, "w") as f:
            json.dump({"key": "val"}, f)

        # Make file unreadable
        os.chmod(srv.SERVER_CONFIG_FILE, 0o000)
        try:
            cfg = srv._load_server_config()
            assert cfg == {}
        finally:
            os.chmod(srv.SERVER_CONFIG_FILE, 0o644)


# ---------------------------------------------------------------------------
# _init_server_globals — wires config → module-level state
# ---------------------------------------------------------------------------


class TestInitServerGlobals:
    def test_sets_api_key(self, tmp_path):
        import one.server as srv

        srv._init_server_globals()
        assert srv._api_key is not None
        assert srv._api_key.startswith("one_")

    def test_default_cors_origins_when_no_config(self, tmp_path):
        import one.server as srv

        srv._init_server_globals()
        assert srv._cors_origins == {"*"}

    def test_custom_cors_origins_from_config(self, tmp_path):
        import one.server as srv

        origins = ["https://example.com", "https://other.com"]
        os.makedirs(srv.CONFIG_DIR, exist_ok=True)
        with open(srv.SERVER_CONFIG_FILE, "w") as f:
            json.dump({"cors_origins": origins}, f)

        srv._init_server_globals()
        assert srv._cors_origins == set(origins)

    def test_auth_enabled_default_true(self, tmp_path):
        import one.server as srv

        srv._init_server_globals()
        assert srv._auth_enabled is True

    def test_auth_disabled_via_config(self, tmp_path):
        import one.server as srv

        os.makedirs(srv.CONFIG_DIR, exist_ok=True)
        with open(srv.SERVER_CONFIG_FILE, "w") as f:
            json.dump({"auth_enabled": False}, f)

        srv._init_server_globals()
        assert srv._auth_enabled is False

    def test_rate_limit_config_applied(self, tmp_path):
        import one.server as srv

        os.makedirs(srv.CONFIG_DIR, exist_ok=True)
        with open(srv.SERVER_CONFIG_FILE, "w") as f:
            json.dump({"rate_limit": {"requests": 5, "window": 10}}, f)

        srv._init_server_globals()
        assert srv.rate_limiter.max_requests == 5
        assert srv.rate_limiter.window == 10

    def test_rate_limit_defaults_unchanged_when_no_config(self, tmp_path):
        import one.server as srv
        from one.server import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW

        srv._init_server_globals()
        assert srv.rate_limiter.max_requests == RATE_LIMIT_REQUESTS
        assert srv.rate_limiter.window == RATE_LIMIT_WINDOW

    def test_partial_rate_limit_config_uses_defaults_for_missing_keys(self, tmp_path):
        """Only 'requests' provided — 'window' should fall back to the module default."""
        import one.server as srv
        from one.server import RATE_LIMIT_WINDOW

        os.makedirs(srv.CONFIG_DIR, exist_ok=True)
        with open(srv.SERVER_CONFIG_FILE, "w") as f:
            json.dump({"rate_limit": {"requests": 7}}, f)

        srv._init_server_globals()
        assert srv.rate_limiter.max_requests == 7
        assert srv.rate_limiter.window == RATE_LIMIT_WINDOW

    def test_api_key_persisted_between_inits(self, tmp_path):
        """Second call to _init_server_globals should reuse the same key."""
        import one.server as srv

        srv._init_server_globals()
        first_key = srv._api_key

        srv._init_server_globals()
        assert srv._api_key == first_key


# ---------------------------------------------------------------------------
# CORS helper: _send_cors_headers logic (tested via the module-level state it
# reads, without starting the server)
# ---------------------------------------------------------------------------


class TestCorsOriginMatching:
    """Unit-test the CORS origin-matching logic extracted from _send_cors_headers.

    Rather than driving the full HTTP handler, we replicate the conditional
    that the handler executes so the behaviour can be verified in isolation.
    """

    @staticmethod
    def _cors_decision(cors_origins: set, request_origin: str) -> str | None:
        """Mirror the logic in OneAPIHandler._send_cors_headers."""
        if "*" in cors_origins or request_origin in cors_origins:
            return request_origin if "*" not in cors_origins else "*"
        return None

    def test_wildcard_reflects_star(self):
        result = self._cors_decision({"*"}, "https://any.site.com")
        assert result == "*"

    def test_exact_origin_match(self):
        result = self._cors_decision({"https://app.example.com"}, "https://app.example.com")
        assert result == "https://app.example.com"

    def test_unknown_origin_blocked(self):
        result = self._cors_decision({"https://allowed.com"}, "https://evil.com")
        assert result is None

    def test_multiple_allowed_origins(self):
        origins = {"https://a.com", "https://b.com"}
        assert self._cors_decision(origins, "https://a.com") == "https://a.com"
        assert self._cors_decision(origins, "https://b.com") == "https://b.com"
        assert self._cors_decision(origins, "https://c.com") is None

    def test_wildcard_overrides_exact_restriction(self):
        # If "*" is in the set the handler always emits "*", regardless of other entries
        result = self._cors_decision({"*", "https://specific.com"}, "https://specific.com")
        assert result == "*"


# ---------------------------------------------------------------------------
# SessionManager — does not start proxy processes
# ---------------------------------------------------------------------------


class TestSessionManager:
    """Tests for SessionManager that avoid touching ClaudeProxy or AutoLoop."""

    @pytest.fixture()
    def manager(self, monkeypatch):
        """Return a SessionManager whose Session.start() is a no-op."""
        import one.server as srv

        # Prevent Session.__init__ from calling get_backend() and store.set_project()
        monkeypatch.setattr("one.server.get_backend", lambda: None)
        monkeypatch.setattr("one.store.set_project", lambda project: None)

        return srv.SessionManager()

    def test_create_adds_session(self, manager):
        session = manager.create("proj_a")
        assert session.id in manager.sessions

    def test_get_returns_session(self, manager):
        session = manager.create("proj_b")
        assert manager.get(session.id) is session

    def test_get_missing_returns_none(self, manager):
        assert manager.get("nonexistent-id") is None

    def test_get_by_project_finds_active_session(self, manager):
        session = manager.create("my_project")
        found = manager.get_by_project("my_project")
        assert found is session

    def test_get_by_project_ignores_stopped_sessions(self, manager):
        session = manager.create("proj_c")
        session.status = "stopped"
        found = manager.get_by_project("proj_c")
        assert found is None

    def test_get_by_project_returns_none_when_absent(self, manager):
        assert manager.get_by_project("unknown_project") is None

    def test_list_all_excludes_stopped(self, manager):
        s1 = manager.create("proj_active")
        s2 = manager.create("proj_stopped")
        s2.status = "stopped"
        listed = manager.list_all()
        ids = [s["id"] for s in listed]
        assert s1.id in ids
        assert s2.id not in ids

    def test_list_all_returns_dicts(self, manager):
        manager.create("proj_x")
        listed = manager.list_all()
        assert isinstance(listed, list)
        assert all(isinstance(item, dict) for item in listed)

    def test_stop_all_marks_sessions_stopped(self, manager):
        s1 = manager.create("p1")
        s2 = manager.create("p2")
        manager.stop_all()
        assert s1.status == "stopped"
        assert s2.status == "stopped"

    def test_create_is_thread_safe(self, manager):
        """Concurrent creates should each get a unique session."""
        results = []
        lock = threading.Lock()

        def create_one():
            s = manager.create("concurrent_proj")
            with lock:
                results.append(s.id)

        threads = [threading.Thread(target=create_one) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert len(set(results)) == 20  # all IDs are unique


# ---------------------------------------------------------------------------
# Session.to_dict — state serialisation (no subprocess/proxy involvement)
# ---------------------------------------------------------------------------


class TestSessionToDict:
    @pytest.fixture()
    def session(self, monkeypatch):
        monkeypatch.setattr("one.server.get_backend", lambda: None)
        monkeypatch.setattr("one.store.set_project", lambda project: None)

        from one.server import Session

        return Session("test_project")

    def test_required_keys_present(self, session):
        d = session.to_dict()
        for key in ("id", "project", "model", "status", "turns", "cost", "auto_running"):
            assert key in d, f"Missing key: {key}"

    def test_project_reflects_constructor_arg(self, session):
        assert session.to_dict()["project"] == "test_project"

    def test_auto_running_false_by_default(self, session):
        assert session.to_dict()["auto_running"] is False

    def test_no_auto_progress_key_when_not_running(self, session):
        assert "auto_progress" not in session.to_dict()

    def test_turn_count_and_cost_start_at_zero(self, session):
        d = session.to_dict()
        assert d["turns"] == 0
        assert d["cost"] == 0.0

    def test_id_is_eight_chars(self, session):
        # Session.id is str(uuid.uuid4())[:8]
        assert len(session.id) == 8

    def test_status_is_idle_initially(self, session):
        assert session.to_dict()["status"] == "idle"


# ---------------------------------------------------------------------------
# Session._emit and listener management (no proxy)
# ---------------------------------------------------------------------------


class TestSessionListeners:
    @pytest.fixture()
    def session(self, monkeypatch):
        monkeypatch.setattr("one.server.get_backend", lambda: None)
        monkeypatch.setattr("one.store.set_project", lambda project: None)

        from one.server import Session

        return Session("listener_proj")

    def test_add_and_emit(self, session):
        events = []
        session.add_listener(events.append)
        session._emit("test_event", "hello")
        assert len(events) == 1
        assert events[0]["type"] == "test_event"
        assert events[0]["data"] == "hello"
        assert events[0]["project"] == "listener_proj"

    def test_remove_listener_stops_delivery(self, session):
        events = []
        session.add_listener(events.append)
        session.remove_listener(events.append)
        session._emit("test_event", "ignored")
        assert events == []

    def test_multiple_listeners_all_receive(self, session):
        bucket_a, bucket_b = [], []
        session.add_listener(bucket_a.append)
        session.add_listener(bucket_b.append)
        session._emit("ping", "payload")
        assert len(bucket_a) == 1
        assert len(bucket_b) == 1

    def test_failing_listener_does_not_propagate_exception(self, session):
        def bad_listener(_event):
            raise RuntimeError("listener failure")

        good_events = []
        session.add_listener(bad_listener)
        session.add_listener(good_events.append)

        # Should not raise
        session._emit("safe_event", "data")
        assert len(good_events) == 1
