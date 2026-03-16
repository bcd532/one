"""Ground Truth Population — Self-introspection engine.

Introspects the one codebase and populates the memory graph with verified
facts about schemas, signatures, wiring, and contracts. This gives Claude
an actual grounded worldview instead of hallucinating function names.

Usage:
    from one.ground import populate_ground_truths
    populate_ground_truths(project="one")

Or from CLI:
    one ground
"""

import ast
import inspect
import sqlite3
import os
import importlib
import threading
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import push_memory, DB_DIR, DB_PATH, set_project
from .hdc import encode_tagged
from .entities import extract_entities
from .store import ensure_entity, link_memory_entity


# ── Schema introspection ──────────────────────────────────────────────


def _get_real_schemas() -> list[dict]:
    """Read actual SQLite schemas from the live database."""
    db = os.path.expanduser("~/.one/one.db")
    if not os.path.exists(db):
        return []

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name"
    ).fetchall()

    schemas = []
    for t in tables:
        name = t["name"]
        cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
        col_defs = []
        for c in cols:
            col_def = f"{c['name']} {c['type']}"
            if c["pk"]:
                col_def += " PK"
            if c["dflt_value"] is not None:
                col_def += f" DEFAULT {c['dflt_value']}"
            col_defs.append(col_def)

        schemas.append({
            "table": name,
            "columns": col_defs,
            "column_names": [c["name"] for c in cols],
        })
    conn.close()
    return schemas


# ── Module introspection ──────────────────────────────────────────────


def _get_module_signatures() -> list[dict]:
    """Introspect every module's public classes and functions."""
    base = os.path.dirname(os.path.abspath(__file__))
    sigs = []

    module_names = [
        "app", "auto", "proxy", "cli", "backend", "hdc", "gate",
        "excitation", "rules", "entities", "store", "client",
        "research", "synthesis", "dialectic", "contradictions",
        "analogy", "verification", "experiments", "playbook",
        "swarm", "morgoth", "health", "audit", "sessions",
        "gemma", "server", "graph", "watch", "claudemd", "init",
    ]

    for mod_name in module_names:
        try:
            mod = importlib.import_module(f"one.{mod_name}")
        except Exception:
            continue

        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            obj = getattr(mod, attr_name)

            if inspect.isclass(obj) and obj.__module__ == f"one.{mod_name}":
                # Get class __init__ signature
                try:
                    sig = inspect.signature(obj.__init__)
                    params = []
                    for pname, param in sig.parameters.items():
                        if pname == "self":
                            continue
                        p = pname
                        if param.annotation != inspect.Parameter.empty:
                            p += f": {param.annotation.__name__}" if hasattr(param.annotation, "__name__") else ""
                        if param.default != inspect.Parameter.empty:
                            p += f" = {param.default!r}"
                        params.append(p)

                    # Get public methods
                    methods = []
                    for mname in dir(obj):
                        if mname.startswith("_"):
                            continue
                        mobj = getattr(obj, mname, None)
                        if callable(mobj) and hasattr(mobj, "__func__"):
                            try:
                                msig = inspect.signature(mobj)
                                mparams = []
                                for mp, mv in msig.parameters.items():
                                    if mp == "self":
                                        continue
                                    s = mp
                                    if mv.default != inspect.Parameter.empty:
                                        s += f"={mv.default!r}"
                                    mparams.append(s)
                                methods.append(f".{mname}({', '.join(mparams)})")
                            except (ValueError, TypeError):
                                methods.append(f".{mname}()")

                    sigs.append({
                        "module": mod_name,
                        "type": "class",
                        "name": attr_name,
                        "params": ", ".join(params),
                        "methods": methods,
                    })
                except (ValueError, TypeError):
                    pass

            elif inspect.isfunction(obj) and obj.__module__ == f"one.{mod_name}":
                try:
                    sig = inspect.signature(obj)
                    params = []
                    for pname, param in sig.parameters.items():
                        p = pname
                        if param.default != inspect.Parameter.empty:
                            p += f"={param.default!r}"
                        params.append(p)

                    ret = ""
                    if sig.return_annotation != inspect.Signature.empty:
                        ret = f" -> {sig.return_annotation}"

                    sigs.append({
                        "module": mod_name,
                        "type": "function",
                        "name": attr_name,
                        "params": ", ".join(params),
                        "return": ret,
                    })
                except (ValueError, TypeError):
                    pass

    return sigs


# ── Ground truth memory creation ──────────────────────────────────────


def _store_ground_truth(text: str, label: str, project: str, confidence: float = 0.95) -> str:
    """Store a verified ground truth memory with high confidence."""
    vec = encode_tagged(text, role="system")
    mid = push_memory(
        raw_text=text,
        source="ground_truth",
        tm_label=label,
        regime_tag="verified",
        aif_confidence=confidence,
        hdc_vector=vec.tolist(),
        project=project,
    )

    # Extract and link entities
    ents = extract_entities(text, source="ground_truth")
    for ent in ents:
        eid = ensure_entity(ent)
        if mid and eid:
            link_memory_entity(mid, eid)

    return mid


def populate_ground_truths(
    project: str = "one",
    on_log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Populate the knowledge graph with verified ground truths about the codebase.

    Returns stats about what was stored.
    """
    log = on_log or (lambda m: None)
    set_project(project)
    stats = {"schemas": 0, "signatures": 0, "contracts": 0, "traps": 0}

    # ── 1. Schema ground truths ──────────────────────────────────

    log("introspecting database schemas...")
    schemas = _get_real_schemas()

    for s in schemas:
        cols_text = ", ".join(s["columns"])
        text = (
            f"SCHEMA GROUND TRUTH: Table '{s['table']}' has columns: {cols_text}. "
            f"Column names are: {', '.join(s['column_names'])}. "
            f"This is the verified schema from the live database."
        )
        _store_ground_truth(text, "schema_ground_truth", project)
        stats["schemas"] += 1

    # Critical schema corrections that have caused bugs before
    critical_schemas = [
        "CRITICAL: The 'entities' table has column 'type', NOT 'entity_type'. The 'entities' table has NO 'project' column. Queries must use 'type' and must not filter by project.",
        "CRITICAL: The rules table is named 'rule_nodes', NOT 'rules'. All queries must use 'rule_nodes'.",
        "CRITICAL: The 'memories' table has column 'timestamp' for time, NOT 'created'. There is no 'created' column on memories.",
        "CRITICAL: store.recall('') with an empty string returns EMPTY because the zero vector has zero norm. Use store.get_recent() to get memories without vector search.",
        "CRITICAL: CREATE TABLE IF NOT EXISTS does NOT add new columns to existing tables. New columns require ALTER TABLE ADD COLUMN with a try/except sqlite3.OperationalError guard.",
    ]
    for text in critical_schemas:
        _store_ground_truth(text, "schema_critical", project, confidence=0.99)
        stats["traps"] += 1

    # ── 2. Signature ground truths ────────────────────────────────

    log("introspecting module signatures...")
    sigs = _get_module_signatures()

    for s in sigs:
        if s["type"] == "class":
            methods_text = "; ".join(s["methods"][:10]) if s["methods"] else "none"
            text = (
                f"SIGNATURE GROUND TRUTH: Class {s['name']} in one.{s['module']} — "
                f"constructor takes ({s['params']}). "
                f"Public methods: {methods_text}. "
                f"Verified by runtime introspection."
            )
        else:
            text = (
                f"SIGNATURE GROUND TRUTH: Function {s['name']} in one.{s['module']} — "
                f"signature: {s['name']}({s['params']}){s.get('return', '')}. "
                f"Verified by runtime introspection."
            )
        _store_ground_truth(text, "signature_ground_truth", project, confidence=0.95)
        stats["signatures"] += 1

    # ── 3. Integration contracts ──────────────────────────────────

    log("storing integration contracts...")
    contracts = [
        "CONTRACT: app.py _show_health() calls HealthDashboard(self.project).format_report(). HealthDashboard is in health.py. format_report takes optional report dict, defaults to generating fresh via full_report().",
        "CONTRACT: app.py _run_audit(flags) calls AuditEngine(self.project).run_full_audit(auto_fix=bool). AuditEngine is in audit.py. run_full_audit returns dict with keys: project, timestamp, total_issues, garbage, duplicates, continuous_check, health. The 'health' key has values: 'healthy', 'needs_attention', 'critical'. There is NO 'overall_score' key.",
        "CONTRACT: app.py _start_swarm(goal) calls SwarmCoordinator(goal=goal, project=self.project).start(). SwarmCoordinator is in swarm.py. start() takes optional num_agents=6, strategy='deep'.",
        "CONTRACT: app.py _start_morgoth(goal) calls MorgothMode(goal=goal, project=self.project).start(). MorgothMode is in morgoth.py. start() takes NO arguments.",
        "CONTRACT: app.py _start_auto(goal) calls AutoLoop(proxy=self.proxy, on_status=fn, on_log=fn, on_complete=fn, project=self.project).start(goal). AutoLoop is in auto.py.",
        "CONTRACT: app.py _start_watch(dir) must use self.backend, NOT get_backend(). The watch module's start_watch takes (directory, project, backend).",
        "CONTRACT: app.py _start_research(topic) calls research.start_research(topic=topic, project=self.project, turn_budget=10, on_log=fn). Returns dict with keys: status, findings, gaps_remaining, turns_used.",
        "CONTRACT: Backend.recall(query, n) does vector search. If query is empty string, returns []. Use store.get_recent(n, project) for non-vector retrieval of recent memories.",
        "CONTRACT: All Foundry SDK imports (orion_push_sdk, foundry_sdk_runtime) MUST be lazy — inside function bodies, never at module top level. The system must work with zero external services.",
        "CONTRACT: After any edit to one/*.py files, run: cp ~/projects/open-sourced-projects/one/one/*.py ~/projects/orion/one/ to sync to live install.",
    ]
    for text in contracts:
        _store_ground_truth(text, "contract_ground_truth", project, confidence=0.98)
        stats["contracts"] += 1

    # ── 4. Testing ground truths ──────────────────────────────────

    log("storing testing patterns...")
    testing = [
        "TESTING GROUND TRUTH: To verify a command works, call the actual function chain. Example: from one.health import HealthDashboard; d = HealthDashboard('project'); r = d.format_report(). If this crashes, the /health command is broken.",
        "TESTING GROUND TRUTH: Running 'one --help' is NOT testing. Testing means calling every function in the chain from TUI command to database query.",
        "TESTING GROUND TRUTH: When changing a schema, ALL test files that create their own test schemas must be updated to match. Test INSERT statements must use exact column names from the real table.",
        "TESTING GROUND TRUTH: Every module must import cleanly. Verify with: python -c 'from one.MODULE import CLASS'. If this fails, the module is broken.",
        "TESTING GROUND TRUTH: The test suite has 883 tests. ALL must pass before commit. Run: .venv/bin/python -m pytest tests/ -v --tb=short",
        "TESTING GROUND TRUTH: When fixing recall('') calls, the corresponding test mocks must change from mocking 'recall' to mocking 'get_recent'. The get_recent signature is (n=50, project=None), NOT (query, n=10, project=None).",
    ]
    for text in testing:
        _store_ground_truth(text, "testing_ground_truth", project, confidence=0.97)
        stats["traps"] += 1

    total = sum(stats.values())
    log(f"ground truth population complete: {total} facts stored ({stats})")
    return stats
