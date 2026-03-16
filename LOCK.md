# LOCK.md — Ground Truth for `one`

**This file is the single source of truth. If LOCK.md disagrees with the code, the CODE is wrong.**

Every Claude Code session touching this repo MUST read this file. Every edit to a `.py` file MUST be validated against this file's schemas, signatures, and wiring. No exceptions. No "probably fine." No skipping.

---

## SQLite Schema — EXACT columns, EXACT types

The database is `~/.one/one.db`. These are the real columns. Not what you think they are. Not what some old version had. THESE.

### memories
```
id TEXT PK, raw_text TEXT, source TEXT, timestamp TEXT,
project TEXT DEFAULT 'global', hdc_vector BLOB,
tm_label TEXT DEFAULT 'unclassified', regime_tag TEXT DEFAULT 'default',
aif_confidence REAL DEFAULT 0.0
```
- Time column is `timestamp`, NOT `created`
- `id` is a UUID string, NOT integer

### entities
```
id INTEGER PK, name TEXT, type TEXT,
observation_count INTEGER DEFAULT 1,
first_seen TEXT, last_seen TEXT
```
- Type column is `type`, NOT `entity_type`
- There is NO `project` column on entities
- Queries MUST use `type`, never `entity_type`

### rule_nodes
```
id INTEGER PK, project TEXT, parent_id INTEGER, rule_text TEXT,
activation_keywords TEXT DEFAULT '*', hdc_vector BLOB,
confidence REAL DEFAULT 0.5, source_count INTEGER DEFAULT 1,
active INTEGER DEFAULT 1, superseded_by INTEGER,
created TEXT, updated TEXT
```
- Table is `rule_nodes`, NOT `rules`

### memory_entities (junction)
```
memory_id TEXT PK, entity_id INTEGER PK
```

### sessions
```
id TEXT PK, project TEXT, model TEXT, start_time TEXT,
end_time TEXT, turn_count INTEGER DEFAULT 0,
total_cost REAL DEFAULT 0.0
```

### session_messages
```
id INTEGER PK, session_id TEXT, role TEXT, content TEXT,
timestamp TEXT, turn_number INTEGER
```

### syntheses
```
id INTEGER PK, project TEXT, entity_a TEXT, entity_b TEXT,
hypothesis TEXT, confidence REAL, novelty_score REAL DEFAULT 0.5,
tested INTEGER DEFAULT 0, test_result TEXT, parent_id INTEGER,
depth INTEGER DEFAULT 0, created TEXT
```

### contradictions
```
id INTEGER PK, project TEXT, finding_a TEXT, finding_b TEXT,
finding_a_source TEXT, finding_b_source TEXT,
severity TEXT DEFAULT 'moderate', status TEXT DEFAULT 'active',
resolution TEXT, resolution_type TEXT,
confidence_a REAL DEFAULT 0.5, confidence_b REAL DEFAULT 0.5,
created TEXT, resolved_at TEXT
```

### research_topics
```
id INTEGER PK, project TEXT, topic TEXT,
status TEXT DEFAULT 'active', turns_used INTEGER DEFAULT 0,
turn_budget INTEGER DEFAULT 10, findings_count INTEGER DEFAULT 0,
gaps_count INTEGER DEFAULT 0, synthesis_depth INTEGER DEFAULT 0,
created TEXT, updated TEXT, depth_level INTEGER DEFAULT 0
```

### research_findings
```
id INTEGER PK, topic_id INTEGER, content TEXT,
finding_type TEXT DEFAULT 'finding', source_query TEXT,
confidence REAL DEFAULT 0.5, created TEXT,
source_quality REAL DEFAULT 0.5, quantitative_data TEXT,
published_date TEXT, contradicts_finding_id INTEGER,
depth_level INTEGER DEFAULT 0
```

### research_gaps
```
id INTEGER PK, topic_id INTEGER, question TEXT,
status TEXT DEFAULT 'open', resolved_by INTEGER,
created TEXT, priority REAL DEFAULT 0.5
```

### research_citations
```
id INTEGER PK, finding_a INTEGER, finding_b INTEGER,
relation TEXT DEFAULT 'supports', created TEXT
```

### playbooks
```
id INTEGER PK, project TEXT, task_description TEXT,
category TEXT DEFAULT 'general', key_decisions TEXT,
reusable_patterns TEXT, pitfalls TEXT, full_playbook TEXT,
confidence REAL DEFAULT 0.9, times_recalled INTEGER DEFAULT 0,
created TEXT, updated TEXT
```

### verification_log
```
id INTEGER PK, project TEXT, memory_id TEXT,
finding_id INTEGER, previous_confidence REAL,
new_confidence REAL, verification_type TEXT,
evidence TEXT, created TEXT
```

### knowledge_frontier
```
id INTEGER PK, project TEXT, goal TEXT, question TEXT,
information_value REAL DEFAULT 0.5,
status TEXT DEFAULT 'unexplored',
category TEXT DEFAULT 'general', created TEXT, explored_at TEXT
```

---

## Class Signatures — EXACT constructors and methods

### AutoLoop (auto.py)
```python
AutoLoop(
    proxy: ClaudeProxy,
    on_status: Callable[[str], None] = None,
    on_log: Callable[[str, str], None] = None,
    on_complete: Callable[[str], None] = None,
    project: str = "global",
)
.start(goal: str) -> None
.stop() -> None
.on_response_complete(text: str) -> None
```

### SwarmCoordinator (swarm.py)
```python
SwarmCoordinator(
    goal: str,
    project: str,
    model: str = "sonnet",
    conductor_model: str = "opus",
    cwd: str = ".",
    on_finding: Callable = None,
    on_status: Callable = None,
    on_eureka: Callable = None,
    on_log: Callable = None,
)
.start(num_agents: int = 6, strategy: str = "deep") -> None
.stop() -> dict
.inject(text: str) -> None
.status() -> dict
.get_findings() -> list[Finding]
```

### MorgothMode (morgoth.py)
```python
MorgothMode(
    goal: str,
    project: str,
    proxy_factory: Callable = None,
    on_log: Callable = None,
)
.start() -> None  # NO arguments
.stop() -> None
.status() -> dict
```

### HealthDashboard (health.py)
```python
HealthDashboard(project: str, on_log: Callable = None)
.format_report(report: dict = None) -> str
.full_report() -> dict
.volume() -> dict
.entities() -> dict
.intelligence() -> dict
.quality() -> dict
.warnings() -> list[dict[str, str]]
```

### AuditEngine (audit.py)
```python
AuditEngine(project: str, on_log: Callable = None)
.run_full_audit(auto_fix: bool = False) -> dict
.format_report(report: dict = None) -> str
.find_garbage(score_threshold: float = 3.0) -> dict
.find_duplicates() -> dict
.score_memory(memory: dict) -> dict
.score_entity(entity: dict) -> dict
```
- `run_full_audit` returns: `{project, timestamp, total_issues, garbage, duplicates, continuous_check, health, fixed?}`
- `health` key values: "healthy", "needs_attention", "critical"
- There is NO `overall_score` key

### ClaudeProxy (proxy.py)
```python
ClaudeProxy(
    model: str = None,
    cwd: str = ".",
    permission_mode: str = "default",
    system_prompt: str = None,
    append_system_prompt: str = None,
    allowed_tools: list[str] = None,
    disallowed_tools: list[str] = None,
    resume: str = None,
    session: str = None,
    continue_: bool = False,
)
.start() -> None
.stop() -> None
.send(text: str) -> None
.on_event(callback: Callable[[dict], None]) -> None
.alive -> bool
.model -> str | None
.cwd -> str
```

### Backend (backend.py)
```python
# Protocol
Backend.push_memory(raw_text, source, tm_label, regime_tag, aif_confidence, hdc_vector, project) -> str
Backend.recall(query: str, n: int) -> list[dict]
Backend.recall_context(query: str, n: int, max_chars: int, use_gemma: bool) -> str
Backend.ensure_entity(entity: dict) -> None
Backend.stats() -> dict

get_backend(foundry=None) -> SqliteBackend | FoundryBackend
```

### Store functions (store.py)
```python
push_memory(raw_text, source, tm_label="unclassified", regime_tag="default",
            aif_confidence=0.0, hdc_vector=None, project=None) -> str
recall(query: str, n: int = 10, project: str = None) -> list[dict]
get_recent(n: int = 50, project: str = None) -> list[dict]
get_entities(entity_type: str = None, limit: int = 50) -> list[dict]
get_related_entities(entity_name: str, limit: int = 10) -> list[dict]
get_memory_by_time(since=None, until=None, source=None, limit=50) -> list[dict]
ensure_entity(entity: dict) -> int
stats() -> dict
```
- `recall("")` returns EMPTY (zero vector). Use `get_recent()` for "get all memories".
- `get_entities` returns dicts with `name`, `type`, `observation_count` — NOT `entity_type`

### Research functions (research.py)
```python
start_research(topic, project, turn_budget=10, max_depth=3,
               adversarial=True, on_log=None) -> dict
research_frontier(project: str) -> dict
```

### Other functions
```python
# rules.py
add_rule(project, text, activation_keywords="*", confidence=0.9, source_count=1) -> int
get_all_rules(project: str) -> list[dict]
get_active_rules(project, current_text, recent_files, recent_tools) -> list[dict]
format_rules_for_injection(rules: list, project: str) -> str

# synthesis.py
run_deep_synthesis(project, depth=3) -> list[dict]
get_syntheses_count(project: str) -> int

# sessions.py
create_session(project: str, model: str) -> str
end_session(session_id: str, total_cost: float) -> None
add_message(session_id, role, content, turn_number) -> int
list_sessions(project=None, limit=20) -> list[dict]
get_session_messages(session_id, limit=100) -> list[dict]
export_session_markdown(session_id: str) -> str

# playbook.py
list_playbooks(project: str) -> list[dict]
create_playbook(project, task_desc, decisions, patterns, pitfalls, full) -> int
recall_playbook_context(project, goal, n=3) -> str

# watch.py
start_watch(directory: str, project: str, backend) -> str
stop_watch() -> str

# claudemd.py
generate_claude_md(project: str) -> str

# client.py
get_client() -> FoundryClient
token_status() -> dict  # {remaining_seconds, expired, warning, ok, formatted}
```

---

## Command Wiring — /command → handler → function

| Command | Handler | Calls | Notes |
|---------|---------|-------|-------|
| `/quit` `/exit` `/q` | `self.exit()` | direct | |
| `/clear` | `action_clear_chat()` | direct | |
| `/cost` | inline | `self._total_cost` | |
| `/recall` | `_force_recall()` | `backend.recall()` | |
| `/rules` | `_show_rules()` | `rules.get_all_rules(project)` | |
| `/rule <text>` | `_add_manual_rule(text)` | `rules.add_rule(project, text)` | |
| `/stats` | `_show_stats()` | `backend.stats()` | |
| `/entities` | `action_show_entities()` | `store.get_entities()`, `store.get_related_entities()` | |
| `/search <q>` | `_search_memories(q)` | `backend.recall(q)` | |
| `/undo` | `action_show_undo()` | `git diff --stat HEAD~1` | |
| `/context` | `_show_last_context()` | `self._last_injected_context` | |
| `/think` | `action_toggle_thinking()` | `self._show_thinking` toggle | |
| `/forget <text>` | `_forget_rule(text)` | SQL on `rule_nodes` | |
| `/help` | `_show_help()` | inline display | |
| `/sessions` | `_show_sessions()` | `sessions.list_sessions()` | |
| `/session <id>` | `_show_session_messages(id)` | `sessions.get_session_messages()` | |
| `/export` | `action_export_session()` | `sessions.export_session_markdown()` | |
| `/auto <goal>` | `_start_auto(goal)` | `AutoLoop(proxy, ...).start(goal)` | |
| `/stop` | `_stop_auto()` | `auto_loop.stop()` | Only stops auto, not swarm/morgoth |
| `/watch [dir]` | `_start_watch(dir)` | `watch.start_watch(dir, project, self.backend)` | |
| `/unwatch` | `_stop_watch()` | `watch.stop_watch()` | |
| `/generate` | `_generate_claude_md()` | `claudemd.generate_claude_md(project)` | |
| `/synthesize` | `_run_synthesis()` | `synthesis.run_deep_synthesis(project)` | threaded |
| `/research <topic>` | `_start_research(topic)` | `research.start_research(topic, project)` | threaded |
| `/playbooks` | `_show_playbooks()` | `playbook.list_playbooks(project)` | |
| `/frontier` | `_show_frontier()` | `research.research_frontier(project)` | |
| `/swarm <goal>` | `_start_swarm(goal)` | `SwarmCoordinator(goal, project).start()` | |
| `/morgoth <goal>` | `_start_morgoth(goal)` | `MorgothMode(goal, project).start()` | |
| `/health` | `_show_health()` | `HealthDashboard(project).format_report()` | |
| `/audit [--fix]` | `_run_audit(flags)` | `AuditEngine(project).run_full_audit(auto_fix=bool)` | threaded |
| `/focus <agent>` | `_focus_agent(id)` | stub — not implemented | |
| `/inject <text>` | `_inject_all(text)` | stub — not implemented | |

---

## Known Traps — DO NOT repeat these

### 1. `recall("")` returns empty
An empty string produces a zero HDC vector. `store.recall()` returns `[]` when norm < 1e-10. Use `store.get_recent()` for "get all recent memories."

### 2. `entities` table has NO `project` column
The column is `type`, NOT `entity_type`. Entities are global, not per-project. Every past bug that queried `entity_type` or `WHERE project = ?` on entities was wrong.

### 3. Rules table is `rule_nodes`, not `rules`
Every query to `FROM rules` is wrong.

### 4. Memories time column is `timestamp`, not `created`
The `created` column does not exist on the memories table.

### 5. Foundry imports MUST be lazy
All `orion_push_sdk`, `foundry_sdk_runtime` imports go inside function bodies, never at module top level. The system must work with ZERO external services.

### 6. Schema evolution — CREATE TABLE IF NOT EXISTS doesn't add columns
If a table already exists, `CREATE TABLE IF NOT EXISTS` with new columns is a no-op. New columns MUST use `ALTER TABLE ADD COLUMN` with a try/except guard.

### 7. `push_memory` bypass
8+ modules call `store.push_memory` directly instead of through `Backend`. This means Foundry never receives those memories. Known debt.

### 8. Swarm/Morgoth objects get garbage collected
`_start_swarm` and `_start_morgoth` create objects as local variables. They have no persistent reference. `/stop` only stops auto, not swarm/morgoth.

### 9. Test schemas must match real schemas
Test files create their own SQLite schemas. When the real schema changes, tests MUST be updated to match. Column names in test INSERT statements must exactly match the real table.

---

## Verification Checklist — BEFORE any commit

1. `python -m pytest tests/ -v --tb=short` — ALL 883 tests pass
2. `python -c "from one.MODULE import CLASS"` — for every changed module
3. Cross-reference every SQL query against the schemas above
4. Cross-reference every function call against the signatures above
5. If adding a column: add `ALTER TABLE` migration, not just to CREATE TABLE
6. If Foundry SDK: import is inside a function, not top-level
7. Test THROUGH THE APP: `one --help` works, launch and try the command
8. Sync to live: `cp ~/projects/open-sourced-projects/one/one/*.py ~/projects/orion/one/`

---

## File → Responsibility Map

| File | Does | Depends On |
|------|------|------------|
| `app.py` | TUI, command dispatch, stream rendering | proxy, backend, store, rules, sessions, health, audit, research, synthesis, playbook, swarm, morgoth, watch, claudemd, entities, hdc, gemma |
| `auto.py` | Autonomous Claude loop | proxy, backend, rules, playbook, store, hdc, gemma |
| `proxy.py` | Claude subprocess wrapper | (none — standalone) |
| `cli.py` | Entry point | proxy, app, init, server, remote |
| `backend.py` | Storage abstraction | store, retrieve, entities, rules, hdc, gemma |
| `store.py` | SQLite CRUD, vector search | hdc |
| `hdc.py` | HDC encoding (4096-dim) | numpy |
| `gate.py` | AIF quality gating | hdc, excitation |
| `excitation.py` | Breakthrough/rage detection | (standalone) |
| `rules.py` | Contextual rule tree | store, hdc |
| `entities.py` | Entity extraction | (standalone for extraction; foundry SDK lazy for push) |
| `research.py` | Deep research with gaps | store, hdc, gemma, synthesis, entities |
| `synthesis.py` | Cross-domain insight DAG | store, hdc, gemma |
| `dialectic.py` | Thesis/antithesis/synthesis | store, gemma |
| `contradictions.py` | Contradiction mining | store, hdc, gemma |
| `analogy.py` | Structural template matching | store, hdc, gemma |
| `verification.py` | Self-verifying knowledge | store, hdc, gemma |
| `experiments.py` | Hypothesis testing | store, gemma |
| `playbook.py` | Strategy playbooks | store, hdc, gemma |
| `swarm.py` | Multi-agent coordination | proxy, store, hdc, excitation |
| `morgoth.py` | God mode research+build | swarm, dialectic, analogy, contradictions, verification, experiments, health, store |
| `health.py` | Knowledge graph health | store (via own _get_conn) |
| `audit.py` | Quality enforcement | store (via own _get_conn), gemma |
| `sessions.py` | Session persistence | (standalone SQLite) |
| `gemma.py` | Local LLM (Ollama) | subprocess |
| `server.py` | REST API | store, sessions, research, synthesis, graph, claudemd |
| `remote.py` | Telegram/Discord/Slack | server API |
| `graph.py` | d3.js visualization | store |
| `watch.py` | File change monitoring | backend, entities, hdc |
| `claudemd.py` | Generate CLAUDE.md | rules, store |
| `retrieve.py` | Foundry vector recall | hdc, foundry SDK (lazy) |
| `client.py` | Foundry client lifecycle | foundry SDK (lazy) |
| `init.py` | Environment detection | (standalone) |
