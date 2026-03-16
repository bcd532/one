# one — Claude Operating Instructions

**READ LOCK.md FIRST.** It has exact schemas, exact signatures, known traps. If you write a SQL query or call a function, verify it against LOCK.md. No exceptions.

## What this is

Persistent memory and autonomous research platform for Claude CLI. 35 Python files, 14,500+ lines. SQLite default backend, Palantir AIP Foundry optional.

## Project Structure

```
one/
├── app.py          — Textual TUI (1826 lines)
├── auto.py         — Autonomous loop, Claude drives
├── proxy.py        — Bidirectional stream-json Claude wrapper
├── cli.py          — Entry: one, one remote, one server
├── backend.py      — Storage abstraction (SQLite/Foundry)
├── hdc.py          — HDC encoding (4096-dim trigrams+words+bigrams)
├── gate.py         — AIF gating (noise, novelty, temporal, excitation)
├── excitation.py   — Breakthrough vs rage detection
├── rules.py        — Contextual rule tree
├── entities.py     — Entity extraction (10+ types + relationships)
├── store.py        — SQLite memory store with vector search
├── client.py       — Foundry client with token lifecycle
├── research.py     — Autonomous research with gap analysis
├── synthesis.py    — Cross-domain insight DAG
├── dialectic.py    — Thesis/antithesis/synthesis engine
├── contradictions.py — Active contradiction mining
├── analogy.py      — Structural template matching
├── verification.py — Self-verifying knowledge
├── experiments.py  — Executable hypothesis testing
├── playbook.py     — Reusable strategy playbooks
├── swarm.py        — Multi-agent coordinated research
├── morgoth.py      — Research+build+verify+iterate mode
├── swarm_tui.py    — Swarm dashboard view
├── health.py       — Knowledge graph health (HealthDashboard class)
├── audit.py        — Quality enforcement (AuditEngine class)
├── sessions.py     — Session persistence
├── gemma.py        — Local LLM condensation (Gemma 3 4B)
├── server.py       — REST API (:4111)
├── remote.py       — Telegram/Discord/Slack setup
├── graph.py        — d3.js knowledge graph visualization
├── watch.py        — File change monitoring
├── claudemd.py     — Generate CLAUDE.md from rules
├── retrieve.py     — Foundry vector recall
├── init.py         — Environment detection
└── __init__.py
```

## Two copies — ALWAYS SYNC

- `~/projects/open-sourced-projects/one/` — GitHub repo
- `~/projects/orion/` — LIVE install (`one` command runs from here)

After ANY edit: `cp ~/projects/open-sourced-projects/one/one/*.py ~/projects/orion/one/`

## Build & Test

```bash
cd ~/projects/open-sourced-projects/one
.venv/bin/pip install -e .
.venv/bin/python -m pytest tests/
one --help
```

## Coding Rules

- ALL Foundry SDK imports must be LAZY (inside functions, not top-level)
- System works with ZERO external services (no ollama, no Foundry = SQLite only)
- No Co-Authored-By in commits
- No personal info pushed to Foundry
- Comments must be professional OSS quality
- MIROBEAR.md and BULLETPROOF.md stay in .gitignore
- Test THROUGH THE APP, not just unit tests. Run the actual command.
- Verify imports resolve: `python -c "from one.module import Thing"`

## Key Classes & Functions

- `HealthDashboard(project).format_report()` — not get_health_report()
- `AuditEngine(project).run_full_audit(auto_fix=bool)` — not run_full_audit()
- `SwarmCoordinator(goal, project).start()` — multi-agent
- `MorgothMode(goal, project).start()` — god builder
- `AutoLoop(proxy, ...).start(goal)` — autonomous loop
- `get_backend(foundry=None)` → SqliteBackend or FoundryBackend
- `token_status()` → dict with remaining_seconds, expired, warning, ok

## SSH

GitHub uses `~/.ssh/fartfart` key through port 443 (port 22 blocked).
Dedi server at 100.85.86.13 via Tailscale SSH.

## Foundry

- Token: `~/.one/token` (180 days, good until Sept 2026)
- Config: `~/.one/config` (has host= line)
- SDK: `orion_push_sdk` (private, only in orion venv)
- Objects: MemoryEntry (with hdc_vector 4096-dim) + Entity, linked via hasMemoryEntries
