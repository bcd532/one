# One — Autonomous Project Execution Engine

> You describe what you want built. One figures it out, researches it, specs it,
> and builds it — talking to Claude Code so you don't have to.
> By the end of a full day, your project is complete.

---

## What One Is

One is an autonomous project execution engine for solo developers. It sits between
you and Claude Code, acting as a persistent, topology-aware orchestrator that drives
projects from idea to completion.

You talk to One through Telegram. One talks to Claude Code for you — one agent or many.
It tracks every file, function, and dependency as a live knowledge graph. It detects
when work is incomplete or broken. It keeps driving until the project is actually done —
across as many sessions as it takes.

**One is NOT:**
- Not an IDE or TUI (that's ORION)
- Not a prompt manager or CLI wrapper
- Not tied to Claude Code forever — Claude Code is the first integration, the topology engine is tool-agnostic

---

## The Four Phases

One operates as a four-phase machine. Every project goes through all four phases.
The first three phases are where the real value lives — a perfect spec with a
mediocre executor beats a bad spec with a perfect executor every time.

### How One conducts conversations (Phases 1, 2, 3)

One uses Claude Code as its LLM for all phases — not a separate API integration.
For conversational phases, One spawns a single Claude Code agent with a
phase-specific system prompt:

- **Phase 1 agent:** System prompt includes the knowledge graph state, the list
  of unknowns, and instructions to ask one clarifying question at a time.
  One parses the agent's output, updates the knowledge graph, recalculates
  unknowns, and feeds the updated state back for the next turn.
- **Phase 3 agent:** System prompt includes the full knowledge graph (with
  research findings), and instructions to present design sections one at a time
  for user approval.

One is the orchestrator. Claude Code is the brain. One decides *what* to ask
and *when* to move on. Claude Code decides *how* to phrase it and *what details*
to explore within One's directive.

### Phase transitions

Phases are not strictly linear. Here's the state machine:

```
Phase 1 (Understand) ──▶ Phase 2 (Research) ──▶ Phase 3 (Spec) ──▶ Phase 4 (Build)
         ▲                       │                      │                  │
         └───────────────────────┘                      │                  │
                  (need more context)                   │                  │
                                        ◀───────────────┘                  │
                                    (need more research)                   │
                                        ◀──────────────────────────────────┘
                                    (unexpected finding during build)
```

**Transition triggers:**
- **1→2:** One proposes transition when unknowns list contains only items that
  require external research (not user input). User confirms via Telegram button.
- **2→3:** One proposes when research covers all identified unknowns. User reviews
  findings and confirms. User can say "dig deeper on X" to stay in Phase 2.
- **3→4:** All spec sections approved by user. Task plan generated and approved.
- **4→2 (backward):** Agent hits unexpected technical blocker. One auto-pauses,
  spawns research agent, reports findings to user before resuming.
- **Any→1 (backward):** User realizes scope needs changing. Says "wait, I want
  to change the scope" in Telegram. One returns to Phase 1 with existing graph.

### Phase 1: Understand (Conversation)

You tell One what you want. One asks clarifying questions, one at a time, until it
truly understands the scope, the users, the constraints, and the success criteria.

One maintains a knowledge graph of what's been established. After each response,
it updates the graph and determines what's still unknown. It keeps asking until
the unknowns list is empty.

**Unknowns One resolves in this phase:**
- Purpose — what problem does this solve?
- Users — who uses it and how?
- Scope — what's in, what's out?
- Constraints — tech stack, budget, timeline, deployment?
- Success criteria — how do you know it's done?

### Phase 2: Research (Agent-Driven)

One spawns a lightweight Claude Code agent to eliminate unknowns and challenge
assumptions. Research is context-dependent:

| You're building... | One researches... |
|---|---|
| A startup product | Competitors, market gaps, weaknesses, your edge, pricing |
| An open source tool | Existing projects, what they do well, what they miss |
| A technical system | Architecture patterns, pitfalls, scaling gotchas, library comparisons |
| An integration | API docs, rate limits, auth flows, known bugs, workarounds |
| Something novel | Academic papers, related approaches, why similar attempts failed |

**Anti-thesis is mandatory.** For every major decision, the Research Agent actively
looks for reasons it might be wrong:
- "You chose SQLite. Here's when SQLite breaks down. Does that apply?"
- "You want real-time sync. Here are 5 approaches with trade-offs at your scale."
- "3 similar projects tried this approach. 2 succeeded because X. 1 failed because Y."

Research findings become nodes in the knowledge graph, linked to the decisions they
inform. Nothing is decided without evidence.

Research results come back to the Telegram conversation. You review, steer, ask for
more depth. One keeps researching until there are no unknowns left.

**Research can resume during Phase 4.** If an agent hits something unexpected during
implementation, One can pause, research alternatives, update the graph, and adjust.

### Phase 3: Spec (Conversation)

One walks through the design with you section by section. Every component, every
edge case, every integration point. You approve each section before moving on.

By the end, there's a thorough spec document and a task plan with dependencies
mapped out. The knowledge graph is dense with goals, decisions, components, and
their relationships.

### Phase 4: Build (Autonomous)

One asks how you want to run:
- **Full autonomy** — One runs agents until done, notifies on completion
- **Checkpoints** — One pauses at milestones for your review
- **Multiple agents** — parallel execution on independent tasks

The topology engine tracks everything. If an agent says it's done but the wiring
is broken, One catches it and sends another agent to fix it.

---

## Architecture

```
┌─ Your Phone ─┐     ┌─ Your Machine ──────────────────────────┐
│               │     │                                          │
│  Telegram     │◄───►│  One (background service)                │
│  Bot Chat     │     │    │                                      │
│               │     │    ├── Interface Layer                    │
└───────────────┘     │    │   └── Telegram Bot API               │
                      │    │                                      │
                      │    ├── Brain Layer                        │
                      │    │   ├── Knowledge Graph                │
                      │    │   ├── Topology Engine                │
                      │    │   ├── Session Tracker                │
                      │    │   ├── Error Intelligence             │
                      │    │   └── Planner                        │
                      │    │                                      │
                      │    ├── Agent Layer                        │
                      │    │   ├── Claude Code process(es)        │
                      │    │   └── Research agents                │
                      │    │                                      │
                      │    └── SQLite Brain (.one/brain.db)       │
                      │                                          │
                      └──────────────────────────────────────────┘
```

**Three layers:**

### Interface Layer

Telegram bot. All interaction happens here.

The `one` CLI exists but is minimal — see CLI Commands section for details.
Everything else happens through Telegram.

### Brain Layer

The persistent intelligence that spans every session.

**Knowledge Graph** — an interconnected web of nodes and edges, inspired by
Obsidian's graph view. Not a flat list of facts — a web where you can see
clusters, connections, and orphans.

Node types:
- **Goal** — what the user wants to achieve
- **Constraint** — limitations (tech stack, budget, timeline)
- **Decision** — a choice made (with rationale, alternatives considered)
- **Component** — a piece of the system
- **Dependency** — something a component needs
- **Risk** — something that could go wrong
- **Research** — findings from Phase 2
- **Unknown** — something not yet answered (visible gaps)
- **File** — a source file (added during Phase 4)
- **Function** — a function/class/method (added during Phase 4)
- **Pattern** — a learned error→fix pair or operational knowledge
- **Anti-pattern** — something that doesn't work

Edge types:
- **requires** — Goal requires Component
- **informed_by** — Decision informed by Research
- **implements** — File implements Component
- **imports** — File imports from File
- **calls** — Function calls Function
- **tests** — File tests File
- **depends_on** — Component depends on Component
- **contradicts** — Research contradicts Decision (anti-thesis)
- **fixes** — Pattern fixes a specific error

The knowledge graph and topology graph are the same graph. During Phases 1-3,
it's populated with goals, decisions, components. During Phase 4, code nodes
get added and linked to those same components. A goal with no code path means
the project isn't done.

**Data model clarification:** The `nodes` and `edges` tables are the unified
graph. The `files` and `functions` tables are detail tables — each row has a
`node_id` FK pointing into the `nodes` table. A file IS a node (type=file)
with additional metadata (hash, size, language) stored in the `files` table.
Integrity checks query the `nodes`/`edges` tables for graph-level analysis
(orphans, missing connections) and join to `files`/`functions` for code-level
detail (unresolved imports, stub functions).

**Topology Engine** — parses the codebase into graph nodes after every change.

Reparse trigger: after each agent tool call that modifies files (Edit, Write).
One intercepts agent output, detects file-modifying tool calls, and triggers
an incremental reparse of only the changed files. Full reparse runs on
`one init` and on session start if file hashes have drifted. For a 500-file
TypeScript project, incremental reparse targets <100ms per changed file.

What it parses (language-aware):
- Files — path, hash, size, language, last modified
- Exports/Imports — what each file provides and consumes
- Functions/Classes — signatures, parameters, return types
- Call sites — who calls what, where
- Type references — interfaces implemented, types used
- Config references — env vars, config keys, route definitions
- Test coverage — which functions have tests, which don't

**v1 ships with TypeScript/JavaScript only.** The parser plugin interface is
designed for extensibility, but v1 focuses on one language done well:
- TypeScript/JavaScript (imports, exports, classes, React components)

Future parser plugins (v2+):
- Python (imports, functions, classes, decorators)
- Rust (use, mod, pub fn, impl, traits)
- Go (import, func, interfaces)

Integrity checks:

| Check | What it catches |
|---|---|
| Dangling imports | File A imports from File B, but B doesn't export it |
| Unimplemented interfaces | Interface declared, only 2 of 4 methods exist |
| Dead routes | Route registered but handler function missing |
| Orphan files | File exists but nothing imports or references it |
| Missing tests | Public function with no test coverage |
| Broken call chains | Function signature changed but callers not updated |
| Config gaps | Code references env var not in .env or config |
| Spec→Code gaps | Spec says "auth module" but no auth code exists |

**Session Tracker** — tracks what happened across every session. What changed,
what was planned, what was left incomplete. The memory that spans sessions.

**Error Intelligence** — learns from every failure and success. See dedicated
section below.

**Planner** — takes the spec, breaks it into tasks with dependencies, determines
execution order, assigns tasks to agents. Uses topology to understand what
depends on what.

### Agent Layer

Claude Code CLI processes managed by One.

Each agent is spawned with:
- `claude -p --output-format stream-json --input-format stream-json`
- Custom system prompt with briefing, topology context, known patterns
- Scoped tool permissions
- Specific task assignment

One monitors agent output in real-time:
- Logs every action
- Captures error→fix pairs
- Updates topology after every edit
- Checks integrity continuously

---

## Error Intelligence

The system learns from every failure and success, feeding knowledge forward
so every agent is smarter than the last.

### What One captures (automatically)

One watches every agent session and extracts:
- **Error→Fix pairs** — what went wrong, what solved it, normalized into patterns
- **Approach patterns** — "when building X, tried A then B then C, C worked"
- **Environment quirks** — "this project uses port 2222 for SSH", "DB requires SSL"
- **Anti-patterns** — "agent tried mocking the DB 3 times, failed every time"
- **Tool preferences** — "sed always breaks on this codebase, use Edit tool instead"

Each pattern is a node in the knowledge graph, connected to the components
and files it relates to.

### What agents receive (on startup)

Every time One spawns a Claude Code agent, it builds a custom briefing
assembled dynamically from the knowledge graph — queried for patterns
related to the specific files, components, and task the agent will work on:

```
BRIEFING FOR AGENT: implement-auth-module
─────────────────────────────────────────

PROJECT CONTEXT:
  [topology subgraph relevant to this task]

YOUR TASK:
  [specific task from the plan]

KNOWN PATTERNS FOR THIS AREA:
  - database connections: use connection pooling (learned session 3)
  - auth tokens: JWT with RS256, not HS256 (learned session 1)
  - SSH: key auth, port 2222 (confirmed sessions 1, 3, 5)

COMMON ERRORS & FIXES:
  - "ECONNREFUSED :5432" → postgres runs on :5433
  - "permission denied .env" → use .env.local
  - "test timeout" → increase to 30s, hits real DB

ANTI-PATTERNS (DO NOT):
  - Do not mock the database in tests
  - Do not use relative imports from src/

WHAT'S ALREADY DONE:
  [files and functions implemented]

WHAT'S NOT WIRED YET:
  [dangling connections for this task]
```

### Who writes the patterns

- **One** detects error→fix pairs by watching agent output automatically
- **Agents** can record learnings via a convention: the agent's system prompt
  instructs it to output a specific JSON block (`{"type": "learning", ...}`)
  when it discovers something important. One's watcher parses these from the
  stream-json output and stores them as pattern nodes. No MCP server needed —
  it's a structured output convention within the existing stream.
- **The user** can add patterns via Telegram: "remember: postgres is on port 5433"

---

## Agent Orchestration

### Task decomposition

The plan from Phase 3 is broken into tasks with dependencies. One knows the
execution order.

```
Task Graph:
  [1] Setup project structure
       ├──▶ [2] Database schema ──▶ [4] Auth module ──▶ [7] Auth tests
       └──▶ [3] API scaffolding ──▶ [5] User endpoints ──▶ [8] User tests
                                   └──▶ [6] Middleware ──▶ [9] Integration tests
                                                                 └──▶ [10] Deploy
```

### Single agent mode

One feeds tasks sequentially. After each task:
1. Topology engine reparses
2. Integrity check runs
3. Clean → next task
4. Dangling → agent fixes before moving on

### Multi-agent mode (v2)

Multi-agent is a v2 feature. The orchestrator interface is designed to support
it, but v1 ships with single-agent execution only. The concurrency machinery
(file locking, collision prevention, cross-agent wiring checks) adds significant
complexity that should be built after single-agent mode is proven.

**v2 design (for reference):**
One identifies independent branches in the task graph and runs in parallel:
- Agent 1: Database → Auth → Auth tests
- Agent 2: API scaffold → User endpoints → User tests
- Agent 3: Middleware (after API scaffold completes)

All agents share brain.db.

**Collision prevention (v2):**
- File-level advisory locks tracked in brain.db (agent_id, file_path, locked_at)
- Integrity checks deferred until an agent completes its full task, not after each edit
- Files marked "in-flight" are excluded from integrity scans until owning agent finishes
- After all parallel agents finish, One runs a cross-agent wiring check

### Agent failure handling

- **Stuck in loop** (same error 3+ times) — One kills it, spawns fresh agent
  with updated briefing: "Previous agent failed at X. Tried A, B, C. Try different."
- **Context exhaustion** (approaching token limit) — One checkpoints work, spawns
  new agent with briefing of what's done and what's left
- **Regression** (broke something working) — One detects via topology diff, rolls
  back the specific change

### Verification on task completion

When an agent claims "done," One verifies:
1. Topology integrity check — all imports resolve?
2. All functions implemented (no stubs)?
3. Tests pass?
4. Spec coverage — does this task's goals trace to code?

Pass → next task. Fail → agent fixes or new agent takes over.

---

## Telegram Interface

### Setup

```
$ cd my-project
$ one

ONE: No project found here.
     Let's set up your Telegram bot.

     1. Open @BotFather on Telegram
     2. Create a new bot
     3. Paste your bot token here:

> <token>

ONE: Connected. Open Telegram and message your bot.
     [One is now running as a background service]
```

### Interaction patterns

| Situation | What One sends |
|---|---|
| Clarifying question | Text message with inline button choices |
| Research complete | Summary message + full report as file attachment |
| Design section review | Formatted markdown, asks for approval |
| Checkpoint during build | Status summary + topology integrity report |
| Error agent can't solve | "Agent stuck on X. Tried A, B, C. What should I try?" |
| Milestone complete | Progress update with graph stats |
| Project complete | Final report + full topology status |
| Crash/recovery | "Went down at 3:14 PM. Recovered. Nothing lost. Continuing." |

### Quick actions via inline buttons

```
ONE: Milestone 2 complete: Auth + Database

     Files: 18 created
     Functions: 52 implemented
     Tests: 73 passing
     Spec: 4/12 goals done

     [Continue] [Review Code] [Pause] [Change Plan]
```

### Resuming

```
ONE: Welcome back. Session 12.

     SINCE LAST SESSION:
     • 2 files modified outside One (detected)
     • No broken wiring

     STATUS:
     • Progress: 6/14 goals
     • Current: Payment integration
     • Next: Stripe webhook handlers

     Patterns learned last session:
     • Stripe webhooks need idempotency keys
     • Use signing verification, not IP allowlisting
     • Test with Stripe CLI, not ngrok

     Ready to continue. Chat or run?
```

---

## SQLite Brain Schema

Single file: `.one/brain.db`

```sql
-- Knowledge Graph
nodes (
  id, type, name, description, status, phase_created,
  session_created, metadata_json, created_at, updated_at
)
-- type: goal, constraint, decision, component, dependency,
--       risk, research, unknown, file, function, pattern, anti_pattern

edges (
  id, source_id, target_id, type, weight, metadata_json, created_at
)
-- type: requires, informed_by, implements, imports, calls,
--       tests, depends_on, contradicts, fixes

-- Topology (code-level detail)
files (
  id, path, hash, size, lines, language, node_id,
  last_parsed, created_at, updated_at
)

functions (
  id, file_id, name, signature, start_line, end_line,
  body_hash, complexity, node_id, created_at, updated_at
)

imports (
  id, source_file_id, target_file_id, symbol, resolved, created_at
)

-- Sessions
sessions (
  id, phase, status, focus, started_at, ended_at,
  summary, tokens_used, agents_spawned
)

-- Tasks
tasks (
  id, title, description, status, priority,
  assigned_agent, session_created, session_completed,
  blocked_by_json, created_at, updated_at
)

-- Error Intelligence
patterns (
  id, type, trigger, resolution, confidence,
  times_seen, times_worked, node_id,
  related_files_json, created_at, updated_at
)
-- type: error_fix, approach, environment, anti_pattern, tool_preference

-- Agent Tracking
agents (
  id, session_id, task_id, model, focus, status,
  pid, tokens_used, started_at, ended_at
)

-- Audit
action_log (
  id, session_id, agent_id, action_type, target,
  input_summary, output_summary, success,
  duration_ms, created_at
)

conversation_log (
  id, session_id, phase, role, content,
  knowledge_updates_json, created_at
)

-- Telegram
telegram_config (
  bot_token, chat_id, configured_at
)

telegram_messages (
  id, session_id, direction, content, message_id, created_at
)
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Core service | TypeScript (Node.js) | Fast to build, strong async, good library ecosystem |
| Database | SQLite (better-sqlite3) | Single file brain, no server, travels with repo |
| Telegram | grammy or telegraf | Mature Telegram bot frameworks for Node |
| Claude Code | CLI child process | `claude -p --output-format stream-json --input-format stream-json` |
| Topology parsing | Tree-sitter | Battle-tested multi-language parser, native bindings |
| Process management | Node child_process | Spawn and manage Claude Code processes |
| Background service | systemd / launchd / pm2 | Keep One running when terminal closes |

**Why TypeScript, not C:**
ORION is C because it's a TUI with sub-millisecond rendering requirements.
One is an orchestrator — it manages processes, parses JSON, talks to Telegram,
and queries SQLite. TypeScript is the right tool: fast enough, vastly faster
to develop, better library ecosystem for the integrations One needs.

---

## File Layout

```
one/
├── src/
│   ├── core/
│   │   ├── brain.ts           # SQLite brain — schema, queries, migrations
│   │   ├── knowledge.ts       # Knowledge graph — nodes, edges, queries
│   │   ├── session.ts         # Session lifecycle, crash recovery
│   │   └── config.ts          # Project configuration
│   ├── phases/
│   │   ├── understand.ts      # Phase 1 — conversation logic, unknown tracking
│   │   ├── research.ts        # Phase 2 — research agent spawning, report parsing
│   │   ├── spec.ts            # Phase 3 — spec generation, section-by-section
│   │   └── build.ts           # Phase 4 — task execution, orchestration
│   ├── topology/
│   │   ├── engine.ts          # Topology engine — parse, diff, verify
│   │   ├── parsers/
│   │   │   └── typescript.ts  # TS/JS parser (tree-sitter) — v1 only language
│   │   ├── integrity.ts       # Integrity checks — dangling, orphans, stubs
│   │   └── differ.ts          # Change detection between parses
│   ├── agents/
│   │   ├── orchestrator.ts    # Agent spawning, assignment, monitoring
│   │   ├── claude.ts          # Claude Code process management
│   │   ├── briefing.ts        # Dynamic briefing generation from knowledge graph
│   │   └── watcher.ts         # Real-time agent output monitoring, pattern capture
│   ├── intelligence/
│   │   ├── patterns.ts        # Error→fix pattern detection and storage
│   │   └── planner.ts         # Spec→tasks→dependency graph→execution order
│   ├── telegram/
│   │   ├── bot.ts             # Telegram bot setup, message routing
│   │   ├── handlers.ts        # Message handlers per phase
│   │   └── formatter.ts       # Rich message formatting, inline buttons
│   └── cli/
│       └── index.ts           # Minimal CLI — init, stop, status, logs
├── tests/
│   ├── topology/              # Parser unit tests with fixture files
│   ├── knowledge/             # Knowledge graph CRUD and query tests
│   ├── intelligence/          # Pattern extraction and briefing tests
│   ├── agents/                # Orchestrator integration tests (mock CLI)
│   ├── telegram/              # Bot integration tests (mock API)
│   ├── e2e/                   # End-to-end smoke tests
│   └── fixtures/              # Sample files, recorded streams, schemas
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-03-13-one-design.md
├── package.json
├── tsconfig.json
└── LICENSE
```

---

## Crash Recovery

One is a long-running background service managing child processes. It must
handle crashes gracefully.

**SQLite:** WAL mode enabled on database open. All writes are atomic transactions.
If One crashes mid-write, SQLite recovers automatically on next open.

**Session state machine:** `started → running → interrupted → completed`
If One starts and finds a session in `running` state, it was interrupted.
Recovery steps:
1. Check for orphaned Claude Code processes (stored PIDs in `agents` table).
   Kill any that are still alive.
2. Mark interrupted session as `interrupted` with timestamp.
3. Run full topology reparse to determine actual codebase state.
4. Diff topology against last known good state to detect partial work.
5. Start new session. Send Telegram message: "Recovered from interruption.
   Session N was interrupted. Topology shows X files changed, Y integrity
   issues. Resuming from task Z."

**Agent PID tracking:** Every spawned Claude Code process has its PID stored
in `agents` table. On startup, One iterates all agents with status `running`,
checks if PID is alive, and kills/reaps as needed.

**Checkpoint frequency:** During Phase 4, One commits a checkpoint to brain.db
after every completed task (not after every edit). The checkpoint records:
current task progress, topology snapshot hash, and agent state. Recovery
resumes from the last completed task.

---

## Security

**Threat model:** One runs on a single developer's local machine. The Telegram
bot is the only network-facing surface. The machine and network are assumed
trusted. This is not a multi-tenant system.

**Telegram authorization:**
- On first contact, One records the `chat_id` in `telegram_config`.
- All subsequent messages are checked against this `chat_id`. Messages from
  any other sender are silently dropped.
- If the `chat_id` needs to change, the developer runs `one reset-telegram`
  from the CLI (requires local access).

**Bot token storage:** Stored in plaintext in `.one/brain.db`. This is
acceptable for a local-only tool — the database file has the same access
permissions as the developer's other files. The `.one/` directory should be
added to `.gitignore` (One does this automatically on init).

**Agent sandboxing:** Claude Code agents inherit the developer's permissions.
One does not add additional sandboxing beyond what Claude Code provides.
The developer is responsible for their system's security posture.

---

## Cost Controls

One spawns Claude Code agents that consume API tokens. Uncontrolled autonomous
execution is a real spending risk.

**Budget system:**
- On first run, One asks via Telegram: "Set a daily token budget? (recommended)"
- Budget stored in `.one/config.json` as `daily_token_budget`
- Default: no limit, but One warns if no budget is set before entering Phase 4

**Tracking:**
- Every agent's token usage is tracked in the `agents` table (`tokens_used`)
- One aggregates daily usage from all agents

**Guardrails:**
- At 80% of daily budget: Telegram notification — "You've used 80% of today's
  budget. N tokens remaining."
- At 100%: Hard stop. All agents paused. Telegram: "Daily budget reached.
  [Continue anyway] [Stop for today] [Increase budget]"
- Per-task estimate: Before Phase 4, One estimates token cost based on task
  count and complexity. Shown to user for approval.

---

## Installation & Prerequisites

**Prerequisites:**
- Node.js >= 20
- Claude Code CLI installed and authenticated (`claude --version` works)
- Telegram account (for bot setup)

**Installation:**
```
npm install -g @anthropic/one
```

Tree-sitter has native bindings requiring node-gyp. One uses prebuilt binaries
via `tree-sitter-typescript` npm packages where available, falling back to
compilation. Installation docs will list platform-specific notes.

**Background service:**
One uses pm2 as the universal process manager (cross-platform: Linux, macOS,
Windows WSL). On `one` init:
1. Checks if pm2 is installed, installs it globally if not
2. Starts One as a pm2 process: `pm2 start one-service --name one`
3. Configures pm2 startup hook so One survives reboots

---

## Configuration

Project configuration lives in `.one/config.json`:

```json
{
  "project_root": "/path/to/project",
  "model": "sonnet",
  "daily_token_budget": 500000,
  "checkpoint_mode": "after_each_task",
  "max_retries_per_task": 3,
  "auto_commit": false,
  "claude_code_path": "claude",
  "min_claude_code_version": "1.0.0"
}
```

The `.one/` directory is auto-added to `.gitignore` on init. It contains:
- `brain.db` — the SQLite brain
- `config.json` — project settings
- `logs/` — structured log files

Global settings (shared across projects) live in `~/.one/global.json`:
- Default model
- Default budget
- Telegram bot token (if reusing across projects)

---

## Existing Project Support

Running `one` in a directory with existing code triggers a different flow:

```
$ cd existing-project/
$ one

ONE: I see an existing project here. Let me understand it first.
     [Scanning 247 files...]
     [Building topology graph...]

ONE: Here's what I found:
     • 247 files, primarily TypeScript
     • 12 entry points detected
     • 34 dangling imports (!)
     • Test coverage: 62%

     What do you want to do?
     [Add a feature] [Fix something] [Refactor] [Just explore]
```

One parses the existing codebase into the topology graph first, then enters
Phase 1 with the context of "you have X, what do you want to change/add?"
The knowledge graph starts pre-populated with File and Function nodes from
the existing code.

---

## Database Migrations

The brain.db schema will evolve. Migration strategy:

- `meta` table stores `schema_version` (integer, starts at 1)
- Migration files live in `src/core/migrations/` as numbered TypeScript files:
  `001_initial.ts`, `002_add_cost_tracking.ts`, etc.
- On startup, One reads `schema_version`, runs any migrations with a higher
  number, updates `schema_version`
- Migrations run inside a transaction — if one fails, the DB rolls back
- brain.db is user data that must survive upgrades

---

## Logging & Observability

**Log location:** `.one/logs/`

**Format:** Structured JSON, one object per line (JSONL):
```json
{"ts": "2026-03-13T14:30:00Z", "level": "info", "component": "orchestrator", "event": "task_started", "task_id": 4, "agent_id": "a1"}
```

**Levels:**
- `info` — phase transitions, task start/complete, session events
- `debug` — agent tool calls, topology changes, knowledge graph updates
- `error` — agent failures, recovery actions, integrity violations

**Rotation:** Daily rotation. Logs older than 7 days auto-deleted. Configurable
in `.one/config.json`.

**`one logs` command:** Tails the current day's log file with human-readable
formatting (timestamps, colored levels, component names).

---

## Claude Code Compatibility

One depends on Claude Code CLI flags:
- `--print` (`-p`) for non-interactive mode
- `--output-format stream-json` for structured output parsing
- `--input-format stream-json` for structured input
- `--system-prompt` for agent briefing injection

**On startup:** One runs `claude --version` and checks against a minimum
supported version stored in `src/core/constants.ts`. If below minimum,
One warns: "Claude Code vX.Y detected. One requires vX.Z+. Some features
may not work." Does not hard-block — the developer may know what they're doing.

---

## Testing Strategy

**Framework:** Vitest (fast, native TypeScript, good mocking support).

**Test structure:**

| Area | Test type | What it covers |
|---|---|---|
| Topology parsers | Unit | Parse fixture files, verify extracted nodes/edges. One fixture per language feature (imports, exports, classes, functions). |
| Knowledge graph | Unit | Node/edge CRUD, graph queries (orphans, paths, clusters), edge cases (circular deps). |
| Integrity checker | Unit | Feed known-broken topologies, verify correct violation detection. |
| Session tracker | Integration | Session lifecycle, crash recovery simulation, orphan detection. |
| Error intelligence | Unit | Pattern extraction from mock agent output, briefing generation. |
| Planner | Unit | Spec→task decomposition, dependency ordering, cycle detection. |
| Agent orchestrator | Integration | Mock Claude Code CLI (spawn process that emits fixture stream-json), verify task flow, failure handling. |
| Telegram bot | Integration | Mock Telegram API, verify message formatting, button handling, phase routing. |
| Brain (SQLite) | Integration | Schema creation, migrations, concurrent access, WAL recovery. |
| End-to-end | E2E | Tiny project: init → understand (mock user) → research (mock web) → spec → build (mock agent) → verify topology. Uses mock Telegram bot. |

**Test fixtures:**
- `tests/fixtures/topology/` — sample TypeScript files with known structures
- `tests/fixtures/stream-json/` — recorded Claude Code output for replay
- `tests/fixtures/schemas/` — known-good and known-broken topology states

**CI:** Tests run on push. No merge without green tests.

---

## v1 Scope Summary

What ships in v1:
- Telegram interface (full)
- Four-phase lifecycle (full)
- Knowledge graph (full)
- Topology engine (TypeScript/JavaScript only)
- Single-agent execution
- Error intelligence (full)
- Crash recovery
- Cost controls
- Existing project support

What's deferred to v2+:
- Multi-agent parallel execution
- Python/Rust/Go parsers
- Web dashboard
- Team collaboration features

---

## What Success Looks Like

A developer runs `one` in an empty directory. Sets up Telegram. Describes what
they want to build. One asks questions until it understands. Researches the space.
Walks through the design. Produces a thorough spec. Then builds it — spawning
Claude Code agents, tracking every file and function, catching every dangling
connection, learning from every error.

By the end of a full day, the project is complete. Not "mostly done." Not
"the happy path works." Complete — every goal traces through decisions through
components through tested, wired, working code.

The developer checks their Telegram. Sees the final report. Reviews the code.
It's done.

That's One.
