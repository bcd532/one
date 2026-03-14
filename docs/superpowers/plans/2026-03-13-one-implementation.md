# One Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build One — an autonomous project execution engine that talks to users via Telegram and drives Claude Code agents to build complete projects with topology-aware integrity tracking.

**Architecture:** TypeScript Node.js service with SQLite brain (better-sqlite3), Telegram bot (grammy), tree-sitter for code parsing, and Claude Code CLI child processes for agent execution. Single-agent mode for v1. Background service managed by pm2.

**Tech Stack:** TypeScript, Node.js 20+, SQLite (better-sqlite3), grammy, tree-sitter, Vitest, Commander.js

**Spec:** `docs/superpowers/specs/2026-03-13-one-design.md`

**Dependency notes:**
- Planner (Task 14) must be built before Orchestrator (Task 15) — orchestrator consumes task graph
- Phase Handlers (Task 22) are built after Phase logic (Tasks 23-26) — handlers delegate to phases
- Cost Controls (Task 17) must exist before Phase 4 Build (Task 26)

---

## Chunk 1: Project Scaffolding & SQLite Brain

### Task 1: Initialize project

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `.gitignore`
- Create: `vitest.config.ts`

- [ ] **Step 1: Initialize npm project**

```bash
npm init -y
```

- [ ] **Step 2: Install core dependencies**

```bash
npm install better-sqlite3 grammy commander tree-sitter tree-sitter-typescript
npm install -D typescript vitest @types/node @types/better-sqlite3 tsx
```

- [ ] **Step 3: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "outDir": "dist",
    "rootDir": "src",
    "strict": true,
    "esModuleInterop": true,
    "declaration": true,
    "sourceMap": true,
    "forceConsistentCasingInFileNames": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

- [ ] **Step 4: Create vitest.config.ts**

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    include: ['tests/**/*.test.ts'],
    testTimeout: 10000,
  },
});
```

- [ ] **Step 5: Create .gitignore**

```
node_modules/
dist/
.one/
*.db
*.db-wal
*.db-shm
```

- [ ] **Step 6: Update package.json scripts**

Add to package.json:
```json
{
  "type": "module",
  "bin": { "one": "./dist/cli/index.js" },
  "scripts": {
    "build": "tsc",
    "dev": "tsx src/cli/index.ts",
    "test": "vitest run",
    "test:watch": "vitest"
  }
}
```

- [ ] **Step 7: Create directory structure**

```bash
mkdir -p src/{core/migrations,phases,topology/parsers,agents,intelligence,automation,telegram,cli}
mkdir -p tests/{core,topology,knowledge,intelligence,agents,telegram,e2e,fixtures/{topology,stream-json,schemas}}
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: initialize project scaffolding with TypeScript, Vitest, dependencies"
```

---

### Task 2: SQLite Brain — Schema & Migrations

**Files:**
- Create: `src/core/brain.ts`
- Create: `src/core/migrations/001_initial.ts`
- Test: `tests/core/brain.test.ts`

- [ ] **Step 1: Write failing tests for brain initialization**

```typescript
// tests/core/brain.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Brain } from '../../src/core/brain.js';
import fs from 'fs';
import path from 'path';

const TEST_DIR = '/tmp/one-test-brain';
const DB_PATH = path.join(TEST_DIR, 'brain.db');

describe('Brain', () => {
  beforeEach(() => {
    fs.mkdirSync(TEST_DIR, { recursive: true });
  });

  afterEach(() => {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it('creates database with all tables', () => {
    const brain = new Brain(DB_PATH);
    const tables = brain.listTables();
    expect(tables).toContain('meta');
    expect(tables).toContain('nodes');
    expect(tables).toContain('edges');
    expect(tables).toContain('files');
    expect(tables).toContain('functions');
    expect(tables).toContain('imports');
    expect(tables).toContain('sessions');
    expect(tables).toContain('tasks');
    expect(tables).toContain('patterns');
    expect(tables).toContain('agents');
    expect(tables).toContain('action_log');
    expect(tables).toContain('conversation_log');
    expect(tables).toContain('telegram_config');
    expect(tables).toContain('telegram_messages');
    brain.close();
  });

  it('enables WAL mode', () => {
    const brain = new Brain(DB_PATH);
    const mode = brain.getJournalMode();
    expect(mode).toBe('wal');
    brain.close();
  });

  it('sets schema version in meta', () => {
    const brain = new Brain(DB_PATH);
    const version = brain.getMeta('schema_version');
    expect(version).toBe('1');
    brain.close();
  });

  it('is idempotent — opening twice does not error', () => {
    const brain1 = new Brain(DB_PATH);
    brain1.close();
    const brain2 = new Brain(DB_PATH);
    const version = brain2.getMeta('schema_version');
    expect(version).toBe('1');
    brain2.close();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/core/brain.test.ts`
Expected: FAIL — module not found

- [ ] **Step 3: Write migration 001_initial.ts**

```typescript
// src/core/migrations/001_initial.ts
import type Database from 'better-sqlite3';

export const version = 1;

export function up(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS meta (
      key TEXT PRIMARY KEY,
      value TEXT
    );

    CREATE TABLE IF NOT EXISTS nodes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      type TEXT NOT NULL,
      name TEXT NOT NULL,
      description TEXT,
      status TEXT DEFAULT 'active',
      phase_created INTEGER,
      session_created INTEGER,
      metadata_json TEXT DEFAULT '{}',
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
    CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);

    CREATE TABLE IF NOT EXISTS edges (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
      target_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
      type TEXT NOT NULL,
      weight REAL DEFAULT 1.0,
      metadata_json TEXT DEFAULT '{}',
      created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
    CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);

    CREATE TABLE IF NOT EXISTS files (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      path TEXT NOT NULL UNIQUE,
      hash TEXT,
      size INTEGER,
      lines INTEGER,
      language TEXT,
      node_id INTEGER REFERENCES nodes(id) ON DELETE SET NULL,
      last_parsed TEXT,
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
    CREATE INDEX IF NOT EXISTS idx_files_node ON files(node_id);

    CREATE TABLE IF NOT EXISTS functions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
      name TEXT NOT NULL,
      signature TEXT,
      start_line INTEGER,
      end_line INTEGER,
      body_hash TEXT,
      complexity INTEGER DEFAULT 0,
      node_id INTEGER REFERENCES nodes(id) ON DELETE SET NULL,
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_functions_file ON functions(file_id);
    CREATE INDEX IF NOT EXISTS idx_functions_node ON functions(node_id);

    CREATE TABLE IF NOT EXISTS imports (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
      target_file_id INTEGER REFERENCES files(id) ON DELETE SET NULL,
      symbol TEXT,
      resolved INTEGER DEFAULT 0,
      created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_imports_source ON imports(source_file_id);
    CREATE INDEX IF NOT EXISTS idx_imports_target ON imports(target_file_id);

    CREATE TABLE IF NOT EXISTS sessions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      phase INTEGER,
      status TEXT DEFAULT 'started',
      focus TEXT,
      started_at TEXT DEFAULT (datetime('now')),
      ended_at TEXT,
      summary TEXT,
      tokens_used INTEGER DEFAULT 0,
      agents_spawned INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS tasks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      description TEXT,
      status TEXT DEFAULT 'pending',
      priority INTEGER DEFAULT 0,
      assigned_agent TEXT,
      session_created INTEGER,
      session_completed INTEGER,
      blocked_by_json TEXT DEFAULT '[]',
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

    CREATE TABLE IF NOT EXISTS patterns (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      type TEXT NOT NULL,
      trigger TEXT NOT NULL,
      resolution TEXT,
      confidence REAL DEFAULT 0.0,
      times_seen INTEGER DEFAULT 1,
      times_worked INTEGER DEFAULT 0,
      node_id INTEGER REFERENCES nodes(id) ON DELETE SET NULL,
      related_files_json TEXT DEFAULT '[]',
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(type);

    CREATE TABLE IF NOT EXISTS agents (
      id TEXT PRIMARY KEY,
      session_id INTEGER REFERENCES sessions(id),
      task_id INTEGER REFERENCES tasks(id),
      model TEXT,
      focus TEXT,
      status TEXT DEFAULT 'pending',
      pid INTEGER,
      tokens_used INTEGER DEFAULT 0,
      started_at TEXT DEFAULT (datetime('now')),
      ended_at TEXT
    );

    CREATE TABLE IF NOT EXISTS action_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id INTEGER,
      agent_id TEXT,
      action_type TEXT NOT NULL,
      target TEXT,
      input_summary TEXT,
      output_summary TEXT,
      success INTEGER DEFAULT 1,
      duration_ms INTEGER,
      created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_action_log_session ON action_log(session_id);

    CREATE TABLE IF NOT EXISTS conversation_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id INTEGER,
      phase INTEGER,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      knowledge_updates_json TEXT DEFAULT '[]',
      created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS telegram_config (
      bot_token TEXT,
      chat_id TEXT,
      configured_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS telegram_messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id INTEGER,
      direction TEXT NOT NULL,
      content TEXT NOT NULL,
      message_id TEXT,
      created_at TEXT DEFAULT (datetime('now'))
    );
  `);
}
```

- [ ] **Step 4: Write brain.ts**

```typescript
// src/core/brain.ts
import Database from 'better-sqlite3';
import { up as migrate001, version as v1 } from './migrations/001_initial.js';

export class Brain {
  private db: Database.Database;

  constructor(dbPath: string) {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.runMigrations();
  }

  private runMigrations(): void {
    // Ensure meta table exists for version tracking
    this.db.exec(`CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)`);

    const currentVersion = parseInt(this.getMeta('schema_version') ?? '0', 10);

    const migrations = [{ version: v1, up: migrate001 }];

    for (const migration of migrations) {
      if (currentVersion < migration.version) {
        this.db.transaction(() => {
          migration.up(this.db);
          this.setMeta('schema_version', String(migration.version));
        })();
      }
    }
  }

  getMeta(key: string): string | undefined {
    const row = this.db.prepare('SELECT value FROM meta WHERE key = ?').get(key) as
      | { value: string }
      | undefined;
    return row?.value;
  }

  setMeta(key: string, value: string): void {
    this.db.prepare('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)').run(key, value);
  }

  listTables(): string[] {
    const rows = this.db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
      .all() as { name: string }[];
    return rows.map((r) => r.name);
  }

  getJournalMode(): string {
    const row = this.db.prepare('PRAGMA journal_mode').get() as { journal_mode: string };
    return row.journal_mode;
  }

  getDb(): Database.Database {
    return this.db;
  }

  close(): void {
    this.db.close();
  }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run tests/core/brain.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/core/brain.ts src/core/migrations/001_initial.ts tests/core/brain.test.ts
git commit -m "feat: SQLite brain with schema, migrations, WAL mode"
```

---

### Task 3: Configuration System

**Files:**
- Create: `src/core/config.ts`
- Test: `tests/core/config.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/core/config.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Config, DEFAULT_CONFIG } from '../../src/core/config.js';
import fs from 'fs';
import path from 'path';

const TEST_DIR = '/tmp/one-test-config';

describe('Config', () => {
  beforeEach(() => {
    fs.mkdirSync(path.join(TEST_DIR, '.one'), { recursive: true });
  });

  afterEach(() => {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it('creates default config when none exists', () => {
    const config = Config.load(TEST_DIR);
    expect(config.model).toBe('sonnet');
    expect(config.daily_token_budget).toBe(0);
    expect(config.max_retries_per_task).toBe(3);
  });

  it('saves and reloads config', () => {
    const config = Config.load(TEST_DIR);
    config.model = 'opus';
    Config.save(TEST_DIR, config);
    const reloaded = Config.load(TEST_DIR);
    expect(reloaded.model).toBe('opus');
  });

  it('merges partial config with defaults', () => {
    const configPath = path.join(TEST_DIR, '.one', 'config.json');
    fs.writeFileSync(configPath, JSON.stringify({ model: 'haiku' }));
    const config = Config.load(TEST_DIR);
    expect(config.model).toBe('haiku');
    expect(config.max_retries_per_task).toBe(3); // default preserved
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/core/config.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement config.ts**

```typescript
// src/core/config.ts
import fs from 'fs';
import path from 'path';

export interface OneConfig {
  project_root: string;
  model: string;
  daily_token_budget: number;
  checkpoint_mode: 'after_each_task' | 'after_each_milestone' | 'never';
  max_retries_per_task: number;
  auto_commit: boolean;
  claude_code_path: string;
  hooks: Record<string, string | null>;
  schedules: Record<string, string>;
}

export const DEFAULT_CONFIG: OneConfig = {
  project_root: '',
  model: 'sonnet',
  daily_token_budget: 0,
  checkpoint_mode: 'after_each_task',
  max_retries_per_task: 3,
  auto_commit: false,
  claude_code_path: 'claude',
  hooks: {},
  schedules: {},
};

export class Config {
  static load(projectRoot: string): OneConfig {
    const configPath = path.join(projectRoot, '.one', 'config.json');
    let partial: Partial<OneConfig> = {};
    if (fs.existsSync(configPath)) {
      partial = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    }
    return { ...DEFAULT_CONFIG, project_root: projectRoot, ...partial };
  }

  static save(projectRoot: string, config: OneConfig): void {
    const dotOne = path.join(projectRoot, '.one');
    fs.mkdirSync(dotOne, { recursive: true });
    const configPath = path.join(dotOne, 'config.json');
    const { project_root, ...rest } = config;
    fs.writeFileSync(configPath, JSON.stringify(rest, null, 2) + '\n');
  }
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/core/config.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/config.ts tests/core/config.test.ts
git commit -m "feat: configuration system with defaults and merge"
```

---

### Task 4: Logger

**Files:**
- Create: `src/core/logger.ts`
- Test: `tests/core/logger.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/core/logger.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Logger } from '../../src/core/logger.js';
import fs from 'fs';
import path from 'path';

const TEST_DIR = '/tmp/one-test-logger';
const LOG_DIR = path.join(TEST_DIR, '.one', 'logs');

describe('Logger', () => {
  beforeEach(() => {
    fs.mkdirSync(LOG_DIR, { recursive: true });
  });

  afterEach(() => {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it('writes JSONL to daily log file', () => {
    const logger = new Logger(LOG_DIR);
    logger.info('orchestrator', 'task_started', { task_id: 1 });
    logger.flush();
    const today = new Date().toISOString().split('T')[0];
    const logFile = path.join(LOG_DIR, `${today}.jsonl`);
    expect(fs.existsSync(logFile)).toBe(true);
    const line = JSON.parse(fs.readFileSync(logFile, 'utf-8').trim());
    expect(line.level).toBe('info');
    expect(line.component).toBe('orchestrator');
    expect(line.event).toBe('task_started');
    expect(line.task_id).toBe(1);
  });

  it('supports info, debug, error levels', () => {
    const logger = new Logger(LOG_DIR);
    logger.info('test', 'info_event');
    logger.debug('test', 'debug_event');
    logger.error('test', 'error_event', { err: 'oops' });
    logger.flush();
    const today = new Date().toISOString().split('T')[0];
    const lines = fs.readFileSync(path.join(LOG_DIR, `${today}.jsonl`), 'utf-8')
      .trim().split('\n').map(l => JSON.parse(l));
    expect(lines.map(l => l.level)).toEqual(['info', 'debug', 'error']);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/core/logger.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement logger.ts**

```typescript
// src/core/logger.ts
import fs from 'fs';
import path from 'path';

export class Logger {
  private logDir: string;
  private buffer: string[] = [];

  constructor(logDir: string) {
    this.logDir = logDir;
    fs.mkdirSync(logDir, { recursive: true });
  }

  private log(level: string, component: string, event: string, data?: Record<string, unknown>): void {
    const entry = {
      ts: new Date().toISOString(),
      level,
      component,
      event,
      ...data,
    };
    this.buffer.push(JSON.stringify(entry));
  }

  info(component: string, event: string, data?: Record<string, unknown>): void {
    this.log('info', component, event, data);
  }

  debug(component: string, event: string, data?: Record<string, unknown>): void {
    this.log('debug', component, event, data);
  }

  error(component: string, event: string, data?: Record<string, unknown>): void {
    this.log('error', component, event, data);
  }

  flush(): void {
    if (this.buffer.length === 0) return;
    const today = new Date().toISOString().split('T')[0];
    const logFile = path.join(this.logDir, `${today}.jsonl`);
    fs.appendFileSync(logFile, this.buffer.join('\n') + '\n');
    this.buffer = [];
  }
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/core/logger.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/logger.ts tests/core/logger.test.ts
git commit -m "feat: JSONL structured logger with daily rotation"
```

---

## Chunk 2: Knowledge Graph

### Task 5: Knowledge Graph — Node & Edge CRUD

**Files:**
- Create: `src/core/knowledge.ts`
- Test: `tests/knowledge/knowledge.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/knowledge/knowledge.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Brain } from '../../src/core/brain.js';
import { KnowledgeGraph } from '../../src/core/knowledge.js';
import fs from 'fs';
import path from 'path';

const TEST_DIR = '/tmp/one-test-knowledge';
const DB_PATH = path.join(TEST_DIR, 'brain.db');

describe('KnowledgeGraph', () => {
  let brain: Brain;
  let kg: KnowledgeGraph;

  beforeEach(() => {
    fs.mkdirSync(TEST_DIR, { recursive: true });
    brain = new Brain(DB_PATH);
    kg = new KnowledgeGraph(brain);
  });

  afterEach(() => {
    brain.close();
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  describe('nodes', () => {
    it('creates and retrieves a node', () => {
      const id = kg.addNode({ type: 'goal', name: 'Build auth', description: 'User auth system' });
      const node = kg.getNode(id);
      expect(node?.type).toBe('goal');
      expect(node?.name).toBe('Build auth');
    });

    it('lists nodes by type', () => {
      kg.addNode({ type: 'goal', name: 'Goal 1' });
      kg.addNode({ type: 'goal', name: 'Goal 2' });
      kg.addNode({ type: 'component', name: 'Comp 1' });
      const goals = kg.getNodesByType('goal');
      expect(goals).toHaveLength(2);
    });

    it('updates a node', () => {
      const id = kg.addNode({ type: 'unknown', name: 'Tech stack?' });
      kg.updateNode(id, { type: 'decision', name: 'Use TypeScript', status: 'resolved' });
      const node = kg.getNode(id);
      expect(node?.type).toBe('decision');
      expect(node?.status).toBe('resolved');
    });

    it('deletes a node and cascades edges', () => {
      const a = kg.addNode({ type: 'goal', name: 'A' });
      const b = kg.addNode({ type: 'component', name: 'B' });
      kg.addEdge({ source_id: a, target_id: b, type: 'requires' });
      kg.deleteNode(a);
      expect(kg.getNode(a)).toBeUndefined();
      expect(kg.getEdgesFrom(a)).toHaveLength(0);
    });
  });

  describe('edges', () => {
    it('creates and retrieves edges', () => {
      const a = kg.addNode({ type: 'goal', name: 'A' });
      const b = kg.addNode({ type: 'component', name: 'B' });
      kg.addEdge({ source_id: a, target_id: b, type: 'requires' });
      const edges = kg.getEdgesFrom(a);
      expect(edges).toHaveLength(1);
      expect(edges[0].type).toBe('requires');
      expect(edges[0].target_id).toBe(b);
    });

    it('finds incoming edges', () => {
      const a = kg.addNode({ type: 'goal', name: 'A' });
      const b = kg.addNode({ type: 'component', name: 'B' });
      kg.addEdge({ source_id: a, target_id: b, type: 'requires' });
      const incoming = kg.getEdgesTo(b);
      expect(incoming).toHaveLength(1);
      expect(incoming[0].source_id).toBe(a);
    });
  });

  describe('queries', () => {
    it('finds orphan nodes (no edges)', () => {
      const a = kg.addNode({ type: 'component', name: 'Connected' });
      const b = kg.addNode({ type: 'component', name: 'Orphan' });
      const c = kg.addNode({ type: 'goal', name: 'Goal' });
      kg.addEdge({ source_id: c, target_id: a, type: 'requires' });
      const orphans = kg.findOrphans();
      expect(orphans).toHaveLength(1);
      expect(orphans[0].name).toBe('Orphan');
    });

    it('finds nodes with status unknown', () => {
      kg.addNode({ type: 'unknown', name: 'What DB?' });
      kg.addNode({ type: 'goal', name: 'Build it' });
      const unknowns = kg.getNodesByType('unknown');
      expect(unknowns).toHaveLength(1);
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/knowledge/knowledge.test.ts`
Expected: FAIL

- [ ] **Step 3: Implement knowledge.ts**

```typescript
// src/core/knowledge.ts
import type { Brain } from './brain.js';

export interface NodeInput {
  type: string;
  name: string;
  description?: string;
  status?: string;
  phase_created?: number;
  session_created?: number;
  metadata_json?: string;
}

export interface NodeRow {
  id: number;
  type: string;
  name: string;
  description: string | null;
  status: string;
  phase_created: number | null;
  session_created: number | null;
  metadata_json: string;
  created_at: string;
  updated_at: string;
}

export interface EdgeInput {
  source_id: number;
  target_id: number;
  type: string;
  weight?: number;
  metadata_json?: string;
}

export interface EdgeRow {
  id: number;
  source_id: number;
  target_id: number;
  type: string;
  weight: number;
  metadata_json: string;
  created_at: string;
}

export class KnowledgeGraph {
  private db;

  constructor(brain: Brain) {
    this.db = brain.getDb();
  }

  addNode(input: NodeInput): number {
    const result = this.db.prepare(
      `INSERT INTO nodes (type, name, description, status, phase_created, session_created, metadata_json)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    ).run(
      input.type, input.name, input.description ?? null,
      input.status ?? 'active', input.phase_created ?? null,
      input.session_created ?? null, input.metadata_json ?? '{}'
    );
    return Number(result.lastInsertRowid);
  }

  getNode(id: number): NodeRow | undefined {
    return this.db.prepare('SELECT * FROM nodes WHERE id = ?').get(id) as NodeRow | undefined;
  }

  getNodesByType(type: string): NodeRow[] {
    return this.db.prepare('SELECT * FROM nodes WHERE type = ?').all(type) as NodeRow[];
  }

  updateNode(id: number, updates: Partial<NodeInput>): void {
    const fields: string[] = [];
    const values: unknown[] = [];
    for (const [key, value] of Object.entries(updates)) {
      fields.push(`${key} = ?`);
      values.push(value);
    }
    fields.push("updated_at = datetime('now')");
    values.push(id);
    this.db.prepare(`UPDATE nodes SET ${fields.join(', ')} WHERE id = ?`).run(...values);
  }

  deleteNode(id: number): void {
    this.db.prepare('DELETE FROM nodes WHERE id = ?').run(id);
  }

  addEdge(input: EdgeInput): number {
    const result = this.db.prepare(
      `INSERT INTO edges (source_id, target_id, type, weight, metadata_json)
       VALUES (?, ?, ?, ?, ?)`
    ).run(input.source_id, input.target_id, input.type, input.weight ?? 1.0, input.metadata_json ?? '{}');
    return Number(result.lastInsertRowid);
  }

  getEdgesFrom(nodeId: number): EdgeRow[] {
    return this.db.prepare('SELECT * FROM edges WHERE source_id = ?').all(nodeId) as EdgeRow[];
  }

  getEdgesTo(nodeId: number): EdgeRow[] {
    return this.db.prepare('SELECT * FROM edges WHERE target_id = ?').all(nodeId) as EdgeRow[];
  }

  deleteEdge(id: number): void {
    this.db.prepare('DELETE FROM edges WHERE id = ?').run(id);
  }

  findOrphans(): NodeRow[] {
    return this.db.prepare(`
      SELECT n.* FROM nodes n
      WHERE n.id NOT IN (SELECT source_id FROM edges)
        AND n.id NOT IN (SELECT target_id FROM edges)
    `).all() as NodeRow[];
  }
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run tests/knowledge/knowledge.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/knowledge.ts tests/knowledge/knowledge.test.ts
git commit -m "feat: knowledge graph with node/edge CRUD and orphan detection"
```

---

### Task 6: Session Tracker

**Files:**
- Create: `src/core/session.ts`
- Test: `tests/core/session.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/core/session.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Brain } from '../../src/core/brain.js';
import { SessionTracker } from '../../src/core/session.js';
import fs from 'fs';
import path from 'path';

const TEST_DIR = '/tmp/one-test-session';
const DB_PATH = path.join(TEST_DIR, 'brain.db');

describe('SessionTracker', () => {
  let brain: Brain;
  let tracker: SessionTracker;

  beforeEach(() => {
    fs.mkdirSync(TEST_DIR, { recursive: true });
    brain = new Brain(DB_PATH);
    tracker = new SessionTracker(brain);
  });

  afterEach(() => {
    brain.close();
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it('starts and ends a session', () => {
    const id = tracker.start(1, 'Initial setup');
    expect(id).toBeGreaterThan(0);
    const session = tracker.get(id);
    expect(session?.status).toBe('running');
    expect(session?.phase).toBe(1);
    tracker.end(id, 'Completed initial setup');
    const ended = tracker.get(id);
    expect(ended?.status).toBe('completed');
  });

  it('detects and recovers interrupted sessions', () => {
    const id = tracker.start(1, 'Will be interrupted');
    // Simulate crash — session left in 'running' state
    const interrupted = tracker.findInterrupted();
    expect(interrupted).toHaveLength(1);
    expect(interrupted[0].id).toBe(id);
    tracker.markInterrupted(id);
    const session = tracker.get(id);
    expect(session?.status).toBe('interrupted');
  });

  it('tracks token usage', () => {
    const id = tracker.start(1, 'Token test');
    tracker.addTokens(id, 5000);
    tracker.addTokens(id, 3000);
    const session = tracker.get(id);
    expect(session?.tokens_used).toBe(8000);
  });
});
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement session.ts**

```typescript
// src/core/session.ts
import type { Brain } from './brain.js';

export interface SessionRow {
  id: number;
  phase: number | null;
  status: string;
  focus: string | null;
  started_at: string;
  ended_at: string | null;
  summary: string | null;
  tokens_used: number;
  agents_spawned: number;
}

export class SessionTracker {
  private db;

  constructor(brain: Brain) {
    this.db = brain.getDb();
  }

  start(phase: number, focus: string): number {
    const result = this.db.prepare(
      `INSERT INTO sessions (phase, status, focus) VALUES (?, 'running', ?)`
    ).run(phase, focus);
    return Number(result.lastInsertRowid);
  }

  end(id: number, summary?: string): void {
    this.db.prepare(
      `UPDATE sessions SET status = 'completed', ended_at = datetime('now'), summary = ? WHERE id = ?`
    ).run(summary ?? null, id);
  }

  get(id: number): SessionRow | undefined {
    return this.db.prepare('SELECT * FROM sessions WHERE id = ?').get(id) as SessionRow | undefined;
  }

  getCurrent(): SessionRow | undefined {
    return this.db.prepare("SELECT * FROM sessions WHERE status = 'running' ORDER BY id DESC LIMIT 1")
      .get() as SessionRow | undefined;
  }

  findInterrupted(): SessionRow[] {
    return this.db.prepare("SELECT * FROM sessions WHERE status = 'running'").all() as SessionRow[];
  }

  markInterrupted(id: number): void {
    this.db.prepare(
      `UPDATE sessions SET status = 'interrupted', ended_at = datetime('now') WHERE id = ?`
    ).run(id);
  }

  addTokens(id: number, tokens: number): void {
    this.db.prepare('UPDATE sessions SET tokens_used = tokens_used + ? WHERE id = ?').run(tokens, id);
  }

  incrementAgents(id: number): void {
    this.db.prepare('UPDATE sessions SET agents_spawned = agents_spawned + 1 WHERE id = ?').run(id);
  }
}
```

- [ ] **Step 4: Run tests → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/core/session.ts tests/core/session.test.ts
git commit -m "feat: session tracker with lifecycle, crash recovery, token tracking"
```

---

## Chunk 3: Topology Engine

### Task 7: TypeScript Parser (tree-sitter)

**Files:**
- Create: `src/topology/parsers/typescript.ts`
- Create: `tests/fixtures/topology/sample.ts`
- Test: `tests/topology/parser.test.ts`

- [ ] **Step 1: Create fixture file**

```typescript
// tests/fixtures/topology/sample.ts
import { Database } from 'better-sqlite3';
import { readFileSync } from 'fs';
import type { Config } from './config';

export interface UserService {
  getUser(id: string): Promise<User>;
  createUser(data: CreateUserInput): Promise<User>;
}

export class AuthService implements UserService {
  private db: Database;

  constructor(db: Database) {
    this.db = db;
  }

  async getUser(id: string): Promise<User> {
    return this.db.prepare('SELECT * FROM users WHERE id = ?').get(id);
  }

  async createUser(data: CreateUserInput): Promise<User> {
    const result = this.db.prepare('INSERT INTO users (name) VALUES (?)').run(data.name);
    return this.getUser(String(result.lastInsertRowid));
  }
}

export function validateToken(token: string): boolean {
  return token.length > 0;
}
```

- [ ] **Step 2: Write failing tests**

```typescript
// tests/topology/parser.test.ts
import { describe, it, expect } from 'vitest';
import { TypeScriptParser } from '../../src/topology/parsers/typescript.js';
import path from 'path';

const FIXTURE = path.join(__dirname, '../fixtures/topology/sample.ts');

describe('TypeScriptParser', () => {
  const parser = new TypeScriptParser();

  it('extracts imports', () => {
    const result = parser.parseFile(FIXTURE);
    expect(result.imports.length).toBeGreaterThanOrEqual(3);
    const sources = result.imports.map(i => i.source);
    expect(sources).toContain('better-sqlite3');
    expect(sources).toContain('fs');
    expect(sources).toContain('./config');
  });

  it('extracts exported functions', () => {
    const result = parser.parseFile(FIXTURE);
    const fns = result.functions.map(f => f.name);
    expect(fns).toContain('validateToken');
  });

  it('extracts classes', () => {
    const result = parser.parseFile(FIXTURE);
    const classes = result.classes.map(c => c.name);
    expect(classes).toContain('AuthService');
  });

  it('extracts class methods', () => {
    const result = parser.parseFile(FIXTURE);
    const authClass = result.classes.find(c => c.name === 'AuthService');
    expect(authClass).toBeDefined();
    const methods = authClass!.methods.map(m => m.name);
    expect(methods).toContain('getUser');
    expect(methods).toContain('createUser');
  });

  it('extracts interfaces', () => {
    const result = parser.parseFile(FIXTURE);
    const ifaces = result.interfaces.map(i => i.name);
    expect(ifaces).toContain('UserService');
  });

  it('extracts exported symbols', () => {
    const result = parser.parseFile(FIXTURE);
    expect(result.exports).toContain('AuthService');
    expect(result.exports).toContain('validateToken');
    expect(result.exports).toContain('UserService');
  });
});
```

- [ ] **Step 3: Run test → FAIL**

- [ ] **Step 4: Implement TypeScript parser**

```typescript
// src/topology/parsers/typescript.ts
import Parser from 'tree-sitter';
import TypeScript from 'tree-sitter-typescript';
import fs from 'fs';

export interface ImportInfo {
  symbols: string[];
  source: string;
  isType: boolean;
}

export interface FunctionInfo {
  name: string;
  signature: string;
  startLine: number;
  endLine: number;
  exported: boolean;
}

export interface ClassInfo {
  name: string;
  startLine: number;
  endLine: number;
  exported: boolean;
  implements: string[];
  methods: FunctionInfo[];
}

export interface InterfaceInfo {
  name: string;
  methods: string[];
  exported: boolean;
}

export interface ParseResult {
  imports: ImportInfo[];
  functions: FunctionInfo[];
  classes: ClassInfo[];
  interfaces: InterfaceInfo[];
  exports: string[];
}

export class TypeScriptParser {
  private parser: Parser;

  constructor() {
    this.parser = new Parser();
    this.parser.setLanguage(TypeScript.typescript);
  }

  parseFile(filePath: string): ParseResult {
    const source = fs.readFileSync(filePath, 'utf-8');
    return this.parseSource(source);
  }

  parseSource(source: string): ParseResult {
    const tree = this.parser.parse(source);
    const root = tree.rootNode;

    const imports: ImportInfo[] = [];
    const functions: FunctionInfo[] = [];
    const classes: ClassInfo[] = [];
    const interfaces: InterfaceInfo[] = [];
    const exports: string[] = [];

    for (const child of root.children) {
      if (child.type === 'import_statement') {
        imports.push(this.extractImport(child));
      } else if (child.type === 'export_statement') {
        const decl = child.namedChildren.find(c =>
          ['function_declaration', 'class_declaration', 'interface_declaration',
           'lexical_declaration', 'type_alias_declaration'].includes(c.type)
        );
        if (decl) {
          if (decl.type === 'function_declaration') {
            const fn = this.extractFunction(decl, true);
            if (fn) { functions.push(fn); exports.push(fn.name); }
          } else if (decl.type === 'class_declaration') {
            const cls = this.extractClass(decl, true);
            if (cls) { classes.push(cls); exports.push(cls.name); }
          } else if (decl.type === 'interface_declaration') {
            const iface = this.extractInterface(decl, true);
            if (iface) { interfaces.push(iface); exports.push(iface.name); }
          }
        }
      } else if (child.type === 'function_declaration') {
        const fn = this.extractFunction(child, false);
        if (fn) functions.push(fn);
      } else if (child.type === 'class_declaration') {
        const cls = this.extractClass(child, false);
        if (cls) classes.push(cls);
      } else if (child.type === 'interface_declaration') {
        const iface = this.extractInterface(child, false);
        if (iface) interfaces.push(iface);
      }
    }

    return { imports, functions, classes, interfaces, exports };
  }

  private extractImport(node: Parser.SyntaxNode): ImportInfo {
    const source = node.descendantsOfType('string')[0]?.text?.replace(/['"]/g, '') ?? '';
    const isType = node.text.includes('import type');
    const symbols: string[] = [];
    const clause = node.descendantsOfType('import_clause')[0];
    if (clause) {
      for (const spec of clause.descendantsOfType('import_specifier')) {
        symbols.push(spec.childForFieldName('name')?.text ?? spec.text);
      }
      const named = clause.childForFieldName('name');
      if (named) symbols.push(named.text);
    }
    return { symbols, source, isType };
  }

  private extractFunction(node: Parser.SyntaxNode, exported: boolean): FunctionInfo | null {
    const name = node.childForFieldName('name')?.text;
    if (!name) return null;
    return {
      name,
      signature: node.text.split('{')[0].trim(),
      startLine: node.startPosition.row + 1,
      endLine: node.endPosition.row + 1,
      exported,
    };
  }

  private extractClass(node: Parser.SyntaxNode, exported: boolean): ClassInfo | null {
    const name = node.childForFieldName('name')?.text;
    if (!name) return null;
    const implementsClause = node.descendantsOfType('implements_clause')[0];
    const implementsList: string[] = [];
    if (implementsClause) {
      for (const type of implementsClause.descendantsOfType('type_identifier')) {
        implementsList.push(type.text);
      }
    }
    const body = node.childForFieldName('body');
    const methods: FunctionInfo[] = [];
    if (body) {
      for (const member of body.namedChildren) {
        if (member.type === 'method_definition') {
          const mName = member.childForFieldName('name')?.text;
          if (mName) {
            methods.push({
              name: mName,
              signature: member.text.split('{')[0].trim(),
              startLine: member.startPosition.row + 1,
              endLine: member.endPosition.row + 1,
              exported: false,
            });
          }
        }
      }
    }
    return { name, startLine: node.startPosition.row + 1, endLine: node.endPosition.row + 1, exported, implements: implementsList, methods };
  }

  private extractInterface(node: Parser.SyntaxNode, exported: boolean): InterfaceInfo | null {
    const name = node.childForFieldName('name')?.text;
    if (!name) return null;
    const body = node.childForFieldName('body');
    const methods: string[] = [];
    if (body) {
      for (const member of body.namedChildren) {
        if (member.type === 'method_signature') {
          const mName = member.childForFieldName('name')?.text;
          if (mName) methods.push(mName);
        }
      }
    }
    return { name, methods, exported };
  }
}
```

- [ ] **Step 5: Run tests → PASS**

- [ ] **Step 6: Commit**

```bash
git add src/topology/parsers/typescript.ts tests/topology/parser.test.ts tests/fixtures/topology/sample.ts
git commit -m "feat: tree-sitter TypeScript parser — imports, functions, classes, interfaces"
```

---

### Task 8: Topology Engine — Scan, Parse, Store

**Files:**
- Create: `src/topology/engine.ts`
- Test: `tests/topology/engine.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
// tests/topology/engine.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Brain } from '../../src/core/brain.js';
import { KnowledgeGraph } from '../../src/core/knowledge.js';
import { TopologyEngine } from '../../src/topology/engine.js';
import fs from 'fs';
import path from 'path';

const TEST_DIR = '/tmp/one-test-topology';
const DB_PATH = path.join(TEST_DIR, '.one', 'brain.db');
const PROJECT_DIR = path.join(TEST_DIR, 'project');

describe('TopologyEngine', () => {
  let brain: Brain;
  let kg: KnowledgeGraph;
  let engine: TopologyEngine;

  beforeEach(() => {
    fs.mkdirSync(path.join(TEST_DIR, '.one'), { recursive: true });
    fs.mkdirSync(path.join(PROJECT_DIR, 'src'), { recursive: true });
    brain = new Brain(DB_PATH);
    kg = new KnowledgeGraph(brain);
    engine = new TopologyEngine(brain, kg, PROJECT_DIR);

    // Write test files
    fs.writeFileSync(path.join(PROJECT_DIR, 'src', 'index.ts'),
      `import { helper } from './helper';\nexport function main() { helper(); }\n`);
    fs.writeFileSync(path.join(PROJECT_DIR, 'src', 'helper.ts'),
      `export function helper() { return 42; }\n`);
  });

  afterEach(() => {
    brain.close();
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it('scans project and creates file nodes', () => {
    engine.fullScan();
    const files = kg.getNodesByType('file');
    expect(files.length).toBeGreaterThanOrEqual(2);
  });

  it('creates function nodes linked to files', () => {
    engine.fullScan();
    const functions = kg.getNodesByType('function');
    const fnNames = functions.map(f => f.name);
    expect(fnNames).toContain('main');
    expect(fnNames).toContain('helper');
  });

  it('creates import edges between files', () => {
    engine.fullScan();
    const files = kg.getNodesByType('file');
    const indexNode = files.find(f => f.name.includes('index.ts'));
    expect(indexNode).toBeDefined();
    const edges = kg.getEdgesFrom(indexNode!.id);
    const importEdges = edges.filter(e => e.type === 'imports');
    expect(importEdges.length).toBeGreaterThanOrEqual(1);
  });

  it('incremental reparse updates changed files only', () => {
    engine.fullScan();
    const before = kg.getNodesByType('function').length;
    fs.writeFileSync(path.join(PROJECT_DIR, 'src', 'helper.ts'),
      `export function helper() { return 42; }\nexport function newFn() { return 1; }\n`);
    engine.reparseFile(path.join(PROJECT_DIR, 'src', 'helper.ts'));
    const after = kg.getNodesByType('function').length;
    expect(after).toBe(before + 1);
  });
});
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement engine.ts**

The topology engine walks the project directory, parses TypeScript files with the tree-sitter parser, and stores results as nodes/edges in the knowledge graph. It handles:
- Full scan: walk all files, parse, store
- Incremental reparse: reparse single file, update graph
- File hashing for change detection
- Import resolution (relative paths → file nodes)

Implementation should use `fs.readdirSync` recursive walk, skip `node_modules`/`.git`/`.one`/`dist`, hash files with `crypto.createHash('sha256')`, and delegate parsing to `TypeScriptParser`.

- [ ] **Step 4: Run tests → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/topology/engine.ts tests/topology/engine.test.ts
git commit -m "feat: topology engine with full scan, incremental reparse, import resolution"
```

---

### Task 9: Integrity Checker

**Files:**
- Create: `src/topology/integrity.ts`
- Test: `tests/topology/integrity.test.ts`

- [ ] **Step 1: Write failing tests**

Tests should verify detection of:
- Dangling imports (import from file that doesn't exist)
- Orphan files (no imports to or from)
- Stub functions (declared but empty body)
- Unimplemented interface methods

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement integrity.ts**

Queries the knowledge graph for:
- Files with unresolved imports (`imports` table where `resolved = 0`)
- Nodes with no edges (orphans, excluding goal/decision/research types)
- Functions with body hash indicating empty/stub

Returns array of `Violation` objects: `{ type, severity, source, target, message }`.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/topology/integrity.ts tests/topology/integrity.test.ts
git commit -m "feat: integrity checker — dangling imports, orphans, stubs"
```

---

### Task 10: Topology Differ

**Files:**
- Create: `src/topology/differ.ts`
- Test: `tests/topology/differ.test.ts`

- [ ] **Step 1: Write failing tests**

Tests verify differ detects:
- New files added
- Files removed
- Functions added/removed/modified (body hash changed)

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement differ.ts**

Compares current file hashes and function body hashes against stored values. Returns `Diff[]` with `{ type: 'added'|'removed'|'modified', entity: 'file'|'function', name, path }`.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/topology/differ.ts tests/topology/differ.test.ts
git commit -m "feat: topology differ — detect file/function changes between scans"
```

---

## Chunk 4: Agent Management & Error Intelligence

### Task 11: Claude Code Process Manager

**Files:**
- Create: `src/agents/claude.ts`
- Test: `tests/agents/claude.test.ts`

- [ ] **Step 1: Write failing tests**

Test spawning a mock process (use `echo` or a fixture script that emits stream-json), verifying:
- Process spawns with correct flags
- Stream-json output is parsed
- Process can be killed
- PID is tracked

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement claude.ts**

Uses `child_process.spawn` to launch `claude -p --output-format stream-json --input-format stream-json --system-prompt <briefing>`. Pipes stdin/stdout. Parses NDJSON output events. Emits typed events via EventEmitter: `text`, `tool_use`, `tool_result`, `error`, `done`.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/agents/claude.ts tests/agents/claude.test.ts
git commit -m "feat: Claude Code process manager — spawn, stream-json parse, lifecycle"
```

---

### Task 12: Agent Watcher — Stream Monitor & Pattern Capture

**Files:**
- Create: `src/agents/watcher.ts`
- Create: `tests/fixtures/stream-json/sample-session.jsonl` (fixture with tool calls, errors, learnings)
- Test: `tests/agents/watcher.test.ts`

- [ ] **Step 1: Create stream-json fixture**

Write a sample JSONL file representing a Claude Code session with: text output, Edit tool calls, an error + retry + success sequence, and a learning block.

- [ ] **Step 2: Write failing tests**

Using fixture stream-json data, verify:
- Detects file-modifying tool calls (Edit, Write)
- Extracts error messages from tool results
- Captures `{"type": "learning", ...}` blocks from agent output
- Tracks token usage from stream events

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement watcher.ts**

Wraps a `ClaudeProcess`, subscribes to events, and:
- Logs all actions to `action_log` table
- Detects errors and successful fixes → stores as patterns
- Parses learning blocks → stores as pattern nodes
- Triggers topology reparse on file-modifying tool calls
- Accumulates token counts

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/agents/watcher.ts tests/agents/watcher.test.ts
git commit -m "feat: agent watcher — stream monitoring, error capture, pattern extraction"
```

---

### Task 13: Error Intelligence — Pattern Storage & Briefing Generation

**Files:**
- Create: `src/intelligence/patterns.ts`
- Create: `src/agents/briefing.ts`
- Test: `tests/intelligence/patterns.test.ts`
- Test: `tests/intelligence/briefing.test.ts`

- [ ] **Step 1: Write failing tests for patterns**

Test pattern CRUD: store error→fix pair, increment confidence on re-encounter, query patterns by related files.

- [ ] **Step 2: Write failing tests for briefing**

Test briefing generation: given a task and topology context, generates markdown briefing string including relevant patterns, known errors, anti-patterns, completed work, dangling connections.

- [ ] **Step 3: Run → FAIL**

- [ ] **Step 4: Implement patterns.ts and briefing.ts**

`patterns.ts` — CRUD for `patterns` table. Methods: `addPattern`, `findByTrigger`, `incrementSeen`, `incrementWorked`, `getByRelatedFiles`.

`briefing.ts` — Queries knowledge graph for task context, topology subgraph, patterns, and formats into the briefing template from the spec.

- [ ] **Step 5: Run → PASS**

- [ ] **Step 6: Commit**

```bash
git add src/intelligence/patterns.ts src/agents/briefing.ts tests/intelligence/
git commit -m "feat: error intelligence — pattern storage, confidence tracking, agent briefing generation"
```

---

### Task 14: Planner — Spec to Task Graph

**Files:**
- Create: `src/intelligence/planner.ts`
- Test: `tests/intelligence/planner.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Creates tasks from a list of components in the knowledge graph
- Sets up blocked_by relationships based on component dependencies
- Returns topologically sorted execution order
- Detects cycles and errors

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement planner.ts**

Reads component and dependency nodes from knowledge graph. For each component, creates a task. For each `depends_on` edge, sets `blocked_by`. Topological sort for execution order (Kahn's algorithm). Stores tasks in `tasks` table.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/intelligence/planner.ts tests/intelligence/planner.test.ts
git commit -m "feat: planner — knowledge graph to task graph with dependency ordering"
```

---

### Task 15: Agent Orchestrator — Task Execution Flow

**Files:**
- Create: `src/agents/orchestrator.ts`
- Test: `tests/agents/orchestrator.test.ts`

- [ ] **Step 1: Write failing tests**

Test single-agent sequential task execution:
- Picks next pending task (respecting blocked_by)
- Spawns agent with briefing
- On task complete: runs integrity check
- On integrity pass: marks task complete, picks next
- On integrity fail: re-sends agent to fix
- On agent stuck (3+ same error): kills, spawns new with updated briefing
- Tracks cost and stops at budget limit

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement orchestrator.ts**

State machine per task: `pending → assigned → running → verifying → complete|failed`.
Uses `ClaudeProcess`, `AgentWatcher`, `TopologyEngine.reparseFile`, `IntegrityChecker`, `BriefingGenerator`. Reports progress via callback (used by Telegram later).

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/agents/orchestrator.ts tests/agents/orchestrator.test.ts
git commit -m "feat: agent orchestrator — sequential task execution, verification, failure handling"
```

---

## Chunk 5: Automation & Cost Controls

### Task 16: Hook System

**Files:**
- Create: `src/automation/hooks.ts`
- Test: `tests/automation/hooks.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Fires hook by name, passes JSON to stdin of child process
- Blocking hooks (on_task_complete, on_integrity_fail) return exit code
- Non-existent/null hooks are silently skipped
- Hook timeout (5s default)

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement hooks.ts**

Reads hooks config, spawns child process with `shell: true`, pipes JSON to stdin, captures exit code. Blocking hooks await completion; fire-and-forget hooks don't.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/automation/hooks.ts tests/automation/hooks.test.ts
git commit -m "feat: hook system — lifecycle event hooks with blocking/fire-and-forget modes"
```

---

### Task 17: Scheduler

**Files:**
- Create: `src/automation/scheduler.ts`
- Test: `tests/automation/scheduler.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Parses cron expressions
- Schedules and fires tasks at the right time (use fake timers)
- Runs integrity check on schedule
- Stops cleanly

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement scheduler.ts**

Simple cron scheduler using `setInterval` with minute-level granularity. Parses cron expressions for the three built-in schedules. Calls registered callbacks (integrity check, full reparse, brain backup).

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/automation/scheduler.ts tests/automation/scheduler.test.ts
git commit -m "feat: scheduler — cron-like recurring tasks for integrity checks and reparses"
```

---

### Task 17a: Cost Control System

**Files:**
- Create: `src/core/costs.ts`
- Test: `tests/core/costs.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Tracks daily token usage across agents (aggregates from `agents` table)
- Returns percentage of daily budget consumed
- Triggers warning callback at 80% threshold
- Triggers hard stop callback at 100% threshold
- Estimates task cost based on task count and average tokens per task
- Handles unlimited budget (daily_token_budget = 0) — no warnings

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement costs.ts**

`CostController` class. Methods: `getDailyUsage()`, `checkBudget()` returns `{ok, percentage, remaining}`, `estimateTaskCost(taskCount)`. Takes callbacks for `onWarning` and `onBudgetExhausted` (used by Telegram later). Queries `agents` table for today's token totals.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/core/costs.ts tests/core/costs.test.ts
git commit -m "feat: cost control system — budget tracking, 80%/100% thresholds, estimation"
```

---

### Task 17b: Automation-Aware Topology Parsers

**Files:**
- Create: `src/topology/parsers/config-parsers.ts`
- Test: `tests/topology/config-parsers.test.ts`
- Create: `tests/fixtures/topology/package.json`
- Create: `tests/fixtures/topology/ci-workflow.yml`
- Create: `tests/fixtures/topology/Dockerfile`

- [ ] **Step 1: Create fixture files**

Create sample `package.json` with scripts, a `.github/workflows/ci.yml`, and a `Dockerfile` for testing.

- [ ] **Step 2: Write failing tests**

Test:
- Parses `package.json` scripts into script nodes with edges to referenced files
- Parses GitHub Actions YAML into CI nodes with edges to referenced scripts/commands
- Parses Dockerfile into infrastructure nodes with edges to copied files
- Returns empty results for non-existent files (graceful)

- [ ] **Step 3: Run → FAIL**

- [ ] **Step 4: Implement config-parsers.ts**

Simple parsers using JSON.parse (package.json), a YAML parser (js-yaml — add as dependency), and regex-based extraction for Dockerfile COPY/ADD commands. Each returns nodes and edges compatible with the knowledge graph. No tree-sitter needed — these are config files.

- [ ] **Step 5: Run → PASS**

- [ ] **Step 6: Commit**

```bash
git add src/topology/parsers/config-parsers.ts tests/topology/config-parsers.test.ts tests/fixtures/topology/
git commit -m "feat: automation-aware topology — parse package.json scripts, GitHub Actions, Dockerfile"
```

---

### Task 17c: Phase State Machine

**Files:**
- Create: `src/core/phase-machine.ts`
- Test: `tests/core/phase-machine.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Starts at Phase 1
- Allows forward transitions: 1→2, 2→3, 3→4
- Allows backward transitions: 4→2, 3→2, any→1
- Rejects invalid transitions (e.g., 1→4 direct)
- Stores current phase in session
- Preserves knowledge graph on backward transitions
- `requestScopeChange()` returns to Phase 1 from any phase

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement phase-machine.ts**

`PhaseMachine` class with `currentPhase`, `transition(to)` with validation, `canTransition(to)` predicate. Emits events on transition (consumed by hooks and Telegram). Stores phase in current session.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/core/phase-machine.ts tests/core/phase-machine.test.ts
git commit -m "feat: phase state machine — forward/backward transitions, scope change support"
```

---

### Task 17d: Log Rotation & Global Config

**Files:**
- Modify: `src/core/logger.ts`
- Create: `src/core/global-config.ts`
- Test: `tests/core/logger-rotation.test.ts`
- Test: `tests/core/global-config.test.ts`

- [ ] **Step 1: Write failing tests for log rotation**

Test:
- `cleanOldLogs(maxAgeDays)` deletes log files older than N days
- Keeps recent logs untouched
- Handles empty log directory gracefully

- [ ] **Step 2: Write failing tests for global config**

Test:
- Loads from `~/.one/global.json` (use temp dir in tests)
- Falls back to defaults when file doesn't exist
- Merges global → project config (project overrides global)
- Stores default model, default budget, shared Telegram token

- [ ] **Step 3: Run → FAIL**

- [ ] **Step 4: Add `cleanOldLogs` method to Logger, implement GlobalConfig**

`Logger.cleanOldLogs(maxAgeDays)` scans log dir, checks file dates, removes old ones.
`GlobalConfig` loads from `~/.one/global.json`, same pattern as project Config.

- [ ] **Step 5: Run → PASS**

- [ ] **Step 6: Commit**

```bash
git add src/core/logger.ts src/core/global-config.ts tests/core/logger-rotation.test.ts tests/core/global-config.test.ts
git commit -m "feat: log rotation (7-day auto-delete) and global config (~/.one/global.json)"
```

---

### Task 17e: Claude Code Version Check

**Files:**
- Create: `src/core/constants.ts`
- Create: `src/core/compat.ts`
- Test: `tests/core/compat.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Parses `claude --version` output to extract version number
- Compares against minimum supported version
- Returns `{compatible, currentVersion, minVersion}` result
- Handles missing `claude` binary gracefully (returns incompatible with error message)

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement constants.ts and compat.ts**

`constants.ts` exports `MIN_CLAUDE_CODE_VERSION = '1.0.0'`.
`compat.ts` exports `checkClaudeCodeVersion()` — runs `claude --version` via `execSync`, parses output, compares semver.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/core/constants.ts src/core/compat.ts tests/core/compat.test.ts
git commit -m "feat: Claude Code version compatibility check"
```

---

## Chunk 6: Telegram Interface

### Task 18: Telegram Bot Setup & Auth

**Files:**
- Create: `src/telegram/bot.ts`
- Test: `tests/telegram/bot.test.ts`

- [ ] **Step 1: Write failing tests**

Test (with mocked grammy):
- Bot initializes with token from brain.db
- First message pins chat_id
- Subsequent messages from wrong chat_id are dropped
- Messages route to correct phase handler

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement bot.ts**

Uses grammy `Bot` class. Middleware checks `chat_id` authorization. Routes messages to phase-specific handlers based on current session phase. Stores messages in `telegram_messages` table.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/telegram/bot.ts tests/telegram/bot.test.ts
git commit -m "feat: Telegram bot — setup, chat_id authorization, phase routing"
```

---

### Task 19: Message Formatter & Inline Buttons

**Files:**
- Create: `src/telegram/formatter.ts`
- Test: `tests/telegram/formatter.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Formats status report with topology stats
- Creates inline keyboard for choices (A/B/C/D)
- Creates checkpoint buttons (Continue/Review/Pause/Change)
- Formats error report
- Truncates messages >4096 chars (Telegram limit)

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement formatter.ts**

Pure functions that take data and return grammy-compatible message objects with `reply_markup` for inline keyboards.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/telegram/formatter.ts tests/telegram/formatter.test.ts
git commit -m "feat: Telegram message formatter — status reports, inline buttons, truncation"
```

---

## Chunk 7: Phase Logic, Handlers & CLI

### Task 21: Phase 1 — Understand

**Files:**
- Create: `src/phases/understand.ts`
- Test: `tests/phases/understand.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Initializes with empty unknowns list (purpose, users, scope, constraints, criteria)
- After user answers a question, corresponding unknown is resolved
- Generates knowledge graph nodes for each answered unknown
- Proposes transition to Phase 2 when only research-type unknowns remain
- Can return to Phase 1 from other phases (preserves existing graph)

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement understand.ts**

Manages the unknowns list. Spawns a Claude Code agent with system prompt containing current knowledge graph state + unknowns. Parses agent response, updates knowledge graph, recalculates unknowns. Determines when to propose Phase 2 transition.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/phases/understand.ts tests/phases/understand.test.ts
git commit -m "feat: Phase 1 understand — structured conversation, unknown tracking, knowledge graph updates"
```

---

### Task 22: Phase 2 — Research

**Files:**
- Create: `src/phases/research.ts`
- Test: `tests/phases/research.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Spawns research agent with correct system prompt
- Parses research findings into knowledge graph nodes (type=research)
- Creates `informed_by` edges between decisions and research
- Creates `contradicts` edges for anti-thesis findings
- Reports findings back to user via Telegram
- Supports "dig deeper on X" follow-up

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement research.ts**

Spawns a Claude Code agent with web search capability, system prompt instructing competitive analysis, anti-thesis exploration. Parses structured output into research nodes. Links to relevant decision/component nodes.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/phases/research.ts tests/phases/research.test.ts
git commit -m "feat: Phase 2 research — agent-driven competitive analysis, anti-thesis, knowledge graph linking"
```

---

### Task 23: Phase 3 — Spec

**Files:**
- Create: `src/phases/spec.ts`
- Test: `tests/phases/spec.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Generates spec sections from knowledge graph
- Presents sections one at a time
- On approval: marks section as approved, moves to next
- On revision: re-generates section with feedback
- After all sections approved: generates task plan via Planner
- Writes spec document to project directory

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement spec.ts**

Uses Claude Code agent with full knowledge graph in system prompt. Iterates through component groups. Captures user approval per section. On completion, calls Planner to generate task graph.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/phases/spec.ts tests/phases/spec.test.ts
git commit -m "feat: Phase 3 spec — section-by-section design, approval flow, task plan generation"
```

---

### Task 24: Phase 4 — Build

**Files:**
- Create: `src/phases/build.ts`
- Test: `tests/phases/build.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Asks user for execution mode (autonomy/checkpoints)
- Delegates to orchestrator for task execution
- Sends checkpoint reports via Telegram at milestones
- Handles user responses to checkpoints (continue/pause/change)
- On project complete: sends final report with topology status
- Triggers Phase 2 backward transition on unexpected blocker

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement build.ts**

Thin coordination layer between orchestrator, Telegram, and cost controls. Manages the execution loop: orchestrator runs tasks → reports progress → checks budget → handles user input.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/phases/build.ts tests/phases/build.test.ts
git commit -m "feat: Phase 4 build — autonomous execution with checkpoints, cost controls, backward transitions"
```

---

### Task 24a: Telegram Phase Handlers

**Files:**
- Create: `src/telegram/handlers.ts`
- Test: `tests/telegram/handlers.test.ts`

- [ ] **Step 1: Write failing tests**

Test each phase handler (now that phase logic exists to delegate to):
- Phase 1: receives user message, forwards to understand module, returns response
- Phase 2: receives research results, formats and sends to user
- Phase 3: presents spec section, handles approve/revise buttons
- Phase 4: handles checkpoint buttons (Continue/Pause/Change Plan)
- Handles `remember:` messages → stores as pattern
- Handles "wait, I want to change the scope" → triggers phase machine backward transition

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement handlers.ts**

Each handler function takes the Telegram context + One's brain state, delegates to the appropriate phase module, and sends formatted responses via the formatter.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/telegram/handlers.ts tests/telegram/handlers.test.ts
git commit -m "feat: Telegram phase handlers — message routing to phase modules"
```

---

### Task 24b: Existing Project Support

**Files:**
- Modify: `src/cli/index.ts` (init flow)
- Modify: `src/telegram/handlers.ts` (existing project UX)
- Test: `tests/cli/existing-project.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- Detects existing files in project directory (non-empty dir with source files)
- Runs topology scan on existing codebase
- Reports stats: file count, languages, dangling imports, test coverage
- Presents action buttons: [Add a feature] [Fix something] [Refactor] [Just explore]
- Enters Phase 1 with pre-populated knowledge graph (File/Function nodes from scan)

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement existing project flow**

On `one` init in non-empty directory: run `TopologyEngine.fullScan()`, query integrity checker for stats, format stats message, send via Telegram with action buttons. On button selection, enter Phase 1 with context.

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/cli/index.ts src/telegram/handlers.ts tests/cli/existing-project.test.ts
git commit -m "feat: existing project support — scan, stats, action selection"
```

---

### Task 25: CLI — Init, Stop, Status, Logs

**Files:**
- Create: `src/cli/index.ts`
- Test: `tests/cli/cli.test.ts`

- [ ] **Step 1: Write failing tests**

Test:
- `one` in empty dir → prompts for Telegram token, creates `.one/`, starts service
- `one` in existing project → detects files, scans topology, connects to Telegram
- `one status` → prints project state from brain.db
- `one stop` → kills pm2 process
- `one logs` → reads today's log file

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement cli/index.ts**

Uses Commander.js. Commands:
- default (no subcommand): init flow — create `.one/`, prompt for Telegram token (via readline), set up brain.db, scan existing files, start pm2 service
- `status`: open brain.db, query sessions/tasks/topology stats, print
- `stop`: `pm2 stop one`
- `logs`: read and format today's JSONL log
- `reset-telegram`: clear telegram_config, re-prompt

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/cli/index.ts tests/cli/cli.test.ts
git commit -m "feat: CLI — init, status, stop, logs, reset-telegram commands"
```

---

## Chunk 8: Integration & E2E

### Task 26: Main Service Entry Point

**Files:**
- Create: `src/service.ts`
- Test: `tests/core/service.test.ts`

- [ ] **Step 1: Write failing tests**

Test (with mocked subsystems):
- Startup initializes brain, config, logger, telegram bot, scheduler
- Detects interrupted sessions on startup and runs recovery
- Checks Claude Code version and logs warning if incompatible
- Runs log rotation on startup (clean old logs)
- Resumes Phase 4 orchestrator if session was in build phase
- Shuts down cleanly: stops scheduler, closes brain, kills agents

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement service.ts**

The main service that pm2 runs. On start:
1. Load config (project + global)
2. Open brain.db, run migrations
3. Check Claude Code version
4. Clean old logs
5. Check for interrupted sessions → recover (mark interrupted, kill orphaned PIDs, full reparse)
6. Start Telegram bot
7. Start scheduler
8. If existing session in Phase 4: resume orchestrator
9. Otherwise: wait for Telegram messages

- [ ] **Step 4: Run → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/service.ts tests/core/service.test.ts
git commit -m "feat: main service entry point — startup, recovery, version check, scheduler"
```

---

### Task 27: End-to-End Smoke Test

**Files:**
- Create: `tests/e2e/smoke.test.ts`

- [ ] **Step 1: Write E2E test**

Tiny project flow with all mocks:
1. Init brain.db
2. Start Phase 1 with mock user messages
3. Transition to Phase 2 with mock research agent
4. Transition to Phase 3, approve all sections
5. Transition to Phase 4 with mock build agent
6. Verify topology graph has expected nodes/edges
7. Verify patterns were captured
8. Verify session history is correct

- [ ] **Step 2: Run → PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/smoke.test.ts
git commit -m "test: end-to-end smoke test — full lifecycle with mocked agents"
```

---

### Task 28: Final Wiring & Polish

- [ ] **Step 1: Ensure all tests pass**

Run: `npx vitest run`
Expected: All tests pass

- [ ] **Step 2: Build TypeScript**

Run: `npm run build`
Expected: Clean build, no errors

- [ ] **Step 3: Add .gitignore entry for .one/**

Verify `.one/` is in the project `.gitignore` for target projects (handled by init).

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: One v1 — autonomous project execution engine"
```
