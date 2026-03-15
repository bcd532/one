"""Executable Verification Engine — Hypotheses that can be tested MUST be tested.

System 7: Experiment design, execution, measurement, comparison.

When a hypothesis involves testable claims, this engine:
1. Detects testability
2. Designs the experiment
3. Executes it
4. Measures quantitative results
5. Compares to baseline
6. Stores full experiment record with reproducibility info

Experiment types:
- Code experiments: modify parameter, run tests, measure metrics
- Data experiments: query dataset, compute statistics, verify claims
- Mathematical experiments: compute proof, verify identity, test conjecture
- Simulation experiments: model system, run scenarios, compare outcomes
"""

import sqlite3
import subprocess
import threading
import json
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import push_memory, DB_PATH, DB_DIR
from .gemma import _call_ollama


_local = threading.local()

EXPERIMENT_TYPES = ["code", "data", "mathematical", "simulation"]

EXPERIMENT_STATUSES = ["designed", "running", "passed", "failed", "error", "inconclusive"]


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        import os
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
    return _local.conn


def init_schema() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            experiment_type TEXT DEFAULT 'code',
            plan TEXT,
            baseline TEXT,
            result TEXT,
            delta TEXT,
            status TEXT DEFAULT 'designed',
            confidence REAL DEFAULT 0.5,
            reproducible INTEGER DEFAULT 0,
            created TEXT NOT NULL,
            executed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project);
        CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments(experiment_type);
    """)
    conn.commit()


# ── Prompts ─────────────────────────────────────────────────────

TESTABILITY_PROMPT = """Assess whether this hypothesis is testable with available tools.

HYPOTHESIS: {hypothesis}

A hypothesis is testable if:
1. It makes a specific, measurable claim
2. There's a way to run a test, compute a value, or compare outcomes
3. We can define what "pass" and "fail" look like

Respond with:
TESTABLE: [YES|NO]
TYPE: [code|data|mathematical|simulation]
REASONING: [why it is or isn't testable]
MEASUREMENT: [what we would measure, if testable]"""

DESIGN_PROMPT = """Design an experiment to test this hypothesis.

HYPOTHESIS: {hypothesis}
EXPERIMENT TYPE: {experiment_type}

Design a concrete, reproducible experiment:
1. What is the independent variable (what we change)?
2. What is the dependent variable (what we measure)?
3. What is the baseline (control condition)?
4. What is the test procedure?
5. What constitutes pass vs fail?

Respond with:
INDEPENDENT_VARIABLE: [what we change]
DEPENDENT_VARIABLE: [what we measure]
BASELINE: [control condition or expected value]
PROCEDURE: [step-by-step test procedure]
PASS_CRITERION: [what result means hypothesis is supported]
FAIL_CRITERION: [what result means hypothesis is refuted]
COMMAND: [if code experiment, the shell command to run; otherwise N/A]"""


# ── Experiment Engine ──────────────────────────────────────────

class ExperimentEngine:
    """Designs and executes experiments to test hypotheses."""

    def __init__(self, project: str, on_log: Optional[Callable] = None):
        self.project = project
        self._log = on_log or (lambda _m: None)
        init_schema()

    def is_testable(self, hypothesis: str) -> dict:
        """Determine if a hypothesis can be tested. Returns testability + type."""
        self._log(f"assessing testability: {hypothesis[:60]}...")

        prompt = TESTABILITY_PROMPT.format(hypothesis=hypothesis)
        result = _call_ollama(prompt, timeout=60)

        testable = False
        experiment_type = "code"
        reasoning = ""
        measurement = ""

        if result:
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("TESTABLE:"):
                    val = line.split(":", 1)[1].strip().upper()
                    testable = val == "YES"
                elif line.startswith("TYPE:"):
                    t = line.split(":", 1)[1].strip().lower()
                    if t in EXPERIMENT_TYPES:
                        experiment_type = t
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("MEASUREMENT:"):
                    measurement = line.split(":", 1)[1].strip()

        return {
            "testable": testable,
            "experiment_type": experiment_type,
            "reasoning": reasoning,
            "measurement": measurement,
        }

    def design_experiment(self, hypothesis: str, experiment_type: str = "code") -> dict:
        """Design an experiment plan for a hypothesis."""
        self._log(f"designing experiment: {hypothesis[:60]}...")

        prompt = DESIGN_PROMPT.format(
            hypothesis=hypothesis,
            experiment_type=experiment_type,
        )
        result = _call_ollama(prompt, timeout=90)

        plan: dict[str, str] = {
            "independent_variable": "",
            "dependent_variable": "",
            "baseline": "",
            "procedure": "",
            "pass_criterion": "",
            "fail_criterion": "",
            "command": "",
        }

        if result:
            for line in result.split("\n"):
                line = line.strip()
                for key in [
                    "INDEPENDENT_VARIABLE", "DEPENDENT_VARIABLE", "BASELINE",
                    "PROCEDURE", "PASS_CRITERION", "FAIL_CRITERION", "COMMAND",
                ]:
                    if line.startswith(f"{key}:"):
                        plan[key.lower()] = line.split(":", 1)[1].strip()

        # Store designed experiment
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """INSERT INTO experiments
               (project, hypothesis, experiment_type, plan, baseline, status, created)
               VALUES (?, ?, ?, ?, ?, 'designed', ?)""",
            (self.project, hypothesis[:1000], experiment_type,
             json.dumps(plan), plan.get("baseline", ""), now),
        )
        conn.commit()

        experiment_id = cur.lastrowid or 0
        self._log(f"experiment {experiment_id} designed")

        return {
            "experiment_id": experiment_id,
            "hypothesis": hypothesis,
            "experiment_type": experiment_type,
            "plan": plan,
        }

    def run_experiment(self, plan: dict) -> dict:
        """Execute an experiment and capture results.

        For code experiments, runs the shell command.
        For other types, uses LLM to reason about the outcome.
        """
        experiment_id = plan.get("experiment_id", 0)
        experiment_type = plan.get("experiment_type", "code")
        hypothesis = plan.get("hypothesis", "")
        experiment_plan = plan.get("plan", {})

        self._log(f"running experiment {experiment_id}...")

        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()

        # Mark as running
        if experiment_id:
            conn.execute(
                "UPDATE experiments SET status = 'running' WHERE id = ?",
                (experiment_id,),
            )
            conn.commit()

        result_data: dict[str, object] = {
            "output": "",
            "exit_code": -1,
            "measurement": "",
            "passed": False,
        }

        try:
            if experiment_type == "code" and isinstance(experiment_plan, dict):
                command = experiment_plan.get("command", "")
                if command and command != "N/A":
                    result_data = self._run_code_experiment(command)
                else:
                    result_data = self._run_llm_experiment(hypothesis, experiment_plan)
            else:
                result_data = self._run_llm_experiment(hypothesis, experiment_plan)

        except Exception as exc:
            result_data = {
                "output": str(exc),
                "exit_code": -1,
                "measurement": "",
                "passed": False,
                "error": str(exc),
            }

        # Determine status
        passed = bool(result_data.get("passed", False))
        status = "passed" if passed else "failed"
        if result_data.get("error"):
            status = "error"

        # Update experiment record
        if experiment_id:
            conn.execute(
                """UPDATE experiments
                   SET status = ?, result = ?, executed_at = ?,
                       confidence = ?
                   WHERE id = ?""",
                (status, json.dumps(result_data, default=str), now,
                 0.8 if passed else 0.3, experiment_id),
            )
            conn.commit()

        # Store result as memory
        verdict = "PASSED" if passed else "FAILED"
        push_memory(
            f"[EXPERIMENT {verdict}] {hypothesis[:200]} — {result_data.get('measurement', '')}",
            source="experiment",
            tm_label="experiment_result",
            project=self.project,
            aif_confidence=0.8 if passed else 0.3,
        )

        self._log(f"experiment {experiment_id}: {status}")

        return {
            "experiment_id": experiment_id,
            "status": status,
            "passed": passed,
            "result": result_data,
        }

    def _run_code_experiment(self, command: str) -> dict[str, object]:
        """Run a shell command and capture output."""
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd="/tmp",
            )
            output = proc.stdout + proc.stderr
            passed = proc.returncode == 0

            return {
                "output": output[:5000],
                "exit_code": proc.returncode,
                "measurement": f"exit_code={proc.returncode}",
                "passed": passed,
            }
        except subprocess.TimeoutExpired:
            return {
                "output": "Command timed out after 120 seconds",
                "exit_code": -1,
                "measurement": "timeout",
                "passed": False,
                "error": "timeout",
            }

    def _run_llm_experiment(self, hypothesis: str, plan: object) -> dict[str, object]:
        """Use LLM to reason about an experiment's outcome."""
        plan_str = json.dumps(plan, default=str) if isinstance(plan, dict) else str(plan)

        prompt = f"""Execute this thought experiment and determine the outcome.

HYPOTHESIS: {hypothesis}

EXPERIMENT PLAN:
{plan_str}

Reason step-by-step through the experiment:
1. Set up the baseline condition
2. Apply the modification
3. Compute or reason about the measurement
4. Compare to the pass/fail criteria

Respond with:
OUTCOME: [PASSED|FAILED|INCONCLUSIVE]
MEASUREMENT: [the measured/computed value]
REASONING: [step-by-step reasoning]"""

        result = _call_ollama(prompt, timeout=90)

        outcome = "INCONCLUSIVE"
        measurement = ""
        reasoning = ""

        if result:
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("OUTCOME:"):
                    o = line.split(":", 1)[1].strip().upper()
                    if o in ("PASSED", "FAILED", "INCONCLUSIVE"):
                        outcome = o
                elif line.startswith("MEASUREMENT:"):
                    measurement = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

        return {
            "output": reasoning,
            "exit_code": 0 if outcome == "PASSED" else 1,
            "measurement": measurement,
            "passed": outcome == "PASSED",
        }

    def compare_to_baseline(self, results: dict, baseline: str) -> dict:
        """Compare experiment results to baseline."""
        measurement = ""
        if isinstance(results.get("result"), dict):
            measurement = str(results["result"].get("measurement", ""))

        prompt = f"""Compare this experiment result to the baseline.

BASELINE: {baseline}
RESULT: {measurement}

Respond with:
DELTA: [the difference between result and baseline]
SIGNIFICANT: [YES|NO — is the difference meaningful?]
INTERPRETATION: [what this means for the hypothesis]"""

        result = _call_ollama(prompt, timeout=60)

        delta = ""
        significant = False
        interpretation = ""

        if result:
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("DELTA:"):
                    delta = line.split(":", 1)[1].strip()
                elif line.startswith("SIGNIFICANT:"):
                    significant = line.split(":", 1)[1].strip().upper() == "YES"
                elif line.startswith("INTERPRETATION:"):
                    interpretation = line.split(":", 1)[1].strip()

        # Update experiment with delta
        experiment_id = results.get("experiment_id", 0)
        if experiment_id:
            conn = _get_conn()
            conn.execute(
                "UPDATE experiments SET delta = ? WHERE id = ?",
                (json.dumps({"delta": delta, "significant": significant}), experiment_id),
            )
            conn.commit()

        return {
            "delta": delta,
            "significant": significant,
            "interpretation": interpretation,
        }

    def store_experiment(self, hypothesis: str, plan: dict, results: dict) -> int:
        """Store a complete experiment record. Returns experiment ID."""
        conn = _get_conn()
        now = datetime.now(timezone.utc).isoformat()

        passed = bool(results.get("passed", False))
        status = "passed" if passed else "failed"
        if results.get("error"):
            status = "error"

        cur = conn.execute(
            """INSERT INTO experiments
               (project, hypothesis, experiment_type, plan, baseline,
                result, status, confidence, created, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.project, hypothesis[:1000],
             plan.get("experiment_type", "code"),
             json.dumps(plan, default=str),
             plan.get("baseline", ""),
             json.dumps(results, default=str),
             status,
             0.8 if passed else 0.3,
             now, now),
        )
        conn.commit()

        experiment_id = cur.lastrowid or 0
        self._log(f"experiment {experiment_id} stored: {status}")
        return experiment_id

    def list_experiments(self, status: str = "") -> list[dict]:
        """Return all experiments, optionally filtered by status."""
        conn = _get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE project = ? AND status = ? ORDER BY created DESC",
                (self.project, status),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE project = ? ORDER BY created DESC",
                (self.project,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_experiment(self, experiment_id: int) -> Optional[dict]:
        """Get a single experiment by ID."""
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        return dict(row) if row else None

    def experiment_dashboard(self) -> dict:
        """Summary stats for experiments."""
        conn = _get_conn()

        total = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE project = ?",
            (self.project,),
        ).fetchone()[0]

        passed = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE project = ? AND status = 'passed'",
            (self.project,),
        ).fetchone()[0]

        failed = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE project = ? AND status = 'failed'",
            (self.project,),
        ).fetchone()[0]

        errors = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE project = ? AND status = 'error'",
            (self.project,),
        ).fetchone()[0]

        designed = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE project = ? AND status = 'designed'",
            (self.project,),
        ).fetchone()[0]

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "designed": designed,
            "pass_rate": round(passed / max(total - designed, 1), 2),
        }

    def run_full_experiment(self, hypothesis: str) -> dict:
        """Full pipeline: testability → design → execute → compare → store."""
        # Step 1: Check testability
        testability = self.is_testable(hypothesis)
        if not testability["testable"]:
            self._log(f"hypothesis not testable: {testability['reasoning']}")
            return {
                "testable": False,
                "reasoning": testability["reasoning"],
            }

        # Step 2: Design
        design = self.design_experiment(
            hypothesis, testability["experiment_type"],
        )

        # Step 3: Execute
        results = self.run_experiment(design)

        # Step 4: Compare to baseline
        baseline = design["plan"].get("baseline", "")
        comparison = self.compare_to_baseline(results, baseline)

        self._log(f"full experiment complete: {'PASSED' if results['passed'] else 'FAILED'}")

        return {
            "testable": True,
            "experiment_id": design["experiment_id"],
            "hypothesis": hypothesis,
            "experiment_type": testability["experiment_type"],
            "plan": design["plan"],
            "results": results,
            "comparison": comparison,
            "passed": results["passed"],
        }
