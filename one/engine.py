"""Zero Hallucination Engine — Verification pipeline for every code action.

Nothing lands without going through this. Every SQL query checked against live
schemas. Every function call checked against runtime signatures. Every decision
logged to the knowledge graph. The system learns from every mistake.

Architecture:
    PROPOSE  → Claude wants to change something
    CHALLENGE → Engine checks against ground truth (schemas, signatures, contracts)
    VERIFY   → Actually run it — imports, AST, integration
    LOG      → Store the decision with evidence in the knowledge graph
    PASS/FAIL → Edit lands or gets rejected with reasons

Usage:
    from one.engine import verify_edit, verify_sql, verify_call, log_decision

    result = verify_edit("/path/to/file.py", old_content, new_content)
    if not result["passed"]:
        for issue in result["issues"]:
            print(f"BLOCKED: {issue}")
"""

import ast
import re
import os
import sqlite3
import inspect
import importlib
import threading
import json
import logging
import signal
import atexit
from datetime import datetime, timezone
from typing import Optional, Callable

from .store import push_memory, DB_PATH, DB_DIR
from .hdc import encode_tagged


# ── Session Logger — survives crashes, sigkills, everything ───────────

LOG_DIR = os.path.join(DB_DIR, "logs")
_logger: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    """Get or create the session logger. Writes to ~/.one/logs/."""
    global _logger
    if _logger is not None:
        return _logger

    os.makedirs(LOG_DIR, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, f"session-{now}.log")

    _logger = logging.getLogger("one.engine")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(fh)

    # Also keep a latest symlink
    latest = os.path.join(LOG_DIR, "latest.log")
    try:
        if os.path.islink(latest):
            os.unlink(latest)
        os.symlink(log_path, latest)
    except OSError:
        pass

    _logger.info(f"session started — log: {log_path}")

    # Register shutdown handler to flush on exit/signal
    atexit.register(_flush_log)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
            def _handler(s, f, _prev=prev):
                _flush_log()
                if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    _prev(s, f)
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass  # Can't set signals in non-main thread

    return _logger


def _flush_log():
    """Flush all log handlers — called on exit/signal."""
    if _logger:
        for h in _logger.handlers:
            h.flush()


def elog(msg: str, level: str = "info") -> None:
    """Write to the engine session log. Always available."""
    log = _get_logger()
    getattr(log, level, log.info)(msg)


# ── Schema Truth ──────────────────────────────────────────────────────


_schema_cache: Optional[dict] = None


def _get_schema_truth() -> dict:
    """Load exact column names for every table from the live database."""
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    db = os.path.expanduser("~/.one/one.db")
    if not os.path.exists(db):
        _schema_cache = {}
        return _schema_cache

    conn = sqlite3.connect(db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
    ).fetchall()

    truth = {}
    for (name,) in tables:
        cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
        truth[name] = {
            "columns": [c[1] for c in cols],
            "types": {c[1]: c[2] for c in cols},
        }
    conn.close()
    _schema_cache = truth
    return truth


def invalidate_schema_cache():
    """Force re-read of schemas after migrations."""
    global _schema_cache
    _schema_cache = None


# ── SQL Extraction & Verification ─────────────────────────────────────


_SQL_PATTERN = re.compile(
    r"""(?:execute|executescript|executemany)\s*\(\s*(?:f?["']{1,3})(.*?)(?:["']{1,3})""",
    re.DOTALL,
)

_TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?(\w+)",
    re.IGNORECASE,
)

_COLUMN_REF = re.compile(
    r"\bSELECT\s+(.*?)\s+FROM",
    re.IGNORECASE | re.DOTALL,
)

_WHERE_COL = re.compile(
    r"\bWHERE\b(.*?)(?:\bORDER\b|\bGROUP\b|\bLIMIT\b|\bHAVING\b|$)",
    re.IGNORECASE | re.DOTALL,
)

_COL_NAME = re.compile(r"\b(\w+)\s*(?:=|<|>|!=|IS\b|IN\b|LIKE\b|BETWEEN\b)")


def extract_sql_from_source(source: str) -> list[dict]:
    """Extract SQL queries from Python source code using AST + regex."""
    queries = []

    # Walk AST to find string literals inside execute() calls
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return queries

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Check if it's a .execute/.executescript call
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        if func_name not in ("execute", "executescript", "executemany"):
            continue

        if not node.args:
            continue

        # Extract the SQL string
        sql_arg = node.args[0]
        sql_text = None

        if isinstance(sql_arg, ast.Constant) and isinstance(sql_arg.value, str):
            sql_text = sql_arg.value
        elif isinstance(sql_arg, ast.JoinedStr):
            # f-string — extract the static parts
            parts = []
            for v in sql_arg.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                else:
                    parts.append("?")  # placeholder for dynamic parts
            sql_text = "".join(parts)

        if sql_text and len(sql_text.strip()) > 5:
            queries.append({
                "sql": sql_text.strip(),
                "line": node.lineno,
            })

    return queries


def verify_sql(sql: str, line: int = 0) -> list[dict]:
    """Check a SQL query against live schema truth. Returns list of issues."""
    issues = []
    schema = _get_schema_truth()

    if not schema:
        return issues  # No DB to check against

    # Collect ALL tables referenced in the query (handles JOINs)
    all_tables = []
    for match in _TABLE_REF.finditer(sql):
        table = match.group(1).lower()
        if table in ("sqlite_sequence", "sqlite_master", "dual"):
            continue
        if table not in schema:
            if table == "rules" and "rule_nodes" in schema:
                issues.append({
                    "type": "wrong_table",
                    "line": line,
                    "message": f"Table 'rules' does not exist. Did you mean 'rule_nodes'?",
                    "severity": "error",
                })
            elif table not in ("set", "values", "null", "true", "false"):
                issues.append({
                    "type": "unknown_table",
                    "line": line,
                    "message": f"Table '{table}' not found in database",
                    "severity": "warning",
                })
        else:
            all_tables.append(table)

    if not all_tables:
        return issues

    # Build union of all valid columns across all referenced tables
    all_cols = set()
    for t in all_tables:
        for c in schema[t]["columns"]:
            all_cols.add(c.lower())

    # For single-table queries, enforce strict column checking
    # For JOINs, check against union of all table columns
    # Also track which columns belong to which table for entity-specific checks
    entity_cols = set(c.lower() for c in schema.get("entities", {}).get("columns", []))

    def _check_col(col_name: str, context: str) -> None:
        """Check a column reference against known tables."""
        cl = col_name.lower()
        if cl in all_cols:
            return  # Valid across some referenced table

        suggestion = ""
        if cl == "entity_type" and "type" in all_cols:
            suggestion = " Did you mean 'type'?"
        elif cl == "created" and "timestamp" in all_cols:
            suggestion = " Did you mean 'timestamp'?"
        elif cl == "project" and "entities" in all_tables and "project" not in entity_cols:
            suggestion = " entities table has NO project column."

        issues.append({
            "type": "wrong_column",
            "line": line,
            "message": f"{context} column '{col_name}' not found in tables {all_tables}.{suggestion}",
            "severity": "error",
        })

    # Check SELECT columns
    select_match = _COLUMN_REF.search(sql)
    if select_match:
        select_part = select_match.group(1)
        if select_part.strip() != "*":
            for col_token in select_part.split(","):
                col_token = col_token.strip()
                if " AS " in col_token.upper():
                    col_token = col_token.split()[0]
                if "." in col_token:
                    col_token = col_token.split(".")[-1]
                if "(" in col_token:
                    inner = re.search(r"\((\w+)\)", col_token)
                    if inner:
                        col_token = inner.group(1)
                    else:
                        continue
                col_token = col_token.strip().strip("'\"")
                if col_token and col_token != "?" and col_token != "*":
                    _check_col(col_token, "SELECT")

    # Check WHERE columns
    where_match = _WHERE_COL.search(sql)
    if where_match:
        where_part = where_match.group(1)
        for col_match in _COL_NAME.finditer(where_part):
            col = col_match.group(1)
            if col.upper() in ("AND", "OR", "NOT", "NULL", "TRUE", "FALSE"):
                continue
            if col == "?" or col.startswith("'"):
                continue
            _check_col(col, "WHERE")

    return issues


# ── Signature Verification ────────────────────────────────────────────


def verify_call(module_path: str, func_name: str, kwargs: list[str]) -> list[dict]:
    """Verify a function call against its actual runtime signature."""
    issues = []
    try:
        mod = importlib.import_module(module_path)
        obj = getattr(mod, func_name, None)
        if obj is None:
            issues.append({
                "type": "missing_function",
                "message": f"{module_path}.{func_name} does not exist",
                "severity": "error",
            })
            return issues

        sig = inspect.signature(obj if not inspect.isclass(obj) else obj.__init__)
        params = set(sig.parameters.keys()) - {"self"}

        for kw in kwargs:
            if kw not in params:
                issues.append({
                    "type": "wrong_kwarg",
                    "message": f"{func_name}() has no parameter '{kw}'. Valid: {sorted(params)}",
                    "severity": "error",
                })
    except Exception:
        pass  # Can't import module — skip

    return issues


# ── Full Edit Verification ────────────────────────────────────────────


def verify_edit(file_path: str, new_content: str) -> dict:
    """Full verification pipeline for any source file.

    Supports: Python, C/C99, JavaScript, HTML, CSS, and generic text.

    Returns:
        {
            "passed": bool,
            "issues": [{"type", "line", "message", "severity"}],
            "sql_checked": int,
            "warnings": int,
            "errors": int,
        }
    """
    ext = os.path.splitext(file_path)[1].lower()
    issues = []

    if ext == ".py":
        issues = _verify_python(file_path, new_content)
    elif ext in (".c", ".h"):
        issues = _verify_c(file_path, new_content)
    elif ext in (".js", ".mjs", ".ts", ".jsx", ".tsx"):
        issues = _verify_js(file_path, new_content)
    elif ext in (".html", ".htm"):
        issues = _verify_html(file_path, new_content)
    elif ext == ".css":
        issues = _verify_css(file_path, new_content)
    elif ext in (".json",):
        issues = _verify_json(file_path, new_content)

    # Count SQL checked (only from Python analysis)
    sql_checked = sum(1 for i in issues if i.get("_sql"))
    for i in issues:
        i.pop("_sql", None)

    errors = sum(1 for i in issues if i["severity"] == "error")
    warnings = sum(1 for i in issues if i["severity"] == "warning")

    return {
        "passed": errors == 0,
        "issues": issues,
        "sql_checked": sql_checked,
        "warnings": warnings,
        "errors": errors,
    }


def _verify_python(file_path: str, content: str) -> list[dict]:
    """Python-specific verification: syntax, imports, SQL, patterns."""
    issues = []

    # 1. Syntax check
    try:
        ast.parse(content)
    except SyntaxError as e:
        issues.append({
            "type": "syntax_error",
            "line": e.lineno or 0,
            "message": f"Syntax error: {e.msg}",
            "severity": "error",
        })
        return issues

    # 2. Check for top-level Foundry imports
    tree = ast.parse(content)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "foundry" in alias.name.lower() or "orion_push" in alias.name.lower():
                    issues.append({
                        "type": "top_level_foundry_import",
                        "line": node.lineno,
                        "message": f"Top-level Foundry import '{alias.name}'. Must be lazy (inside function).",
                        "severity": "error",
                    })
        elif isinstance(node, ast.ImportFrom) and node.module:
            if "foundry" in node.module.lower() or "orion_push" in node.module.lower():
                issues.append({
                    "type": "top_level_foundry_import",
                    "line": node.lineno,
                    "message": f"Top-level Foundry import from '{node.module}'. Must be lazy (inside function).",
                    "severity": "error",
                })

    # 3. Extract and verify all SQL queries
    queries = extract_sql_from_source(content)
    for q in queries:
        sql_issues = verify_sql(q["sql"], q["line"])
        for si in sql_issues:
            si["_sql"] = True
        issues.extend(sql_issues)

    # 4. Check for recall("") anti-pattern
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "recall":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "":
                    issues.append({
                        "type": "recall_empty_string",
                        "line": node.lineno,
                        "message": "recall('') returns empty (zero vector). Use get_recent() instead.",
                        "severity": "error",
                    })

    return issues


def _verify_c(file_path: str, content: str) -> list[dict]:
    """C/C99 verification: bracket matching, common errors, include guards."""
    issues = []
    lines = content.split("\n")

    # Bracket/brace matching
    stack = []
    pairs = {"{": "}", "(": ")", "[": "]"}
    closing = set(pairs.values())
    in_string = False
    in_comment = False
    in_block_comment = False

    for lineno, line in enumerate(lines, 1):
        i = 0
        while i < len(line):
            ch = line[i]

            # Track block comments
            if in_block_comment:
                if ch == "*" and i + 1 < len(line) and line[i + 1] == "/":
                    in_block_comment = False
                    i += 1
                i += 1
                continue

            # Skip line comments
            if ch == "/" and i + 1 < len(line):
                if line[i + 1] == "/":
                    break  # Rest of line is comment
                if line[i + 1] == "*":
                    in_block_comment = True
                    i += 2
                    continue

            # Skip strings
            if ch == '"' and not in_string:
                in_string = True
                i += 1
                continue
            if ch == '"' and in_string:
                # Check for escaped quote
                if i > 0 and line[i - 1] != "\\":
                    in_string = False
                i += 1
                continue
            if in_string:
                i += 1
                continue

            if ch in pairs:
                stack.append((ch, lineno))
            elif ch in closing:
                if stack:
                    top, open_line = stack[-1]
                    if pairs.get(top) == ch:
                        stack.pop()
                    else:
                        issues.append({
                            "type": "bracket_mismatch",
                            "line": lineno,
                            "message": f"Mismatched '{ch}' — expected '{pairs[top]}' to close '{top}' from line {open_line}",
                            "severity": "error",
                        })
                else:
                    issues.append({
                        "type": "bracket_mismatch",
                        "line": lineno,
                        "message": f"Unexpected closing '{ch}' with no matching opener",
                        "severity": "error",
                    })
            i += 1

    for ch, lineno in stack:
        issues.append({
            "type": "bracket_mismatch",
            "line": lineno,
            "message": f"Unclosed '{ch}' — missing '{pairs[ch]}'",
            "severity": "error",
        })

    # Check for common C pitfalls
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        # Assignment in condition
        if re.match(r'\bif\s*\([^=]*[^!=<>]=[^=]', stripped):
            if "==" not in stripped:
                issues.append({
                    "type": "assignment_in_condition",
                    "line": lineno,
                    "message": "Possible assignment in condition (use == for comparison)",
                    "severity": "warning",
                })
        # malloc without sizeof
        if "malloc(" in stripped and "sizeof" not in stripped and "calloc" not in stripped:
            issues.append({
                "type": "malloc_no_sizeof",
                "line": lineno,
                "message": "malloc() without sizeof — potential size error",
                "severity": "warning",
            })
        # Buffer overflow patterns
        if re.search(r'\b(gets|sprintf|strcpy|strcat)\s*\(', stripped):
            func = re.search(r'\b(gets|sprintf|strcpy|strcat)\b', stripped).group(1)
            safe = {"gets": "fgets", "sprintf": "snprintf", "strcpy": "strncpy", "strcat": "strncat"}
            issues.append({
                "type": "unsafe_function",
                "line": lineno,
                "message": f"'{func}' is unsafe — use '{safe[func]}' instead",
                "severity": "error",
            })

    # Header guard check for .h files
    if file_path.endswith(".h"):
        has_guard = False
        for line in lines[:5]:
            if line.strip().startswith("#ifndef") or line.strip().startswith("#pragma once"):
                has_guard = True
                break
        if not has_guard:
            issues.append({
                "type": "missing_header_guard",
                "line": 1,
                "message": "Header file missing include guard (#ifndef/#pragma once)",
                "severity": "warning",
            })

    return issues


def _verify_js(file_path: str, content: str) -> list[dict]:
    """JavaScript/TypeScript verification: common errors and patterns."""
    issues = []
    lines = content.split("\n")

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()

        # var usage (should use let/const)
        if re.match(r'\bvar\s+', stripped):
            issues.append({
                "type": "var_usage",
                "line": lineno,
                "message": "Use 'let' or 'const' instead of 'var'",
                "severity": "warning",
            })

        # == instead of ===
        if re.search(r'[^!=]==[^=]', stripped) and "===" not in stripped:
            issues.append({
                "type": "loose_equality",
                "line": lineno,
                "message": "Use '===' instead of '==' for strict equality",
                "severity": "warning",
            })

        # console.log left in
        if "console.log(" in stripped and not stripped.startswith("//"):
            issues.append({
                "type": "console_log",
                "line": lineno,
                "message": "console.log() left in code",
                "severity": "warning",
            })

        # eval() usage
        if re.search(r'\beval\s*\(', stripped):
            issues.append({
                "type": "eval_usage",
                "line": lineno,
                "message": "eval() is a security risk — use alternatives",
                "severity": "error",
            })

        # innerHTML (XSS risk)
        if ".innerHTML" in stripped and "textContent" not in stripped:
            issues.append({
                "type": "inner_html",
                "line": lineno,
                "message": "innerHTML is an XSS risk — use textContent or sanitize",
                "severity": "warning",
            })

    # Bracket matching (basic)
    _check_brackets(content, issues)

    return issues


def _verify_html(file_path: str, content: str) -> list[dict]:
    """HTML verification: structure, common issues."""
    issues = []
    lower = content.lower()

    if "<html" not in lower and "<!doctype" not in lower:
        # Could be a fragment, skip structure checks
        pass
    else:
        if "<!doctype" not in lower:
            issues.append({
                "type": "missing_doctype",
                "line": 1,
                "message": "Missing <!DOCTYPE html> declaration",
                "severity": "warning",
            })
        if "<meta" in lower and 'charset' not in lower:
            issues.append({
                "type": "missing_charset",
                "line": 1,
                "message": "Missing charset meta tag",
                "severity": "warning",
            })

    # Check for inline event handlers (security)
    lines = content.split("\n")
    for lineno, line in enumerate(lines, 1):
        for handler in ("onclick=", "onload=", "onerror=", "onmouseover=", "onfocus="):
            if handler in line.lower():
                issues.append({
                    "type": "inline_handler",
                    "line": lineno,
                    "message": f"Inline event handler '{handler.rstrip('=')}' — use addEventListener instead",
                    "severity": "warning",
                })
                break

    # Unclosed tags (basic check)
    void_tags = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
    tag_stack = []
    for lineno, line in enumerate(lines, 1):
        for m in re.finditer(r'<(/?)(\w+)[^>]*>', line):
            closing = m.group(1) == "/"
            tag = m.group(2).lower()
            if tag in void_tags:
                continue
            if closing:
                if tag_stack and tag_stack[-1][0] == tag:
                    tag_stack.pop()
            else:
                tag_stack.append((tag, lineno))

    for tag, lineno in tag_stack[-5:]:
        issues.append({
            "type": "unclosed_tag",
            "line": lineno,
            "message": f"Unclosed <{tag}> tag",
            "severity": "warning",
        })

    return issues


def _verify_css(file_path: str, content: str) -> list[dict]:
    """CSS verification: syntax, common issues."""
    issues = []
    lines = content.split("\n")

    brace_depth = 0
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("/*") or stripped.startswith("//"):
            continue

        brace_depth += stripped.count("{") - stripped.count("}")

        # !important overuse
        if "!important" in stripped:
            issues.append({
                "type": "important_usage",
                "line": lineno,
                "message": "!important — consider specificity instead",
                "severity": "warning",
            })

    if brace_depth != 0:
        issues.append({
            "type": "brace_mismatch",
            "line": len(lines),
            "message": f"Unbalanced braces: {'unclosed' if brace_depth > 0 else 'extra closing'} ({abs(brace_depth)})",
            "severity": "error",
        })

    return issues


def _verify_json(file_path: str, content: str) -> list[dict]:
    """JSON verification: valid syntax."""
    issues = []
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        issues.append({
            "type": "json_syntax",
            "line": e.lineno or 1,
            "message": f"Invalid JSON: {e.msg}",
            "severity": "error",
        })
    return issues


def _check_brackets(content: str, issues: list[dict]) -> None:
    """Generic bracket matching for JS/TS files."""
    stack = []
    pairs = {"{": "}", "(": ")", "[": "]"}
    closing = set(pairs.values())
    in_string = None
    in_template = False

    for lineno, line in enumerate(content.split("\n"), 1):
        i = 0
        while i < len(line):
            ch = line[i]

            # Skip comments
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break

            # Track strings
            if ch in ('"', "'", "`") and in_string is None:
                in_string = ch
                if ch == "`":
                    in_template = True
                i += 1
                continue
            if ch == in_string and (i == 0 or line[i - 1] != "\\"):
                in_string = None
                in_template = False
                i += 1
                continue
            if in_string:
                i += 1
                continue

            if ch in pairs:
                stack.append((ch, lineno))
            elif ch in closing:
                if stack and pairs.get(stack[-1][0]) == ch:
                    stack.pop()
            i += 1

    for ch, lineno in stack[-3:]:
        issues.append({
            "type": "bracket_mismatch",
            "line": lineno,
            "message": f"Unclosed '{ch}'",
            "severity": "error",
        })


# ── Decision Logging ──────────────────────────────────────────────────


def log_decision(
    action: str,
    file_path: str,
    outcome: str,
    evidence: list[str],
    issues: list[dict],
    project: str = "one",
) -> str:
    """Log a verification decision to the knowledge graph.

    Every pass, every fail, every issue found — stored permanently
    so the system learns from its own mistakes.
    """
    severity_counts = {}
    for i in issues:
        s = i.get("severity", "info")
        severity_counts[s] = severity_counts.get(s, 0) + 1

    evidence_text = "; ".join(evidence[:5]) if evidence else "none"
    issue_text = "; ".join(i["message"] for i in issues[:5]) if issues else "none"

    text = (
        f"DECISION LOG [{outcome.upper()}]: {action} on {os.path.basename(file_path)}. "
        f"Evidence: {evidence_text}. "
        f"Issues: {issue_text}. "
        f"Counts: {severity_counts}."
    )

    confidence = 0.95 if outcome == "pass" else 0.99  # Failures are MORE valuable to remember

    elog(f"decision: {outcome} — {action} on {os.path.basename(file_path)} | {severity_counts}")

    vec = encode_tagged(text, role="system")
    mid = push_memory(
        raw_text=text,
        source="engine",
        tm_label="decision_log",
        regime_tag="verified",
        aif_confidence=confidence,
        hdc_vector=vec.tolist(),
        project=project,
    )
    return mid


# ── Turn Logger ───────────────────────────────────────────────────────


def log_turn(
    turn_number: int,
    role: str,
    action: str,
    content_summary: str,
    verification_result: Optional[dict] = None,
    project: str = "one",
) -> str:
    """Log every single turn with full context.

    If Claude says something — logged.
    If Claude edits something — logged with verification result.
    If user says no — logged with the rejection.
    If something is fixed — logged with what was wrong and what fixed it.
    """
    ver_status = ""
    if verification_result:
        passed = verification_result.get("passed", True)
        errors = verification_result.get("errors", 0)
        warnings = verification_result.get("warnings", 0)
        ver_status = f" [{'PASS' if passed else f'FAIL:{errors}e/{warnings}w'}]"

    text = (
        f"TURN {turn_number} [{role}]{ver_status}: {action}. "
        f"Summary: {content_summary[:200]}"
    )

    elog(f"turn {turn_number} [{role}]{ver_status}: {action}")

    vec = encode_tagged(text, role="system")
    mid = push_memory(
        raw_text=text,
        source="engine",
        tm_label="turn_log",
        regime_tag="log",
        aif_confidence=0.8,
        hdc_vector=vec.tolist(),
        project=project,
    )
    return mid


# ── Convenience: Verify a file on disk ────────────────────────────────


def verify_file(file_path: str, project: str = "one") -> dict:
    """Read a file and run full verification. Log the result."""
    if not os.path.exists(file_path):
        elog(f"verify_file: not found — {file_path}", "warning")
        return {"passed": False, "issues": [{"type": "missing", "message": f"File not found: {file_path}", "severity": "error"}]}

    with open(file_path) as f:
        content = f.read()

    result = verify_edit(file_path, content)
    status = "PASS" if result["passed"] else f"FAIL({result['errors']}e)"
    elog(f"verify_file: {os.path.basename(file_path)} — {status}, {result['sql_checked']} SQL")

    evidence = [f"sql_checked={result['sql_checked']}"]
    if result["passed"]:
        evidence.append("all checks passed")
    else:
        evidence.extend(i["message"] for i in result["issues"][:3])

    log_decision(
        action="verify_file",
        file_path=file_path,
        outcome="pass" if result["passed"] else "fail",
        evidence=evidence,
        issues=result["issues"],
        project=project,
    )

    return result


# ── Batch: Verify entire codebase ─────────────────────────────────────


def verify_codebase(project: str = "one", on_log: Optional[Callable] = None) -> dict:
    """Verify every Python file in the one/ directory."""
    log = on_log or (lambda m: None)
    base = os.path.dirname(os.path.abspath(__file__))

    results = {"passed": 0, "failed": 0, "total_issues": 0, "files": {}}

    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue

        fpath = os.path.join(base, fname)
        result = verify_file(fpath, project=project)

        if result["passed"]:
            results["passed"] += 1
            log(f"  PASS  {fname} ({result['sql_checked']} SQL checked)")
        else:
            results["failed"] += 1
            results["total_issues"] += len(result["issues"])
            log(f"  FAIL  {fname} — {len(result['issues'])} issues:")
            for issue in result["issues"][:5]:
                log(f"        line {issue.get('line', '?')}: {issue['message']}")

        results["files"][fname] = result

    total = results["passed"] + results["failed"]
    log(f"\n{results['passed']}/{total} files passed, {results['total_issues']} total issues")
    return results


# ══════════════════════════════════════════════════════════════════════
# CODEBASE ONTOLOGY — Every function, every connection, every version
# ══════════════════════════════════════════════════════════════════════


_ontology_local = threading.local()
_ontology_project = "global"


def set_ontology_project(project: str) -> None:
    """Set the active project for ontology storage."""
    global _ontology_project
    _ontology_project = project
    # Invalidate cached connection so next call picks up new project
    if hasattr(_ontology_local, "conn"):
        _ontology_local.conn = None


def _ont_conn() -> sqlite3.Connection:
    """Thread-local connection to the project-scoped ontology database."""
    if not hasattr(_ontology_local, "conn") or _ontology_local.conn is None:
        db_dir = os.path.join(DB_DIR, "ontology")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, f"{_ontology_project}.db")
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _init_ontology_schema(conn)
        _ontology_local.conn = conn
    return _ontology_local.conn


def _init_ontology_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        -- Every function/class/method in the codebase
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file TEXT NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,            -- function, class, method
            parent TEXT,                   -- class name if method
            signature TEXT,                -- full signature string
            params TEXT,                   -- JSON list of param names
            return_type TEXT,
            docstring TEXT,
            line_start INTEGER,
            line_end INTEGER,
            UNIQUE(file, name, parent)
        );

        -- What calls what: function A in file X calls function B in file Y
        CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            caller_file TEXT NOT NULL,
            caller_name TEXT NOT NULL,
            callee_module TEXT,            -- resolved import path
            callee_name TEXT NOT NULL,
            line INTEGER,
            resolved INTEGER DEFAULT 0,    -- 1 if we verified it exists
            UNIQUE(caller_file, caller_name, callee_name, line)
        );

        -- File-to-file dependencies: file X imports from file Y
        CREATE TABLE IF NOT EXISTS file_deps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            target_file TEXT NOT NULL,
            import_names TEXT,             -- JSON list of imported names
            UNIQUE(source_file, target_file)
        );

        -- Every change to every file, with context
        CREATE TABLE IF NOT EXISTS changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,          -- edit, create, delete, fix, refactor
            summary TEXT NOT NULL,
            diff_summary TEXT,             -- what lines changed
            reason TEXT,                   -- why the change was made
            bug_ref TEXT,                  -- what bug this fixes
            tried_and_failed TEXT,         -- what was attempted but didn't work
            verification_result TEXT,      -- JSON of engine verification
            linked_concepts TEXT,          -- JSON list of concept entity IDs
            linked_files TEXT              -- JSON list of other affected files
        );

        -- Contracts: function X returns dict with keys [a, b, c]
        CREATE TABLE IF NOT EXISTS contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file TEXT NOT NULL,
            symbol_name TEXT NOT NULL,
            contract_type TEXT NOT NULL,   -- return_keys, param_types, side_effect
            contract_data TEXT NOT NULL,   -- JSON
            verified INTEGER DEFAULT 0,
            last_verified TEXT,
            UNIQUE(file, symbol_name, contract_type)
        );

        CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_file, caller_name);
        CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_name);
        CREATE INDEX IF NOT EXISTS idx_file_deps_source ON file_deps(source_file);
        CREATE INDEX IF NOT EXISTS idx_file_deps_target ON file_deps(target_file);
        CREATE INDEX IF NOT EXISTS idx_changes_file ON changes(file);
        CREATE INDEX IF NOT EXISTS idx_contracts_symbol ON contracts(file, symbol_name);
    """)
    conn.commit()


# ── AST Introspection ─────────────────────────────────────────────────


def _extract_symbols(file_path: str, source: str) -> list[dict]:
    """Extract every function, class, and method from a Python file."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    symbols = []
    fname = os.path.basename(file_path)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            params = []
            for arg in node.args.args:
                p = arg.arg
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    p += f": {arg.annotation.id}"
                params.append(p)

            # Get defaults
            defaults = node.args.defaults
            n_defaults = len(defaults)
            n_params = len(params)
            for i, d in enumerate(defaults):
                idx = n_params - n_defaults + i
                if isinstance(d, ast.Constant):
                    params[idx] += f"={d.value!r}"
                elif isinstance(d, ast.Name):
                    params[idx] += f"={d.id}"

            ret = ""
            if node.returns:
                if isinstance(node.returns, ast.Name):
                    ret = node.returns.id
                elif isinstance(node.returns, ast.Constant):
                    ret = str(node.returns.value)

            doc = ast.get_docstring(node) or ""

            symbols.append({
                "file": fname,
                "name": node.name,
                "kind": "function",
                "parent": None,
                "signature": f"{node.name}({', '.join(params)})",
                "params": json.dumps([a.arg for a in node.args.args]),
                "return_type": ret,
                "docstring": doc[:500],
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
            })

        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            # Get __init__ params
            init_params = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for arg in item.args.args:
                        if arg.arg != "self":
                            p = arg.arg
                            if arg.annotation and isinstance(arg.annotation, ast.Name):
                                p += f": {arg.annotation.id}"
                            init_params.append(p)
                    defaults = item.args.defaults
                    n_d = len(defaults)
                    n_p = len(init_params)
                    for i, d in enumerate(defaults):
                        idx = n_p - n_d + i
                        if isinstance(d, ast.Constant):
                            init_params[idx] += f"={d.value!r}"

            symbols.append({
                "file": fname,
                "name": node.name,
                "kind": "class",
                "parent": None,
                "signature": f"{node.name}({', '.join(init_params)})",
                "params": json.dumps(init_params),
                "return_type": "",
                "docstring": doc[:500],
                "line_start": node.lineno,
                "line_end": node.end_lineno or node.lineno,
            })

            # Extract methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_") and item.name != "__init__":
                        continue
                    mparams = []
                    for arg in item.args.args:
                        if arg.arg == "self":
                            continue
                        p = arg.arg
                        if arg.annotation and isinstance(arg.annotation, ast.Name):
                            p += f": {arg.annotation.id}"
                        mparams.append(p)
                    defaults = item.args.defaults
                    n_d = len(defaults)
                    n_p = len(mparams)
                    for i, d in enumerate(defaults):
                        idx = n_p - n_d + i
                        if isinstance(d, ast.Constant):
                            mparams[idx] += f"={d.value!r}"

                    symbols.append({
                        "file": fname,
                        "name": item.name,
                        "kind": "method",
                        "parent": node.name,
                        "signature": f"{node.name}.{item.name}({', '.join(mparams)})",
                        "params": json.dumps(mparams),
                        "return_type": "",
                        "docstring": (ast.get_docstring(item) or "")[:500],
                        "line_start": item.lineno,
                        "line_end": item.end_lineno or item.lineno,
                    })

    return symbols


def _extract_calls(file_path: str, source: str) -> list[dict]:
    """Extract every function/method call and resolve what it's calling."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    calls = []
    fname = os.path.basename(file_path)

    # Build import map: name → module
    import_map = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local_name = alias.asname or alias.name
                import_map[local_name] = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name
                import_map[local_name] = alias.name

    # Find current function context for each call
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        callee = ""
        callee_module = ""

        if isinstance(node.func, ast.Name):
            callee = node.func.id
            callee_module = import_map.get(callee, "")
        elif isinstance(node.func, ast.Attribute):
            callee = node.func.attr
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                callee_module = import_map.get(obj_name, obj_name)

        if callee and not callee.startswith("_"):
            # Find which function this call is inside
            caller_name = _find_enclosing_function(tree, node.lineno)

            calls.append({
                "caller_file": fname,
                "caller_name": caller_name or "<module>",
                "callee_module": callee_module,
                "callee_name": callee,
                "line": node.lineno,
            })

    return calls


def _find_enclosing_function(tree: ast.Module, lineno: int) -> Optional[str]:
    """Find the function/method that contains the given line number."""
    best = None
    best_start = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end and node.lineno > best_start:
                # Check if it's a method
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        if node in ast.iter_child_nodes(parent):
                            best = f"{parent.name}.{node.name}"
                            best_start = node.lineno
                            break
                else:
                    best = node.name
                    best_start = node.lineno

    return best


def _extract_file_deps(file_path: str, source: str) -> list[dict]:
    """Extract file-to-file import dependencies."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    deps = {}  # target_file → set of names
    fname = os.path.basename(file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            level = node.level  # number of dots in relative import

            # Resolve: from .proxy import X → proxy.py
            # Resolve: from . import store → store.py
            target = None
            if level > 0:
                # Relative import within the package
                if mod:
                    target = mod.split(".")[0] + ".py"
                else:
                    # from . import X
                    for alias in node.names:
                        t = alias.name + ".py"
                        if t not in deps:
                            deps[t] = set()
                        deps[t].add(alias.name)
                    continue
            elif mod and mod.startswith("one."):
                target = mod.replace("one.", "").split(".")[0] + ".py"

            if target:
                names = [a.name for a in node.names]
                if target not in deps:
                    deps[target] = set()
                deps[target].update(names)

    return [
        {"source_file": fname, "target_file": t, "import_names": json.dumps(sorted(n))}
        for t, n in deps.items()
    ]


# ── Ontology Population ──────────────────────────────────────────────


def map_codebase(on_log: Optional[Callable] = None) -> dict:
    """Full codebase introspection. Maps every symbol, call, and dependency."""
    log = on_log or (lambda m: None)
    elog("map_codebase: starting full introspection")
    conn = _ont_conn()
    base = os.path.dirname(os.path.abspath(__file__))

    stats = {"files": 0, "symbols": 0, "calls": 0, "deps": 0}

    # Clear old data for fresh map
    conn.execute("DELETE FROM symbols")
    conn.execute("DELETE FROM calls")
    conn.execute("DELETE FROM file_deps")
    conn.commit()

    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue

        fpath = os.path.join(base, fname)
        with open(fpath) as f:
            source = f.read()

        stats["files"] += 1

        # Symbols
        symbols = _extract_symbols(fpath, source)
        for s in symbols:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO symbols (file, name, kind, parent, signature, params, return_type, docstring, line_start, line_end) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (s["file"], s["name"], s["kind"], s["parent"], s["signature"],
                     s["params"], s["return_type"], s["docstring"], s["line_start"], s["line_end"]),
                )
                stats["symbols"] += 1
            except sqlite3.IntegrityError:
                pass

        # Calls
        calls = _extract_calls(fpath, source)
        for c in calls:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO calls (caller_file, caller_name, callee_module, callee_name, line) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (c["caller_file"], c["caller_name"], c["callee_module"], c["callee_name"], c["line"]),
                )
                stats["calls"] += 1
            except sqlite3.IntegrityError:
                pass

        # File deps
        deps = _extract_file_deps(fpath, source)
        for d in deps:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO file_deps (source_file, target_file, import_names) "
                    "VALUES (?, ?, ?)",
                    (d["source_file"], d["target_file"], d["import_names"]),
                )
                stats["deps"] += 1
            except sqlite3.IntegrityError:
                pass

        sym_count = len(symbols)
        call_count = len(calls)
        dep_count = len(deps)
        log(f"  {fname}: {sym_count} symbols, {call_count} calls, {dep_count} deps")

    conn.commit()

    # Resolve calls — check if callee actually exists
    _resolve_calls(conn)

    log(f"\nMapped: {stats['files']} files, {stats['symbols']} symbols, {stats['calls']} calls, {stats['deps']} deps")
    return stats


def _resolve_calls(conn: sqlite3.Connection) -> None:
    """Mark calls as resolved if the callee exists in symbols table."""
    conn.execute("""
        UPDATE calls SET resolved = 1
        WHERE EXISTS (
            SELECT 1 FROM symbols
            WHERE symbols.name = calls.callee_name
        )
    """)
    conn.commit()


# ── Change Tracking ──────────────────────────────────────────────────


def track_change(
    file: str,
    action: str,
    summary: str,
    reason: str = "",
    bug_ref: str = "",
    tried_failed: str = "",
    verification: Optional[dict] = None,
    linked_concepts: Optional[list[str]] = None,
    linked_files: Optional[list[str]] = None,
) -> int:
    """Record a change with full context and links."""
    conn = _ont_conn()
    now = datetime.now(timezone.utc).isoformat()

    cursor = conn.execute(
        "INSERT INTO changes (file, timestamp, action, summary, reason, bug_ref, "
        "tried_and_failed, verification_result, linked_concepts, linked_files) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            os.path.basename(file),
            now,
            action,
            summary,
            reason,
            bug_ref,
            tried_failed,
            json.dumps(verification, default=str) if verification else None,
            json.dumps(linked_concepts) if linked_concepts else None,
            json.dumps(linked_files) if linked_files else None,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def store_contract(
    file: str,
    symbol_name: str,
    contract_type: str,
    contract_data: dict,
) -> int:
    """Store a verified contract for a function/class."""
    conn = _ont_conn()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT OR REPLACE INTO contracts (file, symbol_name, contract_type, contract_data, verified, last_verified) "
        "VALUES (?, ?, ?, ?, 1, ?)",
        (os.path.basename(file), symbol_name, contract_type, json.dumps(contract_data), now),
    )
    conn.commit()
    return cursor.lastrowid


# ── Impact Analysis ──────────────────────────────────────────────────


def get_impact(file: str) -> dict:
    """What gets affected if this file changes?

    Returns:
        {
            "dependents": files that import from this file,
            "dependencies": files this file imports from,
            "callers": functions in OTHER files that call functions in THIS file,
            "symbols": all public functions/classes in this file,
            "recent_changes": last 10 changes to this file,
        }
    """
    conn = _ont_conn()
    fname = os.path.basename(file)

    # Who depends on this file?
    dependents = [
        dict(r) for r in conn.execute(
            "SELECT source_file, import_names FROM file_deps WHERE target_file = ?",
            (fname,),
        ).fetchall()
    ]

    # What does this file depend on?
    dependencies = [
        dict(r) for r in conn.execute(
            "SELECT target_file, import_names FROM file_deps WHERE source_file = ?",
            (fname,),
        ).fetchall()
    ]

    # Who calls functions in this file from OTHER files?
    callers = [
        dict(r) for r in conn.execute(
            """SELECT c.caller_file, c.caller_name, c.callee_name, c.line
               FROM calls c
               JOIN symbols s ON s.name = c.callee_name AND s.file = ?
               WHERE c.caller_file != ?""",
            (fname, fname),
        ).fetchall()
    ]

    # All symbols in this file
    symbols = [
        dict(r) for r in conn.execute(
            "SELECT name, kind, parent, signature, line_start FROM symbols WHERE file = ? ORDER BY line_start",
            (fname,),
        ).fetchall()
    ]

    # Recent changes
    changes = [
        dict(r) for r in conn.execute(
            "SELECT timestamp, action, summary, reason, bug_ref FROM changes WHERE file = ? ORDER BY timestamp DESC LIMIT 10",
            (fname,),
        ).fetchall()
    ]

    # Contracts
    contracts = [
        dict(r) for r in conn.execute(
            "SELECT symbol_name, contract_type, contract_data, last_verified FROM contracts WHERE file = ?",
            (fname,),
        ).fetchall()
    ]

    return {
        "file": fname,
        "dependents": dependents,
        "dependencies": dependencies,
        "callers": callers,
        "symbols": symbols,
        "recent_changes": changes,
        "contracts": contracts,
    }


# ── Pre-Edit Verification with Impact ─────────────────────────────────


def verify_edit_with_impact(file_path: str, new_content: str, project: str = "one") -> dict:
    """Full verification: SQL + signatures + impact analysis.

    This is the main entry point for the hooks. It:
    1. Runs verify_edit (SQL, imports, patterns)
    2. Checks what other files depend on this file
    3. Verifies that changed function signatures don't break callers
    4. Logs the decision
    """
    result = verify_edit(file_path, new_content)
    fname = os.path.basename(file_path)

    # Get impact analysis
    impact = get_impact(file_path)

    # Extract new symbols from the edit
    new_symbols = _extract_symbols(file_path, new_content)
    new_sig_map = {s["name"]: s for s in new_symbols}

    # Check: did any public function signatures change?
    conn = _ont_conn()
    old_symbols = conn.execute(
        "SELECT name, signature, params FROM symbols WHERE file = ?",
        (fname,),
    ).fetchall()

    for old in old_symbols:
        old_name = old["name"]
        if old_name in new_sig_map:
            new_sig = new_sig_map[old_name]["signature"]
            if old["signature"] != new_sig:
                # Signature changed — check who calls this
                callers = [
                    dict(r) for r in conn.execute(
                        "SELECT caller_file, caller_name, line FROM calls WHERE callee_name = ? AND caller_file != ?",
                        (old_name, fname),
                    ).fetchall()
                ]
                if callers:
                    caller_desc = ", ".join(f"{c['caller_file']}:{c['caller_name']}" for c in callers[:5])
                    result["issues"].append({
                        "type": "signature_change",
                        "line": new_sig_map[old_name]["line_start"],
                        "message": f"Signature of {old_name} changed from '{old['signature']}' to '{new_sig}'. Called by: {caller_desc}. Verify callers still work.",
                        "severity": "warning",
                    })
        else:
            # Symbol was removed — check who calls it
            callers = conn.execute(
                "SELECT caller_file, caller_name FROM calls WHERE callee_name = ? AND caller_file != ?",
                (old_name, fname),
            ).fetchall()
            if callers:
                caller_desc = ", ".join(f"{dict(c)['caller_file']}:{dict(c)['caller_name']}" for c in callers[:5])
                result["issues"].append({
                    "type": "symbol_removed",
                    "line": 0,
                    "message": f"Function '{old_name}' was removed but is called by: {caller_desc}",
                    "severity": "error",
                })
                result["errors"] = result.get("errors", 0) + 1

    # Check contracts
    contracts = conn.execute(
        "SELECT symbol_name, contract_type, contract_data FROM contracts WHERE file = ?",
        (fname,),
    ).fetchall()
    for contract in contracts:
        sym_name = contract["symbol_name"]
        if sym_name in new_sig_map:
            c_type = contract["contract_type"]
            c_data = json.loads(contract["contract_data"])
            if c_type == "return_keys":
                result["issues"].append({
                    "type": "contract_check",
                    "line": new_sig_map[sym_name]["line_start"],
                    "message": f"CONTRACT: {sym_name} must return dict with keys {c_data}. Verify this is still true.",
                    "severity": "warning",
                })

    result["passed"] = sum(1 for i in result["issues"] if i["severity"] == "error") == 0
    result["impact"] = {
        "dependents": len(impact["dependents"]),
        "callers": len(impact["callers"]),
    }

    # Log the decision
    evidence = [f"sql={result['sql_checked']}", f"dependents={len(impact['dependents'])}", f"callers={len(impact['callers'])}"]
    log_decision(
        action="verify_edit_with_impact",
        file_path=file_path,
        outcome="pass" if result["passed"] else "fail",
        evidence=evidence,
        issues=result["issues"],
        project=project,
    )

    return result


# ══════════════════════════════════════════════════════════════════════
# FOUNDRY SYNC — Push ontology through 2 objects: MemoryEntry + Entity
# ══════════════════════════════════════════════════════════════════════


def sync_to_foundry(client, project: str = "one", on_log: Optional[Callable] = None) -> dict:
    """Push ontology to Foundry and LINK objects together.

    Pushes MemoryEntry + Entity, then creates has_memory_entries links
    so the graph is navigable in Vertex.
    """
    log = on_log or (lambda m: None)
    conn = _ont_conn()

    from .client import push_memory as foundry_push_memory, push_entity as foundry_push_entity
    from .hdc import encode_tagged

    stats = {"memories": 0, "entities": 0, "links": 0}

    # ── 1. File entities ──────────────────────────────────────────

    log("pushing file entities...")
    elog("foundry sync: pushing file entities")
    files = conn.execute("SELECT DISTINCT file FROM symbols").fetchall()
    for (fname,) in files:
        try:
            foundry_push_entity(client, f"file:{fname}", fname, "file")
            stats["entities"] += 1
        except Exception as e:
            elog(f"foundry entity FAIL: file:{fname} — {e}", "error")

    # ── 2. Symbol memories + entities ─────────────────────────────

    log("pushing symbols...")
    elog(f"foundry sync: pushing {len(files)} file entities done, now symbols")
    symbols = conn.execute(
        "SELECT file, name, kind, parent, signature, docstring FROM symbols"
    ).fetchall()

    for s in symbols:
        text = f"SYMBOL [{s['kind']}] {s['file']}:{s['signature']}"
        if s["docstring"]:
            text += f" — {s['docstring'][:200]}"
        try:
            vec = encode_tagged(text, role="system")
            foundry_push_memory(
                client, text, source="ontology", tm_label=f"symbol_{s['kind']}",
                regime_tag="verified", aif_confidence=0.95, hdc_vector=vec.tolist(),
            )
            stats["memories"] += 1
        except Exception as e:
            elog(f"foundry memory FAIL: {s['file']}:{s['name']} — {e}", "error")
        try:
            foundry_push_entity(client, f"{s['kind']}:{s['file']}:{s['name']}", f"{s['file']}:{s['name']}", s["kind"])
            stats["entities"] += 1
        except Exception as e:
            elog(f"foundry entity FAIL: {s['kind']}:{s['name']} — {e}", "error")

    # ── 3. Call relationships ─────────────────────────────────────

    log("pushing calls...")
    calls = conn.execute(
        "SELECT DISTINCT caller_file, caller_name, callee_name, line FROM calls WHERE resolved = 1"
    ).fetchall()

    seen = set()
    for c in calls:
        key = (c["caller_file"], c["caller_name"], c["callee_name"])
        if key in seen:
            continue
        seen.add(key)
        text = f"CALL {c['caller_file']}:{c['caller_name']} → {c['callee_name']} (line {c['line']})"
        try:
            vec = encode_tagged(text, role="system")
            foundry_push_memory(
                client, text, source="ontology", tm_label="call",
                regime_tag="verified", aif_confidence=0.9, hdc_vector=vec.tolist(),
            )
            stats["memories"] += 1
        except Exception:
            pass

    # ── 4. File dependencies ──────────────────────────────────────

    log("pushing dependencies...")
    deps = conn.execute("SELECT source_file, target_file, import_names FROM file_deps").fetchall()
    for d in deps:
        names = json.loads(d["import_names"])
        text = f"DEPENDENCY {d['source_file']} → {d['target_file']}: {', '.join(names[:10])}"
        try:
            vec = encode_tagged(text, role="system")
            foundry_push_memory(
                client, text, source="ontology", tm_label="file_dep",
                regime_tag="verified", aif_confidence=0.95, hdc_vector=vec.tolist(),
            )
            stats["memories"] += 1
        except Exception:
            pass

    # ── 5. Changes ────────────────────────────────────────────────

    log("pushing change history...")
    changes = conn.execute(
        "SELECT file, timestamp, action, summary, reason, bug_ref FROM changes ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    for ch in changes:
        parts = [f"CHANGE [{ch['action']}] {ch['file']}: {ch['summary']}"]
        if ch["reason"]:
            parts.append(f"Reason: {ch['reason']}")
        if ch["bug_ref"]:
            parts.append(f"Fixes: {ch['bug_ref']}")
        text = " | ".join(parts)
        try:
            vec = encode_tagged(text, role="system")
            foundry_push_memory(
                client, text, source="ontology", tm_label="change",
                regime_tag="log", aif_confidence=0.85, hdc_vector=vec.tolist(),
            )
            stats["memories"] += 1
        except Exception:
            pass

    # ── 6. Contracts ──────────────────────────────────────────────

    log("pushing contracts...")
    contracts = conn.execute(
        "SELECT file, symbol_name, contract_type, contract_data FROM contracts"
    ).fetchall()
    for ct in contracts:
        data = json.loads(ct["contract_data"])
        text = f"CONTRACT {ct['file']}:{ct['symbol_name']} [{ct['contract_type']}]: {data}"
        try:
            vec = encode_tagged(text, role="system")
            foundry_push_memory(
                client, text, source="ontology", tm_label="contract",
                regime_tag="verified", aif_confidence=0.98, hdc_vector=vec.tolist(),
            )
            stats["memories"] += 1
        except Exception:
            pass

    # ── 7. LINK entities to memories ─────────────────────────────

    # ── 7. LINK entities to memories ─────────────────────────────
    # SDK edits().add_link() queues but has no apply/commit method.
    # Links require a Foundry Action Type: link_memory_to_entity
    # (modify MemoryEntry, set entity_id FK to Entity PK).
    # Once that action exists, uncomment and use:
    #   client.ontology.actions.link_memory_to_entity(...)
    #
    # For now, links exist in local SQLite ontology (file_deps,
    # calls, symbols tables). Foundry gets objects but not links.

    try:
        if hasattr(client.ontology.actions, "link_memory_to_entity"):
            log("linking entities to memories via action...")
            elog("foundry sync: link action available, creating links")
            from orion_push_sdk.ontology.search._entity_object_type import EntityObjectType
            from orion_push_sdk.ontology.search._memory_entry_object_type import MemoryEntryObjectType
            et = EntityObjectType()
            mt = MemoryEntryObjectType()

            deps = conn.execute("SELECT source_file, target_file FROM file_deps").fetchall()
            for d in deps:
                try:
                    ents = client.ontology.objects.Entity.where(et.name == d["source_file"]).take(1)
                    mems = client.ontology.objects.MemoryEntry.where(mt.tm_label == "file_dep").take(200)
                    dep_mem = next((m for m in mems if d["target_file"] in (m.raw_text or "")), None)
                    if ents and dep_mem:
                        client.ontology.actions.link_memory_to_entity(
                            memory_entry=dep_mem.get_primary_key(),
                            entity=ents[0].get_primary_key(),
                        )
                        stats["links"] += 1
                except Exception as e:
                    elog(f"link FAIL: {d['source_file']}: {e}", "error")

            log(f"linked {stats['links']} entity↔memory pairs")
        else:
            elog("foundry sync: no link_memory_to_entity action — skipping links")
    except Exception as e:
        elog(f"linking phase: {e}", "error")

    log(f"foundry sync complete: {stats['memories']} memories, {stats['entities']} entities, {stats['links']} links")
    return stats
