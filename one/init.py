"""First-run initialization and environment detection."""

import os
import subprocess
import shutil

DB_DIR = os.path.expanduser("~/.one")


def detect_claude() -> tuple[bool, str]:
    """Check if Claude CLI is installed and accessible."""
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, ""


def detect_ollama() -> tuple[bool, str]:
    """Check if ollama is installed."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, ""


def detect_gemma() -> bool:
    """Check if gemma3:4b is pulled in ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5,
        )
        return "gemma3" in result.stdout
    except Exception:
        return False


def detect_foundry() -> bool:
    """Check if Foundry credentials are configured."""
    if os.environ.get("FOUNDRY_TOKEN"):
        return True
    token_file = os.path.expanduser("~/.one/token")
    return os.path.exists(token_file)


def get_project_name(cwd: str = None) -> str:
    """Determine the project name from the git root or directory name."""
    cwd = cwd or os.getcwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=3, cwd=cwd,
        )
        if result.returncode == 0:
            return os.path.basename(result.stdout.strip())
    except Exception:
        pass
    return os.path.basename(cwd)


def ensure_dirs() -> None:
    """Create ~/.one/ directory structure if missing."""
    os.makedirs(DB_DIR, exist_ok=True)


def run_init() -> dict:
    """Run full initialization check. Returns status dict."""
    ensure_dirs()

    claude_ok, claude_ver = detect_claude()
    ollama_ok, ollama_ver = detect_ollama()
    gemma_ok = detect_gemma() if ollama_ok else False
    foundry_ok = detect_foundry()
    project = get_project_name()

    return {
        "claude": claude_ok,
        "claude_version": claude_ver,
        "ollama": ollama_ok,
        "gemma": gemma_ok,
        "foundry": foundry_ok,
        "project": project,
        "db_dir": DB_DIR,
    }
