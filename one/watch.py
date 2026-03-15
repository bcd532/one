"""Watch mode — monitors a directory for file changes and auto-logs diffs.

Uses simple mtime polling (no external dependencies). Detects meaningful
file changes, computes diffs, and stores them as memories with entity links.
"""

import os
import difflib
import threading
from typing import Optional

# Directories and extensions to ignore
IGNORE_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".pytest_cache", ".tox", "dist", "build"}
IGNORE_FILES = {".DS_Store"}
IGNORE_EXTS = {".pyc", ".pyo", ".so", ".o", ".a", ".dylib"}

_watcher_thread: Optional[threading.Thread] = None
_watcher_stop = threading.Event()
_watcher_dir: Optional[str] = None


def _should_ignore(path: str) -> bool:
    """Check whether a file path should be skipped."""
    parts = path.split(os.sep)
    for part in parts:
        if part in IGNORE_DIRS:
            return True
    basename = os.path.basename(path)
    if basename in IGNORE_FILES:
        return True
    _, ext = os.path.splitext(basename)
    if ext in IGNORE_EXTS:
        return True
    return False


def _is_text_file(path: str) -> bool:
    """Heuristic check for text files (skip binary)."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(512)
        if b"\x00" in chunk:
            return False
        return True
    except (OSError, PermissionError):
        return False


def _compute_diff(old_content: str, new_content: str, filename: str) -> str:
    """Compute a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{filename}", tofile=f"b/{filename}", lineterm="")
    return "".join(diff)


def _is_meaningful_diff(diff_text: str) -> bool:
    """Return True if the diff contains non-whitespace changes."""
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            stripped = line[1:].strip()
            if stripped:
                return True
        if line.startswith("-") and not line.startswith("---"):
            stripped = line[1:].strip()
            if stripped:
                return True
    return False


def _scan_directory(directory: str) -> dict[str, tuple[float, str]]:
    """Scan a directory tree and return {relpath: (mtime, content)} for text files."""
    result = {}
    for root, dirs, files in os.walk(directory):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, directory)
            if _should_ignore(relpath):
                continue
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                continue
            if not _is_text_file(fpath):
                continue
            try:
                with open(fpath, "r", errors="replace") as f:
                    content = f.read()
            except (OSError, PermissionError):
                continue
            result[relpath] = (mtime, content)
    return result


def _watch_loop(directory: str, project: str, backend) -> None:
    """Main polling loop. Runs in a background thread."""
    # Initial snapshot
    snapshot: dict[str, tuple[float, str]] = _scan_directory(directory)

    while not _watcher_stop.is_set():
        _watcher_stop.wait(timeout=2.0)
        if _watcher_stop.is_set():
            break

        new_snapshot = _scan_directory(directory)

        for relpath, (new_mtime, new_content) in new_snapshot.items():
            old = snapshot.get(relpath)
            if old is None:
                # New file
                _log_change(backend, project, directory, relpath, "", new_content, "created")
            else:
                old_mtime, old_content = old
                if new_mtime > old_mtime and new_content != old_content:
                    _log_change(backend, project, directory, relpath, old_content, new_content, "modified")

        # Detect deletions
        for relpath in set(snapshot.keys()) - set(new_snapshot.keys()):
            old_mtime, old_content = snapshot[relpath]
            _log_change(backend, project, directory, relpath, old_content, "", "deleted")

        snapshot = new_snapshot


def _log_change(backend, project: str, directory: str, relpath: str, old_content: str, new_content: str, change_type: str) -> None:
    """Log a file change as a memory with entity links."""
    filename = os.path.basename(relpath)
    diff_text = _compute_diff(old_content, new_content, relpath)

    if change_type == "modified" and not _is_meaningful_diff(diff_text):
        return

    # Truncate diff for storage
    max_diff = 2000
    if len(diff_text) > max_diff:
        diff_text = diff_text[:max_diff] + f"\n... ({len(diff_text) - max_diff} chars truncated)"

    summary = f"[{change_type}] {relpath}"
    if diff_text:
        summary += f"\n{diff_text}"

    try:
        mid = backend.push_memory(
            raw_text=summary,
            source="watch",
            tm_label="file_change",
            regime_tag="default",
            aif_confidence=0.3,
        )
    except Exception:
        return

    # Link the file as an entity
    try:
        from . import store
        full_path = os.path.join(directory, relpath)
        eid = store.ensure_entity({"name": full_path, "type": "file"})
        if mid:
            store.link_memory_entity(mid, eid)
    except Exception:
        pass


def start_watch(directory: str, project: str, backend) -> str:
    """Start watching a directory for file changes in a background thread.

    Returns a status message.
    """
    global _watcher_thread, _watcher_dir

    directory = os.path.abspath(os.path.expanduser(directory))
    if not os.path.isdir(directory):
        return f"not a directory: {directory}"

    if _watcher_thread is not None and _watcher_thread.is_alive():
        return f"already watching {_watcher_dir} — /unwatch first"

    _watcher_stop.clear()
    _watcher_dir = directory
    _watcher_thread = threading.Thread(
        target=_watch_loop,
        args=(directory, project, backend),
        daemon=True,
    )
    _watcher_thread.start()
    return f"watching {directory}"


def stop_watch() -> str:
    """Stop the file watcher if running. Returns a status message."""
    global _watcher_thread, _watcher_dir

    if _watcher_thread is None or not _watcher_thread.is_alive():
        return "not watching anything"

    _watcher_stop.set()
    _watcher_thread.join(timeout=5.0)
    stopped_dir = _watcher_dir
    _watcher_thread = None
    _watcher_dir = None
    return f"stopped watching {stopped_dir}"


def is_watching() -> bool:
    """Return True if the watcher is currently active."""
    return _watcher_thread is not None and _watcher_thread.is_alive()


def watching_directory() -> Optional[str]:
    """Return the directory being watched, or None."""
    if is_watching():
        return _watcher_dir
    return None
