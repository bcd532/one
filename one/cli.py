#!/usr/bin/env python3
"""Command-line entry point for One."""

import argparse
import os
import sys

from .proxy import ClaudeProxy
from .app import OneApp


def main():
    parser = argparse.ArgumentParser(description="one — persistent-memory Claude interface")
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("-m", "--model", default="opus")
    parser.add_argument("-d", "--dir", default=None)
    parser.add_argument("-c", "--continue", dest="continue_last", action="store_true")
    parser.add_argument("-r", "--resume", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--no-foundry", action="store_true")
    parser.add_argument("--no-recall", action="store_true")
    parser.add_argument("--permission-mode", default="acceptEdits",
                        choices=["acceptEdits", "bypassPermissions", "default", "dontAsk", "plan", "auto"])
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--append-system-prompt", default=None)
    parser.add_argument("--allowed-tools", nargs="*", default=None)
    parser.add_argument("--disallowed-tools", nargs="*", default=None)

    args = parser.parse_args()

    # Detect environment
    from .init import run_init, get_project_name
    env = run_init()

    if not env["claude"]:
        print("error: claude CLI not found — install from https://claude.ai/download")
        sys.exit(1)

    # Set project scope
    cwd = args.dir or os.getcwd()
    project = get_project_name(cwd)
    from . import store
    store.set_project(project)

    # Foundry (optional)
    foundry = None
    if not args.no_foundry and env["foundry"]:
        try:
            from .client import get_client
            foundry = get_client()
        except Exception:
            pass

    proxy = ClaudeProxy(
        model=args.model,
        cwd=cwd,
        session_id=args.session or args.resume,
        resume=args.resume is not None,
        continue_last=args.continue_last,
        permission_mode=args.permission_mode,
        system_prompt=args.system_prompt,
        append_system_prompt=args.append_system_prompt,
        allowed_tools=args.allowed_tools,
        disallowed_tools=args.disallowed_tools,
    )

    app = OneApp(proxy, foundry_client=foundry, project=project)
    app.run()


if __name__ == "__main__":
    main()
