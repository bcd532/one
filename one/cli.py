#!/usr/bin/env python3
"""Command-line entry point for One."""

import argparse
import os

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

    foundry = None
    if not args.no_foundry:
        try:
            from .client import get_client
            foundry = get_client()
        except Exception:
            pass

    proxy = ClaudeProxy(
        model=args.model,
        cwd=args.dir or os.getcwd(),
        session_id=args.session or args.resume,
        resume=args.resume is not None,
        continue_last=args.continue_last,
        permission_mode=args.permission_mode,
        system_prompt=args.system_prompt,
        append_system_prompt=args.append_system_prompt,
        allowed_tools=args.allowed_tools,
        disallowed_tools=args.disallowed_tools,
    )

    app = OneApp(proxy, foundry_client=foundry)
    app.run()


if __name__ == "__main__":
    main()
