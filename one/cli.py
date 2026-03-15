#!/usr/bin/env python3
"""Command-line entry point for One."""

import argparse
import os
import sys


def main():
    # Check for subcommands first
    if len(sys.argv) > 1 and sys.argv[1] == "remote":
        from .remote import setup_interactive, start_configured
        if len(sys.argv) > 2 and sys.argv[2] == "--start":
            start_configured()
        else:
            setup_interactive()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        from .server import start_server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 4111
        server = start_server(port=port)
        print(f"one server running on http://localhost:{port}")
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    from .proxy import ClaudeProxy
    from .app import OneApp

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
    parser.add_argument("--yolo", action="store_true", help="bypass all permissions")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--append-system-prompt", default=None)
    parser.add_argument("--allowed-tools", nargs="*", default=None)
    parser.add_argument("--disallowed-tools", nargs="*", default=None)

    args = parser.parse_args()

    from .init import run_init, get_project_name
    env = run_init()

    if not env["claude"]:
        print("error: claude CLI not found — install from https://claude.ai/download")
        sys.exit(1)

    cwd = args.dir or os.getcwd()
    project = get_project_name(cwd)
    from . import store
    store.set_project(project)

    foundry = None
    if not args.no_foundry and env["foundry"]:
        try:
            from .client import get_client
            foundry = get_client()
        except Exception:
            pass

    perm_mode = "bypassPermissions" if args.yolo else args.permission_mode

    proxy = ClaudeProxy(
        model=args.model,
        cwd=cwd,
        session_id=args.session or args.resume,
        resume=args.resume is not None,
        continue_last=args.continue_last,
        permission_mode=perm_mode,
        system_prompt=args.system_prompt,
        append_system_prompt=args.append_system_prompt,
        allowed_tools=args.allowed_tools,
        disallowed_tools=args.disallowed_tools,
    )

    app = OneApp(proxy, foundry_client=foundry, project=project)
    app.run()


if __name__ == "__main__":
    main()
