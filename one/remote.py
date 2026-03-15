"""one remote — setup and manage integrations.

Handles Telegram, Discord, Slack bots, and webhook configuration.
Run `one remote` to interactively set up integrations, or use
`one remote --telegram` etc. for specific platforms.
"""

import os
import json
import threading
import time
from typing import Optional

CONFIG_DIR = os.path.expanduser("~/.one")
INTEGRATIONS_FILE = os.path.join(CONFIG_DIR, "integrations.json")


def _load_config() -> dict:
    if os.path.exists(INTEGRATIONS_FILE):
        with open(INTEGRATIONS_FILE) as f:
            return json.load(f)
    return {}


def _save_config(config: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(INTEGRATIONS_FILE, "w") as f:
        json.dump(config, f, indent=2)
    os.chmod(INTEGRATIONS_FILE, 0o600)


# ── Telegram ────────────────────────────────────────────────────────

def start_telegram(token: str, server_url: str = "http://localhost:4111") -> None:
    """Start the Telegram bot integration."""
    try:
        from telegram import Update, Bot
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
    except ImportError:
        print("Install python-telegram-bot: pip install python-telegram-bot")
        return

    import httpx

    def _api(method: str, path: str, data: dict = None) -> dict:
        if method == "GET":
            r = httpx.get(f"{server_url}{path}")
        else:
            r = httpx.post(f"{server_url}{path}", json=data or {})
        return r.json()

    async def cmd_start(update: Update, context) -> None:
        await update.message.reply_text(
            "one — persistent memory for AI coding tools\n\n"
            "Commands:\n"
            "/new <project> — create a session\n"
            "/auto <goal> — start autonomous mode\n"
            "/stop — stop auto\n"
            "/status — show all sessions\n"
            "/send <text> — send to active session\n"
            "/switch <project> — switch active project\n"
            "/rules — show rules\n"
            "/token <token> — refresh Foundry token"
        )

    async def cmd_new(update: Update, context) -> None:
        project = " ".join(context.args) if context.args else "default"
        result = _api("POST", "/api/session/new", {"project": project})
        await update.message.reply_text(f"Session created: {result.get('id')} ({project})")

    async def cmd_auto(update: Update, context) -> None:
        goal = " ".join(context.args) if context.args else ""
        if not goal:
            await update.message.reply_text("Usage: /auto <goal or file path>")
            return
        project = _get_active_project(update.effective_chat.id)
        result = _api("POST", "/api/auto", {"goal": goal, "project": project})
        await update.message.reply_text(f"Auto started: {result.get('session')}")

    async def cmd_stop(update: Update, context) -> None:
        project = _get_active_project(update.effective_chat.id)
        session = _api("GET", "/api/sessions")
        for s in session:
            if s.get("project") == project:
                _api("POST", "/api/stop", {"session_id": s["id"]})
                await update.message.reply_text("Stopped.")
                return
        await update.message.reply_text("No active session.")

    async def cmd_status(update: Update, context) -> None:
        sessions = _api("GET", "/api/sessions")
        if not sessions:
            await update.message.reply_text("No active sessions.")
            return
        lines = ["Sessions:"]
        for s in sessions:
            icon = "🟢" if s["status"] in ("ready", "auto") else "⚪"
            auto = " [AUTO]" if s.get("auto_running") else ""
            lines.append(f"{icon} {s['project']} — {s['status']}{auto} — {s['turns']}t ${s['cost']:.2f}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_switch(update: Update, context) -> None:
        project = " ".join(context.args) if context.args else ""
        if not project:
            await update.message.reply_text("Usage: /switch <project>")
            return
        _set_active_project(update.effective_chat.id, project)
        await update.message.reply_text(f"Switched to: {project}")

    async def cmd_token(update: Update, context) -> None:
        new_token = " ".join(context.args) if context.args else ""
        if not new_token:
            status = _api("GET", "/api/token")
            await update.message.reply_text(f"Token: {status.get('remaining_human', '?')}")
            return
        _api("POST", "/api/token/refresh", {"token": new_token})
        await update.message.reply_text("Token refreshed.")

    async def cmd_rules(update: Update, context) -> None:
        project = _get_active_project(update.effective_chat.id)
        sessions = _api("GET", "/api/sessions")
        for s in sessions:
            if s.get("project") == project:
                await update.message.reply_text(f"Rules for {project} — check terminal or /status")
                return
        await update.message.reply_text("No active session for this project.")

    async def handle_message(update: Update, context) -> None:
        text = update.message.text
        project = _get_active_project(update.effective_chat.id)
        result = _api("POST", "/api/send", {"text": text, "project": project})
        # Response comes through the event listener
        await update.message.reply_text(f"Sent to {project}. Waiting...")

    # Simple per-chat project tracking
    _active_projects: dict[int, str] = {}

    def _get_active_project(chat_id: int) -> str:
        return _active_projects.get(chat_id, "default")

    def _set_active_project(chat_id: int, project: str) -> None:
        _active_projects[chat_id] = project

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("auto", cmd_auto))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("switch", cmd_switch))
    app.add_handler(CommandHandler("token", cmd_token))
    app.add_handler(CommandHandler("rules", cmd_rules))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print(f"Telegram bot started. Server: {server_url}")
    app.run_polling()


# ── Discord ─────────────────────────────────────────────────────────

def start_discord(token: str, server_url: str = "http://localhost:4111") -> None:
    """Start the Discord bot integration."""
    try:
        import discord
    except ImportError:
        print("Install discord.py: pip install discord.py")
        return

    import httpx

    def _api(method, path, data=None):
        if method == "GET":
            return httpx.get(f"{server_url}{path}").json()
        return httpx.post(f"{server_url}{path}", json=data or {}).json()

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    _channel_projects: dict[int, str] = {}

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        text = message.content
        channel_id = message.channel.id

        if text.startswith("/new "):
            project = text[5:].strip()
            _channel_projects[channel_id] = project
            result = _api("POST", "/api/session/new", {"project": project})
            await message.channel.send(f"Session: {result.get('id')} ({project})")

        elif text.startswith("/auto "):
            goal = text[6:].strip()
            project = _channel_projects.get(channel_id, "default")
            _api("POST", "/api/auto", {"goal": goal, "project": project})
            await message.channel.send(f"Auto started for {project}")

        elif text == "/stop":
            project = _channel_projects.get(channel_id, "default")
            sessions = _api("GET", "/api/sessions")
            for s in sessions:
                if s.get("project") == project:
                    _api("POST", "/api/stop", {"session_id": s["id"]})
            await message.channel.send("Stopped.")

        elif text == "/status":
            sessions = _api("GET", "/api/sessions")
            lines = ["```"]
            for s in sessions:
                auto = " [AUTO]" if s.get("auto_running") else ""
                lines.append(f"{s['project']:20s} {s['status']}{auto} {s['turns']}t ${s['cost']:.2f}")
            lines.append("```")
            await message.channel.send("\n".join(lines))

        elif text.startswith("/"):
            await message.channel.send("Unknown command. Use /new, /auto, /stop, /status")

        else:
            project = _channel_projects.get(channel_id, "default")
            _api("POST", "/api/send", {"text": text, "project": project})
            await message.channel.send(f"Sent to {project}.")

    print(f"Discord bot started. Server: {server_url}")
    client.run(token)


# ── Slack ───────────────────────────────────────────────────────────

def start_slack(token: str, signing_secret: str, server_url: str = "http://localhost:4111") -> None:
    """Start the Slack bot integration."""
    try:
        from slack_bolt import App as SlackApp
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError:
        print("Install slack-bolt: pip install slack-bolt")
        return

    import httpx

    def _api(method, path, data=None):
        if method == "GET":
            return httpx.get(f"{server_url}{path}").json()
        return httpx.post(f"{server_url}{path}", json=data or {}).json()

    app = SlackApp(token=token, signing_secret=signing_secret)

    @app.message("")
    def handle_message(message, say):
        text = message.get("text", "")
        channel = message.get("channel", "")

        if text.startswith("/one "):
            cmd = text[5:].strip()
            if cmd.startswith("new "):
                project = cmd[4:].strip()
                result = _api("POST", "/api/session/new", {"project": project})
                say(f"Session: {result.get('id')} ({project})")
            elif cmd.startswith("auto "):
                goal = cmd[5:].strip()
                _api("POST", "/api/auto", {"goal": goal, "project": "default"})
                say("Auto started.")
            elif cmd == "stop":
                sessions = _api("GET", "/api/sessions")
                for s in sessions:
                    _api("POST", "/api/stop", {"session_id": s["id"]})
                say("Stopped.")
            elif cmd == "status":
                sessions = _api("GET", "/api/sessions")
                lines = []
                for s in sessions:
                    lines.append(f"• {s['project']} — {s['status']} — {s['turns']}t")
                say("\n".join(lines) or "No sessions.")
        else:
            _api("POST", "/api/send", {"text": text, "project": "default"})

    print(f"Slack bot started. Server: {server_url}")
    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN", ""))
    handler.start()


# ── Setup wizard ────────────────────────────────────────────────────

def setup_interactive() -> None:
    """Interactive setup for integrations."""
    config = _load_config()

    print("\none remote — integration setup\n")
    print("Available integrations:")
    print("  1. Telegram")
    print("  2. Discord")
    print("  3. Slack")
    print("  4. Webhook (REST API only)")
    print("  5. Show status")
    print("  6. Start server only")
    print()

    choice = input("Select (1-6): ").strip()

    if choice == "1":
        token = input("Telegram bot token: ").strip()
        if token:
            config["telegram"] = {"token": token}
            _save_config(config)
            print("Saved. Starting...")
            from .server import start_server
            start_server()
            start_telegram(token)

    elif choice == "2":
        token = input("Discord bot token: ").strip()
        if token:
            config["discord"] = {"token": token}
            _save_config(config)
            print("Saved. Starting...")
            from .server import start_server
            start_server()
            start_discord(token)

    elif choice == "3":
        token = input("Slack bot token: ").strip()
        secret = input("Slack signing secret: ").strip()
        if token and secret:
            config["slack"] = {"token": token, "signing_secret": secret}
            _save_config(config)
            print("Saved. Starting...")
            from .server import start_server
            start_server()
            start_slack(token, secret)

    elif choice == "4":
        print("\nREST API starts on port 4111.")
        print("POST /api/webhook with {\"project\": \"...\", \"text\": \"...\"}")
        print("GET /api/sessions for status")
        print("Starting server...")
        from .server import start_server
        server = start_server()
        print(f"Server running on http://localhost:4111")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    elif choice == "5":
        print(f"\nConfig: {INTEGRATIONS_FILE}")
        for k, v in config.items():
            token_preview = v.get("token", "")[:10] + "..." if v.get("token") else "none"
            print(f"  {k}: {token_preview}")

    elif choice == "6":
        from .server import start_server
        server = start_server()
        print(f"Server running on http://localhost:4111")
        print("Endpoints:")
        print("  GET  /api/health")
        print("  GET  /api/sessions")
        print("  GET  /api/token")
        print("  POST /api/session/new  {project, model}")
        print("  POST /api/send         {session_id|project, text}")
        print("  POST /api/auto         {session_id|project, goal}")
        print("  POST /api/stop         {session_id}")
        print("  POST /api/webhook      {project, text}")
        print("  POST /api/token/refresh {token}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def start_configured() -> None:
    """Start all configured integrations."""
    config = _load_config()
    if not config:
        setup_interactive()
        return

    from .server import start_server
    start_server()
    print("Server running on http://localhost:4111")

    threads = []

    if "telegram" in config:
        t = threading.Thread(target=start_telegram, args=(config["telegram"]["token"],), daemon=True)
        t.start()
        threads.append(t)

    if "discord" in config:
        t = threading.Thread(target=start_discord, args=(config["discord"]["token"],), daemon=True)
        t.start()
        threads.append(t)

    if "slack" in config:
        t = threading.Thread(
            target=start_slack,
            args=(config["slack"]["token"], config["slack"]["signing_secret"]),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print(f"Started {len(threads)} integration(s).")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
