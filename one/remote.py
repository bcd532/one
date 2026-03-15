"""one remote — guided setup and management for integrations.

Provides step-by-step setup wizards for Telegram, Discord, Slack,
and webhook integrations with animated branding and connection validation.
"""

import os
import json
import threading
import time
import sys
from typing import Optional

CONFIG_DIR = os.path.expanduser("~/.one")
INTEGRATIONS_FILE = os.path.join(CONFIG_DIR, "integrations.json")

# ANSI
B = "\033[1m"
D = "\033[2m"
R = "\033[0m"
C = "\033[36m"
G = "\033[32m"
Y = "\033[33m"
RE = "\033[31m"
M = "\033[35m"
GR = "\033[90m"
CLR = "\033[2J\033[H"
HIDE = "\033[?25l"
SHOW = "\033[?25h"

LOGO_FRAMES = [
    f"""{GR}
     ██████╗ ███╗   ██╗███████╗
    ██╔═══██╗████╗  ██║██╔════╝
    ██║   ██║██╔██╗ ██║█████╗
    ██║   ██║██║╚██╗██║██╔══╝
    ╚██████╔╝██║ ╚████║███████╗
     ╚═════╝ ╚═╝  ╚═══╝╚══════╝{R}""",
    f"""{D}{C}
     ██████╗ ███╗   ██╗███████╗
    ██╔═══██╗████╗  ██║██╔════╝
    ██║   ██║██╔██╗ ██║█████╗
    ██║   ██║██║╚██╗██║██╔══╝
    ╚██████╔╝██║ ╚████║███████╗
     ╚═════╝ ╚═╝  ╚═══╝╚══════╝{R}""",
    f"""{C}
     ██████╗ ███╗   ██╗███████╗
    ██╔═══██╗████╗  ██║██╔════╝
    ██║   ██║██╔██╗ ██║█████╗
    ██║   ██║██║╚██╗██║██╔══╝
    ╚██████╔╝██║ ╚████║███████╗
     ╚═════╝ ╚═╝  ╚═══╝╚══════╝{R}""",
    f"""{B}{C}
     ██████╗ ███╗   ██╗███████╗
    ██╔═══██╗████╗  ██║██╔════╝
    ██║   ██║██╔██╗ ██║█████╗
    ██║   ██║██║╚██╗██║██╔══╝
    ╚██████╔╝██║ ╚████║███████╗
     ╚═════╝ ╚═╝  ╚═══╝╚══════╝{R}""",
]


def _animate_logo():
    """Fade in the logo."""
    sys.stdout.write(HIDE)
    for frame in LOGO_FRAMES:
        sys.stdout.write(CLR)
        print(frame)
        print()
        sys.stdout.flush()
        time.sleep(0.15)
    sys.stdout.write(SHOW)
    print(f"    {B}{C}one{R} {D}remote{R}")
    print(f"    {D}persistent memory everywhere{R}")
    print()


def _hr():
    print(f"    {D}{'─' * 50}{R}")


def _step(num, text):
    print(f"    {C}{B}{num}.{R} {text}")


def _ok(text):
    print(f"    {G}✓{R} {text}")


def _fail(text):
    print(f"    {RE}✗{R} {text}")


def _info(text):
    print(f"    {D}{text}{R}")


def _prompt(text) -> str:
    return input(f"    {Y}>{R} {text}: ").strip()


def _spinner(text, duration=2):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end = time.time() + duration
    sys.stdout.write(HIDE)
    i = 0
    while time.time() < end:
        sys.stdout.write(f"\r    {C}{frames[i % len(frames)]}{R} {D}{text}{R}  ")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1
    sys.stdout.write(f"\r    {'':60}\r")
    sys.stdout.write(SHOW)
    sys.stdout.flush()


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


# ── Main menu ───────────────────────────────────────────────────────

def setup_interactive() -> None:
    _animate_logo()
    _hr()

    config = _load_config()

    # Show current status
    if config:
        print(f"    {D}configured:{R}")
        for k in config:
            print(f"      {G}●{R} {k}")
        print()

    print(f"    {B}integrations{R}")
    print()
    print(f"      {C}1{R}  Telegram")
    print(f"      {C}2{R}  Discord")
    print(f"      {C}3{R}  Slack")
    print(f"      {C}4{R}  Webhook / REST API")
    print(f"      {C}5{R}  Start all configured")
    print(f"      {C}6{R}  Status")
    print()

    choice = _prompt("select")

    if choice == "1":
        _setup_telegram(config)
    elif choice == "2":
        _setup_discord(config)
    elif choice == "3":
        _setup_slack(config)
    elif choice == "4":
        _setup_webhook()
    elif choice == "5":
        start_configured()
    elif choice == "6":
        _show_status(config)
    else:
        print(f"    {D}invalid choice{R}")


# ── Telegram ────────────────────────────────────────────────────────

def _setup_telegram(config: dict) -> None:
    print()
    print(f"    {B}{C}Telegram Setup{R}")
    _hr()
    print()
    _step(1, "Open Telegram and message @BotFather")
    _step(2, "Send /newbot and follow the prompts")
    _step(3, "Name your bot (e.g. 'one memory')")
    _step(4, "Copy the bot token BotFather gives you")
    print()
    _info("It looks like: 7104583921:AAH3kx9...")
    print()

    token = _prompt("paste bot token")
    if not token:
        _fail("no token provided")
        return

    _spinner("validating token")

    # Validate
    try:
        import httpx
        r = httpx.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        data = r.json()
        if data.get("ok"):
            bot_name = data["result"].get("username", "?")
            _ok(f"connected as @{bot_name}")
        else:
            _fail(f"invalid token: {data.get('description', 'unknown error')}")
            return
    except ImportError:
        _info("httpx not installed — skipping validation")
        _info("run: pip install httpx")
    except Exception as e:
        _fail(f"connection failed: {e}")
        return

    config["telegram"] = {"token": token}
    _save_config(config)

    print()
    _ok("saved to ~/.one/integrations.json")
    print()
    _step(5, "Add the bot to a group or message it directly")
    _step(6, f"Run: {C}one remote --start{R}")
    print()
    _info("commands in Telegram:")
    _info("  /new <project>   — create a session")
    _info("  /auto <goal>     — start autonomous mode")
    _info("  /stop            — stop auto")
    _info("  /status          — show all sessions")
    _info("  /switch <project>— switch active project")
    _info("  /token <token>   — refresh Foundry token")
    print()

    start = _prompt("start the bot now? (y/n)")
    if start.lower() in ("y", "yes"):
        _start_with_server("telegram", config)


# ── Discord ─────────────────────────────────────────────────────────

def _setup_discord(config: dict) -> None:
    print()
    print(f"    {B}{C}Discord Setup{R}")
    _hr()
    print()
    _step(1, f"Go to {C}https://discord.com/developers/applications{R}")
    _step(2, "Click 'New Application' → name it 'one'")
    _step(3, "Go to Bot → click 'Reset Token' → copy it")
    _step(4, "Enable 'Message Content Intent' under Privileged Intents")
    _step(5, "Go to OAuth2 → URL Generator")
    _info("     select scope: bot")
    _info("     select permissions: Send Messages, Read Message History")
    _step(6, "Copy the invite URL and open it to add bot to your server")
    print()

    token = _prompt("paste bot token")
    if not token:
        _fail("no token provided")
        return

    _spinner("validating token")

    try:
        import httpx
        r = httpx.get("https://discord.com/api/v10/users/@me",
                      headers={"Authorization": f"Bot {token}"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            bot_name = data.get("username", "?")
            _ok(f"connected as {bot_name}#{data.get('discriminator', '0')}")
        else:
            _fail(f"invalid token (HTTP {r.status_code})")
            return
    except ImportError:
        _info("httpx not installed — skipping validation")
    except Exception as e:
        _fail(f"connection failed: {e}")
        return

    config["discord"] = {"token": token}
    _save_config(config)

    print()
    _ok("saved to ~/.one/integrations.json")
    print()
    _info("commands in Discord:")
    _info("  /new <project>   — create a session")
    _info("  /auto <goal>     — start autonomous mode")
    _info("  /stop            — stop auto")
    _info("  /status          — show all sessions")
    print()

    start = _prompt("start the bot now? (y/n)")
    if start.lower() in ("y", "yes"):
        _start_with_server("discord", config)


# ── Slack ───────────────────────────────────────────────────────────

def _setup_slack(config: dict) -> None:
    print()
    print(f"    {B}{C}Slack Setup{R}")
    _hr()
    print()
    _step(1, f"Go to {C}https://api.slack.com/apps{R}")
    _step(2, "Click 'Create New App' → 'From scratch'")
    _step(3, "Name it 'one', select your workspace")
    _step(4, "Go to 'Socket Mode' → enable it → generate an app token")
    _step(5, "Go to 'OAuth & Permissions' → add scopes:")
    _info("     chat:write, channels:history, groups:history, im:history")
    _step(6, "Go to 'Event Subscriptions' → enable → subscribe to:")
    _info("     message.channels, message.groups, message.im")
    _step(7, "Install app to workspace")
    _step(8, "Copy the Bot User OAuth Token (xoxb-...)")
    _step(9, "Copy the Signing Secret from Basic Information")
    print()

    token = _prompt("bot token (xoxb-...)")
    if not token:
        _fail("no token provided")
        return

    secret = _prompt("signing secret")
    if not secret:
        _fail("no signing secret provided")
        return

    _spinner("validating")

    try:
        import httpx
        r = httpx.post("https://slack.com/api/auth.test",
                       headers={"Authorization": f"Bearer {token}"}, timeout=10)
        data = r.json()
        if data.get("ok"):
            _ok(f"connected as {data.get('user', '?')} in {data.get('team', '?')}")
        else:
            _fail(f"invalid: {data.get('error', 'unknown')}")
            return
    except ImportError:
        _info("httpx not installed — skipping validation")
    except Exception as e:
        _fail(f"connection failed: {e}")
        return

    config["slack"] = {"token": token, "signing_secret": secret}
    _save_config(config)

    print()
    _ok("saved to ~/.one/integrations.json")
    print()
    _info("commands in Slack:")
    _info("  /one new <project>  — create a session")
    _info("  /one auto <goal>    — start autonomous mode")
    _info("  /one stop           — stop auto")
    _info("  /one status         — show all sessions")
    print()

    start = _prompt("start the bot now? (y/n)")
    if start.lower() in ("y", "yes"):
        _start_with_server("slack", config)


# ── Webhook ─────────────────────────────────────────────────────────

def _setup_webhook() -> None:
    print()
    print(f"    {B}{C}Webhook / REST API{R}")
    _hr()
    print()
    _info("The one server exposes a REST API on port 4111.")
    _info("Any HTTP client can connect — no bot tokens needed.")
    print()
    print(f"    {B}endpoints:{R}")
    print(f"      {G}GET{R}  /api/health              server status")
    print(f"      {G}GET{R}  /api/sessions             list sessions")
    print(f"      {G}GET{R}  /api/token                token status")
    print(f"      {Y}POST{R} /api/session/new          create session")
    print(f"      {Y}POST{R} /api/send                 send message")
    print(f"      {Y}POST{R} /api/auto                 start auto mode")
    print(f"      {Y}POST{R} /api/stop                 stop auto")
    print(f"      {Y}POST{R} /api/webhook              generic webhook")
    print(f"      {Y}POST{R} /api/token/refresh        refresh token")
    print()
    _info("example:")
    print(f"    {D}curl -X POST localhost:4111/api/send \\{R}")
    print(f"    {D}  -d '{{\"project\":\"myapp\",\"text\":\"fix the auth bug\"}}'{R}")
    print()

    start = _prompt("start the server now? (y/n)")
    if start.lower() in ("y", "yes"):
        from .server import start_server
        start_server()
        _ok("server running on http://localhost:4111")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


# ── Status ──────────────────────────────────────────────────────────

def _show_status(config: dict) -> None:
    print()
    print(f"    {B}status{R}")
    _hr()
    print()

    if not config:
        _info("no integrations configured")
        _info(f"run: {C}one remote{R}")
        return

    for name, data in config.items():
        token = data.get("token", "")
        preview = token[:15] + "..." if len(token) > 15 else token
        print(f"    {G}●{R} {B}{name}{R}")
        print(f"      token: {D}{preview}{R}")
        if name == "slack":
            print(f"      secret: {D}{data.get('signing_secret', '')[:10]}...{R}")
    print()

    # Token status
    try:
        from .client import token_status
        status = token_status()
        if status["expired"]:
            print(f"    {RE}●{R} foundry token: {RE}expired{R}")
        elif status["warning"]:
            print(f"    {Y}●{R} foundry token: {Y}{status['remaining_human']} remaining{R}")
        elif status["ok"]:
            print(f"    {G}●{R} foundry token: {G}{status['remaining_human']} remaining{R}")
    except Exception:
        print(f"    {GR}○{R} foundry: not configured")
    print()


# ── Start helpers ───────────────────────────────────────────────────

def _start_with_server(integration: str, config: dict) -> None:
    from .server import start_server
    start_server()
    _ok("API server running on :4111")
    print()

    if integration == "telegram":
        _ok("starting Telegram bot...")
        start_telegram(config["telegram"]["token"])
    elif integration == "discord":
        _ok("starting Discord bot...")
        start_discord(config["discord"]["token"])
    elif integration == "slack":
        _ok("starting Slack bot...")
        start_slack(
            config["slack"]["token"],
            config["slack"]["signing_secret"],
        )


def start_configured() -> None:
    """Start all configured integrations with the API server."""
    config = _load_config()
    if not config:
        print()
        _fail("no integrations configured")
        _info(f"run: {C}one remote{R}")
        return

    _animate_logo()

    from .server import start_server
    start_server()
    _ok("API server on :4111")

    threads = []

    if "telegram" in config:
        t = threading.Thread(
            target=start_telegram,
            args=(config["telegram"]["token"],),
            daemon=True,
        )
        t.start()
        threads.append(("telegram", t))
        _ok("Telegram bot started")

    if "discord" in config:
        t = threading.Thread(
            target=start_discord,
            args=(config["discord"]["token"],),
            daemon=True,
        )
        t.start()
        threads.append(("discord", t))
        _ok("Discord bot started")

    if "slack" in config:
        t = threading.Thread(
            target=start_slack,
            args=(config["slack"]["token"], config["slack"]["signing_secret"]),
            daemon=True,
        )
        t.start()
        threads.append(("slack", t))
        _ok("Slack bot started")

    print()
    _hr()
    print(f"    {G}{len(threads)} integration(s) running{R}")
    _info("press ctrl+c to stop")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n    {D}shutting down{R}")


# ── Bot implementations ────────────────────────────────────────────
# (These are the actual bot runners — imported by the start functions)

def start_telegram(token: str, server_url: str = "http://localhost:4111") -> None:
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
    except ImportError:
        _fail("missing: pip install one[telegram]")
        return

    import httpx

    def _api(method, path, data=None):
        if method == "GET":
            return httpx.get(f"{server_url}{path}").json()
        return httpx.post(f"{server_url}{path}", json=data or {}).json()

    _active: dict[int, str] = {}

    async def cmd_start(update: Update, context) -> None:
        await update.message.reply_text(
            "one — persistent memory for AI coding tools\n\n"
            "/new <project> — create a session\n"
            "/auto <goal> — autonomous mode\n"
            "/stop — stop auto\n"
            "/status — all sessions\n"
            "/switch <project> — switch project\n"
            "/token <token> — refresh Foundry token"
        )

    async def cmd_new(update: Update, context) -> None:
        project = " ".join(context.args) if context.args else "default"
        _active[update.effective_chat.id] = project
        result = _api("POST", "/api/session/new", {"project": project})
        await update.message.reply_text(f"Session: {result.get('id')} ({project})")

    async def cmd_auto(update: Update, context) -> None:
        goal = " ".join(context.args) if context.args else ""
        if not goal:
            await update.message.reply_text("Usage: /auto <goal>")
            return
        project = _active.get(update.effective_chat.id, "default")
        result = _api("POST", "/api/auto", {"goal": goal, "project": project})
        await update.message.reply_text(f"Auto started: {result.get('session')}")

    async def cmd_stop(update: Update, context) -> None:
        project = _active.get(update.effective_chat.id, "default")
        for s in _api("GET", "/api/sessions"):
            if s.get("project") == project:
                _api("POST", "/api/stop", {"session_id": s["id"]})
                await update.message.reply_text("Stopped.")
                return
        await update.message.reply_text("No active session.")

    async def cmd_status(update: Update, context) -> None:
        sessions = _api("GET", "/api/sessions")
        if not sessions:
            await update.message.reply_text("No sessions.")
            return
        lines = []
        for s in sessions:
            icon = "🟢" if s["status"] in ("ready", "auto") else "⚪"
            auto = " [AUTO]" if s.get("auto_running") else ""
            lines.append(f"{icon} {s['project']}{auto} — {s['turns']}t ${s['cost']:.2f}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_switch(update: Update, context) -> None:
        project = " ".join(context.args) if context.args else ""
        if not project:
            await update.message.reply_text("Usage: /switch <project>")
            return
        _active[update.effective_chat.id] = project
        await update.message.reply_text(f"Switched to: {project}")

    async def cmd_token(update: Update, context) -> None:
        new_token = " ".join(context.args) if context.args else ""
        if not new_token:
            status = _api("GET", "/api/token")
            await update.message.reply_text(f"Token: {status.get('remaining_human', '?')}")
            return
        _api("POST", "/api/token/refresh", {"token": new_token})
        await update.message.reply_text("Refreshed.")

    async def handle_msg(update: Update, context) -> None:
        project = _active.get(update.effective_chat.id, "default")
        _api("POST", "/api/send", {"text": update.message.text, "project": project})

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("auto", cmd_auto))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("switch", cmd_switch))
    app.add_handler(CommandHandler("token", cmd_token))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))
    app.run_polling()


def start_discord(token: str, server_url: str = "http://localhost:4111") -> None:
    try:
        import discord
    except ImportError:
        _fail("missing: pip install one[discord]")
        return

    import httpx

    def _api(method, path, data=None):
        if method == "GET":
            return httpx.get(f"{server_url}{path}").json()
        return httpx.post(f"{server_url}{path}", json=data or {}).json()

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    _channels: dict[int, str] = {}

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        text = message.content
        cid = message.channel.id

        if text.startswith("/new "):
            project = text[5:].strip()
            _channels[cid] = project
            r = _api("POST", "/api/session/new", {"project": project})
            await message.channel.send(f"Session: {r.get('id')} ({project})")
        elif text.startswith("/auto "):
            goal = text[6:].strip()
            project = _channels.get(cid, "default")
            _api("POST", "/api/auto", {"goal": goal, "project": project})
            await message.channel.send(f"Auto started for {project}")
        elif text == "/stop":
            for s in _api("GET", "/api/sessions"):
                if s.get("project") == _channels.get(cid, "default"):
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
        elif not text.startswith("/"):
            project = _channels.get(cid, "default")
            _api("POST", "/api/send", {"text": text, "project": project})

    client.run(token)


def start_slack(token: str, signing_secret: str, server_url: str = "http://localhost:4111") -> None:
    try:
        from slack_bolt import App as SlackApp
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError:
        _fail("missing: pip install one[slack]")
        return

    import httpx

    def _api(method, path, data=None):
        if method == "GET":
            return httpx.get(f"{server_url}{path}").json()
        return httpx.post(f"{server_url}{path}", json=data or {}).json()

    app = SlackApp(token=token, signing_secret=signing_secret)

    @app.message("")
    def handle(message, say):
        text = message.get("text", "")
        if text.startswith("/one "):
            cmd = text[5:].strip()
            if cmd.startswith("new "):
                r = _api("POST", "/api/session/new", {"project": cmd[4:].strip()})
                say(f"Session: {r.get('id')}")
            elif cmd.startswith("auto "):
                _api("POST", "/api/auto", {"goal": cmd[5:].strip()})
                say("Auto started.")
            elif cmd == "stop":
                for s in _api("GET", "/api/sessions"):
                    _api("POST", "/api/stop", {"session_id": s["id"]})
                say("Stopped.")
            elif cmd == "status":
                sessions = _api("GET", "/api/sessions")
                say("\n".join(f"• {s['project']} — {s['status']}" for s in sessions) or "No sessions.")
        else:
            _api("POST", "/api/send", {"text": text, "project": "default"})

    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN", ""))
    handler.start()
