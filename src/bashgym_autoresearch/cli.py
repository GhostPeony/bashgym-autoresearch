"""Command-line interface for humans and agents.

Local commands (init, token, serve) work on the state directory directly. All
other commands go through the HTTP API with the token in ``BGAR_TOKEN``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

from bashgym_autoresearch import __version__

DEFAULT_URL = "http://127.0.0.1:8765"
HUMAN_TOKEN_FILENAME = "human.token"


def _home(args: argparse.Namespace) -> Path:
    return Path(args.home or os.environ.get("BGAR_HOME") or Path.home() / ".bashgym-autoresearch")


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _client(args: argparse.Namespace):
    from bashgym_autoresearch.client import Client

    token = args.token or os.environ.get("BGAR_TOKEN")
    if not token:
        raise SystemExit("set BGAR_TOKEN or pass --token")
    return Client(args.url or os.environ.get("BGAR_URL", DEFAULT_URL), token)


def _json_arg(value: str | None) -> object:
    if value is None:
        return None
    path = Path(value)
    if not value.lstrip().startswith(("{", "[")) and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _restrict(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass


def cmd_init(args: argparse.Namespace) -> int:
    """Create the state directory and the first human token; later tokens need a human."""
    from bashgym_autoresearch.api import open_home
    from bashgym_autoresearch.auth import create_token

    home = _home(args)
    home.mkdir(parents=True, exist_ok=True)
    _restrict(home, 0o700)
    service = open_home(home)
    if service.store.read_one("SELECT 1 FROM tokens LIMIT 1") is not None:
        print(f"already initialized: {home}")
        return 0
    token_path = home / HUMAN_TOKEN_FILENAME
    token = create_token(service.store, "human", "owner")
    token_path.write_text(token, encoding="utf-8")
    _restrict(token_path, 0o600)
    print(f"initialized {home}")
    print(f"human token written to {token_path}")
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    """Mint a token through the API; this requires a human token."""
    _print(_client(args).create_token(args.role, args.label))
    return 0


def check_home_permissions(home: Path) -> str | None:
    """On POSIX the state directory must not be readable by other users."""
    if os.name == "nt":
        return None
    mode = home.stat().st_mode & 0o777
    if mode & 0o077:
        return f"{home} has mode {oct(mode)}; run `chmod 700 {home}` (state holds tokens and keys)"
    return None


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from bashgym_autoresearch.api import create_app, open_home
    from bashgym_autoresearch.executors import LocalExecutor
    from bashgym_autoresearch.worker import Worker

    home = _home(args)
    service = open_home(home)
    problem = check_home_permissions(home)
    if problem:
        raise SystemExit(problem)
    stop = threading.Event()
    worker = threading.Thread(
        target=Worker(service, LocalExecutor(service.home / "control")).run,
        args=(stop, args.interval),
        daemon=True,
    )
    worker.start()
    try:
        uvicorn.run(create_app(service), host=args.host, port=args.port, log_level="info")
    finally:
        stop.set()
        worker.join(timeout=10)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from bashgym_autoresearch.mcp_server import main

    main()
    return 0


def cmd_profile_add(args: argparse.Namespace) -> int:
    script = Path(args.script).resolve()
    profile = {
        "name": args.name,
        "kind": args.kind,
        "argv": args.argv or [sys.executable, "{script}", "{run_dir}", "{inputs}"],
        "script": str(script),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "timeout_seconds": args.timeout,
    }
    _print(_client(args).register_profile(profile))
    return 0


def cmd_campaign_create(args: argparse.Namespace) -> int:
    _print(_client(args).create_campaign(_json_arg(args.spec), key=args.key))
    return 0


def cmd_campaign_list(args: argparse.Namespace) -> int:
    _print(_client(args).list_campaigns())
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    _print(_client(args).decide_approval(args.approval_id, args.command == "approve"))
    return 0


def cmd_guidance(args: argparse.Namespace) -> int:
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    _print(_client(args).set_guidance(args.campaign_id, text or ""))
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    body = {
        "role": args.role,
        "change": _json_arg(args.change),
        "recipe": _json_arg(args.recipe) or {},
        "hypothesis": args.hypothesis,
        "estimated_cost": args.cost,
    }
    _print(_client(args).propose(args.campaign_id, body, key=args.key))
    return 0


def cmd_request_approval(args: argparse.Namespace) -> int:
    _print(_client(args).request_approval(args.campaign_id, args.kind, _json_arg(args.payload)))
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    _print(_client(args).wait(args.campaign_id, args.after, args.timeout))
    return 0


def _simple(method: str):
    def run(args: argparse.Namespace) -> int:
        _print(getattr(_client(args), method)(args.campaign_id))
        return 0

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bashgym-ar", description=__doc__)
    parser.add_argument("--version", action="version", version=f"bashgym-ar {__version__}")
    parser.add_argument(
        "--home", help="state directory (default: $BGAR_HOME or ~/.bashgym-autoresearch)"
    )
    parser.add_argument("--url", help=f"API URL (default: $BGAR_URL or {DEFAULT_URL})")
    parser.add_argument("--token", help="bearer token (default: $BGAR_TOKEN)")
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("init", help="create the state directory and a human token").set_defaults(
        run=cmd_init
    )
    token = commands.add_parser(
        "token", help="create an agent or human token (needs a human token)"
    )
    token.add_argument("role", choices=["agent", "human"])
    token.add_argument("--label", default="agent")
    token.set_defaults(run=cmd_token)
    serve = commands.add_parser("serve", help="run the HTTP API and the worker")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--interval", type=float, default=1.0)
    serve.set_defaults(run=cmd_serve)
    commands.add_parser("mcp", help="run the MCP server over stdio").set_defaults(run=cmd_mcp)

    profile = commands.add_parser("profile", help="register a stage program (human)")
    profile.add_argument("name")
    profile.add_argument("--kind", choices=["train", "evaluate"], required=True)
    profile.add_argument("--script", required=True)
    profile.add_argument("--timeout", type=float, required=True)
    profile.add_argument("--argv", nargs="+", help="argv template; default runs the script")
    profile.set_defaults(run=cmd_profile_add)

    campaign = commands.add_parser("campaign", help="create or list campaigns (human)")
    campaign_commands = campaign.add_subparsers(dest="campaign_command", required=True)
    create = campaign_commands.add_parser("create")
    create.add_argument("spec", help="campaign spec JSON or path to a JSON file")
    create.add_argument("--key")
    create.set_defaults(run=cmd_campaign_create)
    campaign_commands.add_parser("list").set_defaults(run=cmd_campaign_list)

    for name in ("approve", "deny"):
        decide = commands.add_parser(name, help=f"{name} a pending approval (human)")
        decide.add_argument("approval_id")
        decide.set_defaults(run=cmd_decide)
    guidance = commands.add_parser("guidance", help="set campaign guidance (human)")
    guidance.add_argument("campaign_id")
    guidance.add_argument("--text")
    guidance.add_argument("--file")
    guidance.set_defaults(run=cmd_guidance)

    for name in ("brief", "results", "failures", "pause", "resume", "cancel", "report"):
        verb = commands.add_parser(name, help=f"agent operation: {name}")
        verb.add_argument("campaign_id")
        verb.set_defaults(run=_simple(name))
    wait = commands.add_parser("wait", help="agent operation: wait for events")
    wait.add_argument("campaign_id")
    wait.add_argument("--after", type=int, default=0)
    wait.add_argument("--timeout", type=float, default=60.0)
    wait.set_defaults(run=cmd_wait)
    propose = commands.add_parser("propose", help="agent operation: propose an experiment")
    propose.add_argument("campaign_id")
    propose.add_argument("--role", choices=["baseline", "candidate"], required=True)
    propose.add_argument("--hypothesis", required=True)
    propose.add_argument("--cost", type=float, required=True)
    propose.add_argument("--change", help='JSON: {"variable": ..., "before": ..., "after": ...}')
    propose.add_argument("--recipe", help="recipe JSON or path to a JSON file")
    propose.add_argument("--key", help="idempotency key (default: random)")
    propose.set_defaults(run=cmd_propose)
    approval = commands.add_parser("request-approval", help="agent operation: request approval")
    approval.add_argument("campaign_id")
    approval.add_argument(
        "--kind", choices=["start", "budget", "promote", "publish"], required=True
    )
    approval.add_argument("--payload", help="payload JSON")
    approval.set_defaults(run=cmd_request_approval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "run"):
        parser.print_help()
        return 0
    from bashgym_autoresearch.client import ClientError

    try:
        return args.run(args)
    except ClientError as exc:
        _print({"status": exc.status, "error": exc.detail})
        return 1
