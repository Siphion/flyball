"""flyballd command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .config import MODES, Settings


def _settings_from(args: argparse.Namespace) -> Settings:
    settings = Settings()
    for name in ("upstream", "host", "port", "mode"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(settings, name, value)
    if getattr(args, "data_dir", None):
        settings.data_dir = Path(args.data_dir)
    if getattr(args, "governor", False):
        settings.governor = True
    settings.__post_init__()
    return settings


def cmd_stats(args: argparse.Namespace) -> int:
    from .stats import report

    return report(_settings_from(args).data_dir)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .proxy import create_app

    settings = _settings_from(args)
    print(f"flyball {__version__}  mode={settings.mode}")
    print(f"  listening on http://{settings.host}:{settings.port}")
    print(f"  upstream     {settings.upstream}")
    if settings.mode == "off":
        print("  pure passthrough: no decision is taken. Use --mode shadow to start logging.")
    if settings.governor:
        if settings.mode == "active":
            print("  governor     on: confidence {:.2f}, cap {} chars, {:.0%} of conversations".format(
                settings.governor_confidence, settings.governor_cap, settings.ab_treatment_share))
        else:
            print("  governor     requested, but it only acts in --mode active")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from .upstream import probe_sync

    settings = _settings_from(args)
    caps = probe_sync(settings.upstream, settings.upstream_api_key)

    if args.json:
        print(json.dumps(caps.as_dict(), indent=2))
        return 0 if caps.reachable else 1

    def mark(ok: bool) -> str:
        return "yes" if ok else "no "

    print(f"upstream             {settings.upstream}")
    print(f"  reachable          {mark(caps.reachable)}")
    print(f"  server             {caps.server or 'unknown'}")
    print(f"  model              {caps.model or 'unknown'}")
    print(f"  thinking toggle    {mark(caps.thinking_toggle)}   (chat_template_kwargs.enable_thinking)")
    print(f"  reasoning control  {mark(caps.reasoning_control)}   (POST /v1/chat/completions/control)")
    print(f"  probe action       {mark(caps.probe_action)}   (ask the model mid-reasoning)")
    for note in caps.notes:
        print(f"  note: {note}")
    if caps.auth_required:
        print("\nSet FLYBALL_UPSTREAM_API_KEY and probe again.")
    elif caps.reasoning_control and caps.probe_action:
        print("\nThis upstream can be governed: flyball can ask the model mid-reasoning and close it.")
    elif caps.reasoning_control:
        print("\nThis upstream can be governed: flyball can close the reasoning mid-stream.")
    elif caps.reachable:
        print("\nThis upstream can only be configured, not governed: no mid-stream control.")
    return 0 if caps.reachable else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="flyballd",
        description="A System-1 governor for your local model's reasoning.",
    )
    parser.add_argument("--version", action="version", version=f"flyball {__version__}")
    # Shared options live on a parent parser so that they can be written after the
    # subcommand, which is where everyone types them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--upstream", help="OpenAI-compatible server that runs the model")
    common.add_argument("--data-dir", help="where the decision ledger is written")
    common.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the proxy", parents=[common])
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--mode", choices=MODES)
    serve.add_argument(
        "--governor",
        action="store_true",
        help="in active mode, close the reasoning once the model has settled (needs the probe action)",
    )
    serve.set_defaults(func=cmd_serve)

    probe = sub.add_parser("probe", help="report what the upstream supports", parents=[common])
    probe.add_argument("--json", action="store_true", help="machine-readable output")
    probe.set_defaults(func=cmd_probe)

    stats = sub.add_parser("stats", help="summarise the decision ledger", parents=[common])
    stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
