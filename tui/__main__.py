"""Run the TUI: ``python -m tui``."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List, Optional

from config.config import ConfigError, load_config
from internal.agent.agent import create_agent
from internal.session.session import SessionError
from tui.agent_bridge import AgentBridge, build_cmd_context
from tui.model import initial_model
from tui.runtime import Program
from tui.view import view


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyagent-tui",
        description="Elm-architecture terminal UI for pyagent.",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--session",
        default=None,
        help="session id to open (defaults to config session.default_id)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="render ticks per second while idle",
    )
    parser.add_argument(
        "--no-mouse",
        action="store_true",
        help=(
            "do not enable terminal mouse reporting; the wheel will then be "
            "sent as arrow keys, and plain drag selects text. Also set "
            "PYAGENT_INPUT_DEBUG=<file> to log raw input bytes."
        ),
    )
    return parser


def _print_exit_summary(agent: object, session_id: str, turns: int) -> None:
    """Leave a breadcrumb once the alt screen closes.

    The transcript goes away with the screen, so the last line on the terminal
    should say which session this was and where the history landed.
    """
    try:
        messages = len(getattr(agent, "messages", ()) or ())
    except Exception:  # noqa: BLE001 - a summary must never mask the exit
        messages = 0

    storage = getattr(agent, "session_storage_dir", None)
    location = f"{session_id}.jsonl" if not storage else str(
        os.path.join(str(storage), f"{session_id}.jsonl")
    )
    print(
        f"pyagent: 已退出 · 会话 {session_id} · {messages} 条消息 · "
        f"本轮 {turns} 次 · 历史 {location}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # An un-encodable character in model output must never kill the UI.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        agent = create_agent(config, session_id=args.session)
    except SessionError as exc:
        print(f"Invalid session: {exc}", file=sys.stderr)
        return 2

    size = shutil.get_terminal_size((100, 30))
    model = initial_model(agent, width=size.columns, height=size.lines)
    program = Program(model, view_fn=view, fps=args.fps, mouse=not args.no_mouse)

    def _quit() -> None:
        program.model.should_quit = True

    bridge = AgentBridge(agent, program.dispatch)
    program.commands = build_cmd_context(bridge, _quit)
    bridge.refresh_all()

    try:
        program.run()
    except KeyboardInterrupt:
        # Only reached when a second Ctrl+C forces the shutdown, or when the
        # loop itself is interrupted before ``run`` can restore anything.
        pass
    finally:
        agent.close()

    _print_exit_summary(agent, program.model.session_id, program.model.turn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
