"""Entry point: TUI by default, line-based CLI with ``--cli``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from config.config import ConfigError, load_config
from internal.agent import create_agent
from internal.session import SessionError


def _configure_stdio() -> None:
    """Never let an un-encodable character kill the process.

    Windows consoles often default to a legacy code page (GBK here), so model
    output containing emoji would otherwise raise UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # detached or non-text stream
            pass


# --------------------------------------------------------------------------
# Line-based CLI
# --------------------------------------------------------------------------


def _print_skill_list(agent) -> None:
    skills = agent.describe_skills()
    if not skills:
        print("尚未加载技能。")
        return
    print(f"已加载 {len(skills)} 个技能：")
    for skill in skills:
        keywords = ", ".join(skill["keywords"]) or "（无）"
        tools = ", ".join(skill["tools"]) or "（无）"
        print(f"  - {skill['name']}: 关键词=[{keywords}] 工具=[{tools}]")


def _print_skill_errors(agent) -> None:
    errors = agent.state.metadata.get("skill_errors") or {}
    for source, message in errors.items():
        print(f"[技能] 加载失败 {source}: {message}")


def _print_mcp_errors(agent) -> None:
    errors = agent.state.metadata.get("mcp_errors") or {}
    for alias, message in errors.items():
        print(f"[MCP] 服务 '{alias}' 启动失败: {message}")


def _handle_skills_command(agent, argument: str) -> None:
    action = argument.strip().lower()
    if action in {"", "list"}:
        _print_skill_list(agent)
        _print_skill_errors(agent)
    elif action == "reload":
        result = agent.reload_skills()
        print(f"已重新加载 {len(result.skills)} 个技能。")
        for source, message in result.errors.items():
            print(f"[技能] 加载失败 {source}: {message}")
    elif action == "on":
        agent.set_skills_enabled(True)
        print("技能已启用。")
    elif action == "off":
        agent.set_skills_enabled(False)
        print("技能已关闭，所有工具都可见。")
    elif action == "status":
        active = agent.state.metadata.get("active_skills") or []
        visible = agent.state.metadata.get("visible_tools") or []
        print(f"技能开关: {'开' if agent.skills_enabled() else '关'}")
        print(f"未命中策略: {agent.skill_unmatched_policy}")
        print(f"常驻来源: {', '.join(agent.always_visible_sources)}")
        print(f"本轮命中的技能: {active or '（无）'}")
        print(f"本轮可见的工具: {visible or '（无）'}")
    else:
        print("用法: /skills [list|reload|on|off|status]")


def _handle_mcp_command(agent, argument: str) -> None:
    parts = argument.split()
    action = parts[0].lower() if parts else "list"
    name = parts[1] if len(parts) > 1 else ""

    if action == "list":
        servers = agent.describe_mcp()
        if not servers:
            print("未配置 MCP 服务。")
            return
        print(f"{len(servers)} 个 MCP 服务：")
        for server in servers:
            state = "已连接" if server["started"] else (
                "已启用" if server["enabled"] else "已停用"
            )
            print(
                f"  - {server['name']} [{server['mode']}] {state} "
                f"工具={len(server['tools'])}"
            )
        _print_mcp_errors(agent)
    elif action == "reload":
        failures = agent.reload_mcp()
        if failures:
            print(f"重新加载完成，{len(failures)} 个错误：")
            for alias, error in failures.items():
                print(f"  - {alias}: {error}")
        else:
            print("MCP 服务已重新加载。")
    elif action in {"enable", "disable"}:
        if not name:
            print(f"用法: /mcp {action} <name>")
            return
        enabled = action == "enable"
        if agent.set_mcp_server_enabled(name, enabled):
            print(f"MCP 服务 '{name}' {'已启用' if enabled else '已停用'}。")
        else:
            print(f"未知的 MCP 服务: {name}")
    elif action == "tools":
        servers = agent.describe_mcp()
        selected = [s for s in servers if s["name"] == name] if name else servers
        if not selected:
            print(f"未知的 MCP 服务: {name}" if name else "未配置 MCP 服务。")
            return
        for server in selected:
            tools = ", ".join(server["tools"]) or "（无）"
            print(f"{server['name']}: {tools}")
    else:
        print(
            "用法: /mcp [list|reload|enable <name>|disable <name>|tools [name]]"
        )


def _format_timestamp(value) -> str:
    try:
        return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (AttributeError, ValueError):
        return str(value)


def _handle_session_command(agent, argument: str) -> None:
    parts = argument.split()
    action = parts[0].lower() if parts else "list"
    name = parts[1] if len(parts) > 1 else ""

    if action == "list":
        sessions = agent.list_sessions()
        if not sessions:
            print("暂无会话。")
            return
        print(f"{len(sessions)} 个会话：")
        for summary in sessions:
            marker = "*" if summary.session_id == agent.session_id else " "
            preview = summary.preview or "（空）"
            print(
                f" {marker} {summary.session_id}  "
                f"消息={summary.message_count}  "
                f"更新={_format_timestamp(summary.updated_at)}  "
                f"{preview}"
            )
    elif action == "new":
        created = agent.create_session(name or None)
        print(f"已创建并切换到会话 '{created}'。")
    elif action == "use":
        if not name:
            print("用法: /session use <name>")
            return
        restored = agent.switch_session(name)
        if restored:
            print(
                f"已切换到会话 '{agent.session_id}'："
                f"恢复了 {agent.restored_message_count} 条消息。"
            )
        else:
            print(f"已切换到会话 '{agent.session_id}'：暂无历史。")
    elif action == "current":
        print(f"会话: {agent.session_id}")
        print(f"消息数: {len(agent.messages)}")
        print(f"归档目录: {agent.tool_output_path}")
    elif action == "delete":
        if not name:
            print("用法: /session delete <name>")
            return
        try:
            deleted = agent.delete_session(name)
        except SessionError as exc:
            print(f"无法删除 '{name}': {exc}")
            return
        if deleted:
            print(f"已删除会话 '{name}'。")
        else:
            print(f"未找到会话 '{name}'。")
    else:
        print("用法: /session [list|new [name]|use <name>|current|delete <name>]")


def _print_help() -> None:
    # Left column is pure ASCII, so ``ljust`` lines the descriptions up; the
    # Chinese text on the right is double-width and needs no padding.
    rows = (
        ("/skills [list|reload|on|off|status]", "查看或控制技能"),
        ("/mcp [list|reload|enable|disable|tools]", "查看或控制 MCP 服务"),
        ("/session [list|new|use|current|delete]", "管理会话"),
        ("/help", "显示本帮助"),
        ("exit | quit", "退出"),
    )
    print("可用命令：")
    for command, description in rows:
        print(f"  {command.ljust(42)} {description}")


def run_cli(argv: Optional[List[str]] = None) -> int:
    """Line-based interactive loop (the pre-TUI interface)."""
    parser = argparse.ArgumentParser(
        prog="pyagent --cli",
        description="pyagent 的行式命令行界面。",
    )
    parser.add_argument("--config", default=None, help="config.yaml 路径")
    parser.add_argument("--session", default=None, help="要打开的会话 id")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        agent = create_agent(config, session_id=args.session)
    except (ConfigError, SessionError) as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        return 2

    if agent.restored_message_count > 0:
        print(
            f"会话 '{agent.session_id}'："
            f"恢复了 {agent.restored_message_count} 条消息。"
        )
    else:
        print(f"会话 '{agent.session_id}'：暂无历史。")
    _print_skill_errors(agent)
    _print_mcp_errors(agent)
    print("pyagent 已就绪。输入 exit 退出，输入 /help 查看命令。")

    try:
        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl+C leaves the caret on the "^C" line; EOF (piped stdin)
                # gets the same courtesy so the farewell never lands mid-line.
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            if user_input.startswith("/"):
                command, _, argument = user_input[1:].partition(" ")
                if command == "skills":
                    _handle_skills_command(agent, argument)
                elif command == "mcp":
                    _handle_mcp_command(agent, argument)
                elif command == "session":
                    _handle_session_command(agent, argument)
                elif command == "help":
                    _print_help()
                else:
                    print(f"未知命令: /{command}。输入 /help 查看可用命令。")
                continue

            answer = agent.chat(user_input)
            if answer and not agent.state.is_streaming:
                print()
    finally:
        agent.close()

    # Printed once, after the loop, so every way out says goodbye -- including
    # the silent ``exit`` break, which used to just stop the process.
    print("再见。")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyagent",
        description="编码 agent。默认启动 TUI，加 --cli 使用行式界面。",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="使用行式 CLI 而不是 TUI",
    )
    parser.add_argument("--config", default=None, help="config.yaml 路径")
    parser.add_argument("--session", default=None, help="要打开的会话 id")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)

    forwarded: List[str] = []
    if args.config:
        forwarded += ["--config", args.config]
    if args.session:
        forwarded += ["--session", args.session]

    if args.cli:
        return run_cli(forwarded)

    from tui.__main__ import main as tui_main

    return tui_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
