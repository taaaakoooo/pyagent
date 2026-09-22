"""Pure view: ``Model -> rich Renderable``."""

from __future__ import annotations

from functools import lru_cache
from typing import List

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tui.model import ActiveView, LineRole, Model, agent_status_label
from tui.update import permission_options

ROLE_STYLES = {
    LineRole.USER: "bold cyan",
    LineRole.ASSISTANT: "white",
    LineRole.TOOL: "dim yellow",
    LineRole.SYSTEM: "dim",
    LineRole.ERROR: "bold red",
}

ROLE_PREFIX = {
    LineRole.USER: "你",
    LineRole.ASSISTANT: "助手",
    LineRole.TOOL: "工具",
    LineRole.SYSTEM: "提示",
    LineRole.ERROR: "错误",
}

CHAT_HINTS = "Enter 发送  Esc 命令  ↑/↓ 或滚轮 滚动  Tab 切换面板"
COMMAND_HINTS = "Shift+Q 退出  Shift+C 取消  i 输入  Esc 返回  Tab 面板"
PANEL_HINTS = "↑/↓ 移动  空格 开关  Enter 应用  Shift+R 刷新  Shift+Q 退出"

# Transcription lines are capped at TRANSCRIPT_LIMIT (400); the extra headroom
# covers streaming rows and stale entries left behind by a resize.
_FORMAT_CACHE_SIZE = 2048

# Block caret drawn at the end of the input line while it has focus. The "off"
# phase is a space so the box geometry never shifts while blinking.
#
# Deliberately restricted to glyphs that survive a legacy console code page:
# U+2588 encodes in both cp936 (GBK) and cp437, unlike fancier dingbats such as
# U+270E which raise UnicodeEncodeError -- or render as "?" after the
# errors="replace" reconfiguration done in tui/__main__.py.
CURSOR_ON = "\u2588"
CURSOR_OFF = " "
ELLIPSIS = "\u2026"


def view(model: Model) -> RenderableType:
    """Render the whole screen for ``model``."""
    layout = Layout()
    layout.split_column(
        Layout(_header(model), name="header", size=1),
        Layout(_body(model), name="body"),
        Layout(_composer(model), name="composer", size=3),
    )
    if model.permission is not None:
        return Group(layout, _permission_panel(model))
    return layout


def _header(model: Model) -> Text:
    header = Text()
    header.append(" pyagent ", style="bold white on blue")
    header.append(f" 会话:{model.session_id} ", style="cyan")
    header.append(f" 状态:{agent_status_label(model.status)} ", style="magenta")
    header.append(f" 轮次:{model.turn}/{model.max_turns} ", style="green")
    header.append(
        f" tokens:{model.token_total}/{model.token_limit} ",
        style="yellow",
    )
    if model.skills_enabled:
        header.append(" 技能:开 ", style="green")
    else:
        header.append(" 技能:关 ", style="dim")
    return header


def _composer(model: Model) -> Panel:
    """The persistent input box -- the main signal that typing lands here.

    Four states, distinguished by border colour and title so it is always
    obvious whether keystrokes go to the chat input or to a command:

    * command -> magenta, Shift+letter commands active (chat view)
    * running -> yellow, locked (the agent owns the turn)
    * focused -> cyan, live caret
    * blurred -> grey, hint to Tab back into the chat
    """
    running = model.is_running
    focused = model.view == ActiveView.CHAT and not running

    if running:
        border = "yellow"
        title = " 输入已锁定（agent 回复中） "
        body_style = "dim"
        hint = "Esc 取消本轮"
    elif model.view == ActiveView.CHAT and model.command_mode:
        border = "magenta"
        title = " 命令层 "
        body_style = "dim"
        hint = model.notice or COMMAND_HINTS
    elif focused:
        border = "cyan"
        title = " > 输入 "
        body_style = "bold white"
        hint = model.notice or CHAT_HINTS
    else:
        border = "grey37"
        title = " 输入 · 按 Tab 回到对话 "
        body_style = "dim"
        hint = PANEL_HINTS

    body = Text()
    typing = focused and not model.command_mode
    body.append("> " if typing else "  ", style=body_style)
    if typing:
        # Budget: panel borders + padding take 4 cells, "> " takes 2, and the
        # caret takes 1. Overshooting makes Rich wrap, and the extra row is
        # then cropped away by the fixed 3-row layout, hiding the input.
        # Keep the tail visible so the caret never scrolls out of view.
        body.append(_clip_tail(model.input_buffer, model.width - 7), style=body_style)
        body.append(CURSOR_ON if model.cursor_on else CURSOR_OFF, style="bold cyan")
    else:
        body.append(_clip_tail(model.input_buffer, model.width - 6), style=body_style)

    return Panel(
        body,
        title=title,
        subtitle=hint,
        border_style=border,
        expand=True,
    )


def _clip_tail(text: str, width: int) -> str:
    """Keep the end of ``text`` so a long input line stays readable."""
    available = max(1, width)
    if cell_len(text) <= available:
        return text

    kept: List[str] = []
    used = 0
    for char in reversed(text):
        size = cell_len(char)
        if used + size > available - 1:
            break
        kept.append(char)
        used += size
    return ELLIPSIS + "".join(reversed(kept))


def _body(model: Model) -> RenderableType:
    if model.view == ActiveView.SKILLS:
        return _skills_view(model)
    if model.view == ActiveView.MCP:
        return _mcp_view(model)
    if model.view == ActiveView.SESSIONS:
        return _sessions_view(model)
    if model.view == ActiveView.HELP:
        return _help_view()
    return _chat_view(model)


def _chat_view(model: Model) -> RenderableType:
    lines: List[Text] = []
    for line in model.transcript:
        lines.append(_format_line(line.role, line.text, model.width))

    if model.stream_buffer:
        lines.append(
            _format_line(LineRole.ASSISTANT, model.stream_buffer, model.width)
        )

    if not lines:
        lines.append(Text("输入内容开始对话。", style="dim"))

    window = max(1, model.visible_lines)
    total = len(lines)
    end = total - model.scroll
    start = max(0, end - window)
    visible = lines[start:max(start + 1, end)]

    body = Group(*visible)
    return Panel(body, title=ActiveView.CHAT.title, border_style="blue")


@lru_cache(maxsize=_FORMAT_CACHE_SIZE)
def _format_line(role: LineRole, text: str, width: int) -> Text:
    """Render one transcript entry.

    ``lru_cache`` matters a lot here: the chat view formats the whole
    transcript, but only a couple of dozen rows are visible, so without
    memoization every frame re-wraps hundreds of lines. Measured on a
    400-entry transcript: 1.17 ms uncached vs 0.03 ms cached.

    The returned ``Text`` is shared between frames, so callers must treat it
    as immutable -- never ``append`` to what this hands back. The regression
    for that lives in tests/test_tui_view.py::FormatLineCacheTests.
    """
    style = ROLE_STYLES.get(role, "white")
    prefix = f"{ROLE_PREFIX.get(role, role.value)} | "
    pad = " " * cell_len(prefix)
    wrap_width = max(10, width - cell_len(prefix) - 6)

    rendered = Text()
    for index, raw in enumerate(text.splitlines() or [""]):
        if index:
            rendered.append("\n")
        for chunk_index, chunk in enumerate(_wrap(raw, wrap_width) or [""]):
            if chunk_index:
                rendered.append("\n")
                rendered.append(pad, style=style)
            else:
                rendered.append(prefix, style=style)
            rendered.append(chunk, style=style)
    return rendered


def _wrap(text: str, width: int) -> List[str]:
    if width <= 0 or cell_len(text) <= width:
        return [text]
    chunks: List[str] = []
    current = ""
    for char in text:
        if cell_len(current + char) > width:
            chunks.append(current)
            current = char
        else:
            current += char
    if current or not chunks:
        chunks.append(current)
    return chunks


def _skills_view(model: Model) -> RenderableType:
    table = Table(
        title=ActiveView.SKILLS.title,
        expand=True,
        border_style="blue",
        header_style="bold",
    )
    table.add_column("", width=2, no_wrap=True)
    table.add_column("名称", no_wrap=True)
    table.add_column("关键词", overflow="fold")
    table.add_column("工具", overflow="fold")

    if not model.skills:
        table.add_row("", "[dim]尚未加载技能[/dim]", "", "")

    for index, skill in enumerate(model.skills):
        marker = ">" if index == model.skill_selected else " "
        table.add_row(
            marker,
            skill.name,
            ", ".join(skill.keywords) or "[dim]-[/dim]",
            ", ".join(skill.tools) or "[dim]-[/dim]",
        )

    info = Text()
    info.append(
        f"路由: {'开' if model.skills_enabled else '关'}"
        f"   未命中策略: {model.skill_unmatched_policy}\n",
        style="cyan",
    )
    info.append(
        f"常驻来源: {', '.join(model.always_visible) or '-'}",
        style="dim",
    )
    for source, message in model.skill_errors.items():
        info.append(f"\n[加载失败] {source}: {message}", style="red")

    return Group(table, info)


def _mcp_view(model: Model) -> RenderableType:
    table = Table(
        title=ActiveView.MCP.title,
        expand=True,
        border_style="blue",
        header_style="bold",
    )
    table.add_column("", width=2, no_wrap=True)
    table.add_column("服务", no_wrap=True)
    table.add_column("模式", no_wrap=True)
    table.add_column("状态", no_wrap=True)
    table.add_column("工具", overflow="fold")
    table.add_column("错误", overflow="fold")

    if not model.mcp_servers:
        table.add_row(
            "",
            "[dim]未配置 MCP 服务[/dim]",
            "",
            "",
            "",
            "",
        )

    for index, server in enumerate(model.mcp_servers):
        marker = ">" if index == model.mcp_selected else " "
        state = "已启用" if server.enabled else "已停用"
        if server.enabled and server.started:
            state = "已连接"
        table.add_row(
            marker,
            server.name,
            server.mode,
            state,
            str(len(server.tools)),
            server.error or "",
        )

    detail = Text()
    if model.mcp_servers:
        selected = model.mcp_servers[
            min(model.mcp_selected, len(model.mcp_servers) - 1)
        ]
        detail.append(f"{selected.name} 的工具:\n", style="cyan")
        if selected.tools:
            for name in selected.tools:
                detail.append(f"  - {name}\n", style="dim")
        else:
            detail.append("  （未发现工具）\n", style="dim")
    for alias, message in model.mcp_errors.items():
        detail.append(f"[启动失败] {alias}: {message}\n", style="red")

    return Group(table, detail)


def _sessions_view(model: Model) -> RenderableType:
    table = Table(
        title=ActiveView.SESSIONS.title,
        expand=True,
        border_style="blue",
        header_style="bold",
    )
    table.add_column("", width=2, no_wrap=True)
    table.add_column("会话", no_wrap=True)
    table.add_column("消息", no_wrap=True)
    table.add_column("更新时间", no_wrap=True)
    table.add_column("摘要", overflow="fold")

    if not model.sessions:
        table.add_row("", "[dim]暂无会话[/dim]", "", "", "")

    for index, session in enumerate(model.sessions):
        current = session.session_id == model.session_id
        marker = "*" if current else (" " if index != model.session_selected else ">")
        table.add_row(
            marker,
            session.session_id,
            str(session.message_count),
            session.updated_at,
            session.preview or "[dim]（空）[/dim]",
        )

    info = Text()
    info.append(f"归档目录: {model.archive_dir}", style="dim")
    return Group(table, info)


def _help_view() -> RenderableType:
    # Kept compact on purpose: the composer eats three rows, so a verbose
    # cheat sheet would get cropped on a 24-row terminal. Every line here has
    # to fit that budget -- adding one pushes the permission list out of view.
    body = Text()
    body.append("对话\n", style="bold cyan")
    body.append("  Enter            发送输入的内容\n")
    body.append("  Esc              清空输入 / 取消本轮 / 进入命令层\n")
    body.append("  ↑ / ↓ 或滚轮      滚动聊天记录（正在输入时翻历史）\n")
    body.append("  PgUp / PgDn      上下翻页（Shift+拖拽可选中文字）\n")
    body.append("命令层（输入框为空时按 Esc 进入，按 i 返回）\n", style="bold cyan")
    body.append("  Shift+Q          退出（回复中需再按一次确认）\n")
    body.append("  Shift+C          取消本轮回复；空闲时退出\n")
    body.append("面板\n", style="bold cyan")
    body.append("  Tab / Shift+Tab  切换 对话 / 技能 / MCP / 会话 / 帮助\n")
    body.append("  空格 / Enter      开关技能、开关 MCP、应用会话\n")
    body.append("  滚轮              移动高亮项\n")
    body.append("  Shift+R          刷新当前面板\n")
    body.append("  Shift+N / Shift+D 新建 / 删除会话\n")
    body.append("退出  在输入框直接键入 /exit 后回车，或 Esc 后按 Shift+Q\n")
    body.append("权限（a / d = 本次，A / D = 本会话，Esc 拒绝）\n", style="bold cyan")
    options = permission_options()
    for index, (label, _decision) in enumerate(options):
        body.append(f"  {index + 1}.{label}")
        body.append("\n" if index % 2 else "    ")
    return Panel(body, title=ActiveView.HELP.title, border_style="blue")


def _permission_panel(model: Model) -> RenderableType:
    request = model.permission
    body = Text()
    if request is None:
        return Panel(body, title="权限审批", border_style="yellow")

    body.append(f"工具: {request.tool_name}\n", style="bold yellow")
    if request.server_name:
        body.append(f"服务: {request.server_name}\n", style="magenta")
    body.append(
        f"风险: {request.risk_level}  标签: {', '.join(request.risk_tags) or '-'}\n",
        style="red",
    )
    if request.risk_description:
        body.append(f"{request.risk_description}\n")

    arguments = request.arguments or {}
    if arguments:
        body.append("\n参数\n", style="bold")
        for key, value in list(arguments.items())[:8]:
            body.append(f"  {key} = {_clip(str(value), 120)}\n", style="dim")

    diff = request.preview_diff
    if isinstance(diff, dict):
        unified = diff.get("unified")
        if unified:
            body.append("\n改动预览\n", style="bold")
            body.append(_clip(str(unified), 2000) + "\n", style="dim")

    body.append("\n", style="")
    for index, (label, _decision) in enumerate(permission_options()):
        marker = ">" if index == model.permission_choice else " "
        body.append(f" {marker} {label}\n", style="bold green" if index == model.permission_choice else "white")

    return Panel(
        body,
        title="需要审批",
        border_style="yellow",
        expand=False,
    )


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_to_text(model: Model, width: int = 100, height: int = 30) -> str:
    """Render the view into a plain string (used by tests and snapshots)."""
    import io

    from rich.console import Console

    console = Console(
        file=io.StringIO(),
        width=width,
        height=height,
        force_terminal=False,
        legacy_windows=False,
        color_system=None,
    )
    console.print(view(model))
    return console.file.getvalue()


__all__ = ["render_to_text", "view"]
