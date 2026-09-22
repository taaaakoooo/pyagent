"""Pure update function: ``(Model, Msg) -> (Model, Cmd | None)``."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from internal.types.types import PermissionDecision
from tui.model import (
    ActiveView,
    BLINK_TICKS,
    LineRole,
    McpServerItem,
    Model,
    SessionItem,
    SkillItem,
    tool_status_label,
)
from tui.msgs import (
    KEY_BACKSPACE,
    KEY_BACKTAB,
    KEY_DELETE,
    KEY_DOWN,
    KEY_END,
    KEY_ENTER,
    KEY_ESC,
    KEY_HOME,
    KEY_RESUME_TYPING,
    KEY_LEFT,
    KEY_PAGEDOWN,
    KEY_PAGEUP,
    KEY_RIGHT,
    KEY_SCROLL_DOWN,
    KEY_SCROLL_UP,
    KEY_SPACE,
    KEY_TAB,
    KEY_UP,
    SHIFT_COMMANDS,
    AgentDeltaMsg,
    AgentDoneMsg,
    AgentErrorMsg,
    AgentPermissionMsg,
    AgentStartedMsg,
    AgentToolMsg,
    InterruptMsg,
    KeyMsg,
    McpChangedMsg,
    Msg,
    NoticeMsg,
    ResizeMsg,
    SessionChangedMsg,
    SkillsChangedMsg,
    TextMsg,
    TickMsg,
)

# A Cmd is an opaque side-effecting action executed by the runtime.
Cmd = Callable[["CmdContext"], None]

CANCEL_NOTICE = "正在取消本轮回复…"

# Typed instead of pressed: works while the composer is accepting text, where
# Shift+letter cannot be told apart from a capital letter.
EXIT_COMMANDS = frozenset({"/exit", "/quit", "/q"})

QUIT_CONFIRM_NOTICE = "agent 仍在回复：再按一次 Shift+Q 立即退出，其他键继续"

# Transcript rows moved per wheel notch. Smaller than a page so the wheel
# feels like fine-grained scrolling rather than page flipping.
WHEEL_SCROLL_LINES = 3

PERMISSION_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("a", PermissionDecision.ALLOW_ONCE),
    ("d", PermissionDecision.DENY_ONCE),
    ("A", PermissionDecision.ALLOW_SESSION),
    ("D", PermissionDecision.DENY_SESSION),
)

_PERMISSION_LABELS: Dict[str, str] = {
    PermissionDecision.ALLOW_ONCE: "本次允许",
    PermissionDecision.DENY_ONCE: "本次拒绝",
    PermissionDecision.ALLOW_SESSION: "本会话允许",
    PermissionDecision.DENY_SESSION: "本会话拒绝",
}


class CmdContext:
    """Hook the runtime injects so ``update`` can request side effects."""

    def __init__(
        self,
        run_agent: Optional[Callable[[str], None]] = None,
        resolve_permission: Optional[Callable[[str], None]] = None,
        cancel_agent: Optional[Callable[[], None]] = None,
        save_skills: Optional[Callable[[bool], None]] = None,
        reload_skills: Optional[Callable[[], None]] = None,
        reload_mcp: Optional[Callable[[], None]] = None,
        set_mcp_enabled: Optional[Callable[[str, bool], None]] = None,
        create_session: Optional[Callable[[str], None]] = None,
        switch_session: Optional[Callable[[str], None]] = None,
        delete_session: Optional[Callable[[str], None]] = None,
        refresh_view: Optional[Callable[[], None]] = None,
        quit: Optional[Callable[[], None]] = None,
    ) -> None:
        self.run_agent = run_agent
        self.resolve_permission = resolve_permission
        self.cancel_agent = cancel_agent
        self.save_skills = save_skills
        self.reload_skills = reload_skills
        self.reload_mcp = reload_mcp
        self.set_mcp_enabled = set_mcp_enabled
        self.create_session = create_session
        self.switch_session = switch_session
        self.delete_session = delete_session
        self.refresh_view = refresh_view
        self.quit = quit


def permission_options() -> List[Tuple[str, str]]:
    """Return ``(label, decision)`` pairs in display order."""
    return [
        (_PERMISSION_LABELS[decision], decision)
        for _, decision in PERMISSION_CHOICES
    ]


def update(model: Model, msg: Msg) -> Tuple[Model, Optional[Cmd]]:
    """Fold one message into the model, returning an optional command."""
    model = model.copy()

    if isinstance(msg, KeyMsg):
        return _update_key(model, msg)
    if isinstance(msg, TextMsg):
        return _update_text(model, msg)
    if isinstance(msg, ResizeMsg):
        model.width = max(20, msg.width)
        model.height = max(8, msg.height)
        return model, None
    if isinstance(msg, TickMsg):
        if BLINK_TICKS > 0:
            model.cursor_on = (msg.tick // BLINK_TICKS) % 2 == 0
        return model, None
    if isinstance(msg, NoticeMsg):
        model.notice = msg.text
        return model, None
    if isinstance(msg, InterruptMsg):
        # A signal is the process-level twin of Shift+C, so it takes the same
        # route: never tear the terminal down from inside a handler.
        return _cancel_or_quit(model)
    if isinstance(msg, AgentStartedMsg):
        model.is_running = True
        model.error = None
        model.notice = ""
        model.status = "running"
        return model, None
    if isinstance(msg, AgentDeltaMsg):
        model.stream_buffer += msg.delta
        model.scroll = 0
        return model, None
    if isinstance(msg, AgentToolMsg):
        return _update_tool(model, msg)
    if isinstance(msg, AgentPermissionMsg):
        model.permission = msg.request
        model.permission_choice = 0
        model.status = "waiting_permission"
        return model, None
    if isinstance(msg, AgentDoneMsg):
        return _update_done(model, msg)
    if isinstance(msg, AgentErrorMsg):
        model.flush_stream()
        model.append_line(LineRole.ERROR, msg.message)
        model.error = msg.message
        model.is_running = False
        model.status = "error"
        return model, None
    if isinstance(msg, SkillsChangedMsg):
        model.skills = [
            SkillItem(
                name=item.get("name", ""),
                description=item.get("description", ""),
                keywords=tuple(item.get("keywords") or ()),
                tools=tuple(item.get("tools") or ()),
            )
            for item in msg.skills
        ]
        model.skills_enabled = msg.enabled
        model.skill_unmatched_policy = msg.unmatched_policy
        model.always_visible = list(msg.always_visible)
        model.skill_errors = dict(msg.errors)
        model.clamp_selection()
        return model, None
    if isinstance(msg, McpChangedMsg):
        model.mcp_servers = [
            McpServerItem(
                name=item.get("name", ""),
                mode=item.get("mode", "stdio"),
                enabled=bool(item.get("enabled", True)),
                started=bool(item.get("started", False)),
                tools=tuple(item.get("tools") or ()),
                error=item.get("error"),
            )
            for item in msg.servers
        ]
        model.mcp_errors = dict(msg.errors)
        model.clamp_selection()
        return model, None
    if isinstance(msg, SessionChangedMsg):
        model.session_id = msg.session_id
        model.sessions = [
            SessionItem(
                session_id=item.get("session_id", ""),
                message_count=item.get("message_count", 0),
                updated_at=item.get("updated_at", ""),
                preview=item.get("preview", ""),
            )
            for item in msg.sessions
        ]
        model.archive_dir = msg.archive_dir
        model.clamp_selection()
        return model, None

    return model, None


def _update_key(model: Model, msg: KeyMsg) -> Tuple[Model, Optional[Cmd]]:
    key = msg.key

    # Any keystroke makes the caret visible again, so typing never feels
    # like it went nowhere during the "off" phase of the blink.
    model.cursor_on = True

    if model.permission is not None:
        return _update_permission_key(model, msg)

    # Shift+letter is checked before the per-view handlers, but never while
    # the composer is accepting text: there the same keystroke has to mean
    # "type this capital letter".
    command = SHIFT_COMMANDS.get(msg.char) if msg.char else None
    if command is not None and not _composer_is_typing(model):
        if command == "cancel":
            return _cancel_or_quit(model)
        return _request_quit(model)

    # Anything else means the user is still working, so a pending quit
    # confirmation lapses rather than lingering as a trap.
    model.quit_pending = False

    if key == KEY_TAB:
        model.view = model.view.next(1)
        model.command_mode = False
        return model, _hook("refresh_view")
    if key == KEY_BACKTAB:
        model.view = model.view.next(-1)
        model.command_mode = False
        return model, _hook("refresh_view")

    if model.view == ActiveView.CHAT:
        return _update_chat_key(model, msg)
    if model.view == ActiveView.SKILLS:
        return _update_skills_key(model, msg)
    if model.view == ActiveView.MCP:
        return _update_mcp_key(model, msg)
    if model.view == ActiveView.SESSIONS:
        return _update_sessions_key(model, msg)
    return model, None


def _composer_is_typing(model: Model) -> bool:
    """True when a printable character belongs in the prompt, not a command."""
    return model.view == ActiveView.CHAT and not model.command_mode


def _cancel_or_quit(model: Model) -> Tuple[Model, Optional[Cmd]]:
    """Shift+C and Ctrl+C: stop the current turn, or leave when idle."""
    if model.is_running:
        model.notice = CANCEL_NOTICE
        return model, _hook("cancel_agent")
    return _request_quit(model)


def _request_quit(model: Model) -> Tuple[Model, Optional[Cmd]]:
    """Leave the app, but never discard a running turn without asking.

    Quitting mid-turn silently throws away the answer the user is waiting for,
    and a stray Shift+Q is easy to hit. So the first request only arms the
    confirmation; any other keypress disarms it again.
    """
    if model.is_running and not model.quit_pending:
        model.quit_pending = True
        model.notice = QUIT_CONFIRM_NOTICE
        return model, None
    model.should_quit = True
    return model, None


def _update_permission_key(model: Model, msg: KeyMsg) -> Tuple[Model, Optional[Cmd]]:
    key = msg.key

    if key == KEY_ESC:
        return _resolve_permission(model, PermissionDecision.DENY_ONCE)
    if key in (KEY_UP, KEY_LEFT):
        model.permission_choice = max(0, model.permission_choice - 1)
        return model, None
    if key in (KEY_DOWN, KEY_RIGHT, KEY_TAB):
        model.permission_choice = min(
            len(PERMISSION_CHOICES) - 1, model.permission_choice + 1
        )
        return model, None
    if key == KEY_ENTER or key == KEY_SPACE:
        decision = PERMISSION_CHOICES[model.permission_choice][1]
        return _resolve_permission(model, decision)
    if msg.char:
        return _resolve_permission_from_char(model, msg.char)
    return model, None


def _resolve_permission_from_char(
    model: Model, char: str
) -> Tuple[Model, Optional[Cmd]]:
    for shortcut, decision in PERMISSION_CHOICES:
        if char == shortcut:
            return _resolve_permission(model, decision)
    lowered = char.lower()
    for shortcut, decision in PERMISSION_CHOICES:
        if lowered == shortcut.lower():
            return _resolve_permission(model, decision)
    return model, None


def _resolve_permission(
    model: Model, decision: str
) -> Tuple[Model, Optional[Cmd]]:
    request = model.permission
    model.permission = None
    model.permission_choice = 0
    if request is not None:
        label = _PERMISSION_LABELS.get(decision, decision)
        model.append_line(
            LineRole.SYSTEM,
            f"permission: {label} for {request.tool_name}",
        )
    model.status = "running"

    def _cmd(context: CmdContext) -> None:
        if context.resolve_permission is not None:
            context.resolve_permission(decision)

    return model, _cmd


def _update_chat_key(model: Model, msg: KeyMsg) -> Tuple[Model, Optional[Cmd]]:
    key = msg.key

    if model.command_mode:
        return _update_command_layer_key(model, msg)

    if key == KEY_ESC:
        if model.is_running:
            model.notice = CANCEL_NOTICE
            model.command_mode = True
            return model, _hook("cancel_agent")
        if model.input_buffer or model.stream_buffer:
            model.input_buffer = ""
            model.stream_buffer = ""
            return model, None
        # Nothing left to clear, so Esc means "leave the composer". This is
        # how you reach the Shift+letter commands from a fresh prompt.
        model.command_mode = True
        return model, None
    if key == KEY_ENTER:
        return _submit_input(model)
    if key == KEY_BACKSPACE:
        model.input_buffer = model.input_buffer[:-1]
        return model, None
    if key == KEY_DELETE:
        model.input_buffer = ""
        return model, None
    if key == KEY_HOME:
        model.input_buffer = ""
        return model, None
    if key == KEY_END:
        return model, None
    if key == KEY_UP:
        return _arrow_or_history(model, 1)
    if key == KEY_DOWN:
        return _arrow_or_history(model, -1)
    if key == KEY_PAGEUP:
        model.scroll_by(model.visible_lines)
        return model, None
    if key == KEY_PAGEDOWN:
        model.scroll_by(-model.visible_lines)
        return model, None
    if key == KEY_SCROLL_UP:
        model.scroll_by(WHEEL_SCROLL_LINES)
        return model, None
    if key == KEY_SCROLL_DOWN:
        model.scroll_by(-WHEEL_SCROLL_LINES)
        return model, None
    if msg.char:
        model.input_buffer += msg.char
        return model, None
    return model, None


def _update_command_layer_key(
    model: Model, msg: KeyMsg
) -> Tuple[Model, Optional[Cmd]]:
    """Handle keys while the composer is in its command layer.

    Shift+letter commands are resolved by the caller. Anything else that
    prints hands control back to the prompt, so a stray keypress is never
    swallowed -- except ``i``, which is the explicit "let me type" key and
    would otherwise have to be typed twice.
    """
    key = msg.key

    if key == KEY_ESC:
        model.command_mode = False
        return model, None
    if key == KEY_UP:
        return _arrow_or_history(model, 1)
    if key == KEY_DOWN:
        return _arrow_or_history(model, -1)
    if key == KEY_PAGEUP:
        model.scroll_by(model.visible_lines)
        return model, None
    if key == KEY_PAGEDOWN:
        model.scroll_by(-model.visible_lines)
        return model, None
    if key == KEY_SCROLL_UP:
        model.scroll_by(WHEEL_SCROLL_LINES)
        return model, None
    if key == KEY_SCROLL_DOWN:
        model.scroll_by(-WHEEL_SCROLL_LINES)
        return model, None
    if msg.char and msg.char.lower() == KEY_RESUME_TYPING:
        model.command_mode = False
        return model, None
    if msg.char:
        model.command_mode = False
        model.input_buffer += msg.char
        return model, None
    return model, None


def _arrow_or_history(model: Model, step: int) -> Tuple[Model, Optional[Cmd]]:
    """Split the arrow keys between the transcript and the input history.

    Terminals that do not report the mouse turn a wheel scroll into Up/Down,
    so the wheel and the history keys are literally the same bytes; there is
    nothing to tell them apart. What *is* observable is whether a prompt is in
    progress: mid-composition the arrows almost certainly mean "recall what I
    typed before", and on an empty prompt they almost certainly mean "scroll".
    """
    if model.input_buffer:
        return _history_step(model, step)
    model.scroll_by(WHEEL_SCROLL_LINES * step)
    return model, None


def _history_step(model: Model, step: int) -> Tuple[Model, Optional[Cmd]]:
    if not model.history:
        return model, None
    if model.history_index is None:
        index = len(model.history) - 1 if step > 0 else None
    else:
        index = model.history_index - step
    if index is None or index < 0:
        model.history_index = None
        model.input_buffer = ""
        return model, None
    index = min(index, len(model.history) - 1)
    model.history_index = index
    model.input_buffer = model.history[index]
    return model, None


def _wheel_as_arrow(key: str) -> str:
    """Let the wheel drive list selection in the panels.

    The chat view treats the wheel as transcript scrolling, but in a list the
    natural meaning is "move the highlight", which is already what Up/Down do.
    """
    if key == KEY_SCROLL_UP:
        return KEY_UP
    if key == KEY_SCROLL_DOWN:
        return KEY_DOWN
    return key


def _update_skills_key(model: Model, msg: KeyMsg) -> Tuple[Model, Optional[Cmd]]:
    key = _wheel_as_arrow(msg.key)
    if key == KEY_UP:
        model.skill_selected = max(0, model.skill_selected - 1)
        return model, None
    if key == KEY_DOWN:
        model.skill_selected = min(
            max(0, len(model.skills) - 1), model.skill_selected + 1
        )
        return model, None
    if key == KEY_SPACE or key == KEY_ENTER:
        model.skills_enabled = not model.skills_enabled

        def _cmd(context: CmdContext) -> None:
            if context.save_skills is not None:
                context.save_skills(model.skills_enabled)

        return model, _cmd
    if msg.char in ("r", "R"):
        return model, _hook("reload_skills")
    if msg.char in ("o", "O"):
        model.skills_enabled = not model.skills_enabled

        def _cmd(context: CmdContext) -> None:
            if context.save_skills is not None:
                context.save_skills(model.skills_enabled)

        return model, _cmd
    return model, None


def _update_mcp_key(model: Model, msg: KeyMsg) -> Tuple[Model, Optional[Cmd]]:
    key = _wheel_as_arrow(msg.key)
    if key == KEY_UP:
        model.mcp_selected = max(0, model.mcp_selected - 1)
        return model, None
    if key == KEY_DOWN:
        model.mcp_selected = min(
            max(0, len(model.mcp_servers) - 1), model.mcp_selected + 1
        )
        return model, None
    if msg.char in ("r", "R"):
        return model, _hook("reload_mcp")
    if key in (KEY_SPACE, KEY_ENTER) and model.mcp_servers:
        selected = model.mcp_servers[model.mcp_selected]
        target = not selected.enabled
        for index, server in enumerate(model.mcp_servers):
            if server.name == selected.name:
                model.mcp_servers[index] = type(selected)(
                    name=server.name,
                    mode=server.mode,
                    enabled=target,
                    started=server.started,
                    tools=server.tools,
                    error=server.error,
                )
        alias = selected.name

        def _cmd(context: CmdContext) -> None:
            if context.set_mcp_enabled is not None:
                context.set_mcp_enabled(alias, target)

        return model, _cmd
    return model, None


def _update_sessions_key(model: Model, msg: KeyMsg) -> Tuple[Model, Optional[Cmd]]:
    key = _wheel_as_arrow(msg.key)
    if key == KEY_UP:
        model.session_selected = max(0, model.session_selected - 1)
        return model, None
    if key == KEY_DOWN:
        model.session_selected = min(
            max(0, len(model.sessions) - 1), model.session_selected + 1
        )
        return model, None
    if key == KEY_ENTER and model.sessions:
        target = model.sessions[model.session_selected].session_id

        def _cmd(context: CmdContext) -> None:
            if context.switch_session is not None:
                context.switch_session(target)

        return model, _cmd
    if msg.char in ("n", "N"):
        model.notice = "正在创建新会话…"
        return model, _hook("create_session")
    if msg.char in ("d", "D") and model.sessions:
        target = model.sessions[model.session_selected].session_id

        def _cmd(context: CmdContext) -> None:
            if context.delete_session is not None:
                context.delete_session(target)

        return model, _cmd
    if msg.char in ("r", "R"):
        return model, _hook("refresh_view")
    return model, None


def _update_text(model: Model, msg: TextMsg) -> Tuple[Model, Optional[Cmd]]:
    """Type a whole string, honouring the active overlay and view."""
    for char in msg.text:
        if char == "\n":
            model, cmd = _update_key(model, KeyMsg(KEY_ENTER))
            if cmd is not None:
                return model, cmd
        else:
            model, cmd = _update_key(model, KeyMsg("char", char))
            if cmd is not None:
                return model, cmd
    return model, None


def _submit_input(model: Model) -> Tuple[Model, Optional[Cmd]]:
    text = model.input_buffer.strip()
    if not text:
        return model, None
    if text.lower() in EXIT_COMMANDS:
        model.input_buffer = ""
        return _request_quit(model)
    if model.is_running:
        model.notice = "agent 仍在回复，请稍候或按 Esc 取消本轮"
        return model, None

    model.input_buffer = ""
    model.history_index = None
    model.history.append(text)
    model.append_line(LineRole.USER, text)
    model.is_running = True
    model.error = None
    model.notice = ""
    model.status = "running"

    def _cmd(context: CmdContext) -> None:
        if context.run_agent is not None:
            context.run_agent(text)

    return model, _cmd


def _update_tool(model: Model, msg: AgentToolMsg) -> Tuple[Model, Optional[Cmd]]:
    # Flush first so the transcript stays chronological: any text streamed
    # before this tool call belongs above it.
    model.flush_stream()

    detail = ""
    if msg.status in ("completed", "error"):
        content = str(msg.details.get("content", "")).strip()
        detail = _first_line(content)
    elif msg.status == "running":
        detail = _summarize_arguments(msg.details)
    elif msg.status == "skipped":
        detail = str(msg.details.get("reason", "skipped"))
    elif msg.status == "warning":
        detail = str(msg.details.get("message", ""))

    suffix = f" {detail}" if detail else ""
    model.append_line(
        LineRole.TOOL,
        f"[{msg.tool_name}] {tool_status_label(msg.status)}{suffix}",
    )
    return model, None


def _update_done(model: Model, msg: AgentDoneMsg) -> Tuple[Model, Optional[Cmd]]:
    model.flush_stream()
    if msg.error:
        model.append_line(LineRole.ERROR, msg.error)
        model.error = msg.error
    model.is_running = False
    model.status = "cancelled" if msg.cancelled else ("error" if msg.error else "idle")
    return model, None


def _summarize_arguments(arguments: Dict[str, object], limit: int = 80) -> str:
    if not arguments:
        return ""
    parts = [f"{key}={_short(value)}" for key, value in arguments.items()]
    return _first_line(", ".join(parts))[:limit]


def _short(value: object) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= 40 else text[:37] + "..."


def _first_line(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= 100 else line[:97] + "..."


def _hook(name: str) -> Cmd:
    """Build a Cmd that calls the named hook on the runtime context."""

    def _run(context: CmdContext) -> None:
        hook = getattr(context, name, None)
        if callable(hook):
            hook()

    return _run


__all__ = [
    "CANCEL_NOTICE",
    "Cmd",
    "CmdContext",
    "PERMISSION_CHOICES",
    "permission_options",
    "update",
]
