"""Elm-style application model for the TUI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from internal.types.types import PermissionRequest

TRANSCRIPT_LIMIT = 400

# Ticks between caret blinks. The runtime ticks at DEFAULT_FPS (12), so this
# is roughly a 0.5 s on/off cycle.
BLINK_TICKS = 6


class ActiveView(str, Enum):
    """Top-level panels selectable with Tab."""

    CHAT = "chat"
    SKILLS = "skills"
    MCP = "mcp"
    SESSIONS = "sessions"
    HELP = "help"

    @property
    def title(self) -> str:
        return {
            ActiveView.CHAT: "对话",
            ActiveView.SKILLS: "技能",
            ActiveView.MCP: "MCP",
            ActiveView.SESSIONS: "会话",
            ActiveView.HELP: "帮助",
        }[self]

    def next(self, step: int = 1) -> "ActiveView":
        order = list(ActiveView)
        return order[(order.index(self) + step) % len(order)]


class LineRole(str, Enum):
    """Visual role of a transcript line."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"
    ERROR = "error"


# Display labels for state tokens that cross the update/view boundary.
#
# The model keeps the raw tokens (``"running"``, ``"idle"``) because they are
# also compared programmatically; only the rendered text is localised. Keeping
# the maps here rather than in the view means the transcript lines that
# ``update`` bakes in -- a tool lifecycle row -- get the same wording as the
# header, instead of one being translated and the other not.

TOOL_STATUS_LABELS: Dict[str, str] = {
    "running": "执行中",
    "completed": "完成",
    "error": "失败",
    "skipped": "已跳过",
    "warning": "警告",
}

AGENT_STATUS_LABELS: Dict[str, str] = {
    "idle": "空闲",
    "running": "回复中",
    "waiting_permission": "等待审批",
    "cancelled": "已取消",
    "error": "错误",
}


def tool_status_label(status: str) -> str:
    """Localised tool status, falling back to the raw token if unknown."""
    return TOOL_STATUS_LABELS.get(status, status)


def agent_status_label(status: str) -> str:
    """Localised header status, falling back to the raw token if unknown."""
    return AGENT_STATUS_LABELS.get(status, status)


@dataclass(frozen=True)
class ChatLine:
    """One rendered transcript entry."""

    role: LineRole
    text: str


@dataclass(frozen=True)
class SkillItem:
    """A skill row in the skills panel."""

    name: str
    description: str = ""
    keywords: Sequence[str] = ()
    tools: Sequence[str] = ()


@dataclass(frozen=True)
class McpServerItem:
    """A server row in the MCP panel."""

    name: str
    mode: str = "stdio"
    enabled: bool = True
    started: bool = False
    tools: Sequence[str] = ()
    error: Optional[str] = None


@dataclass(frozen=True)
class SessionItem:
    """A session row in the sessions panel."""

    session_id: str
    message_count: int = 0
    updated_at: str = ""
    preview: str = ""


@dataclass
class Model:
    """Everything the view needs; mutated only inside ``update``."""

    view: ActiveView = ActiveView.CHAT
    width: int = 100
    height: int = 30
    should_quit: bool = False

    # Set when quitting would discard a turn that is still running. The first
    # quit request arms this and asks again; any other key disarms it, so an
    # accidental Shift+Q mid-turn cannot lose the answer.
    quit_pending: bool = False

    # Chat
    transcript: List[ChatLine] = field(default_factory=list)
    stream_buffer: str = ""
    input_buffer: str = ""
    history: List[str] = field(default_factory=list)
    history_index: Optional[int] = None
    scroll: int = 0
    is_running: bool = False
    cursor_on: bool = True

    # The composer has two layers. ``False`` (the default) types into the
    # prompt; ``True`` interprets Shift+letter as a command instead. They have
    # to be separate because a terminal reports Shift+Q as the character "Q",
    # so a command layer is the only way a command can never eat typed text.
    command_mode: bool = False

    # Status bar
    status: str = "idle"
    turn: int = 0
    max_turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_limit: int = 0
    notice: str = ""
    error: Optional[str] = None

    # Permission overlay
    permission: Optional[PermissionRequest] = None
    permission_choice: int = 0

    # Skills panel
    skills: List[SkillItem] = field(default_factory=list)
    skills_enabled: bool = True
    skill_unmatched_policy: str = "none"
    always_visible: List[str] = field(default_factory=lambda: ["mcp"])
    skill_errors: Dict[str, str] = field(default_factory=dict)
    skill_selected: int = 0

    # MCP panel
    mcp_servers: List[McpServerItem] = field(default_factory=list)
    mcp_errors: Dict[str, str] = field(default_factory=dict)
    mcp_selected: int = 0

    # Sessions panel
    session_id: str = "default"
    sessions: List[SessionItem] = field(default_factory=list)
    session_selected: int = 0
    archive_dir: str = ""

    @property
    def visible_lines(self) -> int:
        """Roughly how many transcript rows fit in the chat view.

        Chrome is the 1-row header, the 3-row composer, the chat panel's two
        border rows, and a little slack.
        """
        return max(1, self.height - 10)

    @property
    def token_total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def append_line(self, role: LineRole, text: str) -> None:
        self.transcript.append(ChatLine(role=role, text=text))
        if len(self.transcript) > TRANSCRIPT_LIMIT:
            del self.transcript[: len(self.transcript) - TRANSCRIPT_LIMIT]
        self.scroll = 0

    def flush_stream(self) -> None:
        """Move the streaming buffer into the transcript as an assistant line."""
        if self.stream_buffer.strip():
            self.append_line(LineRole.ASSISTANT, self.stream_buffer.rstrip())
        self.stream_buffer = ""

    def clamp_selection(self) -> None:
        self.skill_selected = _clamp(self.skill_selected, len(self.skills))
        self.mcp_selected = _clamp(self.mcp_selected, len(self.mcp_servers))
        self.session_selected = _clamp(self.session_selected, len(self.sessions))

    def scroll_by(self, delta: int) -> None:
        max_scroll = max(0, len(self.transcript) - 1)
        self.scroll = min(max(0, self.scroll + delta), max_scroll)

    def copy(self) -> "Model":
        """Shallow copy with independent list containers for safe updates."""
        clone = replace(self)
        clone.transcript = list(self.transcript)
        clone.history = list(self.history)
        clone.skills = list(self.skills)
        clone.mcp_servers = list(self.mcp_servers)
        clone.sessions = list(self.sessions)
        clone.skill_errors = dict(self.skill_errors)
        clone.mcp_errors = dict(self.mcp_errors)
        return clone


def _clamp(value: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(value, length - 1))


def initial_model(
    agent: Optional[Any] = None,
    *,
    width: int = 100,
    height: int = 30,
) -> Model:
    """Seed the model from a live agent, or empty defaults when absent."""
    model = Model(width=width, height=height)
    if agent is None:
        return model

    model.session_id = getattr(agent, "session_id", "default")
    model.max_turns = getattr(agent.state, "max_turns", 0)
    model.token_limit = getattr(agent, "context_window_limit", 0)
    model.skill_unmatched_policy = getattr(
        agent, "skill_unmatched_policy", "none"
    )
    model.skills_enabled = agent.skills_enabled()
    model.archive_dir = str(getattr(agent, "tool_output_path", ""))

    if getattr(agent, "token_budget", None) is not None:
        model.prompt_tokens = agent.token_budget.prompt_tokens
        model.completion_tokens = agent.token_budget.completion_tokens

    for message in getattr(agent, "messages", []) or []:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role == "system" or not content:
            continue
        if role == "user":
            model.append_line(LineRole.USER, content)
        elif role == "assistant":
            model.append_line(LineRole.ASSISTANT, content)
        elif role == "tool":
            model.append_line(LineRole.TOOL, content)

    apply_skills(model, agent)
    apply_mcp(model, agent)
    apply_sessions(model, agent)
    model.scroll = 0
    return model


def apply_skills(model: Model, agent: Any) -> None:
    """Refresh the skills panel from the agent."""
    model.skills = [
        SkillItem(
            name=item.get("name", ""),
            description=item.get("description", ""),
            keywords=tuple(item.get("keywords") or ()),
            tools=tuple(item.get("tools") or ()),
        )
        for item in agent.describe_skills()
    ]
    model.skills_enabled = agent.skills_enabled()
    model.skill_unmatched_policy = agent.skill_unmatched_policy
    model.always_visible = list(getattr(agent, "always_visible_sources", ()) or ())
    model.skill_errors = dict(agent.state.metadata.get("skill_errors") or {})
    model.clamp_selection()


def apply_mcp(model: Model, agent: Any) -> None:
    """Refresh the MCP panel from the agent."""
    model.mcp_servers = [
        McpServerItem(
            name=item.get("name", ""),
            mode=item.get("mode", "stdio"),
            enabled=bool(item.get("enabled", True)),
            started=bool(item.get("started", False)),
            tools=tuple(item.get("tools") or ()),
            error=item.get("error"),
        )
        for item in agent.describe_mcp()
    ]
    model.mcp_errors = dict(agent.state.metadata.get("mcp_errors") or {})
    model.clamp_selection()


def apply_sessions(model: Model, agent: Any) -> None:
    """Refresh the sessions panel from the agent."""
    model.session_id = agent.session_id
    model.archive_dir = str(agent.tool_output_path)
    model.sessions = [
        SessionItem(
            session_id=summary.session_id,
            message_count=summary.message_count,
            updated_at=_format_timestamp(summary.updated_at),
            preview=summary.preview,
        )
        for summary in agent.list_sessions()
    ]
    model.clamp_selection()


def _format_timestamp(value: Any) -> str:
    try:
        return value.astimezone().strftime("%Y-%m-%d %H:%M")
    except (AttributeError, ValueError):
        return str(value)


__all__ = [
    "AGENT_STATUS_LABELS",
    "ActiveView",
    "BLINK_TICKS",
    "ChatLine",
    "LineRole",
    "McpServerItem",
    "Model",
    "SessionItem",
    "SkillItem",
    "TOOL_STATUS_LABELS",
    "TRANSCRIPT_LIMIT",
    "agent_status_label",
    "apply_mcp",
    "apply_sessions",
    "apply_skills",
    "initial_model",
    "tool_status_label",
]
