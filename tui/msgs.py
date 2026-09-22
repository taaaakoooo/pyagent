"""Messages that drive the Elm-style TUI update loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from internal.types.types import PermissionRequest


@dataclass(frozen=True)
class KeyMsg:
    """A normalized key press."""

    key: str
    char: Optional[str] = None


@dataclass(frozen=True)
class TextMsg:
    """A text fragment produced without a key press (for tests and reuse)."""

    text: str


@dataclass(frozen=True)
class TickMsg:
    """Emitted by the runtime on every idle render tick."""

    tick: int = 0


@dataclass(frozen=True)
class ResizeMsg:
    """Terminal size changed."""

    width: int
    height: int


@dataclass(frozen=True)
class InterruptMsg:
    """The process was signalled (Ctrl+C, ``SIGTERM``).

    Routed through ``update`` rather than tearing the process down so an
    interrupt behaves exactly like Shift+C: cancel a running turn, otherwise
    quit through the same path as everything else.
    """


@dataclass(frozen=True)
class AgentStartedMsg:
    """A ``run_agent`` command began executing."""

    prompt: str


@dataclass(frozen=True)
class AgentDeltaMsg:
    """A streamed content fragment from the LLM."""

    delta: str


@dataclass(frozen=True)
class AgentToolMsg:
    """A tool lifecycle update emitted by ``on_tool_status``."""

    tool_name: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentPermissionMsg:
    """An approval request waiting on a decision."""

    request: PermissionRequest
    resolver: Optional[Callable[[str], None]] = None


@dataclass(frozen=True)
class AgentDoneMsg:
    """The agent turn finished."""

    answer: str = ""
    error: Optional[str] = None
    cancelled: bool = False


@dataclass(frozen=True)
class AgentErrorMsg:
    """The agent turn raised an unexpected exception."""

    message: str


@dataclass(frozen=True)
class SkillsChangedMsg:
    """Skill state was refreshed from the agent."""

    skills: tuple = ()
    enabled: bool = True
    unmatched_policy: str = "none"
    always_visible: tuple = ()
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class McpChangedMsg:
    """MCP state was refreshed from the agent."""

    servers: tuple = ()
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionChangedMsg:
    """Session state was refreshed from the agent."""

    session_id: str = "default"
    sessions: tuple = ()
    message_count: int = 0
    archive_dir: str = ""
    restored_count: int = 0


@dataclass(frozen=True)
class NoticeMsg:
    """A transient status line for the footer."""

    text: str


Msg = Union[
    KeyMsg,
    TextMsg,
    TickMsg,
    ResizeMsg,
    AgentStartedMsg,
    AgentDeltaMsg,
    AgentToolMsg,
    AgentPermissionMsg,
    AgentDoneMsg,
    AgentErrorMsg,
    SkillsChangedMsg,
    McpChangedMsg,
    SessionChangedMsg,
    NoticeMsg,
]


KEY_ENTER = "enter"
KEY_ESC = "esc"
KEY_TAB = "tab"
KEY_BACKTAB = "backtab"
KEY_BACKSPACE = "backspace"
KEY_DELETE = "delete"
KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_HOME = "home"
KEY_END = "end"
KEY_PAGEUP = "pageup"
KEY_PAGEDOWN = "pagedown"
KEY_SPACE = "space"

# Mouse wheel notches. The terminal reports these as SGR mouse events
# (``ESC [ < 64 ; x ; y M``), which the runtime decodes into these two keys so
# the wheel can scroll the transcript instead of being mistaken for typing.
KEY_SCROLL_UP = "scroll_up"
KEY_SCROLL_DOWN = "scroll_down"

# Shift+letter commands.
#
# A terminal does not report the Shift modifier for letters -- Shift+Q is
# delivered as the character "Q", exactly like a capital typed into the
# composer. These are therefore matched on the uppercase character, and the
# composer keeps a separate command layer (see ``Model.command_mode``) so a
# command can never swallow typed text.
KEY_SHIFT_Q = "Q"
KEY_SHIFT_C = "C"

SHIFT_COMMANDS = {
    KEY_SHIFT_Q: "quit",
    KEY_SHIFT_C: "cancel",
}

# Leave the composer's command layer and start typing again. Named distinctly
# from the ANSI "insert" key to avoid confusion.
KEY_RESUME_TYPING = "i"


__all__ = [
    "AgentDeltaMsg",
    "AgentDoneMsg",
    "AgentErrorMsg",
    "AgentPermissionMsg",
    "AgentStartedMsg",
    "AgentToolMsg",
    "KEY_BACKSPACE",
    "KEY_BACKTAB",
    "KEY_DELETE",
    "KEY_DOWN",
    "KEY_END",
    "KEY_ENTER",
    "KEY_ESC",
    "KEY_HOME",
    "KEY_LEFT",
    "KEY_PAGEDOWN",
    "KEY_PAGEUP",
    "KEY_RESUME_TYPING",
    "KEY_RIGHT",
    "KEY_SCROLL_DOWN",
    "KEY_SCROLL_UP",
    "KEY_SPACE",
    "KEY_TAB",
    "KEY_UP",
    "KEY_SHIFT_C",
    "KEY_SHIFT_Q",
    "SHIFT_COMMANDS",
    "InterruptMsg",
    "KeyMsg",
    "McpChangedMsg",
    "Msg",
    "NoticeMsg",
    "ResizeMsg",
    "SessionChangedMsg",
    "SkillsChangedMsg",
    "TextMsg",
    "TickMsg",
]
