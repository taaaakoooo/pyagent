"""Shared data structures used by the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# types.py中定义公共的数据结构，这些数据在多个模块中共享：
# llm/client.py     → 产出 Message、LLMResponse、ToolCall
# agent/agent.py    → 读写 Message，调度 ToolCall
# tools/tools.py    → 接收 ToolCall，返回 ToolResult
# session/session.py → 持久化 Message
# tui/tui.py        → 展示 Message、AgentState
# mcp/mcp.py        → 把 MCP 工具转成 ToolDefinition

# 项目里同时存在三种“被调用方”：

# LLM API（远程 AI）
# 本地 tools（进程内函数）
# MCP Server（外部工具进程）
# 它们对 Agent 来说都叫“工具”，但实现方式不同。Agent 在 tools/tools.py 
# 里把它们统一注册成 ToolDefinition，对 LLM 来说看起来一样，但底层调用路径不同。

class AgentStatus:
    """Known values for :attr:`AgentState.status`."""

    IDLE = "idle"
    THINKING = "thinking"
    STREAMING = "streaming"
    WAITING_PERMISSION = "waiting_permission"
    EXECUTING_TOOL = "executing_tool"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ToolDefinition:
    """Description and JSON Schema for a tool exposed to the LLM."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    source: str = "local"
    server_name: Optional[str] = None


@dataclass
class FunctionCall:
    """A function name and its JSON-encoded arguments.

    During streaming, ``name`` and ``arguments`` may arrive in multiple
    fragments. The LLM client is responsible for joining those fragments.
    """

    name: str = ""
    arguments: str = ""


@dataclass
class ToolCall:
    """A tool call returned by an LLM, including streaming information."""

    index: Optional[int] = None
    id: str = ""
    type: str = "function"
    function: FunctionCall = field(default_factory=FunctionCall)


@dataclass
class Message:
    """A single conversation message."""

    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The result returned after a tool call."""

    tool_call_id: str
    name: str
    content: Any
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Normalized response returned by an LLM provider."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMStreamChunk:
    """One incremental chunk from a streaming LLM response."""

    content_delta: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current runtime status of the agent loop."""

    status: str = AgentStatus.IDLE
    turn: int = 0
    max_turns: int = 20
    is_streaming: bool = False
    active_tool_name: Optional[str] = None
    active_tool_call_id: Optional[str] = None
    pending_permission: Optional[PermissionRequest] = None
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PermissionDecision:
    """User decision for a permission prompt."""

    ALLOW_ONCE = "allow_once"
    DENY_ONCE = "deny_once"
    ALLOW_SESSION = "allow_session"
    DENY_SESSION = "deny_session"


@dataclass
class PermissionRequest:
    """A request for approval before performing a potentially unsafe action."""

    request_id: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    risk_level: str = "medium"
    risk_tags: List[str] = field(default_factory=list)
    risk_description: str = ""
    preview_diff: Optional[Dict[str, Any]] = None
    tool_source: str = "local"
    server_name: Optional[str] = None


@dataclass
class MCPServerConfig:
    """Connection settings for an MCP server."""

    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    enabled: bool = True


__all__ = [
    "AgentState",
    "AgentStatus",
    "FunctionCall",
    "LLMResponse",
    "LLMStreamChunk",
    "MCPServerConfig",
    "Message",
    "PermissionDecision",
    "PermissionRequest",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
]
