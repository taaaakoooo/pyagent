"""Core state models for MCP server connections."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from internal.types.types import MCPServerConfig, ToolDefinition

RequestId = Union[int, str]
MCP_PROTOCOL_VERSION = "2026-07-28"
MCP_DEFAULT_TIMEOUT_SECONDS = 60.0


class MCPMode:
    """Supported MCP transport modes."""

    STDIO = "stdio"
    HTTP = "http"

    @classmethod
    def normalize(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized == cls.STDIO:
            return cls.STDIO
        if normalized in {cls.HTTP, "streamable-http", "streamable_http", "sse"}:
            return cls.HTTP
        raise ValueError(f"Unsupported MCP mode: {value}")


@dataclass(frozen=True)
class MCPStdio:
    """Configuration used to start a local MCP child process."""

    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    timeout_seconds: float = MCP_DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class MCPHttp:
    """Configuration used to connect to an MCP HTTP endpoint."""

    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = MCP_DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class MCPRequest:
    """A JSON-RPC request sent to an MCP server."""

    id: RequestId
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    jsonrpc: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params,
        }


@dataclass(frozen=True)
class MCPResponse:
    """A JSON-RPC response received from an MCP server."""

    id: RequestId
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResponse":
        if data.get("jsonrpc") != "2.0":
            raise ValueError("MCP response must use JSON-RPC 2.0")
        if "id" not in data:
            raise ValueError("MCP response is missing request id")
        if "result" in data and "error" in data:
            raise ValueError("MCP response cannot contain both result and error")
        return cls(
            id=data["id"],
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class MCPExchange:
    """A matched request and response pair."""

    request: MCPRequest
    response: MCPResponse


class MCPRequestMatcher:
    """Thread-safe correlation of JSON-RPC responses to pending requests."""

    def __init__(self) -> None:
        self._pending: Dict[RequestId, MCPRequest] = {}
        self._completed: Dict[RequestId, MCPExchange] = {}
        self._lock = threading.Lock()

    def add(self, request: MCPRequest) -> None:
        with self._lock:
            if request.id in self._pending:
                raise ValueError(f"Duplicate pending MCP request id: {request.id}")
            self._pending[request.id] = request

    def match(self, response: MCPResponse) -> MCPExchange:
        with self._lock:
            request = self._pending.pop(response.id, None)
            if request is None:
                raise KeyError(f"No pending MCP request for response id: {response.id}")
            exchange = MCPExchange(request=request, response=response)
            self._completed[response.id] = exchange
            return exchange

    def get_completed(self, request_id: RequestId) -> Optional[MCPExchange]:
        with self._lock:
            return self._completed.get(request_id)

    def is_pending(self, request_id: RequestId) -> bool:
        with self._lock:
            return request_id in self._pending

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._completed.clear()


@dataclass
class MCPCache:
    """Connection-local cache for MCP capabilities and discovered tools."""

    initialized: bool = False
    protocol_version: Optional[str] = None
    server_info: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    tools: List[ToolDefinition] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.initialized = False
        self.protocol_version = None
        self.server_info.clear()
        self.capabilities.clear()
        self.tools.clear()
        self.values.clear()


@dataclass
class MCP:
    """State for one MCP server connection.

    Transport execution is deliberately separate from this model. Exactly one
    of ``stdio`` and ``http`` must be configured according to ``mode``.
    """

    server_alias: str
    mode: str
    stdio: Optional[MCPStdio] = None
    http: Optional[MCPHttp] = None
    matcher: MCPRequestMatcher = field(default_factory=MCPRequestMatcher)
    cache: MCPCache = field(default_factory=MCPCache)
    user_config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.server_alias = self.server_alias.strip()
        if not self.server_alias:
            raise ValueError("MCP server alias must not be empty")

        self.mode = MCPMode.normalize(self.mode)
        if self.mode == MCPMode.STDIO:
            if self.stdio is None:
                raise ValueError("stdio mode requires stdio configuration")
            if self.http is not None:
                raise ValueError("stdio mode cannot include http configuration")
        elif self.mode == MCPMode.HTTP:
            if self.http is None:
                raise ValueError("http mode requires http configuration")
            if self.stdio is not None:
                raise ValueError("http mode cannot include stdio configuration")

    @classmethod
    def from_server_config(
        cls,
        config: MCPServerConfig,
        *,
        user_config: Optional[Dict[str, Any]] = None,
    ) -> "MCP":
        mode = MCPMode.normalize(config.transport)
        stdio: Optional[MCPStdio] = None
        http: Optional[MCPHttp] = None

        if mode == MCPMode.STDIO:
            if not config.command:
                raise ValueError(
                    f"MCP stdio server '{config.name}' requires a command"
                )
            stdio = MCPStdio(
                command=config.command,
                args=list(config.args),
                env=dict(config.env),
                cwd=config.cwd,
                timeout_seconds=config.timeout_seconds,
            )
        else:
            if not config.url:
                raise ValueError(f"MCP HTTP server '{config.name}' requires a URL")
            http = MCPHttp(
                url=config.url,
                headers=dict(config.headers),
                timeout_seconds=config.timeout_seconds,
            )

        return cls(
            server_alias=config.name,
            mode=mode,
            stdio=stdio,
            http=http,
            user_config=dict(user_config or {}),
            enabled=config.enabled,
        )


__all__ = [
    "MCP",
    "MCP_DEFAULT_TIMEOUT_SECONDS",
    "MCP_PROTOCOL_VERSION",
    "MCPCache",
    "MCPExchange",
    "MCPHttp",
    "MCPMode",
    "MCPRequest",
    "MCPRequestMatcher",
    "MCPResponse",
    "MCPStdio",
    "RequestId",
]
