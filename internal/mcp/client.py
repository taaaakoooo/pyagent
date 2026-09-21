"""One-client-per-server MCP connection wrapper."""

from __future__ import annotations

import itertools
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from internal.mcp.mcp import (
    MCP,
    MCPCache,
    MCPMode,
    MCPRequest,
    MCPResponse,
    MCP_PROTOCOL_VERSION,
)
from internal.mcp.transport import (
    MCPHttpTransport,
    MCPStdioTransport,
    MCPTransport,
)
from internal.types.types import MCPServerConfig, ToolResult

DEFAULT_CLIENT_INFO = {"name": "pyagent", "version": "0.1.0"}


class MCPRemoteError(RuntimeError):
    """JSON-RPC error returned by an MCP server."""

    def __init__(self, error: Dict[str, Any]) -> None:
        self.error = error
        code = error.get("code", "unknown")
        message = error.get("message", "MCP request failed")
        super().__init__(f"MCP error {code}: {message}")


@dataclass
class MCPClient:
    """A client bound to exactly one MCP server and one transport."""

    server_alias: str
    mode: str
    transport: MCPTransport
    cache: MCPCache = field(default_factory=MCPCache)
    user_config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    _started: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _request_ids: itertools.count = field(
        default_factory=lambda: itertools.count(1),
        init=False,
        repr=False,
    )
    _id_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.server_alias = self.server_alias.strip()
        if not self.server_alias:
            raise ValueError("MCP server alias must not be empty")
        self.mode = MCPMode.normalize(self.mode)

    @property
    def started(self) -> bool:
        return self._started and not self._closed

    @classmethod
    def from_server_config(
        cls,
        config: MCPServerConfig,
        *,
        user_config: Optional[Dict[str, Any]] = None,
    ) -> "MCPClient":
        state = MCP.from_server_config(config, user_config=user_config)
        if state.mode == MCPMode.STDIO:
            if state.stdio is None:
                raise ValueError("stdio MCP state is missing stdio configuration")
            transport: MCPTransport = MCPStdioTransport(
                state.server_alias,
                state.stdio,
            )
        else:
            if state.http is None:
                raise ValueError("HTTP MCP state is missing HTTP configuration")
            transport = MCPHttpTransport(state.server_alias, state.http)

        return cls(
            server_alias=state.server_alias,
            mode=state.mode,
            transport=transport,
            cache=state.cache,
            user_config=state.user_config,
            enabled=state.enabled,
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError(f"MCP client '{self.server_alias}' is closed")
        if not self.enabled or self._started:
            return
        self.transport.start()
        self._started = True

    def ensure_started(self) -> None:
        if not self.started:
            self.start()

    def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> MCPResponse:
        self.ensure_started()
        request_params = dict(params or {})
        meta = dict(request_params.get("_meta") or {})
        meta["io.modelcontextprotocol/protocolVersion"] = MCP_PROTOCOL_VERSION
        meta.setdefault(
            "io.modelcontextprotocol/clientInfo",
            dict(self.user_config.get("client_info") or DEFAULT_CLIENT_INFO),
        )
        meta.setdefault(
            "io.modelcontextprotocol/clientCapabilities",
            dict(self.user_config.get("client_capabilities") or {}),
        )
        request_params["_meta"] = meta
        request = MCPRequest(
            id=self.next_request_id(),
            method=method,
            params=request_params,
        )
        return self.transport.send(request)

    def discover_server(self) -> Dict[str, Any]:
        response = self.send_request("server/discover")
        result = self._require_result(response)
        supported = result.get("supportedVersions")
        if not isinstance(supported, list) or MCP_PROTOCOL_VERSION not in supported:
            raise MCPRemoteError(
                {
                    "code": "unsupported_protocol_version",
                    "message": (
                        f"Server '{self.server_alias}' does not support "
                        f"{MCP_PROTOCOL_VERSION}"
                    ),
                    "data": {"supported": supported or []},
                }
            )

        capabilities = result.get("capabilities") or {}
        if not isinstance(capabilities, dict):
            raise ValueError("MCP discover capabilities must be an object")
        result_meta = result.get("_meta") or {}
        if not isinstance(result_meta, dict):
            result_meta = {}
        server_info = result_meta.get("io.modelcontextprotocol/serverInfo") or {}
        if not isinstance(server_info, dict):
            server_info = {}

        self.cache.initialized = True
        self.cache.protocol_version = MCP_PROTOCOL_VERSION
        self.cache.capabilities = dict(capabilities)
        self.cache.server_info = dict(server_info)
        self.cache.values["discover"] = dict(result)
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        seen_cursors = set()

        while True:
            params: Dict[str, Any] = {}
            if cursor is not None:
                params["cursor"] = cursor
            response = self.send_request("tools/list", params)
            result = self._require_result(response)
            page = result.get("tools") or []
            if not isinstance(page, list):
                raise ValueError("MCP tools/list result.tools must be an array")
            for raw_tool in page:
                if not isinstance(raw_tool, dict):
                    raise ValueError("Each MCP tool definition must be an object")
                tools.append(dict(raw_tool))

            next_cursor = result.get("nextCursor")
            if not next_cursor:
                break
            cursor = str(next_cursor)
            if cursor in seen_cursors:
                raise ValueError(f"MCP tools/list repeated cursor: {cursor}")
            seen_cursors.add(cursor)

        self.cache.values["raw_tools"] = tools
        return tools

    def start_and_discover(self) -> List[Dict[str, Any]]:
        self.ensure_started()
        self.discover_server()
        return self.list_tools()

    def call_tool(
        self,
        remote_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        response = self.send_request(
            "tools/call",
            {"name": remote_name, "arguments": dict(arguments)},
        )
        if response.error is not None:
            return ToolResult(
                tool_call_id="",
                name=remote_name,
                content=str(MCPRemoteError(response.error)),
                is_error=True,
                metadata={"mcp_error": dict(response.error)},
            )

        result = response.result
        if not isinstance(result, dict):
            return ToolResult(
                tool_call_id="",
                name=remote_name,
                content="MCP tools/call returned a non-object result",
                is_error=True,
            )

        content = self._normalize_tool_content(result)
        metadata: Dict[str, Any] = {"mcp_result": dict(result)}
        if "structuredContent" in result:
            metadata["structured_content"] = result["structuredContent"]
        return ToolResult(
            tool_call_id="",
            name=remote_name,
            content=content,
            is_error=bool(result.get("isError", False)),
            metadata=metadata,
        )

    def next_request_id(self) -> int:
        with self._id_lock:
            return next(self._request_ids)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._started = False
        self.transport.close()

    @staticmethod
    def _require_result(response: MCPResponse) -> Dict[str, Any]:
        if response.error is not None:
            raise MCPRemoteError(response.error)
        if not isinstance(response.result, dict):
            raise ValueError("MCP response result must be an object")
        return response.result

    @staticmethod
    def _normalize_tool_content(result: Dict[str, Any]) -> str:
        parts: List[str] = []
        content_items = result.get("content") or []
        if isinstance(content_items, list):
            for item in content_items:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
        elif content_items:
            parts.append(str(content_items))

        if not parts and "structuredContent" in result:
            parts.append(
                json.dumps(result["structuredContent"], ensure_ascii=False, indent=2)
            )
        return "\n".join(part for part in parts if part)


__all__ = ["DEFAULT_CLIENT_INFO", "MCPClient", "MCPRemoteError"]
