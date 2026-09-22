"""Manager for MCP clients and stable public tool names."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from internal.mcp.client import MCPClient
from internal.mcp.mcp import MCPResponse
from internal.mcp.tooling import normalize_mcp_tool
from internal.tools.tools import ToolRegistry
from internal.types.types import ToolDefinition, ToolResult


@dataclass(frozen=True)
class MCPToolRoute:
    """Route from an LLM-visible tool name to its MCP server tool."""

    public_name: str
    server_alias: str
    remote_name: str


class MCPManager:
    """Own MCP clients and maintain collision-free public tool names."""

    def __init__(self) -> None:
        self._clients: Dict[str, MCPClient] = {}
        self._public_routes: Dict[str, MCPToolRoute] = {}
        self._remote_to_public: Dict[Tuple[str, str], str] = {}
        self._next_suffix: Dict[str, int] = {}
        self._server_public_names: Dict[str, Set[str]] = {}
        self._registry: Optional[ToolRegistry] = None
        self._closed = False
        self._lock = threading.RLock()

    def add_client(self, client: MCPClient) -> None:
        alias = client.server_alias
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP manager is closed")
            if alias in self._clients:
                raise ValueError(f"MCP server alias already registered: {alias}")
            self._clients[alias] = client

    def get_client(self, server_alias: str) -> MCPClient:
        with self._lock:
            client = self._clients.get(server_alias)
        if client is None:
            raise KeyError(f"Unknown MCP server alias: {server_alias}")
        return client

    def list_clients(self) -> List[MCPClient]:
        with self._lock:
            return list(self._clients.values())

    def remove_client(
        self,
        server_alias: str,
        *,
        close: bool = True,
    ) -> Optional[MCPClient]:
        with self._lock:
            client = self._clients.pop(server_alias, None)
            if client is None:
                return None
            route_keys = [
                public_name
                for public_name, route in self._public_routes.items()
                if route.server_alias == server_alias
            ]
            for public_name in route_keys:
                route = self._public_routes.pop(public_name)
                self._remote_to_public.pop(
                    (route.server_alias, route.remote_name),
                    None,
                )
                if self._registry is not None:
                    self._registry.unregister(public_name)
            self._server_public_names.pop(server_alias, None)
        if close:
            client.close()
        return client

    def set_client_enabled(
        self,
        server_alias: str,
        enabled: bool,
        registry: Optional[ToolRegistry] = None,
    ) -> MCPClient:
        """Toggle a server; disabling also unregisters its public tools."""
        client = self.get_client(server_alias)
        client.enabled = bool(enabled)
        if not enabled and registry is not None:
            self._remove_stale_tools(server_alias, set(), registry)
        return client

    def start_all(self) -> Dict[str, BaseException]:
        """Start all enabled clients and return failures keyed by alias."""
        failures: Dict[str, BaseException] = {}
        for client in self.list_clients():
            if not client.enabled:
                continue
            try:
                client.start()
            except BaseException as exc:
                failures[client.server_alias] = exc
        return failures

    def start(self, registry: ToolRegistry) -> Dict[str, BaseException]:
        """Start, discover, normalize, and register tools from all clients."""
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP manager is closed")
            self._registry = registry

        failures: Dict[str, BaseException] = {}
        for client in self.list_clients():
            if not client.enabled:
                continue
            try:
                self.discover_client(client.server_alias, registry)
            except BaseException as exc:
                failures[client.server_alias] = exc
        return failures

    def discover_client(
        self,
        server_alias: str,
        registry: ToolRegistry,
    ) -> List[ToolDefinition]:
        """Discover one server and register its normalized tools."""
        client = self.get_client(server_alias)
        raw_tools = client.start_and_discover()
        occupied = {tool.name for tool in registry.list_tools()}
        discovered_remote_names: Set[str] = set()
        definitions: List[ToolDefinition] = []
        new_remote_keys: List[Tuple[str, str]] = []
        registered_names: List[str] = []

        try:
            for raw_tool in raw_tools:
                remote_name = raw_tool.get("name")
                if not isinstance(remote_name, str) or not remote_name.strip():
                    raise ValueError(
                        f"MCP server '{server_alias}' returned a tool without a name"
                    )
                remote_name = remote_name.strip()
                discovered_remote_names.add(remote_name)
                remote_key = (server_alias, remote_name)
                if self.get_public_tool_name(*remote_key) is None:
                    new_remote_keys.append(remote_key)
                public_name = self.reserve_tool_name(
                    server_alias,
                    remote_name,
                    occupied_names=occupied,
                )
                definition = normalize_mcp_tool(
                    server_alias,
                    raw_tool,
                    public_name,
                )
                definitions.append(definition)
                occupied.add(public_name)
                if not registry.has_tool(public_name):
                    registry.register(
                        definition,
                        self._make_tool_handler(public_name),
                    )
                    registered_names.append(public_name)
                else:
                    registry.update_definition(definition)
        except BaseException:
            for public_name in registered_names:
                registry.unregister(public_name)
            with self._lock:
                for remote_key in new_remote_keys:
                    public_name = self._remote_to_public.pop(remote_key, None)
                    if public_name is not None:
                        self._public_routes.pop(public_name, None)
            raise

        self._remove_stale_tools(
            server_alias,
            discovered_remote_names,
            registry,
        )
        client.cache.tools = definitions
        with self._lock:
            self._server_public_names[server_alias] = {
                definition.name for definition in definitions
            }
            self._registry = registry
        return definitions

    def send_request(
        self,
        server_alias: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> MCPResponse:
        """Dispatch a JSON-RPC request through the server's selected transport."""
        with self._lock:
            if self._closed:
                raise RuntimeError("MCP manager is closed")
        client = self.get_client(server_alias)
        return client.send_request(method, params)

    def call_tool(
        self,
        public_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Route an LLM-visible tool call to the owning MCP server."""
        with self._lock:
            if self._closed:
                return ToolResult(
                    tool_call_id="",
                    name=public_name,
                    content="MCP manager is closed",
                    is_error=True,
                )
        route = self.resolve_tool_name(public_name)
        client = self.get_client(route.server_alias)
        result = client.call_tool(route.remote_name, arguments)
        result.name = public_name
        result.metadata.setdefault("mcp_server", route.server_alias)
        result.metadata.setdefault("mcp_remote_name", route.remote_name)
        return result

    def close_all(self) -> Dict[str, BaseException]:
        """Close all clients and return failures keyed by alias."""
        failures: Dict[str, BaseException] = {}
        for client in self.list_clients():
            try:
                client.close()
            except BaseException as exc:
                failures[client.server_alias] = exc
        return failures

    def close(self) -> Dict[str, BaseException]:
        """Stop accepting requests and close every client. Idempotent."""
        with self._lock:
            if self._closed:
                return {}
            self._closed = True
        return self.close_all()

    def reserve_tool_name(
        self,
        server_alias: str,
        remote_name: str,
        *,
        occupied_names: Optional[Iterable[str]] = None,
    ) -> str:
        """Return a stable, collision-free public name for a remote tool."""
        alias = server_alias.strip()
        name = remote_name.strip()
        if not alias:
            raise ValueError("MCP server alias must not be empty")
        if not name:
            raise ValueError("MCP remote tool name must not be empty")

        with self._lock:
            if alias not in self._clients:
                raise KeyError(f"Unknown MCP server alias: {alias}")

            remote_key = (alias, name)
            existing = self._remote_to_public.get(remote_key)
            if existing is not None:
                return existing

            occupied = set(occupied_names or ())
            suffix = self._next_suffix.get(name, 0)
            while True:
                public_name = name if suffix == 0 else f"{name}_{suffix}"
                suffix += 1
                if (
                    public_name not in self._public_routes
                    and public_name not in occupied
                ):
                    break

            self._next_suffix[name] = suffix
            route = MCPToolRoute(
                public_name=public_name,
                server_alias=alias,
                remote_name=name,
            )
            self._public_routes[public_name] = route
            self._remote_to_public[remote_key] = public_name
            return public_name

    def resolve_tool_name(self, public_name: str) -> MCPToolRoute:
        with self._lock:
            route = self._public_routes.get(public_name)
        if route is None:
            raise KeyError(f"Unknown MCP public tool name: {public_name}")
        return route

    def get_public_tool_name(
        self,
        server_alias: str,
        remote_name: str,
    ) -> Optional[str]:
        with self._lock:
            return self._remote_to_public.get((server_alias, remote_name))

    def list_tool_routes(self) -> List[MCPToolRoute]:
        with self._lock:
            return list(self._public_routes.values())

    def _make_tool_handler(self, public_name: str):
        def handler(arguments: Dict[str, Any]) -> ToolResult:
            return self.call_tool(public_name, arguments)

        return handler

    def _remove_stale_tools(
        self,
        server_alias: str,
        current_remote_names: Set[str],
        registry: ToolRegistry,
    ) -> None:
        with self._lock:
            stale_keys = [
                remote_key
                for remote_key in self._remote_to_public
                if remote_key[0] == server_alias
                and remote_key[1] not in current_remote_names
            ]
            for remote_key in stale_keys:
                public_name = self._remote_to_public.pop(remote_key)
                self._public_routes.pop(public_name, None)
                registry.unregister(public_name)


__all__ = ["MCPManager", "MCPToolRoute"]
