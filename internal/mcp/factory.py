"""Build an MCP manager from declarative configuration."""

from __future__ import annotations

from typing import List, Optional, Tuple

from config.config import MCPConfig, MCPServerEntry
from internal.mcp.client import MCPClient
from internal.mcp.manager import MCPManager
from internal.types.types import MCPServerConfig


def to_server_config(entry: MCPServerEntry) -> MCPServerConfig:
    """Convert a config-layer entry into an MCP connection config."""
    return MCPServerConfig(
        name=entry.name.strip(),
        transport=entry.transport.strip(),
        command=entry.command,
        args=list(entry.args),
        env=dict(entry.env),
        cwd=entry.cwd,
        url=entry.url,
        headers=dict(entry.headers),
        timeout_seconds=entry.timeout_seconds,
        enabled=entry.enabled,
    )


def build_mcp_manager(config: MCPConfig) -> MCPManager:
    """Build a manager holding one client per enabled server entry."""
    manager = MCPManager()
    for entry in config.enabled_servers():
        manager.add_client(
            MCPClient.from_server_config(to_server_config(entry))
        )
    return manager


def build_mcp_manager_if_configured(
    config: Optional[MCPConfig],
) -> Tuple[Optional[MCPManager], List[str]]:
    """Return a manager plus the aliases it manages, or ``None`` when empty."""
    if config is None or not config.enabled:
        return None, []
    servers = config.enabled_servers()
    if not servers:
        return None, []
    return build_mcp_manager(config), [entry.name.strip() for entry in servers]


__all__ = [
    "build_mcp_manager",
    "build_mcp_manager_if_configured",
    "to_server_config",
]
