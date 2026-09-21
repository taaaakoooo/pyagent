"""Normalization helpers for MCP tool definitions."""

from __future__ import annotations

from typing import Any, Dict

from internal.types.types import ToolDefinition


def normalize_mcp_tool(
    server_alias: str,
    raw_tool: Dict[str, Any],
    public_name: str,
) -> ToolDefinition:
    """Convert a raw MCP tool schema into the project's common definition."""
    remote_name = raw_tool.get("name")
    if not isinstance(remote_name, str) or not remote_name.strip():
        raise ValueError("MCP tool definition requires a non-empty name")

    description = raw_tool.get("description")
    if not isinstance(description, str) or not description.strip():
        description = (
            f"MCP tool '{remote_name.strip()}' provided by server '{server_alias}'."
        )

    input_schema = raw_tool.get("inputSchema")
    if input_schema is None:
        input_schema = {"type": "object", "properties": {}}
    if not isinstance(input_schema, dict):
        raise ValueError(
            f"MCP tool '{remote_name}' inputSchema must be an object"
        )
    if input_schema.get("type", "object") != "object":
        raise ValueError(
            f"MCP tool '{remote_name}' inputSchema type must be 'object'"
        )

    return ToolDefinition(
        name=public_name,
        description=description.strip(),
        input_schema=dict(input_schema),
        source="mcp",
        server_name=server_alias,
    )


__all__ = ["normalize_mcp_tool"]
