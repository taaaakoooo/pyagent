"""Local tool registry."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List

from internal.diff.diff import ToolExecutorResult
from internal.tools.definitions import TOOL_NAMES, build_all_definitions
from internal.tools.executors import EXECUTOR_MAP, ToolContext
from internal.tools.path import ensure_tool_output_in_workspace
from internal.types.types import ToolDefinition, ToolResult

ToolHandler = Callable[[Dict[str, Any]], Any]


class ToolRegistry:
    """In-memory registry for local tools."""

    def __init__(self) -> None:
        self._definitions: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, ToolHandler] = {}
        self._registration_order: List[str] = []

    def register(
        self,
        definition: ToolDefinition,
        handler: ToolHandler,
    ) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler
        self._registration_order.append(definition.name)

    def list_tools(self) -> List[ToolDefinition]:
        local_names = [name for name in TOOL_NAMES if name in self._definitions]
        dynamic_names = [
            name
            for name in self._registration_order
            if name in self._definitions and name not in TOOL_NAMES
        ]
        return [self._definitions[name] for name in local_names + dynamic_names]

    def has_tool(self, name: str) -> bool:
        return name in self._definitions

    def update_definition(self, definition: ToolDefinition) -> None:
        if definition.name not in self._definitions:
            raise KeyError(f"Unknown tool: {definition.name}")
        self._definitions[definition.name] = definition

    def unregister(self, name: str) -> bool:
        if name not in self._definitions:
            return False
        self._definitions.pop(name, None)
        self._handlers.pop(name, None)
        self._registration_order = [
            registered_name
            for registered_name in self._registration_order
            if registered_name != name
        ]
        return True

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                tool_call_id="",
                name=name,
                content=f"Unknown tool: {name}",
                is_error=True,
            )

        try:
            result = handler(arguments)
            if isinstance(result, ToolResult):
                return result
            if isinstance(result, ToolExecutorResult):
                return ToolResult(
                    tool_call_id="",
                    name=name,
                    content=result.summary,
                    metadata=result.to_metadata(),
                )
            return ToolResult(
                tool_call_id="",
                name=name,
                content=result,
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id="",
                name=name,
                content=str(exc),
                is_error=True,
            )


def create_default_tool_registry(
    workspace_root: Path,
    tool_output_path: Path,
) -> ToolRegistry:
    """Register built-in workspace tools in fixed order."""
    resolved_workspace = workspace_root.resolve()
    resolved_tool_output = tool_output_path.resolve()
    ensure_tool_output_in_workspace(resolved_workspace, resolved_tool_output)

    registry = ToolRegistry()
    context = ToolContext(
        workspace_root=resolved_workspace,
        tool_output_path=resolved_tool_output,
    )

    for definition in build_all_definitions():
        executor = EXECUTOR_MAP.get(definition.name)
        if executor is None:
            continue
        registry.register(definition, partial(executor, ctx=context))

    return registry


__all__ = ["ToolHandler", "ToolRegistry", "create_default_tool_registry"]
