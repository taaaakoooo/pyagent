"""Tool definitions exposed to the LLM (schemas only, no executors)."""

from __future__ import annotations

from typing import Dict, List, Tuple

from internal.types.types import ToolDefinition

TOOL_NAMES: Tuple[str, ...] = (
    "read_file",
    "write_file",
    "replace_line",
    "list_dir",
    "read_regex",
    "run_shell",
    "read_disk_data",
)


def _object_schema(
    properties: Dict[str, Dict[str, object]],
    required: List[str],
) -> Dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def build_read_file_definition() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read a text file inside the workspace.",
        input_schema=_object_schema(
            {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "offset": {
                    "type": "integer",
                    "description": "1-based start line (optional).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (optional).",
                },
            },
            ["path"],
        ),
    )


def build_write_file_definition() -> ToolDefinition:
    return ToolDefinition(
        name="write_file",
        description="Create or overwrite a text file inside the workspace.",
        input_schema=_object_schema(
            {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "content": {"type": "string", "description": "Full file content to write."},
            },
            ["path", "content"],
        ),
    )


def build_replace_line_definition() -> ToolDefinition:
    return ToolDefinition(
        name="replace_line",
        description="Replace a single line in a workspace file by 1-based line number.",
        input_schema=_object_schema(
            {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "line_number": {
                    "type": "integer",
                    "description": "1-based line number to replace.",
                },
                "new_content": {"type": "string", "description": "Replacement line content."},
            },
            ["path", "line_number", "new_content"],
        ),
    )


def build_list_dir_definition() -> ToolDefinition:
    return ToolDefinition(
        name="list_dir",
        description="List files and directories inside the workspace.",
        input_schema=_object_schema(
            {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory path.",
                    "default": ".",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list recursively.",
                    "default": False,
                },
            },
            [],
        ),
    )


def build_read_regex_definition() -> ToolDefinition:
    return ToolDefinition(
        name="read_regex",
        description="Search a workspace file with a regular expression and return matching lines.",
        input_schema=_object_schema(
            {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "pattern": {"type": "string", "description": "Regular expression pattern."},
                "max_matches": {
                    "type": "integer",
                    "description": "Maximum number of matches to return.",
                    "default": 100,
                },
            },
            ["path", "pattern"],
        ),
    )


def build_run_shell_definition() -> ToolDefinition:
    return ToolDefinition(
        name="run_shell",
        description="Execute a shell command with cwd restricted to the workspace.",
        input_schema=_object_schema(
            {
                "command": {"type": "string", "description": "Shell command to execute."},
                "cwd": {
                    "type": "string",
                    "description": "Optional workspace-relative working directory.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Optional command timeout in seconds.",
                    "default": 60,
                },
            },
            ["command"],
        ),
    )


def build_read_disk_data_definition() -> ToolDefinition:
    return ToolDefinition(
        name="read_disk_data",
        description=(
            "Read archived tool output from .tool_outputs/tool_results/. "
            "Use tool_call_id or filename; omit both to list recent archived outputs."
        ),
        input_schema=_object_schema(
            {
                "tool_call_id": {
                    "type": "string",
                    "description": "Tool call id used in the archived filename.",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename relative to tool_results/.",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based start line when reading file content (optional).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (optional).",
                },
            },
            [],
        ),
    )


_DEFINITION_BUILDERS = {
    "read_file": build_read_file_definition,
    "write_file": build_write_file_definition,
    "replace_line": build_replace_line_definition,
    "list_dir": build_list_dir_definition,
    "read_regex": build_read_regex_definition,
    "run_shell": build_run_shell_definition,
    "read_disk_data": build_read_disk_data_definition,
}


def build_all_definitions() -> List[ToolDefinition]:
    """Return all tool definitions in fixed ``TOOL_NAMES`` order."""
    return [_DEFINITION_BUILDERS[name]() for name in TOOL_NAMES]


__all__ = [
    "TOOL_NAMES",
    "build_all_definitions",
    "build_list_dir_definition",
    "build_read_disk_data_definition",
    "build_read_file_definition",
    "build_read_regex_definition",
    "build_replace_line_definition",
    "build_run_shell_definition",
    "build_write_file_definition",
]
