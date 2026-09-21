"""Permission approval for tool execution."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from internal.diff.diff import build_file_diff
from internal.tools.path import resolve_workspace_path_info
from internal.types.types import PermissionDecision, PermissionRequest, ToolDefinition

PermissionCallback = Callable[[PermissionRequest], PermissionDecision]

AUTO_ALLOW_PREFIXES = ("read_", "list_", "get_")
INTERACTIVE_LOCAL_TOOLS = frozenset({"write_file", "replace_line", "run_shell"})

SHELL_RISK_PATTERNS: Tuple[Tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\brm\s+(-[^\s]*f[^\s]*\s+.*-[^\s]*r|-[^\s]*r[^\s]*\s+.*-[^\s]*f|-rf|-fr)\b", re.I), "destructive_delete", "Recursive force delete"),
    (re.compile(r"\brm\s+-[^\s]*f\b", re.I), "force_delete", "Force delete files"),
    (re.compile(r"\bsudo\b", re.I), "elevated_privilege", "Uses sudo / elevated privileges"),
    (re.compile(r"\b(chmod\s+777|chmod\s+-R\s+777)\b", re.I), "permission_change", "Makes files world-writable"),
    (re.compile(r"\b(mkfs|diskpart|format\s+[a-z]:)\b", re.I), "disk_format", "Formats or repartitions storage"),
    (re.compile(r"\bdd\s+if=", re.I), "raw_disk_write", "Writes directly to a block device"),
    (re.compile(r"\bcurl[^\n|]*\|\s*(ba)?sh\b", re.I), "remote_code_exec", "Pipes remote content into a shell"),
    (re.compile(r"\bwget[^\n|]*\|\s*(ba)?sh\b", re.I), "remote_code_exec", "Pipes downloaded content into a shell"),
    (re.compile(r">\s*/dev/[a-z]", re.I), "device_redirect", "Redirects output to a device file"),
    (re.compile(r"\b(powershell\s+-enc|iex\s*\(|invoke-expression)\b", re.I), "obfuscated_exec", "Executes obfuscated or encoded commands"),
    (re.compile(r"\b(rmdir\s+/s|rd\s+/s|del\s+/f)\b", re.I), "destructive_delete", "Force deletes directories or files on Windows"),
)


def analyze_shell_risk(command: str) -> Tuple[List[str], str]:
    """Return matched risk tags and a human-readable description for a shell command."""
    tags: List[str] = []
    descriptions: List[str] = []
    seen: Set[str] = set()

    for pattern, tag, description in SHELL_RISK_PATTERNS:
        if tag in seen:
            continue
        if pattern.search(command):
            seen.add(tag)
            tags.append(tag)
            descriptions.append(description)

    if not tags:
        return [], "Shell command may modify the workspace or run arbitrary programs."

    return tags, "; ".join(descriptions)


def is_read_only_tool(tool_name: str, tool_source: str = "local") -> bool:
    """Return True when a tool is considered read-only and low risk."""
    if tool_source == "mcp":
        return False
    return any(tool_name.startswith(prefix) for prefix in AUTO_ALLOW_PREFIXES)


def requires_interactive_approval(
    tool_name: str,
    tool_definition: Optional[ToolDefinition] = None,
) -> bool:
    """Return True when a tool needs user confirmation."""
    source = tool_definition.source if tool_definition is not None else "local"
    if source == "mcp":
        return True
    return tool_name in INTERACTIVE_LOCAL_TOOLS


def build_preview_diff(
    tool_name: str,
    arguments: Dict[str, object],
    workspace_root: Path,
) -> Optional[Dict[str, object]]:
    """Build a file diff preview for mutating tools before execution."""
    if tool_name == "write_file":
        path_arg = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))
        resolved = resolve_workspace_path_info(workspace_root, path_arg)
        old_content = (
            resolved.absolute.read_text(encoding="utf-8")
            if resolved.absolute.is_file()
            else ""
        )
        return build_file_diff(resolved.relative, "write", old_content, content).to_dict()

    if tool_name == "replace_line":
        path_arg = str(arguments.get("path", ""))
        line_number = int(arguments.get("line_number", 0))
        new_content = str(arguments.get("new_content", ""))
        resolved = resolve_workspace_path_info(workspace_root, path_arg)
        if not resolved.absolute.is_file():
            return None
        old_content = resolved.absolute.read_text(encoding="utf-8")
        lines = old_content.splitlines()
        if line_number < 1 or line_number > len(lines):
            return None
        lines[line_number - 1] = new_content
        new_file_content = "\n".join(lines) + ("\n" if lines else "")
        return build_file_diff(
            resolved.relative,
            "replace_line",
            old_content,
            new_file_content,
        ).to_dict()

    return None


def enrich_permission_request(
    request: PermissionRequest,
    *,
    tool_definition: Optional[ToolDefinition],
    workspace_root: Path,
) -> PermissionRequest:
    """Attach tool source, risk metadata, and diff preview to a permission request."""
    source = tool_definition.source if tool_definition is not None else "local"
    server_name = tool_definition.server_name if tool_definition is not None else None
    risk_tags: List[str] = []
    risk_description = request.description or f"Execute tool '{request.tool_name}'"
    risk_level = request.risk_level
    preview_diff = request.preview_diff

    if request.tool_name == "run_shell":
        command = str(request.arguments.get("command", ""))
        risk_tags, risk_description = analyze_shell_risk(command)
        risk_level = "high" if risk_tags else "medium"
    elif source == "mcp":
        risk_tags = ["mcp_tool"]
        risk_description = (
            f"MCP tool '{request.tool_name}'"
            + (f" from server '{server_name}'" if server_name else "")
            + " runs outside the local sandbox."
        )
        risk_level = "high"
    elif request.tool_name in {"write_file", "replace_line"}:
        risk_tags = ["file_mutation"]
        risk_description = f"Tool '{request.tool_name}' will modify workspace files."
        risk_level = "medium"
        preview_diff = build_preview_diff(
            request.tool_name,
            request.arguments,
            workspace_root,
        )

    return replace(
        request,
        description=risk_description,
        risk_level=risk_level,
        risk_tags=risk_tags,
        risk_description=risk_description,
        preview_diff=preview_diff,
        tool_source=source,
        server_name=server_name,
    )


class ConversationPermissionManager:
    """Session-scoped permission manager with interactive approval support."""

    def __init__(
        self,
        auto_allow_prefixes: Tuple[str, ...] = AUTO_ALLOW_PREFIXES,
    ) -> None:
        self._auto_allow_prefixes = auto_allow_prefixes
        self._session_allowed: Set[str] = set()
        self._session_denied: Set[str] = set()

    def reset_session(self) -> None:
        """Clear conversation-level allow/deny decisions."""
        self._session_allowed.clear()
        self._session_denied.clear()

    def resolve(
        self,
        request: PermissionRequest,
        *,
        callback: Optional[PermissionCallback] = None,
        tool_definition: Optional[ToolDefinition] = None,
        workspace_root: Optional[Path] = None,
    ) -> bool:
        """Run the three-step permission flow and return whether execution is allowed."""
        enriched = enrich_permission_request(
            request,
            tool_definition=tool_definition,
            workspace_root=workspace_root or Path("."),
        )

        if enriched.tool_name in self._session_allowed:
            return True
        if enriched.tool_name in self._session_denied:
            return False

        if is_read_only_tool(enriched.tool_name, enriched.tool_source):
            return True

        if not requires_interactive_approval(enriched.tool_name, tool_definition):
            return True

        if callback is None:
            return is_read_only_tool(enriched.tool_name, enriched.tool_source)

        decision = callback(enriched)
        return self._apply_decision(enriched.tool_name, decision)

    def request(self, permission: PermissionRequest) -> bool:
        """Legacy entry point: resolve without an interactive callback."""
        return self.resolve(permission, callback=None)

    def _apply_decision(self, tool_name: str, decision: PermissionDecision) -> bool:
        if decision == PermissionDecision.ALLOW_SESSION:
            self._session_allowed.add(tool_name)
            self._session_denied.discard(tool_name)
            return True
        if decision == PermissionDecision.DENY_SESSION:
            self._session_denied.add(tool_name)
            self._session_allowed.discard(tool_name)
            return False
        if decision == PermissionDecision.ALLOW_ONCE:
            return True
        return False


DefaultPermissionApprover = ConversationPermissionManager


def create_default_permission_approver() -> ConversationPermissionManager:
    return ConversationPermissionManager()


__all__ = [
    "AUTO_ALLOW_PREFIXES",
    "ConversationPermissionManager",
    "DefaultPermissionApprover",
    "PermissionCallback",
    "PermissionDecision",
    "analyze_shell_risk",
    "build_preview_diff",
    "create_default_permission_approver",
    "enrich_permission_request",
    "is_read_only_tool",
    "requires_interactive_approval",
]
