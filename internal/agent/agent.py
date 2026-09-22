"""Agent runtime structure and callback definitions."""

from __future__ import annotations

import json
import sys
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from config.config import (
    AppConfig,
    AgentConfig,
    MCPConfig,
    TokenBudget,
    load_config,
)
from internal.compact.compact import (
    deterministic_compact,
    estimate_messages_tokens,
    should_force_compact,
)
from internal.llm.client import LLMClient, LLMError, _merge_tool_call
from internal.mcp.factory import build_mcp_manager_if_configured
from internal.mcp.manager import MCPManager
from internal.permission.permission import (
    PermissionCallback,
    create_default_permission_approver,
    enrich_permission_request,
    is_read_only_tool,
)
from internal.skills import (
    Skill,
    SkillError,
    SkillLoadResult,
    SkillManager,
    SkillMatch,
    discover_skill_files,
    load_skill_markdown,
)
from internal.tools.executors import ToolContext
from internal.tools.tools import create_default_tool_registry
from internal.session.session import (
    SessionError,
    SessionStore,
    SessionSummary,
    generate_session_id,
    validate_session_id,
)
from internal.types.types import (
    AgentState,
    AgentStatus,
    LLMResponse,
    Message,
    PermissionDecision,
    PermissionRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

StreamContentCallback = Callable[[str], None]
ToolStatusCallback = Callable[[str, str, Dict[str, Any]], None]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "Use available tools when needed to inspect or modify the workspace. "
    "Prefer safe, minimal changes and explain your reasoning clearly."
)
DEFAULT_TOOL_OUTPUT_DIR = ".tool_outputs"


def _resolve_skill_dir(value: str, workspace_root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    return candidate


def _safe_write(text: str, end: str = "\n") -> None:
    """Write to stdout without ever dying on an un-encodable character.

    The Windows console frequently runs a legacy code page (GBK here), so a
    model reply containing an emoji would raise UnicodeEncodeError and abort
    the whole turn. Fall back to a lossy encode instead.
    """
    stream = sys.stdout
    try:
        stream.write(text + end)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        stream.write(
            text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            + end
        )
    stream.flush()


def _default_stream_delta(delta: str) -> None:
    _safe_write(delta, end="")


def _default_tool_status(tool_name: str, status: str, details: Dict[str, Any]) -> None:
    detail_text = ""
    if details:
        detail_text = " " + " ".join(f"{key}={value}" for key, value in details.items())
    _safe_write(f"\n[{tool_name}] {status}{detail_text}")


def _default_permission_prompt(permission: PermissionRequest) -> PermissionDecision:
    if permission.risk_tags:
        print(f"Risk tags: {', '.join(permission.risk_tags)}")
    if permission.risk_description:
        print(f"Risk: {permission.risk_description}")
    if permission.preview_diff:
        unified = permission.preview_diff.get("unified", "")
        if unified:
            print("--- diff preview ---")
            print(unified.rstrip())

    prompt = (
        f"Allow tool '{permission.tool_name}'? "
        "[y]es once / [n]o once / [a]lways allow / [d]eny session: "
    )
    answer = input(prompt).strip().lower()
    if answer in {"y", "yes"}:
        return PermissionDecision.ALLOW_ONCE
    if answer in {"a", "always"}:
        return PermissionDecision.ALLOW_SESSION
    if answer in {"d", "deny"}:
        return PermissionDecision.DENY_SESSION
    return PermissionDecision.DENY_ONCE


class ToolRegistryProtocol(Protocol):
    """Registry that exposes and executes tools for the agent."""

    def list_tools(self) -> List[ToolDefinition]:
        ...

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        ...


class PermissionApprover(Protocol):
    """Component that decides whether a tool call is allowed.

    An approver may additionally implement ``reset_session() -> None`` to drop
    any conversation-scoped decisions it is caching; ``Agent.switch_session``
    calls it so grants cannot leak across sessions. It is optional, so this
    Protocol deliberately does not declare it.
    """

    def resolve(
        self,
        permission: PermissionRequest,
        *,
        callback: Optional[PermissionCallback],
        tool_definition: Optional[ToolDefinition],
        workspace_root: Path,
    ) -> bool:
        ...

    def request(self, permission: PermissionRequest) -> bool:
        ...


@dataclass
class Agent:
    """Core agent runtime: dependencies, history, callbacks, and token tracking."""

    client: LLMClient
    tool_registry: ToolRegistryProtocol
    permission: PermissionApprover
    config: AgentConfig
    messages: List[Message] = field(default_factory=list)
    system_prompt: str = ""
    tool_output_path: Path = field(default_factory=lambda: Path(".tool_outputs"))
    on_stream_delta: Optional[StreamContentCallback] = None
    on_tool_status: Optional[ToolStatusCallback] = None
    on_permission: Optional[PermissionCallback] = None
    estimated_tokens: int = 0
    context_window_limit: int = 128000
    state: AgentState = field(default_factory=AgentState)
    token_budget: Optional[TokenBudget] = None
    session_store: Optional[SessionStore] = None
    session_id: str = "default"
    session_storage_dir: Optional[Path] = None
    tool_output_root: Optional[Path] = None
    restored_message_count: int = 0
    mcp_manager: Optional[MCPManager] = None
    skill_manager: Optional[SkillManager] = None
    skill_unmatched_policy: str = "readonly"
    always_visible_sources: Tuple[str, ...] = ("mcp",)
    skills_directory: Optional[Path] = None
    _skills_enabled: bool = field(default=True, init=False)
    _base_system_prompt: str = field(default="", init=False)
    _active_skill_match: Optional[SkillMatch] = field(default=None, init=False)
    _cancel_requested: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.state.max_turns != self.config.max_turns:
            self.state.max_turns = self.config.max_turns
        if self.token_budget is not None:
            self.context_window_limit = self.token_budget.limit

    @property
    def remaining_tokens(self) -> int:
        context_used = self._estimate_context_tokens()
        return max(0, self.context_window_limit - context_used)

    @property
    def token_usage_ratio(self) -> float:
        limit = self.context_window_limit
        if limit <= 0:
            return 0.0
        return self._estimate_context_tokens() / limit

    def emit_stream_delta(self, delta: str) -> None:
        if delta and self.on_stream_delta is not None:
            self.on_stream_delta(delta)

    def emit_tool_status(
        self,
        tool_name: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.on_tool_status is not None:
            self.on_tool_status(tool_name, status, details or {})

    def request_permission(
        self,
        permission: PermissionRequest,
        tool_definition: Optional[ToolDefinition] = None,
    ) -> bool:
        self.state.status = AgentStatus.WAITING_PERMISSION
        workspace_root = Path(self.config.workspace_root).resolve()
        enriched = enrich_permission_request(
            permission,
            tool_definition=tool_definition,
            workspace_root=workspace_root,
        )
        self.state.pending_permission = enriched
        approved = self.permission.resolve(
            enriched,
            callback=self.on_permission,
            tool_definition=tool_definition,
            workspace_root=workspace_root,
        )
        self.state.pending_permission = None
        return approved

    def update_estimated_tokens(self, tokens: int) -> None:
        self.estimated_tokens = max(0, tokens)

    def _estimate_context_tokens(self) -> int:
        return estimate_messages_tokens(self.messages)

    def set_max_turns(self, max_turns: int) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.config.max_turns = max_turns
        self.state.max_turns = max_turns

    def restore_session(self) -> bool:
        """Load persisted messages from the session store."""
        if self.session_store is None:
            return False

        loaded = self.session_store.load_messages()
        self.messages = loaded
        self.restored_message_count = len(loaded)
        self._ensure_system_prompt()
        self.update_estimated_tokens(self._estimate_context_tokens())
        return self.restored_message_count >= 1

    def _ensure_system_prompt(self) -> None:
        if self._has_system_message():
            return
        if not self.system_prompt:
            return
        if self.session_store is not None and not self.session_store.is_empty():
            self.messages.insert(0, Message(role="system", content=self.system_prompt))
            return
        system_message = Message(role="system", content=self.system_prompt)
        self.messages.insert(0, system_message)
        self._persist_message(system_message)

    def _has_system_message(self) -> bool:
        return any(message.role == "system" for message in self.messages)

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new empty session and switch to it."""
        target = validate_session_id(session_id or generate_session_id())
        if self.session_storage_dir is not None:
            existing = {summary.session_id for summary in self.list_sessions()}
            if target in existing:
                suffix = 2
                while f"{target}-{suffix}" in existing:
                    suffix += 1
                target = f"{target}-{suffix}"
        self.switch_session(target)
        return target

    def switch_session(self, session_id: str) -> bool:
        """Point the agent at another session, restoring its history."""
        target = validate_session_id(session_id)
        if target == self.session_id and self.session_store is not None:
            return self.restored_message_count >= 1
        if self.session_storage_dir is None:
            raise SessionError("Agent has no session storage directory configured")

        # Grants are keyed by tool name only, so leaving them in place would let
        # an "always allow run_shell" from one session silently authorize the
        # same tool in the next one. This guard sits after the same-session
        # early return above, so re-selecting the active session is a no-op.
        self._reset_permission_session()

        store = SessionStore(self.session_storage_dir, target)
        self.session_store = store
        self.session_id = target
        self.messages = []
        self.restored_message_count = 0

        if self.tool_output_root is not None:
            self.tool_output_path = self.tool_output_root / target
            self.tool_output_path.mkdir(parents=True, exist_ok=True)
            self._rebind_tool_context()

        loaded = store.load_messages()
        self.messages = loaded
        self.restored_message_count = len(loaded)
        self._ensure_system_prompt()
        if self.skill_manager:
            self.import_skills_to_system_prompt()
        self.update_estimated_tokens(self._estimate_context_tokens())
        return self.restored_message_count >= 1

    def list_sessions(self) -> List[SessionSummary]:
        """List sessions available in the current storage directory."""
        if self.session_storage_dir is None:
            return []
        return SessionStore.list_sessions(self.session_storage_dir)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file, refusing to delete the active session."""
        target = validate_session_id(session_id)
        if target == self.session_id:
            raise SessionError("Cannot delete the active session")
        if self.session_storage_dir is None:
            raise SessionError("Agent has no session storage directory configured")
        return SessionStore(self.session_storage_dir, target).delete()

    def _reset_permission_session(self) -> None:
        """Drop conversation-scoped permission decisions, if the approver keeps any.

        ``reset_session`` is optional: ``PermissionApprover`` only requires
        ``resolve``/``request``, so a custom approver without session state is
        perfectly valid and simply needs no reset.
        """
        reset = getattr(self.permission, "reset_session", None)
        if callable(reset):
            reset()

    def _rebind_tool_context(self) -> None:
        set_context = getattr(self.tool_registry, "set_tool_context", None)
        if not callable(set_context):
            return
        workspace_root = Path(self.config.workspace_root).resolve()
        set_context(
            ToolContext(
                workspace_root=workspace_root,
                tool_output_path=self.tool_output_path.resolve(),
            )
        )

    def _activate_skills(self, user_input: str) -> None:
        if not self._skills_enabled or not self.skill_manager:
            self._active_skill_match = None
            self.state.metadata["active_skills"] = []
            return
        self._active_skill_match = self.skill_manager.match(user_input)
        self.state.metadata["active_skills"] = list(
            self._active_skill_match.skill_names
        )

    def _visible_tools(self) -> List[ToolDefinition]:
        """Tools exposed to the LLM for the current turn."""
        tools = self.tool_registry.list_tools()
        if not self._skills_enabled or not self.skill_manager:
            return tools

        pinned = [
            tool
            for tool in tools
            if tool.source in self.always_visible_sources
        ]
        pinned_names = {tool.name for tool in pinned}

        match = self._active_skill_match
        if match is not None and match.tool_names:
            allowed = set(match.tool_names)
            return [
                tool
                for tool in tools
                if tool.name in allowed or tool.source in self.always_visible_sources
            ]

        if self.skill_unmatched_policy == "all":
            return tools
        if self.skill_unmatched_policy == "readonly":
            return pinned + [
                tool
                for tool in tools
                if tool.name not in pinned_names
                and is_read_only_tool(tool.name, tool.source)
            ]
        return pinned

    def set_skills_enabled(self, enabled: bool) -> None:
        """Toggle skill routing without discarding registered skills."""
        self._skills_enabled = bool(enabled)
        if not self._skills_enabled:
            self._active_skill_match = None
            self.state.metadata["active_skills"] = []

    def describe_skills(self) -> List[Dict[str, Any]]:
        """Summarize registered skills for display."""
        if self.skill_manager is None:
            return []
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "keywords": list(skill.keywords),
                "tools": list(skill.tools),
            }
            for skill in self.skill_manager.list_skills()
        ]

    def reload_skills(self, directory: Optional[Path] = None) -> SkillLoadResult:
        """Re-read skill files, keeping previously loaded skills on failure."""
        if self.skill_manager is None:
            self.skill_manager = SkillManager()
        target = directory or self.skills_directory
        if target is None:
            return SkillLoadResult(
                skills=tuple(self.skill_manager.list_skills()),
                errors={},
            )
        return self.skill_manager.load_directory(Path(target))

    def skills_enabled(self) -> bool:
        """Whether skill-based tool routing is currently on."""
        return self._skills_enabled

    def describe_mcp(self) -> List[Dict[str, Any]]:
        """Summarize every configured MCP server for display."""
        if self.mcp_manager is None:
            return []

        errors = self.state.metadata.get("mcp_errors") or {}
        routes = self.mcp_manager.list_tool_routes()
        tools_by_server: Dict[str, List[str]] = {}
        for route in routes:
            tools_by_server.setdefault(route.server_alias, []).append(
                route.public_name
            )

        servers: List[Dict[str, Any]] = []
        for client in self.mcp_manager.list_clients():
            alias = client.server_alias
            error = errors.get(alias)
            servers.append(
                {
                    "name": alias,
                    "mode": client.mode,
                    "enabled": client.enabled,
                    "started": client.started,
                    "tools": sorted(tools_by_server.get(alias, [])),
                    "error": str(error) if error is not None else None,
                }
            )
        return servers

    def reload_mcp(self) -> Dict[str, BaseException]:
        """Re-run discovery for every enabled MCP server."""
        if self.mcp_manager is None:
            return {}

        failures = self.mcp_manager.start(self.tool_registry)
        self.state.metadata["mcp_errors"] = {
            alias: str(error) for alias, error in failures.items()
        }
        if self.skill_manager:
            self.import_skills_to_system_prompt()
        return failures

    def set_mcp_server_enabled(self, server_alias: str, enabled: bool) -> bool:
        """Toggle one MCP server; returns False when the alias is unknown."""
        if self.mcp_manager is None:
            return False
        try:
            self.mcp_manager.set_client_enabled(
                server_alias,
                enabled,
                self.tool_registry,
            )
        except KeyError:
            return False

        if enabled:
            self.reload_mcp()
        else:
            self.state.metadata["mcp_errors"] = {
                alias: error
                for alias, error in (
                    self.state.metadata.get("mcp_errors") or {}
                ).items()
                if alias != server_alias
            }
            if self.skill_manager:
                self.import_skills_to_system_prompt()
        return True

    def cancel(self) -> None:
        """Request cancellation, honoured between agent loop turns."""
        self._cancel_requested = True
        self.state.status = AgentStatus.CANCELLED

    def import_skills_to_system_prompt(
        self,
        skills: Optional[Iterable[Skill]] = None,
        *,
        skill_files: Optional[Iterable[Path]] = None,
        names: Optional[Sequence[str]] = None,
    ) -> str:
        """Register skills and merge their instructions into the system prompt."""
        if self.skill_manager is None:
            self.skill_manager = SkillManager()
        for skill in skills or ():
            self.skill_manager.register(skill)
        for path in skill_files or ():
            self.skill_manager.load_markdown(Path(path))

        system_message = next(
            (message for message in self.messages if message.role == "system"),
            None,
        )
        if not self._base_system_prompt:
            content = (
                system_message.content
                if system_message is not None and system_message.content is not None
                else self.system_prompt
            )
            self._base_system_prompt = content or ""
        self.system_prompt = self.skill_manager.merge_system_prompt(
            self._base_system_prompt,
            names,
        )
        if system_message is None:
            self._ensure_system_prompt()
        else:
            system_message.content = self.system_prompt
        self.update_estimated_tokens(self._estimate_context_tokens())
        return self.system_prompt

    def chat(self, user_input: str) -> str:
        """Append a user message and run the agent loop."""
        self._activate_skills(user_input)
        user_message = Message(role="user", content=user_input)
        self.messages.append(user_message)
        self._persist_message(user_message)
        self.update_estimated_tokens(self._estimate_context_tokens())
        return self.run_loop()

    def run_loop(self) -> str:
        """Execute the agent loop until a final answer or turn limit."""
        self.state.turn = 0
        self._cancel_requested = False
        final_answer: Optional[str] = None

        while self.state.turn < self.state.max_turns:
            if self._cancel_requested:
                self.state.status = AgentStatus.CANCELLED
                self.state.last_error = "Cancelled by user"
                return self.state.last_error

            self.state.turn += 1
            self._maybe_force_compact(self.state.turn)

            try:
                response = self._call_llm()
            except LLMError as exc:
                self.state.status = AgentStatus.ERROR
                self.state.last_error = str(exc)
                self.emit_tool_status(
                    "llm",
                    "error",
                    {"message": str(exc)},
                )
                return self.state.last_error

            self._archive_llm_output(self.state.turn, response)

            if response.tool_calls:
                final_answer = self._handle_tool_calls(response)
                continue

            final_answer = response.content or ""
            self._append_assistant_message(
                Message(role="assistant", content=final_answer)
            )
            self.state.status = AgentStatus.COMPLETED
            return final_answer

        self.state.status = AgentStatus.ERROR
        self.state.last_error = f"Reached max turns ({self.state.max_turns})"
        self.emit_tool_status("agent", "error", {"message": self.state.last_error})
        return self.state.last_error

    def _maybe_force_compact(self, turn: int) -> None:
        if not should_force_compact(self.remaining_tokens, self.context_window_limit):
            return

        hot_messages, cold_messages = deterministic_compact(
            self.messages,
            self.context_window_limit,
        )
        self.messages = hot_messages
        self.update_estimated_tokens(self._estimate_context_tokens())
        if cold_messages:
            self._archive_cold_messages(turn, cold_messages)

    def _call_llm(self) -> LLMResponse:
        tools = self._visible_tools()
        self.state.metadata["visible_tools"] = [tool.name for tool in tools]
        self.state.status = AgentStatus.STREAMING
        self.state.is_streaming = True

        content_parts: List[str] = []
        tool_calls_by_index: Dict[int, ToolCall] = {}
        finish_reason: Optional[str] = None
        usage: Dict[str, int] = {}

        try:
            for chunk in self.client.stream(self.messages, tools):
                if chunk.content_delta:
                    content_parts.append(chunk.content_delta)
                    self.emit_stream_delta(chunk.content_delta)
                for tool_call in chunk.tool_calls:
                    _merge_tool_call(tool_calls_by_index, tool_call)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.usage:
                    usage = chunk.usage
        finally:
            self.state.is_streaming = False

        self._sync_token_usage(usage)
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=[
                tool_calls_by_index[index]
                for index in sorted(tool_calls_by_index)
            ],
            finish_reason=finish_reason,
            usage=usage,
        )

    def _sync_token_usage(self, usage: Dict[str, int]) -> None:
        if usage and self.token_budget is not None:
            warning = self.token_budget.add_usage(usage)
            if warning:
                self.emit_tool_status("token_budget", "warning", {"message": warning})
        self.update_estimated_tokens(self._estimate_context_tokens())

    def _handle_tool_calls(self, response: LLMResponse) -> Optional[str]:
        assistant_content = response.content or self._describe_tool_calls(response.tool_calls)
        assistant_message = Message(
            role="assistant",
            content=assistant_content,
            tool_calls=response.tool_calls,
        )
        self._append_assistant_message(assistant_message)
        if assistant_content:
            self.emit_stream_delta(f"\n{assistant_content}\n")

        prepared: List[Dict[str, Any]] = []
        results: Dict[int, ToolResult] = {}
        for index, tool_call in enumerate(response.tool_calls):
            tool_name = tool_call.function.name
            arguments = self._parse_tool_arguments(tool_call)
            self.state.status = AgentStatus.EXECUTING_TOOL
            self.state.active_tool_name = tool_name
            self.state.active_tool_call_id = tool_call.id

            tool_definition = self._get_tool_definition(tool_name)
            permission = PermissionRequest(
                request_id=str(uuid.uuid4()),
                tool_name=tool_name,
                arguments=arguments,
                description=f"Execute tool '{tool_name}'",
            )
            approved = self.request_permission(permission, tool_definition)
            if not approved:
                self.emit_tool_status(tool_name, "skipped", {"reason": "denied"})
                results[index] = ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_name,
                    content=f"Tool '{tool_name}' was skipped: permission denied.",
                    is_error=True,
                )
            prepared.append(
                {
                    "index": index,
                    "tool_call": tool_call,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "definition": tool_definition,
                    "approved": approved,
                }
            )

        mcp_entries = [
            entry
            for entry in prepared
            if entry["approved"]
            and entry["definition"] is not None
            and entry["definition"].source == "mcp"
        ]
        futures: Dict[int, Future[ToolResult]] = {}
        executor: Optional[ThreadPoolExecutor] = None
        if mcp_entries:
            executor = ThreadPoolExecutor(
                max_workers=len(mcp_entries),
                thread_name_prefix="mcp-tool",
            )
            for entry in mcp_entries:
                self.emit_tool_status(
                    entry["tool_name"],
                    "running",
                    entry["arguments"],
                )
                futures[entry["index"]] = executor.submit(
                    self.tool_registry.call_tool,
                    entry["tool_name"],
                    entry["arguments"],
                )

        try:
            for entry in prepared:
                index = entry["index"]
                if not entry["approved"] or index in futures:
                    continue
                self.emit_tool_status(
                    entry["tool_name"],
                    "running",
                    entry["arguments"],
                )
                try:
                    results[index] = self.tool_registry.call_tool(
                        entry["tool_name"],
                        entry["arguments"],
                    )
                except Exception as exc:
                    results[index] = ToolResult(
                        tool_call_id=entry["tool_call"].id,
                        name=entry["tool_name"],
                        content=str(exc),
                        is_error=True,
                    )

            for index, future in futures.items():
                entry = prepared[index]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    results[index] = ToolResult(
                        tool_call_id=entry["tool_call"].id,
                        name=entry["tool_name"],
                        content=str(exc),
                        is_error=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        for entry in prepared:
            index = entry["index"]
            tool_call = entry["tool_call"]
            tool_name = entry["tool_name"]
            result = results[index]
            result.tool_call_id = tool_call.id
            if entry["approved"]:
                self.emit_tool_status(
                    tool_name,
                    "error" if result.is_error else "completed",
                    {"content": str(result.content)},
                )
            tool_message = Message(
                role="tool",
                name=tool_name,
                tool_call_id=tool_call.id,
                content=str(result.content),
            )
            self._append_tool_message(tool_message)

        self.state.active_tool_name = None
        self.state.active_tool_call_id = None
        self.state.status = AgentStatus.THINKING
        return assistant_content

    def _get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        for tool in self.tool_registry.list_tools():
            if tool.name == tool_name:
                return tool
        return None

    def _parse_tool_arguments(self, tool_call: ToolCall) -> Dict[str, Any]:
        raw = tool_call.function.arguments or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _describe_tool_calls(self, tool_calls: List[ToolCall]) -> str:
        names = [tool_call.function.name for tool_call in tool_calls if tool_call.function.name]
        if not names:
            return "Calling tools to continue the task."
        joined = ", ".join(names)
        return f"Calling tools: {joined}"

    def _append_assistant_message(self, message: Message) -> None:
        self.messages.append(message)
        self._persist_message(message)
        self.update_estimated_tokens(self._estimate_context_tokens())

    def _append_tool_message(self, message: Message) -> None:
        self.messages.append(message)
        self._persist_message(message)
        self._archive_tool_result(message)
        self.update_estimated_tokens(self._estimate_context_tokens())

    def _persist_message(self, message: Message) -> None:
        if self.session_store is not None:
            self.session_store.append_message(message)

    def _archive_llm_output(self, turn: int, response: LLMResponse) -> None:
        turn_dir = self.tool_output_path / "turns"
        turn_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "turn": turn,
            "content": response.content,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
                for tool_call in response.tool_calls
            ],
            "raw": response.raw,
        }
        output_file = turn_dir / f"turn_{turn:03d}.json"
        output_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _archive_cold_messages(self, turn: int, cold_messages: List[Message]) -> None:
        cold_dir = self.tool_output_path / "cold"
        cold_dir.mkdir(parents=True, exist_ok=True)
        cold_file = cold_dir / f"turn_{turn:03d}.jsonl"
        with cold_file.open("a", encoding="utf-8") as handle:
            for message in cold_messages:
                handle.write(
                    json.dumps(
                        {
                            "role": message.role,
                            "content": message.content,
                            "name": message.name,
                            "tool_call_id": message.tool_call_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if self.session_store is not None:
            self.session_store.append_event(
                {
                    "type": "compact",
                    "turn": turn,
                    "cold_count": len(cold_messages),
                    "cold_file": str(cold_file),
                }
            )

    def _archive_tool_result(self, message: Message) -> None:
        results_dir = self.tool_output_path / "tool_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        name = message.name or "tool"
        file_name = f"{name}_{message.tool_call_id or uuid.uuid4().hex}.txt"
        result_file = results_dir / file_name
        result_file.write_text(message.content or "", encoding="utf-8")

    def set_callbacks(
        self,
        *,
        on_stream_delta: Optional[StreamContentCallback] = None,
        on_tool_status: Optional[ToolStatusCallback] = None,
        on_permission: Optional[PermissionCallback] = None,
    ) -> Agent:
        """Inject or replace runtime callbacks. Unspecified callbacks are kept."""
        if on_stream_delta is not None:
            self.on_stream_delta = on_stream_delta
        if on_tool_status is not None:
            self.on_tool_status = on_tool_status
        if on_permission is not None:
            self.on_permission = on_permission
        return self

    def close(self) -> None:
        """Release MCP transports and child processes. Safe to call repeatedly."""
        if self.mcp_manager is not None:
            failures = self.mcp_manager.close()
            if failures:
                self.state.metadata["mcp_close_errors"] = {
                    alias: str(error) for alias, error in failures.items()
                }


def create_agent(
    app_config: Optional[AppConfig] = None,
    *,
    tool_registry: Optional[ToolRegistryProtocol] = None,
    permission: Optional[PermissionApprover] = None,
    system_prompt: Optional[str] = None,
    tool_output_path: Optional[Path] = None,
    on_stream_delta: Optional[StreamContentCallback] = None,
    on_tool_status: Optional[ToolStatusCallback] = None,
    on_permission: Optional[PermissionCallback] = None,
    resolve_api_key: bool = True,
    restore_session: bool = True,
    mcp_manager: Optional[MCPManager] = None,
    mcp_config: Optional[MCPConfig] = None,
    skills: Optional[Iterable[Skill]] = None,
    skill_files: Optional[Iterable[Path]] = None,
    import_skills_into_system_prompt: Optional[bool] = None,
    session_id: Optional[str] = None,
) -> Agent:
    """Build an Agent with project defaults from config."""
    config = app_config or load_config(resolve_key=resolve_api_key)
    token_budget = TokenBudget(config.token_usage)
    workspace_root = Path(config.agent.workspace_root).resolve()
    active_session = validate_session_id(session_id or config.session.default_id)

    tool_output_root = tool_output_path or (workspace_root / DEFAULT_TOOL_OUTPUT_DIR)
    if tool_output_path is None and config.session.isolate_tool_output:
        output_path = tool_output_root / active_session
    else:
        output_path = tool_output_root
    output_path.mkdir(parents=True, exist_ok=True)

    prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
    registry = tool_registry or create_default_tool_registry(
        workspace_root=workspace_root,
        tool_output_path=output_path,
    )
    mcp_errors: Dict[str, BaseException] = {}
    if mcp_manager is None:
        mcp_settings = mcp_config if mcp_config is not None else config.mcp
        mcp_manager, _ = build_mcp_manager_if_configured(mcp_settings)
    if mcp_manager is not None:
        mcp_errors = mcp_manager.start(registry)
    approver = permission or create_default_permission_approver()

    skill_manager = SkillManager()
    skill_errors: Dict[str, str] = {}
    for skill in skills or ():
        skill_manager.register(skill)
    for skill_file in skill_files or ():
        try:
            skill_manager.replace(load_skill_markdown(Path(skill_file)))
        except (SkillError, OSError) as exc:
            skill_errors[str(skill_file)] = str(exc)

    if not skills and not skill_files and config.skills.enabled:
        load_result = skill_manager.load_directory(
            _resolve_skill_dir(config.skills.dir, workspace_root)
        )
        skill_errors.update(load_result.errors)
        for extra in config.skills.files:
            try:
                skill_manager.replace(load_skill_markdown(
                    _resolve_skill_dir(extra, workspace_root)
                ))
            except (SkillError, OSError) as exc:
                skill_errors[str(extra)] = str(exc)

    if import_skills_into_system_prompt is None:
        inject_prompt = config.skills.inject_system_prompt
    else:
        inject_prompt = import_skills_into_system_prompt

    session_store = SessionStore(
        Path(config.session.storage_dir),
        active_session,
    )

    agent = Agent(
        client=LLMClient(config.llm, token_budget),
        tool_registry=registry,
        permission=approver,
        config=config.agent,
        messages=[],
        system_prompt=prompt,
        tool_output_path=output_path,
        on_stream_delta=on_stream_delta or _default_stream_delta,
        on_tool_status=on_tool_status or _default_tool_status,
        on_permission=on_permission or _default_permission_prompt,
        estimated_tokens=0,
        context_window_limit=config.token_usage.context_window,
        state=AgentState(max_turns=config.agent.max_turns),
        token_budget=token_budget,
        session_store=session_store,
        session_id=active_session,
        session_storage_dir=Path(config.session.storage_dir),
        tool_output_root=tool_output_root,
        mcp_manager=mcp_manager,
        skill_manager=skill_manager,
        skill_unmatched_policy=config.skills.unmatched_tools,
        always_visible_sources=tuple(config.skills.always_visible_sources),
        skills_directory=_resolve_skill_dir(config.skills.dir, workspace_root),
    )
    if mcp_errors:
        agent.state.metadata["mcp_errors"] = {
            alias: str(error) for alias, error in mcp_errors.items()
        }
    if skill_errors:
        agent.state.metadata["skill_errors"] = dict(skill_errors)

    restored = agent.restore_session() if restore_session else False
    if not restored:
        agent._ensure_system_prompt()
    if inject_prompt and skill_manager:
        agent.import_skills_to_system_prompt()

    return agent


__all__ = [
    "Agent",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TOOL_OUTPUT_DIR",
    "PermissionApprover",
    "PermissionCallback",
    "StreamContentCallback",
    "ToolRegistryProtocol",
    "ToolStatusCallback",
    "create_agent",
]
