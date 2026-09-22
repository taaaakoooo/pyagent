"""Bridge between the Elm runtime and ``Agent``.

The ``*_cmd`` functions are installed into a ``CmdContext``: the UI thread stays
responsive while an agent turn runs on a worker thread, and every agent callback
becomes a dispatched message.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from internal.agent.agent import Agent
from internal.types.types import PermissionDecision, PermissionRequest
from tui.msgs import (
    AgentDeltaMsg,
    AgentDoneMsg,
    AgentErrorMsg,
    AgentPermissionMsg,
    AgentStartedMsg,
    AgentToolMsg,
    McpChangedMsg,
    Msg,
    NoticeMsg,
    SessionChangedMsg,
    SkillsChangedMsg,
)
from tui.update import CmdContext

Dispatch = Callable[[Msg], None]

PERMISSION_TIMEOUT_SECONDS = 300.0


class PermissionResolver:
    """Hand-off object letting the UI thread answer a blocked callback."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.decision: str = PermissionDecision.DENY_ONCE

    def resolve(self, decision: str) -> None:
        self.decision = decision
        self.event.set()

    def wait(self, timeout: float = PERMISSION_TIMEOUT_SECONDS) -> str:
        """Block until answered; a timeout denies once rather than hanging."""
        if not self.event.wait(timeout):
            return PermissionDecision.DENY_ONCE
        return self.decision


class AgentBridge:
    """Owns the agent callbacks and the worker thread that runs turns."""

    def __init__(self, agent: Agent, dispatch: Dispatch) -> None:
        self.agent = agent
        self.dispatch = dispatch
        self._thread: Optional[threading.Thread] = None
        self._pending_resolver: Optional[PermissionResolver] = None
        self._lock = threading.Lock()

    # -- agent callbacks ------------------------------------------------

    def on_stream_delta(self, delta: str) -> None:
        self.dispatch(AgentDeltaMsg(delta))

    def on_tool_status(
        self,
        tool_name: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dispatch(AgentToolMsg(tool_name, status, dict(details or {})))

    def on_permission(self, request: PermissionRequest) -> str:
        """Block the worker thread until the UI answers the prompt."""
        resolver = PermissionResolver()
        with self._lock:
            self._pending_resolver = resolver
        self.dispatch(AgentPermissionMsg(request, resolver.resolve))
        return resolver.wait()

    def install_callbacks(self) -> None:
        """Replace the agent's default stdout/``input()`` callbacks."""
        self.agent.set_callbacks(
            on_stream_delta=self.on_stream_delta,
            on_tool_status=self.on_tool_status,
            on_permission=self.on_permission,
        )

    # -- commands -------------------------------------------------------

    def run_agent(self, prompt: str) -> None:
        """Start one agent turn on a daemon worker thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self.dispatch(
                    NoticeMsg("agent 仍在回复，请稍候或按 Esc 取消本轮")
                )
                return
            self._thread = threading.Thread(
                target=self._run_turn,
                args=(prompt,),
                name="tui-agent",
                daemon=True,
            )
            self._thread.start()

    def _run_turn(self, prompt: str) -> None:
        self.dispatch(AgentStartedMsg(prompt))
        try:
            self.install_callbacks()
            answer = self.agent.chat(prompt)
        except Exception as exc:  # noqa: BLE001 - surfaced in the transcript
            self.dispatch(AgentErrorMsg(f"{type(exc).__name__}: {exc}"))
            self._clear_pending_resolver()
            return

        state = self.agent.state
        cancelled = state.status == "cancelled"
        self.dispatch(
            AgentDoneMsg(
                answer=answer or "",
                error=None if cancelled else state.last_error,
                cancelled=cancelled,
            )
        )
        self._clear_pending_resolver()
        self.refresh_all()

    def resolve_permission(self, decision: str) -> None:
        """Release the permission callback currently blocked on the UI."""
        with self._lock:
            resolver = self._pending_resolver
            self._pending_resolver = None
        if resolver is not None:
            resolver.resolve(decision)

    def cancel_agent(self) -> None:
        """Ask the agent to stop at the next turn boundary."""
        self.agent.cancel()
        # Unblock any waiting approval so the loop can observe the cancellation.
        self.resolve_permission(PermissionDecision.DENY_ONCE)

    def _clear_pending_resolver(self) -> None:
        with self._lock:
            self._pending_resolver = None

    # -- snapshots ------------------------------------------------------

    def refresh_skills(self) -> None:
        self.dispatch(
            SkillsChangedMsg(
                skills=tuple(self.agent.describe_skills()),
                enabled=self.agent.skills_enabled(),
                unmatched_policy=self.agent.skill_unmatched_policy,
                always_visible=tuple(self.agent.always_visible_sources),
                errors=dict(self.agent.state.metadata.get("skill_errors") or {}),
            )
        )

    def refresh_mcp(self) -> None:
        self.dispatch(
            McpChangedMsg(
                servers=tuple(self.agent.describe_mcp()),
                errors=dict(self.agent.state.metadata.get("mcp_errors") or {}),
            )
        )

    def refresh_sessions(self) -> None:
        sessions = _session_rows(self.agent)
        self.dispatch(
            SessionChangedMsg(
                session_id=self.agent.session_id,
                sessions=tuple(sessions),
                message_count=len(self.agent.messages),
                archive_dir=str(self.agent.tool_output_path),
                restored_count=self.agent.restored_message_count,
            )
        )

    def refresh_all(self) -> None:
        self.refresh_skills()
        self.refresh_mcp()
        self.refresh_sessions()

    # -- panel actions --------------------------------------------------

    def reload_skills(self) -> None:
        result = self.agent.reload_skills()
        self.dispatch(NoticeMsg(f"已重新加载 {len(result.skills)} 个技能。"))
        self.refresh_skills()

    def reload_mcp(self) -> None:
        failures = self.agent.reload_mcp()
        if failures:
            self.dispatch(
                NoticeMsg(f"MCP 重新加载完成，{len(failures)} 个错误。")
            )
        else:
            self.dispatch(NoticeMsg("MCP 已重新加载。"))
        self.refresh_mcp()

    def set_mcp_enabled(self, server_alias: str, enabled: bool) -> None:
        if not self.agent.set_mcp_server_enabled(server_alias, enabled):
            self.dispatch(NoticeMsg(f"未知的 MCP 服务: {server_alias}"))
        else:
            action = "已启用" if enabled else "已停用"
            self.dispatch(NoticeMsg(f"MCP 服务 '{server_alias}' {action}。"))
        self.refresh_mcp()

    def set_skills_enabled(self, enabled: bool) -> None:
        self.agent.set_skills_enabled(enabled)
        self.dispatch(
            NoticeMsg("技能已启用。" if enabled else "技能已关闭。")
        )
        self.refresh_skills()

    def switch_session(self, session_id: str) -> None:
        try:
            restored = self.agent.switch_session(session_id)
        except Exception as exc:  # noqa: BLE001 - surfaced as a notice
            self.dispatch(NoticeMsg(f"无法切换会话: {exc}"))
            return
        detail = "已恢复历史" if restored else "暂无历史"
        self.dispatch(NoticeMsg(f"会话 '{session_id}': {detail}。"))
        self.refresh_all()

    def create_session(self) -> None:
        try:
            created = self.agent.create_session()
        except Exception as exc:  # noqa: BLE001 - surfaced as a notice
            self.dispatch(NoticeMsg(f"无法创建会话: {exc}"))
            return
        self.dispatch(NoticeMsg(f"已创建并切换到 '{created}'。"))
        self.refresh_all()

    def delete_session(self, session_id: str) -> None:
        try:
            deleted = self.agent.delete_session(session_id)
        except Exception as exc:  # noqa: BLE001 - surfaced as a notice
            self.dispatch(NoticeMsg(f"无法删除 '{session_id}': {exc}"))
            return
        self.dispatch(
            NoticeMsg(
                f"已删除会话 '{session_id}'。"
                if deleted
                else f"未找到会话 '{session_id}'。"
            )
        )
        self.refresh_all()


def _session_rows(agent: Agent) -> List[Dict[str, Any]]:
    return [
        {
            "session_id": summary.session_id,
            "message_count": summary.message_count,
            "updated_at": _format_timestamp(summary.updated_at),
            "preview": summary.preview,
        }
        for summary in agent.list_sessions()
    ]


def _format_timestamp(value: Any) -> str:
    try:
        return value.astimezone().strftime("%Y-%m-%d %H:%M")
    except (AttributeError, ValueError):
        return str(value)


def build_cmd_context(
    bridge: AgentBridge,
    quit_callback: Callable[[], None],
) -> CmdContext:
    """Wire every ``Cmd`` the update function can emit."""
    return CmdContext(
        run_agent=bridge.run_agent,
        resolve_permission=bridge.resolve_permission,
        cancel_agent=bridge.cancel_agent,
        save_skills=bridge.set_skills_enabled,
        reload_skills=bridge.reload_skills,
        reload_mcp=bridge.reload_mcp,
        set_mcp_enabled=bridge.set_mcp_enabled,
        create_session=bridge.create_session,
        switch_session=bridge.switch_session,
        delete_session=bridge.delete_session,
        refresh_view=bridge.refresh_all,
        quit=quit_callback,
    )


__all__ = [
    "AgentBridge",
    "PERMISSION_TIMEOUT_SECONDS",
    "PermissionResolver",
    "build_cmd_context",
]
