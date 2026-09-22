"""Tests for the TUI <-> Agent bridge, including the permission handshake."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config import AgentConfig, LLMConfig, TokenBudget, TokenUsageConfig
from internal.agent.agent import Agent
from internal.llm.client import LLMClient
from internal.permission.permission import create_default_permission_approver
from internal.tools.tools import create_default_tool_registry
from internal.types.types import (
    FunctionCall,
    LLMResponse,
    Message,
    PermissionDecision,
    PermissionRequest,
    ToolCall,
)
from tui.agent_bridge import AgentBridge, PermissionResolver, build_cmd_context
from tui.msgs import (
    AgentDeltaMsg,
    AgentDoneMsg,
    AgentErrorMsg,
    AgentPermissionMsg,
    AgentToolMsg,
    McpChangedMsg,
    NoticeMsg,
    SessionChangedMsg,
    SkillsChangedMsg,
)
from tui.update import CmdContext


class Collector:
    """Thread-safe message collector standing in for ``Program.dispatch``."""

    def __init__(self) -> None:
        self.messages: List[Any] = []
        self._lock = threading.Lock()
        self._event = threading.Event()

    def __call__(self, msg: Any) -> None:
        with self._lock:
            self.messages.append(msg)
        self._event.set()

    def wait_for(self, predicate, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if any(predicate(msg) for msg in self.messages):
                    return True
            self._event.clear()
            self._event.wait(0.05)
        return False

    def of_type(self, cls) -> List[Any]:
        with self._lock:
            return [msg for msg in self.messages if isinstance(msg, cls)]


def _make_agent() -> Agent:
    token_budget = TokenBudget(TokenUsageConfig(context_window=128000))
    return Agent(
        client=LLMClient(LLMConfig(api_key="test"), token_budget),
        tool_registry=create_default_tool_registry(
            Path("."), Path(".tool_outputs")
        ),
        permission=create_default_permission_approver(),
        config=AgentConfig(max_turns=3, workspace_root="."),
        messages=[Message(role="system", content="test")],
        token_budget=token_budget,
    )


class PermissionResolverTests(unittest.TestCase):
    def test_wait_returns_the_recorded_decision(self) -> None:
        resolver = PermissionResolver()
        threading.Timer(0.01, resolver.resolve, args=("allow_once",)).start()

        self.assertEqual(resolver.wait(timeout=2.0), "allow_once")

    def test_wait_times_out_to_deny_once(self) -> None:
        resolver = PermissionResolver()

        self.assertEqual(resolver.wait(timeout=0.01), PermissionDecision.DENY_ONCE)


class PermissionHandshakeTests(unittest.TestCase):
    def test_callback_blocks_until_the_ui_answers(self) -> None:
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)
        request = PermissionRequest(request_id="r1", tool_name="run_shell")

        result: Dict[str, str] = {}

        def _call() -> None:
            result["decision"] = bridge.on_permission(request)

        worker = threading.Thread(target=_call, daemon=True)
        worker.start()

        self.assertTrue(
            collector.wait_for(lambda msg: isinstance(msg, AgentPermissionMsg)),
            "the permission prompt was never dispatched",
        )

        bridge.resolve_permission(PermissionDecision.ALLOW_SESSION)
        worker.join(timeout=2.0)

        self.assertEqual(result["decision"], PermissionDecision.ALLOW_SESSION)

    def test_prompt_message_carries_the_request(self) -> None:
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)
        request = PermissionRequest(request_id="r1", tool_name="run_shell")

        threading.Thread(
            target=lambda: bridge.on_permission(request), daemon=True
        ).start()
        collector.wait_for(lambda msg: isinstance(msg, AgentPermissionMsg))

        prompt = collector.of_type(AgentPermissionMsg)[0]
        self.assertIs(prompt.request, request)

    def test_cancel_releases_a_pending_permission(self) -> None:
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)
        request = PermissionRequest(request_id="r1", tool_name="run_shell")

        result: Dict[str, str] = {}

        def _call() -> None:
            result["decision"] = bridge.on_permission(request)

        worker = threading.Thread(target=_call, daemon=True)
        worker.start()
        collector.wait_for(lambda msg: isinstance(msg, AgentPermissionMsg))

        bridge.cancel_agent()
        worker.join(timeout=2.0)

        self.assertEqual(result["decision"], PermissionDecision.DENY_ONCE)

    def test_resolve_without_a_pending_request_is_safe(self) -> None:
        bridge = AgentBridge(_make_agent(), Collector())

        bridge.resolve_permission(PermissionDecision.ALLOW_ONCE)


class RunAgentTests(unittest.TestCase):
    def _bridge(self, responses: List[LLMResponse]):
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)
        calls: List[LLMResponse] = list(responses)
        agent._call_llm = lambda: calls.pop(0)  # type: ignore[assignment]
        return bridge, collector, agent

    def test_run_agent_dispatches_started_deltas_and_done(self) -> None:
        bridge, collector, agent = self._bridge([LLMResponse(content="answer")])

        def _emit(**kwargs) -> None:
            pass

        bridge.run_agent("hello")

        self.assertTrue(
            collector.wait_for(lambda msg: isinstance(msg, AgentDoneMsg)),
            "the turn never completed",
        )
        done = collector.of_type(AgentDoneMsg)[0]
        self.assertEqual(done.answer, "answer")
        self.assertIsNone(done.error)
        self.assertFalse(done.cancelled)

    def test_callbacks_do_not_write_to_stdout(self) -> None:
        bridge, collector, agent = self._bridge([LLMResponse(content="answer")])

        bridge.on_stream_delta("chunk")
        bridge.on_tool_status("read_file", "running", {"path": "a"})

        self.assertEqual(collector.of_type(AgentDeltaMsg)[0].delta, "chunk")
        self.assertEqual(
            collector.of_type(AgentToolMsg)[0].tool_name, "read_file"
        )

    def test_install_callbacks_replaces_the_defaults(self) -> None:
        bridge, _, agent = self._bridge([])

        bridge.install_callbacks()

        self.assertEqual(agent.on_stream_delta, bridge.on_stream_delta)
        self.assertEqual(agent.on_permission, bridge.on_permission)

    def test_error_in_the_turn_is_dispatched(self) -> None:
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)
        agent._call_llm = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("kaboom")
        )

        bridge.run_agent("hello")

        self.assertTrue(
            collector.wait_for(lambda msg: isinstance(msg, AgentErrorMsg)),
            "the failure was never reported to the UI",
        )
        self.assertIn("kaboom", collector.of_type(AgentErrorMsg)[0].message)

    def test_second_run_while_running_is_rejected(self) -> None:
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)
        release = threading.Event()

        def _slow_call() -> LLMResponse:
            release.wait(2.0)
            return LLMResponse(content="late")

        agent._call_llm = _slow_call  # type: ignore[assignment]
        bridge.run_agent("first")
        bridge.run_agent("second")

        self.assertTrue(
            collector.wait_for(
                lambda msg: isinstance(msg, NoticeMsg)
                and "仍在回复" in msg.text
            )
        )
        release.set()


class CommandContextTests(unittest.TestCase):
    def test_build_cmd_context_wires_every_hook(self) -> None:
        agent = _make_agent()
        bridge = AgentBridge(agent, Collector())
        quit_called: List[bool] = []

        context = build_cmd_context(bridge, lambda: quit_called.append(True))

        for hook in (
            "run_agent",
            "resolve_permission",
            "cancel_agent",
            "save_skills",
            "reload_skills",
            "reload_mcp",
            "set_mcp_enabled",
            "create_session",
            "switch_session",
            "delete_session",
            "refresh_view",
            "quit",
        ):
            with self.subTest(hook=hook):
                self.assertTrue(callable(getattr(context, hook)), hook)

        context.quit()
        self.assertEqual(quit_called, [True])

    def test_refresh_all_dispatches_three_snapshots(self) -> None:
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)

        bridge.refresh_all()

        self.assertEqual(len(collector.of_type(SkillsChangedMsg)), 1)
        self.assertEqual(len(collector.of_type(McpChangedMsg)), 1)
        self.assertEqual(len(collector.of_type(SessionChangedMsg)), 1)

    def test_refresh_skills_reports_agent_state(self) -> None:
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)

        bridge.refresh_skills()

        snapshot = collector.of_type(SkillsChangedMsg)[0]
        self.assertTrue(snapshot.enabled)
        self.assertEqual(snapshot.unmatched_policy, "readonly")
        self.assertEqual(snapshot.always_visible, ("mcp",))

    def test_set_mcp_enabled_reports_unknown_alias(self) -> None:
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)

        bridge.set_mcp_enabled("missing", True)

        notices = [msg.text for msg in collector.of_type(NoticeMsg)]
        self.assertTrue(any("未知的 MCP 服务" in text for text in notices))

    def test_reload_mcp_without_a_manager_is_a_noop(self) -> None:
        agent = _make_agent()
        collector = Collector()
        bridge = AgentBridge(agent, collector)

        bridge.reload_mcp()

        notices = [msg.text for msg in collector.of_type(NoticeMsg)]
        self.assertTrue(any("MCP 已重新加载" in text for text in notices))


if __name__ == "__main__":
    unittest.main()
