"""Tests for the Agent's MCP management surface."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest import mock

from config.config import AgentConfig, LLMConfig, TokenBudget, TokenUsageConfig
from internal.agent import agent as agent_module
from internal.agent.agent import Agent, create_agent
from internal.llm.client import LLMClient, LLMHTTPError, LLMTimeoutError
from internal.mcp.manager import MCPManager, MCPToolRoute
from internal.permission.permission import create_default_permission_approver
from internal.tools.tools import create_default_tool_registry
from internal.types.types import (
    FunctionCall,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class FakeClient:
    """Minimal stand-in for ``MCPClient`` used by describe/set APIs."""

    def __init__(
        self,
        alias: str,
        *,
        mode: str = "stdio",
        enabled: bool = True,
        started: bool = False,
    ) -> None:
        self.server_alias = alias
        self.mode = mode
        self.enabled = enabled
        self._started = started

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self._started = False


class FakeManager(MCPManager):
    """MCPManager with discovery and registration stubbed out."""

    def __init__(self, clients: List[FakeClient]) -> None:
        super().__init__()
        self.failures: Dict[str, BaseException] = {}
        self.start_calls = 0
        for client in clients:
            self.add_client(client)

    def start(self, registry) -> Dict[str, BaseException]:
        self.start_calls += 1
        for client in self.list_clients():
            if client.enabled:
                client.start()
        return dict(self.failures)


def _make_agent(
    manager: Optional[MCPManager] = None,
    registry=None,
) -> Agent:
    token_budget = TokenBudget(TokenUsageConfig(context_window=128000))
    return Agent(
        client=LLMClient(LLMConfig(api_key="test"), token_budget),
        tool_registry=registry
        or create_default_tool_registry(Path("."), Path(".tool_outputs")),
        permission=create_default_permission_approver(),
        config=AgentConfig(max_turns=5, workspace_root="."),
        messages=[Message(role="system", content="test")],
        token_budget=token_budget,
        mcp_manager=manager,
    )


class DescribeMCPTests(unittest.TestCase):
    def test_no_manager_returns_empty_list(self) -> None:
        agent = _make_agent()

        self.assertEqual(agent.describe_mcp(), [])

    def test_reports_alias_mode_and_state(self) -> None:
        manager = FakeManager(
            [
                FakeClient("fs", mode="stdio", started=True),
                FakeClient("remote", mode="http", enabled=False),
            ]
        )
        agent = _make_agent(manager)

        described = {item["name"]: item for item in agent.describe_mcp()}

        self.assertEqual(described["fs"]["mode"], "stdio")
        self.assertTrue(described["fs"]["started"])
        self.assertTrue(described["fs"]["enabled"])
        self.assertEqual(described["remote"]["mode"], "http")
        self.assertFalse(described["remote"]["enabled"])

    def test_reports_tools_from_routes(self) -> None:
        manager = FakeManager([FakeClient("fs")])
        manager._public_routes["read_file_1"] = MCPToolRoute(
            public_name="read_file_1",
            server_alias="fs",
            remote_name="read_file",
        )
        agent = _make_agent(manager)

        described = agent.describe_mcp()

        self.assertEqual(described[0]["tools"], ["read_file_1"])

    def test_reports_start_errors(self) -> None:
        manager = FakeManager([FakeClient("fs")])
        manager.failures = {"fs": RuntimeError("boom")}
        agent = _make_agent(manager)
        agent.state.metadata["mcp_errors"] = {"fs": "RuntimeError: boom"}

        described = agent.describe_mcp()

        self.assertEqual(described[0]["error"], "RuntimeError: boom")


class ReloadMCPTests(unittest.TestCase):
    def test_no_manager_is_a_noop(self) -> None:
        agent = _make_agent()

        self.assertEqual(agent.reload_mcp(), {})

    def test_reload_restarts_clients_and_refreshes_tools(self) -> None:
        manager = FakeManager([FakeClient("fs")])
        agent = _make_agent(manager)

        failures = agent.reload_mcp()

        self.assertEqual(failures, {})
        self.assertEqual(manager.start_calls, 1)
        self.assertTrue(manager.get_client("fs").started)

    def test_reload_records_failures_in_metadata(self) -> None:
        manager = FakeManager([FakeClient("fs")])
        manager.failures = {"fs": RuntimeError("boom")}
        agent = _make_agent(manager)

        agent.reload_mcp()

        self.assertIn("boom", agent.state.metadata["mcp_errors"]["fs"])


class SetServerEnabledTests(unittest.TestCase):
    def test_unknown_alias_returns_false(self) -> None:
        agent = _make_agent(FakeManager([FakeClient("fs")]))

        self.assertFalse(agent.set_mcp_server_enabled("nope", False))

    def test_no_manager_returns_false(self) -> None:
        agent = _make_agent()

        self.assertFalse(agent.set_mcp_server_enabled("fs", True))

    def test_disable_unregisters_tools(self) -> None:
        registry = create_default_tool_registry(Path("."), Path(".tool_outputs"))
        registry.register(
            ToolDefinition(
                name="read_file_1",
                description="remote",
                input_schema={},
                source="mcp",
                server_name="fs",
            ),
            lambda arguments: ToolResult(
                tool_call_id="", name="read_file_1", content=""
            ),
        )
        manager = FakeManager([FakeClient("fs")])
        manager._public_routes["read_file_1"] = MCPToolRoute(
            public_name="read_file_1",
            server_alias="fs",
            remote_name="read_file",
        )
        manager._remote_to_public[("fs", "read_file")] = "read_file_1"
        agent = _make_agent(manager, registry=registry)

        self.assertTrue(agent.set_mcp_server_enabled("fs", False))
        self.assertFalse(registry.has_tool("read_file_1"))
        self.assertFalse(manager.get_client("fs").enabled)

    def test_enable_triggers_rediscovery(self) -> None:
        manager = FakeManager([FakeClient("fs", enabled=False)])
        agent = _make_agent(manager)

        self.assertTrue(agent.set_mcp_server_enabled("fs", True))
        self.assertTrue(manager.get_client("fs").enabled)
        self.assertEqual(manager.start_calls, 1)


class CancelTests(unittest.TestCase):
    def test_cancel_sets_status(self) -> None:
        agent = _make_agent()

        agent.cancel()

        self.assertEqual(agent.state.status, "cancelled")

    def test_cancel_during_a_turn_stops_the_loop(self) -> None:
        agent = _make_agent()
        calls: List[int] = []

        def fake_call_llm() -> LLMResponse:
            calls.append(1)
            agent.cancel()
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(
                            name="list_dir", arguments='{"path": "."}'
                        ),
                    )
                ],
            )

        agent._call_llm = fake_call_llm  # type: ignore[assignment]

        result = agent.run_loop()

        self.assertEqual(result, "Cancelled by user")
        self.assertEqual(len(calls), 1)
        self.assertEqual(agent.state.status, "cancelled")

    def test_cancel_flag_is_reset_by_a_new_run(self) -> None:
        agent = _make_agent()
        agent.cancel()
        responses = iter([LLMResponse(content="ok")])
        agent._call_llm = lambda: next(responses)  # type: ignore[assignment]

        result = agent.run_loop()

        self.assertEqual(result, "ok")


class CreateAgentMCPWiringTests(unittest.TestCase):
    def test_empty_servers_do_not_build_a_manager(self) -> None:
        from config.config import AppConfig, MCPConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                llm=LLMConfig(api_key="test"),
                agent=AgentConfig(max_turns=3, workspace_root=tmp),
                mcp=MCPConfig(),
            )
            agent = create_agent(
                config,
                resolve_api_key=False,
                restore_session=False,
                system_prompt="test",
            )
            try:
                self.assertIsNone(agent.mcp_manager)
            finally:
                agent.close()

    def test_disabled_mcp_does_not_build_a_manager(self) -> None:
        from config.config import AppConfig, MCPConfig, MCPServerEntry

        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                llm=LLMConfig(api_key="test"),
                agent=AgentConfig(max_turns=3, workspace_root=tmp),
                mcp=MCPConfig(
                    enabled=False,
                    servers=[MCPServerEntry(name="fs", command="npx")],
                ),
            )
            agent = create_agent(
                config,
                resolve_api_key=False,
                restore_session=False,
                system_prompt="test",
            )
            try:
                self.assertIsNone(agent.mcp_manager)
            finally:
                agent.close()


class LLMFailureHandlingTests(unittest.TestCase):
    """A provider failure must not escape run_loop as a raw traceback."""

    def test_stream_delta_survives_an_unencodable_character(self) -> None:
        """A GBK console must not abort the turn when the model emits an emoji."""

        class GbkStream(io.StringIO):
            encoding = "gbk"

        stream = GbkStream()
        stream.write = lambda text: io.StringIO.write(  # type: ignore[method-assign]
            stream, text.encode("gbk").decode("gbk")
        )

        with mock.patch("sys.stdout", stream):
            agent_module._safe_write("hello \U0001f60a world", end="")

        self.assertIn("hello", stream.getvalue())
        self.assertIn("world", stream.getvalue())

    def test_llm_error_is_reported_gracefully(self) -> None:
        agent = _make_agent()

        def _boom() -> LLMResponse:
            raise LLMHTTPError(
                400,
                "Bad Request",
                '{"error":{"message":"Access denied, please make sure your '
                'account is in good standing.","type":"Arrearage"}}',
            )

        agent._call_llm = _boom  # type: ignore[assignment]

        result = agent.run_loop()

        self.assertIn("account is in good standing", result)
        self.assertEqual(agent.state.status, "error")
        self.assertEqual(agent.state.last_error, result)

    def test_llm_error_does_not_append_an_assistant_message(self) -> None:
        agent = _make_agent()
        before = len(agent.messages)
        agent._call_llm = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            LLMTimeoutError("too slow")
        )

        agent.run_loop()

        self.assertEqual(len(agent.messages), before)


if __name__ == "__main__":
    unittest.main()
