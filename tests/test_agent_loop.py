"""Tests for the agent loop."""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

from config.config import AgentConfig, LLMConfig, TokenUsageConfig, TokenBudget
from internal.agent.agent import Agent, create_agent
from internal.compact.compact import deterministic_compact, should_force_compact
from internal.llm.client import LLMClient
from internal.permission.permission import create_default_permission_approver
from internal.skills import Skill, SkillManager
from internal.tools.tools import create_default_tool_registry
from internal.types.types import (
    FunctionCall,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class AgentLoopTests(unittest.TestCase):
    def test_chat_appends_user_message(self) -> None:
        agent = self._make_agent()
        with mock.patch.object(agent, "run_loop", return_value="done") as run_loop:
            result = agent.chat("hello")

        self.assertEqual(result, "done")
        self.assertEqual(agent.messages[-1].role, "user")
        self.assertEqual(agent.messages[-1].content, "hello")
        run_loop.assert_called_once()

    def test_run_loop_returns_text_response(self) -> None:
        agent = self._make_agent()
        response = LLMResponse(content="final answer", tool_calls=[])

        with mock.patch.object(agent, "_call_llm", return_value=response):
            result = agent.run_loop()

        self.assertEqual(result, "final answer")
        self.assertEqual(agent.messages[-1].role, "assistant")
        self.assertEqual(agent.state.status, "completed")

    def test_run_loop_executes_tools_after_permission(self) -> None:
        agent = self._make_agent()
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(
                name="echo",
                arguments=json.dumps({"text": "hi"}),
            ),
        )
        first = LLMResponse(content="Calling echo", tool_calls=[tool_call])
        second = LLMResponse(content="done", tool_calls=[])

        agent.tool_registry.register(
            ToolDefinition(
                name="echo",
                description="echo text",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            ),
            lambda args: args["text"],
        )

        with mock.patch.object(agent, "_call_llm", side_effect=[first, second]):
            with mock.patch.object(agent, "request_permission", return_value=True):
                result = agent.run_loop()

        self.assertEqual(result, "done")
        tool_messages = [message for message in agent.messages if message.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].content, "hi")

    def test_run_loop_skips_tool_when_permission_denied(self) -> None:
        agent = self._make_agent()
        tool_call = ToolCall(
            id="call_2",
            function=FunctionCall(
                name="echo",
                arguments=json.dumps({"text": "hi"}),
            ),
        )
        first = LLMResponse(content="Calling echo", tool_calls=[tool_call])
        second = LLMResponse(content="done", tool_calls=[])

        with mock.patch.object(agent, "_call_llm", side_effect=[first, second]):
            with mock.patch.object(agent, "request_permission", return_value=False):
                agent.run_loop()

        tool_messages = [message for message in agent.messages if message.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("skipped", tool_messages[0].content)

    def test_set_max_turns(self) -> None:
        agent = self._make_agent()
        agent.set_max_turns(5)
        self.assertEqual(agent.config.max_turns, 5)
        self.assertEqual(agent.state.max_turns, 5)

    def test_force_compact_threshold(self) -> None:
        self.assertTrue(should_force_compact(9000, 128000))
        self.assertFalse(should_force_compact(20000, 128000))

    def test_deterministic_compact_moves_old_messages(self) -> None:
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="old " * 1000),
            Message(role="assistant", content="old reply " * 1000),
            Message(role="user", content="new question"),
            Message(role="assistant", content="new answer"),
        ]
        hot, cold = deterministic_compact(messages, 1200)
        self.assertEqual(hot[0].role, "system")
        self.assertGreaterEqual(len(cold), 1)

    def test_mcp_tool_calls_run_concurrently_and_keep_message_order(self) -> None:
        agent = self._make_agent()
        barrier = threading.Barrier(2)
        state = {"active": 0, "max_active": 0}
        lock = threading.Lock()

        def make_handler(label: str):
            def handler(_: Dict[str, Any]) -> ToolResult:
                with lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                try:
                    barrier.wait(timeout=2)
                    return ToolResult(
                        tool_call_id="",
                        name=label,
                        content=f"{label}-result",
                    )
                finally:
                    with lock:
                        state["active"] -= 1

            return handler

        for name in ("remote_a", "remote_b"):
            agent.tool_registry.register(
                ToolDefinition(
                    name=name,
                    description=name,
                    input_schema={"type": "object", "properties": {}},
                    source="mcp",
                    server_name="demo",
                ),
                make_handler(name),
            )

        response = LLMResponse(
            content="Calling MCP tools",
            tool_calls=[
                ToolCall(
                    id="call_a",
                    function=FunctionCall(name="remote_a", arguments="{}"),
                ),
                ToolCall(
                    id="call_b",
                    function=FunctionCall(name="remote_b", arguments="{}"),
                ),
            ],
        )

        with mock.patch.object(agent, "request_permission", return_value=True):
            agent._handle_tool_calls(response)

        tool_messages = [message for message in agent.messages if message.role == "tool"]
        self.assertEqual(state["max_active"], 2)
        self.assertEqual(
            [message.tool_call_id for message in tool_messages[-2:]],
            ["call_a", "call_b"],
        )
        self.assertEqual(
            [message.content for message in tool_messages[-2:]],
            ["remote_a-result", "remote_b-result"],
        )

    def test_skills_only_expose_tools_for_current_user_match(self) -> None:
        agent = self._make_agent()
        agent.skill_manager = SkillManager(
            [
                Skill(
                    name="files",
                    keywords=("文件",),
                    tools=("read_file", "write_file"),
                    instructions="Handle files.",
                )
            ]
        )
        with mock.patch.object(agent, "run_loop", return_value="done"):
            agent.chat("请读取文件")

        with mock.patch.object(agent.client, "stream", return_value=[]) as stream:
            agent._call_llm()

        tools = stream.call_args.args[1]
        self.assertEqual([tool.name for tool in tools], ["read_file", "write_file"])
        self.assertEqual(agent.state.metadata["active_skills"], ["files"])
        self.assertEqual(
            agent.state.metadata["visible_tools"],
            ["read_file", "write_file"],
        )

    def test_skills_hide_all_tools_when_no_keyword_matches(self) -> None:
        agent = self._make_agent()
        agent.skill_manager = SkillManager(
            [
                Skill(
                    name="shell",
                    keywords=("命令",),
                    tools=("run_shell",),
                    instructions="Run commands carefully.",
                )
            ]
        )
        with mock.patch.object(agent, "run_loop", return_value="done"):
            agent.chat("只回答一个问题")

        with mock.patch.object(agent.client, "stream", return_value=[]) as stream:
            agent._call_llm()

        self.assertEqual(stream.call_args.args[1], [])
        self.assertEqual(agent.state.metadata["active_skills"], [])

    def test_skill_match_changes_between_chats_and_persists_within_loop(self) -> None:
        agent = self._make_agent()
        agent.skill_manager = SkillManager(
            [
                Skill(
                    name="read",
                    keywords=("read",),
                    tools=("read_file",),
                    instructions="Read first.",
                ),
                Skill(
                    name="shell",
                    keywords=("shell",),
                    tools=("run_shell",),
                    instructions="Use shell.",
                ),
            ]
        )
        with mock.patch.object(agent, "run_loop", return_value="done"):
            agent.chat("read config")
        with mock.patch.object(agent.client, "stream", return_value=[]) as stream:
            agent._call_llm()
            agent._call_llm()
        for call in stream.call_args_list:
            self.assertEqual([tool.name for tool in call.args[1]], ["read_file"])

        with mock.patch.object(agent, "run_loop", return_value="done"):
            agent.chat("use shell")
        with mock.patch.object(agent.client, "stream", return_value=[]) as stream:
            agent._call_llm()
        self.assertEqual(
            [tool.name for tool in stream.call_args.args[1]],
            ["run_shell"],
        )

    def test_agent_without_registered_skills_exposes_all_tools(self) -> None:
        agent = self._make_agent()
        agent.skill_manager = SkillManager()
        with mock.patch.object(agent.client, "stream", return_value=[]) as stream:
            agent._call_llm()
        self.assertEqual(
            [tool.name for tool in stream.call_args.args[1]],
            [tool.name for tool in agent.tool_registry.list_tools()],
        )

    def _make_agent(self) -> Agent:
        token_budget = TokenBudget(TokenUsageConfig(context_window=128000))
        config = AgentConfig(max_turns=15, workspace_root=".")
        client = LLMClient(LLMConfig(api_key="test"), token_budget)
        return Agent(
            client=client,
            tool_registry=create_default_tool_registry(Path("."), Path(".tool_outputs")),
            permission=create_default_permission_approver(),
            config=config,
            messages=[Message(role="system", content="test")],
            token_budget=token_budget,
        )


if __name__ == "__main__":
    unittest.main()
