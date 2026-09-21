"""Tests for session restore and message round-trip."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

from config.config import AgentConfig, AppConfig, LLMConfig, SessionConfig
from internal.agent.agent import Agent, DEFAULT_SYSTEM_PROMPT, create_agent
from internal.session.session import SessionStore
from internal.skills import Skill
from internal.types.types import Message, ToolCall, FunctionCall


def _count_jsonl_messages(session_file: Path, role: Optional[str] = None) -> int:
    if not session_file.is_file():
        return 0
    count = 0
    with session_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") != "message":
                continue
            message = event.get("message") or {}
            if role is None or message.get("role") == role:
                count += 1
    return count


def _make_app_config(session_dir: Path, workspace_root: Path) -> AppConfig:
    return AppConfig(
        llm=LLMConfig(api_key="test"),
        agent=AgentConfig(workspace_root=str(workspace_root)),
        session=SessionConfig(storage_dir=str(session_dir)),
    )


class SessionRestoreTests(unittest.TestCase):
    def test_load_messages_empty_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            self.assertEqual(store.load_messages(), [])

    def test_load_messages_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            messages = [
                Message(role="user", content="hello"),
                Message(
                    role="assistant",
                    content="calling tool",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=FunctionCall(
                                name="echo",
                                arguments='{"text": "hi"}',
                            ),
                        )
                    ],
                ),
                Message(role="tool", name="echo", tool_call_id="call_1", content="hi"),
            ]
            store.append_messages(messages)
            loaded = store.load_messages()

        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0].content, "hello")
        self.assertEqual(loaded[1].tool_calls[0].function.name, "echo")
        self.assertEqual(loaded[2].content, "hi")

    def test_load_messages_skips_non_message_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            store.append_message(Message(role="user", content="keep me"))
            store.append_event({"type": "compact", "turn": 1})
            loaded = store.load_messages()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].content, "keep me")

    def test_restore_session_false_when_no_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            agent = Agent(
                client=mock.MagicMock(),
                tool_registry=mock.MagicMock(),
                permission=mock.MagicMock(),
                config=AgentConfig(),
                system_prompt="system prompt",
                session_store=store,
            )

            restored = agent.restore_session()

        self.assertFalse(restored)
        self.assertEqual(agent.restored_message_count, 0)
        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(agent.messages[0].role, "system")
        self.assertEqual(agent.messages[0].content, "system prompt")

    def test_restore_injects_system_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = SessionStore(tmp_path)
            store.append_message(Message(role="user", content="prior question"))

            agent = Agent(
                client=mock.MagicMock(),
                tool_registry=mock.MagicMock(),
                permission=mock.MagicMock(),
                config=AgentConfig(),
                system_prompt="injected system",
                session_store=store,
            )
            restored = agent.restore_session()

            self.assertTrue(restored)
            self.assertEqual(agent.messages[0].role, "system")
            self.assertEqual(agent.messages[0].content, "injected system")
            self.assertEqual(agent.messages[1].content, "prior question")
            self.assertEqual(_count_jsonl_messages(store.session_file, "system"), 1)

    def test_restore_does_not_duplicate_existing_system(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = SessionStore(tmp_path)
            store.append_message(Message(role="system", content="saved system"))
            store.append_message(Message(role="user", content="question"))

            agent = Agent(
                client=mock.MagicMock(),
                tool_registry=mock.MagicMock(),
                permission=mock.MagicMock(),
                config=AgentConfig(),
                system_prompt="default system",
                session_store=store,
            )
            agent.restore_session()

            system_messages = [m for m in agent.messages if m.role == "system"]
            self.assertEqual(len(system_messages), 1)
            self.assertEqual(system_messages[0].content, "saved system")
            self.assertEqual(_count_jsonl_messages(store.session_file, "system"), 1)

    def test_create_agent_fresh_session_has_system_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_dir = tmp_path / "sessions"
            config = _make_app_config(session_dir, tmp_path)
            agent = create_agent(config, resolve_api_key=False, restore_session=True)

        self.assertEqual(agent.restored_message_count, 0)
        self.assertEqual(len(agent.messages), 1)
        self.assertEqual(agent.messages[0].role, "system")
        self.assertEqual(agent.messages[0].content, DEFAULT_SYSTEM_PROMPT)

    def test_create_agent_imports_skill_into_system_prompt(self) -> None:
        skill = Skill(
            name="files",
            keywords=("文件",),
            tools=("read_file",),
            instructions="Inspect files before editing.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config,
                resolve_api_key=False,
                restore_session=True,
                skills=[skill],
            )

        self.assertIn(DEFAULT_SYSTEM_PROMPT, agent.system_prompt)
        self.assertIn("Skill: files", agent.system_prompt)
        self.assertIn("Inspect files before editing.", agent.messages[0].content or "")

    def test_runtime_skill_import_updates_restored_system_in_memory(self) -> None:
        skill = Skill(
            name="shell",
            keywords=("shell",),
            tools=("run_shell",),
            instructions="Explain commands before running them.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            store.append_message(Message(role="system", content="saved system"))
            agent = Agent(
                client=mock.MagicMock(),
                tool_registry=mock.MagicMock(),
                permission=mock.MagicMock(),
                config=AgentConfig(),
                system_prompt="default system",
                session_store=store,
            )
            agent.restore_session()
            updated = agent.import_skills_to_system_prompt([skill])

        self.assertIn("saved system", updated)
        self.assertIn("Skill: shell", updated)
        self.assertEqual(agent.messages[0].content, updated)

    def test_create_agent_restores_previous_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_dir = tmp_path / "sessions"
            config = _make_app_config(session_dir, tmp_path)

            first = create_agent(config, resolve_api_key=False, restore_session=False)
            first.messages.append(Message(role="user", content="first question"))
            first._persist_message(first.messages[-1])
            first.messages.append(Message(role="assistant", content="first answer"))
            first._persist_message(first.messages[-1])

            second = create_agent(config, resolve_api_key=False, restore_session=True)

        self.assertEqual(second.restored_message_count, 3)
        roles = [message.role for message in second.messages]
        self.assertEqual(roles, ["system", "user", "assistant"])
        self.assertEqual(second.messages[1].content, "first question")
        self.assertEqual(second.messages[2].content, "first answer")

    def test_create_agent_starts_and_closes_mcp_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            manager = mock.MagicMock()
            manager.start.return_value = {"broken": RuntimeError("offline")}
            manager.close.return_value = {}

            agent = create_agent(
                config,
                resolve_api_key=False,
                restore_session=False,
                mcp_manager=manager,
            )
            agent.close()
            agent.close()

        manager.start.assert_called_once()
        self.assertEqual(agent.state.metadata["mcp_errors"], {"broken": "offline"})
        self.assertEqual(manager.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
