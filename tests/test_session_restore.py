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
from internal.session.session import SessionError, SessionStore
from internal.skills import Skill
from internal.types.types import (
    FunctionCall,
    Message,
    PermissionDecision,
    PermissionRequest,
    ToolCall,
)


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
            # The session already had history, so the injected prompt is kept in
            # memory only to avoid appending a duplicate system record.
            self.assertEqual(_count_jsonl_messages(store.session_file, "system"), 0)

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

    def test_create_agent_repeated_without_restore_keeps_single_system(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_dir = tmp_path / "sessions"
            config = _make_app_config(session_dir, tmp_path)

            for _ in range(3):
                create_agent(config, resolve_api_key=False, restore_session=False)

            system_count = _count_jsonl_messages(
                session_dir / "default.jsonl", "system"
            )

        self.assertEqual(system_count, 1)


class SessionIsolationTests(unittest.TestCase):
    def test_create_agent_uses_named_session_and_archive_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_dir = tmp_path / "sessions"
            config = _make_app_config(session_dir, tmp_path)

            agent = create_agent(
                config,
                resolve_api_key=False,
                restore_session=True,
                session_id="work",
            )

            self.assertEqual(agent.session_id, "work")
            self.assertEqual(agent.session_store.session_file, session_dir / "work.jsonl")
            self.assertEqual(agent.tool_output_path, tmp_path / ".tool_outputs" / "work")
            self.assertTrue(agent.tool_output_path.is_dir())

    def test_explicit_tool_output_path_skips_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            explicit = tmp_path / "custom_outputs"
            config = _make_app_config(tmp_path / "sessions", tmp_path)

            agent = create_agent(
                config,
                resolve_api_key=False,
                restore_session=True,
                session_id="work",
                tool_output_path=explicit,
            )

            self.assertEqual(agent.tool_output_path, explicit)

    def test_two_sessions_do_not_share_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_dir = tmp_path / "sessions"
            config = _make_app_config(session_dir, tmp_path)

            alpha = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )
            alpha.messages.append(Message(role="user", content="alpha question"))
            alpha._persist_message(alpha.messages[-1])

            beta = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="beta"
            )
            beta.messages.append(Message(role="user", content="beta question"))
            beta._persist_message(beta.messages[-1])

            alpha_reloaded = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )

        contents = [message.content for message in alpha_reloaded.messages]
        self.assertIn("alpha question", contents)
        self.assertNotIn("beta question", contents)

    def test_switch_session_restores_target_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_dir = tmp_path / "sessions"
            config = _make_app_config(session_dir, tmp_path)

            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )
            agent.messages.append(Message(role="user", content="alpha question"))
            agent._persist_message(agent.messages[-1])

            switched_back = agent.switch_session("beta")
            self.assertFalse(switched_back)
            self.assertEqual(agent.session_id, "beta")
            self.assertEqual(agent.restored_message_count, 0)
            self.assertEqual(len(agent.messages), 1)
            self.assertEqual(agent.messages[0].role, "system")
            self.assertEqual(
                agent.tool_output_path, tmp_path / ".tool_outputs" / "beta"
            )
            self.assertTrue(agent.tool_output_path.is_dir())
            self.assertEqual(_count_jsonl_messages(agent.session_store.session_file, "system"), 1)

            switched_back = agent.switch_session("alpha")
            self.assertTrue(switched_back)
            self.assertEqual(agent.session_id, "alpha")
            self.assertEqual(agent.restored_message_count, 2)
            self.assertEqual(agent.messages[1].content, "alpha question")
            self.assertEqual(
                agent.tool_output_path, tmp_path / ".tool_outputs" / "alpha"
            )

    def test_switch_session_rebinds_tool_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )

            agent.switch_session("beta")

            self.assertEqual(
                agent.tool_registry.tool_context.tool_output_path,
                agent.tool_output_path.resolve(),
            )

    def test_switch_session_to_same_session_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            first = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )
            first.messages.append(Message(role="user", content="kept"))
            first._persist_message(first.messages[-1])

            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )
            restored = agent.switch_session("alpha")

            self.assertTrue(restored)
            self.assertEqual(len(agent.messages), 2)
            self.assertEqual(agent.messages[1].content, "kept")

    def test_switch_session_to_same_empty_session_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )

            self.assertFalse(agent.switch_session("alpha"))

    def test_switch_session_requires_storage_dir(self) -> None:
        agent = Agent(
            client=mock.MagicMock(),
            tool_registry=mock.MagicMock(),
            permission=mock.MagicMock(),
            config=AgentConfig(),
            system_prompt="system prompt",
        )
        with self.assertRaises(SessionError):
            agent.switch_session("work")


class SessionPermissionIsolationTests(unittest.TestCase):
    """Conversation-scoped grants must not leak into another session.

    A grant is stored keyed by tool name alone, so without an explicit reset an
    "always allow run_shell" answered in one session would silently authorize the
    same tool in every later session.
    """

    def _allow_run_shell_forever(self, agent) -> None:
        request = PermissionRequest(
            request_id="p1",
            tool_name="run_shell",
            arguments={"command": "echo hi"},
        )
        approved = agent.permission.resolve(
            request,
            callback=lambda _request: PermissionDecision.ALLOW_SESSION,
            workspace_root=Path("."),
        )
        self.assertTrue(approved)

    def _resolve_run_shell(self, agent) -> bool:
        return agent.permission.resolve(
            PermissionRequest(
                request_id="p2",
                tool_name="run_shell",
                arguments={"command": "echo hi"},
            ),
            callback=None,
            workspace_root=Path("."),
        )

    def test_switch_session_clears_session_grants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )

            self._allow_run_shell_forever(agent)
            # Sanity check: the grant is live within the same session.
            self.assertTrue(self._resolve_run_shell(agent))

            agent.switch_session("beta")

            # Without a callback a mutating tool must be denied again.
            self.assertFalse(self._resolve_run_shell(agent))

    def test_switch_session_clears_session_denials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )

            agent.permission.resolve(
                PermissionRequest(
                    request_id="p1",
                    tool_name="run_shell",
                    arguments={"command": "echo hi"},
                ),
                callback=lambda _request: PermissionDecision.DENY_SESSION,
                workspace_root=Path("."),
            )
            self.assertFalse(self._resolve_run_shell(agent))

            agent.switch_session("beta")

            # The denial is gone, so the callback decides again.
            self.assertTrue(
                agent.permission.resolve(
                    PermissionRequest(
                        request_id="p3",
                        tool_name="run_shell",
                        arguments={"command": "echo hi"},
                    ),
                    callback=lambda _request: PermissionDecision.ALLOW_ONCE,
                    workspace_root=Path("."),
                )
            )

    def test_reselecting_the_same_session_keeps_grants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )

            self._allow_run_shell_forever(agent)
            agent.switch_session("alpha")

            self.assertTrue(self._resolve_run_shell(agent))

    def test_create_session_clears_session_grants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )

            self._allow_run_shell_forever(agent)
            agent.create_session("beta")

            self.assertFalse(self._resolve_run_shell(agent))

    def test_approver_without_reset_session_is_supported(self) -> None:
        """A minimal custom approver must not break session switching."""

        class MinimalApprover:
            def resolve(self, permission, *, callback, tool_definition, workspace_root):
                return True

            def request(self, permission):
                return True

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config,
                resolve_api_key=False,
                restore_session=True,
                session_id="alpha",
                permission=MinimalApprover(),
            )

            # Must not raise AttributeError.
            agent.switch_session("beta")

            self.assertEqual(agent.session_id, "beta")


class SessionLifecycleTests(unittest.TestCase):
    def test_create_session_autogenerates_unique_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(config, resolve_api_key=False, restore_session=True)

            first = agent.create_session()
            second = agent.create_session()

            self.assertTrue(first.startswith("session-"))
            self.assertNotEqual(first, second)
            self.assertEqual(agent.session_id, second)

    def test_create_session_appends_suffix_on_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(config, resolve_api_key=False, restore_session=True)

            self.assertEqual(agent.create_session("work"), "work")
            self.assertEqual(agent.create_session("work"), "work-2")
            self.assertEqual(agent.create_session("work"), "work-3")

    def test_create_session_starts_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )
            agent.messages.append(Message(role="user", content="alpha question"))
            agent._persist_message(agent.messages[-1])

            agent.create_session("beta")

            self.assertEqual(agent.restored_message_count, 0)
            self.assertEqual(len(agent.messages), 1)
            self.assertEqual(agent.messages[0].role, "system")

    def test_list_sessions_reports_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(config, resolve_api_key=False, restore_session=True)

            agent.create_session("alpha")
            agent.create_session("beta")

            session_ids = {summary.session_id for summary in agent.list_sessions()}

        self.assertEqual(session_ids, {"default", "alpha", "beta"})

    def test_delete_session_removes_other_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_dir = tmp_path / "sessions"
            config = _make_app_config(session_dir, tmp_path)
            agent = create_agent(config, resolve_api_key=False, restore_session=True)
            agent.create_session("alpha")
            agent.create_session("beta")

            self.assertTrue(agent.delete_session("alpha"))
            self.assertFalse(agent.delete_session("alpha"))
            self.assertFalse((session_dir / "alpha.jsonl").exists())

    def test_delete_session_refuses_active_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(
                config, resolve_api_key=False, restore_session=True, session_id="alpha"
            )

            with self.assertRaises(SessionError):
                agent.delete_session("alpha")

    def test_delete_session_rejects_invalid_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = _make_app_config(tmp_path / "sessions", tmp_path)
            agent = create_agent(config, resolve_api_key=False, restore_session=True)

            with self.assertRaises(SessionError):
                agent.delete_session("../escape")


if __name__ == "__main__":
    unittest.main()
