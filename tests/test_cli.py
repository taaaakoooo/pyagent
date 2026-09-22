"""The line-based CLI's user-facing output.

Written as a golden-output check rather than a behaviour test: the value here
is that a future edit cannot quietly reintroduce an English string, and that
every ``/`` command keeps printing something a user can act on.
"""

from __future__ import annotations

import io
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List
from unittest import mock

import main
from config.config import ConfigError
from internal.session.session import SessionError, SessionSummary

# Words that only ever appeared in the old, untranslated output. If one
# resurfaces it means a string was missed or reverted.
ENGLISH_MARKERS = (
    "No skills loaded",
    "Loaded ",
    "failed to load",
    "Usage:",
    "No MCP servers",
    "MCP server(s)",
    "server(s)",
    "Unknown",
    "No sessions yet",
    "Session(s)",
    "messages=",
    "updated=",
    "Cannot delete",
    "not found",
    "Commands:",
    "Startup error",
    "PyAgent ready",
    "Bye.",
    "You: ",
)


class FakeAgent:
    """Minimal stand-in exposing exactly what the CLI handlers touch."""

    def __init__(self) -> None:
        self.session_id = "default"
        self.messages: List[object] = []
        self.restored_message_count = 0
        self.tool_output_path = ".tool_outputs/default"
        self.skill_unmatched_policy = "readonly"
        self.always_visible_sources = ("mcp",)
        self._skills_enabled = True
        self.state = mock.MagicMock()
        self.state.metadata: dict = {}

    # -- skills ---------------------------------------------------------

    def describe_skills(self) -> List[dict]:
        return [{"name": "files", "keywords": ["file"], "tools": ["read_file"]}]

    def reload_skills(self) -> Any:
        result = mock.MagicMock()
        result.skills = [object(), object()]
        result.errors = {"skills/broken.md": "missing keywords"}
        return result

    def skills_enabled(self) -> bool:
        return self._skills_enabled

    def set_skills_enabled(self, enabled: bool) -> None:
        self._skills_enabled = enabled

    # -- mcp ------------------------------------------------------------

    def describe_mcp(self) -> List[dict]:
        return [
            {
                "name": "fs",
                "mode": "stdio",
                "enabled": True,
                "started": True,
                "tools": ["read_file_1"],
            },
            {
                "name": "remote",
                "mode": "http",
                "enabled": False,
                "started": False,
                "tools": [],
            },
        ]

    def reload_mcp(self) -> dict:
        return {}

    def set_mcp_server_enabled(self, name: str, enabled: bool) -> bool:
        return name == "fs"

    # -- sessions -------------------------------------------------------

    def list_sessions(self) -> List[SessionSummary]:
        return [
            SessionSummary(
                session_id="default",
                path=Path(".sessions/default.jsonl"),
                message_count=4,
                updated_at=datetime(2026, 9, 21, 19, 0, tzinfo=timezone.utc),
                preview="hello",
            ),
            SessionSummary(
                session_id="work",
                path=Path(".sessions/work.jsonl"),
                message_count=0,
                updated_at=datetime(2026, 9, 20, 8, 30, tzinfo=timezone.utc),
                preview="",
            ),
        ]

    def create_session(self, name: Any = None) -> str:
        return name or "s1"

    def switch_session(self, name: str) -> bool:
        self.session_id = name
        self.restored_message_count = 3
        return True

    def delete_session(self, name: str) -> bool:
        return name == "default"

    def chat(self, prompt: str) -> str:
        return "answer"

    def close(self) -> None:
        pass


def capture(function: Any, *args: Any) -> str:
    """Run ``function`` and return everything it printed."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        function(*args)
    return buffer.getvalue()


def assert_no_english(test: unittest.TestCase, text: str) -> None:
    for marker in ENGLISH_MARKERS:
        with test.subTest(marker=marker):
            test.assertNotIn(marker, text)


class CliIsChineseTests(unittest.TestCase):
    """Every printer must emit Chinese; none may fall back to English."""

    def test_skill_list(self) -> None:
        text = capture(main._print_skill_list, FakeAgent())

        self.assertIn("已加载 1 个技能", text)
        self.assertIn("关键词=[file]", text)
        self.assertIn("工具=[read_file]", text)
        assert_no_english(self, text)

    def test_skill_list_empty(self) -> None:
        agent = FakeAgent()
        agent.describe_skills = lambda: []  # type: ignore[method-assign]

        text = capture(main._print_skill_list, agent)

        self.assertIn("尚未加载技能", text)
        assert_no_english(self, text)

    def test_skill_errors(self) -> None:
        agent = FakeAgent()
        agent.state.metadata = {"skill_errors": {"a.md": "boom"}}

        text = capture(main._print_skill_errors, agent)

        self.assertIn("加载失败", text)
        self.assertIn("a.md", text)
        assert_no_english(self, text)

    def test_mcp_errors(self) -> None:
        agent = FakeAgent()
        agent.state.metadata = {"mcp_errors": {"fs": "timeout"}}

        text = capture(main._print_mcp_errors, agent)

        self.assertIn("启动失败", text)
        self.assertIn("fs", text)
        assert_no_english(self, text)

    def test_help(self) -> None:
        text = capture(main._print_help)

        self.assertIn("可用命令", text)
        for command in ("/skills", "/mcp", "/session", "/help", "exit"):
            self.assertIn(command, text)
        assert_no_english(self, text)


class SkillsCommandTests(unittest.TestCase):
    def test_status_reports_the_routing_state(self) -> None:
        agent = FakeAgent()
        agent.state.metadata = {
            "active_skills": ["files"],
            "visible_tools": ["read_file"],
        }

        text = capture(main._handle_skills_command, agent, "status")

        self.assertIn("技能开关: 开", text)
        self.assertIn("未命中策略: readonly", text)
        self.assertIn("常驻来源: mcp", text)
        self.assertIn("本轮命中的技能: ['files']", text)
        assert_no_english(self, text)

    def test_status_says_when_nothing_is_active(self) -> None:
        text = capture(main._handle_skills_command, FakeAgent(), "status")

        self.assertIn("（无）", text)

    def test_off_disables_and_says_which_tools_remain(self) -> None:
        agent = FakeAgent()

        text = capture(main._handle_skills_command, agent, "off")

        self.assertFalse(agent.skills_enabled())
        self.assertIn("技能已关闭", text)

    def test_reload_reports_the_count_and_any_errors(self) -> None:
        text = capture(main._handle_skills_command, FakeAgent(), "reload")

        self.assertIn("已重新加载 2 个技能", text)
        self.assertIn("skills/broken.md", text)

    def test_unknown_action_prints_usage(self) -> None:
        text = capture(main._handle_skills_command, FakeAgent(), "wat")

        self.assertIn("用法:", text)

    def test_every_known_action_is_handled(self) -> None:
        for action in ("", "list", "reload", "on", "off", "status"):
            with self.subTest(action=action):
                text = capture(main._handle_skills_command, FakeAgent(), action)

                self.assertNotIn("用法:", text)


class McpCommandTests(unittest.TestCase):
    def test_list_localises_the_state_column(self) -> None:
        text = capture(main._handle_mcp_command, FakeAgent(), "")

        self.assertIn("2 个 MCP 服务", text)
        self.assertIn("已连接", text)
        self.assertIn("已停用", text)
        self.assertIn("工具=1", text)
        assert_no_english(self, text)

    def test_enable_and_disable_are_distinguishable(self) -> None:
        enabled = capture(main._handle_mcp_command, FakeAgent(), "enable fs")
        disabled = capture(main._handle_mcp_command, FakeAgent(), "disable fs")

        self.assertIn("已启用", enabled)
        self.assertIn("已停用", disabled)

    def test_names_are_still_echoed_verbatim(self) -> None:
        """Server ids are identifiers, not prose -- do not translate them."""
        text = capture(main._handle_mcp_command, FakeAgent(), "tools fs")

        self.assertIn("fs:", text)
        self.assertIn("read_file_1", text)

    def test_missing_name_prints_usage(self) -> None:
        text = capture(main._handle_mcp_command, FakeAgent(), "enable")

        self.assertIn("用法:", text)

    def test_unknown_server_is_reported(self) -> None:
        text = capture(main._handle_mcp_command, FakeAgent(), "enable nope")

        self.assertIn("未知的 MCP 服务", text)

    def test_every_known_action_is_handled(self) -> None:
        for action in ("", "list", "reload", "enable fs", "disable fs", "tools"):
            with self.subTest(action=action):
                text = capture(main._handle_mcp_command, FakeAgent(), action)

                self.assertNotIn("用法:", text)


class SessionCommandTests(unittest.TestCase):
    def test_list_marks_the_current_session(self) -> None:
        text = capture(main._handle_session_command, FakeAgent(), "")

        self.assertIn("2 个会话", text)
        self.assertIn("*", text)
        self.assertIn("消息=4", text)
        self.assertIn("（空）", text)
        assert_no_english(self, text)

    def test_current_reports_the_archive_dir(self) -> None:
        text = capture(main._handle_session_command, FakeAgent(), "current")

        self.assertIn("会话: default", text)
        self.assertIn("归档目录: .tool_outputs/default", text)

    def test_use_reports_how_much_was_restored(self) -> None:
        text = capture(main._handle_session_command, FakeAgent(), "use work")

        self.assertIn("已切换到会话 'work'", text)
        self.assertIn("恢复了 3 条消息", text)

    def test_delete_distinguishes_found_from_missing(self) -> None:
        deleted = capture(
            main._handle_session_command, FakeAgent(), "delete default"
        )
        missing = capture(main._handle_session_command, FakeAgent(), "delete nope")

        self.assertIn("已删除会话", deleted)
        self.assertIn("未找到会话", missing)

    def test_session_errors_are_surfaced_not_raised(self) -> None:
        agent = FakeAgent()

        def _boom(_name: str) -> bool:
            raise SessionError("in use")

        agent.delete_session = _boom  # type: ignore[method-assign]

        text = capture(main._handle_session_command, agent, "delete default")

        self.assertIn("无法删除", text)
        self.assertIn("in use", text)

    def test_missing_name_prints_usage(self) -> None:
        for action in ("use", "delete"):
            with self.subTest(action=action):
                text = capture(main._handle_session_command, FakeAgent(), action)

                self.assertIn("用法:", text)

    def test_every_known_action_is_handled(self) -> None:
        for action in ("", "list", "new x", "use work", "current", "delete default"):
            with self.subTest(action=action):
                text = capture(main._handle_session_command, FakeAgent(), action)

                self.assertNotIn("用法:", text)


class DispatchTests(unittest.TestCase):
    """The interactive loop itself: prompt, quit words, unknown commands."""

    def _run(
        self, inputs: List[Any], agent: Any = None, create_side_effect: Any = None
    ) -> str:
        agent = agent if agent is not None else FakeAgent()
        buffer = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(main, "load_config", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(
                    main,
                    "create_agent",
                    side_effect=create_side_effect,
                    return_value=None if create_side_effect else agent,
                )
            )
            stack.enter_context(
                mock.patch("builtins.input", side_effect=inputs + [EOFError])
            )
            stack.enter_context(redirect_stdout(buffer))
            code = main.run_cli([])

        self.assertEqual(code, 0)
        return buffer.getvalue()

    def test_startup_and_farewell_are_in_chinese(self) -> None:
        text = self._run([""])

        self.assertIn("已就绪", text)
        self.assertIn("再见", text)
        assert_no_english(self, text)

    def test_exit_words_leave_cleanly(self) -> None:
        for word in ("exit", "quit", "EXIT"):
            with self.subTest(word=word):
                text = self._run([word])

                self.assertIn("再见", text)

    def test_an_unknown_command_suggests_help(self) -> None:
        text = self._run(["/nope"])

        self.assertIn("未知命令: /nope", text)
        self.assertIn("/help", text)

    def test_help_is_reachable_from_the_loop(self) -> None:
        text = self._run(["/help"])

        self.assertIn("可用命令", text)

    def test_a_normal_prompt_goes_to_the_agent(self) -> None:
        agent = FakeAgent()
        calls: List[str] = []
        agent.chat = lambda prompt: calls.append(prompt) or "answer"  # type: ignore[method-assign]

        self._run(["hi"], agent=agent)

        self.assertEqual(calls, ["hi"])

    def test_startup_failure_is_reported_in_chinese(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(main, "load_config", return_value=object())
            )
            stack.enter_context(
                mock.patch.object(
                    main, "create_agent", side_effect=ConfigError("bad yaml")
                )
            )
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(err))
            code = main.run_cli([])

        self.assertEqual(code, 2)
        self.assertIn("启动失败", err.getvalue())
        self.assertIn("bad yaml", err.getvalue())


if __name__ == "__main__":
    unittest.main()
