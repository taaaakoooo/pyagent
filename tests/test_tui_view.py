"""Tests for the pure TUI view. These must never need a real TTY."""

from __future__ import annotations

import unittest
from typing import List

from rich.cells import cell_len

from internal.types.types import PermissionRequest
from tui.model import ActiveView, LineRole, McpServerItem, Model, SessionItem, SkillItem
from tui.msgs import (
    AgentDeltaMsg,
    AgentDoneMsg,
    AgentPermissionMsg,
    AgentToolMsg,
    McpChangedMsg,
    SessionChangedMsg,
    SkillsChangedMsg,
)
from tui.update import update
from tui.view import CURSOR_ON, _format_line, render_to_text


def _model(view: ActiveView = ActiveView.CHAT) -> Model:
    return Model(view=view, width=80, height=24, session_id="default")


class ChatViewTests(unittest.TestCase):
    def test_empty_transcript_shows_a_placeholder(self) -> None:
        text = render_to_text(_model(), width=80, height=24)

        self.assertIn("输入内容开始对话", text)

    def test_transcript_lines_are_rendered(self) -> None:
        model = _model()
        model.append_line(LineRole.USER, "hello there")
        model.append_line(LineRole.ASSISTANT, "hi back")

        text = render_to_text(model, width=80, height=24)

        self.assertIn("你 | hello there", text)
        self.assertIn("助手 | hi back", text)

    def test_stream_buffer_is_rendered(self) -> None:
        model, _ = update(_model(), AgentDeltaMsg("partial output"))

        text = render_to_text(model, width=80, height=24)

        self.assertIn("partial output", text)

    def test_header_shows_session_and_status(self) -> None:
        model = _model()
        model.status = "thinking"
        model.turn = 3
        model.max_turns = 10

        text = render_to_text(model, width=100, height=24)

        self.assertIn("会话:default", text)
        # An unknown token is shown verbatim rather than swallowed, so a new
        # status added elsewhere is still visible instead of rendering blank.
        self.assertIn("状态:thinking", text)
        self.assertIn("轮次:3/10", text)

    def test_known_status_tokens_are_localised(self) -> None:
        for token, label in (
            ("idle", "空闲"),
            ("running", "回复中"),
            ("waiting_permission", "等待审批"),
            ("cancelled", "已取消"),
            ("error", "错误"),
        ):
            with self.subTest(token=token):
                model = _model()
                model.status = token

                text = render_to_text(model, width=100, height=24)

                self.assertIn(f"状态:{label}", text)

    def test_long_lines_wrap_without_losing_content(self) -> None:
        model = _model()
        model.append_line(LineRole.USER, "x" * 300)

        text = render_to_text(model, width=60, height=24)

        self.assertIn("x" * 40, text)

    def test_narrow_terminal_does_not_raise(self) -> None:
        model = _model()
        for index in range(20):
            model.append_line(LineRole.USER, f"line {index} " + "y" * 120)

        text = render_to_text(model, width=20, height=8)

        self.assertTrue(text.strip())

    def test_cjk_content_renders(self) -> None:
        model = _model()
        model.append_line(LineRole.USER, "请帮我读取文件内容")

        text = render_to_text(model, width=80, height=24)

        self.assertIn("请帮我读取文件内容", text)


class SkillsViewTests(unittest.TestCase):
    def _model(self) -> Model:
        model, _ = update(
            _model(ActiveView.SKILLS),
            SkillsChangedMsg(
                skills=(
                    {
                        "name": "files",
                        "keywords": ("file", "文件"),
                        "tools": ("read_file",),
                    },
                ),
                enabled=True,
                unmatched_policy="none",
                always_visible=("mcp",),
                errors={"skills/broken.md": "missing keywords"},
            ),
        )
        return model

    def test_skills_table_lists_entries(self) -> None:
        text = render_to_text(self._model(), width=100, height=24)

        self.assertIn("files", text)
        self.assertIn("read_file", text)
        self.assertIn("路由: 开", text)
        self.assertIn("未命中策略: none", text)
        self.assertIn("常驻来源: mcp", text)

    def test_load_errors_are_shown(self) -> None:
        text = render_to_text(self._model(), width=100, height=24)

        self.assertIn("skills/broken.md", text)
        self.assertIn("missing keywords", text)
        self.assertIn("加载失败", text)

    def test_empty_state_renders(self) -> None:
        text = render_to_text(_model(ActiveView.SKILLS), width=100, height=24)

        self.assertIn("尚未加载技能", text)


class MCPViewTests(unittest.TestCase):
    def _model(self) -> Model:
        model, _ = update(
            _model(ActiveView.MCP),
            McpChangedMsg(
                servers=(
                    {
                        "name": "fs",
                        "mode": "stdio",
                        "enabled": True,
                        "started": True,
                        "tools": ["read_file_1", "write_file_1"],
                        "error": None,
                    },
                    {
                        "name": "remote",
                        "mode": "http",
                        "enabled": False,
                        "started": False,
                        "tools": [],
                        "error": None,
                    },
                ),
                errors={"broken": "MCPTimeoutError: too slow"},
            ),
        )
        return model

    def test_server_rows_are_rendered(self) -> None:
        text = render_to_text(self._model(), width=110, height=24)

        self.assertIn("fs", text)
        self.assertIn("stdio", text)
        self.assertIn("已连接", text)
        self.assertIn("remote", text)
        self.assertIn("已停用", text)

    def test_selected_server_tools_are_listed(self) -> None:
        text = render_to_text(self._model(), width=110, height=24)

        self.assertIn("read_file_1", text)
        self.assertIn("write_file_1", text)

    def test_start_errors_are_shown(self) -> None:
        text = render_to_text(self._model(), width=110, height=24)

        self.assertIn("MCPTimeoutError", text)
        self.assertIn("启动失败", text)

    def test_empty_state_renders(self) -> None:
        text = render_to_text(_model(ActiveView.MCP), width=100, height=24)

        self.assertIn("未配置 MCP 服务", text)

    def test_selection_beyond_the_list_does_not_raise(self) -> None:
        model = self._model()
        model.mcp_selected = 99

        text = render_to_text(model, width=100, height=24)

        self.assertIn("fs", text)


class SessionsViewTests(unittest.TestCase):
    def _model(self) -> Model:
        model, _ = update(
            _model(ActiveView.SESSIONS),
            SessionChangedMsg(
                session_id="default",
                sessions=(
                    {
                        "session_id": "default",
                        "message_count": 4,
                        "updated_at": "2026-09-21 19:00",
                        "preview": "hello",
                    },
                    {
                        "session_id": "work",
                        "message_count": 0,
                        "updated_at": "2026-09-20 08:30",
                        "preview": "",
                    },
                ),
                archive_dir=".tool_outputs/default",
            ),
        )
        return model

    def test_sessions_are_rendered_with_the_current_one_marked(self) -> None:
        text = render_to_text(self._model(), width=110, height=24)

        self.assertIn("default", text)
        self.assertIn("work", text)
        self.assertIn("*", text)
        self.assertIn("hello", text)
        self.assertIn("（空）", text)

    def test_archive_dir_is_shown(self) -> None:
        text = render_to_text(self._model(), width=110, height=24)

        self.assertIn(".tool_outputs/default", text)
        self.assertIn("归档目录", text)

    def test_empty_state_renders(self) -> None:
        text = render_to_text(_model(ActiveView.SESSIONS), width=100, height=24)

        self.assertIn("暂无会话", text)


class HelpViewTests(unittest.TestCase):
    def test_help_documents_the_permission_choices(self) -> None:
        text = render_to_text(_model(ActiveView.HELP), width=100, height=24)

        self.assertIn("本次允许", text)
        self.assertIn("本次拒绝", text)
        self.assertIn("本会话允许", text)
        self.assertIn("本会话拒绝", text)
        self.assertIn("Shift+Q", text)

    def test_help_is_chinese(self) -> None:
        """The cheat sheet is user-facing prose, so it is localised."""
        text = render_to_text(_model(ActiveView.HELP), width=100, height=24)

        for fragment in ("对话", "命令层", "面板", "退出", "权限"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_help_fits_a_short_terminal(self) -> None:
        """The permission list must survive the 3-row composer.

        A panel taller than the body is silently cropped, which used to hide
        the permission choices entirely when a line was added.
        """
        text = render_to_text(_model(ActiveView.HELP), width=100, height=24)

        self.assertIn("本会话拒绝", text)


class PermissionOverlayTests(unittest.TestCase):
    def _request(self) -> PermissionRequest:
        return PermissionRequest(
            request_id="r1",
            tool_name="run_shell",
            arguments={"command": "rm -rf build"},
            risk_level="high",
            risk_tags=["destructive", "shell"],
            risk_description="Deletes files.",
            preview_diff={"unified": "--- a\n+++ b\n-old\n+new"},
            tool_source="local",
        )

    def test_overlay_shows_tool_risk_and_diff(self) -> None:
        model, _ = update(_model(), AgentPermissionMsg(self._request()))

        text = render_to_text(model, width=100, height=30)

        self.assertIn("需要审批", text)
        self.assertIn("run_shell", text)
        self.assertIn("destructive", text)
        self.assertIn("Deletes files.", text)
        self.assertIn("rm -rf build", text)
        self.assertIn("+new", text)

    def test_overlay_lists_all_four_choices(self) -> None:
        model, _ = update(_model(), AgentPermissionMsg(self._request()))

        text = render_to_text(model, width=100, height=30)

        self.assertIn("本次允许", text)
        self.assertIn("本会话拒绝", text)

    def test_overlay_renders_without_a_diff(self) -> None:
        request = PermissionRequest(
            request_id="r2",
            tool_name="read_file",
            risk_level="low",
        )
        model, _ = update(_model(), AgentPermissionMsg(request))

        text = render_to_text(model, width=80, height=24)

        self.assertIn("read_file", text)

    def test_overlay_renders_mcp_server_name(self) -> None:
        request = PermissionRequest(
            request_id="r3",
            tool_name="read_file_1",
            tool_source="mcp",
            server_name="fs",
            risk_level="medium",
        )
        model, _ = update(_model(), AgentPermissionMsg(request))

        text = render_to_text(model, width=80, height=24)

        self.assertIn("服务: fs", text)


class ToolLineTests(unittest.TestCase):
    def test_tool_lines_are_rendered(self) -> None:
        model = _model()
        model, _ = update(
            model,
            AgentToolMsg("write_file", "completed", {"content": "wrote 3 lines"}),
        )

        text = render_to_text(model, width=90, height=24)

        self.assertIn("write_file", text)
        self.assertIn("完成", text)
        self.assertIn("工具 |", text)

    def test_tool_status_tokens_are_localised(self) -> None:
        for token, label in (
            ("running", "执行中"),
            ("completed", "完成"),
            ("error", "失败"),
            ("skipped", "已跳过"),
            ("warning", "警告"),
        ):
            with self.subTest(token=token):
                model, _ = update(
                    _model(), AgentToolMsg("read_file", token, {"path": "a"})
                )

                text = render_to_text(model, width=90, height=24)

                self.assertIn(label, text)


class ModelRenderingContractTests(unittest.TestCase):
    """The view must survive any model the update function can produce."""

    def test_every_view_renders_for_a_default_model(self) -> None:
        for view in ActiveView:
            with self.subTest(view=view):
                text = render_to_text(_model(view), width=60, height=18)
                self.assertTrue(text.strip())

    def test_rendering_is_deterministic(self) -> None:
        model = _model()
        model.append_line(LineRole.USER, "deterministic")

        first = render_to_text(model, width=80, height=24)
        second = render_to_text(model, width=80, height=24)

        self.assertEqual(first, second)


def _composer_rows(model: Model, width: int = 80, height: int = 24) -> List[str]:
    """The composer is the last three rows: border, input line, border."""
    rows = render_to_text(model, width=width, height=height).splitlines()
    return rows[-3:]


class FormatLineCacheTests(unittest.TestCase):
    """Frame time depends on these cache hits, not just on correctness."""

    def setUp(self) -> None:
        _format_line.cache_clear()

    def test_repainting_an_unchanged_transcript_hits_the_cache(self) -> None:
        model = _model()
        for index in range(5):
            model.append_line(LineRole.ASSISTANT, f"line {index}")

        render_to_text(model, width=80, height=24)
        before = _format_line.cache_info()
        render_to_text(model, width=80, height=24)
        after = _format_line.cache_info()

        self.assertGreater(after.hits, before.hits)
        self.assertEqual(after.misses, before.misses)

    def test_a_resize_reformats_at_the_new_width(self) -> None:
        model = _model()
        model.append_line(LineRole.ASSISTANT, "same text")

        render_to_text(model, width=80, height=24)
        misses = _format_line.cache_info().misses

        model.width = 60
        render_to_text(model, width=60, height=24)

        self.assertGreater(_format_line.cache_info().misses, misses)

    def test_cached_text_is_never_mutated_by_a_later_frame(self) -> None:
        # The cache hands the same Text object to every frame; an in-place
        # append anywhere would make rows grow without limit.
        model = _model()
        model.append_line(LineRole.ASSISTANT, "stable across frames")

        first = render_to_text(model, width=80, height=24)
        for _ in range(3):
            render_to_text(model, width=80, height=24)

        self.assertEqual(render_to_text(model, width=80, height=24), first)

    def test_streaming_growth_does_not_corrupt_flushed_lines(self) -> None:
        model = _model()
        model.append_line(LineRole.USER, "earlier turn")
        render_to_text(model, width=80, height=24)

        model.stream_buffer = "partial"
        with_stream = render_to_text(model, width=80, height=24)
        model.stream_buffer = ""
        without = render_to_text(model, width=80, height=24)

        self.assertIn("earlier turn", with_stream)
        self.assertIn("earlier turn", without)


class ComposerTests(unittest.TestCase):
    """The composer is the only cue telling the user where typing lands."""

    def test_focused_composer_announces_itself_and_shows_a_caret(self) -> None:
        model = _model()

        top, content, bottom = _composer_rows(model)

        self.assertIn("> 输入", top)
        self.assertIn(CURSOR_ON, content)
        self.assertIn("Enter 发送", bottom)

    def test_caret_vanishes_on_the_off_phase_without_shifting_layout(self) -> None:
        on_model = _model()
        off_model = _model()
        off_model.cursor_on = False

        on_rows = _composer_rows(on_model)
        off_rows = _composer_rows(off_model)

        self.assertIn(CURSOR_ON, on_rows[1])
        self.assertNotIn(CURSOR_ON, off_rows[1])
        self.assertEqual(len(on_rows[1]), len(off_rows[1]))

    def test_running_composer_locks_the_input(self) -> None:
        model = _model()
        model.is_running = True
        model.input_buffer = "typed"

        top, content, bottom = _composer_rows(model)

        self.assertIn("输入已锁定", top)
        self.assertIn("Esc", bottom)
        self.assertNotIn(CURSOR_ON, content)
        self.assertIn("typed", content)

    def test_command_layer_composer_hides_the_prompt_caret(self) -> None:
        model = _model()
        model.command_mode = True

        top, content, bottom = _composer_rows(model)

        self.assertIn("命令层", top)
        self.assertNotIn(CURSOR_ON, content)
        self.assertIn("Shift+Q", bottom)

    def test_command_layer_still_shows_what_is_typed(self) -> None:
        model = _model()
        model.command_mode = True
        model.input_buffer = "kept"

        _top, content, _bottom = _composer_rows(model)

        self.assertIn("kept", content)
        self.assertNotIn(CURSOR_ON, content)

    def test_blurred_composer_points_back_at_the_chat(self) -> None:
        model = _model(ActiveView.SKILLS)
        model.input_buffer = "draft"

        top, content, _bottom = _composer_rows(model)

        self.assertIn("Tab", top)
        self.assertNotIn(CURSOR_ON, content)
        self.assertIn("draft", content)

    def test_long_input_keeps_the_caret_and_the_tail_visible(self) -> None:
        model = _model()
        model.input_buffer = "HEAD" + ("x" * 400) + "TAIL"

        _top, content, _bottom = _composer_rows(model)

        self.assertIn("TAIL", content)
        self.assertIn(CURSOR_ON, content)
        self.assertNotIn("HEAD", content)

    def test_composer_stays_three_rows_on_narrow_terminals(self) -> None:
        for width in (20, 30, 40, 60):
            with self.subTest(width=width):
                model = _model()
                model.input_buffer = "中文输入" * 30

                rows = _composer_rows(model, width=width, height=18)

                self.assertEqual(len(rows), 3)
                for row in rows:
                    self.assertLessEqual(cell_len(row), width)

    def test_composer_is_what_brings_the_frame_to_full_height(self) -> None:
        for height in (18, 24, 30):
            with self.subTest(height=height):
                text = render_to_text(_model(), width=80, height=height)

                self.assertEqual(len(text.splitlines()), height)

    def test_notice_replaces_the_hint_while_focused(self) -> None:
        model = _model()
        model.notice = "saved skills"

        _top, _content, bottom = _composer_rows(model)

        self.assertIn("saved skills", bottom)
        self.assertNotIn("Enter 发送", bottom)


if __name__ == "__main__":
    unittest.main()
