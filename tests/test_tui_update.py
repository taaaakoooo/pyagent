"""Tests for the pure TUI update function."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Tuple

from internal.types.types import PermissionDecision, PermissionRequest
from tui.model import ActiveView, BLINK_TICKS, LineRole, Model
from tui.msgs import (
    KEY_BACKTAB,
    KEY_DOWN,
    KEY_ENTER,
    KEY_ESC,
    KEY_LEFT,
    KEY_PAGEDOWN,
    KEY_PAGEUP,
    KEY_RIGHT,
    KEY_SCROLL_DOWN,
    KEY_SCROLL_UP,
    KEY_SPACE,
    KEY_TAB,
    KEY_UP,
    KEY_SHIFT_C,
    KEY_SHIFT_Q,
    AgentDeltaMsg,
    AgentDoneMsg,
    AgentErrorMsg,
    AgentPermissionMsg,
    AgentStartedMsg,
    AgentToolMsg,
    InterruptMsg,
    KeyMsg,
    McpChangedMsg,
    NoticeMsg,
    ResizeMsg,
    SessionChangedMsg,
    SkillsChangedMsg,
    TextMsg,
    TickMsg,
)
from tui.update import (
    WHEEL_SCROLL_LINES,
    CmdContext,
    permission_options,
    update,
)



class RecordingContext(CmdContext):
    """CmdContext that records which hooks fired."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: List[Tuple[str, Any]] = []

        self.run_agent = self._recorder("run_agent")
        self.resolve_permission = self._recorder("resolve_permission")
        self.cancel_agent = self._recorder("cancel_agent")
        self.save_skills = self._recorder("save_skills")
        self.reload_skills = self._recorder("reload_skills")
        self.reload_mcp = self._recorder("reload_mcp")
        self.set_mcp_enabled = self._recorder("set_mcp_enabled")
        self.create_session = self._recorder("create_session")
        self.switch_session = self._recorder("switch_session")
        self.delete_session = self._recorder("delete_session")
        self.refresh_view = self._recorder("refresh_view")

    def _recorder(self, name: str):
        def _record(*args: Any) -> None:
            self.calls.append((name, args if len(args) != 1 else args[0]))

        return _record

    def names(self) -> List[str]:
        return [name for name, _ in self.calls]


def _type(model: Model, text: str) -> Tuple[Model, Optional[Any]]:
    cmd = None
    for char in text:
        model, cmd = update(model, KeyMsg("char", char))
    return model, cmd


def _press(model: Model, key: str) -> Tuple[Model, Optional[Any]]:
    return update(model, KeyMsg(key))


def _run(model: Model, msg) -> Tuple[Model, Optional[Any], List[Tuple[str, Any]]]:
    model, cmd = update(model, msg)
    context = RecordingContext()
    if cmd is not None:
        cmd(context)
    return model, cmd, context.calls


class InputTests(unittest.TestCase):
    def test_typing_accumulates_in_the_input_buffer(self) -> None:
        model, _ = _type(Model(), "hello")

        self.assertEqual(model.input_buffer, "hello")

    def test_enter_appends_a_user_line_and_runs_the_agent(self) -> None:
        model, _ = _type(Model(), "hi")
        model, cmd, calls = _run(model, KeyMsg(KEY_ENTER))

        self.assertEqual(model.input_buffer, "")
        self.assertEqual(len(model.transcript), 1)
        self.assertEqual(model.transcript[0].role, LineRole.USER)
        self.assertEqual(model.transcript[0].text, "hi")
        self.assertTrue(model.is_running)
        self.assertEqual(calls, [("run_agent", "hi")])

    def test_enter_on_empty_input_does_nothing(self) -> None:
        model, cmd = _press(Model(), KEY_ENTER)

        self.assertIsNone(cmd)
        self.assertEqual(model.transcript, [])

    def test_enter_while_running_is_ignored(self) -> None:
        model, _ = _type(Model(is_running=True), "hi")
        model, cmd, calls = _run(model, KeyMsg(KEY_ENTER))

        self.assertEqual(calls, [])
        self.assertIn("仍在回复", model.notice)

    def test_text_msg_types_a_whole_string(self) -> None:
        model, _ = update(Model(), TextMsg("abc"))

        self.assertEqual(model.input_buffer, "abc")

    def test_text_msg_with_newline_submits(self) -> None:
        model, cmd, calls = _run(Model(), TextMsg("abc\n"))

        self.assertEqual(model.transcript[-1].text, "abc")
        self.assertEqual(calls, [("run_agent", "abc")])

    def test_history_recalls_previous_entries(self) -> None:
        model = Model()
        model, _ = update(model, TextMsg("first\n"))
        model, _ = update(model, AgentDoneMsg(answer="ok"))
        model, _ = update(model, TextMsg("second\n"))
        model, _ = update(model, AgentDoneMsg(answer="ok"))

        # History is reached once there is a prompt in progress: a bare Up on
        # an empty composer scrolls the transcript instead, because that is
        # what a wheel scroll looks like on terminals without mouse reporting.
        model, _ = _type(model, "x")
        model, _ = _press(model, KEY_UP)
        self.assertEqual(model.input_buffer, "second")
        model, _ = _press(model, KEY_UP)
        self.assertEqual(model.input_buffer, "first")
        model, _ = _press(model, KEY_DOWN)
        self.assertEqual(model.input_buffer, "second")

    def test_arrows_scroll_the_transcript_on_an_empty_composer(self) -> None:
        """Wheel scrolling arrives as Up/Down here, so it must scroll."""
        model = Model()
        for index in range(40):
            model.append_line(LineRole.USER, f"line {index}")
        model.scroll = 0

        model, _ = _press(model, KEY_UP)
        self.assertEqual(model.scroll, WHEEL_SCROLL_LINES)
        model, _ = _press(model, KEY_DOWN)
        self.assertEqual(model.scroll, 0)

    def test_arrows_do_not_scroll_while_composing(self) -> None:
        """Mid-prompt the arrows mean history, so the transcript stays put."""
        model = Model()
        for index in range(40):
            model.append_line(LineRole.USER, f"line {index}")
        model.scroll = 0
        model.input_buffer = "draft"

        model, _ = _press(model, KEY_DOWN)
        self.assertEqual(model.scroll, 0)

    def test_submit_is_rejected_while_a_turn_is_running(self) -> None:
        model, _ = update(Model(), TextMsg("first\n"))
        model, _ = update(model, TextMsg("second\n"))

        self.assertEqual(model.history, ["first"])

    def test_esc_clears_the_input_buffer(self) -> None:
        model, _ = _type(Model(), "draft")
        model, _ = _press(model, KEY_ESC)

        self.assertEqual(model.input_buffer, "draft" if False else "")
        self.assertEqual(model.stream_buffer, "")

    def test_paging_scrolls_the_transcript(self) -> None:
        model = Model(height=20)
        for index in range(10):
            model.append_line(LineRole.USER, f"line {index}")

        model, _ = _press(model, KEY_PAGEUP)
        self.assertGreater(model.scroll, 0)

        model, _ = _press(model, KEY_PAGEDOWN)
        self.assertEqual(model.scroll, 0)


class MouseWheelScrollTests(unittest.TestCase):
    """The wheel scrolls the transcript instead of typing stray letters."""

    def _model_with_history(self) -> Model:
        model = Model(height=20)
        for index in range(20):
            model.append_line(LineRole.USER, f"line {index}")
        return model

    def test_wheel_up_scrolls_back_and_down_returns(self) -> None:
        model = self._model_with_history()

        model, _ = _press(model, KEY_SCROLL_UP)
        self.assertGreater(model.scroll, 0)

        model, _ = _press(model, KEY_SCROLL_DOWN)
        self.assertEqual(model.scroll, 0)

    def test_wheel_never_touches_the_composer(self) -> None:
        """The reported bug: scrolling typed "H"/"P" into the prompt."""
        model, _ = _type(Model(height=20), "keep me")

        model, _ = _press(model, KEY_SCROLL_UP)
        model, _ = _press(model, KEY_SCROLL_DOWN)

        self.assertEqual(model.input_buffer, "keep me")

    def test_wheel_scrolls_by_a_few_lines_not_a_page(self) -> None:
        model = self._model_with_history()
        page = model.visible_lines

        model, _ = _press(model, KEY_SCROLL_UP)
        self.assertLess(model.scroll, page)
        self.assertGreater(model.scroll, 1)

    def test_wheel_does_not_move_input_history(self) -> None:
        """Up/Down recall history; the wheel must not hijack them."""
        model = self._model_with_history()
        model.history = ["older"]
        model.history_index = None

        model, _ = _press(model, KEY_SCROLL_UP)

        self.assertEqual(model.input_buffer, "")
        self.assertIsNone(model.history_index)

    def test_wheel_scrolls_in_the_command_layer(self) -> None:
        model = self._model_with_history()
        model, _ = _press(model, KEY_ESC)  # enter the command layer
        self.assertTrue(model.command_mode)

        model, _ = _press(model, KEY_SCROLL_UP)
        self.assertGreater(model.scroll, 0)

    def test_wheel_moves_the_selection_in_panels(self) -> None:
        """In a list the natural meaning is "move the highlight"."""
        model, _ = update(
            Model(view=ActiveView.SKILLS),
            SkillsChangedMsg(
                skills=(
                    {"name": "a", "keywords": ("a",), "tools": ()},
                    {"name": "b", "keywords": ("b",), "tools": ()},
                    {"name": "c", "keywords": ("c",), "tools": ()},
                ),
                enabled=True,
                unmatched_policy="none",
                always_visible=("mcp",),
            ),
        )
        model.skill_selected = 0

        model, _ = _press(model, KEY_SCROLL_DOWN)
        self.assertEqual(model.skill_selected, 1)

        model, _ = _press(model, KEY_SCROLL_UP)
        self.assertEqual(model.skill_selected, 0)

    def test_wheel_is_ignored_while_awaiting_permission(self) -> None:
        """The modal must not shift underneath the user."""
        model = Model(height=20)
        for index in range(10):
            model.append_line(LineRole.USER, f"line {index}")
        request = PermissionRequest(request_id="r1", tool_name="run_shell")
        model, _ = update(model, AgentPermissionMsg(request))

        model, _ = _press(model, KEY_SCROLL_UP)

        self.assertEqual(model.scroll, 0)
        self.assertEqual(model.permission_choice, 0)


class ShiftCommandTests(unittest.TestCase):
    """Shift+letter commands, and the composer layer that protects typing."""

    def _command_layer(self) -> Model:
        model, _ = _press(Model(), KEY_ESC)
        return model

    def test_shift_q_quits_from_the_command_layer(self) -> None:
        model, _ = _type(self._command_layer(), KEY_SHIFT_Q)

        self.assertTrue(model.should_quit)

    def test_shift_c_cancels_while_running(self) -> None:
        model = Model(is_running=True, command_mode=True)
        model, cmd, calls = _run(model, KeyMsg("char", KEY_SHIFT_C))

        self.assertFalse(model.should_quit)
        self.assertEqual(calls, [("cancel_agent", ())])

    def test_shift_c_quits_when_idle(self) -> None:
        model, _ = _type(self._command_layer(), KEY_SHIFT_C)

        self.assertTrue(model.should_quit)

    def test_shift_letters_work_from_any_panel(self) -> None:
        for view in (ActiveView.SKILLS, ActiveView.MCP, ActiveView.SESSIONS):
            with self.subTest(view=view):
                model, _ = _type(Model(view=view), KEY_SHIFT_Q)

                self.assertTrue(model.should_quit)

    def test_shift_q_is_typed_not_interpreted_while_composing(self) -> None:
        # The whole point of the layer: a capital Q in the prompt must reach
        # the buffer instead of quitting the app.
        model, _ = _type(Model(), "Q")

        self.assertFalse(model.should_quit)
        self.assertEqual(model.input_buffer, "Q")


    def test_capitals_keep_working_mid_sentence(self) -> None:
        model, _ = _type(Model(), "Can you read THIS")

        self.assertFalse(model.should_quit)
        self.assertFalse(model.command_mode)
        self.assertEqual(model.input_buffer, "Can you read THIS")

    def test_esc_opens_the_command_layer_only_when_nothing_to_clear(self) -> None:
        with_text, _ = _type(Model(), "draft")
        with_text, _ = _press(with_text, KEY_ESC)

        self.assertEqual(with_text.input_buffer, "")
        self.assertFalse(with_text.command_mode)

        empty, _ = _press(Model(), KEY_ESC)

        self.assertTrue(empty.command_mode)

    def test_esc_closes_the_command_layer_again(self) -> None:
        model, _ = _press(self._command_layer(), KEY_ESC)

        self.assertFalse(model.command_mode)

    def test_shift_i_returns_to_typing_without_inserting_text(self) -> None:
        model, _ = _type(self._command_layer(), "I")

        self.assertFalse(model.command_mode)
        self.assertEqual(model.input_buffer, "")

    def test_any_other_letter_hands_control_back_to_the_prompt(self) -> None:
        model, _ = _type(self._command_layer(), "a")

        self.assertFalse(model.command_mode)
        self.assertEqual(model.input_buffer, "a")

    def test_i_then_a_capital_lets_a_prompt_start_with_q(self) -> None:
        model, _ = _type(self._command_layer(), "i")
        model, _ = _type(model, "Q")

        self.assertFalse(model.should_quit)
        self.assertEqual(model.input_buffer, "Q")

    def test_switching_panels_leaves_the_command_layer(self) -> None:
        model, _ = _press(self._command_layer(), KEY_TAB)

        self.assertFalse(model.command_mode)
        self.assertEqual(model.view, ActiveView.SKILLS)

    def test_history_navigation_still_works_in_the_command_layer(self) -> None:
        model = Model(command_mode=True)
        model.history = ["first", "second"]

        # The composer is empty here, so the arrow scrolls instead of
        # recalling; a draft brings history back.
        model.input_buffer = "draft"
        model, _ = _press(model, KEY_UP)

        self.assertEqual(model.input_buffer, "second")

    def test_shift_letters_drive_the_panel_actions(self) -> None:
        cases = [
            (ActiveView.SKILLS, "R", "reload_skills", ()),
            (ActiveView.MCP, "R", "reload_mcp", ()),
            (ActiveView.SESSIONS, "N", "create_session", ()),
        ]
        for view, char, hook, args in cases:
            with self.subTest(view=view, char=char):
                model, _cmd_, calls = _run(Model(view=view), KeyMsg("char", char))

                self.assertEqual(calls, [(hook, args)])

    def test_lowercase_still_works_in_panels_where_nothing_is_typed(self) -> None:
        model, _cmd_, calls = _run(
            Model(view=ActiveView.SKILLS), KeyMsg("char", "r")
        )

        self.assertEqual(calls, [("reload_skills", ())])

    def test_enter_still_sends_after_returning_to_typing(self) -> None:
        model, _ = _type(self._command_layer(), "i")
        model, _ = _type(model, "hello")
        model, cmd = _press(model, KEY_ENTER)

        self.assertEqual(model.status, "running")
        self.assertIsNotNone(cmd)


class QuitConfirmationTests(unittest.TestCase):
    """A stray quit must not throw away a turn the user is waiting on."""

    def _running(self) -> Model:
        model, _ = _press(Model(), KEY_ESC)  # reach the command layer
        model.is_running = True
        return model

    def test_the_first_quit_while_running_only_asks(self) -> None:
        model, _ = _type(self._running(), KEY_SHIFT_Q)

        self.assertFalse(model.should_quit)
        self.assertTrue(model.quit_pending)
        self.assertIn("Shift+Q", model.notice)

    def test_the_second_quit_goes_through(self) -> None:
        model, _ = _type(self._running(), KEY_SHIFT_Q)
        model, _ = _type(model, KEY_SHIFT_Q)

        self.assertTrue(model.should_quit)

    def test_any_other_key_cancels_the_confirmation(self) -> None:
        """The prompt lapses, so it cannot linger as a trap."""
        model, _ = _type(self._running(), KEY_SHIFT_Q)
        model, _ = _press(model, KEY_DOWN)

        self.assertFalse(model.quit_pending)
        self.assertFalse(model.should_quit)

    def test_quitting_while_idle_needs_no_confirmation(self) -> None:
        model, _ = _press(Model(), KEY_ESC)
        model, _ = _type(model, KEY_SHIFT_Q)

        self.assertTrue(model.should_quit)
        self.assertFalse(model.quit_pending)

    def test_shift_c_never_quits_mid_turn(self) -> None:
        """Cancel is not a stealthy way around the confirmation."""
        model = Model(is_running=True, command_mode=True)
        model, _cmd, calls = _run(model, KeyMsg("char", KEY_SHIFT_C))

        self.assertEqual(calls, [("cancel_agent", ())])
        self.assertFalse(model.should_quit)

    def test_an_interrupt_while_running_cancels_rather_than_exits(self) -> None:
        model = Model(is_running=True)
        model, _cmd, calls = _run(model, InterruptMsg())

        self.assertEqual(calls, [("cancel_agent", ())])
        self.assertFalse(model.should_quit)

    def test_an_interrupt_while_idle_exits(self) -> None:
        model, _cmd, calls = _run(Model(), InterruptMsg())

        self.assertTrue(model.should_quit)
        self.assertEqual(calls, [])


class TypedExitTests(unittest.TestCase):
    """``/exit`` works where Shift+Q cannot: while the prompt has focus."""

    def test_exit_commands_quit(self) -> None:
        for text in ("/exit", "/quit", "/q", "/EXIT"):
            with self.subTest(text=text):
                model, _ = _type(Model(), text)
                model, _cmd, calls = _run(model, KeyMsg(KEY_ENTER))

                self.assertTrue(model.should_quit)
                self.assertEqual(calls, [])

    def test_the_exit_command_is_not_echoed_to_the_transcript(self) -> None:
        model, _ = _type(Model(), "/exit")
        model, _ = _press(model, KEY_ENTER)

        self.assertEqual(model.transcript, [])

    def test_a_message_starting_with_a_slash_is_still_a_message(self) -> None:
        model, _ = _type(Model(), "/exiting soon")
        model, _cmd, calls = _run(model, KeyMsg(KEY_ENTER))

        self.assertFalse(model.should_quit)
        self.assertEqual(calls, [("run_agent", "/exiting soon")])

    def test_exit_while_running_still_asks_first(self) -> None:
        model, _ = _type(Model(is_running=True), "/exit")
        model, _cmd, calls = _run(model, KeyMsg(KEY_ENTER))

        self.assertFalse(model.should_quit)
        self.assertTrue(model.quit_pending)
        self.assertEqual(calls, [])


class ViewRoutingTests(unittest.TestCase):
    def test_tab_cycles_views_forward(self) -> None:
        model = Model()
        seen = [model.view]
        for _ in range(len(ActiveView)):
            model, _ = _press(model, KEY_TAB)
            seen.append(model.view)

        self.assertEqual(seen[0], ActiveView.CHAT)
        self.assertEqual(seen[1], ActiveView.SKILLS)
        self.assertEqual(seen[-1], ActiveView.CHAT)

    def test_backtab_cycles_backwards(self) -> None:
        model, _ = _press(Model(), KEY_BACKTAB)

        self.assertEqual(model.view, ActiveView.HELP)

    def test_resize_updates_dimensions(self) -> None:
        model, _ = update(Model(), ResizeMsg(40, 12))

        self.assertEqual((model.width, model.height), (40, 12))

    def test_resize_clamps_to_a_usable_minimum(self) -> None:
        model, _ = update(Model(), ResizeMsg(4, 2))

        self.assertEqual((model.width, model.height), (20, 8))

    def test_tick_keeps_the_model_unchanged(self) -> None:
        before = Model()
        model, cmd = update(before, TickMsg(1))

        self.assertIsNone(cmd)
        self.assertEqual(model.view, before.view)


class StreamTests(unittest.TestCase):
    def test_deltas_accumulate_in_the_stream_buffer(self) -> None:
        model, _ = update(Model(), AgentDeltaMsg("Hel"))
        model, _ = update(model, AgentDeltaMsg("lo"))

        self.assertEqual(model.stream_buffer, "Hello")

    def test_done_flushes_the_buffer_into_the_transcript(self) -> None:
        model, _ = update(Model(), AgentDeltaMsg("Hello"))
        model, _ = update(model, AgentDoneMsg(answer="Hello"))

        self.assertEqual(model.stream_buffer, "")
        self.assertEqual(model.transcript[-1].role, LineRole.ASSISTANT)
        self.assertEqual(model.transcript[-1].text, "Hello")
        self.assertFalse(model.is_running)
        self.assertEqual(model.status, "idle")

    def test_done_with_error_appends_an_error_line(self) -> None:
        model, _ = update(Model(), AgentDoneMsg(error="boom"))

        self.assertEqual(model.transcript[-1].role, LineRole.ERROR)
        self.assertEqual(model.error, "boom")
        self.assertEqual(model.status, "error")

    def test_cancelled_done_marks_status(self) -> None:
        model, _ = update(
            Model(), AgentDoneMsg(cancelled=True, error="Cancelled by user")
        )

        self.assertEqual(model.status, "cancelled")
        self.assertFalse(model.is_running)

    def test_agent_error_message_is_recorded(self) -> None:
        model, _ = update(Model(), AgentErrorMsg("ValueError: nope"))

        self.assertEqual(model.transcript[-1].role, LineRole.ERROR)
        self.assertEqual(model.error, "ValueError: nope")

    def test_started_clears_previous_errors(self) -> None:
        model = Model(error="old", is_running=False)
        model, _ = update(model, AgentStartedMsg("hi"))

        self.assertIsNone(model.error)
        self.assertTrue(model.is_running)
        self.assertEqual(model.status, "running")

    def test_tool_status_lines_are_appended(self) -> None:
        model, _ = update(
            Model(),
            AgentToolMsg(
                "read_file",
                "completed",
                {"content": "first line\nsecond line"},
            ),
        )

        self.assertEqual(model.transcript[-1].role, LineRole.TOOL)
        self.assertIn("read_file", model.transcript[-1].text)
        self.assertIn("first line", model.transcript[-1].text)
        self.assertNotIn("second line", model.transcript[-1].text)

    def test_running_tool_status_summarizes_arguments(self) -> None:
        model, _ = update(
            Model(),
            AgentToolMsg("write_file", "running", {"path": "a.txt"}),
        )

        self.assertIn("path=a.txt", model.transcript[-1].text)

    def test_notice_is_stored_on_the_model(self) -> None:
        model, _ = update(Model(), NoticeMsg("hello"))

        self.assertEqual(model.notice, "hello")

    def test_transcript_stays_chronological_across_a_tool_call(self) -> None:
        model, _ = update(Model(), AgentDeltaMsg("thinking..."))
        model, _ = update(
            model, AgentToolMsg("read_file", "running", {"path": "a.txt"})
        )
        model, _ = update(model, AgentDoneMsg(answer="thinking..."))

        roles = [line.role for line in model.transcript]
        self.assertEqual(
            roles, [LineRole.ASSISTANT, LineRole.TOOL]
        )
        self.assertEqual(model.transcript[0].text, "thinking...")
        self.assertEqual(model.stream_buffer, "")


class PermissionTests(unittest.TestCase):
    def _request(self) -> PermissionRequest:
        return PermissionRequest(
            request_id="r1",
            tool_name="run_shell",
            arguments={"command": "ls"},
            risk_level="high",
            risk_tags=["shell"],
        )

    def test_permission_message_opens_the_overlay(self) -> None:
        model, _ = update(Model(), AgentPermissionMsg(self._request()))

        self.assertIsNotNone(model.permission)
        self.assertEqual(model.status, "waiting_permission")

    def test_enter_picks_the_highlighted_choice(self) -> None:
        model, _ = update(Model(), AgentPermissionMsg(self._request()))
        model, _ = _press(model, KEY_DOWN)
        model, cmd, calls = _run(model, KeyMsg(KEY_ENTER))

        self.assertIsNone(model.permission)
        self.assertEqual(calls, [("resolve_permission", PermissionDecision.DENY_ONCE)])

    def test_four_decisions_are_offered(self) -> None:
        options = permission_options()

        self.assertEqual(len(options), 4)
        self.assertEqual(
            [decision for _, decision in options],
            [
                PermissionDecision.ALLOW_ONCE,
                PermissionDecision.DENY_ONCE,
                PermissionDecision.ALLOW_SESSION,
                PermissionDecision.DENY_SESSION,
            ],
        )

    def test_shortcut_letters_produce_each_decision(self) -> None:
        cases = {
            "a": PermissionDecision.ALLOW_ONCE,
            "d": PermissionDecision.DENY_ONCE,
            "A": PermissionDecision.ALLOW_SESSION,
            "D": PermissionDecision.DENY_SESSION,
        }
        for shortcut, expected in cases.items():
            with self.subTest(shortcut=shortcut):
                model, _ = update(Model(), AgentPermissionMsg(self._request()))
                model, cmd, calls = _run(model, KeyMsg("char", shortcut))
                self.assertEqual(calls, [("resolve_permission", expected)])

    def test_esc_denies_once(self) -> None:
        model, _ = update(Model(), AgentPermissionMsg(self._request()))
        model, cmd, calls = _run(model, KeyMsg(KEY_ESC))

        self.assertEqual(calls, [("resolve_permission", PermissionDecision.DENY_ONCE)])

    def test_selection_is_clamped_to_the_option_count(self) -> None:
        model, _ = update(Model(), AgentPermissionMsg(self._request()))
        for _ in range(10):
            model, _ = _press(model, KEY_DOWN)
        self.assertEqual(model.permission_choice, 3)

        for _ in range(10):
            model, _ = _press(model, KEY_UP)
        self.assertEqual(model.permission_choice, 0)

    def test_permission_swallows_other_panel_keys(self) -> None:
        model, _ = update(Model(), AgentPermissionMsg(self._request()))
        model, cmd = _press(model, KEY_TAB)

        self.assertEqual(model.view, ActiveView.CHAT)
        self.assertIsNone(cmd)

    def test_permission_arrow_keys_do_not_scroll_chat(self) -> None:
        model = Model()
        for index in range(10):
            model.append_line(LineRole.USER, f"line {index}")
        model, _ = update(model, AgentPermissionMsg(self._request()))
        before = model.scroll

        model, _ = _press(model, KEY_RIGHT)

        self.assertEqual(model.scroll, before)


class SkillsPanelTests(unittest.TestCase):
    def _model_with_skills(self) -> Model:
        model, _ = update(
            Model(view=ActiveView.SKILLS),
            SkillsChangedMsg(
                skills=(
                    {"name": "files", "keywords": ("file",), "tools": ("read_file",)},
                    {"name": "shell", "keywords": ("run",), "tools": ("run_shell",)},
                ),
                enabled=True,
                unmatched_policy="none",
                always_visible=("mcp",),
            ),
        )
        return model

    def test_snapshot_populates_the_panel(self) -> None:
        model = self._model_with_skills()

        self.assertEqual([skill.name for skill in model.skills], ["files", "shell"])
        self.assertEqual(model.always_visible, ["mcp"])
        self.assertTrue(model.skills_enabled)

    def test_arrow_keys_move_the_selection(self) -> None:
        model = self._model_with_skills()
        model, _ = _press(model, KEY_DOWN)

        self.assertEqual(model.skill_selected, 1)

    def test_space_toggles_skills_and_calls_back(self) -> None:
        model = self._model_with_skills()
        model, cmd, calls = _run(model, KeyMsg(KEY_SPACE))

        self.assertFalse(model.skills_enabled)
        self.assertEqual(calls, [("save_skills", False)])

    def test_r_reloads_skills(self) -> None:
        model = self._model_with_skills()
        model, cmd, calls = _run(model, KeyMsg("char", "r"))

        self.assertEqual(calls, [("reload_skills", ())])


class MCPPanelTests(unittest.TestCase):
    def _model(self) -> Model:
        model, _ = update(
            Model(view=ActiveView.MCP),
            McpChangedMsg(
                servers=(
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
                )
            ),
        )
        return model

    def test_snapshot_populates_the_panel(self) -> None:
        model = self._model()

        self.assertEqual([s.name for s in model.mcp_servers], ["fs", "remote"])
        self.assertTrue(model.mcp_servers[0].started)
        self.assertFalse(model.mcp_servers[1].enabled)

    def test_space_disables_the_selected_server(self) -> None:
        model = self._model()
        model, cmd, calls = _run(model, KeyMsg(KEY_SPACE))

        self.assertFalse(model.mcp_servers[0].enabled)
        self.assertEqual(calls, [("set_mcp_enabled", ("fs", False))])

    def test_space_enables_the_selected_server(self) -> None:
        model = self._model()
        model, _ = _press(model, KEY_DOWN)
        model, cmd, calls = _run(model, KeyMsg(KEY_SPACE))

        self.assertTrue(model.mcp_servers[1].enabled)
        self.assertEqual(calls, [("set_mcp_enabled", ("remote", True))])

    def test_r_reloads_servers(self) -> None:
        model = self._model()
        model, cmd, calls = _run(model, KeyMsg("char", "R"))

        self.assertEqual(calls, [("reload_mcp", ())])

    def test_space_without_servers_is_a_noop(self) -> None:
        model, cmd = _press(Model(view=ActiveView.MCP), KEY_SPACE)

        self.assertIsNone(cmd)


class SessionsPanelTests(unittest.TestCase):
    def _model(self) -> Model:
        model, _ = update(
            Model(view=ActiveView.SESSIONS),
            SessionChangedMsg(
                session_id="default",
                sessions=(
                    {"session_id": "default", "message_count": 4, "preview": "hi"},
                    {"session_id": "work", "message_count": 2, "preview": "yo"},
                ),
                archive_dir=".tool_outputs/default",
            ),
        )
        return model

    def test_snapshot_populates_the_panel(self) -> None:
        model = self._model()

        self.assertEqual([s.session_id for s in model.sessions], ["default", "work"])
        self.assertEqual(model.archive_dir, ".tool_outputs/default")

    def test_enter_switches_to_the_selected_session(self) -> None:
        model = self._model()
        model, _ = _press(model, KEY_DOWN)
        model, cmd, calls = _run(model, KeyMsg(KEY_ENTER))

        self.assertEqual(calls, [("switch_session", "work")])

    def test_n_creates_a_session(self) -> None:
        model = self._model()
        model, cmd, calls = _run(model, KeyMsg("char", "n"))

        self.assertEqual(calls, [("create_session", ())])

    def test_d_deletes_the_selected_session(self) -> None:
        model = self._model()
        model, _ = _press(model, KEY_DOWN)
        model, cmd, calls = _run(model, KeyMsg("char", "d"))

        self.assertEqual(calls, [("delete_session", "work")])


class ImmutabilityTests(unittest.TestCase):
    def test_update_does_not_mutate_the_input_model(self) -> None:
        original = Model()
        original.append_line(LineRole.USER, "seed")
        snapshot = list(original.transcript)

        updated, _ = _type(original, "abc")

        self.assertEqual(original.input_buffer, "")
        self.assertEqual(original.transcript, snapshot)
        self.assertEqual(updated.input_buffer, "abc")

    def test_update_returns_a_distinct_model(self) -> None:
        original = Model()
        updated, _ = update(original, TickMsg())

        self.assertIsNot(updated, original)


class CaretBlinkTests(unittest.TestCase):
    """The caret must blink while idle and reappear the instant you type."""

    def test_ticks_flip_the_caret_on_a_fixed_period(self) -> None:
        model = Model()

        states = []
        for tick in range(BLINK_TICKS * 3):
            model, _ = update(model, TickMsg(tick))
            states.append(model.cursor_on)

        # On for the first half period, off for the second, and so on.
        self.assertEqual(states[:BLINK_TICKS], [True] * BLINK_TICKS)
        self.assertEqual(states[BLINK_TICKS : BLINK_TICKS * 2], [False] * BLINK_TICKS)
        self.assertEqual(states[BLINK_TICKS * 2 :], [True] * BLINK_TICKS)

    def test_a_tick_only_changes_the_model_on_a_flip(self) -> None:
        model = Model()

        steady, _ = update(model, TickMsg(1))

        self.assertTrue(steady.cursor_on)
        flipping, _ = update(model, TickMsg(BLINK_TICKS))
        self.assertFalse(flipping.cursor_on)

    def test_typing_turns_the_caret_back_on(self) -> None:
        model, _ = update(Model(), TickMsg(BLINK_TICKS))
        self.assertFalse(model.cursor_on)

        model, _ = _type(model, "h")

        self.assertTrue(model.cursor_on)
        self.assertEqual(model.input_buffer, "h")

    def test_every_key_restores_the_caret(self) -> None:
        for key in (KEY_UP, KEY_DOWN, KEY_TAB, KEY_ESC, KEY_SPACE, KEY_PAGEUP):
            with self.subTest(key=key):
                model, _ = update(Model(), TickMsg(BLINK_TICKS))
                model, _ = _press(model, key)

                self.assertTrue(model.cursor_on)


if __name__ == "__main__":
    unittest.main()
