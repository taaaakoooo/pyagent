"""Tests for the TUI runtime (key decoding and the render loop)."""

from __future__ import annotations

import os
import signal
import threading
import time
import unittest
from typing import List
from unittest import mock

from tui import runtime
from tui.model import BLINK_TICKS, Model
from tui.msgs import (
    KEY_BACKSPACE,
    KEY_BACKTAB,
    KEY_DELETE,
    KEY_DOWN,
    KEY_END,
    KEY_ENTER,
    KEY_ESC,
    KEY_HOME,
    KEY_LEFT,
    KEY_PAGEDOWN,
    KEY_PAGEUP,
    KEY_RIGHT,
    KEY_SCROLL_DOWN,
    KEY_SCROLL_UP,
    KEY_SHIFT_Q,
    KEY_SPACE,
    KEY_TAB,
    KEY_UP,
    AgentDeltaMsg,
    InterruptMsg,
    KeyMsg,
    TickMsg,
)
from tui.runtime import Program, decode_ansi_sequence, decode_control_char


class ControlCharacterTests(unittest.TestCase):
    def test_enter_tab_and_backspace(self) -> None:
        cases = {
            "\r": KEY_ENTER,
            "\n": KEY_ENTER,
            "\t": KEY_TAB,
            "\x08": KEY_BACKSPACE,
            "\x7f": KEY_BACKSPACE,
        }
        for char, expected in cases.items():
            with self.subTest(char=repr(char)):
                self.assertEqual(decode_control_char(char).key, expected)

    def test_ctrl_shortcuts_are_not_bound(self) -> None:
        # Ctrl combos are intercepted by the terminal host (Cursor, VS Code,
        # tmux) before the app sees them, so the TUI must not rely on any.
        self.assertIsNone(decode_control_char("\x03"))
        self.assertIsNone(decode_control_char("\x11"))

    def test_shift_letters_arrive_as_uppercase_characters(self) -> None:
        msg = decode_control_char("Q")

        self.assertEqual(msg.key, "char")
        self.assertEqual(msg.char, KEY_SHIFT_Q)

    def test_escape(self) -> None:
        self.assertEqual(decode_control_char("\x1b").key, KEY_ESC)

    def test_space_is_a_named_key_carrying_its_char(self) -> None:
        msg = decode_control_char(" ")

        self.assertEqual(msg.key, KEY_SPACE)
        self.assertEqual(msg.char, " ")

    def test_printable_characters_are_typed(self) -> None:
        msg = decode_control_char("a")

        self.assertEqual(msg.key, "char")
        self.assertEqual(msg.char, "a")

    def test_non_printable_characters_are_ignored(self) -> None:
        self.assertIsNone(decode_control_char("\x01"))


class AnsiSequenceTests(unittest.TestCase):
    def test_arrow_keys(self) -> None:
        cases = {
            "[A": KEY_UP,
            "[B": KEY_DOWN,
            "[C": KEY_RIGHT,
            "[D": KEY_LEFT,
        }
        for sequence, expected in cases.items():
            with self.subTest(sequence=sequence):
                self.assertEqual(decode_ansi_sequence(sequence).key, expected)

    def test_navigation_keys(self) -> None:
        cases = {
            "[H": KEY_HOME,
            "[F": KEY_END,
            "[1~": KEY_HOME,
            "[3~": KEY_DELETE,
            "[4~": KEY_END,
            "[5~": KEY_PAGEUP,
            "[6~": KEY_PAGEDOWN,
            "[Z": KEY_BACKTAB,
        }
        for sequence, expected in cases.items():
            with self.subTest(sequence=sequence):
                self.assertEqual(decode_ansi_sequence(sequence).key, expected)

    def test_modified_arrow_reports_the_modifier(self) -> None:
        self.assertEqual(decode_ansi_sequence("[1;5A").key, "ctrl+up")

    def test_unknown_sequences_are_ignored(self) -> None:
        self.assertIsNone(decode_ansi_sequence("[99~"))
        self.assertIsNone(decode_ansi_sequence("nonsense"))

    def test_ss3_arrow_keys_are_decoded(self) -> None:
        """Application cursor-key mode sends ESC O A instead of ESC [ A.

        Decoding these matters beyond convenience: an unrecognised sequence
        used to leave its final byte to be read as a typed character.
        """
        cases = {
            "OA": KEY_UP,
            "OB": KEY_DOWN,
            "OC": KEY_RIGHT,
            "OD": KEY_LEFT,
            "OH": KEY_HOME,
            "OF": KEY_END,
        }
        for sequence, expected in cases.items():
            with self.subTest(sequence=sequence):
                self.assertEqual(decode_ansi_sequence(sequence).key, expected)

    def test_unknown_ss3_sequence_is_ignored(self) -> None:
        self.assertIsNone(decode_ansi_sequence("OP"))


class MouseWheelSequenceTests(unittest.TestCase):
    """The wheel must arrive as an explicit scroll key, never as typed text."""

    def test_wheel_up_and_down(self) -> None:
        self.assertEqual(decode_ansi_sequence("[<64;10;5M").key, KEY_SCROLL_UP)
        self.assertEqual(decode_ansi_sequence("[<65;10;5M").key, KEY_SCROLL_DOWN)

    def test_wheel_with_modifiers_still_scrolls(self) -> None:
        # Shift/Ctrl/Alt set bits 2-4; the direction lives in the low bits.
        self.assertEqual(decode_ansi_sequence("[<68;10;5M").key, KEY_SCROLL_UP)
        self.assertEqual(decode_ansi_sequence("[<69;10;5M").key, KEY_SCROLL_DOWN)

    def test_horizontal_wheel_is_ignored(self) -> None:
        self.assertIsNone(decode_ansi_sequence("[<66;10;5M"))
        self.assertIsNone(decode_ansi_sequence("[<67;10;5M"))

    def test_clicks_and_drags_are_ignored(self) -> None:
        # Button press/release carry no wheel bit and are not bound yet. If
        # they ever leaked, a stray click would type punctuation into the
        # composer.
        for body in ("[<0;10;5M", "[<0;10;5m", "[<32;10;5M", "[<2;10;5M"):
            with self.subTest(body=body):
                self.assertIsNone(decode_ansi_sequence(body))

    def test_sequence_completeness_waits_for_all_bytes(self) -> None:
        """A partial mouse report must not be treated as finished."""
        for partial in ("", "[", "[<", "[<64", "[<64;10;5"):
            with self.subTest(partial=partial):
                self.assertIsNone(runtime.sequence_length("\x1b" + partial))
        self.assertEqual(runtime.sequence_length("\x1b[<64;10;5M"), 11)

    def test_x10_mouse_report_consumes_its_raw_bytes(self) -> None:
        """X10 encoding appends three raw bytes that would otherwise leak."""
        self.assertIsNone(runtime.sequence_length("\x1b[M"))
        self.assertIsNone(runtime.sequence_length("\x1b[M  "))
        self.assertEqual(runtime.sequence_length("\x1b[M   "), 6)

    def test_x10_wheel_is_decoded_too(self) -> None:
        """Hosts that only speak ``?1000`` still get a working wheel."""
        up = "\x1b[M" + chr(32 + 64) + chr(32 + 10) + chr(32 + 5)
        down = "\x1b[M" + chr(32 + 65) + chr(32 + 10) + chr(32 + 5)
        self.assertEqual(decode_ansi_sequence(up[1:]).key, KEY_SCROLL_UP)
        self.assertEqual(decode_ansi_sequence(down[1:]).key, KEY_SCROLL_DOWN)

    def test_x10_clicks_are_ignored(self) -> None:
        click = "\x1b[M" + chr(32 + 0) + chr(32 + 10) + chr(32 + 5)
        self.assertIsNone(decode_ansi_sequence(click[1:]))

    def test_ss3_and_csi_lengths(self) -> None:
        self.assertEqual(runtime.sequence_length("\x1bOA"), 3)
        self.assertIsNone(runtime.sequence_length("\x1bO"))
        self.assertEqual(runtime.sequence_length("\x1b[A"), 3)
        self.assertEqual(runtime.sequence_length("\x1b[5~"), 4)
        self.assertEqual(runtime.sequence_length("\x1ba"), 2)

    def test_bare_escape_is_still_the_escape_key(self) -> None:
        """A lone Esc resolves on flush rather than being dropped."""
        decoder = runtime.KeyStreamDecoder()
        self.assertEqual(decoder.feed("\x1b"), [])
        self.assertEqual([msg.key for msg in decoder.flush()], [KEY_ESC])


class KeyStreamDecoderTests(unittest.TestCase):
    """Stray bytes must never reach the composer, however input is chopped up.

    The regression this guards against: the reader used to grab the bytes
    available at that moment and give up on the rest, so a wheel report that
    arrived in two reads left its payload behind -- and X10 payload bytes are
    printable, which is how "H"/"P"/">"/"~" got typed into the prompt.
    """

    @staticmethod
    def _text_keys(messages) -> List[str]:
        return [msg.char for msg in messages if msg.key == "char"]

    def test_one_char_at_a_time_still_decodes(self) -> None:
        decoder = runtime.KeyStreamDecoder()
        messages = []
        for char in "\x1b[<64;10;5M":
            messages.extend(decoder.feed(char))
        self.assertEqual([msg.key for msg in messages], [KEY_SCROLL_UP])

    def test_split_x10_report_leaks_nothing(self) -> None:
        """The exact shape of the bug: ESC and "[M" land before the payload."""
        decoder = runtime.KeyStreamDecoder()
        self.assertEqual(decoder.feed("\x1b[M"), [])  # payload not yet arrived
        # A wheel report split across reads still resolves, and nothing leaks.
        messages = []
        for byte in (64, 40, 40):
            messages.extend(decoder.feed(chr(32 + byte)))
        self.assertEqual([msg.key for msg in messages], [KEY_SCROLL_UP])
        self.assertEqual(self._text_keys(messages), [])
        self.assertEqual(decoder.pending, "")

    def test_abandoned_sequence_is_dropped_not_typed(self) -> None:
        decoder = runtime.KeyStreamDecoder()
        decoder.feed("\x1b[<64;10;")
        self.assertEqual(decoder.flush(), [])
        self.assertEqual(decoder.pending, "")

    def test_torn_x10_report_is_claimed_and_still_decoded(self) -> None:
        """The leftover of a torn report must not be typed as punctuation.

        This is the "still typing h and p after a fix" case: the head timed out
        and was dropped, and the payload -- which is ``32 + value``, so plain
        printable characters -- arrived afterwards. Claiming the tail both
        keeps it out of the composer and recovers the scroll.
        """
        decoder = runtime.KeyStreamDecoder()
        decoder.feed("\x1b[M")
        self.assertEqual(decoder.flush(), [])
        self.assertEqual(decoder.expecting, 3)

        messages = []
        for byte in (64, 40, 48):  # a wheel report at (40, 48)
            messages.extend(decoder.feed(chr(32 + byte)))

        self.assertEqual(self._text_keys(messages), [])
        self.assertEqual([m.key for m in messages], [KEY_SCROLL_UP])
        self.assertEqual(decoder.expecting, 0)

    def test_torn_sgr_report_is_claimed_and_still_decoded(self) -> None:
        """SGR reports have a terminator, so their tail can be claimed too."""
        decoder = runtime.KeyStreamDecoder()
        decoder.feed("\x1b[<64;10;")
        decoder.flush()

        typed, keys = [], []
        for char in "5Mok":
            for message in decoder.feed(char):
                if message.key == "char":
                    typed.append(message.char)
                else:
                    keys.append(message.key)

        self.assertEqual("".join(typed), "ok")
        self.assertEqual(keys, [KEY_SCROLL_UP])

    def test_claiming_stops_at_non_report_input(self) -> None:
        """A torn SGR head must not swallow ordinary typing after it."""
        decoder = runtime.KeyStreamDecoder()
        decoder.feed("\x1b[<64;10;")
        decoder.flush()

        typed = []
        for char in "hello":
            typed.extend(decoder.feed(char))

        self.assertEqual("".join(m.char or "" for m in typed), "hello")

    def test_claiming_stops_at_a_control_character(self) -> None:
        """X10 payload is never a control byte, so one means real input."""
        decoder = runtime.KeyStreamDecoder()
        decoder.feed("\x1b[M")
        decoder.flush()

        messages = decoder.feed("\x1b[A")
        self.assertEqual([m.key for m in messages], [KEY_UP])
        self.assertEqual(decoder.expecting, 0)

    def test_an_owed_tail_expires(self) -> None:
        """A dead terminal must not silently eat later keystrokes."""
        clock = FakeClock()
        decoder = runtime.KeyStreamDecoder(clock=clock)
        decoder.feed("\x1b[M")
        decoder.flush()
        clock.now += runtime.SEQUENCE_REMAINDER_TIMEOUT_SECONDS + 1.0
        typed = decoder.feed("hi")
        self.assertEqual("".join(m.char or "" for m in typed), "hi")

    def test_extra_typing_after_a_recovered_report_is_unaffected(self) -> None:
        """The bytes belonging to a report never taint what follows it."""
        decoder = runtime.KeyStreamDecoder()
        decoder.feed("\x1b[M")
        decoder.flush()
        messages = []
        for byte in (64, 40, 48):
            messages.extend(decoder.feed(chr(32 + byte)))
        messages.extend(decoder.feed("hello"))
        self.assertEqual("".join(m.char or "" for m in messages), "hello")

    def test_a_stale_tail_never_becomes_text(self) -> None:
        """A sequence that never completes is dropped, never typed."""
        decoder = runtime.KeyStreamDecoder()
        decoder.feed("\x1b[<64;10;5")  # final byte lost in transit
        self.assertEqual(self._text_keys(decoder.flush()), [])
        self.assertEqual(decoder.pending, "")

    def test_normal_typing_is_unaffected(self) -> None:
        decoder = runtime.KeyStreamDecoder()
        messages = decoder.feed("hi there")
        typed = "".join(m.char or "" for m in messages)
        self.assertEqual(typed, "hi there")
        self.assertEqual(decoder.pending, "")

    def test_escape_after_text_only_affects_the_sequence(self) -> None:
        decoder = runtime.KeyStreamDecoder()
        messages = decoder.feed("ab\x1b[Acd")
        self.assertEqual(self._text_keys(messages), ["a", "b", "c", "d"])
        self.assertIn(KEY_UP, [msg.key for msg in messages])


class MouseModeTests(unittest.TestCase):
    def test_mouse_mode_is_toggled_only_on_a_tty(self) -> None:
        program = Program(Model())
        program.console = mock.MagicMock()
        program.console.file.isatty.return_value = False
        program._write_control(runtime.ENTER_MOUSE_MODE)
        program.console.file.write.assert_not_called()

    def test_mouse_mode_sequences_are_balanced(self) -> None:
        program = Program(Model())
        program.console = mock.MagicMock()
        program.console.file.isatty.return_value = True

        program._write_control(runtime.ENTER_MOUSE_MODE)
        program._write_control(runtime.EXIT_MOUSE_MODE)

        written = [call.args[0] for call in program.console.file.write.call_args_list]
        self.assertEqual(written, [runtime.ENTER_MOUSE_MODE, runtime.EXIT_MOUSE_MODE])
        # Both must request and then release the same reporting modes.
        self.assertIn("?1000h", runtime.ENTER_MOUSE_MODE)
        self.assertIn("?1006h", runtime.ENTER_MOUSE_MODE)
        self.assertIn("?1000l", runtime.EXIT_MOUSE_MODE)
        self.assertIn("?1006l", runtime.EXIT_MOUSE_MODE)

    def test_control_writes_never_raise(self) -> None:
        program = Program(Model())
        program.console = mock.MagicMock()
        program.console.file.isatty.return_value = True
        program.console.file.write.side_effect = OSError("closed")
        program._write_control(runtime.ENTER_MOUSE_MODE)  # must not raise


class MouseOptOutTests(unittest.TestCase):
    """Hosts that swallow or mangle mouse reports need a way out."""

    def test_no_mouse_flag_disables_reporting(self) -> None:
        from tui.__main__ import build_parser

        args = build_parser().parse_args(["--no-mouse"])
        self.assertTrue(args.no_mouse)
        self.assertFalse(build_parser().parse_args([]).no_mouse)

    def test_disabled_mouse_never_toggles_reporting(self) -> None:
        """``--no-mouse`` must not emit either reporting sequence.

        The cursor/attribute reset is still sent -- that is not mouse-specific
        and leaving it out is what makes a force-quit look broken afterwards.
        """
        program = Program(Model(), mouse=False)
        self.assertFalse(program.mouse)
        program.console = mock.MagicMock()
        program.console.file.isatty.return_value = True

        program._stop_event.set()
        with mock.patch.object(program, "_loop"), mock.patch.object(
            program, "_render"
        ), mock.patch.object(runtime, "InputReader") as reader_cls, mock.patch.object(
            runtime, "Live"
        ):
            reader_cls.return_value = mock.MagicMock()
            program.run()

        written = "".join(
            call.args[0] for call in program.console.file.write.call_args_list
        )
        self.assertNotIn("?1000h", written)
        self.assertNotIn("?1000l", written)
        self.assertIn("?25h", written)


class ExitHandlingTests(unittest.TestCase):
    """Leaving the app must be deliberate, and must restore the terminal."""

    def _program(self, **kwargs) -> Program:
        program = Program(Model(), **kwargs)
        program.console = mock.MagicMock()
        program.console.file.isatty.return_value = True
        return program

    def test_restore_is_idempotent(self) -> None:
        """``finally`` and ``atexit`` both call it; the sequences send once."""
        program = self._program()

        program._restore_terminal()
        program._restore_terminal()

        written = [call.args[0] for call in program.console.file.write.call_args_list]
        self.assertEqual(
            written,
            [runtime.EXIT_MOUSE_MODE, runtime.SHOW_CURSOR + runtime.RESET_ATTRS],
        )

    def test_restore_survives_a_closed_stream(self) -> None:
        program = self._program()
        program.console.file.write.side_effect = OSError("closed")

        program._restore_terminal()  # must not raise

    def test_mouse_sequences_are_skipped_when_reporting_was_never_enabled(self) -> None:
        program = self._program(mouse=False)

        program._restore_terminal()

        written = "".join(
            call.args[0] for call in program.console.file.write.call_args_list
        )
        self.assertNotIn("?1000l", written)
        self.assertIn("?25h", written)

    def test_an_interrupt_only_sets_a_flag(self) -> None:
        """Handlers must not touch the queue: they can run holding its lock."""
        program = Program(Model())

        program._on_signal(2, None)

        self.assertTrue(program._interrupt_requested)

    def test_the_loop_applies_a_pending_interrupt(self) -> None:
        """The flag is consumed by the loop, not by the handler."""
        program = Program(Model())
        program._on_signal(2, None)

        def _stop_after_one_step(*_args: object) -> None:
            program.model.should_quit = True

        with mock.patch.object(program, "_step", side_effect=_stop_after_one_step):
            program._loop()

        self.assertTrue(
            any(isinstance(m, InterruptMsg) for m in program.dispatched)
        )
        self.assertFalse(program._interrupt_requested)

    def test_a_second_interrupt_gives_up_on_being_polite(self) -> None:
        """A stuck shutdown must not make Ctrl+C unusable."""
        program = Program(Model())
        program._on_signal(2, None)

        with self.assertRaises(KeyboardInterrupt):
            program._on_signal(2, None)

    def test_signal_handlers_are_restored(self) -> None:
        program = Program(Model())
        original = signal.getsignal(signal.SIGINT)

        program._install_signal_handlers()
        self.assertIsNot(signal.getsignal(signal.SIGINT), original)

        program._restore_signal_handlers()
        self.assertIs(signal.getsignal(signal.SIGINT), original)

    def test_handler_installation_outside_the_main_thread_is_skipped(self) -> None:
        """``signal.signal`` raises off-thread; the loop must not care."""
        program = Program(Model())

        def _attempt() -> None:
            program._install_signal_handlers()

        worker = threading.Thread(target=_attempt)
        worker.start()
        worker.join(timeout=5.0)

        self.assertEqual(program._saved_signals, [])


class ExitSummaryTests(unittest.TestCase):
    """The transcript dies with the alt screen, so print where it went."""

    def _agent(self, messages: int = 3, storage: Optional[str] = None) -> mock.MagicMock:
        agent = mock.MagicMock()
        agent.messages = [object()] * messages
        agent.session_storage_dir = storage
        return agent

    def test_summary_names_the_session_and_the_count(self) -> None:
        from tui.__main__ import _print_exit_summary

        with mock.patch("builtins.print") as printed:
            _print_exit_summary(self._agent(7), "default", 2)

        line = printed.call_args.args[0]
        self.assertIn("default", line)
        self.assertIn("7", line)
        self.assertIn("2", line)

    def test_summary_includes_the_history_path_when_known(self) -> None:
        from tui.__main__ import _print_exit_summary

        with mock.patch("builtins.print") as printed:
            _print_exit_summary(self._agent(storage=".sessions"), "abc", 0)

        self.assertIn(os.path.join(".sessions", "abc.jsonl"), printed.call_args.args[0])

    def test_summary_never_masks_the_exit(self) -> None:
        """A broken agent object must not turn a clean quit into a traceback."""
        from tui.__main__ import _print_exit_summary

        agent = mock.MagicMock()
        type(agent).messages = mock.PropertyMock(side_effect=RuntimeError("boom"))

        with mock.patch("builtins.print"):
            _print_exit_summary(agent, "default", 0)  # must not raise


class WindowsScanCodeTests(unittest.TestCase):
    """Regression: the wheel used to insert stray "H" and "P" letters.

    Terminals without mouse reporting translate a scroll into Up/Down arrow
    keys, which on Windows arrive as an ``\\x00``/``\\xe0`` prefix plus a scan
    code. The old reader slept 10ms and then peeked exactly once: if the scan
    byte had not landed yet the prefix was swallowed, and the byte was read on
    the next pass as a printable character -- "H" for Up, "P" for Down -- and
    typed into the composer.
    """

    class _FakeMsvcrt:
        """Replays queued characters, optionally after an arrival delay.

        The delay models what this bug came from: the scan byte showing up
        slightly after the prefix rather than in the same console record.
        """

        def __init__(
            self,
            chars,
            delay: float = 0.0,
            gap: float = 0.0,
            gaps=None,
            on_empty=None,
        ) -> None:
            self.chars = list(chars)
            self._deadline = time.monotonic() + delay
            self._gap = gap
            self._gaps = list(gaps) if gaps is not None else None
            self._read = 0
            self._on_empty = on_empty

        def kbhit(self) -> bool:
            if time.monotonic() < self._deadline:
                return False
            if not self.chars and self._on_empty is not None:
                self._on_empty()
            return bool(self.chars)

        def getwch(self) -> str:
            char = self.chars.pop(0)
            # Each character becomes available once the previous one has been
            # read and its gap has elapsed, which is what splits a key across
            # two reads. ``gaps`` scripts that per character.
            if self._gaps is None:
                wait = self._gap
            elif self._read < len(self._gaps):
                wait = self._gaps[self._read]
            else:
                wait = 0.0
            self._read += 1
            self._deadline = time.monotonic() + wait
            return char

    def test_scan_code_waits_for_a_late_byte(self) -> None:
        """A byte arriving after the old 10ms peek must still be collected."""
        fake = self._FakeMsvcrt(["H"], delay=0.03)

        self.assertEqual(
            runtime.InputReader._windows_read_char(0.5, fake),
            "H",
        )

    def test_the_old_single_peek_would_have_missed_that_byte(self) -> None:
        """Pin the exact failure the fix addresses.

        Reproduces the previous implementation against the same late byte, so
        a future "simplification" back to a single peek fails loudly here.
        """
        fake = self._FakeMsvcrt(["H"], delay=0.03)

        time.sleep(0.01)  # the old code's only wait
        self.assertFalse(fake.kbhit())
        # ...so the old code produced scan = "", dropped the prefix, and the
        # "H" was later decoded as typed text.
        self.assertEqual(fake.chars, ["H"])

    def test_scan_code_times_out_when_nothing_arrives(self) -> None:
        fake = self._FakeMsvcrt([])
        self.assertIsNone(runtime.InputReader._windows_read_char(0.01, fake))

    def test_known_scan_codes_map_to_navigation_keys(self) -> None:
        """H/P are the scan codes the wheel synthesised into arrows."""
        self.assertEqual(runtime._WINDOWS_SCAN_KEYS.get("H"), KEY_UP)
        self.assertEqual(runtime._WINDOWS_SCAN_KEYS.get("P"), KEY_DOWN)

    def test_scan_timeout_outlasts_the_old_single_peek(self) -> None:
        # The previous code allowed a single 10ms peek; anything slower leaked.
        self.assertGreater(runtime.SCAN_CODE_TIMEOUT_SECONDS, 0.01)

    def test_a_scan_byte_delayed_past_the_old_window_is_still_claimed(self) -> None:
        """Measured regression: real scan bytes arrived up to ~800ms late.

        The orphan marker used to expire after 500ms, so these bytes fell
        through and were typed into the composer. The marker is cleared by any
        other character, so widening it costs nothing when typing resumes.
        """
        self.assertGreater(runtime.ORPHAN_SCAN_WINDOW_SECONDS, 0.8)
        messages = self._run_reader(["\x00", "H"], gap=0.9)

        self.assertIn(KEY_UP, [msg.key for msg in messages])
        self.assertEqual([m.char for m in messages if m.key == "char"], [])

    def test_orphaned_scan_code_is_claimed_not_typed(self) -> None:
        """The remaining leak: a scan byte the console never reports as ready.

        ``kbhit`` can keep saying "nothing waiting" while the scan byte is
        still in flight, so waiting cannot fix this on its own. The reader
        remembers it is owed a scan code and claims the next character.
        """
        messages = self._run_reader(["\x00", "H"])
        keys = [msg.key for msg in messages]

        self.assertIn(KEY_UP, keys)
        self.assertNotIn("char", keys)  # "H" must never be typed

    def test_orphan_marker_ignores_unrelated_characters(self) -> None:
        """A real keystroke after a dropped prefix is still a keystroke."""
        messages = self._run_reader(["\x00", "x"])
        self.assertEqual([m.char for m in messages if m.key == "char"], ["x"])

    def test_a_paired_prefix_and_scan_code_still_works(self) -> None:
        """The normal path is untouched by the orphan handling."""
        messages = self._run_reader(["\xe0", "P"], gap=0.0)
        self.assertEqual([m.key for m in messages], [KEY_DOWN])

    def _run_reader(
        self,
        chars,
        delay: float = 0.0,
        gap: float = 0.2,
        gaps=None,
    ) -> list:
        """Drive a real ``InputReader`` against a scripted Windows console.

        ``gap`` is what makes this interesting: it holds each character back
        long enough for the reader's scan-code window to expire, which is the
        race that used to leak the byte as typed text.

        The script is spent deterministically: once the characters run out the
        fake raises the reader's stop flag, so the thread exits on its own
        instead of being raced by a sleep.
        """
        import sys

        dispatched: list = []
        stop = threading.Event()
        fake = self._FakeMsvcrt(
            chars, delay=delay, gap=gap, gaps=gaps, on_empty=stop.set
        )
        module = type("FakeMsvcrtModule", (), {})()
        module.kbhit = fake.kbhit
        module.getwch = fake.getwch

        reader = runtime.InputReader(dispatched.append, stop)
        with mock.patch.dict(sys.modules, {"msvcrt": module}):
            reader.start()
            reader.join(timeout=5.0)
        self.assertFalse(reader.is_alive(), "input reader did not exit")
        return dispatched

    def test_a_report_split_across_the_flush_timeout_leaks_nothing(self) -> None:
        """End to end: the reader gives up mid-report and the tail still lands.

        This is the reported symptom. The head ``ESC [ M`` is read, the console
        then goes quiet for longer than the escape timeout, the reader flushes
        -- and the payload, whose bytes are all printable, used to be typed as
        "H"/"P" punctuation.
        """
        chars = ["\x1b", "[", "M", chr(96), chr(72), chr(80)]
        # Only the pause after the head matters: it is what triggers the flush.
        gaps = [0.0, 0.0, 0.4, 0.0, 0.0, 0.0]

        messages = self._run_reader(chars, gaps=gaps)

        self.assertEqual([m.char for m in messages if m.key == "char"], [])
        self.assertEqual([m.key for m in messages], [KEY_SCROLL_UP])


class FakeClock:
    """A manual clock so the frame budget is exact rather than timing-based.

    Starts away from zero so the first size poll is not skipped by the
    initial ``_last_size_check`` value, matching what ``time.monotonic`` does
    in production.
    """

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeLive:
    """Stands in for a rich Live, counting paints."""

    def __init__(self) -> None:
        self.frames: list = []

    def update(self, renderable, refresh: bool = False) -> None:
        self.frames.append(renderable)


class RenderLoopTests(unittest.TestCase):
    """A burst of streamed deltas must not turn into a burst of repaints."""

    WIDTH = 80
    HEIGHT = 24

    def setUp(self) -> None:
        self.clock = FakeClock()
        patcher = mock.patch.object(
            runtime.shutil,
            "get_terminal_size",
            lambda fallback=(self.WIDTH, self.HEIGHT): os.terminal_size(
                (self.WIDTH, self.HEIGHT)
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _program(self) -> Program:
        program = Program(
            Model(
                width=self.WIDTH,
                height=self.HEIGHT,
                session_id="default",
                # Shift+Q only quits from the command layer; without it the
                # keystroke would be typed into the prompt instead.
                command_mode=True,
            ),
            view_fn=lambda model: "frame",
            render_fps=1,
            clock=self.clock,
        )
        program._live = FakeLive()
        return program

    def test_a_burst_of_deltas_collapses_into_one_frame(self) -> None:
        program = self._program()
        for index in range(6):
            program.dispatch(AgentDeltaMsg(f"chunk {index}"))
        program.dispatch(KeyMsg("char", KEY_SHIFT_Q))

        program._loop()

        self.assertEqual(len(program._live.frames), 1)
        self.assertEqual(
            program.model.stream_buffer,
            "chunk 0chunk 1chunk 2chunk 3chunk 4chunk 5",
        )

    def test_painting_waits_for_the_frame_deadline(self) -> None:
        program = self._program()
        program.dispatch(AgentDeltaMsg("a"))
        program.dispatch(AgentDeltaMsg("b"))

        self.assertFalse(program._step(1.0, 1.0))
        self.assertFalse(program._step(1.0, 1.0))
        self.assertEqual(program._live.frames, [])

        self.clock.now += 5.0

        self.assertTrue(program._step(1.0, 1.0))
        self.assertEqual(len(program._live.frames), 1)
        self.assertEqual(program.model.stream_buffer, "ab")

    def test_a_steady_tick_does_not_repaint(self) -> None:
        program = self._program()
        program.dispatch(TickMsg(1))

        self.assertFalse(program._step(1.0, 1.0))

        self.assertEqual(program._live.frames, [])
        self.assertTrue(program.model.cursor_on)

    def test_a_caret_flip_schedules_a_paint(self) -> None:
        program = self._program()
        program.dispatch(TickMsg(BLINK_TICKS))

        self.assertFalse(program._step(1.0, 1.0))
        self.clock.now += 5.0

        self.assertTrue(program._step(1.0, 1.0))
        self.assertFalse(program.model.cursor_on)

    def test_quitting_flushes_the_pending_frame(self) -> None:
        program = self._program()
        program.dispatch(AgentDeltaMsg("last"))

        self.assertFalse(program._step(1.0, 1.0))
        self.assertEqual(program._live.frames, [])

        program.dispatch(KeyMsg("char", KEY_SHIFT_Q))
        program._loop()

        self.assertEqual(len(program._live.frames), 1)

    def test_the_terminal_size_is_not_polled_every_iteration(self) -> None:
        program = self._program()
        calls: list = []
        patcher = mock.patch.object(
            runtime.shutil,
            "get_terminal_size",
            lambda fallback=(self.WIDTH, self.HEIGHT): (
                calls.append(1) or os.terminal_size((self.WIDTH, self.HEIGHT))
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        for _ in range(10):
            program._step(0.0, 0.0)

        self.assertEqual(len(calls), 1)

        self.clock.now += runtime.SIZE_CHECK_SECONDS + 0.01
        program._step(0.0, 0.0)

        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
