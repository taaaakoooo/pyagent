"""Terminal runtime: input thread, message queue, and alt-screen rendering."""

from __future__ import annotations

import atexit
import queue
import re
import shutil
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.live import Live

from tui.model import Model
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
    KEY_SPACE,
    KEY_TAB,
    KEY_UP,
    InterruptMsg,
    KeyMsg,
    Msg,
    ResizeMsg,
    TickMsg,
)
from tui.update import Cmd, CmdContext, update

DEFAULT_FPS = 12

# Streaming arrives at roughly 80 deltas/second, which is far more often than
# a terminal can usefully repaint. Cap repaints and let the loop coalesce the
# deltas that land in between.
RENDER_FPS = 30

# How often to poll the terminal size; it is a syscall and cannot change
# faster than the user can drag a window edge.
SIZE_CHECK_SECONDS = 0.5

# How long a half-finished escape sequence may sit before it is abandoned.
# Terminal bytes are not framed: an ESC, its "[" and the rest can land in
# separate reads. Waiting is safe because a held sequence is never typed; this
# only bounds how long an incomplete tail is kept, and how long a lone Esc
# press waits to be recognised.
ESCAPE_SEQUENCE_TIMEOUT_SECONDS = 0.25

# Extended keys arrive as a NUL/E0 prefix followed by a scan code. The pair
# normally arrives together, so this window only has to cover the case where
# the scan byte lands in a later console record. It is deliberately short so a
# real arrow key stays responsive; correctness no longer depends on it, because
# a scan byte that shows up late is claimed by the orphan marker below.
SCAN_CODE_TIMEOUT_SECONDS = 0.05

# How long an orphaned scan code may take to show up before its marker expires.
#
# Measured from a real Cursor terminal on Windows: of 99 dropped scan bytes,
# the delay from the prefix was 0ms at the fastest and 138ms on average, with
# four between 500ms and 800ms and one at 3.2s. 0.5s was therefore too tight
# and let those bytes through as typed "H"/"P". The marker is cleared as soon
# as any other character arrives, so a long window does not eat real typing --
# only a genuinely lost scan byte followed immediately by 'h' or 'p' can.
ORPHAN_SCAN_WINDOW_SECONDS = 2.0

# After an unfinished fixed-length sequence is abandoned, the bytes it is still
# owed keep arriving. They are swallowed rather than typed, but only for this
# long, so a genuinely dead terminal cannot eat later keystrokes.
SEQUENCE_REMAINDER_TIMEOUT_SECONDS = 0.5

ENTER_MOUSE_MODE = "\x1b[?1000h\x1b[?1006h"
EXIT_MOUSE_MODE = "\x1b[?1000l\x1b[?1006l"

# Belt and braces for leaving the terminal usable. A force-killed process
# cannot run cleanup, so these are also sent from an ``atexit`` hook: a stray
# mouse-reporting bit otherwise survives into the next shell and breaks text
# selection until the user resets it by hand.
SHOW_CURSOR = "\x1b[?25h"
RESET_ATTRS = "\x1b[0m"
RESTORE_TERMINAL = EXIT_MOUSE_MODE + SHOW_CURSOR + RESET_ATTRS

# Set to a file path to record every raw input character the reader sees.
# Terminal hosts differ enough in what they send that guessing is unreliable.
INPUT_DEBUG_ENV_VAR = "PYAGENT_INPUT_DEBUG"


def _debug_log_path() -> Optional["Path"]:
    """Resolve the debug log target, or ``None`` when disabled."""
    import os
    from pathlib import Path

    value = os.environ.get(INPUT_DEBUG_ENV_VAR)
    if not value:
        return None
    try:
        return Path(value).expanduser()
    except (OSError, ValueError):
        return None


# SGR mouse reporting: ESC [ <button>;<x>;<y>M  (lowercase "m" on release).
_SGR_MOUSE_RE = re.compile(r"<(\d+);(\d+);(\d+)([Mm])\Z")

# The body of an SGR report is only ever digits and semicolons, so anything
# else means the "report" we are claiming is really user input.
_SGR_PARAMETER_CHARS = frozenset("0123456789;")
_SGR_TERMINATORS = frozenset("Mm")

_ENTER_CHARS = {"\r", "\n"}
_TAB_CHARS = {"\t"}
_BACKSPACE_CHARS = {"\x08", "\x7f"}
_ESC_CHAR = "\x1b"

# Windows console scan codes (after the 0x00 / 0xe0 prefix).
_WINDOWS_SCAN_KEYS = {
    "H": KEY_UP,
    "P": KEY_DOWN,
    "K": KEY_LEFT,
    "M": KEY_RIGHT,
    "G": KEY_HOME,
    "O": KEY_END,
    "I": KEY_PAGEUP,
    "Q": KEY_PAGEDOWN,
    "S": KEY_DELETE,
}

# ANSI sequences produced on POSIX terminals.
_ANSI_KEYS = {
    "A": KEY_UP,
    "B": KEY_DOWN,
    "C": KEY_RIGHT,
    "D": KEY_LEFT,
    "H": KEY_HOME,
    "F": KEY_END,
    "1~": KEY_HOME,
    "2~": "insert",
    "3~": KEY_DELETE,
    "4~": KEY_END,
    "5~": KEY_PAGEUP,
    "6~": KEY_PAGEDOWN,
    "Z": KEY_BACKTAB,
}

# Control characters the app reacts to. Deliberately empty: the TUI avoids
# Ctrl bindings because terminal hosts (Cursor, VS Code, tmux) intercept them
# before the app ever sees the keystroke. Commands use Shift+letter instead,
# which arrives as a plain uppercase character.
_CTRL_LETTERS: Dict[str, str] = {}


def decode_control_char(char: str) -> Optional[KeyMsg]:
    """Map a single control character to a ``KeyMsg`` when recognized."""
    if char in _CTRL_LETTERS:
        return KeyMsg(_CTRL_LETTERS[char])
    if char in _ENTER_CHARS:
        return KeyMsg(KEY_ENTER)
    if char in _TAB_CHARS:
        return KeyMsg(KEY_TAB)
    if char in _BACKSPACE_CHARS:
        return KeyMsg(KEY_BACKSPACE)
    if char == _ESC_CHAR:
        return KeyMsg(KEY_ESC)
    if char == " ":
        return KeyMsg(KEY_SPACE, " ")
    if char.isprintable():
        return KeyMsg("char", char)
    return None


def decode_ansi_sequence(sequence: str) -> Optional[KeyMsg]:
    """Map an ANSI escape sequence body (without the leading ESC) to a key.

    Handles CSI (``ESC [ ...``), SS3 (``ESC O A``, sent for arrows while the
    terminal is in application cursor-key mode) and SGR mouse reporting.
    Returning ``None`` means "recognised but not bound": the caller drops it,
    which is what keeps mouse reports from leaking into the composer as text.
    """
    if sequence.startswith("O"):
        # SS3: application cursor keys, e.g. ESC O A for Up.
        key = _ANSI_KEYS.get(sequence[1:])
        return KeyMsg(key) if key is not None else None

    if not sequence.startswith("["):
        return None

    body = sequence[1:]

    if body.startswith("M") and len(body) >= 4:
        return _decode_x10_mouse(body[1:4])

    mouse = _SGR_MOUSE_RE.fullmatch(body)
    if mouse is not None:
        return _decode_sgr_mouse(
            int(mouse.group(1)),
            int(mouse.group(2)),
            int(mouse.group(3)),
        )

    # Modifier-encoded keys: ESC [ <param> ; <modifier> <final>, e.g. ESC[1;5A
    if ";" in body:
        _, _, rest = body.rpartition(";")
        if len(rest) >= 2 and rest[:-1].isdigit():
            inner = _ANSI_KEYS.get(rest[-1])
            if inner is not None:
                return KeyMsg(f"ctrl+{inner}")
        return None

    key = _ANSI_KEYS.get(body)
    if key is not None:
        return KeyMsg(key)
    return None


def _decode_sgr_mouse(button: int, x: int, y: int) -> Optional[KeyMsg]:
    """Turn a mouse report into a key, or ``None`` for events we do not bind.

    Bit 6 (value 64) marks a wheel event; the low two bits then select the
    direction. Modifier bits are ignored.
    """
    del x, y  # Position is not used yet; only the wheel direction matters.
    if not button & 64:
        # Plain clicks and drags are deliberately unbound for now.
        return None
    direction = button & 0b11
    if direction == 0:
        return KeyMsg(KEY_SCROLL_UP)
    if direction == 1:
        return KeyMsg(KEY_SCROLL_DOWN)
    return None  # Horizontal wheel: nothing to scroll.


def _decode_x10_mouse(raw: str) -> Optional[KeyMsg]:
    """Decode the three raw bytes of an X10 mouse report.

    X10 is the older encoding, reported by a terminal that understands
    ``?1000`` but not ``?1006``. It has no separators, so every byte is the
    real value offset by 32 -- which is why a leaked report reads as ordinary
    punctuation rather than something obviously broken.
    """
    codes = [ord(char) - 32 for char in raw]
    if any(code < 0 for code in codes):
        return None
    return _decode_sgr_mouse(codes[0], codes[1], codes[2])


def sequence_length(text: str) -> Optional[int]:
    """Length of the escape sequence at the start of ``text``.

    ``text`` must start with ESC. Returns ``None`` while the sequence is still
    incomplete, which is the signal to wait for more input rather than guess.

    Getting this length right is what keeps raw bytes out of the composer: a
    sequence whose tail is left behind gets decoded as typed text, and X10
    mouse payload bytes are all printable, so a stray report turns into
    something like ``>``, ``~`` or ``H``.
    """
    if len(text) < 2:
        return None  # a lone Esc, or the start of something longer

    second = text[1]
    if second == "O":
        # SS3: exactly one final byte, e.g. ESC O A for Up.
        return 3 if len(text) >= 3 else None

    if second != "[":
        # ESC followed by a printable char: Alt+key and friends.
        return 2

    index = 2
    # CSI is parameter bytes (0x30-0x3F), then intermediate bytes (0x20-0x2F),
    # then a single final byte (0x40-0x7E).
    while index < len(text) and 0x30 <= ord(text[index]) <= 0x3F:
        index += 1
    while index < len(text) and 0x20 <= ord(text[index]) <= 0x2F:
        index += 1
    if index >= len(text):
        return None
    if not 0x40 <= ord(text[index]) <= 0x7E:
        return None

    end = index + 1
    if index == 2 and text[2] == "M":
        # X10 mouse encoding: "ESC [ M" followed by three raw bytes, which are
        # coordinates offset by 32 and carry no framing of their own.
        end += 3
    return end if len(text) >= end else None


def _report_tail_length(text: str) -> Optional[int]:
    """Bytes still owed for an X10 report whose head is ``text``.

    X10 is fixed-length, so the tail can be counted exactly. ``None`` means
    ``text`` is not an X10 head at all.
    """
    if text.startswith("\x1b[") and text[2:3] == "M":
        return max(0, 6 - len(text))
    return None


class KeyStreamDecoder:
    """Reassemble raw terminal input into key presses.

    Terminal input is a stream of bytes with no message boundaries: a single
    read may hand back all of a sequence, part of one, or several at once. The
    naive approach -- grab whatever is available, then give up -- drops the head
    of a half-arrived sequence and lets its remaining bytes be decoded as typed
    text. That is how a scroll wheel ended up spraying characters into the
    composer.

    So a partial sequence is never emitted as text, and if it looks like the
    head of a mouse report, the tail is *claimed* rather than handed to the
    prompt. A report is recognised by its first three bytes, so a torn one can
    always be finished off instead of leaking its payload.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._buffer = ""
        self._clock = clock
        # Set while the tail of a torn mouse report is being claimed.
        self._report_body = ""
        self._report_missing = 0
        self._report_deadline = 0.0

    @property
    def pending(self) -> str:
        """The not-yet-decodable tail, kept for diagnostics and tests."""
        return self._buffer

    @property
    def expecting(self) -> int:
        """How many bytes of a torn X10 report are still being claimed."""
        return self._report_missing

    def feed(self, text: str) -> List[KeyMsg]:
        """Add raw input, returning whichever key presses it completed."""
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> List[KeyMsg]:
        """Resolve leftovers after the terminal has gone quiet.

        A lone ESC is a real Escape key press. Anything longer is a sequence
        that never finished: it is dropped, and if it was the head of a mouse
        report, the bytes it still owed are claimed so they cannot be typed.
        """
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> List[KeyMsg]:
        messages: List[KeyMsg] = []
        while True:
            if self._report_body:
                if self._clock() > self._report_deadline:
                    self._claim_report_tail()  # stop claiming, move on
                    continue
                claimed = self._claim_report_bytes(messages)
                if claimed is None:
                    return messages  # the rest of the report is still coming
                if not claimed:
                    continue

            if not self._buffer:
                return messages

            if self._buffer.startswith(_ESC_CHAR):
                length = sequence_length(self._buffer)
                if length is None:
                    if not final:
                        return messages  # wait for the rest of the sequence
                    message = self._abandon()
                    if message is not None:
                        messages.append(message)
                    continue
                sequence = self._buffer[1:length]
                self._buffer = self._buffer[length:]
                message = decode_ansi_sequence(sequence)
                if message is not None:
                    messages.append(message)
                continue

            char = self._buffer[0]
            self._buffer = self._buffer[1:]
            message = decode_control_char(char)
            if message is not None:
                messages.append(message)

    def _abandon(self) -> Optional[KeyMsg]:
        """Give up on an incomplete sequence, returning a key if it was one.

        A bare ESC resolves here: waiting is how a lone Escape press is told
        apart from the start of a sequence, and the timeout is what ends the
        wait. Anything longer was cut off mid-flight, and if it began a mouse
        report its payload -- which is printable -- is still on its way.
        """
        abandoned, self._buffer = self._buffer, ""
        if abandoned == _ESC_CHAR:
            return KeyMsg(KEY_ESC)

        missing = _report_tail_length(abandoned)
        if missing is not None:
            self._start_claiming(abandoned[1:], missing)
        elif abandoned.startswith("\x1b[<"):
            # SGR: parameters run to a single "M"/"m" terminator, so the tail
            # can be claimed without knowing how many digits are in it.
            self._start_claiming(abandoned[1:], 0)
        return None

    def _start_claiming(self, body: str, missing: int) -> None:
        self._report_body = body
        self._report_missing = missing
        self._report_deadline = self._clock() + SEQUENCE_REMAINDER_TIMEOUT_SECONDS

    def _claim_report_bytes(self, messages: List[KeyMsg]) -> Optional[bool]:
        """Take the next slice of a torn report's tail.

        Returns ``None`` while more input is needed, ``True`` when the report
        finished, and ``False`` when this pass made no progress and the buffer
        should be handed back for normal parsing.
        """
        if not self._buffer:
            return None

        terminated = False
        if self._report_missing:
            # X10 payload bytes are values offset by 32, so never control
            # characters. One here means we guessed wrong about the head --
            # a real key press -- and it must be handled normally.
            if ord(self._buffer[0]) < 32:
                self._finish_claiming()
                return False
            take = min(self._report_missing, len(self._buffer))
            self._report_body += self._buffer[:take]
            self._buffer = self._buffer[take:]
            self._report_missing -= take
            terminated = self._report_missing == 0
        else:
            # SGR: consume the parameter body, stopping at its terminator.
            index = 0
            while index < len(self._buffer):
                char = self._buffer[index]
                if char in _SGR_TERMINATORS:
                    index += 1
                    terminated = True
                    break
                if char not in _SGR_PARAMETER_CHARS:
                    break  # not a report after all
                index += 1
            if index == 0:
                self._finish_claiming()
                return False
            self._report_body += self._buffer[:index]
            self._buffer = self._buffer[index:]

        if not terminated:
            self._report_deadline = (
                self._clock() + SEQUENCE_REMAINDER_TIMEOUT_SECONDS
            )
            return None

        body = self._report_body
        self._finish_claiming()
        message = decode_ansi_sequence(body)
        if message is not None:
            messages.append(message)
        return True

    def _finish_claiming(self) -> None:
        self._report_body = ""
        self._report_missing = 0

    def _claim_report_tail(self) -> None:
        """Stop claiming: the tail never arrived, so drop what we have."""
        self._finish_claiming()


class InputReader(threading.Thread):
    """Daemon thread that turns raw terminal input into dispatched messages."""

    def __init__(
        self,
        dispatch: Callable[[Msg], None],
        stop_event: threading.Event,
        stream=None,
    ) -> None:
        super().__init__(name="tui-input", daemon=True)
        self._dispatch = dispatch
        self._stop_event = stop_event
        self._stream = stream if stream is not None else sys.stdin
        self._restore_terminal: Optional[Callable[[], None]] = None
        self._debug_path = _debug_log_path()

    def run(self) -> None:  # pragma: no cover - requires a real terminal
        try:
            if sys.platform == "win32":
                self._run_windows()
            else:
                self._run_posix()
        finally:
            if self._restore_terminal is not None:
                try:
                    self._restore_terminal()
                except Exception:  # noqa: BLE001 - best effort restore
                    pass

    def _run_windows(self) -> None:  # pragma: no cover - requires Windows TTY
        import msvcrt

        decoder = KeyStreamDecoder()
        flush_at: Optional[float] = None
        # Set when a NUL/E0 prefix arrived without its scan code. That code is
        # still in flight, so the very next character is it -- reading it as
        # typing is what put "H"/"P" in the composer, and no amount of waiting
        # is guaranteed to help if the console reports the pair in two records.
        orphan_deadline: Optional[float] = None

        while not self._stop_event.is_set():
            if not msvcrt.kbhit():
                now = time.monotonic()
                if flush_at is not None and now >= flush_at:
                    self._log_flush(decoder)
                    self._dispatch_keys(decoder.flush())
                    flush_at = None
                if orphan_deadline is not None and now >= orphan_deadline:
                    orphan_deadline = None
                time.sleep(0.005)
                continue

            char = msvcrt.getwch()
            self._debug_log(char)

            if char in ("\x00", "\xe0"):
                scan = self._windows_read_char(SCAN_CODE_TIMEOUT_SECONDS, msvcrt)
                self._debug_log(f"<scan={scan!r}>")
                orphan_deadline = None
                if scan:
                    key = _WINDOWS_SCAN_KEYS.get(scan.upper())
                    if key is not None:
                        self._dispatch(KeyMsg(key))
                else:
                    orphan_deadline = (
                        time.monotonic() + ORPHAN_SCAN_WINDOW_SECONDS
                    )
                continue

            if orphan_deadline is not None:
                orphan_deadline = None
                key = _WINDOWS_SCAN_KEYS.get(char.upper())
                if key is not None:
                    self._debug_log(f"<orphan scan {char!r}>")
                    self._dispatch(KeyMsg(key))
                    continue

            self._dispatch_keys(decoder.feed(char))
            # While a sequence is still open, keep extending its grace period;
            # it is only abandoned once the terminal really stops talking.
            flush_at = (
                time.monotonic() + ESCAPE_SEQUENCE_TIMEOUT_SECONDS
                if decoder.pending
                else None
            )

    @staticmethod
    def _windows_read_char(timeout: float, msvcrt) -> Optional[str]:  # pragma: no cover
        """Read one character from the console, giving up after ``timeout``."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            if msvcrt.kbhit():
                return msvcrt.getwch()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)

    def _run_posix(self) -> None:  # pragma: no cover - requires POSIX TTY
        import select
        import termios
        import tty

        fd = self._stream.fileno()
        original = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._restore_terminal = lambda: termios.tcsetattr(
            fd, termios.TCSADRAIN, original
        )

        decoder = KeyStreamDecoder()
        flush_at: Optional[float] = None

        while not self._stop_event.is_set():
            if flush_at is not None and time.monotonic() >= flush_at:
                self._log_flush(decoder)
                self._dispatch_keys(decoder.flush())
                flush_at = None

            timeout = 0.05
            if flush_at is not None:
                timeout = max(0.0, min(timeout, flush_at - time.monotonic()))
            ready, _, _ = select.select([self._stream], [], [], timeout)
            if not ready:
                continue

            char = self._stream.read(1)
            self._debug_log(char)
            if not char:
                continue

            self._dispatch_keys(decoder.feed(char))
            flush_at = (
                time.monotonic() + ESCAPE_SEQUENCE_TIMEOUT_SECONDS
                if decoder.pending
                else None
            )

    def _dispatch_keys(self, messages: List[KeyMsg]) -> None:
        for message in messages:
            self._dispatch(message)

    def _log_flush(self, decoder: "KeyStreamDecoder") -> None:
        """Record what was stuck when the terminal went quiet.

        A report that had to be finished off from its head is exactly what a
        leaked "H"/"P" used to look like, so the pending text is worth keeping.
        """
        self._debug_log(
            f"<flush pending={decoder.pending!r} expecting={decoder.expecting}>"
        )

    def _debug_log(self, text: str) -> None:
        """Append raw input to a log when ``PYAGENT_INPUT_DEBUG`` is set.

        Terminal behaviour varies enough between hosts that guessing at the
        byte stream is unproductive; this records it verbatim instead.
        """
        if self._debug_path is None:
            return
        try:
            with self._debug_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{time.monotonic():.3f} {text!r}\n")
        except OSError:  # diagnostics must never break input
            pass


class Program:
    """Drives the Elm loop against a real terminal."""

    def __init__(
        self,
        model: Model,
        commands: Optional[CmdContext] = None,
        update_fn: Callable[[Model, Msg], Tuple[Model, Optional[Cmd]]] = update,
        view_fn: Optional[Callable[[Model], object]] = None,
        fps: int = DEFAULT_FPS,
        console: Optional[Console] = None,
        render_fps: int = RENDER_FPS,
        clock: Callable[[], float] = time.monotonic,
        mouse: bool = True,
    ) -> None:
        self.model = model
        self.commands = commands or CmdContext()
        self._update = update_fn
        self._view = view_fn or _default_view
        self.fps = max(1, fps)
        self.render_fps = max(1, render_fps)
        self._clock = clock
        self.mouse = mouse
        self.console = console or Console()
        self._queue: "queue.Queue[Msg]" = queue.Queue()
        self._stop_event = threading.Event()
        self._reader: Optional[InputReader] = None
        self._live: Optional[Live] = None
        self._tick = 0
        self._dirty = False
        self._last_render = self._clock()
        self._last_size_check = 0.0
        self.dispatched: List[Msg] = []
        self._saved_signals: List[Tuple[int, Any]] = []
        self._interrupt_requested = False
        self._restored = False

    # -- public API -----------------------------------------------------

    def dispatch(self, msg: Msg) -> None:
        """Thread-safe message injection (used by agent callbacks)."""
        self._queue.put(msg)

    def run(self) -> None:
        """Block until the user quits."""
        self._stop_event.clear()
        self._interrupt_requested = False
        self._restored = False
        self._reader = InputReader(self.dispatch, self._stop_event)
        self._reader.start()

        live = Live(
            self._view(self.model),
            console=self.console,
            screen=True,
            auto_refresh=False,
            vertical_overflow="crop",
        )
        self._install_signal_handlers()
        at_register = None
        try:
            # If the process dies without unwinding -- an unhandled exception
            # on a worker thread, or a kill that skips ``finally`` -- the
            # terminal would be left in mouse-reporting mode. This is the last
            # line of defence for that.
            at_register = atexit.register(self._restore_terminal)

            # Ask the terminal to report the wheel as mouse events. Without
            # this, terminals translate a scroll into Up/Down arrow keys, which
            # the composer cannot tell apart from real navigation keys. Some
            # hosts (or a user preference for plain drag-to-select) prefer it
            # off, hence the switch.
            if self.mouse:
                self._write_control(ENTER_MOUSE_MODE)
            with live:
                self._live = live
                self._render()
                self._loop()
        finally:
            self._live = None
            self._restore_terminal()
            self._restore_signal_handlers()
            if at_register is not None:
                atexit.unregister(at_register)
            self._stop_event.set()
            if self._reader is not None:
                self._reader.join(timeout=1.0)
            self.model.should_quit = True

    def _restore_terminal(self) -> None:
        """Leave the terminal as we found it. Safe to call more than once."""
        if self._restored:
            return
        self._restored = True
        if self.mouse:
            self._write_control(EXIT_MOUSE_MODE)
        # Cursor visibility and attributes are cheap to reset and were the two
        # things a force-quit most visibly left behind.
        self._write_control(SHOW_CURSOR + RESET_ATTRS)

    def _install_signal_handlers(self) -> None:
        """Turn Ctrl+C into an in-app interrupt instead of a hard kill.

        Only the main thread may install handlers, and a test may run the loop
        elsewhere, so this quietly does nothing when it cannot.
        """
        if threading.current_thread() is not threading.main_thread():
            return

        signals = [signal.SIGINT, signal.SIGTERM]
        sigbreak = getattr(signal, "SIGBREAK", None)  # Windows Ctrl+Break
        if sigbreak is not None:
            signals.append(sigbreak)

        for signum in signals:
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, self._on_signal)
            except (OSError, ValueError, RuntimeError):
                continue
            self._saved_signals.append((signum, previous))

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self._saved_signals:
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError, RuntimeError):
                continue
        self._saved_signals.clear()

    def _on_signal(self, signum: int, frame: Any) -> None:
        """Request a graceful stop; a second signal stops being polite.

        Deliberately only flips a flag. A handler runs at an arbitrary
        bytecode boundary -- including inside the message queue's own lock --
        so touching the queue here could deadlock the very interrupt that is
        supposed to rescue the user.

        Without the escape hatch a stuck shutdown -- a worker that will not
        join, a blocking MCP read -- would make Ctrl+C useless, which is worse
        than the abrupt exit this replaces.
        """
        if self._interrupt_requested:
            self._restore_signal_handlers()
            raise KeyboardInterrupt
        self._interrupt_requested = True

    def _write_control(self, sequence: str) -> None:
        """Write a raw control sequence, ignoring anything that goes wrong."""
        stream = getattr(self.console, "file", None)
        isatty = getattr(stream, "isatty", None)
        if stream is None or not callable(isatty) or not isatty():
            return
        try:
            stream.write(sequence)
            stream.flush()
        except (OSError, ValueError):  # detached or closing stream
            pass

    # -- internals ------------------------------------------------------

    def _loop(self) -> None:
        frame_interval = 1.0 / self.render_fps
        tick_interval = 1.0 / self.fps
        self._last_render = self._clock()

        while not self.model.should_quit:
            if self._interrupt_requested:
                # Applied here rather than from the handler so the interrupt
                # takes the exact same update+Cmd path as Shift+C.
                self._interrupt_requested = False
                self._apply(InterruptMsg())
            self._step(frame_interval, tick_interval)

        # The batching above can hold the last message past the quit, so make
        # sure the final state still reaches the screen.
        if self._dirty:
            self._render()
            self._dirty = False

    def _step(self, frame_interval: float, tick_interval: float) -> bool:
        """Run one loop iteration; return True when a frame was drawn.

        Messages are drained one at a time but painted at most once per
        ``frame_interval``, so a burst of streamed deltas collapses into a
        single repaint instead of one repaint per delta.
        """
        self._maybe_apply_terminal_size()

        until_frame = frame_interval - (self._clock() - self._last_render)

        if self._dirty:
            if until_frame > 0:
                # Something changed but the frame budget is spent; wait out
                # the remainder so we repaint promptly, not on the next burst.
                timeout = until_frame
            else:
                self._render()
                self._last_render = self._clock()
                self._dirty = False
                return True
        else:
            # Idle: wake at least once per tick so the caret keeps blinking.
            timeout = max(until_frame, tick_interval)

        try:
            msg = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            self._tick += 1
            msg = TickMsg(self._tick)

        if self._apply(msg):
            self._dirty = True
        return False

    def _apply(self, msg: Msg) -> bool:
        """Fold one message into the model; return True when state changed."""
        self.dispatched.append(msg)
        caret_before = self.model.cursor_on
        cmd: Optional[Cmd] = None
        self.model, cmd = self._update(self.model, msg)

        if cmd is not None:
            try:
                cmd(self.commands)
            except Exception as exc:  # noqa: BLE001 - command errors must not kill the UI
                self.dispatch(_command_error(exc))

        if isinstance(msg, TickMsg):
            # Ticks arrive constantly; only a caret flip is worth a repaint.
            return self.model.cursor_on != caret_before
        return True

    def _maybe_apply_terminal_size(self) -> None:
        now = self._clock()
        if now - self._last_size_check < SIZE_CHECK_SECONDS:
            return
        self._last_size_check = now
        size = shutil.get_terminal_size((self.model.width, self.model.height))
        if size.columns != self.model.width or size.lines != self.model.height:
            self.dispatch(ResizeMsg(size.columns, size.lines))

    def _render(self) -> None:
        if self._live is None:
            return
        self._live.update(self._view(self.model), refresh=True)


def _default_view(model: Model):  # pragma: no cover - import shim
    from tui.view import view as rich_view

    return rich_view(model)


def _command_error(exc: Exception) -> Msg:
    from tui.msgs import AgentErrorMsg

    return AgentErrorMsg(f"{type(exc).__name__}: {exc}")


__all__ = [
    "DEFAULT_FPS",
    "ENTER_MOUSE_MODE",
    "ESCAPE_SEQUENCE_TIMEOUT_SECONDS",
    "EXIT_MOUSE_MODE",
    "INPUT_DEBUG_ENV_VAR",
    "ORPHAN_SCAN_WINDOW_SECONDS",
    "RENDER_FPS",
    "RESET_ATTRS",
    "RESTORE_TERMINAL",
    "SCAN_CODE_TIMEOUT_SECONDS",
    "SEQUENCE_REMAINDER_TIMEOUT_SECONDS",
    "SHOW_CURSOR",
    "SIZE_CHECK_SECONDS",
    "InputReader",
    "KeyStreamDecoder",
    "Program",
    "decode_ansi_sequence",
    "decode_control_char",
    "sequence_length",
]
