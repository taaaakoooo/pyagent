"""Tests for session id validation, listing, and deletion."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from internal.session.session import (
    SESSION_ID_MAX_LENGTH,
    SessionError,
    SessionStore,
    generate_session_id,
    validate_session_id,
)
from internal.types.types import Message


class ValidateSessionIdTests(unittest.TestCase):
    def test_accepts_safe_ids(self) -> None:
        for session_id in ("work", "refactor-2", "a.b", "Session_01", "  work  "):
            with self.subTest(session_id=session_id):
                self.assertEqual(
                    validate_session_id(session_id),
                    session_id.strip(),
                )

    def test_rejects_unsafe_ids(self) -> None:
        bad_ids = (
            "",
            "   ",
            ".",
            "..",
            "../x",
            "a/b",
            "a\\b",
            "-lead",
            ".lead",
            "_lead",
            "with space",
            "tab\tid",
            "new\nline",
            "a" * (SESSION_ID_MAX_LENGTH + 1),
        )
        for session_id in bad_ids:
            with self.subTest(session_id=session_id):
                with self.assertRaises(SessionError):
                    validate_session_id(session_id)

    def test_rejects_control_characters(self) -> None:
        with self.assertRaises(SessionError):
            validate_session_id("bad\x00id")

    def test_rejects_non_string(self) -> None:
        with self.assertRaises(SessionError):
            validate_session_id(None)  # type: ignore[arg-type]

    def test_max_length_boundary_is_accepted(self) -> None:
        session_id = "a" * SESSION_ID_MAX_LENGTH
        self.assertEqual(validate_session_id(session_id), session_id)


class GenerateSessionIdTests(unittest.TestCase):
    def test_uses_injected_timestamp(self) -> None:
        moment = datetime(2026, 9, 21, 18, 27, 5)
        self.assertEqual(generate_session_id(moment), "session-20260921-182705")

    def test_generated_id_is_valid(self) -> None:
        session_id = generate_session_id()
        self.assertEqual(validate_session_id(session_id), session_id)

    def test_seconds_resolution(self) -> None:
        base = datetime(2026, 1, 2, 3, 4, 5)
        self.assertEqual(
            generate_session_id(base),
            generate_session_id(base + timedelta(microseconds=1)),
        )


class SessionStoreIdTests(unittest.TestCase):
    def test_invalid_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SessionError):
                SessionStore(Path(tmp), "../escape")

    def test_escape_attempt_does_not_leave_storage_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaises(SessionError):
                SessionStore(base, "../../outside")
            self.assertEqual(list(base.glob("*.jsonl")), [])


class SessionListTests(unittest.TestCase):
    def test_missing_directory_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            self.assertEqual(SessionStore.list_sessions(missing), [])

    def test_ignores_non_jsonl_and_invalid_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = SessionStore(base, "work")
            store.append_message(Message(role="user", content="hello"))
            (base / "notes.txt").write_text("ignored", encoding="utf-8")
            (base / ".hidden.jsonl").write_text("", encoding="utf-8")

            summaries = SessionStore.list_sessions(base)

        self.assertEqual([summary.session_id for summary in summaries], ["work"])

    def test_message_count_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = SessionStore(base, "work")
            store.append_message(Message(role="system", content="system prompt"))
            store.append_message(Message(role="user", content="please refactor the parser"))
            store.append_message(Message(role="assistant", content="done"))

            summary = SessionStore.list_sessions(base)[0]

        self.assertEqual(summary.message_count, 3)
        self.assertEqual(summary.preview, "please refactor the parser")

    def test_preview_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = SessionStore(base, "long")
            store.append_message(Message(role="user", content="x" * 200))

            summary = SessionStore.list_sessions(base)[0]

        self.assertTrue(summary.preview.endswith("..."))
        self.assertLessEqual(len(summary.preview), 60)

    def test_sorted_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            older = SessionStore(base, "older")
            older.append_message(Message(role="user", content="old"))
            newer = SessionStore(base, "newer")
            newer.append_message(Message(role="user", content="new"))

            past = time.time() - 3600
            os.utime(older.session_file, (past, past))

            summaries = SessionStore.list_sessions(base)

        self.assertEqual(
            [summary.session_id for summary in summaries],
            ["newer", "older"],
        )

    def test_corrupted_lines_are_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            store = SessionStore(base, "work")
            store.append_message(Message(role="user", content="kept"))
            with store.session_file.open("a", encoding="utf-8") as handle:
                handle.write("{not json}\n")

            summary = SessionStore.list_sessions(base)[0]

        self.assertEqual(summary.message_count, 1)
        self.assertEqual(summary.preview, "kept")

    def test_updated_at_is_timezone_aware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            SessionStore(base, "work").append_message(
                Message(role="user", content="hi")
            )
            summary = SessionStore.list_sessions(base)[0]

        self.assertIsNotNone(summary.updated_at.tzinfo)


class SessionDeleteTests(unittest.TestCase):
    def test_delete_removes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp), "work")
            store.append_message(Message(role="user", content="hi"))

            self.assertTrue(store.delete())
            self.assertFalse(store.session_file.exists())
            self.assertFalse(store.delete())

    def test_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp), "work")
            self.assertTrue(store.is_empty())

            store.append_event({"type": "compact", "turn": 1})
            self.assertTrue(store.is_empty())

            store.append_message(Message(role="user", content="hi"))
            self.assertFalse(store.is_empty())


if __name__ == "__main__":
    unittest.main()
