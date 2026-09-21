"""Tests for diff models and tool result metadata."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from internal.diff.diff import build_file_diff
from internal.tools.tools import create_default_tool_registry


class DiffModelTests(unittest.TestCase):
    def test_build_file_diff_replace_line(self) -> None:
        file_diff = build_file_diff(
            "sample.txt",
            "replace_line",
            "alpha\nbeta\ngamma\n",
            "alpha\nBETA\ngamma\n",
        )

        self.assertEqual(file_diff.path, "sample.txt")
        self.assertEqual(file_diff.operation, "replace_line")
        self.assertIn("-beta", file_diff.unified)
        self.assertIn("+BETA", file_diff.unified)
        kinds = [line.kind for line in file_diff.lines]
        self.assertIn("remove", kinds)
        self.assertIn("add", kinds)


class ToolDiffMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = create_default_tool_registry(
            workspace_root=self.workspace,
            tool_output_path=self.workspace / ".tool_outputs",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_write_file_attaches_file_diff_metadata(self) -> None:
        result = self.registry.call_tool(
            "write_file",
            {"path": "new.txt", "content": "hello\n"},
        )

        self.assertFalse(result.is_error)
        self.assertEqual(result.content, "Wrote 6 character(s) to new.txt")
        self.assertIn("summary", result.metadata)
        self.assertIn("file_diff", result.metadata)
        self.assertEqual(result.metadata["file_diff"]["operation"], "write")
        self.assertEqual(result.metadata["file_diff"]["new_content"], "hello\n")

    def test_replace_line_attaches_file_diff_metadata(self) -> None:
        target = self.workspace / "sample.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

        result = self.registry.call_tool(
            "replace_line",
            {"path": "sample.txt", "line_number": 2, "new_content": "BETA"},
        )

        self.assertFalse(result.is_error)
        self.assertIn("file_diff", result.metadata)
        self.assertEqual(result.metadata["file_diff"]["operation"], "replace_line")
        self.assertIn("BETA", result.metadata["file_diff"]["new_content"])

    def test_read_file_metadata_contains_summary_only(self) -> None:
        (self.workspace / "notes.txt").write_text("hello\n", encoding="utf-8")

        result = self.registry.call_tool("read_file", {"path": "notes.txt"})

        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["summary"], result.content)
        self.assertNotIn("file_diff", result.metadata)


if __name__ == "__main__":
    unittest.main()
