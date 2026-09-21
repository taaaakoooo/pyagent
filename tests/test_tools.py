"""Tests for the local tool registry and executors."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from internal.tools.definitions import TOOL_NAMES
from internal.tools.path import PathSandboxError
from internal.tools.tools import ToolRegistry, create_default_tool_registry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.tool_output = self.workspace / ".tool_outputs"
        self.registry = create_default_tool_registry(
            workspace_root=self.workspace,
            tool_output_path=self.tool_output,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_create_default_registry_registers_all_tools(self) -> None:
        names = [tool.name for tool in self.registry.list_tools()]
        self.assertEqual(names, list(TOOL_NAMES))
        self.assertEqual(len(names), 7)

    def test_read_file_and_write_file_round_trip(self) -> None:
        write_result = self.registry.call_tool(
            "write_file",
            {"path": "notes.txt", "content": "hello\nworld\n"},
        )
        self.assertFalse(write_result.is_error)

        read_result = self.registry.call_tool(
            "read_file",
            {"path": "notes.txt", "offset": 2, "limit": 1},
        )
        self.assertFalse(read_result.is_error)
        self.assertIn("2|world", str(read_result.content))

    def test_replace_line(self) -> None:
        target = self.workspace / "sample.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

        result = self.registry.call_tool(
            "replace_line",
            {"path": "sample.txt", "line_number": 2, "new_content": "BETA"},
        )
        self.assertFalse(result.is_error)
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "alpha\nBETA\ngamma\n",
        )

    def test_list_dir(self) -> None:
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (self.workspace / "README.md").write_text("# demo\n", encoding="utf-8")

        flat = self.registry.call_tool("list_dir", {"path": "."})
        self.assertFalse(flat.is_error)
        self.assertIn("README.md", str(flat.content))
        self.assertIn("src/", str(flat.content))

        recursive = self.registry.call_tool("list_dir", {"path": ".", "recursive": True})
        self.assertFalse(recursive.is_error)
        self.assertIn("src/main.py", str(recursive.content))

    def test_read_regex(self) -> None:
        (self.workspace / "data.log").write_text(
            "info start\nerror boom\ninfo end\n",
            encoding="utf-8",
        )

        result = self.registry.call_tool(
            "read_regex",
            {"path": "data.log", "pattern": r"error (.+)"},
        )
        self.assertFalse(result.is_error)
        payload = json.loads(str(result.content))
        self.assertEqual(payload["match_count"], 1)
        self.assertEqual(payload["matches"][0]["line"], 2)
        self.assertEqual(payload["matches"][0]["groups"], ["boom"])

    def test_run_shell(self) -> None:
        result = self.registry.call_tool(
            "run_shell",
            {"command": "echo hello", "timeout_seconds": 10},
        )
        self.assertFalse(result.is_error)
        payload = json.loads(str(result.content))
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("hello", payload["stdout"])

    def test_path_escape_is_rejected(self) -> None:
        result = self.registry.call_tool("read_file", {"path": "../outside.txt"})
        self.assertTrue(result.is_error)
        self.assertIsInstance(result.content, str)
        self.assertIn("outside allowed root", str(result.content))

    def test_read_disk_data_by_tool_call_id(self) -> None:
        results_dir = self.tool_output / "tool_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        archived = results_dir / "echo_call123.txt"
        archived.write_text("large tool output\nline two\n", encoding="utf-8")

        result = self.registry.call_tool(
            "read_disk_data",
            {"tool_call_id": "call123", "offset": 2, "limit": 1},
        )
        self.assertFalse(result.is_error)
        self.assertIn("2|line two", str(result.content))

    def test_read_disk_data_lists_recent_files(self) -> None:
        results_dir = self.tool_output / "tool_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "grep_abc.txt").write_text("x", encoding="utf-8")

        result = self.registry.call_tool("read_disk_data", {})
        self.assertFalse(result.is_error)
        payload = json.loads(str(result.content))
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["entries"][0]["tool_call_id"], "abc")

    def test_unknown_tool_returns_error(self) -> None:
        registry = ToolRegistry()
        result = registry.call_tool("missing", {})
        self.assertTrue(result.is_error)


class PathSandboxTests(unittest.TestCase):
    def test_resolve_workspace_path_blocks_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with self.assertRaises(PathSandboxError):
                self.registry_call_resolve(workspace, "/etc/passwd")

    def registry_call_resolve(self, workspace: Path, relative: str) -> None:
        registry = create_default_tool_registry(workspace, workspace / ".tool_outputs")
        result = registry.call_tool("read_file", {"path": relative})
        if result.is_error:
            raise PathSandboxError(str(result.content))


if __name__ == "__main__":
    unittest.main()
