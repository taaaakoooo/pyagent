"""Tests for workspace path sandboxing."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from internal.tools.path import (
    PathSandboxError,
    ensure_tool_output_in_workspace,
    resolve_safe_path,
    resolve_workspace_path,
    resolve_workspace_path_info,
    sanitize_path_input,
)
from internal.tools.tools import create_default_tool_registry


class SanitizePathInputTests(unittest.TestCase):
    def test_strips_whitespace(self) -> None:
        self.assertEqual(sanitize_path_input("  src/a.py  "), "src/a.py")

    def test_rejects_empty_path(self) -> None:
        with self.assertRaises(PathSandboxError):
            sanitize_path_input("   ")

    def test_rejects_null_bytes(self) -> None:
        with self.assertRaises(PathSandboxError):
            sanitize_path_input("src\0/a.py")

    def test_allows_dot(self) -> None:
        self.assertEqual(sanitize_path_input("."), ".")


class ResolveSafePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "a.py").write_text("print('ok')\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_relative_path_returns_absolute_under_root(self) -> None:
        resolved = resolve_safe_path(self.workspace, "src/a.py")

        self.assertTrue(resolved.absolute.is_absolute())
        self.assertEqual(resolved.relative, "src/a.py")
        self.assertEqual(resolved.absolute, self.workspace / "src" / "a.py")

    def test_dot_resolves_to_workspace_root(self) -> None:
        resolved = resolve_safe_path(self.workspace, ".")

        self.assertEqual(resolved.absolute, self.workspace)
        self.assertEqual(resolved.relative, ".")

    def test_normalizes_parent_segments(self) -> None:
        resolved = resolve_safe_path(self.workspace, "src/../src/a.py")

        self.assertEqual(resolved.relative, "src/a.py")
        self.assertEqual(resolved.absolute, self.workspace / "src" / "a.py")

    def test_rejects_escape_with_parent_segments(self) -> None:
        with self.assertRaises(PathSandboxError):
            resolve_safe_path(self.workspace, "../outside.txt")

    def test_allows_absolute_path_inside_workspace(self) -> None:
        absolute = str((self.workspace / "src" / "a.py").resolve())
        resolved = resolve_safe_path(self.workspace, absolute)

        self.assertEqual(resolved.absolute, self.workspace / "src" / "a.py")
        self.assertEqual(resolved.relative, "src/a.py")

    def test_rejects_absolute_path_outside_workspace(self) -> None:
        outside = str((self.workspace.parent / "outside.txt").resolve())
        with self.assertRaises(PathSandboxError):
            resolve_safe_path(self.workspace, outside)

    def test_resolve_workspace_path_wrapper(self) -> None:
        absolute = resolve_workspace_path(self.workspace, "src/a.py")
        self.assertEqual(absolute, self.workspace / "src" / "a.py")

    def test_resolve_workspace_path_info_matches_wrapper(self) -> None:
        info = resolve_workspace_path_info(self.workspace, "src/a.py")
        self.assertEqual(info.absolute, resolve_workspace_path(self.workspace, "src/a.py"))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink not supported")
    def test_rejects_symlink_pointing_outside_workspace(self) -> None:
        outside_dir = self.workspace.parent / "outside_link_target"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret\n", encoding="utf-8")

        link_path = self.workspace / "link.txt"
        try:
            os.symlink(outside_file, link_path)
        except OSError:
            self.skipTest("unable to create symlink in this environment")

        with self.assertRaises(PathSandboxError):
            resolve_safe_path(self.workspace, "link.txt")


@unittest.skipUnless(sys.platform == "win32", "Windows-specific path checks")
class WindowsPathSandboxTests(unittest.TestCase):
    def test_rejects_unc_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            with self.assertRaises(PathSandboxError):
                resolve_safe_path(workspace, r"\\server\share\file.txt")


class ToolOutputGuardTests(unittest.TestCase):
    def test_rejects_tool_output_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as workspace_tmp:
            with tempfile.TemporaryDirectory() as outside_tmp:
                workspace = Path(workspace_tmp).resolve()
                outside = Path(outside_tmp).resolve()
                with self.assertRaises(PathSandboxError):
                    create_default_tool_registry(workspace, outside)

    def test_ensure_tool_output_in_workspace_allows_nested_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            tool_output = workspace / ".tool_outputs"
            ensure_tool_output_in_workspace(workspace, tool_output)


class ExecutorPathIntegrationTests(unittest.TestCase):
    def test_read_file_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            registry = create_default_tool_registry(workspace, workspace / ".tool_outputs")
            result = registry.call_tool("read_file", {"path": "../outside.txt"})
            self.assertTrue(result.is_error)
            self.assertIn("outside allowed root", str(result.content))


if __name__ == "__main__":
    unittest.main()
