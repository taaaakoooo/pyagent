"""Tests for permission approval flow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from internal.permission.permission import (
    ConversationPermissionManager,
    analyze_shell_risk,
    build_preview_diff,
    enrich_permission_request,
)
from internal.types.types import PermissionDecision, PermissionRequest, ToolDefinition


class ShellRiskTests(unittest.TestCase):
    def test_detects_destructive_delete(self) -> None:
        tags, description = analyze_shell_risk("rm -rf /tmp/demo")
        self.assertIn("destructive_delete", tags)
        self.assertIn("delete", description.lower())

    def test_safe_command_has_default_description(self) -> None:
        tags, description = analyze_shell_risk("echo hello")
        self.assertEqual(tags, [])
        self.assertIn("Shell command", description)


class PermissionFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ConversationPermissionManager()

    def test_read_only_tool_auto_allowed_without_callback(self) -> None:
        request = PermissionRequest(request_id="1", tool_name="read_file", arguments={"path": "a.py"})
        self.assertTrue(self.manager.resolve(request, callback=None, workspace_root=Path(".")))

    def test_write_tool_denied_without_callback(self) -> None:
        request = PermissionRequest(request_id="2", tool_name="write_file", arguments={"path": "a.py", "content": "x"})
        self.assertFalse(self.manager.resolve(request, callback=None, workspace_root=Path(".")))

    def test_mcp_tool_requires_callback(self) -> None:
        mcp_tool = ToolDefinition(
            name="remote_read",
            description="remote",
            input_schema={"type": "object", "properties": {}},
            source="mcp",
            server_name="demo",
        )
        request = PermissionRequest(request_id="3", tool_name="remote_read", arguments={})
        self.assertFalse(
            self.manager.resolve(
                request,
                callback=None,
                tool_definition=mcp_tool,
                workspace_root=Path("."),
            )
        )

    def test_allow_once_via_callback(self) -> None:
        request = PermissionRequest(request_id="4", tool_name="run_shell", arguments={"command": "echo hi"})

        def approve_once(_: PermissionRequest) -> PermissionDecision:
            return PermissionDecision.ALLOW_ONCE

        self.assertTrue(
            self.manager.resolve(request, callback=approve_once, workspace_root=Path("."))
        )

    def test_deny_session_blocks_later_calls(self) -> None:
        request = PermissionRequest(request_id="5", tool_name="run_shell", arguments={"command": "echo hi"})

        def deny_session(_: PermissionRequest) -> PermissionDecision:
            return PermissionDecision.DENY_SESSION

        self.assertFalse(self.manager.resolve(request, callback=deny_session, workspace_root=Path(".")))
        self.assertFalse(self.manager.resolve(request, callback=deny_session, workspace_root=Path(".")))

    def test_allow_session_skips_later_prompts(self) -> None:
        request = PermissionRequest(request_id="6", tool_name="run_shell", arguments={"command": "echo hi"})
        calls = {"count": 0}

        def allow_session(_: PermissionRequest) -> PermissionDecision:
            calls["count"] += 1
            return PermissionDecision.ALLOW_SESSION

        self.assertTrue(self.manager.resolve(request, callback=allow_session, workspace_root=Path(".")))
        self.assertTrue(self.manager.resolve(request, callback=None, workspace_root=Path(".")))
        self.assertEqual(calls["count"], 1)


class PermissionEnrichmentTests(unittest.TestCase):
    def test_shell_request_includes_risk_tags(self) -> None:
        request = PermissionRequest(
            request_id="7",
            tool_name="run_shell",
            arguments={"command": "sudo rm -rf /"},
        )
        enriched = enrich_permission_request(request, tool_definition=None, workspace_root=Path("."))
        self.assertIn("elevated_privilege", enriched.risk_tags)
        self.assertIn("destructive_delete", enriched.risk_tags)

    def test_write_file_preview_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            target = workspace / "demo.txt"
            target.write_text("old line\n", encoding="utf-8")

            preview = build_preview_diff(
                "write_file",
                {"path": "demo.txt", "content": "new line\n"},
                workspace,
            )
            self.assertIsNotNone(preview)
            assert preview is not None
            self.assertIn("-old line", preview["unified"])
            self.assertIn("+new line", preview["unified"])

    def test_mcp_request_marked_high_risk(self) -> None:
        tool = ToolDefinition(
            name="search",
            description="search",
            input_schema={"type": "object", "properties": {}},
            source="mcp",
            server_name="docs",
        )
        request = PermissionRequest(request_id="8", tool_name="search", arguments={"q": "hello"})
        enriched = enrich_permission_request(request, tool_definition=tool, workspace_root=Path("."))
        self.assertEqual(enriched.tool_source, "mcp")
        self.assertIn("mcp_tool", enriched.risk_tags)
        self.assertEqual(enriched.risk_level, "high")


if __name__ == "__main__":
    unittest.main()
