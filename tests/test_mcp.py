"""Tests for MCP connection state models."""

from __future__ import annotations

import unittest

from internal.mcp.mcp import (
    MCP,
    MCPHttp,
    MCPMode,
    MCPRequest,
    MCPRequestMatcher,
    MCPResponse,
    MCPStdio,
)
from internal.types.types import MCPServerConfig


class MCPRequestMatcherTests(unittest.TestCase):
    def test_matches_response_to_request(self) -> None:
        matcher = MCPRequestMatcher()
        request = MCPRequest(
            id=1,
            method="tools/call",
            params={"name": "read_file", "arguments": {"path": "README.md"}},
        )
        matcher.add(request)

        exchange = matcher.match(
            MCPResponse(id=1, result={"content": [{"type": "text", "text": "ok"}]})
        )

        self.assertEqual(exchange.request, request)
        self.assertEqual(exchange.response.id, 1)
        self.assertFalse(matcher.is_pending(1))
        self.assertEqual(matcher.get_completed(1), exchange)

    def test_rejects_duplicate_pending_id(self) -> None:
        matcher = MCPRequestMatcher()
        request = MCPRequest(id="one", method="tools/list")
        matcher.add(request)

        with self.assertRaises(ValueError):
            matcher.add(request)

    def test_rejects_unknown_response_id(self) -> None:
        matcher = MCPRequestMatcher()
        with self.assertRaises(KeyError):
            matcher.match(MCPResponse(id=999, result={}))


class MCPStructureTests(unittest.TestCase):
    def test_stdio_structure(self) -> None:
        mcp = MCP(
            server_alias="filesystem",
            mode=MCPMode.STDIO,
            stdio=MCPStdio(command="python", args=["server.py"]),
            user_config={"roots": ["."]},
        )

        self.assertEqual(mcp.server_alias, "filesystem")
        self.assertEqual(mcp.stdio.command, "python")
        self.assertIsNone(mcp.http)
        self.assertEqual(mcp.user_config["roots"], ["."])

    def test_http_structure(self) -> None:
        mcp = MCP(
            server_alias="remote",
            mode="streamable-http",
            http=MCPHttp(url="https://example.com/mcp"),
        )

        self.assertEqual(mcp.mode, MCPMode.HTTP)
        self.assertIsNone(mcp.stdio)
        self.assertEqual(mcp.http.url, "https://example.com/mcp")

    def test_rejects_transport_configuration_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            MCP(
                server_alias="bad",
                mode=MCPMode.STDIO,
                http=MCPHttp(url="https://example.com/mcp"),
            )

    def test_builds_from_existing_server_config(self) -> None:
        config = MCPServerConfig(
            name="docs",
            transport="stdio",
            command="python",
            args=["docs_server.py"],
            env={"TOKEN": "secret"},
        )

        mcp = MCP.from_server_config(config, user_config={"approval": "always"})

        self.assertEqual(mcp.server_alias, "docs")
        self.assertEqual(mcp.mode, MCPMode.STDIO)
        self.assertEqual(mcp.stdio.args, ["docs_server.py"])
        self.assertEqual(mcp.user_config["approval"], "always")

    def test_parses_json_rpc_response(self) -> None:
        response = MCPResponse.from_dict(
            {"jsonrpc": "2.0", "id": 7, "result": {"tools": []}}
        )
        self.assertEqual(response.id, 7)
        self.assertEqual(response.result, {"tools": []})


if __name__ == "__main__":
    unittest.main()
