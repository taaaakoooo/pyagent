"""Tests for modern MCP discovery, tool normalization, and tool calls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from internal.mcp.client import MCPClient
from internal.mcp.manager import MCPManager
from internal.mcp.mcp import MCPMode, MCPRequest, MCPResponse, MCP_PROTOCOL_VERSION
from internal.mcp.tooling import normalize_mcp_tool
from internal.tools.tools import create_default_tool_registry


class _ScriptedTransport:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.requests: List[MCPRequest] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def send(self, request: MCPRequest) -> MCPResponse:
        self.requests.append(request)
        if request.method == "server/discover":
            return MCPResponse(
                id=request.id,
                result={
                    "supportedVersions": [MCP_PROTOCOL_VERSION],
                    "capabilities": {"tools": {}},
                    "_meta": {
                        "io.modelcontextprotocol/serverInfo": {
                            "name": "test-server",
                            "version": "1.0",
                        }
                    },
                },
            )
        if request.method == "tools/list":
            cursor = request.params.get("cursor")
            if cursor is None:
                return MCPResponse(
                    id=request.id,
                    result={
                        "tools": [
                            {
                                "name": "search",
                                "description": "Search docs",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                },
                            }
                        ],
                        "nextCursor": "page-2",
                    },
                )
            return MCPResponse(
                id=request.id,
                result={
                    "tools": [
                        {
                            "name": "fetch",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                },
            )
        if request.method == "tools/call":
            return MCPResponse(
                id=request.id,
                result={
                    "content": [{"type": "text", "text": "tool output"}],
                    "structuredContent": {"count": 1},
                    "isError": False,
                },
            )
        raise AssertionError(f"Unexpected MCP method: {request.method}")


class MCPProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = _ScriptedTransport()
        self.client = MCPClient(
            server_alias="docs",
            mode=MCPMode.STDIO,
            transport=self.transport,
        )

    def test_discovery_injects_modern_request_metadata(self) -> None:
        result = self.client.discover_server()

        request = self.transport.requests[0]
        meta = request.params["_meta"]
        self.assertEqual(
            meta["io.modelcontextprotocol/protocolVersion"],
            MCP_PROTOCOL_VERSION,
        )
        self.assertEqual(result["capabilities"], {"tools": {}})
        self.assertTrue(self.client.cache.initialized)
        self.assertEqual(self.client.cache.protocol_version, MCP_PROTOCOL_VERSION)

    def test_tools_list_follows_pagination(self) -> None:
        tools = self.client.list_tools()

        self.assertEqual([tool["name"] for tool in tools], ["search", "fetch"])
        list_requests = [
            request
            for request in self.transport.requests
            if request.method == "tools/list"
        ]
        self.assertEqual(len(list_requests), 2)
        self.assertEqual(list_requests[1].params["cursor"], "page-2")

    def test_call_tool_normalizes_result(self) -> None:
        result = self.client.call_tool("search", {"query": "MCP"})

        self.assertFalse(result.is_error)
        self.assertEqual(result.content, "tool output")
        self.assertEqual(result.metadata["structured_content"], {"count": 1})
        request = self.transport.requests[-1]
        self.assertEqual(request.params["name"], "search")
        self.assertEqual(request.params["arguments"], {"query": "MCP"})

    def test_normalize_tool_rejects_non_object_schema(self) -> None:
        with self.assertRaises(ValueError):
            normalize_mcp_tool(
                "docs",
                {"name": "bad", "inputSchema": {"type": "array"}},
                "bad",
            )


class MCPRegistryIntegrationTests(unittest.TestCase):
    def test_manager_registers_mcp_tools_after_local_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = create_default_tool_registry(
                workspace,
                workspace / ".tool_outputs",
            )
            manager = MCPManager()
            client = MCPClient(
                server_alias="docs",
                mode=MCPMode.STDIO,
                transport=_ScriptedTransport(),
            )
            manager.add_client(client)

            failures = manager.start(registry)

            self.assertEqual(failures, {})
            definitions = registry.list_tools()
            names = [definition.name for definition in definitions]
            self.assertEqual(names[-2:], ["search", "fetch"])
            self.assertTrue(all(item.source == "mcp" for item in definitions[-2:]))

            result = registry.call_tool("search", {"query": "hello"})
            self.assertFalse(result.is_error)
            self.assertEqual(result.content, "tool output")
            self.assertEqual(result.metadata["mcp_server"], "docs")
            manager.close()


if __name__ == "__main__":
    unittest.main()
