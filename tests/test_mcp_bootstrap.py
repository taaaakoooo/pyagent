"""Tests for MCP device bootstrap, handshake fallback, and notifications."""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional

from config.config import MCPConfig, MCPServerEntry
from internal.mcp.client import MCPClient, MCPRemoteError
from internal.mcp.factory import build_mcp_manager, build_mcp_manager_if_configured
from internal.mcp.mcp import MCPRequest, MCPResponse, MCPStdio
from internal.mcp.transport import (
    MCPProtocolError,
    MCPStdioTransport,
    MCPTimeoutError,
    MCPTransportError,
)


class FakeTransport:
    """Records requests and replays scripted responses."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None) -> None:
        self.responses = responses or {}
        self.sent: List[Dict[str, Any]] = []
        self.notifications: List[Dict[str, Any]] = []
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def send(self, request) -> MCPResponse:
        payload = request.to_dict()
        self.sent.append(payload)
        handler = self.responses.get(request.method)
        if handler is None:
            raise MCPRemoteError(
                {"code": -32601, "message": f"Method not found: {request.method}"}
            )
        if isinstance(handler, BaseException):
            raise handler
        if callable(handler):
            return handler(payload)
        return MCPResponse(id=payload.get("id", 0), result=handler)

    def notify(self, request) -> None:
        self.notifications.append(request.to_dict())

    def close(self) -> None:
        self.closed = True


def _client(transport: FakeTransport) -> MCPClient:
    return MCPClient(server_alias="fake", mode="stdio", transport=transport)


DISCOVER_RESULT = {
    "supportedVersions": ["2026-07-28"],
    "capabilities": {"tools": {}},
    "serverInfo": {"name": "fake", "version": "1.0"},
}

INITIALIZE_RESULT = {
    "protocolVersion": "2026-07-28",
    "capabilities": {"tools": {}},
    "serverInfo": {"name": "fake", "version": "1.0"},
}

TOOLS_RESULT = {"tools": [{"name": "echo", "description": "Echo"}]}


class HandshakeTests(unittest.TestCase):
    def test_discover_is_preferred_when_supported(self) -> None:
        transport = FakeTransport(
            {"server/discover": DISCOVER_RESULT, "tools/list": TOOLS_RESULT}
        )
        client = _client(transport)

        tools = client.start_and_discover()

        self.assertEqual([tool["name"] for tool in tools], ["echo"])
        self.assertEqual(client.cache.values["handshake"], "discover")
        self.assertEqual(transport.notifications, [])

    def test_falls_back_to_initialize_when_discover_is_unknown(self) -> None:
        transport = FakeTransport(
            {"initialize": INITIALIZE_RESULT, "tools/list": TOOLS_RESULT}
        )
        client = _client(transport)

        tools = client.start_and_discover()

        self.assertEqual([tool["name"] for tool in tools], ["echo"])
        self.assertEqual(client.cache.values["handshake"], "initialize")
        methods = [payload["method"] for payload in transport.sent]
        self.assertEqual(methods[0], "server/discover")
        self.assertEqual(methods[1], "initialize")

    def test_initialize_result_is_cached(self) -> None:
        transport = FakeTransport(
            {"initialize": INITIALIZE_RESULT, "tools/list": TOOLS_RESULT}
        )
        client = _client(transport)

        client.start_and_discover()

        self.assertEqual(client.cache.protocol_version, "2026-07-28")
        self.assertEqual(client.cache.server_info["name"], "fake")

    def test_initialize_raises_when_capabilities_are_not_a_mapping(self) -> None:
        transport = FakeTransport(
            {"initialize": {"protocolVersion": "2026-07-28", "capabilities": []}}
        )
        client = _client(transport)

        with self.assertRaises(ValueError):
            client.initialize()

    def test_timeout_is_not_retried_with_initialize(self) -> None:
        transport = FakeTransport({"server/discover": MCPTimeoutError("too slow")})
        client = _client(transport)

        with self.assertRaises(MCPTimeoutError):
            client.start_and_discover()

        self.assertEqual([p["method"] for p in transport.sent], ["server/discover"])

    def test_protocol_error_falls_back(self) -> None:
        transport = FakeTransport(
            {
                "server/discover": MCPProtocolError("bad envelope"),
                "initialize": INITIALIZE_RESULT,
                "tools/list": TOOLS_RESULT,
            }
        )
        client = _client(transport)

        client.start_and_discover()

        self.assertEqual(client.cache.values["handshake"], "initialize")


class NotificationTests(unittest.TestCase):
    def test_initialized_notification_has_no_id(self) -> None:
        transport = FakeTransport(
            {"initialize": INITIALIZE_RESULT, "tools/list": TOOLS_RESULT}
        )
        client = _client(transport)

        client.start_and_discover()

        self.assertEqual(len(transport.notifications), 1)
        notification = transport.notifications[0]
        self.assertEqual(notification["method"], "notifications/initialized")
        self.assertNotIn("id", notification)

    def test_notification_failure_is_swallowed(self) -> None:
        class FailingNotify(FakeTransport):
            def notify(self, request) -> None:
                raise MCPProtocolError("server hung up")

        transport = FailingNotify(
            {"initialize": INITIALIZE_RESULT, "tools/list": TOOLS_RESULT}
        )
        client = _client(transport)

        tools = client.start_and_discover()

        self.assertEqual(len(tools), 1)

    def test_send_notification_omits_id_on_request(self) -> None:
        transport = FakeTransport({})
        client = _client(transport)

        client.send_notification("notifications/cancelled", {"requestId": 3})

        payload = transport.notifications[0]
        self.assertNotIn("id", payload)
        self.assertEqual(payload["params"]["requestId"], 3)


class StdioNotificationFormatTests(unittest.TestCase):
    def test_notify_writes_a_json_line_without_id(self) -> None:
        transport = MCPStdioTransport(
            "fake",
            MCPStdio(command="python", args=["-c", ""]),
        )

        written: List[str] = []

        class FakeStdin:
            def write(self, value: str) -> None:
                written.append(value)

            def flush(self) -> None:
                pass

        class FakeProcess:
            stdin = FakeStdin()

            def poll(self):
                return None

        transport._process = FakeProcess()
        transport._started = True

        transport.notify(MCPRequest(method="notifications/initialized", id=None))

        self.assertEqual(len(written), 1)
        payload = json.loads(written[0])
        self.assertNotIn("id", payload)
        self.assertEqual(payload["method"], "notifications/initialized")
        self.assertTrue(written[0].endswith("\n"))

    def test_notify_requires_a_running_transport(self) -> None:
        transport = MCPStdioTransport(
            "fake",
            MCPStdio(command="python", args=["-c", ""]),
        )

        with self.assertRaises(MCPTransportError):
            transport.notify(MCPRequest(method="notifications/initialized"))


class FactoryTests(unittest.TestCase):
    def test_build_mcp_manager_creates_one_client_per_enabled_server(self) -> None:
        config = MCPConfig(
            servers=[
                MCPServerEntry(name="a", command="a"),
                MCPServerEntry(name="b", command="b", enabled=False),
                MCPServerEntry(name="c", transport="http", url="http://x/mcp"),
            ]
        )

        manager = build_mcp_manager(config)

        aliases = [client.server_alias for client in manager.list_clients()]
        self.assertEqual(sorted(aliases), ["a", "c"])

    def test_empty_servers_yields_no_manager(self) -> None:
        manager, aliases = build_mcp_manager_if_configured(MCPConfig())

        self.assertIsNone(manager)
        self.assertEqual(aliases, [])

    def test_disabled_config_yields_no_manager(self) -> None:
        config = MCPConfig(
            enabled=False,
            servers=[MCPServerEntry(name="a", command="a")],
        )

        manager, _ = build_mcp_manager_if_configured(config)

        self.assertIsNone(manager)

    def test_none_config_yields_no_manager(self) -> None:
        manager, aliases = build_mcp_manager_if_configured(None)

        self.assertIsNone(manager)
        self.assertEqual(aliases, [])

    def test_transport_is_selected_from_config(self) -> None:
        config = MCPConfig(
            servers=[
                MCPServerEntry(name="s", command="npx", args=["-y", "pkg"]),
                MCPServerEntry(
                    name="h", transport="streamable-http", url="http://x/mcp"
                ),
            ]
        )

        manager = build_mcp_manager(config)

        self.assertEqual(manager.get_client("s").mode, "stdio")
        self.assertEqual(manager.get_client("h").mode, "http")


if __name__ == "__main__":
    unittest.main()
