"""Tests for MCP client and manager layers."""

from __future__ import annotations

import unittest
from typing import List

from internal.mcp.client import MCPClient
from internal.mcp.manager import MCPManager
from internal.mcp.mcp import MCPMode, MCPRequest, MCPResponse


class _FakeTransport:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.requests: List[MCPRequest] = []

    def start(self) -> None:
        self.started = True

    def send(self, request: MCPRequest) -> MCPResponse:
        self.requests.append(request)
        return MCPResponse(id=request.id, result={"method": request.method})

    def close(self) -> None:
        self.closed = True


def _client(alias: str) -> MCPClient:
    return MCPClient(
        server_alias=alias,
        mode=MCPMode.STDIO,
        transport=_FakeTransport(),
    )


class MCPClientTests(unittest.TestCase):
    def test_generates_unique_request_ids(self) -> None:
        transport = _FakeTransport()
        client = MCPClient(
            server_alias="demo",
            mode=MCPMode.STDIO,
            transport=transport,
        )
        client.start()

        first = client.send_request("ping")
        second = client.send_request("status")
        client.close()

        self.assertEqual(first.id, 1)
        self.assertEqual(second.id, 2)
        self.assertTrue(transport.closed)


class MCPManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = MCPManager()
        self.manager.add_client(_client("alpha"))
        self.manager.add_client(_client("beta"))
        self.manager.add_client(_client("gamma"))

    def test_rejects_duplicate_alias(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.add_client(_client("alpha"))

    def test_starts_and_closes_all_clients(self) -> None:
        self.assertEqual(self.manager.start_all(), {})
        self.assertTrue(all(client.started for client in self.manager.list_clients()))

        self.assertEqual(self.manager.close_all(), {})
        self.assertTrue(
            all(client.transport.closed for client in self.manager.list_clients())
        )

    def test_allocates_incrementing_names_on_collision(self) -> None:
        first = self.manager.reserve_tool_name("alpha", "search")
        second = self.manager.reserve_tool_name("beta", "search")
        third = self.manager.reserve_tool_name("gamma", "search")

        self.assertEqual(first, "search")
        self.assertEqual(second, "search_1")
        self.assertEqual(third, "search_2")
        route = self.manager.resolve_tool_name("search_1")
        self.assertEqual(route.server_alias, "beta")
        self.assertEqual(route.remote_name, "search")

    def test_name_reservation_is_idempotent(self) -> None:
        first = self.manager.reserve_tool_name("alpha", "search")
        repeated = self.manager.reserve_tool_name("alpha", "search")
        self.assertEqual(first, repeated)
        self.assertEqual(len(self.manager.list_tool_routes()), 1)

    def test_removing_client_cleans_routes_without_renumbering(self) -> None:
        self.manager.reserve_tool_name("alpha", "search")
        self.manager.reserve_tool_name("beta", "search")
        third = self.manager.reserve_tool_name("gamma", "search")

        removed = self.manager.remove_client("beta")

        self.assertIsNotNone(removed)
        self.assertEqual(third, "search_2")
        self.assertEqual(
            self.manager.resolve_tool_name("search_2").server_alias,
            "gamma",
        )
        with self.assertRaises(KeyError):
            self.manager.resolve_tool_name("search_1")

    def test_reservation_avoids_names_already_used_by_local_tools(self) -> None:
        public_name = self.manager.reserve_tool_name(
            "alpha",
            "read_file",
            occupied_names={"read_file"},
        )
        self.assertEqual(public_name, "read_file_1")

    def test_send_request_starts_and_dispatches_client(self) -> None:
        response = self.manager.send_request("alpha", "ping", {"value": 1})

        client = self.manager.get_client("alpha")
        self.assertTrue(client.started)
        self.assertEqual(response.result, {"method": "ping"})
        request = client.transport.requests[-1]
        self.assertEqual(request.params["value"], 1)
        self.assertIn("_meta", request.params)

    def test_close_is_idempotent_and_rejects_new_requests(self) -> None:
        self.assertEqual(self.manager.close(), {})
        self.assertEqual(self.manager.close(), {})
        with self.assertRaises(RuntimeError):
            self.manager.send_request("alpha", "ping")


if __name__ == "__main__":
    unittest.main()
