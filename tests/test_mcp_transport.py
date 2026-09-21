"""Tests for MCP stdio and HTTP transports."""

from __future__ import annotations

import gc
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from internal.mcp.mcp import MCPHttp, MCPRequest, MCPStdio
from internal.mcp.transport import (
    MCPHttpTransport,
    MCPStdioTransport,
    MCPTimeoutError,
    MCPTransportError,
)


ECHO_SERVER_CODE = r"""
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    response = {
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {"method": request["method"], "params": request.get("params", {})},
    }
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    sys.stdout.flush()
"""


STDERR_SERVER_CODE = r"""
import json
import sys

request = json.loads(sys.stdin.readline())
sys.stderr.write("x" * 200000 + "\n")
sys.stderr.flush()
response = {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}
sys.stdout.write(json.dumps(response) + "\n")
sys.stdout.flush()
"""

OUT_OF_ORDER_SERVER_CODE = r"""
import json
import sys

first = json.loads(sys.stdin.readline())
second = json.loads(sys.stdin.readline())
for request in (second, first):
    response = {
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {"id": request["id"]},
    }
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
"""

SLOW_SERVER_CODE = r"""
import sys
import time

sys.stdin.readline()
time.sleep(10)
"""


class MCPStdioTransportTests(unittest.TestCase):
    def test_sends_and_matches_json_rpc_response(self) -> None:
        transport = MCPStdioTransport(
            "echo",
            MCPStdio(
                command=sys.executable,
                args=["-u", "-c", ECHO_SERVER_CODE],
                timeout_seconds=5,
            ),
        )
        transport.start()
        try:
            response = transport.send(
                MCPRequest(id=1, method="ping", params={"value": "hello"})
            )
        finally:
            transport.close()

        self.assertEqual(response.id, 1)
        self.assertEqual(response.result["method"], "ping")
        self.assertEqual(response.result["params"], {"value": "hello"})

    def test_stderr_is_drained_without_blocking_response(self) -> None:
        transport = MCPStdioTransport(
            "noisy",
            MCPStdio(
                command=sys.executable,
                args=["-u", "-c", STDERR_SERVER_CODE],
                timeout_seconds=5,
            ),
        )
        transport._logger = mock.Mock()
        transport.start()
        try:
            response = transport.send(MCPRequest(id=2, method="ping"))
        finally:
            transport.close()

        self.assertEqual(response.result, {"ok": True})
        self.assertTrue(transport._logger.warning.called)

    def test_send_requires_running_transport(self) -> None:
        transport = MCPStdioTransport(
            "idle",
            MCPStdio(command=sys.executable, args=["-c", "pass"]),
        )
        with self.assertRaises(MCPTransportError):
            transport.send(MCPRequest(id=1, method="ping"))

    def test_matches_concurrent_out_of_order_responses(self) -> None:
        transport = MCPStdioTransport(
            "concurrent",
            MCPStdio(
                command=sys.executable,
                args=["-u", "-c", OUT_OF_ORDER_SERVER_CODE],
                timeout_seconds=5,
            ),
        )
        responses = {}
        failures = []

        def send(request_id: int) -> None:
            try:
                responses[request_id] = transport.send(
                    MCPRequest(id=request_id, method="ping")
                )
            except BaseException as exc:
                failures.append(exc)

        transport.start()
        threads = [
            threading.Thread(target=send, args=(1,)),
            threading.Thread(target=send, args=(2,)),
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=6)
        finally:
            transport.close()

        self.assertEqual(failures, [])
        self.assertEqual(responses[1].result, {"id": 1})
        self.assertEqual(responses[2].result, {"id": 2})

    def test_timeout_removes_pending_request(self) -> None:
        transport = MCPStdioTransport(
            "slow",
            MCPStdio(
                command=sys.executable,
                args=["-u", "-c", SLOW_SERVER_CODE],
                timeout_seconds=0.05,
            ),
        )
        transport.start()
        try:
            with self.assertRaises(MCPTimeoutError):
                transport.send(MCPRequest(id=9, method="slow"))
            self.assertEqual(transport._pending, {})
        finally:
            transport.close()


class _MCPHttpHandler(BaseHTTPRequestHandler):
    mcp_status = 200
    root_status = 200
    paths = []
    request_headers = []

    def do_POST(self) -> None:
        type(self).paths.append(self.path)
        type(self).request_headers.append(dict(self.headers.items()))
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        status = (
            type(self).mcp_status
            if self.path == "/mcp"
            else type(self).root_status
        )
        if status != 200:
            self.send_response(status)
            self.end_headers()
            return

        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"path": self.path},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        return


class MCPHttpTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        _MCPHttpHandler.paths = []
        _MCPHttpHandler.request_headers = []
        _MCPHttpHandler.mcp_status = 200
        _MCPHttpHandler.root_status = 200
        self.server = HTTPServer(("127.0.0.1", 0), _MCPHttpHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        del self.server
        gc.collect()

    def test_prefers_mcp_path_and_caches_endpoint(self) -> None:
        transport = MCPHttpTransport(
            "remote",
            MCPHttp(url=self.base_url, timeout_seconds=2),
        )
        transport.start()
        first = transport.send(MCPRequest(id=1, method="ping"))
        second = transport.send(MCPRequest(id=2, method="ping"))
        transport.close()

        self.assertEqual(first.result["path"], "/mcp")
        self.assertEqual(second.result["path"], "/mcp")
        self.assertEqual(_MCPHttpHandler.paths, ["/mcp", "/mcp"])

    def test_falls_back_to_root_on_404(self) -> None:
        _MCPHttpHandler.mcp_status = 404
        transport = MCPHttpTransport(
            "remote",
            MCPHttp(url=self.base_url, timeout_seconds=2),
        )
        transport.start()
        response = transport.send(MCPRequest(id=1, method="ping"))
        transport.close()

        self.assertEqual(response.result["path"], "/")
        self.assertEqual(_MCPHttpHandler.paths, ["/mcp", "/"])

    def test_does_not_fallback_on_server_error(self) -> None:
        _MCPHttpHandler.mcp_status = 500
        transport = MCPHttpTransport(
            "remote",
            MCPHttp(url=self.base_url, timeout_seconds=2),
        )
        transport.start()
        try:
            with self.assertRaises(MCPTransportError):
                transport.send(MCPRequest(id=1, method="ping"))
        finally:
            transport.close()

        self.assertEqual(_MCPHttpHandler.paths, ["/mcp"])

    def test_concurrent_fallback_resolves_mcp_path_only_once(self) -> None:
        _MCPHttpHandler.mcp_status = 404
        transport = MCPHttpTransport(
            "remote",
            MCPHttp(url=self.base_url, timeout_seconds=2),
        )
        transport.start()
        responses = []

        def send(request_id: int) -> None:
            responses.append(
                transport.send(MCPRequest(id=request_id, method="ping"))
            )

        threads = [
            threading.Thread(target=send, args=(1,)),
            threading.Thread(target=send, args=(2,)),
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
        finally:
            transport.close()

        self.assertEqual(len(responses), 2)
        self.assertEqual(_MCPHttpHandler.paths.count("/mcp"), 1)
        self.assertEqual(_MCPHttpHandler.paths.count("/"), 2)

    def test_sends_modern_mcp_routing_headers(self) -> None:
        transport = MCPHttpTransport(
            "remote",
            MCPHttp(url=self.base_url, timeout_seconds=2),
        )
        transport.start()
        try:
            transport.send(
                MCPRequest(
                    id=1,
                    method="tools/call",
                    params={
                        "name": "search",
                        "arguments": {},
                        "_meta": {
                            "io.modelcontextprotocol/protocolVersion": "2026-07-28"
                        },
                    },
                )
            )
        finally:
            transport.close()

        headers = {
            key.lower(): value
            for key, value in _MCPHttpHandler.request_headers[0].items()
        }
        self.assertEqual(headers["mcp-protocol-version"], "2026-07-28")
        self.assertEqual(headers["mcp-method"], "tools/call")
        self.assertEqual(headers["mcp-name"], "search")


if __name__ == "__main__":
    unittest.main()
