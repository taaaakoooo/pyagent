"""JSON-RPC transport implementations for MCP clients."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Protocol
from urllib import error as urlerror
from urllib import request as urlrequest

from internal.mcp.mcp import MCPHttp, MCPRequest, MCPResponse, MCPStdio, RequestId


class MCPTransportError(RuntimeError):
    """Base error raised by an MCP transport."""


class MCPTimeoutError(MCPTransportError):
    """Raised when an MCP request does not complete before its timeout."""


class MCPProcessExitedError(MCPTransportError):
    """Raised when an MCP stdio subprocess exits unexpectedly."""


class MCPProtocolError(MCPTransportError):
    """Raised when a transport receives malformed protocol data."""


class MCPTransport(Protocol):
    """Common interface implemented by MCP transports."""

    def start(self) -> None:
        ...

    def send(self, request: MCPRequest) -> MCPResponse:
        ...

    def close(self) -> None:
        ...


@dataclass
class _PendingResponse:
    event: threading.Event
    response: Optional[MCPResponse] = None
    error: Optional[BaseException] = None


class MCPStdioTransport:
    """MCP transport backed by a local subprocess and standard streams."""

    def __init__(self, alias: str, config: MCPStdio) -> None:
        self.alias = alias
        self.config = config
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._pending: Dict[RequestId, _PendingResponse] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._logger = logging.getLogger(f"pyagent.mcp.{alias}")

    @property
    def running(self) -> bool:
        process = self._process
        return bool(
            self._started
            and not self._closed
            and process is not None
            and process.poll() is None
        )

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise MCPTransportError("Cannot restart a closed stdio transport")
            if self.running:
                return

            environment = os.environ.copy()
            environment.update(self.config.env)
            try:
                process = subprocess.Popen(
                    [self.config.command] + list(self.config.args),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    cwd=self.config.cwd,
                    env=environment,
                )
            except OSError as exc:
                raise MCPTransportError(
                    f"Failed to start MCP server '{self.alias}': {exc}"
                ) from exc

            self._process = process
            self._started = True
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                name=f"mcp-{self.alias}-stdout",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                name=f"mcp-{self.alias}-stderr",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

    def send(self, request: MCPRequest) -> MCPResponse:
        if not self.running:
            raise MCPTransportError(
                f"MCP stdio transport '{self.alias}' is not running"
            )

        pending = _PendingResponse(event=threading.Event())
        with self._pending_lock:
            if request.id in self._pending:
                raise MCPTransportError(
                    f"Duplicate pending MCP request id: {request.id}"
                )
            self._pending[request.id] = pending

        payload = json.dumps(
            request.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        process = self._process
        try:
            if process is None or process.stdin is None:
                raise MCPProcessExitedError(
                    f"MCP server '{self.alias}' has no writable stdin"
                )
            with self._write_lock:
                process.stdin.write(payload + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            with self._pending_lock:
                self._pending.pop(request.id, None)
            raise MCPProcessExitedError(
                f"MCP server '{self.alias}' closed stdin"
            ) from exc

        if not pending.event.wait(self.config.timeout_seconds):
            with self._pending_lock:
                self._pending.pop(request.id, None)
            raise MCPTimeoutError(
                f"MCP request {request.id!r} timed out after "
                f"{self.config.timeout_seconds:g}s"
            )

        with self._pending_lock:
            self._pending.pop(request.id, None)
        if pending.error is not None:
            raise pending.error
        if pending.response is None:
            raise MCPProtocolError(
                f"MCP request {request.id!r} completed without a response"
            )
        return pending.response

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            process = self._process

        self._fail_all(MCPTransportError(f"MCP transport '{self.alias}' closed"))
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)

        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._fail_all(
                MCPProcessExitedError(
                    f"MCP server '{self.alias}' has no readable stdout"
                )
            )
            return

        try:
            for line in process.stdout:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    if not isinstance(data, dict):
                        raise ValueError("JSON-RPC message must be an object")
                    response = MCPResponse.from_dict(data)
                except (json.JSONDecodeError, ValueError) as exc:
                    self._logger.error("Invalid MCP stdout message: %s", stripped)
                    self._fail_all(
                        MCPProtocolError(
                            f"MCP server '{self.alias}' returned invalid JSON-RPC: {exc}"
                        )
                    )
                    continue
                self._deliver(response)
        finally:
            if not self._closed:
                return_code = process.poll()
                self._fail_all(
                    MCPProcessExitedError(
                        f"MCP server '{self.alias}' stdout closed"
                        + (
                            f" with exit code {return_code}"
                            if return_code is not None
                            else ""
                        )
                    )
                )

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            message = line.rstrip("\r\n")
            if message:
                self._logger.warning("%s", message)

    def _deliver(self, response: MCPResponse) -> None:
        with self._pending_lock:
            pending = self._pending.get(response.id)
        if pending is None:
            self._logger.warning(
                "Received response for unknown MCP request id %r", response.id
            )
            return
        pending.response = response
        pending.event.set()

    def _fail_all(self, error: BaseException) -> None:
        with self._pending_lock:
            pending_values = list(self._pending.values())
        for pending in pending_values:
            pending.error = error
            pending.event.set()


class MCPHttpTransport:
    """MCP JSON-RPC transport over HTTP with endpoint fallback."""

    def __init__(self, alias: str, config: MCPHttp) -> None:
        self.alias = alias
        self.config = config
        self.resolved_endpoint: Optional[str] = None
        self._started = False
        self._closed = False
        self._endpoint_lock = threading.Lock()
        self._resolution_lock = threading.Lock()

    def start(self) -> None:
        if self._closed:
            raise MCPTransportError("Cannot restart a closed HTTP transport")
        self._started = True

    def send(self, request: MCPRequest) -> MCPResponse:
        if not self._started or self._closed:
            raise MCPTransportError(
                f"MCP HTTP transport '{self.alias}' is not running"
            )

        with self._endpoint_lock:
            cached_endpoint = self.resolved_endpoint
        if cached_endpoint is not None:
            return self._post(cached_endpoint, request)

        with self._resolution_lock:
            with self._endpoint_lock:
                cached_endpoint = self.resolved_endpoint
            if cached_endpoint is not None:
                return self._post(cached_endpoint, request)
            return self._resolve_and_post(request)

    def close(self) -> None:
        self._closed = True
        self._started = False
        with self._endpoint_lock:
            self.resolved_endpoint = None

    def _candidate_endpoints(self) -> list[str]:
        base = self.config.url.rstrip("/")
        candidates = [base + "/mcp", base]
        return list(dict.fromkeys(candidates))

    def _resolve_and_post(self, request: MCPRequest) -> MCPResponse:
        candidates = self._candidate_endpoints()
        first_error: Optional[BaseException] = None
        for index, endpoint in enumerate(candidates):
            try:
                response = self._post(endpoint, request)
            except urlerror.HTTPError as exc:
                if index == 0 and exc.code in {404, 405} and len(candidates) > 1:
                    exc.close()
                    first_error = exc
                    continue
                try:
                    transport_error = self._http_error(endpoint, exc)
                finally:
                    exc.close()
                raise transport_error from exc
            except urlerror.URLError as exc:
                if (
                    index == 0
                    and len(candidates) > 1
                    and self._is_connection_setup_error(exc)
                ):
                    first_error = exc
                    continue
                if self._is_timeout_error(exc):
                    raise MCPTimeoutError(
                        f"MCP HTTP request to '{endpoint}' timed out"
                    ) from exc
                raise MCPTransportError(
                    f"MCP HTTP request to '{endpoint}' failed: {exc.reason}"
                ) from exc

            with self._endpoint_lock:
                self.resolved_endpoint = endpoint
            return response

        raise MCPTransportError(
            f"No usable HTTP endpoint for MCP server '{self.alias}': {first_error}"
        )

    def _post(self, endpoint: str, request: MCPRequest) -> MCPResponse:
        body = json.dumps(
            request.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        meta = request.params.get("_meta") or {}
        if isinstance(meta, dict):
            protocol_version = meta.get(
                "io.modelcontextprotocol/protocolVersion"
            )
            if protocol_version:
                headers["MCP-Protocol-Version"] = str(protocol_version)
        headers["Mcp-Method"] = request.method
        tool_name = request.params.get("name")
        if request.method == "tools/call" and tool_name:
            headers["Mcp-Name"] = str(tool_name)
        headers.update(self.config.headers)
        http_request = urlrequest.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            response = urlrequest.urlopen(
                http_request,
                timeout=self.config.timeout_seconds,
            )
        except socket.timeout as exc:
            raise MCPTimeoutError(
                f"MCP HTTP request to '{endpoint}' timed out"
            ) from exc

        with response:
            content_type = response.headers.get("Content-Type", "")
            if content_type.lower().startswith("text/event-stream"):
                raise MCPProtocolError(
                    "SSE responses are not supported by this MCP transport yet"
                )
            raw = response.read()

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MCPProtocolError(
                f"MCP HTTP endpoint '{endpoint}' returned invalid JSON"
            ) from exc
        if not isinstance(data, dict):
            raise MCPProtocolError("MCP HTTP response must be a JSON object")
        try:
            return MCPResponse.from_dict(data)
        except ValueError as exc:
            raise MCPProtocolError(str(exc)) from exc

    @staticmethod
    def _is_timeout_error(exc: urlerror.URLError) -> bool:
        return isinstance(exc.reason, (socket.timeout, TimeoutError))

    @classmethod
    def _is_connection_setup_error(cls, exc: urlerror.URLError) -> bool:
        if cls._is_timeout_error(exc):
            return False
        return isinstance(
            exc.reason,
            (ConnectionRefusedError, socket.gaierror),
        )

    @staticmethod
    def _http_error(endpoint: str, exc: urlerror.HTTPError) -> MCPTransportError:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
        detail = f": {body}" if body else ""
        return MCPTransportError(
            f"MCP HTTP endpoint '{endpoint}' returned HTTP {exc.code}{detail}"
        )


__all__ = [
    "MCPHttpTransport",
    "MCPProcessExitedError",
    "MCPProtocolError",
    "MCPStdioTransport",
    "MCPTimeoutError",
    "MCPTransport",
    "MCPTransportError",
]
