"""OpenAI-compatible SSE streaming LLM client using urllib."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from config.config import LLMConfig, TokenBudget
from internal.types.types import (
    FunctionCall,
    LLMResponse,
    LLMStreamChunk,
    Message,
    ToolCall,
    ToolDefinition,
)

MAX_SSE_LINE_BYTES = 1 << 20
READ_CHUNK_BYTES = 8192
CHAT_COMPLETIONS_PATH = "/chat/completions"
MAX_OUTPUT_TOKENS = 4096
COMPACT_MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}
RETRY_BACKOFF_SECONDS = [1, 2, 4]
MAX_RETRIES = 3
_SSE_DONE = object()


class LLMError(Exception):
    """Base class for LLM client errors."""


class LLMTimeoutError(LLMError):
    """Raised when a streaming response exceeds the configured timeout."""


class LLMStreamError(LLMError):
    """Raised when SSE parsing fails or a line exceeds the size limit."""


def _extract_api_error_message(body: str) -> str:
    """Pull the human-readable reason out of a provider error body."""
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body.strip()[:200]

    if not isinstance(payload, dict):
        return ""

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message.strip()[:200]
        code = error.get("code")
        if isinstance(code, str) and code:
            return code
    if isinstance(error, str) and error:
        return error.strip()[:200]

    message = payload.get("message")
    if isinstance(message, str) and message:
        return message.strip()[:200]
    return ""


class LLMHTTPError(LLMError):
    """Raised for non-retryable HTTP failures."""

    def __init__(self, status: int, message: str, body: str = "") -> None:
        self.status = status
        self.body = body
        self.api_message = _extract_api_error_message(body)
        detail = f"HTTP {status}: {message}"
        if self.api_message:
            detail = f"{detail} - {self.api_message}"
        super().__init__(detail)


class LLMRetryExhaustedError(LLMError):
    """Raised when retryable HTTP failures persist after all retries."""


def _serialize_tool(tool: ToolDefinition) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _serialize_tool_call(tool_call: ToolCall) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": tool_call.id,
        "type": tool_call.type or "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }
    if tool_call.index is not None:
        payload["index"] = tool_call.index
    return payload


def _serialize_message(message: Message) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            _serialize_tool_call(tool_call) for tool_call in message.tool_calls
        ]
    return payload


def _merge_tool_call(bucket: Dict[int, ToolCall], delta: ToolCall) -> None:
    index = delta.index if delta.index is not None else 0
    if index not in bucket:
        bucket[index] = ToolCall(
            index=index,
            id=delta.id,
            type=delta.type or "function",
            function=FunctionCall(
                name=delta.function.name,
                arguments=delta.function.arguments,
            ),
        )
        return

    current = bucket[index]
    if delta.id:
        current.id = delta.id
    if delta.type:
        current.type = delta.type
    if delta.function.name:
        current.function.name = delta.function.name
    if delta.function.arguments:
        current.function.arguments += delta.function.arguments


def _parse_tool_call_delta(raw: Dict[str, Any]) -> ToolCall:
    function = raw.get("function") or {}
    return ToolCall(
        index=raw.get("index"),
        id=raw.get("id") or "",
        type=raw.get("type") or "function",
        function=FunctionCall(
            name=function.get("name") or "",
            arguments=function.get("arguments") or "",
        ),
    )


def _parse_stream_chunk(data: Dict[str, Any]) -> LLMStreamChunk:
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or {}

    content_delta = delta.get("content")
    tool_calls = [
        _parse_tool_call_delta(item) for item in (delta.get("tool_calls") or [])
    ]
    usage_raw = data.get("usage") or {}
    usage = {
        key: int(value)
        for key, value in usage_raw.items()
        if isinstance(value, (int, float))
    }

    return LLMStreamChunk(
        content_delta=content_delta,
        tool_calls=tool_calls,
        finish_reason=choice.get("finish_reason"),
        usage=usage,
        raw=data,
    )


class LLMClient:
    """Streaming LLM client with retry, token tracking, and SSE parsing."""

    def __init__(
        self,
        llm_config: LLMConfig,
        token_budget: Optional[TokenBudget] = None,
    ) -> None:
        self._config = llm_config
        self._token_budget = token_budget
        if not self._config.api_key:
            self._config.resolve_api_key()

    def stream(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
    ) -> Iterator[LLMStreamChunk]:
        body = self._build_request_body(messages, tools)
        response = self._open_stream_request(body)
        try:
            yield from self._iter_sse_chunks(response)
        finally:
            response.close()

    def complete(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
    ) -> LLMResponse:
        content_parts: List[str] = []
        tool_calls_by_index: Dict[int, ToolCall] = {}
        finish_reason: Optional[str] = None
        usage: Dict[str, int] = {}

        for chunk in self.stream(messages, tools):
            if chunk.content_delta:
                content_parts.append(chunk.content_delta)
            for tool_call in chunk.tool_calls:
                _merge_tool_call(tool_calls_by_index, tool_call)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.usage:
                usage = chunk.usage

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=[
                tool_calls_by_index[index]
                for index in sorted(tool_calls_by_index)
            ],
            finish_reason=finish_reason,
            usage=usage,
        )

    def chat_compact(self, messages: List[Message]) -> LLMResponse:
        """Simple non-streaming chat without tools, for context compression."""
        body = self._build_compact_request_body(messages)
        data = self._post_json(body)
        return self._parse_completion_response(data)

    def remaining_tokens(self) -> int:
        if self._token_budget is None:
            return 0
        return max(0, self._token_budget.limit - self._token_budget.total_tokens)

    def token_usage_summary(self) -> Dict[str, Any]:
        if self._token_budget is None:
            return {
                "used": 0,
                "limit": 0,
                "remaining": 0,
                "ratio_percent": 0.0,
            }
        return {
            "used": self._token_budget.total_tokens,
            "limit": self._token_budget.limit,
            "remaining": self.remaining_tokens(),
            "ratio_percent": round(self._token_budget.usage_ratio * 100, 2),
        }

    def _build_request_body(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
    ) -> Dict[str, Any]:
        serialized_tools = [_serialize_tool(tool) for tool in tools]
        body: Dict[str, Any] = {
            "model": self._config.model,
            "messages": [_serialize_message(message) for message in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        # Omit the key entirely when there are no tools: several providers
        # (DashScope/Qwen among them) reject an empty array with HTTP 400
        # "[] is too short - 'tools'".
        if serialized_tools:
            body["tools"] = serialized_tools
            body["tool_choice"] = "auto"
        return body

    def _build_compact_request_body(self, messages: List[Message]) -> Dict[str, Any]:
        return {
            "model": self._config.model,
            "messages": [_serialize_message(message) for message in messages],
            "stream": False,
            "temperature": TEMPERATURE,
            "max_tokens": COMPACT_MAX_OUTPUT_TOKENS,
        }

    def _chat_completions_url(self) -> str:
        return self._config.api_base.rstrip("/") + CHAT_COMPLETIONS_PATH

    def _build_http_request(
        self,
        body: Dict[str, Any],
        accept: str,
    ) -> urllib.request.Request:
        payload = json.dumps(body).encode("utf-8")
        return urllib.request.Request(
            self._chat_completions_url(),
            data=payload,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
                "Accept": accept,
            },
            method="POST",
        )

    def _request_with_retry(self, body: Dict[str, Any], accept: str) -> Any:
        last_error: Optional[LLMError] = None

        for attempt in range(MAX_RETRIES + 1):
            request = self._build_http_request(body, accept)
            try:
                response = urllib.request.urlopen(
                    request,
                    timeout=self._config.timeout_seconds,
                )
                status = response.getcode() or 200
                if status in RETRYABLE_STATUS_CODES:
                    error_body = response.read().decode("utf-8", errors="replace")
                    response.close()
                    raise LLMHTTPError(status, "retryable server error", error_body)
                if status >= 400:
                    error_body = response.read().decode("utf-8", errors="replace")
                    response.close()
                    raise LLMHTTPError(status, "request failed", error_body)
                return response
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                http_error = LLMHTTPError(
                    exc.code,
                    exc.reason or "request failed",
                    error_body,
                )
                if (
                    http_error.status not in RETRYABLE_STATUS_CODES
                    or attempt >= MAX_RETRIES
                ):
                    if (
                        http_error.status in RETRYABLE_STATUS_CODES
                        and attempt >= MAX_RETRIES
                    ):
                        raise LLMRetryExhaustedError(
                            f"LLM request failed after {MAX_RETRIES} retries "
                            f"(last status: {http_error.status})"
                        ) from http_error
                    raise http_error from exc
                last_error = http_error
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
            except urllib.error.URLError as exc:
                # Transient socket failures (flaky proxy, DNS hiccup, a local
                # stack briefly refusing connections) surface here. Retrying
                # with backoff turns them into a non-event instead of failing
                # the whole turn.
                last_error = LLMError(f"Network error: {exc.reason}")
                if attempt >= MAX_RETRIES:
                    raise LLMRetryExhaustedError(
                        f"Network error after {MAX_RETRIES + 1} attempts: "
                        f"{exc.reason}"
                    ) from exc
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])

        raise LLMRetryExhaustedError(
            f"LLM request failed after {MAX_RETRIES} retries"
        ) from last_error

    def _open_stream_request(self, body: Dict[str, Any]) -> Any:
        return self._request_with_retry(body, "text/event-stream")

    def _post_json(self, body: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request_with_retry(body, "application/json")
        try:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMStreamError(f"Invalid JSON response: {exc}") from exc
        finally:
            response.close()

        if not isinstance(data, dict):
            raise LLMStreamError("Expected JSON object response")
        return data

    def _parse_completion_response(self, data: Dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}

        usage_raw = data.get("usage") or {}
        usage = {
            key: int(value)
            for key, value in usage_raw.items()
            if isinstance(value, (int, float))
        }
        if usage and self._token_budget is not None:
            self._token_budget.add_usage(usage)

        tool_calls = [
            _parse_tool_call_delta(item) for item in (message.get("tool_calls") or [])
        ]

        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            raw=data,
        )

    def _iter_sse_chunks(self, response: Any) -> Iterator[LLMStreamChunk]:
        buffer = b""
        start = time.monotonic()

        while True:
            if time.monotonic() - start > self._config.timeout_seconds:
                raise LLMTimeoutError(
                    f"Streaming response exceeded {self._config.timeout_seconds}s"
                )

            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                parsed = self._parse_sse_line(line_bytes)
                if parsed is _SSE_DONE:
                    return
                if parsed is not None:
                    yield parsed

            raw_chunk = response.read(READ_CHUNK_BYTES)
            if not raw_chunk:
                if buffer:
                    parsed = self._parse_sse_line(buffer)
                    if parsed is _SSE_DONE:
                        return
                    if parsed is not None:
                        yield parsed
                break

            buffer += raw_chunk
            if len(buffer) > MAX_SSE_LINE_BYTES and b"\n" not in buffer:
                raise LLMStreamError("SSE line exceeds 1MB")

    def _parse_sse_line(self, line_bytes: bytes) -> Any:
        if len(line_bytes) > MAX_SSE_LINE_BYTES:
            raise LLMStreamError("SSE line exceeds 1MB")

        line = line_bytes.decode("utf-8").strip()
        if not line or line.startswith(":"):
            return None
        if not line.startswith("data: "):
            return None

        payload = line[6:].strip()
        if payload == "[DONE]":
            return _SSE_DONE

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMStreamError(f"Invalid SSE JSON payload: {exc}") from exc

        chunk = _parse_stream_chunk(data)
        if chunk.usage and self._token_budget is not None:
            self._token_budget.add_usage(chunk.usage)
        return chunk


__all__ = [
    "LLMClient",
    "LLMError",
    "LLMHTTPError",
    "LLMRetryExhaustedError",
    "LLMStreamError",
    "LLMTimeoutError",
    "COMPACT_MAX_OUTPUT_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "TEMPERATURE",
]
