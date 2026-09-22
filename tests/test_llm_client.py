"""Tests for the streaming LLM client."""

from __future__ import annotations

import io
import json
import time
import unittest
import urllib.error
import urllib.request
from typing import Any, Dict, List
from unittest import mock

from config.config import LLMConfig, TokenUsageConfig, TokenBudget
from internal.llm.client import (
    COMPACT_MAX_OUTPUT_TOKENS,
    LLMClient,
    LLMHTTPError,
    LLMRetryExhaustedError,
    LLMStreamError,
    MAX_OUTPUT_TOKENS,
    MAX_SSE_LINE_BYTES,
    RETRY_BACKOFF_SECONDS,
    TEMPERATURE,
    _merge_tool_call,
    _parse_stream_chunk,
)
from internal.types.types import FunctionCall, Message, ToolCall, ToolDefinition


def _make_client(token_budget: TokenBudget | None = None) -> LLMClient:
    config = LLMConfig(api_key="test-key", timeout_seconds=180.0)
    return LLMClient(config, token_budget)


class FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = io.BytesIO(body)
        self._status = status

    def getcode(self) -> int:
        return self._status

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def close(self) -> None:
        self._body.close()


class LLMClientTests(unittest.TestCase):
    def test_build_request_body_fixed_params(self) -> None:
        client = _make_client()
        body = client._build_request_body(
            [Message(role="user", content="hello")],
            [
                ToolDefinition(
                    name="read_file",
                    description="Read a file",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
        )

        self.assertTrue(body["stream"])
        self.assertEqual(body["temperature"], TEMPERATURE)
        self.assertEqual(body["max_tokens"], MAX_OUTPUT_TOKENS)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(len(body["tools"]), 1)
        self.assertEqual(body["tools"][0]["function"]["name"], "read_file")

    def test_build_request_body_empty_tools(self) -> None:
        client = _make_client()
        body = client._build_request_body(
            [Message(role="user", content="hello")],
            [],
        )
        # The key must be omitted entirely: DashScope rejects an empty array
        # with HTTP 400 "[] is too short - 'tools'".
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)

    def test_stream_parses_content_deltas(self) -> None:
        client = _make_client()
        sse_body = (
            'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'
            "data: [DONE]\n\n"
        ).encode("utf-8")
        response = FakeHTTPResponse(sse_body)

        chunks = list(client._iter_sse_chunks(response))
        self.assertEqual([chunk.content_delta for chunk in chunks], ["你", "好"])

    def test_complete_merges_tool_calls(self) -> None:
        client = _make_client()
        first_event = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path"',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        second_event = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": ':"main.py"}'},
                            }
                        ]
                    }
                }
            ]
        }
        sse_body = (
            f"data: {json.dumps(first_event, ensure_ascii=True)}\n\n"
            f"data: {json.dumps(second_event, ensure_ascii=True)}\n\n"
            "data: [DONE]\n\n"
        ).encode("utf-8")

        with mock.patch.object(client, "_open_stream_request", return_value=FakeHTTPResponse(sse_body)):
            result = client.complete([Message(role="user", content="read main.py")], [])

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "read_file")
        self.assertEqual(
            result.tool_calls[0].function.arguments,
            '{"path":"main.py"}',
        )

    def test_sse_line_limit_raises(self) -> None:
        client = _make_client()
        oversized = b"x" * (MAX_SSE_LINE_BYTES + 1) + b"\n"
        response = FakeHTTPResponse(oversized)

        with self.assertRaises(LLMStreamError):
            list(client._iter_sse_chunks(response))

    def test_remaining_tokens_updates_with_usage(self) -> None:
        budget = TokenBudget(TokenUsageConfig(context_window=128000))
        client = _make_client(budget)
        sse_body = (
            b'data: {"choices":[{"delta":{"content":"hi"}}],'
            b'"usage":{"prompt_tokens":1000,"completion_tokens":500}}\n\n'
            b"data: [DONE]\n\n"
        )

        with mock.patch.object(client, "_open_stream_request", return_value=FakeHTTPResponse(sse_body)):
            list(client.stream([Message(role="user", content="hello")], []))

        self.assertEqual(budget.total_tokens, 1500)
        self.assertEqual(client.remaining_tokens(), 126500)
        summary = client.token_usage_summary()
        self.assertEqual(summary["used"], 1500)
        self.assertEqual(summary["remaining"], 126500)

    def test_retryable_status_retries_with_backoff(self) -> None:
        client = _make_client()
        body = client._build_request_body([Message(role="user", content="hello")], [])
        attempts: List[int] = []

        def fake_urlopen(request: Any, timeout: float = 0) -> FakeHTTPResponse:
            attempts.append(1)
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "Service Unavailable",
                hdrs=None,
                fp=io.BytesIO(b"busy"),
            )

        with mock.patch("internal.llm.client.time.sleep") as sleep_mock:
            with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
                with self.assertRaises(LLMRetryExhaustedError):
                    client._open_stream_request(body)

        self.assertEqual(len(attempts), 4)
        sleep_mock.assert_has_calls(
            [mock.call(delay) for delay in RETRY_BACKOFF_SECONDS]
        )

    def test_non_retryable_status_fails_immediately(self) -> None:
        client = _make_client()
        body = client._build_request_body([Message(role="user", content="hello")], [])
        attempts: List[int] = []

        def fake_urlopen(request: Any, timeout: float = 0) -> FakeHTTPResponse:
            attempts.append(1)
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                hdrs=None,
                fp=io.BytesIO(b"no auth"),
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(LLMHTTPError) as ctx:
                client._open_stream_request(body)

        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(len(attempts), 1)

    def test_merge_tool_call_concatenates_arguments(self) -> None:
        bucket: Dict[int, ToolCall] = {}
        _merge_tool_call(
            bucket,
            ToolCall(
                index=0,
                id="call_1",
                function=FunctionCall(name="add", arguments='{"a":'),
            ),
        )
        _merge_tool_call(
            bucket,
            ToolCall(
                index=0,
                function=FunctionCall(arguments='1}'),
            ),
        )
        self.assertEqual(bucket[0].function.arguments, '{"a":1}')

    def test_parse_stream_chunk_usage(self) -> None:
        chunk = _parse_stream_chunk(
            {
                "choices": [{"delta": {"content": "x"}, "finish_reason": None}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        self.assertEqual(chunk.content_delta, "x")
        self.assertEqual(chunk.usage["prompt_tokens"], 10)

    def test_build_compact_request_body(self) -> None:
        client = _make_client()
        body = client._build_compact_request_body(
            [Message(role="user", content="summarize this")]
        )

        self.assertFalse(body["stream"])
        self.assertEqual(body["temperature"], TEMPERATURE)
        self.assertEqual(body["max_tokens"], COMPACT_MAX_OUTPUT_TOKENS)
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)

    def test_chat_compact_returns_completion(self) -> None:
        budget = TokenBudget(TokenUsageConfig(context_window=128000))
        client = _make_client(budget)
        response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "summary text"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 200, "completion_tokens": 50},
        }

        with mock.patch.object(client, "_post_json", return_value=response_data):
            result = client.chat_compact(
                [Message(role="user", content="compress history")]
            )

        self.assertEqual(result.content, "summary text")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(budget.total_tokens, 250)

    def test_chat_compact_uses_json_accept_header(self) -> None:
        client = _make_client()
        captured: Dict[str, Any] = {}

        def fake_request_with_retry(body: Dict[str, Any], accept: str) -> FakeHTTPResponse:
            captured["accept"] = accept
            captured["body"] = body
            return FakeHTTPResponse(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": "ok"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode("utf-8")
            )

        with mock.patch.object(client, "_request_with_retry", side_effect=fake_request_with_retry):
            client.chat_compact([Message(role="user", content="hello")])

        self.assertEqual(captured["accept"], "application/json")
        self.assertFalse(captured["body"]["stream"])


class LLMHTTPErrorMessageTests(unittest.TestCase):
    """Provider error bodies must reach the user, not just the status code."""

    def test_api_message_is_extracted_from_the_body(self) -> None:
        error = LLMHTTPError(
            400,
            "Bad Request",
            '{"error":{"message":"[] is too short - \'tools\'",'
            '"type":"invalid_request_error"}}',
        )

        self.assertEqual(error.api_message, "[] is too short - 'tools'")
        self.assertIn("[] is too short", str(error))

    def test_arrearage_account_error_is_surfaced(self) -> None:
        error = LLMHTTPError(
            400,
            "Bad Request",
            '{"error":{"message":"Access denied, please make sure your '
            'account is in good standing.","type":"Arrearage",'
            '"code":"Arrearage"}}',
        )

        self.assertIn("account is in good standing", str(error))

    def test_non_json_body_falls_back_to_raw_text(self) -> None:
        error = LLMHTTPError(500, "Server Error", "upstream exploded")

        self.assertEqual(error.api_message, "upstream exploded")

    def test_empty_body_keeps_the_plain_message(self) -> None:
        error = LLMHTTPError(503, "Unavailable")

        self.assertEqual(error.api_message, "")
        self.assertEqual(str(error), "HTTP 503: Unavailable")

    def test_code_field_is_used_when_message_is_missing(self) -> None:
        error = LLMHTTPError(
            400, "Bad Request", '{"error":{"code":"InvalidParameter"}}'
        )

        self.assertEqual(error.api_message, "InvalidParameter")


class NetworkRetryTests(unittest.TestCase):
    """Transient socket failures must be retried, not surfaced immediately."""

    def test_transient_url_error_is_retried_then_succeeds(self) -> None:
        client = _make_client()
        attempts: List[int] = []

        def flaky_urlopen(request: object, timeout: float) -> object:
            attempts.append(1)
            if len(attempts) < 3:
                raise urllib.error.URLError(OSError(2, "No such file or directory"))
            return FakeHTTPResponse(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n')

        with mock.patch.object(urllib.request, "urlopen", side_effect=flaky_urlopen):
            with mock.patch.object(time, "sleep", return_value=None):
                response = client._request_with_retry({"a": 1}, "text/event-stream")

        self.assertEqual(len(attempts), 3)
        self.assertIsNotNone(response)

    def test_persistent_url_error_raises_retry_exhausted(self) -> None:
        client = _make_client()

        def always_fail(request: object, timeout: float) -> object:
            raise urllib.error.URLError(OSError(2, "No such file or directory"))

        with mock.patch.object(urllib.request, "urlopen", side_effect=always_fail):
            with mock.patch.object(time, "sleep", return_value=None):
                with self.assertRaises(LLMRetryExhaustedError) as caught:
                    client._request_with_retry({"a": 1}, "text/event-stream")

        self.assertIn("Network error", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
