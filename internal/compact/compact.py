"""Deterministic context compaction helpers."""

from __future__ import annotations

import json
from typing import List, Tuple

from internal.types.types import Message

COMPACT_REMAINING_RATIO = 0.10
TARGET_MAX_USAGE_RATIO = 0.90
MIN_HOT_NON_SYSTEM_MESSAGES = 2


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(message: Message) -> int:
    total = estimate_text_tokens(message.content or "")
    if message.tool_calls:
        total += estimate_text_tokens(json.dumps(_serialize_tool_calls(message), ensure_ascii=False))
    total += estimate_text_tokens(message.tool_call_id or "")
    total += estimate_text_tokens(message.name or "")
    return total


def estimate_messages_tokens(messages: List[Message]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def _serialize_tool_calls(message: Message) -> List[dict]:
    payload = []
    for tool_call in message.tool_calls:
        payload.append(
            {
                "id": tool_call.id,
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
        )
    return payload


def should_force_compact(remaining_tokens: int, limit: int) -> bool:
    if limit <= 0:
        return False
    return (remaining_tokens / limit) < COMPACT_REMAINING_RATIO


def deterministic_compact(
    messages: List[Message],
    context_window_limit: int,
) -> Tuple[List[Message], List[Message]]:
    """Drop oldest non-system messages until context usage is back under target."""
    system_messages = [message for message in messages if message.role == "system"]
    hot_rest = [message for message in messages if message.role != "system"]
    cold: List[Message] = []

    def hot_messages() -> List[Message]:
        return system_messages + hot_rest

    while (
        hot_rest
        and len(hot_rest) > MIN_HOT_NON_SYSTEM_MESSAGES
        and estimate_messages_tokens(hot_messages()) / context_window_limit
        > TARGET_MAX_USAGE_RATIO
    ):
        cold.append(hot_rest.pop(0))

    return hot_messages(), cold


__all__ = [
    "COMPACT_REMAINING_RATIO",
    "MIN_HOT_NON_SYSTEM_MESSAGES",
    "TARGET_MAX_USAGE_RATIO",
    "deterministic_compact",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "estimate_text_tokens",
    "should_force_compact",
]
