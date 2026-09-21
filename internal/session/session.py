"""JSONL session storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from internal.types.types import FunctionCall, Message, ToolCall


class SessionStore:
    """Append-only JSONL store for conversation history."""

    def __init__(self, storage_dir: Path, session_id: str = "default") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.session_file = self.storage_dir / f"{session_id}.jsonl"

    def append_event(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.session_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_message(self, message: Message) -> None:
        self.append_event({"type": "message", "message": _message_to_dict(message)})

    def append_messages(self, messages: List[Message]) -> None:
        for message in messages:
            self.append_message(message)

    def load_messages(self) -> List[Message]:
        if not self.session_file.is_file():
            return []

        messages: List[Message] = []
        with self.session_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "message":
                    continue
                message_data = event.get("message")
                if not isinstance(message_data, dict):
                    continue
                messages.append(_message_from_dict(message_data))
        return messages


def _message_to_dict(message: Message) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "role": message.role,
        "content": message.content,
        "name": message.name,
        "tool_call_id": message.tool_call_id,
        "metadata": message.metadata,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "index": tool_call.index,
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload


def _message_from_dict(data: Dict[str, Any]) -> Message:
    tool_calls: List[ToolCall] = []
    for raw in data.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function") or {}
        tool_calls.append(
            ToolCall(
                index=raw.get("index"),
                id=raw.get("id") or "",
                type=raw.get("type") or "function",
                function=FunctionCall(
                    name=function.get("name") or "",
                    arguments=function.get("arguments") or "",
                ),
            )
        )

    return Message(
        role=data.get("role") or "user",
        content=data.get("content"),
        name=data.get("name"),
        tool_call_id=data.get("tool_call_id"),
        tool_calls=tool_calls,
        metadata=data.get("metadata") or {},
    )


__all__ = ["SessionStore"]
