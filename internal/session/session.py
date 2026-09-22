"""JSONL session storage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from internal.types.types import FunctionCall, Message, ToolCall

SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SESSION_ID_MAX_LENGTH = 64
PREVIEW_MAX_LENGTH = 60


class SessionError(ValueError):
    """Raised when a session id or session operation is invalid."""


@dataclass(frozen=True)
class SessionSummary:
    """Metadata about one stored session file."""

    session_id: str
    path: Path
    message_count: int
    updated_at: datetime
    preview: str


def validate_session_id(session_id: str) -> str:
    """Return a normalized, filesystem-safe session id or raise ``SessionError``."""
    if not isinstance(session_id, str):
        raise SessionError(
            f"Session id must be a string, got {type(session_id).__name__}"
        )

    cleaned = session_id.strip()
    if not cleaned:
        raise SessionError("Session id must not be empty")
    if len(cleaned) > SESSION_ID_MAX_LENGTH:
        raise SessionError(
            f"Session id must be at most {SESSION_ID_MAX_LENGTH} characters"
        )
    if cleaned in {".", ".."}:
        raise SessionError(f"Invalid session id: {session_id}")
    if "/" in cleaned or "\\" in cleaned:
        raise SessionError(f"Session id must not contain path separators: {session_id}")
    if any(character.isspace() for character in cleaned):
        raise SessionError(f"Session id must not contain whitespace: {session_id}")
    if not SESSION_ID_PATTERN.fullmatch(cleaned):
        raise SessionError(
            "Session id must start with a letter or digit and contain only "
            f"letters, digits, dot, underscore, or dash: {session_id}"
        )
    return cleaned


def generate_session_id(now: Optional[datetime] = None) -> str:
    """Return a timestamp-based session id."""
    moment = now or datetime.now()
    return f"session-{moment:%Y%m%d-%H%M%S}"


class SessionStore:
    """Append-only JSONL store for conversation history."""

    def __init__(self, storage_dir: Path, session_id: str = "default") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = validate_session_id(session_id)
        self.session_file = self.storage_dir / f"{self.session_id}.jsonl"

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

    def iter_events(self) -> List[Dict[str, Any]]:
        """Return every parseable event, skipping blank and corrupted lines."""
        return _read_events(self.session_file)

    def load_messages(self) -> List[Message]:
        messages: List[Message] = []
        for event in self.iter_events():
            if event.get("type") != "message":
                continue
            message_data = event.get("message")
            if not isinstance(message_data, dict):
                continue
            messages.append(_message_from_dict(message_data))
        return messages

    def is_empty(self) -> bool:
        """Return True when the session has no stored messages yet."""
        if not self.session_file.is_file():
            return True
        return not any(
            event.get("type") == "message" for event in self.iter_events()
        )

    def delete(self) -> bool:
        """Remove the session file. Returns False when it does not exist."""
        if not self.session_file.is_file():
            return False
        self.session_file.unlink()
        return True

    @staticmethod
    def list_sessions(storage_dir: Path) -> List[SessionSummary]:
        """Return stored sessions sorted by modification time, newest first."""
        base = Path(storage_dir)
        if not base.is_dir():
            return []

        summaries: List[SessionSummary] = []
        for path in base.glob("*.jsonl"):
            try:
                session_id = validate_session_id(path.stem)
            except SessionError:
                continue
            summaries.append(_summarize_session(path, session_id))

        summaries.sort(
            key=lambda summary: (summary.updated_at, summary.session_id),
            reverse=True,
        )
        return summaries


def _read_events(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL events, skipping blank and corrupted lines."""
    if not path.is_file():
        return []

    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _summarize_session(path: Path, session_id: str) -> SessionSummary:
    message_count = 0
    preview = ""
    for event in _read_events(path):
        if event.get("type") != "message":
            continue
        message_count += 1
        message_data = event.get("message")
        if preview or not isinstance(message_data, dict):
            continue
        if message_data.get("role") == "system":
            continue
        content = message_data.get("content")
        if isinstance(content, str) and content.strip():
            preview = _truncate_preview(content.strip())

    return SessionSummary(
        session_id=session_id,
        path=path,
        message_count=message_count,
        updated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        preview=preview,
    )


def _truncate_preview(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= PREVIEW_MAX_LENGTH:
        return collapsed
    return collapsed[: PREVIEW_MAX_LENGTH - 3] + "..."


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


__all__ = [
    "SESSION_ID_MAX_LENGTH",
    "SESSION_ID_PATTERN",
    "SessionError",
    "SessionStore",
    "SessionSummary",
    "generate_session_id",
    "validate_session_id",
]
