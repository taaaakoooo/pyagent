"""Configuration loading, API key resolution, and token budget tracking."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency at runtime
    yaml = None

T = TypeVar("T")

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class ConfigError(Exception):
    """Raised when configuration is missing, invalid, or incomplete."""


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}****{value[-4:]}"


def _read_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the small YAML subset used by config.yaml without PyYAML."""

    lines: List[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    root: Dict[str, Any] = {}
    stack: List[tuple[int, Any]] = [(-1, root)]
    index = 0

    while index < len(lines):
        indent, line = lines[index]
        while stack and indent <= stack[-1][0]:
            stack.pop()

        container = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(container, list):
                raise ConfigError("Invalid YAML list structure")
            container.append(_parse_scalar(line[2:]))
            index += 1
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ConfigError(f"Invalid YAML line: {line}")

        if value:
            if not isinstance(container, dict):
                raise ConfigError("Invalid YAML mapping structure")
            container[key] = _parse_scalar(value)
            index += 1
            continue

        next_line = lines[index + 1][1] if index + 1 < len(lines) else ""
        new_container: Any = [] if next_line.startswith("- ") else {}
        if not isinstance(container, dict):
            raise ConfigError("Invalid YAML mapping structure")
        container[key] = new_container
        stack.append((indent, new_container))
        index += 1

    return root


def _load_yaml_text(text: str) -> Dict[str, Any]:
    if yaml is not None:
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = _parse_simple_yaml(text)

    if not isinstance(loaded, dict):
        raise ConfigError("Config root must be a mapping")
    return loaded


def _read_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _load_yaml_text(handle.read())


def _merge_dict(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _from_dict(cls: Type[T], data: Optional[Dict[str, Any]]) -> T:
    data = data or {}
    allowed = {item.name for item in fields(cls)}
    kwargs = {key: value for key, value in data.items() if key in allowed}
    return cls(**kwargs)


@dataclass
class LLMConfig:
    provider: str = "dashscope"
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-32b"
    api_key_env: str = "DASHSCOPE_API_KEY"
    fallback_api_key_envs: List[str] = field(
        default_factory=lambda: ["OPENAI_API_KEY"]
    )
    timeout_seconds: float = 180.0
    stream: bool = True
    api_key: str = field(default="", repr=False)

    @property
    def masked_api_key(self) -> str:
        return _mask_secret(self.api_key)

    def candidate_env_names(self) -> List[str]:
        names: List[str] = []
        if self.api_key_env:
            names.append(self.api_key_env)
        for name in self.fallback_api_key_envs:
            if name and name not in names:
                names.append(name)
        return names

    def resolve_api_key(self) -> str:
        for name in self.candidate_env_names():
            value = _read_env(name)
            if value:
                self.api_key = value
                return value

        env_list = ", ".join(self.candidate_env_names())
        raise ConfigError(
            f"API key not found. Set one of these environment variables: {env_list}"
        )

    def __repr__(self) -> str:
        return (
            "LLMConfig("
            f"provider={self.provider!r}, "
            f"api_base={self.api_base!r}, "
            f"model={self.model!r}, "
            f"api_key_env={self.api_key_env!r}, "
            f"api_key={self.masked_api_key!r}, "
            f"stream={self.stream!r})"
        )


@dataclass
class AgentConfig:
    max_turns: int = 15
    workspace_root: str = "."


@dataclass
class TokenUsageConfig:
    context_window: int = 128000
    max_session_tokens: Optional[int] = None
    warning_threshold: float = 0.7
    warn_once: bool = True


@dataclass
class SessionConfig:
    storage_dir: str = ".sessions"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    token_usage: TokenUsageConfig = field(default_factory=TokenUsageConfig)
    session: SessionConfig = field(default_factory=SessionConfig)

    def resolve_api_key(self) -> str:
        return self.llm.resolve_api_key()

    def validate(self) -> None:
        if self.agent.max_turns < 1:
            raise ConfigError("agent.max_turns must be >= 1")

        if not (0 < self.token_usage.warning_threshold <= 1):
            raise ConfigError("token_usage.warning_threshold must be in (0, 1]")

        if self.token_usage.context_window <= 0:
            raise ConfigError("token_usage.context_window must be > 0")

        if (
            self.token_usage.max_session_tokens is not None
            and self.token_usage.max_session_tokens <= 0
        ):
            raise ConfigError("token_usage.max_session_tokens must be > 0 when set")


@dataclass
class TokenBudget:
    config: TokenUsageConfig
    prompt_tokens: int = 0
    completion_tokens: int = 0
    warned: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def limit(self) -> int:
        if self.config.max_session_tokens is not None:
            return self.config.max_session_tokens
        return self.config.context_window

    @property
    def usage_ratio(self) -> float:
        if self.limit <= 0:
            return 0.0
        return self.total_tokens / self.limit

    def add_usage(self, usage: Dict[str, int]) -> Optional[str]:
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        self.prompt_tokens += max(prompt, 0)
        self.completion_tokens += max(completion, 0)

        if self.usage_ratio < self.config.warning_threshold:
            return None

        if self.config.warn_once and self.warned:
            return None

        self.warned = True
        percent = int(round(self.usage_ratio * 100))
        return (
            f"Token 用量已达 {percent}%（{self.total_tokens}/{self.limit}），"
            "建议压缩上下文或结束会话。"
        )


def _default_config_dict() -> Dict[str, Any]:
    return {
        "llm": {
            "provider": LLMConfig.provider,
            "api_base": LLMConfig.api_base,
            "model": LLMConfig.model,
            "api_key_env": LLMConfig.api_key_env,
            "fallback_api_key_envs": list(LLMConfig().fallback_api_key_envs),
            "timeout_seconds": LLMConfig.timeout_seconds,
            "stream": LLMConfig.stream,
        },
        "agent": {
            "max_turns": AgentConfig.max_turns,
            "workspace_root": AgentConfig.workspace_root,
        },
        "token_usage": {
            "context_window": TokenUsageConfig.context_window,
            "max_session_tokens": TokenUsageConfig.max_session_tokens,
            "warning_threshold": TokenUsageConfig.warning_threshold,
            "warn_once": TokenUsageConfig.warn_once,
        },
        "session": {
            "storage_dir": SessionConfig.storage_dir,
        },
    }


def load_config(path: Optional[str] = None, resolve_key: bool = True) -> AppConfig:
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    merged = _merge_dict(_default_config_dict(), _read_yaml_file(config_path))
    config = AppConfig(
        llm=_from_dict(LLMConfig, merged.get("llm")),
        agent=_from_dict(AgentConfig, merged.get("agent")),
        token_usage=_from_dict(TokenUsageConfig, merged.get("token_usage")),
        session=_from_dict(SessionConfig, merged.get("session")),
    )
    config.validate()

    if resolve_key:
        config.resolve_api_key()

    return config


__all__ = [
    "AgentConfig",
    "AppConfig",
    "ConfigError",
    "LLMConfig",
    "SessionConfig",
    "TokenBudget",
    "TokenUsageConfig",
    "load_config",
]
