"""Skill definitions, keyword matching, and system-prompt rendering."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - declared project dependency
    yaml = None


class SkillError(ValueError):
    """Raised when a skill definition is invalid."""


@dataclass(frozen=True)
class Skill:
    """Instructions and tools activated by one or more keywords."""

    name: str
    keywords: Tuple[str, ...]
    tools: Tuple[str, ...]
    instructions: str
    description: str = ""

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise SkillError("Skill name must not be empty")
        keywords = _clean_string_tuple(self.keywords, "keywords")
        if not keywords:
            raise SkillError(f"Skill '{name}' must define at least one keyword")
        tools = _clean_string_tuple(self.tools, "tools")
        instructions = self.instructions.strip()
        if not instructions:
            raise SkillError(f"Skill '{name}' instructions must not be empty")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "keywords", keywords)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "description", self.description.strip())

    @classmethod
    def from_dict(cls, data: Dict[str, Any], instructions: str = "") -> "Skill":
        if not isinstance(data, dict):
            raise SkillError("Skill metadata must be a mapping")
        return cls(
            name=_require_string(data, "name"),
            description=_optional_string(data, "description"),
            keywords=_string_sequence(data.get("keywords"), "keywords"),
            tools=_string_sequence(data.get("tools", ()), "tools"),
            instructions=instructions or _optional_string(data, "instructions"),
        )

    @classmethod
    def from_markdown(cls, path: Path) -> "Skill":
        return load_skill_markdown(path)


@dataclass(frozen=True)
class SkillMatch:
    """Skills and tool names activated by a user query."""

    skills: Tuple[Skill, ...] = field(default_factory=tuple)
    tool_names: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def skill_names(self) -> Tuple[str, ...]:
        return tuple(skill.name for skill in self.skills)


class SkillManager:
    """Ordered skill registry with deterministic keyword routing."""

    def __init__(self, skills: Optional[Iterable[Skill]] = None) -> None:
        self._skills: Dict[str, Skill] = {}
        for skill in skills or ():
            self.register(skill)

    def register(self, skill: Skill) -> Skill:
        existing = self._skills.get(skill.name)
        if existing is not None and existing != skill:
            raise SkillError(f"Skill already registered with different content: {skill.name}")
        self._skills[skill.name] = skill
        return skill

    def load_markdown(self, path: Path) -> Skill:
        return self.register(load_skill_markdown(path))

    def list_skills(self) -> List[Skill]:
        return list(self._skills.values())

    def match(self, text: str) -> SkillMatch:
        normalized_text = _normalize(text)
        matched = tuple(
            skill
            for skill in self._skills.values()
            if any(_keyword_matches(normalized_text, keyword) for keyword in skill.keywords)
        )
        tool_names = tuple(
            dict.fromkeys(tool_name for skill in matched for tool_name in skill.tools)
        )
        return SkillMatch(skills=matched, tool_names=tool_names)

    def render_prompt(self, names: Optional[Sequence[str]] = None) -> str:
        selected = self._select(names)
        sections: List[str] = []
        for skill in selected:
            marker = _prompt_marker(skill.name)
            heading = f"## Skill: {skill.name}"
            description = f"\n{skill.description}" if skill.description else ""
            sections.append(
                f"{marker}\n{heading}{description}\n\n{skill.instructions}\n"
                f"<!-- /pyagent-skill:{skill.name} -->"
            )
        return "\n\n".join(sections)

    def merge_system_prompt(
        self,
        base_prompt: str,
        names: Optional[Sequence[str]] = None,
    ) -> str:
        selected = [
            skill
            for skill in self._select(names)
            if _prompt_marker(skill.name) not in base_prompt
        ]
        if not selected:
            return base_prompt
        addition = SkillManager(selected).render_prompt()
        base = base_prompt.rstrip()
        return f"{base}\n\n# Imported Skills\n\n{addition}" if base else addition

    def _select(self, names: Optional[Sequence[str]]) -> List[Skill]:
        if names is None:
            return self.list_skills()
        selected: List[Skill] = []
        seen = set()
        for name in names:
            if name in seen:
                continue
            try:
                selected.append(self._skills[name])
            except KeyError as exc:
                raise SkillError(f"Unknown skill: {name}") from exc
            seen.add(name)
        return selected

    def __bool__(self) -> bool:
        return bool(self._skills)


def load_skill_markdown(path: Path) -> Skill:
    """Load a skill from YAML front matter followed by Markdown instructions."""
    skill_path = Path(path)
    text = skill_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"Skill file must start with YAML front matter: {skill_path}")

    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing_index is None:
        raise SkillError(f"Skill front matter is not closed: {skill_path}")
    front_matter = "\n".join(lines[1:closing_index])
    if yaml is None:
        metadata = _parse_simple_front_matter(front_matter, skill_path)
    else:
        try:
            metadata = yaml.safe_load(front_matter) or {}
        except yaml.YAMLError as exc:
            raise SkillError(f"Invalid YAML front matter in {skill_path}: {exc}") from exc
    instructions = "\n".join(lines[closing_index + 1 :]).strip()
    return Skill.from_dict(metadata, instructions=instructions)


def import_skills_to_system_prompt(
    base_prompt: str,
    skills: Iterable[Skill],
    names: Optional[Sequence[str]] = None,
) -> str:
    """Return a system prompt containing the selected skill instructions."""
    return SkillManager(skills).merge_system_prompt(base_prompt, names)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _keyword_matches(normalized_text: str, keyword: str) -> bool:
    normalized_keyword = _normalize(keyword)
    if not normalized_keyword:
        return False
    if re.fullmatch(r"[a-z0-9_]+(?:[ -]+[a-z0-9_]+)*", normalized_keyword):
        phrase = r"\s+".join(
            re.escape(part) for part in re.split(r"[ -]+", normalized_keyword)
        )
        return (
            re.search(
                rf"(?<![a-z0-9_]){phrase}(?![a-z0-9_])",
                normalized_text,
            )
            is not None
        )
    return normalized_keyword in normalized_text


def _clean_string_tuple(values: Iterable[str], field_name: str) -> Tuple[str, ...]:
    if isinstance(values, str):
        raise SkillError(f"Skill {field_name} must be a sequence of strings")
    cleaned: List[str] = []
    for value in values:
        if not isinstance(value, str):
            raise SkillError(f"Skill {field_name} must contain only strings")
        stripped = value.strip()
        if stripped and stripped not in cleaned:
            cleaned.append(stripped)
    return tuple(cleaned)


def _string_sequence(value: Any, field_name: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SkillError(f"Skill {field_name} must be a list")
    return _clean_string_tuple(value, field_name)


def _require_string(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillError(f"Skill {key} must be a non-empty string")
    return value


def _optional_string(data: Dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise SkillError(f"Skill {key} must be a string")
    return value


def _prompt_marker(name: str) -> str:
    return f"<!-- pyagent-skill:{name} -->"


def _parse_simple_front_matter(text: str, path: Path) -> Dict[str, Any]:
    """Parse the string/list YAML subset used by skill metadata."""
    result: Dict[str, Any] = {}
    active_list: Optional[List[str]] = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if active_list is None:
                raise SkillError(f"Invalid list in skill front matter: {path}")
            active_list.append(stripped[2:].strip().strip("\"'"))
            continue

        key, separator, raw_value = stripped.partition(":")
        if not separator or not key.strip():
            raise SkillError(f"Invalid skill front matter line in {path}: {stripped}")
        key = key.strip()
        value = raw_value.strip()
        if not value:
            active_list = []
            result[key] = active_list
            continue
        active_list = None
        result[key] = value.strip("\"'")
    return result


__all__ = [
    "Skill",
    "SkillError",
    "SkillManager",
    "SkillMatch",
    "import_skills_to_system_prompt",
    "load_skill_markdown",
]
