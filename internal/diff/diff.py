"""Diff models for tool execution results and future TUI previews."""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiffLine:
    """One line in a file diff."""

    kind: str
    text: str
    old_line_no: Optional[int] = None
    new_line_no: Optional[int] = None


@dataclass
class FileDiff:
    """Before/after diff for a single file change."""

    path: str
    operation: str
    old_content: str
    new_content: str
    unified: str = ""
    lines: List[DiffLine] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolExecutorResult:
    """Structured executor output: summary for the LLM plus optional diff metadata."""

    summary: str
    file_diff: Optional[FileDiff] = None

    @classmethod
    def from_text(cls, summary: str) -> ToolExecutorResult:
        return cls(summary=summary)

    def to_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"summary": self.summary}
        if self.file_diff is not None:
            metadata["file_diff"] = self.file_diff.to_dict()
        return metadata


def build_diff_lines(old_content: str, new_content: str) -> List[DiffLine]:
    """Build line-level diff hunks from two file contents."""
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    result: List[DiffLine] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset, line_index in enumerate(range(i1, i2)):
                new_index = j1 + offset
                result.append(
                    DiffLine(
                        kind="context",
                        old_line_no=line_index + 1,
                        new_line_no=new_index + 1,
                        text=old_lines[line_index],
                    )
                )
        elif tag == "delete":
            for line_index in range(i1, i2):
                result.append(
                    DiffLine(
                        kind="remove",
                        old_line_no=line_index + 1,
                        text=old_lines[line_index],
                    )
                )
        elif tag == "insert":
            for line_index in range(j1, j2):
                result.append(
                    DiffLine(
                        kind="add",
                        new_line_no=line_index + 1,
                        text=new_lines[line_index],
                    )
                )
        elif tag == "replace":
            for line_index in range(i1, i2):
                result.append(
                    DiffLine(
                        kind="remove",
                        old_line_no=line_index + 1,
                        text=old_lines[line_index],
                    )
                )
            for line_index in range(j1, j2):
                result.append(
                    DiffLine(
                        kind="add",
                        new_line_no=line_index + 1,
                        text=new_lines[line_index],
                    )
                )

    return result


def build_unified_diff(path: str, old_content: str, new_content: str) -> str:
    """Return a unified diff string for two file contents."""
    return "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def build_file_diff(
    path: str,
    operation: str,
    old_content: str,
    new_content: str,
) -> FileDiff:
    """Build a complete file diff model."""
    return FileDiff(
        path=path,
        operation=operation,
        old_content=old_content,
        new_content=new_content,
        unified=build_unified_diff(path, old_content, new_content),
        lines=build_diff_lines(old_content, new_content),
    )


__all__ = [
    "DiffLine",
    "FileDiff",
    "ToolExecutorResult",
    "build_diff_lines",
    "build_file_diff",
    "build_unified_diff",
]
