"""Diff utilities for tool output and TUI previews."""

from internal.diff.diff import (
    DiffLine,
    FileDiff,
    ToolExecutorResult,
    build_diff_lines,
    build_file_diff,
    build_unified_diff,
)

__all__ = [
    "DiffLine",
    "FileDiff",
    "ToolExecutorResult",
    "build_diff_lines",
    "build_file_diff",
    "build_unified_diff",
]
