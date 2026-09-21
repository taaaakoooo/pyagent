"""Workspace path resolution and sandbox checks for local tools."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class PathSandboxError(ValueError):
    """Raised when a path escapes an allowed root or is invalid."""


@dataclass(frozen=True)
class ResolvedPath:
    """A path resolved to an absolute location under an allowed root."""

    absolute: Path
    relative: str
    root: Path


def sanitize_path_input(user_path: str) -> str:
    """Normalize and validate a user-supplied path string."""
    if not isinstance(user_path, str):
        raise PathSandboxError(f"Path must be a string, got {type(user_path).__name__}")

    cleaned = user_path.strip()
    if not cleaned:
        raise PathSandboxError("Path must not be empty")

    if "\0" in cleaned:
        raise PathSandboxError("Path must not contain null bytes")

    normalized = cleaned.replace("\\", "/")

    if normalized != "." and all(part == ".." for part in PurePosixPath(normalized).parts):
        raise PathSandboxError(f"Invalid path: {user_path}")

    return normalized


def _reject_unc_path(path: Path) -> None:
    raw = str(path)
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise PathSandboxError(f"UNC paths are not allowed: {path}")


def _reject_cross_drive(root: Path, candidate: Path) -> None:
    if not sys.platform.startswith("win"):
        return
    root_drive = Path(root.drive or root.anchor)
    candidate_drive = Path(candidate.drive or candidate.anchor)
    if root_drive != candidate_drive:
        raise PathSandboxError(
            f"Path '{candidate}' is on a different drive than root '{root}'"
        )


def ensure_under_root(root: Path, path: Path) -> Path:
    """Return ``path`` if it is inside ``root``; otherwise raise."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PathSandboxError(
            f"Path '{path}' is outside allowed root '{root}'"
        ) from exc
    return resolved_path


def _relative_display_path(root: Path, absolute: Path) -> str:
    relative = absolute.relative_to(root)
    if relative.as_posix() in {"", "."}:
        return "."
    return relative.as_posix()


def resolve_safe_path(root: Path, user_path: str) -> ResolvedPath:
    """Resolve ``user_path`` to an absolute path under ``root``."""
    normalized = sanitize_path_input(user_path)
    resolved_root = root.resolve()

    candidate = Path(normalized)
    _reject_unc_path(candidate)

    if candidate.is_absolute():
        _reject_cross_drive(resolved_root, candidate)
        joined = candidate
    else:
        joined = resolved_root / candidate

    absolute = ensure_under_root(resolved_root, joined)
    relative = _relative_display_path(resolved_root, absolute)
    return ResolvedPath(absolute=absolute, relative=relative, root=resolved_root)


def resolve_workspace_path_info(workspace_root: Path, user_path: str) -> ResolvedPath:
    """Resolve a workspace path and return absolute + relative forms."""
    return resolve_safe_path(workspace_root, user_path)


def resolve_workspace_path(workspace_root: Path, user_path: str) -> Path:
    """Resolve a workspace path to an absolute path under the workspace root."""
    return resolve_workspace_path_info(workspace_root, user_path).absolute


def resolve_tool_results_path(tool_output_path: Path, filename: str) -> Path:
    """Resolve a path under ``{tool_output_path}/tool_results``."""
    results_root = tool_results_dir(tool_output_path)
    return resolve_safe_path(results_root, filename).absolute


def tool_results_dir(tool_output_path: Path) -> Path:
    """Return the tool results directory, creating it if needed."""
    results_root = (tool_output_path / "tool_results").resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def ensure_tool_output_in_workspace(workspace_root: Path, tool_output_path: Path) -> None:
    """Ensure archived tool output stays inside the workspace."""
    ensure_under_root(workspace_root.resolve(), tool_output_path.resolve())


__all__ = [
    "PathSandboxError",
    "ResolvedPath",
    "ensure_tool_output_in_workspace",
    "ensure_under_root",
    "resolve_safe_path",
    "resolve_tool_results_path",
    "resolve_workspace_path",
    "resolve_workspace_path_info",
    "sanitize_path_input",
    "tool_results_dir",
]
