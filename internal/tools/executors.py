"""Tool executor implementations (not exposed to the LLM)."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from internal.diff.diff import ToolExecutorResult, build_file_diff
from internal.tools.path import (
    resolve_tool_results_path,
    resolve_workspace_path,
    resolve_workspace_path_info,
    tool_results_dir,
)

MAX_READ_LINES = 2000
DEFAULT_DISK_LIST_LIMIT = 20
DEFAULT_SHELL_TIMEOUT = 60.0


@dataclass
class ToolContext:
    """Runtime dependencies for tool executors."""

    workspace_root: Path
    tool_output_path: Path


def read_file_executor(args: Dict[str, Any], ctx: ToolContext) -> ToolExecutorResult:
    path = resolve_workspace_path(ctx.workspace_root, str(args["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {args['path']}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    offset = int(args.get("offset") or 1)
    limit = args.get("limit")
    if limit is not None:
        limit = int(limit)

    if offset < 1:
        raise ValueError("offset must be >= 1")

    start = offset - 1
    if limit is None:
        selected = lines[start : start + MAX_READ_LINES]
        truncated = len(lines) > start + MAX_READ_LINES
    else:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        selected = lines[start : start + limit]
        truncated = start + limit < len(lines)

    numbered = [f"{start + index + 1}|{line}" for index, line in enumerate(selected)]
    result = "\n".join(numbered)
    if truncated:
        result += (
            f"\n\n[truncated: showing {len(selected)} line(s); "
            "use offset/limit or read_regex/read_disk_data for more]"
        )
    return ToolExecutorResult.from_text(result)


def write_file_executor(args: Dict[str, Any], ctx: ToolContext) -> ToolExecutorResult:
    resolved = resolve_workspace_path_info(ctx.workspace_root, str(args["path"]))
    path = resolved.absolute
    content = str(args.get("content", ""))
    relative_path = resolved.relative
    old_content = path.read_text(encoding="utf-8") if path.is_file() else ""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    file_diff = build_file_diff(relative_path, "write", old_content, content)
    return ToolExecutorResult(
        summary=f"Wrote {len(content)} character(s) to {relative_path}",
        file_diff=file_diff,
    )


def replace_line_executor(args: Dict[str, Any], ctx: ToolContext) -> ToolExecutorResult:
    resolved = resolve_workspace_path_info(ctx.workspace_root, str(args["path"]))
    path = resolved.absolute
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {args['path']}")

    line_number = int(args["line_number"])
    new_content = str(args.get("new_content", ""))
    relative_path = resolved.relative
    if line_number < 1:
        raise ValueError("line_number must be >= 1")

    old_content = path.read_text(encoding="utf-8")
    lines = old_content.splitlines()
    if line_number > len(lines):
        raise ValueError(
            f"line_number {line_number} out of range (file has {len(lines)} line(s))"
        )

    lines[line_number - 1] = new_content
    new_file_content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(new_file_content, encoding="utf-8")

    file_diff = build_file_diff(
        relative_path,
        "replace_line",
        old_content,
        new_file_content,
    )
    return ToolExecutorResult(
        summary=f"Replaced line {line_number} in {relative_path}",
        file_diff=file_diff,
    )


def list_dir_executor(args: Dict[str, Any], ctx: ToolContext) -> ToolExecutorResult:
    relative = str(args.get("path") or ".")
    recursive = bool(args.get("recursive", False))
    directory = resolve_workspace_path(ctx.workspace_root, relative)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {relative}")

    entries: List[str] = []
    if recursive:
        for item in sorted(directory.rglob("*")):
            rel = item.relative_to(ctx.workspace_root.resolve())
            suffix = "/" if item.is_dir() else ""
            entries.append(f"{rel.as_posix()}{suffix}")
    else:
        for item in sorted(directory.iterdir()):
            suffix = "/" if item.is_dir() else ""
            entries.append(f"{item.name}{suffix}")

    if not entries:
        return ToolExecutorResult.from_text(f"(empty directory: {relative})")
    return ToolExecutorResult.from_text("\n".join(entries))


def read_regex_executor(args: Dict[str, Any], ctx: ToolContext) -> ToolExecutorResult:
    path = resolve_workspace_path(ctx.workspace_root, str(args["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {args['path']}")

    pattern = str(args["pattern"])
    max_matches = int(args.get("max_matches") or 100)
    if max_matches < 1:
        raise ValueError("max_matches must be >= 1")

    regex = re.compile(pattern)
    matches: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        found = regex.search(line)
        if found is None:
            continue
        matches.append(
            {
                "line": line_number,
                "content": line,
                "groups": list(found.groups()),
            }
        )
        if len(matches) >= max_matches:
            break

    payload = {"pattern": pattern, "match_count": len(matches), "matches": matches}
    return ToolExecutorResult.from_text(json.dumps(payload, ensure_ascii=False, indent=2))


def run_shell_executor(args: Dict[str, Any], ctx: ToolContext) -> ToolExecutorResult:
    command = str(args["command"])
    cwd_arg = args.get("cwd")
    timeout = float(args.get("timeout_seconds") or DEFAULT_SHELL_TIMEOUT)
    if timeout <= 0:
        raise ValueError("timeout_seconds must be > 0")

    cwd = (
        resolve_workspace_path(ctx.workspace_root, str(cwd_arg))
        if cwd_arg
        else ctx.workspace_root.resolve()
    )
    if not cwd.is_dir():
        raise NotADirectoryError(f"Not a directory: {cwd_arg or '.'}")

    completed = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    payload = {
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    return ToolExecutorResult.from_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _slice_lines(text: str, offset: Optional[int], limit: Optional[int]) -> str:
    lines = text.splitlines()
    start = (int(offset) - 1) if offset else 0
    if start < 0:
        raise ValueError("offset must be >= 1")
    if limit is None:
        selected = lines[start:]
    else:
        limit_value = int(limit)
        if limit_value < 1:
            raise ValueError("limit must be >= 1")
        selected = lines[start : start + limit_value]
    numbered = [f"{start + index + 1}|{line}" for index, line in enumerate(selected)]
    return "\n".join(numbered)


def _find_tool_result_by_call_id(results_root: Path, tool_call_id: str) -> Path:
    matches = sorted(results_root.glob(f"*_{tool_call_id}.txt"))
    if not matches:
        raise FileNotFoundError(
            f"No archived tool output found for tool_call_id '{tool_call_id}'"
        )
    if len(matches) > 1:
        return matches[0]
    return matches[0]


def _list_recent_tool_results(results_root: Path, limit: int = DEFAULT_DISK_LIST_LIMIT) -> str:
    files = sorted(
        (path for path in results_root.glob("*.txt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]

    entries: List[Dict[str, Any]] = []
    for path in files:
        stem = path.stem
        tool_call_id = stem.rsplit("_", 1)[-1] if "_" in stem else ""
        tool_name = stem.rsplit("_", 1)[0] if "_" in stem else stem
        entries.append(
            {
                "filename": path.name,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {"count": len(entries), "entries": entries}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def read_disk_data_executor(args: Dict[str, Any], ctx: ToolContext) -> ToolExecutorResult:
    results_root = tool_results_dir(ctx.tool_output_path)
    tool_call_id = args.get("tool_call_id")
    filename = args.get("filename")
    offset = args.get("offset")
    limit = args.get("limit")

    if not tool_call_id and not filename:
        return ToolExecutorResult.from_text(_list_recent_tool_results(results_root))

    if tool_call_id:
        target = _find_tool_result_by_call_id(results_root, str(tool_call_id))
    else:
        target = resolve_tool_results_path(ctx.tool_output_path, str(filename))

    if not target.is_file():
        raise FileNotFoundError(f"Archived tool output not found: {target.name}")

    content = target.read_text(encoding="utf-8")
    if offset is not None or limit is not None:
        return ToolExecutorResult.from_text(_slice_lines(content, offset, limit))
    return ToolExecutorResult.from_text(content)


EXECUTOR_MAP = {
    "read_file": read_file_executor,
    "write_file": write_file_executor,
    "replace_line": replace_line_executor,
    "list_dir": list_dir_executor,
    "read_regex": read_regex_executor,
    "run_shell": run_shell_executor,
    "read_disk_data": read_disk_data_executor,
}


__all__ = [
    "EXECUTOR_MAP",
    "ToolContext",
    "list_dir_executor",
    "read_disk_data_executor",
    "read_file_executor",
    "read_regex_executor",
    "replace_line_executor",
    "run_shell_executor",
    "write_file_executor",
]
