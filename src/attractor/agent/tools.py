"""Tool definitions and registry for the Coding Agent Loop (Section 3 of coding-agent-loop-spec.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .env import ExecutionEnvironment


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class RegisteredTool:
    definition: ToolDefinition
    executor: Callable[[dict[str, Any], ExecutionEnvironment], str]


class ToolRegistry:
    """Manages available tools, argument schema validation, and execution pipeline."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, arguments: dict[str, Any], env: ExecutionEnvironment) -> str:
        tool = self.get(name)
        if not tool:
            raise KeyError(f"Tool not found: {name}")
        return tool.executor(arguments, env)


def create_standard_tool_registry() -> ToolRegistry:
    """Instantiates the standard Provider-Aligned Toolset from Section 3."""
    reg = ToolRegistry()

    # 1. read_file
    def _read_file_exec(args: dict[str, Any], env: ExecutionEnvironment) -> str:
        path = str(args.get("file_path") or args.get("path") or "")
        offset = args.get("offset")
        limit = args.get("limit")
        return env.read_file(
            path,
            offset=int(offset) if offset is not None else None,
            limit=int(limit) if limit is not None else None,
        )

    reg.register(
        RegisteredTool(
            definition=ToolDefinition(
                name="read_file",
                description="Read a file from the workspace with line numbers ('NNN | content').",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative or absolute path"},
                        "offset": {"type": "integer", "description": "1-based starting line number"},
                        "limit": {"type": "integer", "description": "Maximum lines to read"},
                    },
                    "required": ["file_path"],
                },
            ),
            executor=_read_file_exec,
        )
    )

    # 2. write_file
    def _write_file_exec(args: dict[str, Any], env: ExecutionEnvironment) -> str:
        path = str(args.get("file_path") or args.get("path") or "")
        content = str(args.get("content", ""))
        return env.write_file(path, content)

    reg.register(
        RegisteredTool(
            definition=ToolDefinition(
                name="write_file",
                description="Write full content to a file in the workspace. Creates parent directories as needed.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Target file path"},
                        "content": {"type": "string", "description": "Complete file text"},
                    },
                    "required": ["file_path", "content"],
                },
            ),
            executor=_write_file_exec,
        )
    )

    # 3. edit_file
    def _edit_file_exec(args: dict[str, Any], env: ExecutionEnvironment) -> str:
        path = str(args.get("file_path") or args.get("path") or "")
        old_string = str(args.get("old_string", ""))
        new_string = str(args.get("new_string", ""))
        replace_all = bool(args.get("replace_all", False))
        return env.edit_file(path, old_string, new_string, replace_all=replace_all)

    reg.register(
        RegisteredTool(
            definition=ToolDefinition(
                name="edit_file",
                description="Replace an exact string occurrence in an existing file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "File to modify"},
                        "old_string": {"type": "string", "description": "Exact text to find"},
                        "new_string": {"type": "string", "description": "Replacement text"},
                        "replace_all": {"type": "boolean", "description": "Whether to replace all occurrences"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            ),
            executor=_edit_file_exec,
        )
    )

    # 4. shell (exec_command)
    def _shell_exec(args: dict[str, Any], env: ExecutionEnvironment) -> str:
        cmd = str(args.get("command", ""))
        timeout_ms = int(args.get("timeout_ms", 30000))
        res = env.exec_command(cmd, timeout_ms=timeout_ms)
        output = res.combined
        return f"[exit code: {res.exit_code}, duration: {res.duration_ms}ms]\n{output}"

    reg.register(
        RegisteredTool(
            definition=ToolDefinition(
                name="shell",
                description="Execute a shell command (e.g. pytest, uv, git) inside the workspace.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command line string"},
                        "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds"},
                    },
                    "required": ["command"],
                },
            ),
            executor=_shell_exec,
        )
    )

    # 5. list_directory
    def _list_dir_exec(args: dict[str, Any], env: ExecutionEnvironment) -> str:
        path = str(args.get("path", "."))
        depth = int(args.get("depth", 2))
        entries = env.list_directory(path, depth=depth)
        formatted = [f"{'[DIR] ' if e.is_dir else '      '}{e.path}" for e in entries]
        return "\n".join(formatted) if formatted else "(empty directory)"

    reg.register(
        RegisteredTool(
            definition=ToolDefinition(
                name="list_directory",
                description="List files and folders in the workspace up to a given depth.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative directory path"},
                        "depth": {"type": "integer", "description": "Recursion depth (default: 2)"},
                    },
                },
            ),
            executor=_list_dir_exec,
        )
    )

    # 6. grep
    def _grep_exec(args: dict[str, Any], env: ExecutionEnvironment) -> str:
        pattern = str(args.get("pattern", ""))
        path = str(args.get("path", "."))
        glob_filter = args.get("glob_filter")
        case_insensitive = bool(args.get("case_insensitive", False))
        return env.grep(
            pattern,
            path=path,
            glob_filter=str(glob_filter) if glob_filter else None,
            case_insensitive=case_insensitive,
        )

    reg.register(
        RegisteredTool(
            definition=ToolDefinition(
                name="grep",
                description="Search file contents across the workspace using regular expressions.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern"},
                        "path": {"type": "string", "description": "Search directory or file"},
                        "glob_filter": {"type": "string", "description": "File glob filter like '*.py'"},
                        "case_insensitive": {"type": "boolean"},
                    },
                    "required": ["pattern"],
                },
            ),
            executor=_grep_exec,
        )
    )

    # 7. glob
    def _glob_exec(args: dict[str, Any], env: ExecutionEnvironment) -> str:
        pattern = str(args.get("pattern", "*"))
        path = str(args.get("path", "."))
        matches = env.glob(pattern, path=path)
        return "\n".join(matches) if matches else "No matching files."

    reg.register(
        RegisteredTool(
            definition=ToolDefinition(
                name="glob",
                description="Find workspace files matching a wildcard pattern.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Pattern e.g. '**/*.py'"},
                        "path": {"type": "string", "description": "Base directory"},
                    },
                    "required": ["pattern"],
                },
            ),
            executor=_glob_exec,
        )
    )

    return reg
