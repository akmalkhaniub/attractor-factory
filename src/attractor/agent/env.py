"""Execution Environment abstraction for the Coding Agent Loop (Section 4 of coding-agent-loop-spec.md)."""

from __future__ import annotations

import fnmatch
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class DirEntry:
    name: str
    is_dir: bool
    size: int | None = None
    path: str = ""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int

    @property
    def combined(self) -> str:
        out = self.stdout.strip()
        err = self.stderr.strip()
        if out and err:
            return f"{out}\n{err}"
        return out or err


class ExecutionEnvironment(Protocol):
    """Abstract interface decoupling agent tools from local machine, container, or remote sandbox."""

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str: ...

    def write_file(self, path: str, content: str) -> str: ...

    def edit_file(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str: ...

    def file_exists(self, path: str) -> bool: ...

    def list_directory(self, path: str = ".", depth: int = 2) -> list[DirEntry]: ...

    def exec_command(
        self,
        command: str,
        timeout_ms: int = 30000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult: ...

    def grep(
        self,
        pattern: str,
        path: str = ".",
        glob_filter: str | None = None,
        case_insensitive: bool = False,
        max_results: int = 100,
    ) -> str: ...

    def glob(self, pattern: str, path: str = ".") -> list[str]: ...

    def working_directory(self) -> Path: ...

    def platform_name(self) -> str: ...

    def os_version(self) -> str: ...


class LocalExecutionEnvironment:
    """Local workspace execution environment enforcing workspace root scoping and safe defaults."""

    # Section 4.2: Filter sensitive credentials from subshell environment
    SENSITIVE_ENV_PATTERNS = re.compile(
        r".*(_API_KEY|_SECRET|_TOKEN|_PASSWORD|_CREDENTIAL|_AUTH).*", re.IGNORECASE
    )

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            resolved = (self.workspace_root / p).resolve()
        else:
            resolved = p.resolve()
        return resolved

    def working_directory(self) -> Path:
        return self.workspace_root

    def platform_name(self) -> str:
        return platform.system().lower()

    def os_version(self) -> str:
        return platform.platform()

    def file_exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Section 3.4: Line-numbered content in 'NNN | content' format."""
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if target.is_dir():
            raise IsADirectoryError(f"Target is a directory: {path}")

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"[Binary file: {target.name} ({target.stat().st_size} bytes)]"

        lines = content.splitlines()
        start = max(1, offset if offset is not None else 1)
        end = len(lines) if limit is None else min(len(lines), start + limit - 1)

        result_lines: list[str] = []
        for line_no in range(start, end + 1):
            line_idx = line_no - 1
            if line_idx < len(lines):
                result_lines.append(f"{line_no:4d} | {lines[line_idx]}")

        return "\n".join(result_lines)

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        bytes_written = len(content.encode("utf-8"))
        rel = target.relative_to(self.workspace_root) if target.is_relative_to(self.workspace_root) else target
        return f"Successfully wrote {bytes_written} bytes to {rel}"

    def edit_file(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Section 3.5: Exact string match replace."""
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            raise ValueError(f"old_string not found in {path}")
        if count > 1 and not replace_all:
            raise ValueError(
                f"old_string matched {count} times in {path}. Provide more context or set replace_all=true."
            )

        updated = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        target.write_text(updated, encoding="utf-8")
        return f"Successfully made {count if replace_all else 1} replacement(s) in {path}"

    def list_directory(self, path: str = ".", depth: int = 2) -> list[DirEntry]:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        entries: list[DirEntry] = []
        base_depth = len(target.parts)

        for root, dirs, files in os.walk(target):
            current_depth = len(Path(root).parts) - base_depth
            if current_depth > depth:
                dirs.clear()
                continue
            for d in dirs:
                if d in (".git", ".venv", "__pycache__", "node_modules"):
                    continue
                full = Path(root) / d
                entries.append(
                    DirEntry(
                        name=d,
                        is_dir=True,
                        path=str(full.relative_to(self.workspace_root)),
                    )
                )
            for f in files:
                if f.endswith((".pyc", ".lock", ".tmp")):
                    continue
                full = Path(root) / f
                entries.append(
                    DirEntry(
                        name=f,
                        is_dir=False,
                        size=full.stat().st_size if full.exists() else None,
                        path=str(full.relative_to(self.workspace_root)),
                    )
                )
        return entries

    def exec_command(
        self,
        command: str,
        timeout_ms: int = 30000,
        working_dir: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ExecResult:
        cwd = self._resolve(working_dir) if working_dir else self.workspace_root
        timeout_sec = max(1.0, timeout_ms / 1000.0)

        # Environment variable filtering (Section 4.2)
        env = {
            k: v
            for k, v in os.environ.items()
            if not self.SENSITIVE_ENV_PATTERNS.match(k)
        }
        if env_vars:
            env.update(env_vars)

        start_t = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            duration_ms = int((time.monotonic() - start_t) * 1000)
            return ExecResult(
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
                timed_out=False,
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.monotonic() - start_t) * 1000)
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return ExecResult(
                stdout=stdout,
                stderr=f"{stderr}\n[Command timed out after {timeout_sec:.1f}s]",
                exit_code=124,
                timed_out=True,
                duration_ms=duration_ms,
            )

    def grep(
        self,
        pattern: str,
        path: str = ".",
        glob_filter: str | None = None,
        case_insensitive: bool = False,
        max_results: int = 100,
    ) -> str:
        target = self._resolve(path)
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}")

        matches: list[str] = []
        files_to_search: list[Path] = []
        if target.is_file():
            files_to_search.append(target)
        else:
            for root, _, filenames in os.walk(target):
                if any(ignored in root for ignored in (".git", ".venv", "__pycache__", "node_modules")):
                    continue
                for fn in filenames:
                    if glob_filter and not fnmatch.fnmatch(fn, glob_filter):
                        continue
                    files_to_search.append(Path(root) / fn)

        for fpath in files_to_search:
            try:
                text = fpath.read_text(encoding="utf-8")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = fpath.relative_to(self.workspace_root) if fpath.is_relative_to(self.workspace_root) else fpath
                    matches.append(f"{rel}:{idx}: {line.strip()}")
                    if len(matches) >= max_results:
                        return "\n".join(matches)

        if not matches:
            return "No matches found."
        return "\n".join(matches)

    def glob(self, pattern: str, path: str = ".") -> list[str]:
        target = self._resolve(path)
        results: list[str] = []
        for p in target.glob(pattern):
            if any(part in (".git", ".venv", "__pycache__") for part in p.parts):
                continue
            rel = p.relative_to(self.workspace_root) if p.is_relative_to(self.workspace_root) else p
            results.append(str(rel))
        return sorted(results)
