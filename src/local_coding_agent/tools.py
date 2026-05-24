from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "output": self.output}


@dataclass
class ShellPolicy:
    enabled: bool = True
    auto_approve: bool = False


class WorkspaceTools:
    def __init__(self, workspace: str | Path, shell_policy: ShellPolicy | None = None) -> None:
        self.root = Path(workspace).expanduser().resolve()
        self.shell_policy = shell_policy or ShellPolicy()

    def run(self, action: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            if action == "list_files":
                return self.list_files(
                    path=str(arguments.get("path", ".")),
                    max_depth=int(arguments.get("max_depth", 2)),
                    max_entries=int(arguments.get("max_entries", 250)),
                )
            if action == "read_file":
                return self.read_file(
                    path=_required_str(arguments, "path"),
                    start_line=_optional_int(arguments, "start_line"),
                    end_line=_optional_int(arguments, "end_line"),
                )
            if action == "write_file":
                return self.write_file(
                    path=_required_str(arguments, "path"),
                    content=_required_str(arguments, "content"),
                )
            if action == "replace_in_file":
                return self.replace_in_file(
                    path=_required_str(arguments, "path"),
                    old=_required_str(arguments, "old"),
                    new=_required_str(arguments, "new"),
                    expected_replacements=_optional_int(arguments, "expected_replacements"),
                )
            if action == "run_shell":
                return self.run_shell(
                    command=_required_str(arguments, "command"),
                    timeout_seconds=int(arguments.get("timeout_seconds", 60)),
                )
        except (ToolError, OSError, ValueError) as exc:
            return ToolResult(ok=False, output=str(exc))

        return ToolResult(ok=False, output=f"Unknown action: {action}")

    def list_files(self, path: str = ".", max_depth: int = 2, max_entries: int = 250) -> ToolResult:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"Path does not exist: {path}")
        if max_depth < 0:
            raise ToolError("max_depth must be >= 0")

        if target.is_file():
            return ToolResult(ok=True, output=self._relative(target))

        lines: list[str] = []
        skipped = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}
        base_depth = len(target.relative_to(self.root).parts)

        for current, dirnames, filenames in os.walk(target):
            current_path = Path(current)
            rel_depth = len(current_path.relative_to(self.root).parts) - base_depth
            if rel_depth > max_depth:
                dirnames[:] = []
                continue

            dirnames[:] = sorted(dirname for dirname in dirnames if dirname not in skipped)
            filenames = sorted(filename for filename in filenames if filename not in skipped)

            if current_path != target:
                lines.append(f"{self._relative(current_path)}/")
            for filename in filenames:
                lines.append(self._relative(current_path / filename))
                if len(lines) >= max_entries:
                    lines.append(f"... truncated after {max_entries} entries")
                    return ToolResult(ok=True, output="\n".join(lines))

        return ToolResult(ok=True, output="\n".join(lines) if lines else "(empty)")

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"File does not exist: {path}")
        if not target.is_file():
            raise ToolError(f"Path is not a file: {path}")

        text = target.read_text(encoding="utf-8")
        lines = text.splitlines()

        if start_line is not None or end_line is not None:
            start = 1 if start_line is None else start_line
            end = len(lines) if end_line is None else end_line
            if start < 1 or end < start:
                raise ToolError("Invalid line range.")
            selected = lines[start - 1 : end]
            text = "\n".join(f"{line_number}: {line}" for line_number, line in enumerate(selected, start=start))
            if target.read_text(encoding="utf-8").endswith("\n") and end >= len(lines):
                text += "\n"

        if len(text) > 120_000:
            text = text[:120_000] + "\n... truncated after 120000 characters"

        return ToolResult(ok=True, output=text)

    def write_file(self, path: str, content: str) -> ToolResult:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        line_count = len(content.splitlines())
        return ToolResult(ok=True, output=f"Wrote {self._relative(target)} ({line_count} lines).")

    def replace_in_file(
        self,
        path: str,
        old: str,
        new: str,
        expected_replacements: int | None = 1,
    ) -> ToolResult:
        if not old:
            raise ToolError("old text must not be empty.")
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"File does not exist: {path}")
        if not target.is_file():
            raise ToolError(f"Path is not a file: {path}")

        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ToolError("old text was not found.")
        if expected_replacements is not None and count != expected_replacements:
            raise ToolError(f"Expected {expected_replacements} replacements but found {count}.")

        target.write_text(text.replace(old, new), encoding="utf-8")
        return ToolResult(ok=True, output=f"Replaced {count} occurrence(s) in {self._relative(target)}.")

    def run_shell(self, command: str, timeout_seconds: int = 60) -> ToolResult:
        if not self.shell_policy.enabled:
            raise ToolError("Shell commands are disabled.")
        if not command.strip():
            raise ToolError("Command must not be empty.")
        if timeout_seconds < 1 or timeout_seconds > 600:
            raise ToolError("timeout_seconds must be between 1 and 600.")

        if not self.shell_policy.auto_approve and not self._prompt_approval(command):
            raise ToolError("User rejected shell command.")

        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return ToolResult(
                ok=False,
                output=f"Command timed out after {timeout_seconds}s.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
            )

        output = (
            f"exit_code={completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
        if len(output) > 120_000:
            output = output[:120_000] + "\n... truncated after 120000 characters"
        return ToolResult(ok=completed.returncode == 0, output=output)

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError(f"Path escapes workspace: {path}")
        return resolved

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def _prompt_approval(self, command: str) -> bool:
        print(f"\nThe agent wants to run this command in {self.root}:\n\n  {command}\n", file=sys.stderr)
        if not sys.stdin.isatty():
            print("Shell approval requires an interactive terminal. Re-run with --yes to allow commands.", file=sys.stderr)
            return False
        answer = input("Allow command? [y/N] ").strip().lower()
        return answer in {"y", "yes"}


def _required_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ToolError(f'Missing required string argument "{key}".')
    return value


def _optional_int(arguments: dict[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f'Optional argument "{key}" must be an integer.')
    return value
