from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .llm import ChatMessage, LLMError, LocalLLM
from .protocol import SYSTEM_PROMPT, ProtocolError, parse_action
from .tools import ToolResult, WorkspaceTools


Observer = Callable[[str], None]


@dataclass
class AgentResult:
    ok: bool
    message: str
    steps: int


class CodingAgent:
    def __init__(
        self,
        llm: LocalLLM,
        tools: WorkspaceTools,
        max_steps: int = 20,
        observer: Observer | None = None,
        debug_model_output: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.observer = observer
        self.debug_model_output = debug_model_output

    def run(self, task: str) -> AgentResult:
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=f"Task: {task}"),
        ]
        protocol_error_count = 0
        completed_tool_count = 0

        for step in range(1, self.max_steps + 1):
            try:
                assistant_text = self.llm.chat(messages)
            except LLMError as exc:
                return AgentResult(ok=False, message=str(exc), steps=step - 1)

            if self.debug_model_output:
                self._observe(f"model output at step {step}:\n{assistant_text}\n---")

            messages.append(ChatMessage(role="assistant", content=assistant_text))

            try:
                action = parse_action(assistant_text)
            except ProtocolError as exc:
                protocol_error_count += 1
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"Your previous response was invalid: {exc}\n"
                            "The user's task is natural-language text. Your response must be valid JSON.\n"
                            "Return exactly one JSON object using one allowed action.\n"
                            "Do not explain the error. Do not use Markdown fences. Do not use triple-quoted strings.\n"
                            "If the task asks you to create a file, return a write_file action."
                        ),
                    )
                )
                self._observe(f"step {step}: protocol error, asking model to repair response")
                continue

            self._observe(_format_step(step, action.action, action.arguments, action.thought))

            if action.action == "finish":
                if protocol_error_count > 0 and completed_tool_count == 0:
                    return AgentResult(
                        ok=False,
                        message=(
                            "The model stopped after producing invalid tool JSON and did not take any action. "
                            "Re-run with --debug-model to inspect the raw model output, or try a stronger model."
                        ),
                        steps=step,
                    )
                message = action.arguments.get("message", "")
                if not isinstance(message, str) or not message:
                    message = "Done."
                return AgentResult(ok=True, message=message, steps=step)

            result = self.tools.run(action.action, action.arguments)
            tool_observation = _format_tool_observation(action.action, result)
            if tool_observation:
                self._observe(tool_observation)
            if result.ok:
                completed_tool_count += 1
            messages.append(ChatMessage(role="user", content=_tool_result_message(action.action, result)))

        return AgentResult(
            ok=False,
            message=f"Stopped after reaching max_steps={self.max_steps}.",
            steps=self.max_steps,
        )

    def _observe(self, message: str) -> None:
        if self.observer:
            self.observer(message)


def _tool_result_message(action: str, result: ToolResult) -> str:
    return json.dumps(
        {
            "tool_result": action,
            "ok": result.ok,
            "output": result.output,
        },
        ensure_ascii=False,
    )


def _format_step(step: int, action: str, arguments: dict, thought: str) -> str:
    pieces = [f"step {step}: {action}"]
    if action in {"read_file", "write_file", "replace_in_file", "list_files"} and "path" in arguments:
        pieces.append(str(arguments["path"]))
    if action == "run_shell" and "command" in arguments:
        pieces.append(str(arguments["command"]))
    if thought:
        pieces.append(f"- {thought}")
    return " ".join(pieces)


def _format_tool_observation(action: str, result: ToolResult) -> str:
    if action != "run_shell":
        return ""

    stdout = _extract_section(result.output, "STDOUT:", "STDERR:")
    stderr = _extract_section(result.output, "STDERR:", None)
    pieces = []

    if stdout.strip():
        pieces.append(f"command stdout:\n{stdout.rstrip()}")
    else:
        pieces.append("command stdout: (empty)")

    if stderr.strip():
        pieces.append(f"command stderr:\n{stderr.rstrip()}")

    return "\n".join(pieces)


def _extract_section(text: str, start_marker: str, end_marker: str | None) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    if text.startswith("\n", start):
        start += 1
    if end_marker is None:
        return text[start:]
    end = text.find(end_marker, start)
    if end == -1:
        return text[start:]
    return text[start:end]
