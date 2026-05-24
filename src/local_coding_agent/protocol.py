from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class AgentAction:
    action: str
    arguments: dict[str, Any]
    thought: str = ""


SYSTEM_PROMPT = """You are a local coding agent running on the user's machine.

You must solve the user's coding task by taking one tool action at a time.
Respond with exactly one JSON object and no prose outside the JSON.
Do not wrap the JSON in Markdown fences.

Valid actions:

1. list_files
   {"action":"list_files","path":".","max_depth":2}

2. read_file
   {"action":"read_file","path":"README.md","start_line":1,"end_line":200}

3. write_file
   {"action":"write_file","path":"new_file.py","content":"full file contents"}

4. replace_in_file
   {"action":"replace_in_file","path":"existing_file.py","old":"old text","new":"new text","expected_replacements":1}

5. run_shell
   {"action":"run_shell","command":"python3 -m unittest discover -s tests","timeout_seconds":60}

6. finish
   {"action":"finish","message":"Brief final answer for the user."}

Rules:
- Include a short "thought" string if useful, but keep it concise.
- Use relative paths inside the workspace.
- If the user names an exact file path, use that exact path.
- Do not add a directory prefix such as "src/" unless the user requested it.
- Every response must be valid JSON parseable by json.loads.
- For multiline file content, use a normal JSON string with escaped newline characters.
- Never use Python triple-quoted strings inside JSON.
- Read files before changing them unless you are creating a new file.
- Prefer replace_in_file for small edits and write_file for new files or complete rewrites.
- Run focused verification commands when possible.
- If the user asks for printed output, run the relevant command and inspect STDOUT.
- Empty STDOUT is not success when the user asked for output on screen.
- If a command fails, inspect the output and either fix the problem or finish with the blocker.
- Never ask for network access unless the user task truly requires it.
- Finish only when the task is complete or blocked.
"""


def parse_action(text: str) -> AgentAction:
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        raise ProtocolError("Expected a JSON object.")

    action = data.get("action")
    if not isinstance(action, str) or not action:
        raise ProtocolError('JSON object must include a non-empty string field named "action".')

    thought = data.get("thought", "")
    if thought is None:
        thought = ""
    if not isinstance(thought, str):
        raise ProtocolError('"thought" must be a string when provided.')

    arguments = {key: value for key, value in data.items() if key not in {"action", "thought"}}
    return AgentAction(action=action, arguments=arguments, thought=thought)


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    candidates = []

    if stripped.startswith("{"):
        candidates.append(stripped)

    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(match.group(1).strip())

    decoder = json.JSONDecoder()
    for start in [match.start() for match in re.finditer(r"\{", text)]:
        try:
            value, _ = decoder.raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue

    for candidate in candidates:
        try:
            parsed = _loads_json_or_repaired(candidate)
            if isinstance(parsed, dict):
                return parsed
        except ProtocolError:
            continue

    raise ProtocolError("Could not find a valid JSON object in the model response.")


def _loads_json_or_repaired(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        pass

    repaired = _replace_triple_quoted_strings(candidate)
    if repaired != candidate:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            pass

    raise ProtocolError("Candidate was not valid JSON.")


def _replace_triple_quoted_strings(text: str) -> str:
    result = []
    index = 0

    while index < len(text):
        delimiter = _next_triple_quote(text, index)
        if delimiter is None:
            result.append(text[index:])
            break

        start, quote = delimiter
        result.append(text[index:start])
        content_start = start + 3
        end = text.find(quote * 3, content_start)
        if end == -1:
            result.append(text[start:])
            break

        content = text[content_start:end]
        result.append(json.dumps(content))
        index = end + 3

    return "".join(result)


def _next_triple_quote(text: str, start: int) -> tuple[int, str] | None:
    double = text.find('"""', start)
    single = text.find("'''", start)
    if double == -1 and single == -1:
        return None
    if double == -1:
        return single, "'"
    if single == -1:
        return double, '"'
    if double < single:
        return double, '"'
    return single, "'"
