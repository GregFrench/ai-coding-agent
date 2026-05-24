from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import CodingAgent
from .llm import LLMError, OllamaClient, OpenAICompatibleClient, TransformersClient
from .tools import ShellPolicy, WorkspaceTools


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        print(f"Workspace does not exist: {workspace}", file=sys.stderr)
        return 2
    if not workspace.is_dir():
        print(f"Workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    llm = build_llm(args)
    tools = WorkspaceTools(
        workspace=workspace,
        shell_policy=ShellPolicy(enabled=not args.no_shell, auto_approve=args.yes),
    )
    agent = CodingAgent(
        llm=llm,
        tools=tools,
        max_steps=args.max_steps,
        observer=(print if args.verbose or args.debug_model else None),
        debug_model_output=args.debug_model,
    )

    try:
        result = agent.run(args.task)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except LLMError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(result.message)
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-agent",
        description="Run a local coding agent against a workspace.",
    )
    parser.add_argument("task", help="Coding task for the agent to complete.")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Directory the agent may read and write. Defaults to the current directory.",
    )
    parser.add_argument(
        "--provider",
        choices=["transformers", "ollama", "openai-compatible"],
        default="transformers",
        help="Local model runtime.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Model server base URL for Ollama or OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Local model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Model temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Model request timeout in seconds.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens generated per model step for the transformers provider.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=8192,
        help="Maximum prompt tokens kept per model step for the transformers provider.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for the transformers provider: auto, cpu, mps, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for the transformers provider: auto, float32, float16, bfloat16, or none.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when loading with transformers.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load transformers model files only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum model/tool iterations.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Allow shell commands without prompting.",
    )
    parser.add_argument(
        "--no-shell",
        action="store_true",
        help="Disable shell command execution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print each model action as it runs.",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Only print the final result.",
    )
    parser.add_argument(
        "--debug-model",
        action="store_true",
        help="Print raw model outputs before the agent parses them.",
    )
    return parser


def build_llm(args: argparse.Namespace):
    api_key = os.environ.get("LOCAL_AGENT_API_KEY")
    if args.provider == "transformers":
        return TransformersClient(
            model=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            device=args.device,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    if args.provider == "ollama":
        return OllamaClient(
            model=args.model,
            base_url=args.base_url or "http://localhost:11434",
            temperature=args.temperature,
            timeout_seconds=args.timeout,
        )
    return OpenAICompatibleClient(
        model=args.model,
        base_url=args.base_url or "http://localhost:8000",
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        api_key=api_key,
    )
