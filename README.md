# Local Coding Agent

A small coding agent that runs on your machine using an open-source Hugging Face Transformers model. Ollama and OpenAI-compatible local servers are still supported as optional adapters, but Transformers is the default runtime.

## What It Does

- Loads a local Transformers causal language model in-process.
- Gives the model a narrow JSON action protocol instead of relying on hosted tool-calling APIs.
- Exposes workspace-scoped tools for listing, reading, writing, replacing text, and running commands.
- Keeps shell commands approval-gated unless you explicitly pass `--yes`.

## Recommended Local Model

For a coding-focused local model, start with one of these Hugging Face model IDs:

- `Qwen/Qwen2.5-Coder-1.5B-Instruct` if you want the easiest first run.
- `Qwen/Qwen2.5-Coder-7B-Instruct` for stronger coding behavior on a machine with enough memory.
- `deepseek-ai/deepseek-coder-1.3b-instruct` as another small coding model option.

The model is configurable, so the agent is not tied to any one model.

## Setup With Transformers

Install the project into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Run the default Transformers-backed agent:

```bash
local-agent "Add a README section explaining how to run tests"
```

The first run may download model weights from Hugging Face. After the model is cached, inference runs locally. To require pre-downloaded files only:

```bash
local-agent --local-files-only "Create a small Python script that prints hello"
```

Try a larger model when the machine has enough memory:

```bash
local-agent \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --device auto \
  "Find and fix failing tests"
```

If your Python interpreter cannot install PyTorch wheels, create the virtual environment with a Python version supported by your PyTorch install.

## Optional Local Server Adapters

There is no strong architectural reason to prefer Ollama here; Transformers is the primary path. The adapters are useful only when you already have a local server managing model loading for you.

For Ollama:

```bash
local-agent \
  --provider ollama \
  --model qwen2.5-coder:7b \
  "Create a small Python script that prints hello"
```

For llama.cpp server, LM Studio, vLLM, or any local OpenAI-compatible endpoint:


```bash
local-agent \
  --provider openai-compatible \
  --base-url http://localhost:8000 \
  --model qwen2.5-coder-7b \
  "Find and fix failing tests"
```

For LM Studio, the base URL is often:

```bash
local-agent --provider openai-compatible --base-url http://localhost:1234 ...
```

## CLI Options

```bash
local-agent --help
```

Useful flags:

- `--workspace PATH`: directory the agent may read and write. Defaults to the current directory.
- `--model NAME`: Hugging Face model ID or local model path.
- `--provider transformers|ollama|openai-compatible`: local runtime to use.
- `--device auto|cpu|mps|cuda`: Torch device for Transformers.
- `--torch-dtype auto|float32|float16|bfloat16|none`: model dtype for Transformers.
- `--local-files-only`: do not download model files.
- `--base-url URL`: local model server URL for server adapters.
- `--max-steps N`: maximum number of model/tool iterations.
- `--debug-model`: print raw model outputs before parsing.
- `--yes`: allow shell commands without prompting.
- `--no-shell`: disable shell commands entirely.

## Safety Model

File tools cannot access paths outside the workspace. Shell commands run with the workspace as the current directory and require approval by default. This is still powerful software: run it in a project directory you are comfortable letting an agent edit.

## Smoke Test Without A Model

The unit tests exercise the JSON parser, agent loop, and workspace tools without requiring a model:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
