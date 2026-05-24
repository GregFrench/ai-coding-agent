from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class LocalLLM(Protocol):
    def chat(self, messages: list[ChatMessage]) -> str:
        """Return the assistant content for a chat completion."""


class LLMError(RuntimeError):
    pass


@dataclass
class TransformersClient:
    model: str
    temperature: float = 0.1
    max_new_tokens: int = 2048
    max_input_tokens: int = 8192
    device: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)
    _selected_device: str = field(default="", init=False, repr=False)

    def chat(self, messages: list[ChatMessage]) -> str:
        self._load()
        prompt = self._render_prompt(messages)

        old_truncation_side = getattr(self._tokenizer, "truncation_side", "right")
        self._tokenizer.truncation_side = "left"
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            )
        finally:
            self._tokenizer.truncation_side = old_truncation_side

        inputs = {key: value.to(self._selected_device) for key, value in inputs.items()}
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": pad_token_id,
        }
        if self.temperature <= 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature

        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        prompt_token_count = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_token_count:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMError(
                "The transformers provider requires Hugging Face Transformers and PyTorch. "
                "Install them with: python3 -m pip install -e ."
            ) from exc

        self._torch = torch
        self._selected_device = self._choose_device(torch)
        dtype = self._resolve_torch_dtype(torch)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
        except OSError as exc:
            hint = (
                "If this is the first run, allow Transformers to download the model once, "
                "or pre-download it with huggingface-cli. Use --local-files-only only after "
                "the model is already cached."
            )
            raise LLMError(f"Could not load model {self.model!r}. {hint}\n{exc}") from exc

        try:
            self._model.to(self._selected_device)
            self._model.eval()
        except Exception as exc:
            raise LLMError(f"Could not move model to device {self._selected_device!r}: {exc}") from exc

    def _render_prompt(self, messages: list[ChatMessage]) -> str:
        chat_messages = [{"role": message.role, "content": message.content} for message in messages]
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if apply_chat_template is not None:
            try:
                return str(
                    apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                pass

        rendered = []
        for message in messages:
            rendered.append(f"{message.role.upper()}:\n{message.content}")
        rendered.append("ASSISTANT:\n")
        return "\n\n".join(rendered)

    def _choose_device(self, torch: Any) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_torch_dtype(self, torch: Any) -> Any:
        if self.torch_dtype in {"", "none", "None"}:
            return None
        if self.torch_dtype == "auto":
            return "auto"
        try:
            return getattr(torch, self.torch_dtype)
        except AttributeError as exc:
            raise LLMError(
                f"Unknown torch dtype {self.torch_dtype!r}. Try auto, float32, float16, or bfloat16."
            ) from exc


def _post_json(url: str, payload: dict, timeout_seconds: float, headers: dict[str, str] | None = None) -> dict:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"Model server returned HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Could not reach model server at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"Timed out waiting for model server at {url}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model server returned invalid JSON: {raw[:500]}") from exc


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    timeout_seconds: float = 300

    def chat(self, messages: list[ChatMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = _post_json(
            f"{self.base_url.rstrip('/')}/api/chat",
            payload,
            timeout_seconds=self.timeout_seconds,
        )

        try:
            return str(data["message"]["content"])
        except KeyError as exc:
            raise LLMError(f"Ollama response did not include message.content: {data}") from exc


@dataclass
class OpenAICompatibleClient:
    model: str
    base_url: str = "http://localhost:8000"
    temperature: float = 0.1
    timeout_seconds: float = 300
    api_key: str | None = None

    def chat(self, messages: list[ChatMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": self.temperature,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = _post_json(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            payload,
            timeout_seconds=self.timeout_seconds,
            headers=headers,
        )

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"OpenAI-compatible response did not include choices[0].message.content: {data}") from exc
