"""A tiny chat client for any OpenAI-compatible endpoint (standard library only).

Ollama serves this API at http://localhost:11434/v1, so the same code runs a
local model or, later, a hosted one: change --base-url and set an API key.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

OLLAMA_URL = "http://localhost:11434/v1"


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResponse:
    message: dict[str, Any]          # {"role": "assistant", "content": ..., "tool_calls": [...]}
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatClient:
    def __init__(self, model: str, base_url: str = OLLAMA_URL, api_key: str | None = None,
                 temperature: float = 0.0, seed: int = 42, timeout: int = 1800):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or "ollama"
        self.temperature, self.seed, self.timeout = temperature, seed, timeout

    def _request(self, path: str, payload: dict | None = None) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code} from {self.base_url}: {e.read()[:500]!r}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Cannot reach {self.base_url} ({e.reason}). Is Ollama running?") from e

    def check(self) -> None:
        """Fail early with a clear message if the server or model is missing."""
        models = [m.get("id") for m in self._request("/models").get("data", [])]
        if models and self.model not in models:
            raise LLMError(f"Model '{self.model}' not found. Available: {', '.join(models)}. "
                           f"Try: ollama pull {self.model}")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": messages,
                                   "temperature": self.temperature, "seed": self.seed}
        if tools:
            payload["tools"] = tools
        data = self._request("/chat/completions", payload)
        usage = data.get("usage") or {}
        return ChatResponse(data["choices"][0]["message"],
                            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


class FakeLLM:
    """Scripted stand-in for tests and dry runs: `respond(messages, tools) -> message dict`."""

    def __init__(self, respond: Callable[[list[dict], list[dict] | None], dict], model: str = "fake"):
        self.respond, self.model = respond, model

    def check(self) -> None:
        pass

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        msg = {"role": "assistant", "content": None, **self.respond(messages, tools)}
        return ChatResponse(msg, prompt_tokens=sum(len(str(m.get("content") or "")) // 4 for m in messages),
                            completion_tokens=len(str(msg.get("content") or "")) // 4)
