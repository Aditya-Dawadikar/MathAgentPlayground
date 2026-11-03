"""Provider-agnostic chat model factory.

Set ``LLM_PROVIDER`` to choose the backend:

- ``ollama`` (default): uses ``OLLAMA_BASE_URL`` and per-role models
  ``OLLAMA_MODEL`` (agent), ``OLLAMA_MODEL_2`` (planner), ``OLLAMA_JUDGE_MODEL`` (judge).
- ``gemini``: uses ``langchain-google-genai`` with ``GOOGLE_API_KEY`` and
  ``GEMINI_MODEL`` (plus optional ``GEMINI_PLANNER_MODEL`` / ``GEMINI_JUDGE_MODEL``).
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.language_models import BaseChatModel

Role = Literal["agent", "planner", "judge"]

_OLLAMA_MODEL_ENV = {"agent": "OLLAMA_MODEL", "planner": "OLLAMA_MODEL_2", "judge": "OLLAMA_JUDGE_MODEL"}
_OLLAMA_DEFAULTS = {"agent": "llama3.1:8b", "planner": "llama3.1:8b", "judge": "llama3.1:8b"}
_GEMINI_MODEL_ENV = {"agent": "GEMINI_MODEL", "planner": "GEMINI_PLANNER_MODEL", "judge": "GEMINI_JUDGE_MODEL"}
_GEMINI_DEFAULT = "gemini-2.5-flash"


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def model_name(role: Role = "agent") -> str:
    if provider() == "gemini":
        return os.getenv(_GEMINI_MODEL_ENV[role]) or os.getenv("GEMINI_MODEL", _GEMINI_DEFAULT)
    return os.getenv(_OLLAMA_MODEL_ENV[role], _OLLAMA_DEFAULTS[role]).strip()


def build_chat_model(role: Role = "agent", temperature: float | None = 0) -> BaseChatModel:
    """Return a chat model for the given role using the configured provider."""
    name = provider()
    if name == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"temperature": temperature} if temperature is not None else {}
        return ChatOllama(
            model=model_name(role),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            **kwargs,
        )
    if name == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LLM_PROVIDER=gemini needs `pip install langchain-google-genai`") from exc
        kwargs = {"temperature": temperature} if temperature is not None else {}
        return ChatGoogleGenerativeAI(model=model_name(role), **kwargs)
    raise ValueError(f"Unknown LLM_PROVIDER {name!r}; use 'ollama' or 'gemini'")
