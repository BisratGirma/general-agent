"""LLM client builders and a unified response helper.

This module provides:
- :func:`get_hf_token` — resolves a Hugging Face API token from the environment.
- :func:`build_hf_client` — creates a ``huggingface_hub.InferenceClient``.
- :func:`build_ollama_client` — creates an ``openai.OpenAI`` client pointed at Ollama.
- :func:`llm_response` — sends a prompt to *any* of the above clients and returns text.
"""

from __future__ import annotations

import os

from tools.common import _coerce_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)

HF_MODEL = (
    os.getenv("HF_MODEL")
    or os.getenv("HUGGINGFACE_MODEL")
    or "Qwen/Qwen3-4B-Instruct-2507"
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")


# ---------------------------------------------------------------------------
# Token helper
# ---------------------------------------------------------------------------

def get_hf_token() -> str | None:
    """Return the first non-empty Hugging Face token found in the environment."""
    for env_name in HF_TOKEN_ENV_VARS:
        token = os.getenv(env_name)
        if token and token.strip():
            return token.strip()
    return None


# ---------------------------------------------------------------------------
# Client builders
# ---------------------------------------------------------------------------

def build_hf_client():
    """Build a ``huggingface_hub.InferenceClient`` using the HF token.

    Returns:
        An :class:`InferenceClient` instance, or ``None`` if the token is
        missing or the library is not installed.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return None

    hf_token = get_hf_token()
    print(f"Attempting to retrieve HF token from: {HF_TOKEN_ENV_VARS}")
    print(f"HF token found: {'Yes' if hf_token else 'No'}")
    if not hf_token:
        return None

    try:
        client = InferenceClient(model=HF_MODEL, token=hf_token)
        print(f"HF InferenceClient created. model={HF_MODEL}")
        return client
    except Exception as exc:
        print(f"Failed to create HF InferenceClient: {exc}")
        return None


def build_ollama_client():
    """Build an OpenAI-compatible client pointed at the local Ollama server.

    Ollama exposes the OpenAI API at ``http://localhost:11434/v1``; no real API
    key is needed but the ``openai`` library requires a non-empty string.

    Returns:
        An :class:`openai.OpenAI` instance, or ``None`` on failure.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        print(f"Ollama client created. base_url={OLLAMA_BASE_URL}, model={OLLAMA_MODEL}")
        return client
    except Exception as exc:
        print(f"Failed to create Ollama client: {exc}")
        return None


# ---------------------------------------------------------------------------
# Unified response helper
# ---------------------------------------------------------------------------

def llm_response(client, prompt: str, *, max_tokens: int = 256) -> str:
    """Send *prompt* to *client* and return the reply as plain text.

    Handles both ``huggingface_hub.InferenceClient`` and ``openai.OpenAI``
    clients, as well as plain-dict fallback responses.

    Args:
        client: An HF ``InferenceClient`` or OpenAI-compatible ``OpenAI`` client.
        prompt: The user message to send.
        max_tokens: Maximum tokens for the completion.

    Returns:
        The model's reply as a string, or an empty string on failure.
    """
    # Determine call style: HF InferenceClient uses .chat.completions.create
    # with extra_body; OpenAI client uses model= keyword.
    is_hf = not hasattr(client, "base_url")

    kwargs: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
        "stream": False,
    }
    if is_hf:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    else:
        kwargs["model"] = OLLAMA_MODEL

    response = client.chat.completions.create(**kwargs)
    print(f"LLM response preview: {str(response)[:120]}")

    # Standard path: ChatCompletion object
    if hasattr(response, "choices") and response.choices:
        first = response.choices[0]
        if hasattr(first, "message"):
            msg = first.message
            content = getattr(msg, "content", None) or getattr(msg, "reasoning", None)
            return _coerce_text(content)
        if hasattr(first, "text"):
            return _coerce_text(getattr(first, "text", None))

    # Fallback: plain dict
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    return _coerce_text(message.get("content") or message.get("text"))
                return _coerce_text(choice.get("text"))
            return _coerce_text(choice)

    return _coerce_text(response)
