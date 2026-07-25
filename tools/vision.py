"""Image analysis using a local Ollama vision model."""

from __future__ import annotations

import base64
import os
from typing import Optional

import requests

from tools.common import _format_error, _normalize_query

OLLAMA_VISION_MODEL = "gemma4:e4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Task-specific default prompts
_PROMPTS: dict[str, str] = {
    "chess": (
        "You are a chess expert. Carefully examine this chess board image. "
        "Describe the exact position of every piece you can see (use standard algebraic notation for squares). "
        "Then provide a brief evaluation of the position, noting which side has the advantage and why."
    ),
    "general": "Describe the image in detail, noting key objects, colors, layout, and any text visible.",
}


def analyze_image(
    image_path: str,
    task_type: str = "general",
    prompt: Optional[str] = None,
) -> str:
    """Analyze *image_path* with the local Ollama vision model.

    Args:
        image_path: Absolute or relative path to the image file.
        task_type: ``"chess"`` for board analysis, ``"general"`` for open-ended description.
        prompt: Optional custom prompt; overrides the default for *task_type*.

    Returns:
        The model's text response, or a formatted error string on failure.
    """
    image_path = _normalize_query(image_path)
    if not image_path or not os.path.exists(image_path):
        return _format_error("Vision Tool", "image file does not exist")

    # Health-check: ensure Ollama is reachable
    try:
        health = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if health.status_code != 200:
            raise requests.RequestException("unexpected status from Ollama")
    except requests.RequestException:
        return _format_error("Vision Tool", "Ollama service not running. Start it with 'ollama serve'.")

    # Choose prompt
    prompt_text = prompt or _PROMPTS.get(task_type.lower(), _PROMPTS["general"])

    # Base64-encode the image
    try:
        with open(image_path, "rb") as img_file:
            image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    except OSError as exc:
        return _format_error("Vision Tool", f"could not read image file: {exc}")

    # Call Ollama /api/generate
    payload = {
        "model": OLLAMA_VISION_MODEL,
        "prompt": prompt_text,
        "images": [image_b64],
        "stream": False,
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,  # vision inference can be slow on CPU
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        return _format_error("Vision Tool", f"Ollama API call failed: {exc}")
    except ValueError as exc:
        return _format_error("Vision Tool", f"could not parse Ollama response: {exc}")

    model_response = result.get("response", "").strip()
    if not model_response:
        return _format_error("Vision Tool", "Ollama returned an empty response")

    return model_response
