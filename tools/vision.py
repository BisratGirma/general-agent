"""Image analysis using a local Ollama vision model."""

from __future__ import annotations

import base64
import os
from typing import Optional

import requests

from tools.common import _format_error, _normalize_query

OLLAMA_VISION_MODEL = "qwen3-vl:2b"
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


import re


def _extract_image_path(text: str) -> tuple[str | None, str]:
    """Extract a valid local image filepath and remaining prompt text from input string."""
    if not text:
        return None, ""
    
    # 1. Direct check if text itself is a valid file path
    clean_text = text.strip().strip('"\'')
    if os.path.exists(clean_text) and os.path.isfile(clean_text):
        return clean_text, ""

    # 2. Regex search for Windows or POSIX file paths with image extensions
    path_pattern = r'([A-Za-z]:\\[^\s"\'\(\)]+\.(?:png|jpg|jpeg|webp|bmp)|/[^\s"\'\(\)]+\.(?:png|jpg|jpeg|webp|bmp))'
    match = re.search(path_pattern, text, re.IGNORECASE)
    
    if match and os.path.exists(match.group(1)):
        img_path = match.group(1)
        # Extract remaining text to use as prompt if user provided a question
        remaining_prompt = text.replace(match.group(0), "").replace("(File: )", "").strip(" :(),\"'")
        return img_path, remaining_prompt

    # 3. Fallback regex search for any path in quotes or after 'file:'
    quote_pattern = r'(?:file:\s*|["\'])([A-Za-z]:\\[^"\'\n]+|/[^"\'\n]+)(?:["\']|\s|$)'
    match_quote = re.search(quote_pattern, text, re.IGNORECASE)
    if match_quote:
        possible_path = match_quote.group(1).strip()
        if os.path.exists(possible_path) and os.path.isfile(possible_path):
            remaining_prompt = text.replace(match_quote.group(0), "").strip(" :(),\"'")
            return possible_path, remaining_prompt

    return None, text


def analyze_image(
    image_path: str,
    task_type: str = "general",
    prompt: Optional[str] = None,
) -> str:
    """Analyze *image_path* with the local Ollama vision model.

    Args:
        image_path: Absolute or relative path to the image file, or text containing the path.
        task_type: ``"chess"`` for board analysis, ``"general"`` for open-ended description.
        prompt: Optional custom prompt; overrides the default for *task_type*.

    Returns:
        The model's text response, or a formatted error string on failure.
    """
    input_text = _normalize_query(image_path)
    actual_path, extracted_prompt = _extract_image_path(input_text)
    if not actual_path:
        return _format_error("Vision Tool", f"image file does not exist (received: {input_text!r})")

    # Health-check: ensure Ollama is reachable
    try:
        health = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if health.status_code != 200:
            raise requests.RequestException("unexpected status from Ollama")
    except requests.RequestException:
        return _format_error("Vision Tool", "Ollama service not running. Start it with 'ollama serve'.")

    # Choose prompt (custom prompt > prompt extracted from query string > task_type default)
    prompt_text = prompt or extracted_prompt or _PROMPTS.get(task_type.lower(), _PROMPTS["general"])

    # Base64-encode the image
    try:
        with open(actual_path, "rb") as img_file:
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
            timeout=1000,  # vision inference can be slow on CPU
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
