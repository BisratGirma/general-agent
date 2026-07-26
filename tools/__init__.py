"""Public API for the tools package.

Re-exports everything that was previously available at the ``tools_local``
module level, so that existing imports of the form::

    from tools_local import analyze_image, web_search, ...

continue to work via the ``tools_local.py`` compatibility shim.
"""

from tools.classifier import classify_query
from tools.code_exec import execute_python_code
from tools.common import _coerce_text, _format_error, _normalize_query
from tools.media import process_media, process_youtube_transcript, transcribe_audio_file
from tools.spreadsheet import parse_spreadsheet
from tools.vision import analyze_image, analyze_image_hf, analyze_image_ollama
from tools.web import (
    _build_page_selection_prompt,
    _is_text_webpage_url,
    _is_wikipedia_query,
    _select_best_page_from_results,
    scrape_url,
    web_search,
)

__all__ = [
    # Public tools
    "classify_query",
    "execute_python_code",
    "process_media",
    "process_youtube_transcript",
    "transcribe_audio_file",
    "parse_spreadsheet",
    "analyze_image",
    "analyze_image_ollama",
    "analyze_image_hf",
    "scrape_url",
    "web_search",
    # Private helpers (exposed for testing)
    "_coerce_text",
    "_format_error",
    "_normalize_query",
    "_build_page_selection_prompt",
    "_is_text_webpage_url",
    "_is_wikipedia_query",
    "_select_best_page_from_results",
]
