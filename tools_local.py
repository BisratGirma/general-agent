"""Backward-compatibility shim for tools_local.

All symbols that were previously defined directly in this file are now
maintained in the ``tools/`` package.  This shim re-exports every public and
semi-private name so that existing imports of the form::

    from tools_local import web_search, analyze_image, ...

continue to work without any changes to call sites or tests.
"""

# Re-export everything from the new tools package
from tools import (  # noqa: F401, F403
    _build_page_selection_prompt,
    _coerce_text,
    _format_error,
    _is_text_webpage_url,
    _is_wikipedia_query,
    _normalize_query,
    _select_best_page_from_results,
    analyze_image,
    classify_query,
    execute_python_code,
    parse_spreadsheet,
    process_media,
    process_youtube_transcript,
    scrape_url,
    transcribe_audio_file,
    web_search,
)
