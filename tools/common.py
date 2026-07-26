"""Shared utility helpers used by all tool modules."""


def _coerce_text(value) -> str:
    """Coerce any LLM response value to a plain string."""
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(value.content)
    if isinstance(value, list):
        return "\n".join(str(part) for part in value)
    return str(value)


def _format_error(tool_name: str, message: str) -> str:
    """Return a consistently formatted error string."""
    return f"Error: {tool_name} failed - {message}"


def _normalize_query(query: str) -> str:
    """Strip whitespace from a query string, returning empty string for None."""
    return (query or "").strip()


def _extract_file_path(text: str, allowed_extensions: tuple[str, ...] | None = None) -> tuple[str | None, str]:
    """Extract a local file path and remaining prompt/query text from an input string."""
    import os
    import re

    if not text:
        return None, ""

    # 1. Direct check if text itself is a valid file path
    clean_text = text.strip().strip('"\'')
    if os.path.exists(clean_text) and os.path.isfile(clean_text):
        return clean_text, ""

    # 2. Check for Gradio (File: <path>) pattern
    file_match = re.search(r'\(File:\s*([^\)]+)\)', text, re.IGNORECASE)
    if file_match:
        possible_path = file_match.group(1).strip().strip('"\'')
        if os.path.exists(possible_path):
            remaining = text.replace(file_match.group(0), "").strip(" :(),\"'")
            return possible_path, remaining

    # 3. Regex search for Windows or POSIX file paths
    if allowed_extensions:
        ext_pattern = "|".join(re.escape(ext.lstrip(".")) for ext in allowed_extensions)
        path_pattern = rf'([A-Za-z]:\\[^\s"\'\(\)]+\.(?:{ext_pattern})|/[^\s"\'\(\)]+\.(?:{ext_pattern}))'
    else:
        path_pattern = r'([A-Za-z]:\\[^\s"\'\(\)]+\.[A-Za-z0-9]+|/[^\s"\'\(\)]+\.[A-Za-z0-9]+)'

    match = re.search(path_pattern, text, re.IGNORECASE)
    if match and os.path.exists(match.group(1)):
        found_path = match.group(1)
        remaining = text.replace(match.group(0), "").replace("(File: )", "").replace("Process file:", "").strip(" :(),\"'")
        return found_path, remaining

    # 4. Fallback regex search for paths in quotes or after 'file:' / 'Process file:'
    quote_pattern = r'(?:file:\s*|Process file:\s*|["\'])([A-Za-z]:\\[^"\'\n]+|/[^"\'\n]+)(?:["\']|\s|$)'
    match_quote = re.search(quote_pattern, text, re.IGNORECASE)
    if match_quote:
        possible_path = match_quote.group(1).strip()
        if os.path.exists(possible_path) and os.path.isfile(possible_path):
            remaining = text.replace(match_quote.group(0), "").strip(" :(),\"'")
            return possible_path, remaining

    return None, text

