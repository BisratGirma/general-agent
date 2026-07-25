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
