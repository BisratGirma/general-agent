"""Keyword-based query classifier."""

from tools.common import _normalize_query

# Ordered from most-specific to least-specific so that overlapping keywords
# (e.g. "analyze" could be image *or* general) resolve correctly.
_CLASSIFICATION_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("image",   ("image", "photo", "picture", "screenshot", "analyze", "describe")),
    ("video",   ("youtube", "video", "transcript", "watch", "clip")),
    ("audio",   ("audio", "transcribe", "speech", "recording")),
    ("code",    ("python", "execute", "run code", "calculate", "compute", "script")),
    ("excel",   ("csv", "xlsx", "excel", "spreadsheet", "sheet", "data file")),
    ("website", ("website", "webpage", "url", "site", "browse", "review", "search")),
]


def classify_query(question: str) -> str:
    """Return the task type that best matches *question*.

    Possible return values: ``"image"``, ``"video"``, ``"audio"``, ``"code"``,
    ``"excel"``, ``"website"``, or ``"general"`` (the default).
    """
    normalized = _normalize_query(question).lower()
    for task_type, keywords in _CLASSIFICATION_RULES:
        if any(keyword in normalized for keyword in keywords):
            return task_type
    return "general"
