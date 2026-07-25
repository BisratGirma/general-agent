"""Agent state definition shared by all graph nodes."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Mutable state passed through the LangGraph workflow.

    Attributes:
        question: The original user question.
        task_type: Classification label (e.g. ``"general"``, ``"image"``).
        evidence: Raw output collected by the tool node.
        answer: Final answer produced by the answer node.
    """

    question: str
    task_type: str
    evidence: str
    answer: str
