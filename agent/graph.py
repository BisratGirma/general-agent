"""LangGraph workflow and the LangGraphAgent class."""

from __future__ import annotations

from agent.llm import llm_response
from agent.state import AgentState
from langgraph.graph import END, START, StateGraph
from tools import (
    analyze_image,
    classify_query,
    execute_python_code,
    parse_spreadsheet,
    process_media,
    web_search,
)

# ---------------------------------------------------------------------------
# Template answers (used when LLM is unavailable)
# ---------------------------------------------------------------------------

_FALLBACK_ANSWERS: dict[str, str] = {
    "image": (
        "Image-analysis workflow ready. I would inspect the image, identify the key "
        "visual details, and answer the question with grounded observations."
    ),
    "website": (
        "Website-review workflow ready. I would inspect the page content, assess clarity, "
        "trustworthiness, usability, and summarize strengths and risks."
    ),
    "video": (
        "Video-review workflow ready. I would extract the transcript or key moments, "
        "summarize the content, and answer the question using evidence from the video."
    ),
}

_DEFAULT_FALLBACK = (
    "General Q&A workflow ready. I would answer directly, clarify ambiguity when needed, "
    "and provide a concise, helpful response."
)

# Task-type → tool mapping
_TOOL_DISPATCH: dict[str, object] = {
    "website": lambda q: web_search(q, max_results=2),
    "video":   lambda q: process_media("youtube", url=q),
    "audio":   lambda q: process_media("audio_file", file_path=q),
    "code":    execute_python_code,
    "excel":   parse_spreadsheet,
    "image":   analyze_image,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_fallback_answer(task_type: str, _question: str) -> str:
    return _FALLBACK_ANSWERS.get(task_type, _DEFAULT_FALLBACK)


def _classify_task(question: str) -> str:
    return classify_query(question)


def _build_search_query(llm, user_question: str) -> str:
    """Ask the LLM to turn a natural-language question into a short search-engine query.

    Falls back to the original question if the LLM is unavailable or fails.
    """
    if llm is None:
        return user_question

    prompt = (
        "Convert the following user question into a concise, keyword-focused search query "
        "suitable for DuckDuckGo (max 8 words). "
        "Return ONLY the search query, nothing else.\n\n"
        f"User question: {user_question}"
    )
    try:
        search_query = llm_response(llm, prompt).strip()
        # Sanity-check: if the model returned something reasonable, use it
        if search_query and len(search_query) < len(user_question):
            print(f"[SEARCH QUERY] reformulated: {search_query!r}")
            return search_query
    except Exception as exc:
        print(f"[SEARCH QUERY] LLM reformulation failed, using original: {exc}")

    return user_question


# ---------------------------------------------------------------------------
# LangGraphAgent
# ---------------------------------------------------------------------------

class LangGraphAgent:
    """A LangGraph-based agent that routes, calls tools, and answers questions.

    Args:
        llm: An optional LLM client (HF ``InferenceClient`` or Ollama ``OpenAI``).
             When ``None``, the agent falls back to template responses.
    """

    def __init__(self, llm=None) -> None:
        print("LangGraphAgent initialized.")
        self.llm = llm
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("route_question", self._route_question)
        workflow.add_node("use_tool",        self._use_tool)
        workflow.add_node("answer_question", self._answer_question)
        workflow.add_edge(START,            "route_question")
        workflow.add_edge("route_question", "use_tool")
        workflow.add_edge("use_tool",       "answer_question")
        workflow.add_edge("answer_question", END)
        return workflow.compile()

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _route_question(self, state: AgentState) -> dict[str, str]:
        """Classify the question into a task type using heuristics + optional LLM."""
        question = state.get("question", "")
        task_type = _classify_task(question)

        if self.llm is not None:
            try:
                prompt = (
                    "Classify the user's request into EXACTLY ONE of these words: "
                    "general, image, website, video, audio, code, or excel.\n"
                    f"Question: {question}\n"
                    "Return ONLY the single label word and nothing else."
                )
                raw_llm_type = llm_response(self.llm, prompt).strip().lower()
                valid_types = {"general", "image", "website", "video", "audio", "code", "excel"}
                # Extract exact label word if LLM added explanatory preamble
                for valid_t in valid_types:
                    if valid_t in raw_llm_type.split():
                        task_type = valid_t
                        break
                    elif raw_llm_type.endswith(valid_t):
                        task_type = valid_t
                        break
            except Exception as exc:
                print(f"LLM routing failed, falling back to heuristic: {exc}")

        print(f"[ROUTE] task_type={task_type}")
        return {"task_type": task_type}

    def _use_tool(self, state: AgentState) -> dict[str, str]:
        """Dispatch to the appropriate tool and collect evidence."""
        question = state.get("question", "")
        task_type = state.get("task_type", "general")
        evidence = ""

        tool = _TOOL_DISPATCH.get(task_type)
        try:
            if tool is not None:
                if task_type == "website":
                    # For web searches, let the LLM craft a better search query
                    search_query = _build_search_query(self.llm, question)
                    evidence = web_search(search_query, max_results=2)
                else:
                    evidence = tool(question)  # type: ignore[operator]
            # else: task_type == "general" — no tool needed; LLM answers directly
        except Exception as exc:
            evidence = f"Error: tool execution failed - {exc}"

        print(f"[TOOL] evidence length: {len(evidence)} chars")
        return {"evidence": evidence}

    def _answer_question(self, state: AgentState) -> dict[str, str]:
        """Synthesize a final answer using the LLM (or fallback template)."""
        import re

        question = state.get("question", "")
        task_type = state.get("task_type", "general")
        evidence = state.get("evidence", "")

        # Clean file attachment string from question for LLM instruction clarity
        clean_question = re.sub(r"\(File:\s*[^\)]+\)", "", question, flags=re.IGNORECASE).strip()
        if not clean_question:
            clean_question = question

        if self.llm is not None:
            try:
                prompt = f"User Request: {clean_question}\n"
                if evidence:
                    prompt += f"\nContext/Evidence from tools:\n{evidence}\n\n"
                prompt += f"Based on the evidence above, directly answer the user request: {clean_question}"
                answer = llm_response(self.llm, prompt).strip() or _build_fallback_answer(task_type, question)
            except Exception as exc:
                print(f"LLM inference failed, falling back to template: {exc}")
                answer = _build_fallback_answer(task_type, question)
        else:
            answer = _build_fallback_answer(task_type, question)

        return {"answer": answer}


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 80 chars): {question[:80]}...")
        state = self.graph.invoke({"question": question})
        answer = state.get("answer", "")
        print(f"Agent returning answer: {answer[:120]}...")
        return answer or "I could not produce an answer for that request."
