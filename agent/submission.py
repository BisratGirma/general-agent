"""Evaluation submission logic: fetch questions, run agent, post answers."""

from __future__ import annotations

import os

import pandas as pd
import requests

from agent.llm import get_hf_token

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def run_and_submit_all(agent, username: str | None = None) -> tuple[str, pd.DataFrame | None]:
    """Fetch all benchmark questions, run *agent* on each, and submit answers.

    Args:
        agent: A callable that accepts a question string and returns an answer string.
        username: Hugging Face username used for submission.

    Returns:
        A ``(status_message, results_dataframe)`` tuple.  The dataframe is
        ``None`` when an early error prevents any questions from being fetched.
    """
    space_id = os.getenv("SPACE_ID")

    if not username or not username.strip():
        return "Please enter your Hugging Face username before running the evaluation.", None

    username = username.strip()
    print(f"Using submitted username: {username}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    hf_token = get_hf_token()
    auth_headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    print("HF token found." if hf_token else "No HF token found — continuing without auth header.")

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "local"
    print(f"Agent code URL: {agent_code}")

    # --- Fetch questions ---
    questions_data = _fetch_questions(questions_url, auth_headers)
    if isinstance(questions_data, str):
        # Error message returned
        return questions_data, None

    # --- Run agent ---
    results_log, answers_payload = _run_agent(agent, questions_data)

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # --- Submit answers ---
    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    print(f"Submitting {len(answers_payload)} answers for user '{username}'...")
    return _submit_answers(submit_url, auth_headers, submission_data, results_log)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fetch_questions(url: str, headers: dict) -> list | str:
    """Fetch the list of questions from *url*.  Returns the list or an error string."""
    print(f"Fetching questions from: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data:
            return "Fetched questions list is empty or invalid format."
        print(f"Fetched {len(data)} questions.")
        return data
    except requests.exceptions.JSONDecodeError as exc:
        return f"Error decoding server response for questions: {exc}"
    except requests.exceptions.RequestException as exc:
        return f"Error fetching questions: {exc}"
    except Exception as exc:
        return f"An unexpected error occurred fetching questions: {exc}"


def _run_agent(agent, questions_data: list) -> tuple[list[dict], list[dict]]:
    """Run *agent* on every item in *questions_data* and collect results."""
    results_log: list[dict] = []
    answers_payload: list[dict] = []

    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as exc:
            print(f"Error running agent on task {task_id}: {exc}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {exc}"})

    return results_log, answers_payload


def _submit_answers(
    url: str,
    headers: dict,
    submission_data: dict,
    results_log: list[dict],
) -> tuple[str, pd.DataFrame]:
    """POST *submission_data* to *url* and return a human-readable status."""
    results_df = pd.DataFrame(results_log)
    try:
        response = requests.post(url, headers=headers, json=submission_data, timeout=60)
        response.raise_for_status()
        data = response.json()
        status = (
            f"Submission Successful!\n"
            f"User: {data.get('username')}\n"
            f"Overall Score: {data.get('score', 'N/A')}% "
            f"({data.get('correct_count', '?')}/{data.get('total_attempted', '?')} correct)\n"
            f"Message: {data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return status, results_df

    except requests.exceptions.HTTPError as exc:
        detail = f"Server responded with status {exc.response.status_code}."
        try:
            err_json = exc.response.json()
            detail += f" Detail: {err_json.get('detail', exc.response.text)}"
        except requests.exceptions.JSONDecodeError:
            detail += f" Response: {exc.response.text[:500]}"
        msg = f"Submission Failed: {detail}"
        print(msg)
        return msg, results_df

    except requests.exceptions.Timeout:
        msg = "Submission Failed: The request timed out."
        print(msg)
        return msg, results_df

    except requests.exceptions.RequestException as exc:
        msg = f"Submission Failed: Network error - {exc}"
        print(msg)
        return msg, results_df

    except Exception as exc:
        msg = f"An unexpected error occurred during submission: {exc}"
        print(msg)
        return msg, results_df
