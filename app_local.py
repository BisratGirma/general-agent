import os
from typing import TypedDict

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from tools_local import (
    analyze_image,
    classify_query,
    execute_python_code,
    parse_spreadsheet,
    process_media,
    web_search,
)

load_dotenv()

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Ollama runs an OpenAI-compatible server at this address by default.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Set OLLAMA_MODEL in your .env or environment, e.g. "llama3", "mistral", "phi3"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")


class AgentState(TypedDict, total=False):
    question: str
    task_type: str
    answer: str
    evidence: str


def get_hf_token() -> str | None:
    for env_name in HF_TOKEN_ENV_VARS:
        token = os.getenv(env_name)
        if token and token.strip():
            return token.strip()
    return None


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(value.content)
    if isinstance(value, list):
        return "\n".join(str(part) for part in value)
    return str(value)


def _classify_task(question: str) -> str:
    return classify_query(question)


def _build_answer(task_type: str, question: str) -> str:
    if task_type == "image":
        return (
            "Image-analysis workflow ready. I would inspect the image, identify the key visual details, "
            "and answer the question with grounded observations."
        )
    if task_type == "website":
        return (
            "Website-review workflow ready. I would inspect the page content, assess clarity, trustworthiness, "
            "usability, and summarize strengths and risks."
        )
    if task_type == "video":
        return (
            "Video-review workflow ready. I would extract the transcript or key moments, summarize the content, "
            "and answer the question using evidence from the video."
        )
    return (
        "General Q&A workflow ready. I would answer directly, clarify ambiguity when needed, "
        "and provide a concise, helpful response."
    )


def _build_ollama_client() -> OpenAI | None:
    """
    Build an OpenAI-compatible client pointed at the local Ollama server.
    Ollama exposes the OpenAI API at http://localhost:11434/v1 — no real API
    key is required, but the openai library needs a non-empty string.
    """
    try:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",  # placeholder; Ollama ignores the key
        )
        print(f"Ollama client created. base_url={OLLAMA_BASE_URL}, model={OLLAMA_MODEL}")
        return client
    except Exception as exc:
        print(f"Failed to create Ollama client: {exc}")
        return None


class LangGraphAgent:
    def __init__(self):
        print("LangGraphAgent (local/Ollama) initialized.")
        self.llm = _build_ollama_client()
        self.graph = self._build_graph()

    def _build_graph(self):
        def _ollama_response(client: OpenAI, prompt: str) -> str:
            """Send a prompt to the local Ollama model and return the reply text."""
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.2,
                top_p=0.95,
                stream=False,
            )

            print(f"response : {response.choices[0].message.content}")

            # Standard path: response is a ChatCompletion object
            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message"):
                    msg = first_choice.message
                    # Thinking models put the answer in `reasoning` when `content` is empty
                    content = getattr(msg, "content", None)
                    if not content:
                        content = getattr(msg, "reasoning", None)
                    return _coerce_text(content)
                if hasattr(first_choice, "text"):
                    return _coerce_text(getattr(first_choice, "text", None))

            # Fallback: response came back as a plain dict
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        message = choice.get("message")
                        if isinstance(message, dict):
                            return _coerce_text(message.get("content") or message.get("text"))
                        return _coerce_text(choice.get("text"))
                    return _coerce_text(choice)

            return _coerce_text(response)

        def route_question(state: AgentState) -> dict[str, str]:
            question = state.get("question", "")
            task_type = _classify_task(question)
            if self.llm is not None:
                try:
                    prompt = (
                        "Classify the user's request into one of: general, image, website, or video.\n"
                        f"Question: {question}\n"
                        "Return only one label."
                    )
                    response_text = _ollama_response(self.llm, prompt)
                    task_type = response_text.strip().lower() or task_type
                except Exception as exc:
                    print(f"Ollama routing failed, falling back to heuristic: {exc}")
            print(f"[ROUTE] task_type={task_type}")
            return {"task_type": task_type}

        def use_tool(state: AgentState) -> dict[str, str]:
            """Execute the appropriate tool based on task_type."""
            question = state.get("question", "")
            task_type = state.get("task_type", "general")
            evidence = ""

            try:
                if task_type == "website":
                    evidence = web_search(question, max_results=2)
                elif task_type == "video":
                    evidence = process_media("youtube", url=question)
                elif task_type == "audio":
                    evidence = process_media("audio_file", file_path=question)
                elif task_type == "code":
                    evidence = execute_python_code(question)
                elif task_type == "excel":
                    evidence = parse_spreadsheet(question)
                elif task_type == "image":
                    evidence = analyze_image(question)
                else:
                    evidence = web_search(question, max_results=2)
            except Exception as exc:
                evidence = f"Error: tool execution failed - {exc}"

            print(f"[TOOL] evidence length: {len(evidence)} chars")
            return {"evidence": evidence}

        def answer_question(state: AgentState) -> dict[str, str]:
            question = state.get("question", "")
            task_type = state.get("task_type", "general")
            evidence = state.get("evidence", "")
            
            if self.llm is not None:
                try:
                    prompt = (
                        f"Task type: {task_type}\n"
                        f"Question: {question}\n"
                    )
                    if evidence:
                        prompt += f"\n\nEvidence from tools:\n{evidence}\n\n"
                    prompt += "Provide a concise, accurate answer based on the evidence provided."
                    
                    answer_text = _ollama_response(self.llm, prompt)

                    answer = answer_text.strip() or _build_answer(task_type, question)
                except Exception as exc:
                    print(f"Ollama inference failed, falling back to template response: {exc}")
                    answer = _build_answer(task_type, question)
            else:
                answer = _build_answer(task_type, question)
            return {"answer": answer}

        workflow = StateGraph(AgentState)
        workflow.add_node("route_question", route_question)
        workflow.add_node("use_tool", use_tool)
        workflow.add_node("answer_question", answer_question)
        workflow.add_edge(START, "route_question")
        workflow.add_edge("route_question", "use_tool")
        workflow.add_edge("use_tool", "answer_question")
        workflow.add_edge("answer_question", END)
        return workflow.compile()

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question}...")
        state = self.graph.invoke({"question": question})
        answer = state.get("answer", "")
        
        return answer or "I could not produce an answer for that request."


AGENT = LangGraphAgent()


def run_and_submit_all(username: str | None = None):
    """
    Fetches all questions, runs the LangGraph-based agent on them, submits all answers,
    and displays the results.
    """
    space_id = os.getenv("SPACE_ID")

    if not username or not username.strip():
        print("No username provided.")
        return "Please enter your Hugging Face username before running the evaluation.", None

    username = username.strip()
    print(f"Using submitted username: {username}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    hf_token = get_hf_token()
    auth_headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    if hf_token:
        print("Hugging Face token found in environment.")
    else:
        print("No Hugging Face token found. Continuing without an authorization header.")

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "local"
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, headers=auth_headers, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = AGENT(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, headers=auth_headers, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        return status_message, pd.DataFrame(results_log)
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)


with gr.Blocks() as demo:
    gr.Markdown("# LangGraph Agent — Local Ollama Mode")
    gr.Markdown(
        f"""
        **Running against a local Ollama instance.**

        - Ollama endpoint: `{OLLAMA_BASE_URL}`
        - Model: `{OLLAMA_MODEL}` (set `OLLAMA_MODEL` in your `.env` to change)

        Make sure Ollama is running (`ollama serve`) and the model is pulled
        (`ollama pull {OLLAMA_MODEL}`) before using this app.
        """
    )

    question_input = gr.Textbox(
        label="Ask the agent",
        placeholder="Try: review this website, analyze this image, or summarize this video topic",
        lines=3,
    )
    ask_button = gr.Button("Ask Agent")
    agent_output = gr.Textbox(label="Agent Response", lines=8, interactive=False)

    username_input = gr.Textbox(
        label="Hugging Face username",
        placeholder="Enter your username for submission",
        lines=1,
    )
    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    ask_button.click(fn=lambda question: AGENT(question), inputs=[question_input], outputs=[agent_output])
    run_button.click(
        fn=run_and_submit_all,
        inputs=[username_input],
        outputs=[status_output, results_table],
    )


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting (Local Ollama Mode) " + "-" * 30)
    print(f"Ollama base URL : {OLLAMA_BASE_URL}")
    print(f"Ollama model    : {OLLAMA_MODEL}")
    print("-" * (60 + len(" App Starting (Local Ollama Mode) ")) + "\n")

    print("Launching Gradio Interface...")
    demo.launch(
        debug=True,
        share=False,
        server_name="localhost",
        server_port=7863,  # different port so both apps can run side-by-side
        ssr_mode=False,
    )
