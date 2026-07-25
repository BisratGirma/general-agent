"""Entry point — Local Ollama mode.

Run with:
    python app_local.py

Requires Ollama to be running (`ollama serve`) and the model to be available
(`ollama pull <OLLAMA_MODEL>`).

This module is intentionally thin: all logic lives in the ``agent/``,
``tools/``, and ``ui/`` packages.
"""

from dotenv import load_dotenv

load_dotenv()

from agent.llm import OLLAMA_BASE_URL, OLLAMA_MODEL, build_ollama_client  # noqa: E402
from agent.graph import LangGraphAgent  # noqa: E402
from ui.gradio_app import build_demo  # noqa: E402

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

AGENT = LangGraphAgent(llm=build_ollama_client())

demo = build_demo(
    AGENT,
    mode="Local Ollama",
    mode_label="Running against a local Ollama instance.",
    extra_info=(
        f"- Ollama endpoint: `{OLLAMA_BASE_URL}`\n"
        f"- Model: `{OLLAMA_MODEL}` (set `OLLAMA_MODEL` in your `.env` to change)\n"
        f"- Make sure Ollama is running (`ollama serve`) and the model is pulled "
        f"(`ollama pull {OLLAMA_MODEL}`) before using this app."
    ),
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep = "-" * 30
    print(f"\n{sep} App Starting (Local Ollama Mode) {sep}")
    print(f"Ollama base URL : {OLLAMA_BASE_URL}")
    print(f"Ollama model    : {OLLAMA_MODEL}")
    print(f"{sep * 2}\n")

    print("Launching Gradio Interface...")
    demo.launch(
        debug=True,
        share=False,
        server_name="localhost",
        server_port=7863,   # different port so both apps can run side-by-side
        ssr_mode=False,
    )
