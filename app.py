"""Entry point — Hugging Face Inference API mode.

Run with:
    python app.py

This module is intentionally thin: all logic lives in the ``agent/``,
``tools/``, and ``ui/`` packages.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Normalise HF token aliases before any package imports read the env.
from agent.llm import HF_MODEL, build_hf_client, get_hf_token  # noqa: E402
from agent.graph import LangGraphAgent  # noqa: E402
from ui.gradio_app import build_demo  # noqa: E402

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _initialize_environment() -> None:
    """Ensure HF token aliases are normalised."""
    os.environ.setdefault("HF_MODEL", HF_MODEL)
    if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        for alias in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            token = os.getenv(alias)
            if token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = token.strip()
                break


_initialize_environment()

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

AGENT = LangGraphAgent(llm=build_hf_client())

demo = build_demo(
    AGENT,
    mode="HF Inference",
    mode_label="Running against the Hugging Face Inference API.",
    extra_info=(
        f"- Model: `{HF_MODEL}` (set `HF_MODEL` in your `.env` to change)\n"
        "- Make sure `HF_TOKEN` is set in your `.env` before using this app."
    ),
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hf_token = get_hf_token()
    sep = "-" * 30
    print(f"\n{sep} App Starting (HF Inference Mode) {sep}")
    print(f"HF model       : {HF_MODEL}")
    print(f"HF token found : {'Yes' if hf_token else 'No — set HF_TOKEN in your .env'}")
    print(f"{sep * 2}\n")

    print("Launching Gradio Interface...")
    demo.launch(
        debug=True,
        share=True,
        server_name="localhost",
        server_port=7862,
        ssr_mode=False,
    )