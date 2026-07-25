"""Shared Gradio UI builder.

Call :func:`build_demo` with an agent and a mode string to get a fully
configured ``gr.Blocks`` instance.  The two entry points (``app.py`` for HF
Inference and ``app_local.py`` for local Ollama) each call this function with
different parameters instead of duplicating the Gradio layout.
"""

from __future__ import annotations

from typing import Callable

import gradio as gr

from agent.submission import run_and_submit_all


def build_demo(
    agent,
    *,
    mode: str,
    mode_label: str,
    extra_info: str = "",
) -> gr.Blocks:
    """Construct the Gradio application for *agent*.

    Args:
        agent: A callable ``(question: str) -> str``.
        mode: Short identifier shown in the header (e.g. ``"HF Inference"``).
        mode_label: Longer description shown in the sub-heading.
        extra_info: Additional markdown appended to the info block.

    Returns:
        A compiled ``gr.Blocks`` Gradio application.
    """

    def _submit(username: str):
        return run_and_submit_all(agent, username)

    with gr.Blocks(title=f"LangGraph Agent — {mode}") as demo:
        gr.Markdown(f"# LangGraph Agent — {mode}")
        gr.Markdown(
            f"**{mode_label}**\n\n"
            "Routes questions into: general, image, website, video, audio, code, or excel workflows.\n"
            "Uses a LangGraph graph for the orchestration layer.\n\n"
            + extra_info
        )

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Ask the agent",
                    placeholder="Try: review this website, analyze this image, or summarize this video topic",
                    lines=3,
                )
                file_input = gr.File(
                    label="Attach File (Spreadsheet / Image / Audio)",
                    file_types=[".csv", ".xlsx", ".xls", "image", "audio"],
                    type="filepath",
                )
                ask_button = gr.Button("Ask Agent", variant="primary")
                agent_output = gr.Textbox(label="Agent Response", lines=8, interactive=False)

        gr.Markdown("---")
        gr.Markdown("### 📊 Evaluation & Submission")

        with gr.Row():
            with gr.Column(scale=2):
                username_input = gr.Textbox(
                    label="Hugging Face username",
                    placeholder="Enter your username for submission",
                    lines=1,
                )
                run_button = gr.Button("Run Evaluation & Submit All Answers", variant="secondary")

        status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
        results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

        def _handle_ask(question: str, file_path: str | None) -> str:
            full_query = question.strip() if question else ""
            if file_path:
                if full_query:
                    full_query = f"{full_query} (File: {file_path})"
                else:
                    full_query = f"Process file: {file_path}"
            return agent(full_query)

        # Wire up events
        ask_button.click(
            fn=_handle_ask,
            inputs=[question_input, file_input],
            outputs=[agent_output],
        )
        run_button.click(
            fn=_submit,
            inputs=[username_input],
            outputs=[status_output, results_table],
        )

    return demo
