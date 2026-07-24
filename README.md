---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Local tool suite

This project now includes a local-first tool suite for:
- web search with Wikipedia and DuckDuckGo fallback
- YouTube transcript extraction and local Whisper transcription
- safe Python code execution in a subprocess
- CSV/XLSX spreadsheet parsing with human-readable summaries
- local image-analysis placeholders for Ollama-based workflows

Run the local app with:
- python app_local.py

The implementation is designed to work without commercial API keys and to report clear errors when optional dependencies are not installed.