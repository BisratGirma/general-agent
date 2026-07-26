import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import analyze_image, analyze_image_hf, analyze_image_ollama


def test_vision_tools_missing_file():
    res_ollama = analyze_image_ollama("non_existent_file.png")
    assert "Error" in res_ollama
    assert "Ollama Vision Tool" in res_ollama

    res_hf = analyze_image_hf("non_existent_file.png")
    assert "Error" in res_hf
    assert "HF Vision Tool" in res_hf


def test_vision_tool_dispatch(monkeypatch):
    monkeypatch.setenv("VISION_BACKEND", "hf")
    monkeypatch.setenv("HF_TOKEN", "")
    res = analyze_image("non_existent_file.png")
    assert "HF Vision Tool" in res

    monkeypatch.setenv("VISION_BACKEND", "ollama")
    res_ollama = analyze_image("non_existent_file.png")
    assert "Ollama Vision Tool" in res_ollama
