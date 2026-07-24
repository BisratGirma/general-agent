import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools_local as tools_local_module
from tools_local import execute_python_code, parse_spreadsheet


def test_execute_python_code_blocks_dangerous_import():
    result = execute_python_code("import os\nprint('blocked')")
    assert result.startswith("Error:")
    assert "Security" in result or "danger" in result.lower()


def test_execute_python_code_runs_safe_script():
    result = execute_python_code("x = 2 + 3\nprint(x)")
    assert result.startswith("Output:")
    assert "5" in result


def test_parse_spreadsheet_handles_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name", "Score"])
        writer.writerow(["Ada", 10])
        writer.writerow(["Grace", 8])

    result = parse_spreadsheet(str(csv_path), query="Ada")
    assert result.startswith("Parsed")
    assert "Ada" in result


def test_build_page_selection_prompt_includes_user_query_and_results():
    user_query = "Find the latest news about renewable energy"
    results = [
        {"title": "Energy News", "url": "https://example.com/news", "snippet": "..."},
        {"title": "Renewables Today", "url": "https://example.com/renew", "snippet": "..."},
    ]

    prompt = tools_local_module._build_page_selection_prompt(user_query, results)

    assert "user requested this: Find the latest news about renewable energy" in prompt
    assert "from the list" in prompt
    assert "- Energy News / https://example.com/news" in prompt
    assert "- Renewables Today / https://example.com/renew" in prompt
    assert "Choose the single website link (page) that most directly contains the information the user requested." in prompt


def test_select_best_page_falls_back_to_first_valid_url(monkeypatch):
    results = [
        {"title": "A", "url": "https://example.com/a"},
        {"title": "B", "url": "https://example.com/b"},
    ]

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OLLAMA_BASE_URL", "")

    selected = tools_local_module._select_best_page_from_results("Any query", results)
    assert selected == "https://example.com/a"


def test_text_webpage_filter_blocks_video_sites():
    assert tools_local_module._is_text_webpage_url("https://www.youtube.com/watch?v=xyz") is False
    assert tools_local_module._is_text_webpage_url("https://youtu.be/xyz") is False
    assert tools_local_module._is_text_webpage_url("https://en.wikipedia.org/wiki/Mercedes_Sosa") is True


def test_wikipedia_query_preference():
    results = [
        {"title": "YouTube video", "url": "https://www.youtube.com/watch?v=xyz"},
        {"title": "Wikipedia entry", "url": "https://en.wikipedia.org/wiki/Mercedes_Sosa"},
    ]
    selected = tools_local_module._select_best_page_from_results(
        "Use wikipedia to answer the question about Mercedes Sosa.",
        results,
    )
    assert selected == "https://en.wikipedia.org/wiki/Mercedes_Sosa"
