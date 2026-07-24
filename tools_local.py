import ast
import base64
import csv
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
OLLAMA_VISION_MODEL = "gemma4:e4b"
OLLAMA_BASE_URL = "http://localhost:11434"


def _format_error(tool_name: str, message: str) -> str:
    return f"Error: {tool_name} failed - {message}"


def _normalize_query(query: str) -> str:
    return (query or "").strip()


def _coerce_text(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(value.content)
    if isinstance(value, list):
        return "\n".join(str(part) for part in value)
    return str(value)


def _build_page_selection_prompt(user_query: str, results: list[dict]) -> str:
    prompt_lines = [
        f"user requested this: {user_query}",
        "",
        "from the list",
    ]
    for result in results:
        title = result.get("title", "Untitled") or "Untitled"
        url = result.get("url") or result.get("href") or result.get("link") or ""
        snippet = (result.get("snippet") or result.get("body") or "").strip()
        if not url:
            continue
        line = f"- {title} / {url}"
        if snippet:
            line += f" / snippet: {snippet}"
        prompt_lines.append(line)
    prompt_lines.extend([
        "",
        "Choose the single website link (page) that most directly contains the information the user requested.",
        "Prefer text-based pages such as Wikipedia, news articles, or encyclopedia pages.",
        "Avoid selecting video pages, streaming pages, audio-only content, or social media posts.",
        "If a Wikipedia page is available, prefer it. Otherwise choose the best text-based page.",
        "Respond with only the chosen URL.",
    ])
    return "\n".join(prompt_lines)


def _is_text_webpage_url(url: str) -> bool:
    url = (url or "").lower()
    blocked_suffixes = (
        "youtube.com/watch",
        "youtu.be/",
        "vimeo.com/",
        "tiktok.com/",
        "instagram.com/",
        "facebook.com/",
        "twitter.com/",
        "soundcloud.com/",
    )
    return not any(block in url for block in blocked_suffixes)


def _is_wikipedia_query(query: str) -> bool:
    normalized = (query or "").lower()
    return "wikipedia" in normalized or "wikimedia" in normalized


def _build_llm_client():
    try:
        from openai import OpenAI
    except ImportError:
        return None

    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not base_url and not api_key:
        return None

    try:
        if base_url:
            return OpenAI(base_url=base_url, api_key=api_key or "openai")
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _select_best_page_from_results(user_query: str, results: list[dict]) -> str:
    valid_pages = []
    for result in results:
        url = result.get("url") or result.get("href") or result.get("link") or ""
        if url and url.startswith(("http://", "https://")):
            valid_pages.append({
                "title": result.get("title", "Untitled"),
                "url": url,
                "snippet": result.get("snippet", result.get("body", "")),
            })

    if not valid_pages:
        return ""

    text_pages = [page for page in valid_pages if _is_text_webpage_url(page["url"])]
    if not text_pages:
        text_pages = valid_pages

    if _is_wikipedia_query(user_query):
        for page in text_pages:
            if "wikipedia.org" in page["url"].lower():
                return page["url"]

    client = _build_llm_client()
    if client is None:
        return text_pages[0]["url"]

    prompt = _build_page_selection_prompt(user_query, text_pages)
    model_name = os.getenv("WEB_SEARCH_SELECTION_MODEL", "gpt-3.5-turbo")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.1,
            top_p=1.0,
            stream=False,
        )

        if hasattr(response, "choices") and response.choices:
            first_choice = response.choices[0]
            if hasattr(first_choice, "message"):
                selected = _coerce_text(getattr(first_choice.message, "content", None)).strip()
            else:
                selected = _coerce_text(getattr(first_choice, "text", None)).strip()
        elif isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict):
                        selected = _coerce_text(message.get("content") or message.get("text"))
                    else:
                        selected = _coerce_text(choice.get("text"))
                else:
                    selected = _coerce_text(choice)
            else:
                selected = ""
        else:
            selected = _coerce_text(response).strip()

        selected_url = selected.strip().split()[0] if selected else ""
        if selected_url.startswith(("http://", "https://")) and _is_text_webpage_url(selected_url):
            return selected_url
    except Exception:
        pass

    return text_pages[0]["url"]


def classify_query(question: str) -> str:
    normalized = _normalize_query(question).lower()
    if any(keyword in normalized for keyword in ("image", "photo", "picture", "screenshot", "analyze", "describe")):
        return "image"
    if any(keyword in normalized for keyword in ("youtube", "video", "transcript", "watch", "clip")):
        return "video"
    if any(keyword in normalized for keyword in ("audio", "transcribe", "speech", "recording")):
        return "audio"
    if any(keyword in normalized for keyword in ("python", "execute", "run code", "calculate", "compute", "script")):
        return "code"
    if any(keyword in normalized for keyword in ("csv", "xlsx", "excel", "spreadsheet", "sheet", "data file")):
        return "excel"
    if any(keyword in normalized for keyword in ("website", "webpage", "url", "site", "browse", "review", "search")):
        return "website"
    return "general"


def _web_search(query: str, max_results: int = 10) -> str:
    """Perform a web search using DuckDuckGo and scrape the content of top results."""
    query = _normalize_query(query)
    if not query:
        return _format_error("Web Search", "empty query")

    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            print(f"Performing web search for: {query}")
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No web search results found."

        output = []
        for idx, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            url = result.get("href") or result.get("link", "")
            if not url:
                continue
            output.append({"title": title, "url": url, "snippet": result.get("body", "")})

        if not output:
            return "No processable URLs found."

        selected_url = _select_best_page_from_results(query, output)
        if not selected_url:
            return "No valid page could be selected from search results."

        print(f"Selected page to scrape: {selected_url}")
        scraped_content = scrape_url(selected_url)

        return (
            f"=== Selected Source ===\n"
            f"URL: {selected_url}\n"
            f"Content:\n{scraped_content}"
        )

    except ImportError:
        return _format_error("Web Search", "ddgs package is not installed")
    except Exception as exc:
        return _format_error("Web Search", str(exc))

def web_search(query: str, max_results: int = 10) -> str:
    """Perform a web search using DuckDuckGo."""

    web_result = _web_search(query, max_results=max_results)

    print(f"Web search result: {web_result}")
    return f"{web_result}"


import requests

def scrape_url(url: str) -> str:
    """Fetch a web page and convert it to readable text."""
    url = _normalize_query(url)
    if not url.startswith(("http://", "https://")):
        return _format_error("HTML Scrape", "URL must start with http:// or https://")

    # Pass a descriptive User-Agent or a standard browser User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return _format_error("HTML Scrape", str(exc))

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _format_error("HTML Scrape", "beautifulsoup4 is not installed")

    soup = BeautifulSoup(response.text, "html.parser")
    text_parts = []
    for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        cleaned = " ".join(tag.get_text(" ", strip=True).split())
        if cleaned:
            text_parts.append(cleaned)
            
    text = "\n".join(text_parts)[:6000]
    return text or _format_error("HTML Scrape", "no readable text found")

def process_youtube_transcript(url: str, preferred_language: str = "en") -> str:
    """Extract a YouTube transcript using youtube-transcript-api when available."""
    url = _normalize_query(url)
    if not url:
        return _format_error("YouTube Transcript", "missing URL")

    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        return _format_error("YouTube Transcript", "could not find a YouTube video ID")
    video_id = match.group(1)

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return _format_error("YouTube Transcript", "youtube-transcript-api is not installed")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript([preferred_language])
        except Exception:
            transcript = transcript_list.find_generated_transcript([preferred_language]) if hasattr(transcript_list, "find_generated_transcript") else None
        if transcript is None:
            transcript = next(iter(transcript_list), None)
        if transcript is None:
            return _format_error("YouTube Transcript", "no transcripts were found for this video")
        fetched = transcript.fetch()
        if not fetched:
            return _format_error("YouTube Transcript", "transcript fetch returned no content")
        segments = [item.get("text", "") for item in fetched if isinstance(item, dict)]
        return "\n".join(segments)[:6000]
    except Exception as exc:
        return _format_error("YouTube Transcript", str(exc))


def transcribe_audio_file(file_path: str, model_size: str = "base") -> str:
    """Transcribe supported audio files with local Whisper when available."""
    file_path = _normalize_query(file_path)
    if not file_path or not os.path.exists(file_path):
        return _format_error("Whisper", "file does not exist")

    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(SUPPORTED_AUDIO_EXTENSIONS)
        return _format_error("Whisper", f"unsupported format '{suffix}'. Supported: {supported}")

    try:
        import whisper
    except ImportError:
        return _format_error("Whisper", "openai-whisper is not installed")

    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(file_path)
        text = result.get("text", "")
        return text or _format_error("Whisper", "no transcription text was produced")
    except Exception as exc:
        return _format_error("Whisper", str(exc))


def process_media(source_type: str, url: Optional[str] = None, file_path: Optional[str] = None, whisper_model: str = "base", preferred_language: str = "en") -> str:
    """Route media requests to the appropriate local processing function."""
    source_type = (source_type or "").lower()
    if source_type == "youtube":
        return process_youtube_transcript(url or "", preferred_language=preferred_language)
    if source_type == "audio_file":
        return transcribe_audio_file(file_path or "", model_size=whisper_model)
    return _format_error("Media Tool", "unsupported source type")


def execute_python_code(code: str, timeout: int = 30) -> str:
    """Execute Python code safely in an isolated subprocess."""
    code = (code or "").strip()
    if not code:
        return _format_error("Code Interpreter", "empty code")

    blocked_names = {
        "os",
        "subprocess",
        "sys",
        "socket",
        "requests",
        "shutil",
        "pathlib",
        "ctypes",
        "pickle",
        "urllib",
        "http",
        "ssl",
        "multiprocessing",
        "importlib",
        "builtins",
    }
    blocked_calls = {"open", "eval", "exec", "compile", "__import__", "input", "exit", "quit"}

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return _format_error("Code Interpreter", f"syntax error: {exc}")

    class SecurityVisitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name.split(".")[0] in blocked_names:
                    raise ValueError(f"Forbidden import: {alias.name}")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module and node.module.split(".")[0] in blocked_names:
                raise ValueError(f"Forbidden import: {node.module}")
            self.generic_visit(node)

        def visit_Call(self, node):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name in blocked_calls:
                raise ValueError(f"Forbidden call: {func_name}")
            self.generic_visit(node)

    try:
        SecurityVisitor().visit(tree)
    except ValueError as exc:
        return _format_error("Code Interpreter", f"Security violation - {exc}")

    script = textwrap.dedent(code)
    kwargs = {"capture_output": True, "text": True, "timeout": timeout}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    try:
        result = subprocess.run([sys.executable, "-c", script], **kwargs)
    except subprocess.TimeoutExpired:
        return _format_error("Code Interpreter", "execution timed out")
    except Exception as exc:
        return _format_error("Code Interpreter", str(exc))

    output_parts = []
    if result.stdout.strip():
        output_parts.append(result.stdout.strip())
    if result.stderr.strip():
        output_parts.append(f"STDERR:\n{result.stderr.strip()}")
    if not output_parts:
        return "Output:\n<no output>"
    return "Output:\n" + "\n".join(output_parts)


def parse_spreadsheet(file_path: str, sheet_name: Optional[str] = None, query: Optional[str] = None, row_range: Optional[tuple[int, int]] = None, column_range: Optional[tuple[int, int]] = None) -> str:
    """Parse CSV or XLSX files into a human-readable summary."""
    file_path = _normalize_query(file_path)
    if not file_path or not os.path.exists(file_path):
        return _format_error("Spreadsheet Parser", "file does not exist")

    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        try:
            with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            return _format_error("Spreadsheet Parser", str(exc))
        if not rows:
            return _format_error("Spreadsheet Parser", "no rows found")

        headers = list(rows[0].keys())
        filtered_rows = list(rows)
        if query:
            search_term = query.lower()
            filtered_rows = [row for row in rows if any(search_term in str(value).lower() for value in row.values())]

        lines = [f"Parsed {Path(file_path).name} ({len(filtered_rows)} matching rows, {len(headers)} columns)"]
        lines.append(" | ".join(headers))
        for row in filtered_rows[:10]:
            lines.append(" | ".join(str(row.get(header, "")) for header in headers))
        return "\n".join(lines)

    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError:
            return _format_error("Spreadsheet Parser", "pandas/openpyxl is not installed")
        try:
            data_frame = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as exc:
            return _format_error("Spreadsheet Parser", str(exc))
        if query:
            search_term = query.lower()
            mask = data_frame.astype(str).apply(lambda column: column.str.contains(search_term, case=False, na=False)).any(axis=1)
            data_frame = data_frame[mask]
        return f"Parsed {Path(file_path).name} ({len(data_frame)} rows, {len(data_frame.columns)} columns)\n" + data_frame.head(10).to_string(index=False)

    return _format_error("Spreadsheet Parser", f"unsupported file type '{suffix}'")


def analyze_image(image_path: str, task_type: str = "general", prompt: Optional[str] = None) -> str:
    """Analyze an image using the local Ollama vision model (gemma4:e4b).

    Args:
        image_path: Absolute or relative path to the image file.
        task_type: "chess" for board analysis, "general" for open-ended description.
        prompt: Optional custom prompt; overrides the default for the task type.

    Returns:
        The model's text response, or a formatted error string on failure.
    """
    image_path = _normalize_query(image_path)
    if not image_path or not os.path.exists(image_path):
        return _format_error("Vision Tool", "image file does not exist")

    # --- Health-check: ensure Ollama is reachable ---
    try:
        health = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if health.status_code != 200:
            raise requests.RequestException("unexpected status from Ollama")
    except requests.RequestException:
        return _format_error("Vision Tool", "Ollama service not running. Start it with 'ollama serve'.")

    # --- Build task-aware prompt ---
    if prompt:
        prompt_text = prompt
    elif task_type.lower() == "chess":
        prompt_text = (
            "You are a chess expert. Carefully examine this chess board image. "
            "Describe the exact position of every piece you can see (use standard algebraic notation for squares). "
            "Then provide a brief evaluation of the position, noting which side has the advantage and why."
        )
    else:
        prompt_text = "Describe the image in detail, noting key objects, colors, layout, and any text visible."

    # --- Base64-encode the image ---
    try:
        with open(image_path, "rb") as img_file:
            image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    except OSError as exc:
        return _format_error("Vision Tool", f"could not read image file: {exc}")

    # --- Call Ollama /api/generate ---
    payload = {
        "model": OLLAMA_VISION_MODEL,
        "prompt": prompt_text,
        "images": [image_b64],
        "stream": False,
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,  # vision inference can be slow on CPU
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        return _format_error("Vision Tool", f"Ollama API call failed: {exc}")
    except ValueError as exc:
        return _format_error("Vision Tool", f"could not parse Ollama response: {exc}")

    model_response = result.get("response", "").strip()
    if not model_response:
        return _format_error("Vision Tool", "Ollama returned an empty response")

    return model_response
