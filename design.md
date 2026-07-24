# Design Document: Agent Tool Suite

## Overview

This document describes the architecture, components, interfaces, data models, and correctness properties for the Agent Tool Suite. The suite provides five integrated tools for a LangGraph-based agent system, all using **free and open-source solutions** with no commercial API dependencies.

**Critical Constraint**: All tools must operate without commercial API keys (OpenAI, Anthropic, Google). The system runs entirely locally on Windows environments.

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph Agent                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Query Router │───▶│ Tool Nodes   │───▶│ Response     │      │
│  │ (Hybrid)     │    │ (5 Tools)    │    │ Formatter    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ LLM Classify│    │ Tool State   │    │ Error Handler│      │
│  │ + Heuristics│    │ Management   │    │ + Retries    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Tool Suite                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │Web_Search   │ │ Media_Tool  │ │Code_Interp  │               │
│  │Tool         │ │             │ │Tool         │               │
│  │- Wikipedia  │ │- YouTube    │ │- Sandbox    │               │
│  │- DuckDuckGo │ │- Whisper    │ │- Validation │               │
│  │- BeautifulSoup│            │ │- Execution  │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│  ┌─────────────┐ ┌─────────────┐                                │
│  │Excel_Parser │ │ Vision_Tool │                                │
│  │Tool         │ │             │                                │
│  │- pandas     │ │- Ollama     │                                │
│  │- openpyxl   │ │- Stockfish  │                                │
│  └─────────────┘ └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   External Dependencies                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ Ollama      │ │ ffmpeg      │ │ Stockfish   │               │
│  │ (gemma4:e4b)│ │ (audio)     │ │ (chess)     │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph Workflow

The agent uses a state machine workflow with the following nodes:

```python
class AgentState(TypedDict, total=False):
    question: str           # Input query
    task_type: str          # Classified type
    tool_results: dict      # Results from tool execution
    answer: str             # Final answer
    evidence: str           # Supporting evidence
    error: str              # Error information
    retries_remaining: int  # Retry counter
```

Graph structure: `START → route_question → [tool_node_*] → synthesize_answer → END`

### Query Classification Flow

**Classification Heuristics**:
- `web` / `wikipedia`: "wikipedia", "search", "look up", factual questions
- `video` / `youtube`: "youtube", "video", "transcript", "watch"
- `audio`: audio file paths, "transcribe", "audio", "speech"
- `code`: "execute", "run code", "python", "calculate", "compute"
- `excel`: ".xlsx", ".csv", "spreadsheet", "excel", "data file"
- `image` / `vision`: "image", "photo", "picture", "chess board", "analyze image"
- `general`: default fallback

---

## Component Interfaces

### 1. Web_Search_Tool

**Purpose**: Search Wikipedia and the web using free tools.

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class SearchSource(Enum):
    WIKIPEDIA_API = "wikipedia_api"
    WIKIPEDIA_FULL = "wikipedia_full"
    DUCKDUCKGO = "duckduckgo"
    GENERAL_URL = "general_url"

@dataclass
class WebSearchInput:
    query: str
    source: Optional[SearchSource] = None
    url: Optional[str] = None
    max_results: int = 3

@dataclass
class WebSearchOutput:
    content: str
    source: SearchSource
    url: Optional[str] = None
    error: Optional[str] = None

def web_search(input: WebSearchInput) -> WebSearchOutput:
    """
    Search the web using free sources.
    Priority: URL fetch → Wikipedia API → DuckDuckGo fallback
    """
    ...
```

**Key Implementation Notes**:
- Uses Wikipedia API (`https://en.wikipedia.org/w/api.php`) for summaries
- Uses DuckDuckGo via `ddgs` library (no API key required)
- BeautifulSoup for HTML parsing and text extraction
- Fallback chain: Wikipedia → DuckDuckGo on empty results

---

### 2. Media_Tool

**Purpose**: Process YouTube videos and audio files using local Whisper.

```python
@dataclass
class MediaInput:
    source_type: str  # "youtube" or "audio_file"
    url: Optional[str] = None
    file_path: Optional[str] = None
    whisper_model: str = "base"
    preferred_language: str = "en"

@dataclass
class MediaOutput:
    transcript: str
    source: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

def process_media(input: MediaInput) -> MediaOutput:
    """
    Process video or audio content.
    YouTube: API first, fall back to Whisper
    Audio: Local Whisper directly
    """
    ...
```

**Key Implementation Notes**:
- `youtube_transcript_api` for YouTube transcripts (free)
- Local `openai-whisper` library for audio transcription (no API calls)
- Supported audio formats: MP3, WAV, M4A, FLAC, OGG
- Whisper model sizes: tiny, base, small, medium, large

---

### 3. Code_Interpreter_Tool

**Purpose**: Execute Python code safely in an isolated sandbox.

```python
@dataclass
class CodeInput:
    code: str
    timeout: int = 30
    max_output_size: int = 10000

@dataclass
class CodeOutput:
    stdout: str
    stderr: str
    return_value: Optional[str] = None
    execution_time: float = 0.0
    error: Optional[str] = None

def execute_code(input: CodeInput) -> CodeOutput:
    """
    Execute Python code in isolated subprocess.
    Security validation blocks dangerous operations.
    """
    ...
```

**Security Model**:
- AST-based validation before execution
- Blocked modules: os, subprocess, socket, requests, shutil, etc.
- Blocked functions: open, eval, exec, compile, __import__
- Subprocess isolation with timeout enforcement
- Windows-specific: `CREATE_NO_WINDOW` flag for clean execution

---

### 4. Excel_Parser_Tool

**Purpose**: Parse and analyze spreadsheet files.

```python
@dataclass
class ExcelInput:
    file_path: str
    sheet_name: Optional[str] = None
    query: Optional[str] = None
    row_range: Optional[tuple[int, int]] = None
    column_range: Optional[tuple[int, int]] = None

@dataclass
class ExcelOutput:
    data: str  # Human-readable formatted text
    rows: int
    columns: int
    source: str
    error: Optional[str] = None

def parse_excel(input: ExcelInput) -> ExcelOutput:
    """
    Parse Excel or CSV file and return data.
    Supports .xlsx, .xls, .csv
    """
    ...
```

**Key Implementation Notes**:
- pandas for data manipulation
- openpyxl for Excel file reading
- Human-readable text output (not raw serialization)
- Query capability for searching within data

---

### 5. Vision_Tool

**Purpose**: Analyze images using local Ollama with chess analysis support.

```python
@dataclass
class VisionInput:
    image_path: str
    task_type: str = "general"  # "general" or "chess"
    prompt: Optional[str] = None

@dataclass
class VisionOutput:
    description: str
    chess_fen: Optional[str] = None
    chess_best_move: Optional[str] = None
    error: Optional[str] = None

def analyze_image(input: VisionInput) -> VisionOutput:
    """
    Analyze image using local Ollama (gemma4:e4b).
    Chess boards: identify position, generate FEN, use Stockfish.
    """
    ...
```

**Key Implementation Notes**:
- Local Ollama at `http://localhost:11434/v1` (OpenAI-compatible)
- Model: `gemma4:e4b` (vision-capable, no API key required)
- Stockfish integration for chess position evaluation
- Helpful error messages for service/model availability

---

## Data Models

### Tool Results Schema

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class ToolType(Enum):
    WEB_SEARCH = "web_search"
    MEDIA = "media"
    CODE_INTERPRETER = "code_interpreter"
    EXCEL_PARSER = "excel_parser"
    VISION = "vision"

@dataclass
class ToolResult:
    success: bool
    output: str
    tool: ToolType
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error or self.output}"
```

### Configuration Schema

```python
@dataclass
class ToolConfig:
    # Web Search
    wikipedia_timeout: int = 10
    duckduckgo_max_results: int = 3
    
    # Media
    whisper_model: str = "base"
    preferred_transcript_language: str = "en"
    
    # Code Interpreter
    code_timeout: int = 30
    max_output_size: int = 10000
    
    # Vision
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma4:e4b"
    
    # Error Handling
    max_retries: int = 2
    retry_delay: float = 1.0
```

---

## Error Handling Strategy

### Error Types

```python
class ErrorType(Enum):
    NETWORK = "network"       # Network request failed
    TIMEOUT = "timeout"       # Operation timed out
    NOT_FOUND = "not_found"   # Resource not found
    INVALID_INPUT = "invalid" # Invalid user input
    SECURITY = "security"     # Security violation
    SERVICE = "service"       # External service unavailable
    PARSING = "parsing"       # Data parsing failed
    INTERNAL = "internal"     # Internal error
```

### Error Response Format

All tools return errors in consistent format:
- Success: Return result string directly
- Failure: Return `"Error: <description>"` prefix

### Retry Strategy

```python
def with_retry(tool_func, *args, max_retries=2, delay=1.0, **kwargs):
    """Execute tool with retry logic."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = tool_func(*args, **kwargs)
            if not result.startswith("Error:"):
                return result
            last_error = result
        except Exception as e:
            last_error = f"Error: {e}"
        
        if attempt < max_retries:
            time.sleep(delay * (attempt + 1))
    
    return last_error
```

### Graceful Degradation

1. **Wikipedia API failure** → Fall back to DuckDuckGo
2. **YouTube transcript API failure** → Fall back to Whisper
3. **Ollama unavailable** → Return instructions to start service
4. **Code validation failure** → Return specific security violation
5. **File parsing failure** → Return error for that file, continue with others

---

## Security Model

### Code Interpreter Security

**Blocked Operations**:
- File system: `open`, `os.remove`, `shutil`, `pathlib` write operations
- Network: `requests`, `urllib`, `socket`, `http.client`
- System: `subprocess`, `os.system`, `eval`, `exec`
- Imports: All dangerous modules blocked at import time

**Validation Process**:
1. Parse code into AST
2. Walk AST tree for violations
3. Reject if any violations found
4. Execute in isolated subprocess if clean

### Input Validation

- File paths: Check existence and extension
- URLs: Validate format and protocol (http/https only)
- Code: AST-based security scan
- Queries: Sanitize for injection attacks

---

## Dependencies

### Python Packages (requirements.txt)

```
# Core
langgraph>=1.2.0
gradio==5.25.2
python-dotenv>=1.0.0
requests>=2.31.0

# Web Search
ddgs>=2.0.0
beautifulsoup4>=4.12.0

# Media Processing
youtube-transcript-api>=0.6.0
openai-whisper>=20231117
yt-dlp>=2024.1.0

# Data Processing
pandas>=2.0.0
openpyxl>=3.1.0

# Vision
openai>=1.0.0  # For Ollama OpenAI-compatible API

# Chess Analysis
python-chess>=1.10.0
```

### System Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| ffmpeg | Audio processing for Whisper | `winget install ffmpeg` or `choco install ffmpeg` |
| Ollama | Local LLM inference | Download from ollama.ai |
| gemma4:e4b model | Vision model | `ollama pull gemma4:e4b` |
| Stockfish | Chess engine | Download from stockfishchess.org |

---

## Windows-Specific Considerations

### Path Handling

```python
import os
from pathlib import Path

def normalize_path(path: str) -> str:
    """Handle Windows path separators and drive letters."""
    return str(Path(path).resolve())

def validate_windows_path(path: str) -> bool:
    """Validate path for Windows filesystem."""
    try:
        Path(path).resolve()
        return True
    except:
        return False
```

### Subprocess Creation

```python
import subprocess
import sys

def run_subprocess(command: list[str], timeout: int = 30) -> tuple[str, str]:
    """Run subprocess with Windows-compatible settings."""
    kwargs = {
        'capture_output': True,
        'text': True,
        'timeout': timeout,
    }
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    
    result = subprocess.run(command, **kwargs)
    return result.stdout, result.stderr
```

### Environment Variables

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
```

---

## Integration with LangGraph

### Graph Definition

```python
from langgraph.graph import StateGraph, START, END

def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("route_question", route_question)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("media_node", media_node)
    workflow.add_node("code_node", code_node)
    workflow.add_node("excel_node", excel_node)
    workflow.add_node("vision_node", vision_node)
    workflow.add_node("synthesize_answer", synthesize_answer)
    
    # Add edges
    workflow.add_edge(START, "route_question")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "route_question",
        route_to_tool,
        {
            "web": "web_search_node",
            "video": "media_node",
            "audio": "media_node",
            "code": "code_node",
            "excel": "excel_node",
            "image": "vision_node",
            "general": "synthesize_answer"
        }
    )
    
    # All tools lead to answer synthesis
    for node in ["web_search_node", "media_node", "code_node", 
                 "excel_node", "vision_node"]:
        workflow.add_edge(node, "synthesize_answer")
    
    workflow.add_edge("synthesize_answer", END)
    
    return workflow.compile()
```

### Tool Node Implementation

```python
def web_search_node(state: AgentState) -> dict:
    """LangGraph node for web search tool."""
    question = state.get("question", "")
    result = web_search(query=question)
    return {"tool_results": {"web_search": result}}

def media_node(state: AgentState) -> dict:
    """LangGraph node for media processing tool."""
    question = state.get("question", "")
    task_type = state.get("task_type", "")
    
    if task_type == "video":
        result = process_youtube(url=extract_url(question))
    else:
        result = process_audio_file(file_path=extract_file_path(question))
    
    return {"tool_results": {"media": result}}
```

---

## Correctness Properties

*Properties are formal statements about what the system should do. Each property must hold for all valid inputs.*

### Property 1: Wikipedia API Response Format

*For any* valid Wikipedia query, the Web_Search_Tool SHALL return a response prefixed with "Wikipedia" and containing page summary text.

**Validates: Requirements 1.1**

### Property 2: DuckDuckGo Search Results Format

*For any* search query, DuckDuckGo results SHALL contain formatted entries with titles and body snippets.

**Validates: Requirements 1.2**

### Property 3: Wikipedia Fallback Behavior

*For any* query that returns no Wikipedia results, the Web_Search_Tool SHALL automatically fall back to DuckDuckGo search.

**Validates: Requirements 1.5**

### Property 4: YouTube Transcript Language Preference

*For any* YouTube video with multiple transcript languages, the Media_Tool SHALL prefer and return the English transcript.

**Validates: Requirements 2.5**

### Property 5: Audio Transcription with Whisper

*For any* valid audio file in supported formats (MP3, WAV, M4A, FLAC, OGG), the Media_Tool SHALL return a transcription string.

**Validates: Requirements 3.1**

### Property 6: Whisper Model Size Support

*For any* supported Whisper model size (tiny, base, small, medium, large), the Media_Tool SHALL produce valid transcription output.

**Validates: Requirements 3.2**

### Property 7: Code Security Validation

*For any* Python code containing dangerous operations (file system, network, system commands), the Code_Interpreter_Tool SHALL block execution and return a security error.

**Validates: Requirements 4.1, 4.5**

### Property 8: Safe Code Execution

*For any* Python code that passes security validation, the Code_Interpreter_Tool SHALL execute it in an isolated subprocess and return output.

**Validates: Requirements 4.2, 4.3**

### Property 9: Excel File Parsing

*For any* valid .xlsx or .csv file, the Excel_Parser_Tool SHALL load and return structured data in human-readable text format.

**Validates: Requirements 5.1, 5.2, 5.6**

### Property 10: Excel Data Querying

*For any* valid search query against loaded spreadsheet data, the Excel_Parser_Tool SHALL return matching rows.

**Validates: Requirements 5.3**

### Property 11: Vision Analysis with Ollama

*For any* valid image file, the Vision_Tool SHALL process it using local Ollama and return a description string.

**Validates: Requirements 6.1**

### Property 12: Chess Board Recognition

*For any* chess board image explicitly provided, the Vision_Tool SHALL identify the position, generate FEN notation, and provide Stockfish analysis.

**Validates: Requirements 6.2, 6.3**

### Property 13: Query Routing Correctness

*For any* query with a recognizable type pattern, the Query_Router SHALL route to the appropriate tool node.

**Validates: Requirements 7.2**

### Property 14: Heuristic Fallback Classification

*For any* query when the LLM is unavailable, the Query_Router SHALL successfully classify using heuristic keyword matching.

**Validates: Requirements 7.3, 7.4**

### Property 15: Tool Error Handling on Failure

*For any* tool that returns an error, the LangGraph_Agent SHALL attempt alternative tools or return a user-friendly error message.

**Validates: Requirements 7.6**

### Property 16: String Return Type

*For any* tool that completes successfully, the return value SHALL be a string containing the result.

**Validates: Requirements 8.1**

### Property 17: Error Prefix Format

*For any* tool that encounters an error, the return value SHALL be a string prefixed with "Error:" followed by a description.

**Validates: Requirements 8.2**

### Property 18: Citation Inclusion

*For any* tool that uses external sources, the response SHALL include citations or URLs.

**Validates: Requirements 8.4**

### Property 19: Retry Behavior

*For any* tool failure that is potentially transient, the LangGraph_Agent SHALL retry the operation up to the configured maximum.

**Validates: Requirements 9.2**

### Property 20: Windows Path Handling

*For any* file path containing Windows-specific elements (backslashes, drive letters), the tools SHALL process it correctly.

**Validates: Requirements 11.2**

### Property 21: API Key Independence

*For any* execution, the Agent_Tool_Suite SHALL operate correctly without requiring OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY.

**Validates: Requirements 12.1, 12.2, 12.3**

### Property 22: Local Tools Priority

*For any* execution where commercial API keys are present in environment variables, the tools SHALL ignore them and use local alternatives.

**Validates: Requirements 12.4**

---

## Testing Strategy

### Property-Based Tests

Run with minimum 100 iterations per property using a property-based testing framework (hypothesis for Python):

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=200))
def test_wikipedia_response_format(query):
    result = web_search(query)
    # Either returns Wikipedia content, DuckDuckGo results, or error
    assert isinstance(result, str)
    assert len(result) > 0 or result.startswith("Error:")

@given(st.sampled_from(['tiny', 'base', 'small', 'medium', 'large']))
def test_whisper_model_sizes(model_size):
    # Test that all model sizes can be loaded
    # (would use sample audio file)
    pass
```

### Integration Tests

- Test end-to-end workflows with representative examples
- Verify tool integration in LangGraph graph
- Test error recovery and fallback behavior

### Unit Tests

- Test individual functions with specific inputs
- Verify error message formats
- Test edge cases (empty inputs, invalid formats)

---

## Appendix: Response Format Examples

### Web Search Success

```
Wikipedia (Mercedes Sosa): Haydée Mercedes "Mecha" Sosa was an Argentine singer...
```

### Web Search Error

```
Error: Wikipedia search failed - Connection timeout
```

### Media Transcription Success

```
Transcription:
[Content of transcribed audio]

Execution time: 12.34s
```

### Code Execution Success

```
Output:
['filtered', 'results']

Execution time: 0.15s
```

### Code Security Error

```
Error: Security violation - Forbidden import: os
```

### Vision Analysis Success

```
This image shows a chess board from the white player's perspective. 
The position shows...

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Best move: e2e4
```

### Vision Service Unavailable

```
Error: Ollama service not running. Start with 'ollama serve'.
```
