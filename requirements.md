# Requirements Document

## Introduction

This document specifies the requirements for a comprehensive agent tool suite that integrates with an existing LangGraph-based agent system. The suite provides five primary tools: web search with Wikipedia and DuckDuckGo, audio/video intelligence processing via local Whisper, Python code execution sandbox, document/spreadsheet analysis, and vision/multimodal inspection capabilities via local Ollama. **Critical constraint: All tools must use FREE and OPEN-SOURCE solutions only. No commercial APIs (OpenAI, Anthropic, Google) are permitted.** The system is designed to run entirely locally on Windows environments.

## Glossary

- **Agent_Tool_Suite**: The collection of five integrated tools that extend the LangGraph agent's capabilities
- **Web_Search_Tool**: Tool for Wikipedia API queries, DuckDuckGo search, and HTML page scraping
- **Media_Tool**: Tool for YouTube transcript extraction and local Whisper-based audio transcription
- **Code_Interpreter_Tool**: Isolated Python execution environment for running user-provided code
- **Excel_Parser_Tool**: Tool for analyzing spreadsheet files (.xlsx, .csv) and extracting structured data
- **Vision_Tool**: Multimodal tool using local Ollama (gemma4:e4b) with chess analysis via Stockfish
- **LangGraph_Agent**: The existing agent orchestration system in app.py
- **Tools_Module**: The existing tools_local.py module containing helper functions
- **Wikipedia_API**: The MediaWiki API endpoint for querying Wikipedia content
- **DuckDuckGo**: Free, open-source search engine API for web searches
- **YouTube_Transcript_API**: Library for extracting transcripts from YouTube videos
- **Local_Whisper**: The openai-whisper library running locally for audio transcription
- **Ollama_Vision**: Local Ollama instance with gemma4:e4b model for image analysis
- **Stockfish**: Open-source chess engine for position evaluation
- **Query_Router**: The LangGraph node responsible for classifying and routing queries to appropriate tools

## Requirements

### Requirement 1: Web Search and Wikipedia Integration

**User Story:** As an agent user, I want to search Wikipedia and the web using free tools, so that I can answer questions about factual topics without relying on commercial APIs.

#### Acceptance Criteria

1. WHEN a Wikipedia query is received, THE Web_Search_Tool SHALL query the Wikipedia API and return the page summary
2. WHEN a general web search is needed, THE Web_Search_Tool SHALL use DuckDuckGo (via ddgs library) to retrieve search results at no cost
3. WHEN a full Wikipedia page is needed, THE Web_Search_Tool SHALL scrape the complete page content and extract text using BeautifulSoup
4. WHEN a non-Wikipedia URL is provided, THE Web_Search_Tool SHALL fetch and parse HTML content into readable text using BeautifulSoup
5. IF the Wikipedia API returns no results, THEN THE Web_Search_Tool SHALL fall back to DuckDuckGo search automatically
6. WHEN a query mentions "Mercedes Sosa", "Dinosaur", "Roy White", "NASA Award", "Kuznetzov paper", "1928 Olympics", "Taishō Tamai", or "Malko Competition", THE Web_Search_Tool SHALL prioritize Wikipedia as the information source
7. WHEN an error occurs during content retrieval, THE Web_Search_Tool SHALL return both the partial content (if available) and the error message together

### Requirement 2: YouTube Video Processing

**User Story:** As an agent user, I want to extract transcripts from YouTube videos, so that I can answer questions about video content without watching the entire video.

#### Acceptance Criteria

1. WHEN a YouTube URL is provided, THE Media_Tool SHALL extract the video transcript using the free youtube_transcript_api library
2. IF a transcript is unavailable via the API, THEN THE Media_Tool SHALL attempt to download audio and use local openai-whisper for transcription
3. WHEN a video about bird species, Teal'c character, or other specific topics is queried, THE Media_Tool SHALL return relevant content from the transcript
4. IF any failure occurs in the video processing pipeline, THEN THE Media_Tool SHALL generate a descriptive error message indicating the specific failure reason
5. WHEN multiple transcript languages are available, THE Media_Tool SHALL prefer English transcripts

### Requirement 3: Audio Transcription with Local Whisper

**User Story:** As an agent user, I want to transcribe audio files locally without API costs, so that I can process spoken content like lectures, recipes, or exam questions.

#### Acceptance Criteria

1. WHEN an audio file is provided, THE Media_Tool SHALL transcribe it using the local openai-whisper library (no API calls required)
2. THE Media_Tool SHALL support multiple Whisper model sizes (tiny, base, small, medium, large) for balancing speed versus accuracy
3. WHEN audio contains a pie recipe, calculus midterm, or other specific content, THE Media_Tool SHALL return accurate transcribed text
4. IF transcription fails due to file format, THEN THE Media_Tool SHALL return a descriptive error message listing supported formats (MP3, WAV, M4A, FLAC, OGG)
5. WHILE audio quality is poor, THE Media_Tool SHALL still attempt transcription with a confidence warning

### Requirement 4: Python Code Execution Sandbox

**User Story:** As an agent user, I want to execute Python code safely, so that I can perform computations like string manipulation, mathematical operations, and data filtering.

#### Acceptance Criteria

1. WHEN Python code is provided, THE Code_Interpreter_Tool SHALL validate it for dangerous operations before execution
2. WHEN code passes security validation, THE Code_Interpreter_Tool SHALL execute it in an isolated subprocess environment
3. WHEN code execution completes, THE Code_Interpreter_Tool SHALL return the standard output and any results
4. IF code execution times out, THEN THE Code_Interpreter_Tool SHALL terminate the process and return a timeout error
5. IF dangerous operations are detected (file system access, network calls, system commands), THEN THE Code_Interpreter_Tool SHALL block execution immediately and return a security error
6. WHEN string reversal, matrix operations, or botanical filtering tasks are requested, THE Code_Interpreter_Tool SHALL execute appropriate Python code and return results
7. IF code raises an exception, THEN THE Code_Interpreter_Tool SHALL return the exception message and traceback

### Requirement 5: Document and Spreadsheet Analysis

**User Story:** As an agent user, I want to analyze Excel files and documents, so that I can extract data from spreadsheets and answer questions about tabular data.

#### Acceptance Criteria

1. WHEN an .xlsx file is provided, THE Excel_Parser_Tool SHALL load it using pandas or openpyxl and return the data
2. WHEN a .csv file is provided, THE Excel_Parser_Tool SHALL parse it using pandas and return structured data
3. WHEN a query about LibreText exercises or fast-food chain sales data is received, THE Excel_Parser_Tool SHALL search within the loaded data and return matching results
4. IF a file fails to parse, THEN THE Excel_Parser_Tool SHALL return an error message for that file and continue processing any remaining files successfully
5. WHEN specific rows, columns, or cells are requested, THE Excel_Parser_Tool SHALL extract and return the targeted data
6. THE Excel_Parser_Tool SHALL format output as human-readable text, not raw data serialization

### Requirement 6: Vision and Multimodal Analysis with Local Ollama

**User Story:** As an agent user, I want to analyze images including chess boards using local tools, so that I can answer visual questions without commercial API costs.

#### Acceptance Criteria

1. WHEN an image is provided, THE Vision_Tool SHALL process it using local Ollama with the gemma4:e4b vision model
2. WHEN a chess board image is explicitly provided, THE Vision_Tool SHALL identify the position and use Stockfish for evaluation
3. WHEN a chess position is recognized from an explicitly provided chess board image, THE Vision_Tool SHALL return the FEN notation and suggested best moves
4. IF the Ollama service is unavailable, THEN THE Vision_Tool SHALL return a descriptive error message with instructions to start the Ollama service
5. IF the vision model is not found, THEN THE Vision_Tool SHALL return an error with instructions to pull the model (e.g., `ollama pull gemma4:e4b`)
6. THE Vision_Tool SHALL NOT require any API keys (OpenAI, Anthropic, Google) and shall run entirely locally

### Requirement 7: Tool Integration with LangGraph Agent

**User Story:** As a developer, I want all five tools integrated as dedicated nodes in the existing LangGraph agent, so that queries are automatically routed to appropriate tools.

#### Acceptance Criteria

1. WHEN a query is received by the LangGraph_Agent, THE Agent_Tool_Suite SHALL be available as tool options
2. WHEN the Query_Router classifies a query type, THE Agent_Tool_Suite SHALL route to the appropriate dedicated tool node
3. THE LangGraph_Agent SHALL use a hybrid classification approach: LLM-based classification with heuristic keyword fallback
4. WHEN the primary LLM is unavailable for classification, THE Query_Router SHALL use heuristic keyword matching to determine tool routing
5. WHEN multiple tools might apply, THE Agent_Tool_Suite SHALL analyze the tools and choose between sequential or parallel execution
6. IF a tool returns an error, THEN THE LangGraph_Agent SHALL either attempt alternative tools or inform the user of the failure with graceful degradation
7. WHEN tools_local.py is imported, THE Agent_Tool_Suite functions SHALL be available for use

### Requirement 8: Tool Response Formatting

**User Story:** As an agent user, I want consistent response formats from all tools, so that the agent can reliably parse and present results.

#### Acceptance Criteria

1. WHEN any tool completes successfully, THE tool SHALL return a string response containing the result
2. WHEN any tool encounters an error, THE tool SHALL return a string prefixed with "Error:" followed by the error description
3. WHEN a tool returns structured data, THE tool SHALL format it as human-readable text
4. WHEN evidence or sources are used, THE tool SHALL include citations or URLs in the response

### Requirement 9: Error Handling and Graceful Degradation

**User Story:** As a developer, I want robust error handling across all tools, so that the agent continues operating even when individual components fail.

#### Acceptance Criteria

1. WHEN any tool encounters an error, THE tool SHALL log the error with descriptive context
2. WHEN a tool fails, THE LangGraph_Agent SHALL attempt to retry the operation up to a configurable maximum (default: 2 retries)
3. IF retries are exhausted, THEN THE LangGraph_Agent SHALL attempt fallback tools if applicable
4. IF no fallback is available, THEN THE LangGraph_Agent SHALL return a user-friendly error message explaining what went wrong
5. WHEN the Ollama service is not running, THE Vision_Tool SHALL return instructions for starting the service rather than crashing
6. WHEN a network request times out, THE Web_Search_Tool SHALL attempt an alternative search method or return cached results if available

### Requirement 10: Dependency Management

**User Story:** As a developer, I want all required dependencies specified with free/open-source constraints, so that the agent tool suite can be reliably installed without commercial API dependencies.

#### Acceptance Criteria

1. THE requirements.txt file SHALL include all dependencies with pinned versions for reproducibility
2. THE dependencies SHALL include only free and open-source packages: youtube-transcript-api, openai-whisper (local), pandas, openpyxl, python-chess, beautifulsoup4, requests, and ddgs (DuckDuckGo search)
3. THE requirements.txt SHALL NOT include commercial API packages: openai (for API calls), anthropic, google-generativeai
4. WHEN optional dependencies are available, THE requirements.txt SHALL document which are required versus optional
5. THE documentation SHALL specify system dependencies: ffmpeg (for Whisper), Stockfish binary (for chess), and Ollama with gemma4:e4b model (for vision)

### Requirement 11: Windows Environment Compatibility

**User Story:** As a developer, I want the agent tool suite to run on Windows environments, so that users on Windows platforms can use all features without modification.

#### Acceptance Criteria

1. WHEN the Code_Interpreter_Tool executes code on Windows, THE subprocess calls SHALL use appropriate Windows-compatible command syntax
2. WHEN file paths are processed, THE tools SHALL handle Windows path separators and drive letters correctly
3. THE documentation SHALL provide Windows-specific installation instructions for system dependencies (ffmpeg, Stockfish, Ollama)
4. WHEN environment variables are accessed, THE tools SHALL read from Windows environment correctly via python-dotenv
5. THE default Ollama endpoint SHALL be http://localhost:11434 (the Windows default for Ollama)

### Requirement 12: No Commercial API Keys Required

**User Story:** As a developer, I want the agent tool suite to operate without any commercial API keys, so that users can run the system without ongoing costs or external dependencies.

#### Acceptance Criteria

1. THE Agent_Tool_Suite SHALL NOT require OPENAI_API_KEY for any functionality
2. THE Agent_Tool_Suite SHALL NOT require ANTHROPIC_API_KEY for any functionality
3. THE Agent_Tool_Suite SHALL NOT require GOOGLE_API_KEY for any functionality
4. IF API key environment variables are present, THE tools SHALL ignore them and use local alternatives instead
5. WHEN users attempt to configure commercial API keys, THE documentation SHALL clearly state that only free/open-source tools are supported
