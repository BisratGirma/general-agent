"""Media processing tools: YouTube transcripts and audio transcription."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from tools.common import _extract_file_path, _format_error, _normalize_query

SUPPORTED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")



def process_youtube_transcript(url: str, preferred_language: str = "en") -> str:
    """Extract a YouTube video transcript using youtube-transcript-api."""
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
        api = YouTubeTranscriptApi()
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        elif hasattr(api, "list"):
            transcript_list = api.list(video_id)
        elif hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            return _format_error("YouTube Transcript", "unsupported youtube-transcript-api version")

        if isinstance(transcript_list, list):
            segments = [item.get("text", "") for item in transcript_list if isinstance(item, dict)]
            return "\n".join(segments)[:6000]

        try:
            transcript = transcript_list.find_transcript([preferred_language])
        except Exception:
            transcript = (
                transcript_list.find_generated_transcript([preferred_language])
                if hasattr(transcript_list, "find_generated_transcript")
                else None
            )
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
    """Transcribe a local audio file using OpenAI Whisper."""
    import os

    input_text = _normalize_query(file_path)
    actual_path, _ = _extract_file_path(input_text, allowed_extensions=SUPPORTED_AUDIO_EXTENSIONS)
    file_path = actual_path or input_text

    if not file_path or not os.path.exists(file_path):
        return _format_error("Whisper", "file does not exist")


    suffix = Path(file_path).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(SUPPORTED_AUDIO_EXTENSIONS)
        return _format_error("Whisper", f"unsupported format '{suffix}'. Supported: {supported}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return _format_error("Whisper", "faster-whisper is not installed")

    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(file_path)
        text = " ".join([segment.text for segment in segments]).strip()
        return text or _format_error("Whisper", "no transcription text was produced")
    except Exception as exc:
        return _format_error("Whisper", str(exc))


def process_media(
    source_type: str,
    url: Optional[str] = None,
    file_path: Optional[str] = None,
    whisper_model: str = "base",
    preferred_language: str = "en",
) -> str:
    """Route media requests to the appropriate processing function.

    Args:
        source_type: ``"youtube"`` or ``"audio_file"``.
        url: URL for YouTube sources.
        file_path: Local file path for audio sources.
        whisper_model: Whisper model size (e.g. ``"base"``, ``"small"``).
        preferred_language: BCP-47 language code for transcripts.
    """
    source_type = (source_type or "").lower()
    if source_type == "youtube":
        return process_youtube_transcript(url or "", preferred_language=preferred_language)
    if source_type == "audio_file":
        return transcribe_audio_file(file_path or "", model_size=whisper_model)
    return _format_error("Media Tool", "unsupported source type")
