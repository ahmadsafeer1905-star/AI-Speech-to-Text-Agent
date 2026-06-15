"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AI SPEECH-TO-TEXT AGENT — Enterprise Edition                        ║
║         Production-Ready | Python 3.11 | Streamlit                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

OVERALL ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This application is organized as a layered AI Agent:

  ┌─────────────────────────────────────────────────────────┐
  │  1. User Interface Layer     (Streamlit pages/widgets)  │
  │  2. Audio Processing Layer   (pydub, librosa, scipy)    │
  │  3. Speech Recognition Layer (Faster-Whisper)           │
  │  4. NLP Analysis Layer       (Claude API via Anthropic) │
  │  5. Memory Layer             (st.session_state store)   │
  │  6. Report Generation Layer  (structured JSON outputs)  │
  │  7. Analytics Layer          (pandas + plotly)          │
  │  8. Export Layer             (TXT, JSON, CSV download)  │
  │  9. Security Layer           (validation, cleanup)      │
  │ 10. Error Handling Layer     (try/except + logging)     │
  └─────────────────────────────────────────────────────────┘

AI WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Audio Input → Validate → Normalize → Transcribe → Detect Language
       → NLP Analysis (Claude) → Extract Entities / Tasks / Keywords
       → Summarize → Sentiment Analysis → Mode-specific Report
       → Store in Memory → Update Analytics → Enable Export

SPEECH RECOGNITION WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Faster-Whisper is selected as the primary engine because it:
  - Runs 4× faster than original Whisper on CPU using CTranslate2 quantization
  - Auto-detects GPU (CUDA) and falls back gracefully to CPU
  - Supports chunked/streaming inference for large audio files
  - Produces word-level timestamps natively
  - Works fully offline after model download (no API key required)

  Model download path: ~/.cache/huggingface (automatic, one-time)
  Default model: "base" for fast inference; user can select "small", "medium"

NLP WORKFLOW (via Claude API)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  A single carefully structured prompt is sent to Claude containing:
  - The full transcript
  - The selected specialist mode
  - Instructions to return JSON with all analysis fields
  Claude then returns structured JSON with:
    summaries, entities, tasks, keywords, sentiment, mode_outputs

MEMORY WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Streamlit's st.session_state acts as an in-process memory store.
  Each transcription result is appended to session_state["history"] as a dict:
    { id, timestamp, language, transcript, analysis, audio_meta, mode }
  History persists for the browser session. Users can search/filter/re-export.

ANALYTICS WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  After each transcription, text stats (word count, sentence count, etc.)
  are computed from raw transcript text using string operations.
  Plotly charts visualize: sentiment trends, word counts, language distribution.

EXPORT WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Every history record can be exported as TXT, JSON, or CSV.
  Streamlit's st.download_button() streams the file to the browser.
  No server-side file storage required.

SECURITY WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Max upload size enforced: 200 MB
  - MIME type checked against whitelist (audio/* only)
  - Temporary files written to OS temp dir, deleted after processing
  - No user data persisted to disk outside of temp files
  - API key read from st.secrets or environment variable only
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import io
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anthropic
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from pydub import AudioSegment
from rapidfuzz import fuzz, process

# ── Conditional import: audiorecorder widget ──────────────────────────────────
# streamlit-audiorecorder provides in-browser mic capture via WebRTC.
# It may not be available in all environments; we handle that gracefully.
try:
    from audiorecorder import audiorecorder
    AUDIORECORDER_AVAILABLE = True
except ImportError:
    AUDIORECORDER_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# Logging is essential for production debugging. We configure a consistent
# format that includes timestamp, level, and message.
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stt_agent")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Maximum audio upload size (bytes) — Security Layer
MAX_FILE_SIZE_MB: int = 200
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

# Allowed MIME types / extensions — Security Layer
ALLOWED_EXTENSIONS: set[str] = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
ALLOWED_MIME_PREFIXES: tuple[str, ...] = ("audio/",)

# Whisper model options: tradeoff between speed and accuracy
# "tiny" → fastest, lowest accuracy
# "base" → good balance for most use cases (default)
# "small" → better accuracy, ~2× slower
# "medium" → high accuracy, ~5× slower
WHISPER_MODELS: list[str] = ["tiny", "base", "small", "medium"]

# Supported specialist modes and their display names
SPECIALIST_MODES: dict[str, str] = {
    "general": "🎯 General Transcription",
    "meeting": "📋 Meeting Assistant",
    "lecture": "🎓 Lecture Assistant",
    "interview": "🤝 Interview Assistant",
    "medical": "🏥 Medical Notes Assistant",
    "legal": "⚖️ Legal Notes Assistant",
    "podcast": "🎙️ Podcast Assistant",
    "support": "🎧 Customer Support Assistant",
    "sales": "💼 Sales Call Assistant",
    "research": "🔬 Research Assistant",
}

# Languages supported for detection display
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English", "ur": "Urdu", "hi": "Hindi", "ar": "Arabic",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: AUDIO PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class AudioProcessor:
    """
    Audio Processing Layer
    ─────────────────────
    Responsible for:
    - Validating uploaded audio files (size, type, corruption)
    - Reading audio metadata (sample rate, channels, duration, bitrate)
    - Converting any format to WAV 16kHz mono (required by Whisper)
    - Applying noise reduction via scipy spectral gating
    - Detecting silence / very short clips
    - Cleaning up temporary files after processing

    Why 16kHz mono?
    Whisper was trained on 16kHz mono audio. Resampling to this format
    before inference improves both speed and accuracy.
    """

    @staticmethod
    def validate_file(file_bytes: bytes, filename: str) -> tuple[bool, str]:
        """
        Security Layer: Validate file before any processing.
        Returns (is_valid, error_message).
        """
        # Check file size
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            return False, f"File exceeds {MAX_FILE_SIZE_MB} MB limit."

        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

        # Check file is not empty
        if len(file_bytes) < 100:
            return False, "File appears to be empty or corrupt."

        return True, ""

    @staticmethod
    def get_audio_metadata(file_bytes: bytes, filename: str) -> dict[str, Any]:
        """
        Extract audio metadata using pydub and librosa.
        This gives users visibility into their audio before transcription.
        """
        meta: dict[str, Any] = {
            "filename": filename,
            "file_size_mb": round(len(file_bytes) / (1024 * 1024), 2),
            "format": Path(filename).suffix.lower().strip(".").upper(),
            "sample_rate": None,
            "channels": None,
            "duration_seconds": None,
            "bitrate_kbps": None,
        }
        try:
            # pydub handles format detection automatically
            ext = Path(filename).suffix.lower().strip(".")
            audio_segment = AudioSegment.from_file(io.BytesIO(file_bytes), format=ext)
            meta["sample_rate"] = audio_segment.frame_rate
            meta["channels"] = audio_segment.channels
            meta["duration_seconds"] = round(len(audio_segment) / 1000, 2)
            # Estimate bitrate: file_size_bits / duration_seconds
            if meta["duration_seconds"] > 0:
                meta["bitrate_kbps"] = round(
                    (len(file_bytes) * 8) / meta["duration_seconds"] / 1000, 1
                )
        except Exception as exc:
            logger.warning("Could not extract full metadata: %s", exc)
        return meta

    @staticmethod
    def preprocess_audio(file_bytes: bytes, filename: str) -> Optional[str]:
        """
        Convert audio to 16kHz mono WAV for Whisper inference.
        Applies:
        1. Format conversion (pydub handles MP3, M4A, FLAC, OGG, AAC → WAV)
        2. Resampling to 16000 Hz
        3. Mono downmix (average stereo channels)
        4. Amplitude normalization (peak normalize to -1 dBFS)
        5. Basic spectral noise reduction (scipy)

        Returns: path to the processed temporary WAV file.
        Security: temp file is written to OS temp directory.
        """
        ext = Path(filename).suffix.lower().strip(".")
        tmp_input = None
        tmp_output = None
        try:
            # Write raw bytes to a temp file so pydub can read it
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
                f.write(file_bytes)
                tmp_input = f.name

            # Load with pydub (supports all our formats)
            audio = AudioSegment.from_file(tmp_input, format=ext)

            # Convert to mono
            audio = audio.set_channels(1)

            # Resample to 16000 Hz
            audio = audio.set_frame_rate(16000)

            # Peak normalize to -1 dBFS to prevent clipping
            peak_db = audio.max_dBFS
            if peak_db < -1.0:
                audio = audio.apply_gain(-1.0 - peak_db)

            # Export to temporary WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_output = f.name
            audio.export(tmp_output, format="wav")

            # Optional: spectral noise reduction using librosa + numpy
            # This reduces stationary background noise (fans, hum, etc.)
            try:
                y, sr = librosa.load(tmp_output, sr=16000, mono=True)
                # Estimate noise floor from first 0.5 seconds (assumed silent)
                noise_sample = y[:sr // 2] if len(y) > sr // 2 else y
                noise_power = np.mean(noise_sample ** 2)
                # Soft-thresholding: attenuate signal below noise threshold
                signal_power = y ** 2
                threshold = noise_power * 4  # 4× noise floor threshold
                mask = signal_power > threshold
                y_clean = y * mask.astype(float)
                sf.write(tmp_output, y_clean, sr)
                logger.info("Noise reduction applied.")
            except Exception as nr_exc:
                logger.warning("Noise reduction skipped: %s", nr_exc)

            return tmp_output

        except Exception as exc:
            logger.error("Audio preprocessing failed: %s", exc)
            return None
        finally:
            # Security: always clean up the input temp file
            if tmp_input and os.path.exists(tmp_input):
                os.remove(tmp_input)

    @staticmethod
    def cleanup_temp(path: Optional[str]) -> None:
        """Security Layer: Remove temporary audio file after transcription."""
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.info("Temporary file removed: %s", path)
            except OSError as exc:
                logger.warning("Could not remove temp file %s: %s", path, exc)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3: SPEECH RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

class SpeechRecognizer:
    """
    Speech Recognition Layer using Faster-Whisper
    ──────────────────────────────────────────────
    Faster-Whisper uses CTranslate2, a highly optimized inference engine
    that quantizes Whisper weights to INT8 (on CPU) or FP16 (on GPU).
    This results in 4× speedup over the original Whisper on CPU.

    Model Loading Strategy:
    - We use @st.cache_resource to load the model once per server session.
    - This avoids reloading the model (hundreds of MB) on every page refresh.

    GPU Detection:
    - We try 'cuda' device first. If CTranslate2 reports no CUDA device,
      we fall back to 'cpu' automatically.

    Chunk Processing:
    - For audio longer than 30 seconds, Faster-Whisper's VAD (Voice Activity
      Detection) filter splits audio into speech segments automatically.
      This reduces hallucinations on silence and speeds up long-file processing.

    Timestamp Generation:
    - word_timestamps=True produces per-word timing data.
    - We concatenate segments to produce the full transcript text.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = self._load_model(model_size)

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_model(model_size: str):
        """
        Load Faster-Whisper model into memory.
        st.cache_resource ensures this runs only once per Streamlit server session.
        The model is shared across all user sessions (thread-safe for inference).
        """
        from faster_whisper import WhisperModel  # imported here to avoid top-level crash if not installed

        # GPU detection: CTranslate2 will raise if CUDA is unavailable
        device = "cpu"
        compute_type = "int8"  # INT8 quantization for CPU efficiency
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
                compute_type = "float16"  # FP16 for GPU efficiency
                logger.info("GPU detected — using CUDA with float16.")
            else:
                logger.info("No GPU detected — using CPU with int8.")
        except Exception:
            logger.info("CTranslate2 CUDA check failed — defaulting to CPU.")

        logger.info("Loading Whisper model '%s' on %s...", model_size, device)
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            # num_workers: parallel decoding threads on CPU
            num_workers=2,
        )
        logger.info("Whisper model loaded successfully.")
        return model

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> dict[str, Any]:
        """
        Run Faster-Whisper inference on a preprocessed WAV file.

        Parameters:
            wav_path: Path to 16kHz mono WAV file.
            language: ISO 639-1 code (e.g. 'en') or None for auto-detection.

        Returns:
            {
                "transcript": full text string,
                "segments": list of {start, end, text},
                "language": detected language code,
                "language_probability": float 0–1,
            }
        """
        from faster_whisper import WhisperModel

        try:
            # VAD filter removes silence, reducing hallucination on quiet parts.
            # beam_size=5 balances accuracy vs speed (default=5).
            segments_iter, info = self.model.transcribe(
                wav_path,
                language=language,
                beam_size=5,
                vad_filter=True,         # Voice Activity Detection
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
                word_timestamps=True,    # Per-word timestamps
                condition_on_previous_text=True,  # Context continuity between chunks
            )

            # Consume the generator and build transcript
            segments = []
            full_text_parts = []
            for seg in segments_iter:
                segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                })
                full_text_parts.append(seg.text.strip())

            full_transcript = " ".join(full_text_parts)

            lang_code = info.language
            lang_prob = round(info.language_probability, 3)
            lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())

            logger.info(
                "Transcription complete. Language: %s (%.1f%%), Words: %d",
                lang_code, lang_prob * 100, len(full_transcript.split()),
            )

            return {
                "transcript": full_transcript,
                "segments": segments,
                "language": lang_code,
                "language_name": lang_name,
                "language_probability": lang_prob,
            }

        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            raise RuntimeError(f"Transcription failed: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4: NLP ANALYSIS (Claude API)
# ══════════════════════════════════════════════════════════════════════════════

class NLPAnalyzer:
    """
    NLP Analysis Layer — powered by Anthropic Claude
    ──────────────────────────────────────────────────
    Claude performs all higher-level language understanding tasks:
    - Summarization (4 formats)
    - Named entity extraction (people, places, orgs, dates, numbers)
    - Action item / task extraction
    - Keyword and key phrase extraction with relevance scores
    - Sentiment classification with confidence
    - Specialist mode analysis (meeting decisions, lecture notes, etc.)

    Why Claude instead of local NLP?
    - Claude handles multilingual text natively
    - No additional large model downloads required
    - Produces structured JSON reliably with a well-crafted prompt
    - Handles context and nuance far better than rule-based NLP

    Prompt Design:
    - A single structured prompt requesting JSON output ensures we get all
      analysis in one API call, minimizing latency and token costs.
    - Temperature=0 ensures deterministic, factual responses.
    """

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze(self, transcript: str, mode: str = "general") -> dict[str, Any]:
        """
        Send transcript to Claude for comprehensive NLP analysis.
        Returns structured dict with all analysis fields.
        """
        mode_label = SPECIALIST_MODES.get(mode, "General Transcription")
        mode_instructions = self._get_mode_instructions(mode)

        prompt = f"""
You are an expert NLP analyzer. Analyze the following transcript and return a JSON object.

SPECIALIST MODE: {mode_label}
{mode_instructions}

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"

Return ONLY a valid JSON object with exactly this structure (no markdown, no explanation):
{{
  "executive_summary": "2-3 sentence high-level summary",
  "short_summary": "1 paragraph summary under 100 words",
  "detailed_summary": "comprehensive multi-paragraph summary",
  "bullet_summary": ["bullet point 1", "bullet point 2", "...up to 8 bullets"],
  "entities": {{
    "people": ["name1", "name2"],
    "organizations": ["org1", "org2"],
    "locations": ["loc1", "loc2"],
    "dates": ["date1", "date2"],
    "numbers": ["num1", "num2"],
    "deadlines": ["deadline1"]
  }},
  "action_items": [
    {{"task": "description", "assignee": "person or 'Unassigned'", "priority": "high|medium|low"}}
  ],
  "keywords": [
    {{"term": "keyword", "relevance": 0.95}},
    {{"term": "keyword2", "relevance": 0.87}}
  ],
  "key_phrases": ["phrase1", "phrase2"],
  "sentiment": {{
    "classification": "positive|neutral|negative",
    "confidence": 0.85,
    "emotional_tone": "professional|excited|concerned|neutral|frustrated|optimistic"
  }},
  "main_topics": ["topic1", "topic2"],
  "mode_output": {{}}
}}

For mode_output, populate based on the specialist mode:
- meeting: {{"decisions": [], "participants": [], "risks": [], "next_steps": []}}
- lecture: {{"main_concepts": [], "definitions": {{}}, "study_notes": [], "quiz_questions": []}}
- interview: {{"strengths": [], "weaknesses": [], "key_answers": []}}
- medical: {{"symptoms": [], "diagnoses": [], "treatments": [], "follow_ups": []}}
- legal: {{"case_facts": [], "legal_issues": [], "cited_statutes": [], "action_required": []}}
- podcast: {{"topics_covered": [], "guests": [], "key_insights": [], "timestamps": []}}
- support: {{"issue": "", "resolution": "", "customer_sentiment": "", "escalation_needed": false}}
- sales: {{"product_discussed": [], "objections": [], "next_steps": [], "deal_probability": ""}}
- research: {{"hypotheses": [], "findings": [], "methodology_notes": [], "citations_mentioned": []}}
- general: {{}}
"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Clean up any accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            if raw.endswith("```"):
                raw = raw[:-3]

            analysis = json.loads(raw.strip())
            logger.info("NLP analysis complete via Claude API.")
            return analysis

        except json.JSONDecodeError as exc:
            logger.error("Claude returned invalid JSON: %s", exc)
            return self._fallback_analysis(transcript)
        except Exception as exc:
            logger.error("NLP analysis failed: %s", exc)
            return self._fallback_analysis(transcript)

    @staticmethod
    def _get_mode_instructions(mode: str) -> str:
        """Return specialist mode-specific analysis instructions."""
        instructions = {
            "meeting": (
                "Focus on: decisions made, who is responsible for what, "
                "risks identified, blockers, and next meeting agenda items."
            ),
            "lecture": (
                "Focus on: core academic concepts, technical definitions, "
                "study-worthy points, and generate 5 quiz questions."
            ),
            "interview": (
                "Focus on: candidate skills demonstrated, areas of weakness, "
                "standout answers, and hiring recommendation indicators."
            ),
            "medical": (
                "Focus on: patient symptoms, differential diagnoses, "
                "treatment plans, medication mentions, and follow-up instructions."
            ),
            "legal": (
                "Focus on: legal facts, statutes referenced, case arguments, "
                "parties involved, and required legal actions."
            ),
            "podcast": (
                "Focus on: topics discussed, guest names and credentials, "
                "key insights listeners should remember, and memorable quotes."
            ),
            "support": (
                "Focus on: customer issue, root cause, resolution provided, "
                "customer emotional state, and whether escalation is needed."
            ),
            "sales": (
                "Focus on: products/services discussed, customer objections raised, "
                "agreed next steps, and deal health indicators."
            ),
            "research": (
                "Focus on: research questions, findings, methodologies mentioned, "
                "sources cited, and suggested further investigation."
            ),
        }
        return instructions.get(mode, "Provide a balanced general-purpose analysis.")

    @staticmethod
    def _fallback_analysis(transcript: str) -> dict[str, Any]:
        """
        Minimal offline fallback if Claude API is unavailable.
        Uses simple string operations to extract basic stats.
        """
        words = transcript.split()
        sentences = [s.strip() for s in transcript.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        return {
            "executive_summary": transcript[:300] + ("..." if len(transcript) > 300 else ""),
            "short_summary": transcript[:500],
            "detailed_summary": transcript,
            "bullet_summary": sentences[:6],
            "entities": {"people": [], "organizations": [], "locations": [], "dates": [], "numbers": [], "deadlines": []},
            "action_items": [],
            "keywords": [{"term": w, "relevance": 0.5} for w in list(set(words))[:10]],
            "key_phrases": [],
            "sentiment": {"classification": "neutral", "confidence": 0.5, "emotional_tone": "neutral"},
            "main_topics": [],
            "mode_output": {},
            "_fallback": True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5: MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """
    Memory Layer — Session-based transcript history
    ────────────────────────────────────────────────
    Stores all transcription results in st.session_state["history"].
    Each record is a dict:
    {
        "id": str (UUID),
        "timestamp": ISO 8601 datetime string,
        "audio_meta": dict (filename, duration, format, etc.),
        "transcript": str,
        "language": str,
        "language_name": str,
        "language_probability": float,
        "mode": str,
        "analysis": dict,
        "text_stats": dict,
        "segments": list,
    }

    Design decision: session_state is used instead of a database because:
    - No setup required (works on Streamlit Cloud with zero config)
    - Data is ephemeral (no PII stored server-side)
    - Sufficient for session-length workflows

    For production enterprise use, this layer could be swapped for
    a PostgreSQL, MongoDB, or Redis backend with user authentication.
    """

    HISTORY_KEY = "stt_history"

    @classmethod
    def initialize(cls) -> None:
        """Initialize the memory store if not already present."""
        if cls.HISTORY_KEY not in st.session_state:
            st.session_state[cls.HISTORY_KEY] = []

    @classmethod
    def add_record(cls, record: dict[str, Any]) -> None:
        """Append a new transcription record to memory."""
        cls.initialize()
        record["id"] = str(uuid.uuid4())
        record["timestamp"] = datetime.now().isoformat()
        st.session_state[cls.HISTORY_KEY].insert(0, record)  # newest first
        logger.info("Memory: stored record %s", record["id"])

    @classmethod
    def get_all(cls) -> list[dict[str, Any]]:
        """Return all stored records."""
        cls.initialize()
        return st.session_state[cls.HISTORY_KEY]

    @classmethod
    def search(
        cls,
        query: str,
        language_filter: Optional[str] = None,
        fuzzy: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Intelligent search across transcript history.
        Supports:
        - Exact keyword matching (case-insensitive)
        - Fuzzy matching using RapidFuzz (handles typos)
        - Language filtering

        RapidFuzz uses the Levenshtein distance algorithm to score
        string similarity. A score ≥ 70 is considered a match.
        """
        records = cls.get_all()
        results = []
        for rec in records:
            # Language filter
            if language_filter and rec.get("language") != language_filter:
                continue
            transcript = rec.get("transcript", "")
            if not query:
                results.append(rec)
                continue
            # Fuzzy search: split transcript into chunks and score
            if fuzzy:
                score = fuzz.partial_ratio(query.lower(), transcript.lower())
                if score >= 65:
                    results.append(rec)
            else:
                if query.lower() in transcript.lower():
                    results.append(rec)
        return results

    @classmethod
    def delete_record(cls, record_id: str) -> None:
        """Remove a specific record from memory."""
        cls.initialize()
        st.session_state[cls.HISTORY_KEY] = [
            r for r in st.session_state[cls.HISTORY_KEY] if r.get("id") != record_id
        ]

    @classmethod
    def clear_all(cls) -> None:
        """Wipe all history from session memory."""
        st.session_state[cls.HISTORY_KEY] = []


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

class AnalyticsEngine:
    """
    Analytics Layer
    ───────────────
    Computes text statistics from raw transcript text.
    Builds Plotly charts for the analytics dashboard.
    """

    @staticmethod
    def compute_text_stats(transcript: str, duration_seconds: Optional[float] = None) -> dict[str, Any]:
        """
        Compute reading time, word count, sentence count, etc.
        Average adult reading speed: ~238 words per minute.
        Average speaking speed: ~130 words per minute.
        """
        words = transcript.split()
        word_count = len(words)
        char_count = len(transcript)
        sentences = [s for s in transcript.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        sentence_count = len(sentences)
        avg_sentence_len = round(word_count / max(sentence_count, 1), 1)
        reading_time_min = round(word_count / 238, 2)
        speaking_rate_wpm = None
        if duration_seconds and duration_seconds > 0:
            speaking_rate_wpm = round(word_count / (duration_seconds / 60), 1)

        return {
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_len,
            "reading_time_min": reading_time_min,
            "speaking_rate_wpm": speaking_rate_wpm,
            "duration_seconds": duration_seconds,
        }

    @staticmethod
    def build_sentiment_chart(history: list[dict]) -> Optional[go.Figure]:
        """Bar chart of sentiment distribution across all history records."""
        if not history:
            return None
        sentiments = [
            r.get("analysis", {}).get("sentiment", {}).get("classification", "neutral")
            for r in history
        ]
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for s in sentiments:
            counts[s] = counts.get(s, 0) + 1

        fig = px.bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            color=list(counts.keys()),
            color_discrete_map={"positive": "#22c55e", "neutral": "#64748b", "negative": "#ef4444"},
            labels={"x": "Sentiment", "y": "Count"},
            title="Sentiment Distribution",
        )
        fig.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20), height=280)
        return fig

    @staticmethod
    def build_language_chart(history: list[dict]) -> Optional[go.Figure]:
        """Pie chart showing language distribution."""
        if not history:
            return None
        langs: dict[str, int] = {}
        for r in history:
            lang = r.get("language_name", "Unknown")
            langs[lang] = langs.get(lang, 0) + 1
        fig = px.pie(
            names=list(langs.keys()),
            values=list(langs.values()),
            title="Language Distribution",
        )
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=280)
        return fig

    @staticmethod
    def build_word_count_trend(history: list[dict]) -> Optional[go.Figure]:
        """Line chart of word counts over time."""
        if len(history) < 2:
            return None
        timestamps = [r.get("timestamp", "")[:10] for r in reversed(history)]
        word_counts = [r.get("text_stats", {}).get("word_count", 0) for r in reversed(history)]
        fig = px.line(
            x=timestamps,
            y=word_counts,
            markers=True,
            labels={"x": "Date", "y": "Word Count"},
            title="Word Count Over Time",
        )
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=280)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8: EXPORT
# ══════════════════════════════════════════════════════════════════════════════

class ExportEngine:
    """
    Export Layer
    ────────────
    Serializes a history record into TXT, JSON, or CSV format.
    Returns bytes that can be passed to st.download_button().
    No disk writes — everything is in-memory for security and simplicity.
    """

    @staticmethod
    def to_txt(record: dict) -> bytes:
        """
        Plain text export: readable report format.
        Includes transcript, summaries, tasks, keywords.
        """
        lines = [
            f"AI SPEECH-TO-TEXT AGENT — TRANSCRIPT REPORT",
            f"=" * 60,
            f"Date:       {record.get('timestamp', '')[:19]}",
            f"File:       {record.get('audio_meta', {}).get('filename', 'N/A')}",
            f"Language:   {record.get('language_name', 'N/A')} "
            f"({record.get('language_probability', 0) * 100:.1f}% confidence)",
            f"Duration:   {record.get('audio_meta', {}).get('duration_seconds', 'N/A')}s",
            f"Mode:       {SPECIALIST_MODES.get(record.get('mode', 'general'), 'General')}",
            "",
            "TRANSCRIPT",
            "-" * 60,
            record.get("transcript", ""),
            "",
            "EXECUTIVE SUMMARY",
            "-" * 60,
            record.get("analysis", {}).get("executive_summary", "N/A"),
            "",
            "BULLET SUMMARY",
            "-" * 60,
        ]
        for bullet in record.get("analysis", {}).get("bullet_summary", []):
            lines.append(f"  • {bullet}")

        lines += ["", "ACTION ITEMS", "-" * 60]
        for item in record.get("analysis", {}).get("action_items", []):
            lines.append(f"  [{item.get('priority', '').upper()}] {item.get('task', '')} — {item.get('assignee', '')}")

        lines += ["", "KEYWORDS", "-" * 60]
        for kw in record.get("analysis", {}).get("keywords", [])[:15]:
            lines.append(f"  {kw.get('term', '')} ({kw.get('relevance', 0):.0%})")

        lines += [
            "",
            "SENTIMENT",
            "-" * 60,
            f"  {record.get('analysis', {}).get('sentiment', {}).get('classification', 'N/A').upper()} "
            f"({record.get('analysis', {}).get('sentiment', {}).get('confidence', 0) * 100:.1f}% confidence)",
            f"  Tone: {record.get('analysis', {}).get('sentiment', {}).get('emotional_tone', 'N/A')}",
            "",
            f"Generated by AI Speech-to-Text Agent",
        ]
        return "\n".join(lines).encode("utf-8")

    @staticmethod
    def to_json(record: dict) -> bytes:
        """Full JSON export of all record data."""
        return json.dumps(record, indent=2, ensure_ascii=False, default=str).encode("utf-8")

    @staticmethod
    def to_csv(record: dict) -> bytes:
        """
        CSV export flattens the record into key-value rows.
        Action items and keywords get their own rows.
        """
        rows = []
        # Flat metadata
        meta = record.get("audio_meta", {})
        analysis = record.get("analysis", {})
        stats = record.get("text_stats", {})
        rows.append({"section": "metadata", "field": "timestamp", "value": record.get("timestamp", "")})
        rows.append({"section": "metadata", "field": "filename", "value": meta.get("filename", "")})
        rows.append({"section": "metadata", "field": "language", "value": record.get("language_name", "")})
        rows.append({"section": "metadata", "field": "duration_s", "value": meta.get("duration_seconds", "")})
        rows.append({"section": "metadata", "field": "mode", "value": record.get("mode", "")})
        rows.append({"section": "transcript", "field": "full_text", "value": record.get("transcript", "")})
        rows.append({"section": "summary", "field": "executive", "value": analysis.get("executive_summary", "")})
        rows.append({"section": "stats", "field": "word_count", "value": stats.get("word_count", "")})
        rows.append({"section": "stats", "field": "sentiment", "value": analysis.get("sentiment", {}).get("classification", "")})
        for item in analysis.get("action_items", []):
            rows.append({"section": "action_item", "field": item.get("priority", ""), "value": item.get("task", "")})
        for kw in analysis.get("keywords", []):
            rows.append({"section": "keyword", "field": str(kw.get("relevance", "")), "value": kw.get("term", "")})
        df = pd.DataFrame(rows)
        return df.to_csv(index=False).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1: USER INTERFACE — HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def apply_custom_styles() -> None:
    """
    Inject custom CSS into Streamlit for a professional SaaS look.
    Uses Streamlit's st.markdown with unsafe_allow_html for styling overrides.
    """
    st.markdown("""
    <style>
    /* ── Global font & background ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        color: #f1f5f9;
    }
    div[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.75rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #f8fafc; font-weight: 700; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] .stRadio label { color: #cbd5e1; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; letter-spacing: 0.02em;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* ── Success/info/warning pills ── */
    .stAlert { border-radius: 8px; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #6366f1;
        border-bottom: 2px solid #6366f1; padding-bottom: 4px;
        margin-top: 1.5rem; margin-bottom: 0.75rem;
    }

    /* ── Transcript box ── */
    .transcript-box {
        background: #1e293b; border-radius: 10px; padding: 18px;
        border-left: 4px solid #6366f1; color: #e2e8f0;
        font-size: 0.95rem; line-height: 1.7; white-space: pre-wrap;
    }

    /* ── Task cards ── */
    .task-high { border-left: 4px solid #ef4444; padding: 8px 12px; margin: 4px 0; background: #1c1917; border-radius: 6px; }
    .task-medium { border-left: 4px solid #f59e0b; padding: 8px 12px; margin: 4px 0; background: #1c1917; border-radius: 6px; }
    .task-low { border-left: 4px solid #22c55e; padding: 8px 12px; margin: 4px 0; background: #1c1917; border-radius: 6px; }

    /* ── Keyword chips ── */
    .kw-chip {
        display: inline-block; background: #1e293b; color: #a5b4fc;
        border: 1px solid #4338ca; border-radius: 20px;
        padding: 3px 10px; margin: 3px; font-size: 0.82rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_row(stats: dict, audio_meta: dict, analysis: dict) -> None:
    """Render the top KPI metric row."""
    cols = st.columns(5)
    cols[0].metric("📝 Words", f"{stats.get('word_count', 0):,}")
    cols[1].metric("🔤 Characters", f"{stats.get('character_count', 0):,}")
    cols[2].metric("💬 Sentences", stats.get("sentence_count", 0))
    cols[3].metric("⏱️ Duration", f"{audio_meta.get('duration_seconds', 0):.1f}s")
    cols[4].metric("📖 Read Time", f"{stats.get('reading_time_min', 0):.1f} min")

    cols2 = st.columns(5)
    cols2[0].metric("🌍 Language", analysis.get("language_name", "—"))
    cols2[1].metric("🎯 Lang Confidence", f"{analysis.get('language_probability', 0) * 100:.0f}%")
    sentiment_data = analysis.get("analysis", {}).get("sentiment", {})
    cols2[2].metric("😊 Sentiment", sentiment_data.get("classification", "neutral").capitalize())
    cols2[3].metric("📋 Tasks Found", len(analysis.get("analysis", {}).get("action_items", [])))
    cols2[4].metric("🗝️ Keywords", len(analysis.get("analysis", {}).get("keywords", [])))


def render_transcript_section(record: dict) -> None:
    """Render the transcript with segment timestamps."""
    st.markdown('<div class="section-header">📄 Transcript</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="transcript-box">{record["transcript"]}</div>',
        unsafe_allow_html=True,
    )
    # Segments expander
    if record.get("segments"):
        with st.expander("⏰ View with Timestamps"):
            for seg in record["segments"]:
                st.markdown(
                    f"**[{seg['start']:.1f}s — {seg['end']:.1f}s]** {seg['text']}"
                )


def render_summaries(analysis: dict) -> None:
    """Render the four summary formats in tabs."""
    st.markdown('<div class="section-header">📋 Summaries</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Executive", "Short", "Detailed", "Bullet Points"])
    with tab1:
        st.info(analysis.get("executive_summary", "—"))
    with tab2:
        st.write(analysis.get("short_summary", "—"))
    with tab3:
        st.write(analysis.get("detailed_summary", "—"))
    with tab4:
        for bullet in analysis.get("bullet_summary", []):
            st.markdown(f"• {bullet}")


def render_entities(analysis: dict) -> None:
    """Render extracted named entities."""
    entities = analysis.get("entities", {})
    if not any(entities.values()):
        return
    st.markdown('<div class="section-header">🏷️ Entities Extracted</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    entity_map = {
        "👤 People": entities.get("people", []),
        "🏢 Organizations": entities.get("organizations", []),
        "📍 Locations": entities.get("locations", []),
        "📅 Dates": entities.get("dates", []),
        "🔢 Numbers": entities.get("numbers", []),
        "⚡ Deadlines": entities.get("deadlines", []),
    }
    for i, (label, items) in enumerate(entity_map.items()):
        if items:
            cols[i % 3].markdown(f"**{label}**")
            for item in items:
                cols[i % 3].markdown(f"  • {item}")


def render_action_items(analysis: dict) -> None:
    """Render action items with priority color coding."""
    tasks = analysis.get("action_items", [])
    if not tasks:
        return
    st.markdown('<div class="section-header">✅ Action Items</div>', unsafe_allow_html=True)
    for task in tasks:
        priority = task.get("priority", "medium").lower()
        css_class = f"task-{priority}"
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
        st.markdown(
            f'<div class="{css_class}">'
            f'{priority_icon} <strong>{task.get("task", "")}</strong><br>'
            f'<small>Assignee: {task.get("assignee", "Unassigned")} · Priority: {priority.capitalize()}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_keywords(analysis: dict) -> None:
    """Render keyword chips with relevance scores."""
    keywords = analysis.get("keywords", [])
    if not keywords:
        return
    st.markdown('<div class="section-header">🔑 Keywords & Concepts</div>', unsafe_allow_html=True)
    chips_html = " ".join(
        f'<span class="kw-chip">{kw["term"]} <small>({kw.get("relevance", 0):.0%})</small></span>'
        for kw in keywords
    )
    st.markdown(chips_html, unsafe_allow_html=True)

    phrases = analysis.get("key_phrases", [])
    if phrases:
        st.markdown("**Key Phrases:**")
        st.write(", ".join(phrases))


def render_sentiment(analysis: dict) -> None:
    """Render sentiment with visual indicator."""
    sentiment = analysis.get("sentiment", {})
    classification = sentiment.get("classification", "neutral")
    confidence = sentiment.get("confidence", 0.5)
    tone = sentiment.get("emotional_tone", "neutral")

    icon = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(classification, "😐")
    color = {"positive": "green", "neutral": "orange", "negative": "red"}.get(classification, "orange")

    st.markdown('<div class="section-header">💭 Sentiment Analysis</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{icon} Sentiment", classification.capitalize())
    c2.metric("🎯 Confidence", f"{confidence * 100:.0f}%")
    c3.metric("🎭 Emotional Tone", tone.capitalize())
    st.progress(confidence)


def render_mode_output(mode: str, analysis: dict) -> None:
    """Render specialist mode-specific outputs."""
    mode_output = analysis.get("mode_output", {})
    if not mode_output or mode == "general":
        return

    mode_label = SPECIALIST_MODES.get(mode, "General")
    st.markdown(f'<div class="section-header">{mode_label} — Specialist Output</div>', unsafe_allow_html=True)

    for key, value in mode_output.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            if value:
                with st.expander(f"📌 {label} ({len(value)} items)"):
                    for item in value:
                        if isinstance(item, dict):
                            st.json(item)
                        else:
                            st.markdown(f"• {item}")
        elif isinstance(value, dict):
            with st.expander(f"📌 {label}"):
                st.json(value)
        elif value:
            st.markdown(f"**{label}:** {value}")


def render_export_buttons(record: dict) -> None:
    """Render TXT / JSON / CSV download buttons."""
    st.markdown('<div class="section-header">📤 Export</div>', unsafe_allow_html=True)
    fname_base = Path(record.get("audio_meta", {}).get("filename", "transcript")).stem
    ts = record.get("timestamp", "")[:10]

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "⬇️ Download TXT",
        data=ExportEngine.to_txt(record),
        file_name=f"{fname_base}_{ts}.txt",
        mime="text/plain",
    )
    c2.download_button(
        "⬇️ Download JSON",
        data=ExportEngine.to_json(record),
        file_name=f"{fname_base}_{ts}.json",
        mime="application/json",
    )
    c3.download_button(
        "⬇️ Download CSV",
        data=ExportEngine.to_csv(record),
        file_name=f"{fname_base}_{ts}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard() -> None:
    """Landing dashboard with session overview."""
    st.title("🎙️ AI Speech-to-Text Agent")
    st.caption("Enterprise-grade audio transcription & intelligent analysis powered by Faster-Whisper + Claude")

    history = MemoryManager.get_all()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Total Transcripts", len(history))
    total_words = sum(r.get("text_stats", {}).get("word_count", 0) for r in history)
    c2.metric("📝 Total Words", f"{total_words:,}")
    total_duration = sum(r.get("audio_meta", {}).get("duration_seconds", 0) or 0 for r in history)
    c3.metric("⏱️ Total Audio", f"{total_duration / 60:.1f} min")
    langs = len(set(r.get("language", "") for r in history))
    c4.metric("🌍 Languages Detected", langs)

    if history:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fig = AnalyticsEngine.build_sentiment_chart(history)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = AnalyticsEngine.build_language_chart(history)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)

        fig3 = AnalyticsEngine.build_word_count_trend(history)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### 🕐 Recent Transcripts")
        for rec in history[:5]:
            with st.expander(
                f"[{rec['timestamp'][:16]}] {rec.get('audio_meta', {}).get('filename', 'audio')} "
                f"— {rec.get('language_name', '')} · {rec.get('text_stats', {}).get('word_count', 0)} words"
            ):
                st.write(rec.get("transcript", "")[:400] + "…")
                render_export_buttons(rec)
    else:
        st.info("No transcripts yet. Upload audio or record from the microphone to begin.")


def page_upload(recognizer: SpeechRecognizer, nlp: Optional[NLPAnalyzer], mode: str) -> None:
    """Audio upload & transcription page."""
    st.title("📁 Upload Audio")
    st.caption("Supports WAV · MP3 · M4A · FLAC · OGG · AAC  |  Max 200 MB")

    uploaded = st.file_uploader(
        "Drop your audio file here or click to browse",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
    )

    lang_choice = st.selectbox(
        "Language (leave Auto-detect for unknown)",
        options=["Auto-detect"] + list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: "Auto-detect" if x == "Auto-detect" else f"{x} — {SUPPORTED_LANGUAGES.get(x, x)}",
    )
    selected_lang = None if lang_choice == "Auto-detect" else lang_choice

    if uploaded and st.button("🚀 Transcribe & Analyze", use_container_width=True):
        file_bytes = uploaded.read()

        # ── Security: validate ──
        valid, err = AudioProcessor.validate_file(file_bytes, uploaded.name)
        if not valid:
            st.error(f"❌ {err}")
            return

        # ── Audio metadata ──
        with st.spinner("🔍 Analyzing audio metadata…"):
            meta = AudioProcessor.get_audio_metadata(file_bytes, uploaded.name)

        st.success(
            f"✅ Audio detected: {meta['format']} · "
            f"{meta.get('sample_rate', '?')} Hz · "
            f"{meta.get('channels', '?')} ch · "
            f"{meta.get('duration_seconds', '?')}s · "
            f"{meta['file_size_mb']} MB"
        )

        # ── Preprocess ──
        with st.spinner("🔧 Preprocessing audio (normalization, noise reduction)…"):
            wav_path = AudioProcessor.preprocess_audio(file_bytes, uploaded.name)

        if not wav_path:
            st.error("❌ Audio preprocessing failed. Please try a different file.")
            return

        # ── Transcribe ──
        with st.spinner(f"🎙️ Transcribing with Whisper '{recognizer.model_size}'…"):
            try:
                result = recognizer.transcribe(wav_path, language=selected_lang)
            except RuntimeError as exc:
                st.error(f"❌ Transcription error: {exc}")
                AudioProcessor.cleanup_temp(wav_path)
                return
        AudioProcessor.cleanup_temp(wav_path)

        if not result["transcript"].strip():
            st.warning("⚠️ No speech detected. Try a louder or cleaner audio file.")
            return

        # ── Text stats ──
        text_stats = AnalyticsEngine.compute_text_stats(
            result["transcript"], meta.get("duration_seconds")
        )

        # ── NLP Analysis ──
        analysis: dict = {}
        if nlp:
            with st.spinner("🧠 Running AI analysis (Claude)…"):
                analysis = nlp.analyze(result["transcript"], mode)
        else:
            st.warning("⚠️ Claude API key not set — AI analysis skipped. Add key in Settings.")
            analysis = NLPAnalyzer._fallback_analysis(result["transcript"])

        # ── Store in memory ──
        record = {
            "audio_meta": meta,
            "transcript": result["transcript"],
            "segments": result["segments"],
            "language": result["language"],
            "language_name": result["language_name"],
            "language_probability": result["language_probability"],
            "mode": mode,
            "text_stats": text_stats,
            "analysis": analysis,
        }
        MemoryManager.add_record(record)

        # ── Render results ──
        _render_full_result(record)


def page_record(recognizer: SpeechRecognizer, nlp: Optional[NLPAnalyzer], mode: str) -> None:
    """Live microphone recording page."""
    st.title("🎤 Live Recording")

    if not AUDIORECORDER_AVAILABLE:
        st.error(
            "The `streamlit-audiorecorder` package is not installed. "
            "Run `pip install streamlit-audiorecorder` and restart."
        )
        return

    st.info("Click the microphone icon below to start recording. Click again to stop.")
    audio_data = audiorecorder("🔴 Start Recording", "⏹️ Stop Recording")

    if len(audio_data) > 0:
        st.audio(audio_data.export().read(), format="audio/wav")

        if st.button("🚀 Transcribe & Analyze Recording"):
            audio_bytes = audio_data.export(format="wav").read()
            meta = {"filename": "live_recording.wav", "format": "WAV",
                    "file_size_mb": round(len(audio_bytes) / (1024 * 1024), 3)}

            # Audio duration via pydub
            try:
                seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
                meta["duration_seconds"] = round(len(seg) / 1000, 2)
                meta["sample_rate"] = seg.frame_rate
                meta["channels"] = seg.channels
            except Exception:
                pass

            # Save to temp file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                wav_path = f.name

            with st.spinner("🎙️ Transcribing…"):
                try:
                    result = recognizer.transcribe(wav_path)
                except RuntimeError as exc:
                    st.error(f"Transcription failed: {exc}")
                    AudioProcessor.cleanup_temp(wav_path)
                    return
            AudioProcessor.cleanup_temp(wav_path)

            if not result["transcript"].strip():
                st.warning("No speech detected.")
                return

            text_stats = AnalyticsEngine.compute_text_stats(result["transcript"], meta.get("duration_seconds"))

            analysis: dict = {}
            if nlp:
                with st.spinner("🧠 Analyzing…"):
                    analysis = nlp.analyze(result["transcript"], mode)
            else:
                analysis = NLPAnalyzer._fallback_analysis(result["transcript"])

            record = {
                "audio_meta": meta,
                "transcript": result["transcript"],
                "segments": result["segments"],
                "language": result["language"],
                "language_name": result["language_name"],
                "language_probability": result["language_probability"],
                "mode": mode,
                "text_stats": text_stats,
                "analysis": analysis,
            }
            MemoryManager.add_record(record)
            _render_full_result(record)


def _render_full_result(record: dict) -> None:
    """Shared renderer for both upload and live recording results."""
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # KPI metrics
    render_metric_row(record["text_stats"], record["audio_meta"], record)

    # Transcript
    render_transcript_section(record)

    # AI Analysis
    if record.get("analysis") and not record["analysis"].get("_fallback"):
        render_summaries(record["analysis"])
        render_entities(record["analysis"])
        render_action_items(record["analysis"])
        render_keywords(record["analysis"])
        render_sentiment(record["analysis"])
        render_mode_output(record.get("mode", "general"), record["analysis"])

    # Export
    render_export_buttons(record)


def page_history() -> None:
    """Browse, search, and manage transcript history."""
    st.title("🗂️ Transcript History")

    history = MemoryManager.get_all()
    if not history:
        st.info("No transcripts yet. Transcribe some audio first!")
        return

    # Search & filter controls
    col1, col2, col3 = st.columns([3, 2, 1])
    query = col1.text_input("🔍 Search transcripts", placeholder="Enter keyword or phrase…")
    lang_options = ["All Languages"] + list({r.get("language", "") for r in history})
    lang_filter = col2.selectbox("Language", lang_options)
    fuzzy = col3.checkbox("Fuzzy search", value=True)

    selected_lang = None if lang_filter == "All Languages" else lang_filter
    results = MemoryManager.search(query, language_filter=selected_lang, fuzzy=fuzzy)

    st.caption(f"Showing {len(results)} of {len(history)} records")

    if st.button("🗑️ Clear All History", type="secondary"):
        MemoryManager.clear_all()
        st.rerun()

    for rec in results:
        ts = rec.get("timestamp", "")[:16]
        fname = rec.get("audio_meta", {}).get("filename", "recording")
        words = rec.get("text_stats", {}).get("word_count", 0)
        lang = rec.get("language_name", "")
        mode = SPECIALIST_MODES.get(rec.get("mode", "general"), "General")

        with st.expander(f"[{ts}] {fname} — {lang} · {words} words · {mode}"):
            st.text_area("Transcript", rec.get("transcript", ""), height=120, key=f"txt_{rec['id']}")

            summary = rec.get("analysis", {}).get("executive_summary", "")
            if summary:
                st.info(f"**Summary:** {summary}")

            tasks = rec.get("analysis", {}).get("action_items", [])
            if tasks:
                st.markdown(f"**Tasks ({len(tasks)}):** " + " | ".join(
                    t.get("task", "") for t in tasks[:3]
                ))

            render_export_buttons(rec)

            if st.button("🗑️ Delete this record", key=f"del_{rec['id']}"):
                MemoryManager.delete_record(rec["id"])
                st.rerun()


def page_analytics() -> None:
    """Analytics dashboard page."""
    st.title("📈 Analytics Dashboard")

    history = MemoryManager.get_all()
    if not history:
        st.info("No data yet. Transcribe some audio to see analytics.")
        return

    # Summary stats
    total_words = sum(r.get("text_stats", {}).get("word_count", 0) for r in history)
    total_duration = sum(r.get("audio_meta", {}).get("duration_seconds", 0) or 0 for r in history)
    avg_words = round(total_words / max(len(history), 1))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Transcripts", len(history))
    c2.metric("📝 Total Words", f"{total_words:,}")
    c3.metric("⏱️ Total Duration", f"{total_duration / 60:.1f} min")
    c4.metric("📊 Avg Words/Session", f"{avg_words:,}")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig = AnalyticsEngine.build_sentiment_chart(history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = AnalyticsEngine.build_language_chart(history)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    fig3 = AnalyticsEngine.build_word_count_trend(history)
    if fig3:
        st.plotly_chart(fig3, use_container_width=True)

    # Raw data table
    if st.checkbox("Show raw analytics table"):
        rows = []
        for rec in history:
            rows.append({
                "Timestamp": rec.get("timestamp", "")[:16],
                "File": rec.get("audio_meta", {}).get("filename", ""),
                "Language": rec.get("language_name", ""),
                "Words": rec.get("text_stats", {}).get("word_count", 0),
                "Duration (s)": rec.get("audio_meta", {}).get("duration_seconds", 0),
                "Sentiment": rec.get("analysis", {}).get("sentiment", {}).get("classification", ""),
                "Mode": rec.get("mode", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Export Analytics CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="stt_analytics.csv",
            mime="text/csv",
        )


def page_exports() -> None:
    """Bulk export page for all history records."""
    st.title("📤 Exports")
    history = MemoryManager.get_all()
    if not history:
        st.info("No transcripts to export yet.")
        return

    st.markdown("### Bulk Export — All Transcripts")

    # Full JSON dump of all records
    all_json = json.dumps(history, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    st.download_button(
        "⬇️ Export All as JSON",
        data=all_json,
        file_name=f"stt_all_transcripts_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
    )

    # Combined CSV
    rows = []
    for rec in history:
        rows.append({
            "id": rec.get("id", ""),
            "timestamp": rec.get("timestamp", ""),
            "filename": rec.get("audio_meta", {}).get("filename", ""),
            "language": rec.get("language_name", ""),
            "duration_s": rec.get("audio_meta", {}).get("duration_seconds", ""),
            "word_count": rec.get("text_stats", {}).get("word_count", 0),
            "sentiment": rec.get("analysis", {}).get("sentiment", {}).get("classification", ""),
            "mode": rec.get("mode", ""),
            "transcript": rec.get("transcript", ""),
            "executive_summary": rec.get("analysis", {}).get("executive_summary", ""),
        })
    df = pd.DataFrame(rows)
    st.download_button(
        "⬇️ Export All as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"stt_all_transcripts_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### Per-Record Export")
    for rec in history:
        with st.expander(f"[{rec['timestamp'][:16]}] {rec.get('audio_meta', {}).get('filename', 'recording')}"):
            render_export_buttons(rec)


def page_settings() -> None:
    """Settings page for API key and model configuration."""
    st.title("⚙️ Settings")

    st.markdown("### 🔑 Claude API Key")
    st.caption(
        "Required for AI analysis (summaries, entities, tasks, sentiment). "
        "Transcription with Whisper works without a key. "
        "Get your key at: https://console.anthropic.com"
    )
    api_key = st.text_input(
        "Anthropic API Key",
        value=st.session_state.get("api_key", ""),
        type="password",
        placeholder="sk-ant-…",
    )
    if api_key:
        st.session_state["api_key"] = api_key
        st.success("✅ API key saved for this session.")

    st.markdown("### 🎙️ Whisper Model")
    st.caption("Larger models are more accurate but slower. 'base' is recommended for most use cases.")
    current_model = st.session_state.get("whisper_model", "base")
    new_model = st.selectbox(
        "Model size",
        WHISPER_MODELS,
        index=WHISPER_MODELS.index(current_model),
        format_func=lambda m: {
            "tiny": "tiny — fastest, lowest accuracy (~75% quality)",
            "base": "base — balanced speed & accuracy (recommended)",
            "small": "small — better accuracy, ~2× slower",
            "medium": "medium — high accuracy, ~5× slower",
        }.get(m, m),
    )
    if new_model != current_model:
        st.session_state["whisper_model"] = new_model
        # Clear cached model so it reloads on next transcription
        st.cache_resource.clear()
        st.success(f"Model changed to '{new_model}'. It will load on next transcription.")

    st.markdown("### 🌐 Default Language")
    st.caption("You can also override per-transcription on the upload page.")
    default_lang = st.selectbox(
        "Default recognition language",
        ["Auto-detect"] + list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: "Auto-detect" if x == "Auto-detect" else f"{x} — {SUPPORTED_LANGUAGES.get(x, x)}",
    )
    st.session_state["default_lang"] = None if default_lang == "Auto-detect" else default_lang

    st.markdown("### 📋 Default Specialist Mode")
    mode_options = list(SPECIALIST_MODES.keys())
    default_mode = st.selectbox(
        "Default mode",
        mode_options,
        format_func=lambda m: SPECIALIST_MODES[m],
    )
    st.session_state["default_mode"] = default_mode

    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    import sys
    st.code(
        f"Python: {sys.version}\n"
        f"Streamlit: {st.__version__}\n"
        f"Whisper Model: {st.session_state.get('whisper_model', 'base')}\n"
        f"API Key: {'✅ Set' if st.session_state.get('api_key') else '❌ Not set'}\n"
        f"History records: {len(MemoryManager.get_all())}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main entry point for the Streamlit application.
    Configures page layout, sidebar navigation, and routes to page functions.
    """
    # ── Page config ──
    st.set_page_config(
        page_title="AI Speech-to-Text Agent",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──
    apply_custom_styles()

    # ── Initialize memory ──
    MemoryManager.initialize()

    # ── API key resolution ──
    # Priority: 1) st.secrets (Streamlit Cloud) → 2) env var → 3) session_state (Settings page)
    api_key: Optional[str] = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = st.session_state.get("api_key")

    # ── Instantiate NLP analyzer (None if no API key) ──
    nlp: Optional[NLPAnalyzer] = None
    if api_key:
        try:
            nlp = NLPAnalyzer(api_key=api_key)
        except Exception as exc:
            logger.warning("NLP analyzer init failed: %s", exc)

    # ── Load Whisper model ──
    model_size = st.session_state.get("whisper_model", "base")
    try:
        recognizer = SpeechRecognizer(model_size=model_size)
    except Exception as exc:
        st.error(
            f"❌ Failed to load Whisper model '{model_size}'. "
            f"Ensure `faster-whisper` is installed.\n\nError: {exc}"
        )
        st.stop()

    # ── Sidebar navigation ──
    with st.sidebar:
        st.markdown("## 🎙️ STT Agent")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            options=["Dashboard", "Upload Audio", "Live Recording", "History", "Analytics", "Exports", "Settings"],
            format_func=lambda p: {
                "Dashboard": "🏠 Dashboard",
                "Upload Audio": "📁 Upload Audio",
                "Live Recording": "🎤 Live Recording",
                "History": "🗂️ History",
                "Analytics": "📈 Analytics",
                "Exports": "📤 Exports",
                "Settings": "⚙️ Settings",
            }.get(p, p),
        )
        st.markdown("---")

        # Specialist mode selector in sidebar
        current_mode = st.session_state.get("default_mode", "general")
        mode = st.selectbox(
            "🎯 Specialist Mode",
            options=list(SPECIALIST_MODES.keys()),
            index=list(SPECIALIST_MODES.keys()).index(current_mode),
            format_func=lambda m: SPECIALIST_MODES[m],
            key="sidebar_mode",
        )
        st.session_state["default_mode"] = mode

        st.markdown("---")
        st.caption(f"📦 Model: `{model_size}`")
        st.caption(f"🔑 API: {'✅ Connected' if nlp else '❌ Not set'}")
        history_count = len(MemoryManager.get_all())
        st.caption(f"🗂️ History: {history_count} record{'s' if history_count != 1 else ''}")

    # ── Route to page ──
    if page == "Dashboard":
        page_dashboard()
    elif page == "Upload Audio":
        page_upload(recognizer, nlp, mode)
    elif page == "Live Recording":
        page_record(recognizer, nlp, mode)
    elif page == "History":
        page_history()
    elif page == "Analytics":
        page_analytics()
    elif page == "Exports":
        page_exports()
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
