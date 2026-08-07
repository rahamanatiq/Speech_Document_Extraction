from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """The shape every transcription provider must return, real or mock."""

    raw_text: str
    language: str
    duration_seconds: float
    confidence: float | None = None


class TranscriptionProvider(ABC):
    """Contract that both the real Whisper adapter and the mock adapter must satisfy."""

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        """Transcribe audio bytes into text. Implementations must never raise
        raw provider exceptions — wrap them in a domain-level error instead."""
        raise NotImplementedError