import io
import wave

from adapters.transcription.base import TranscriptionProvider, TranscriptionResult
from core.config import settings


class WhisperTranscriptionProvider(TranscriptionProvider):
    """Real speech-to-text using faster-whisper, self-hosted, CPU-friendly.

    The model is loaded once, lazily, on first use — not at import time,
    and not in __init__ either — so simply constructing this class (which
    happens once per request in our current services/ code) never pays the
    multi-second model-load cost more than once per process lifetime.
    """

    _model = None  # class-level cache, shared across all instances in this process

    def _get_model(self):
        if WhisperTranscriptionProvider._model is None:
            # Imported here, not at module level, so this heavy library is
            # never loaded at all when running on mocks — consistent with
            # the lazy-import pattern already used in the service factories.
            from faster_whisper import WhisperModel

            WhisperTranscriptionProvider._model = WhisperModel(
                settings.whisper_model_size,
                device="cpu",
                compute_type="int8",  # quantized — meaningfully faster on CPU, small accuracy trade-off
            )
        return WhisperTranscriptionProvider._model

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if len(audio_bytes) == 0:
            # Mirrors the mock's behavior for this exact edge case, so
            # callers see consistent handling regardless of provider.
            return TranscriptionResult(raw_text="", language="en", duration_seconds=0.0, confidence=0.0)

        model = self._get_model()

        # faster-whisper's language param expects None for auto-detection,
        # not the string "auto" — this is the adapter's job to translate,
        # so callers everywhere else in the codebase only ever deal with
        # our own "auto"/"en"/"bn" convention.
        whisper_language = None if language == "auto" else language

        segments, info = model.transcribe(
            io.BytesIO(audio_bytes),
            language=whisper_language,
            vad_filter=True,
        )
        segments = list(segments)  # faster-whisper returns a generator; materialize it to measure/use safely

        full_text = " ".join(segment.text.strip() for segment in segments)
        avg_confidence = (
            sum(_segment_confidence(s) for s in segments) / len(segments) if segments else 0.0
        )

        return TranscriptionResult(
            raw_text=full_text,
            language=info.language,
            duration_seconds=round(info.duration, 2),
            confidence=round(avg_confidence, 2),
        )


def _segment_confidence(segment) -> float:
    """faster-whisper exposes avg_logprob (a log-probability, negative,
    closer to 0 = more confident), not a 0-1 confidence score directly.
    This converts it to a rough 0-1 scale for consistency with our
    TranscriptionResult contract, which the mock also populates as 0-1.
    """
    import math

    return max(0.0, min(1.0, math.exp(segment.avg_logprob)))