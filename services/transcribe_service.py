from adapters.transcription.base import TranscriptionProvider, TranscriptionResult
from adapters.transcription.mock_adapter import MockTranscriptionProvider
from core.config import settings
from core.exceptions import FileTooLargeError, ProviderError, UnsupportedFormatError

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB — generous for short clips, rejects accidental huge uploads
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}


def get_transcription_provider() -> TranscriptionProvider:
    """Factory: returns whichever provider is configured, without the caller
    needing to know or care which concrete class it is.

    This one function is the single switch point for the whole app — change
    SPEECH_PROVIDER in .env, and every caller of this function automatically
    gets the new provider, with zero other code changes.
    """
    if settings.speech_provider == "mock":
        return MockTranscriptionProvider()
    if settings.speech_provider == "whisper":
        # Imported lazily, inside the branch, so the whisper library and its
        # heavy dependencies are never even imported when running on mocks —
        # this keeps the default (mock) path fast and dependency-free.
        from adapters.transcription.whisper_adapter import WhisperTranscriptionProvider

        return WhisperTranscriptionProvider()

    raise ValueError(f"Unknown speech_provider configured: {settings.speech_provider!r}")


class TranscriptionService:
    """Orchestrates a transcription request: validates input, delegates to
    whichever provider is configured, and returns the result.

    This class is what api/ actually talks to — it never talks to adapters/
    directly.
    """

    def __init__(self, provider: TranscriptionProvider | None = None) -> None:
        # Accepting an optional provider (instead of always calling the
        # factory internally) is what makes this class unit-testable in
        # isolation — tests can inject their own provider without touching
        # global settings.
        self._provider = provider or get_transcription_provider()

    def transcribe(self, filename: str, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        self._validate(filename, audio_bytes)

        try:
            return self._provider.transcribe(audio_bytes, language=language)
        except Exception as exc:
            # Any unexpected failure from the provider is re-raised as our
            # own domain exception — api/ only ever needs to handle
            # ProviderError, never a provider-specific exception type.
            raise ProviderError(f"Transcription provider failed: {exc}") from exc

    def _validate(self, filename: str, audio_bytes: bytes) -> None:
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"'{extension}' is not supported. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise FileTooLargeError(
                f"File is {len(audio_bytes)} bytes, exceeds the {MAX_AUDIO_BYTES} byte limit"
            )