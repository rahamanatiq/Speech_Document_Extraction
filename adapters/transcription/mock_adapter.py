import time

from adapters.transcription.base import TranscriptionProvider, TranscriptionResult


class MockTranscriptionProvider(TranscriptionProvider):
    """Returns a canned transcription instead of running a real model.

    Used as the default provider so the service runs with zero setup and
    zero credentials. Also makes tests fast and deterministic — no model
    inference means no flakiness and no waiting.
    """

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        # A real model would take real time to run; a tiny artificial delay
        # here keeps the mock's timing behavior *realistic* for anyone
        # testing latency/timeout handling against it.
        time.sleep(0.05)

        if len(audio_bytes) == 0:
            # Mirrors a real edge case (silent/empty audio) so services/
            # can be tested against it without needing a real silent file.
            return TranscriptionResult(
                raw_text="",
                language=language if language != "auto" else "en",
                duration_seconds=0.0,
                confidence=0.0,
            )

        resolved_language = "bn" if language == "bn" else "en"
        sample_text = (
            "রোগীর রক্তচাপ স্বাভাবিক রয়েছে।"
            if resolved_language == "bn"
            else "The patient's blood pressure is within normal range."
        )

        return TranscriptionResult(
            raw_text=sample_text,
            language=resolved_language,
            duration_seconds=round(len(audio_bytes) / 16000, 2),
            confidence=0.95,
        )