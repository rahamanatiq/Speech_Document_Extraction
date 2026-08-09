from fastapi import APIRouter, File, Form, UploadFile

from api.schemas import TranscribeResponse
from services.transcribe_service import TranscriptionService
from core.config import settings

router = APIRouter()

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file: .wav, .mp3, .m4a, or .ogg"),
    language: str = Form("auto", description="'en', 'bn', or 'auto'"),
) -> TranscribeResponse:
    """Transcribe an uploaded audio file to text.

    This function's only job is HTTP plumbing: read the upload, hand it to
    the service, translate the result into the response schema. All real
    logic — validation, provider selection, error handling — lives in
    TranscriptionService. If this function ever needs an `if`/`else` beyond
    picking a status code, that's a sign logic has leaked into the wrong layer.
    """
    audio_bytes = await file.read()

    service = TranscriptionService()
    result = service.transcribe(filename=file.filename or "", audio_bytes=audio_bytes, language=language)

    return TranscribeResponse(
        raw_text=result.raw_text,
        language=result.language,
        duration_seconds=result.duration_seconds,
        confidence=result.confidence,
        provider=settings.speech_provider,
    )

