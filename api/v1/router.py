from fastapi import APIRouter

from api.v1.endpoints import document, transcribe

router = APIRouter(prefix="/api/v1")

router.include_router(transcribe.router, tags=["transcription"])
router.include_router(document.router, tags=["documents"])