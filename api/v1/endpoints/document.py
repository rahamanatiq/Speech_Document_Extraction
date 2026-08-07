from fastapi import APIRouter, File, UploadFile

from api.schemas import (
    DocumentExtractResponse,
    LabResultLineResponse,
    NormalizedValueSchema,
    ValueTypeSchema,
)
from services.document_service import DocumentService

router = APIRouter()


@router.post("/documents/extract", response_model=DocumentExtractResponse)
async def extract_document(
    file: UploadFile = File(..., description="Lab report image: .jpg, .jpeg, .png, or .webp"),
) -> DocumentExtractResponse:
    """Extract structured lab-report data from an uploaded image.

    Same principle as transcribe_audio: this function only handles HTTP
    plumbing. All validation, OCR provider selection, and normalization
    logic lives in DocumentService.
    """
    image_bytes = await file.read()

    service = DocumentService()
    result = service.extract(filename=file.filename, image_bytes=image_bytes)

    return DocumentExtractResponse(
        raw_text=result.raw_text,
        meta=result.meta,
        results=[
            LabResultLineResponse(
                raw_line=line.raw_line,
                test_name=line.test_name,
                value=line.value,
                unit=line.unit,
                reference_range=line.reference_range,
                flag=line.flag,
                normalized=NormalizedValueSchema(
                    raw=line.normalized.raw,
                    # Explicit boundary conversion: internal ValueType enum
                    # -> public ValueTypeSchema enum. This line IS the layer
                    # boundary made visible — api/ decides what's exposed,
                    # services/ decides what's true internally.
                    value_type=ValueTypeSchema(line.normalized.value_type.value),
                    numeric_value=line.normalized.numeric_value,
                    range_low=line.normalized.range_low,
                    range_high=line.normalized.range_high,
                ),
            )
            for line in result.results
        ],
        confidence=result.confidence,
    )