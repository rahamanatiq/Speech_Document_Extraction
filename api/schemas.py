from enum import Enum

from pydantic import BaseModel, Field


# --- Transcription ---

class TranscribeResponse(BaseModel):
    raw_text: str = Field(..., description="The transcribed text, exactly as returned by the provider")
    language: str = Field(..., description="Resolved language code, e.g. 'en' or 'bn'")
    duration_seconds: float
    confidence: float | None = Field(None, description="Provider confidence score, if available")
    provider: str = Field(..., description="Which provider produced this result, e.g. 'mock' or 'whisper'")


# --- Document extraction ---

class ValueTypeSchema(str, Enum):
    EXACT = "exact"
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    RANGE = "range"
    UNPARSEABLE = "unparseable"


class NormalizedValueSchema(BaseModel):
    raw: str
    value_type: ValueTypeSchema
    numeric_value: float | None = None
    range_low: float | None = None
    range_high: float | None = None


class UnitTypeSchema(str, Enum):
    KNOWN = "known"
    UNPARSEABLE = "unparseable"


class NormalizedUnitSchema(BaseModel):
    raw: str
    unit_type: UnitTypeSchema
    canonical: str | None = None


class LabResultLineResponse(BaseModel):
    raw_line: str = Field(..., description="Exactly what OCR read for this line, unmodified")
    test_name: str | None = None
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None
    normalized: NormalizedValueSchema
    normalized_unit: NormalizedUnitSchema


class DocumentExtractResponse(BaseModel):
    raw_text: str = Field(..., description="The full, untouched OCR dump of the page")
    meta: dict[str, str] = Field(default_factory=dict)
    results: list[LabResultLineResponse]
    confidence: float | None = None


# --- Shared error shape ---

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Machine-readable error type, e.g. 'UnsupportedFormatError'")
    detail: str = Field(..., description="Human-readable explanation of what went wrong")