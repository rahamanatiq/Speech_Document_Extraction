
from dataclasses import dataclass
 
from adapters.ocr.base import OCRProvider, OCRResult
from adapters.ocr.mock_adapter import MockOCRProvider
from core.config import settings
from core.exceptions import FileTooLargeError, ProviderError, UnsupportedFormatError
from services.normalizer import NormalizedValue, ValueType, normalize_value
 
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB — generous for phone photos, rejects accidental huge uploads
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
 
 
@dataclass
class EnrichedLabResultLine:
    """A LabResultLine plus its normalized value — built by the service
    layer, on top of the adapter's raw output. Keeps adapters/ ignorant
    of normalization entirely, preserving the one-way layer dependency
    (api/ -> services/ -> adapters/, never the reverse).
    """
 
    raw_line: str
    test_name: str | None
    value: str | None
    unit: str | None
    reference_range: str | None
    flag: str | None
    normalized: NormalizedValue
 
 
@dataclass
class DocumentExtractionResult:
    """What DocumentService returns to api/ — the OCRResult's data,
    enriched with normalized values for each result line."""
 
    raw_text: str
    meta: dict[str, str]
    results: list[EnrichedLabResultLine]
    confidence: float | None
 
 
def get_ocr_provider() -> OCRProvider:
    """Factory: returns whichever OCR provider is configured. One single
    switch point — change OCR_PROVIDER in .env, and every caller of this
    function automatically gets the new provider, with zero other changes.
    """
    if settings.ocr_provider == "mock":
        return MockOCRProvider()
    if settings.ocr_provider == "easyocr":
        # Lazy import — EasyOCR pulls in PyTorch, which is large and slow
        # to import. Deferring this keeps the default (mock) path fast and
        # dependency-free.
        from adapters.ocr.easyocr_adapter import EasyOCRProvider
 
        return EasyOCRProvider()
 
    raise ValueError(f"Unknown ocr_provider configured: {settings.ocr_provider!r}")
 
 
class DocumentService:
    """Orchestrates a document extraction request: validates input,
    delegates to whichever OCR provider is configured, and enriches the
    raw result with normalized values. api/ talks to this class only,
    never to adapters/ directly.
    """
 
    def __init__(self, provider: OCRProvider | None = None) -> None:
        # Optional injected provider makes this class unit-testable in
        # isolation, without touching global settings.
        self._provider = provider or get_ocr_provider()
 
    def extract(self, filename: str, image_bytes: bytes) -> DocumentExtractionResult:
        self._validate(filename, image_bytes)
 
        try:
            raw_result = self._provider.extract(image_bytes)
        except Exception as exc:
            # Any unexpected provider failure becomes our own domain
            # exception — api/ only ever needs to handle ProviderError.
            raise ProviderError(f"OCR provider failed: {exc}") from exc
 
        return self._enrich(raw_result)
 
    def _enrich(self, raw_result: OCRResult) -> DocumentExtractionResult:
        """Adds a NormalizedValue to every result line, without ever
        touching or reinterpreting what the adapter already returned.
        """
        enriched_lines = [
            EnrichedLabResultLine(
                raw_line=line.raw_line,
                test_name=line.test_name,
                value=line.value,
                unit=line.unit,
                reference_range=line.reference_range,
                flag=line.flag,
                normalized=(
                    normalize_value(line.value)
                    if line.value is not None
                    # The adapter already couldn't read this value (e.g. the
                    # OCR-garbled "l2.S" case) — don't bother normalizing
                    # None, just mark it unparseable directly.
                    else NormalizedValue(raw="", value_type=ValueType.UNPARSEABLE)
                ),
            )
            for line in raw_result.results
        ]
 
        return DocumentExtractionResult(
            raw_text=raw_result.raw_text,
            meta=raw_result.meta,
            results=enriched_lines,
            confidence=raw_result.confidence,
        )
 
    def _validate(self, filename: str, image_bytes: bytes) -> None:
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"'{extension}' is not supported. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
 
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise FileTooLargeError(
                f"File is {len(image_bytes)} bytes, exceeds the {MAX_IMAGE_BYTES} byte limit"
            )