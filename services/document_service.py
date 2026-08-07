from adapters.ocr.base import OCRProvider, OCRResult
from adapters.ocr.mock_adapter import MockOCRProvider
from core.config import settings
from core.exceptions import FileTooLargeError, ProviderError, UnsupportedFormatError

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB — generous for phone photos, rejects accidental huge uploads
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_ocr_provider() -> OCRProvider:
    """Factory: returns whichever OCR provider is configured. Same single
    switch-point pattern as get_transcription_provider() — one place
    decides mock vs. real, nothing downstream needs to know or care.
    """
    if settings.ocr_provider == "mock":
        return MockOCRProvider()
    if settings.ocr_provider == "easyocr":
        # Lazy import — EasyOCR pulls in PyTorch, which is large and slow
        # to import. Deferring this means the mock path stays fast and
        # this heavy dependency is only loaded when actually needed.
        from adapters.ocr.easyocr_adapter import EasyOCRProvider

        return EasyOCRProvider()

    raise ValueError(f"Unknown ocr_provider configured: {settings.ocr_provider!r}")


class DocumentService:
    """Orchestrates a document extraction request: validates input,
    delegates to whichever OCR provider is configured, and returns the
    structured result. api/ talks to this class, never to adapters/ directly.
    """

    def __init__(self, provider: OCRProvider | None = None) -> None:
        self._provider = provider or get_ocr_provider()

    def extract(self, filename: str, image_bytes: bytes) -> OCRResult:
        self._validate(filename, image_bytes)

        try:
            return self._provider.extract(image_bytes)
        except Exception as exc:
            raise ProviderError(f"OCR provider failed: {exc}") from exc

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