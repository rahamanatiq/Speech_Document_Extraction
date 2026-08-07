from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LabResultLine:
    """One row from the lab report table — one test, one value."""

    raw_line: str                      # exactly what OCR read, untouched — never discarded
    test_name: str | None = None
    value: str | None = None           # kept as raw string here; numeric parsing happens later, in services/normalizer.py
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None            # e.g. "High", "Low", "Normal" — only if explicitly present in the source


@dataclass
class OCRResult:
    """The shape every OCR provider must return, real or mock."""

    raw_text: str                              # the full, untouched OCR dump of the page
    meta: dict[str, str] = field(default_factory=dict)   # e.g. patient name, report date — best-effort, may be partial
    results: list[LabResultLine] = field(default_factory=list)
    confidence: float | None = None


class OCRProvider(ABC):
    """Contract that both the real EasyOCR adapter and the mock adapter must satisfy."""

    @abstractmethod
    def extract(self, image_bytes: bytes) -> OCRResult:
        """Extract structured lab-report data from an image. Implementations must
        never invent a value that wasn't actually read — if a field is unclear,
        leave it None and preserve the raw_line so the caller can see why."""
        raise NotImplementedError