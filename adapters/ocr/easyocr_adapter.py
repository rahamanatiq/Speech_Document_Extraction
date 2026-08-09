import io
import re
 
from adapters.ocr.base import LabResultLine, OCRProvider, OCRResult
 
# --- Column-splitting and field classification ---
# Lab reports are visually columnar (test name, value, unit, range, flag,
# separated by wide gaps). We approximate "columns" by splitting on 2+
# consecutive spaces, since EasyOCR's line reconstruction (below) inserts
# double spaces between separate detected text boxes.
 
_VALUE_RE = re.compile(r"^[<>]?\s*-?\d+(\.\d+)?$")
_RANGE_RE = re.compile(r"^=?\s*-?\d+(\.\d+)?\s*-\s*-?\d+(\.\d+)?$")
_UNIT_RE = re.compile(r"^[A-Za-zµ%/^0-9.]+$")
_FLAG_WORDS = {"H", "HIGH", "L", "LOW", "N", "NORMAL"}
 
# A stricter check than _UNIT_RE: used specifically to decide "does this
# line have genuine lab-report signal", not just "does this column look
# unit-shaped". Deliberately narrower to avoid a bare number or word being
# mistaken for a unit.
_LAB_UNIT_HINT_RE = re.compile(
    r"(g/dl|mg/dl|mmol|meq|k/[uµ]l|10\^|10\*|%|fl|pg|um\^3|cmm|/cmm)", re.IGNORECASE
)
 
 
def _split_columns(line: str) -> list[str]:
    parts = re.split(r"\s{2,}", line.strip())
    return [p.strip() for p in parts if p.strip()]
 
 
def _looks_like_data_row(line: str) -> bool:
    """Distinguishes an actual lab-result row from a section header, a
    column-title row, or an unrelated document's line (e.g. a receipt's
    price list).
 
    Two conditions must BOTH hold — a bare number is not enough on its own,
    since that also matches a receipt's price column:
      1. at least one column looks like a plain value, comparator value,
         or numeric range
      2. at least one column (anywhere in the line) contains a recognizable
         lab-unit token (g/dL, mg/dl, %, K/uL, etc.) OR a numeric range,
         since ranges are themselves a strong lab-report-specific signal
 
    A line that only satisfies #1 (e.g. a receipt line with a price and an
    item code) is excluded entirely, rather than guessed at.
    """
    columns = _split_columns(line)
    if len(columns) < 2:
        return False
 
    has_value_or_range = any(_VALUE_RE.match(c) or _RANGE_RE.match(c) for c in columns[1:])
    has_lab_signal = any(_LAB_UNIT_HINT_RE.search(c) or _RANGE_RE.match(c) for c in columns)
 
    return has_value_or_range and has_lab_signal
 
 
def _parse_line(raw_line: str) -> LabResultLine:
    """Best-effort field extraction from one reconstructed line. Any column
    that doesn't confidently match a known pattern is left None — raw_line
    always preserves exactly what OCR read, regardless of parse success.
    """
    columns = _split_columns(raw_line)
    if len(columns) < 2:
        return LabResultLine(raw_line=raw_line)
 
    first = columns[0]
    test_name = first if not (_VALUE_RE.match(first) or _RANGE_RE.match(first)) else None
 
    value = unit = reference_range = flag = None
    for col in columns[1:]:
        if flag is None and col.upper() in _FLAG_WORDS:
            flag = col
        elif value is None and _VALUE_RE.match(col):
            value = col
        elif reference_range is None and _RANGE_RE.match(col):
            reference_range = col.lstrip("=").strip()
        elif unit is None and _UNIT_RE.match(col) and not _VALUE_RE.match(col):
            unit = col
 
    return LabResultLine(
        raw_line=raw_line, test_name=test_name, value=value,
        unit=unit, reference_range=reference_range, flag=flag,
    )
 
 
def _group_into_lines(ocr_results, y_threshold: float = 12.0) -> list[str]:
    """EasyOCR returns individual text detections with pixel bounding boxes,
    not reconstructed lines. This groups detections into rows by vertical
    (y) proximity, then orders each row left-to-right by x position —
    turning scattered word-level boxes back into readable table rows.
    """
    detections = []
    for bbox, text, _confidence in ocr_results:
        y_center = sum(point[1] for point in bbox) / 4
        x_left = min(point[0] for point in bbox)
        detections.append((y_center, x_left, text))
    detections.sort(key=lambda d: d[0])
 
    groups: list[list[tuple[float, float, str]]] = []
    for y, x, text in detections:
        if groups and abs(y - groups[-1][0][0]) < y_threshold:
            groups[-1].append((y, x, text))
        else:
            groups.append([(y, x, text)])
 
    lines = []
    for group in groups:
        group.sort(key=lambda item: item[1])
        lines.append("  ".join(text for _, _, text in group))
    return lines
 
 
def _extract_meta(lines: list[str]) -> dict[str, str]:
    """Best-effort header extraction. Deliberately does NOT try to parse out
    a clean 'patient_name' value — that would risk guessing which part of a
    messy header line is actually the name. Instead, whole lines that look
    header-like are kept verbatim, consistent with 'never guess, preserve
    what was actually read.'
    """
    meta = {}
    for line in lines:
        lower = line.lower()
        if "patient" in lower or "name" in lower:
            meta.setdefault("patient_line", line.strip())
        if "date" in lower:
            meta.setdefault("date_line", line.strip())
    return meta
 
 
def _preprocess(image):
    """Upscales small/low-res images and boosts contrast before OCR.
 
    Directly targets two real failure modes observed during testing:
      - a low-resolution screenshot that EasyOCR read almost entirely
        wrong (confidence ~0.15, 0 parsed rows), far worse than genuine
        phone photos of the same kind of document
      - generally low-contrast phone photos taken in poor lighting
 
    EasyOCR's text detector works on a fixed grid of pixel regions; a
    low-res image gives it too little detail per character to detect
    reliably. Upscaling via LANCZOS (a high-quality resampling filter)
    gives it more pixels to work with. This doesn't add real information
    that wasn't there, but it does make existing detail easier for the
    model to resolve — a standard, well-established OCR preprocessing step.
    """
    from PIL import Image, ImageEnhance
 
    min_dimension = 1000
    if min(image.size) < min_dimension:
        scale = min_dimension / min(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
 
    return ImageEnhance.Contrast(image).enhance(1.3)
 
 
class EasyOCRProvider(OCRProvider):
    """Real OCR using EasyOCR (deep-learning based), self-hosted, no
    credentials. Chosen over Tesseract for better accuracy on angled/
    poor-light photos — see DECISIONS.md for the full trade-off.
    """
 
    _reader = None  # class-level cache — same lazy-load pattern as Whisper
 
    def _get_reader(self):
        if EasyOCRProvider._reader is None:
            import easyocr  # deferred — keeps the mock path dependency-free
 
            EasyOCRProvider._reader = easyocr.Reader(["en"], gpu=False)
        return EasyOCRProvider._reader
 
    def extract(self, image_bytes: bytes) -> OCRResult:
        if len(image_bytes) == 0:
            return OCRResult(raw_text="", meta={}, results=[], confidence=0.0)
 
        import numpy as np
        from PIL import Image
 
        reader = self._get_reader()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = _preprocess(image)
        ocr_results = reader.readtext(np.array(image))
 
        lines = _group_into_lines(ocr_results)
        raw_text = "\n".join(lines)
 
        confidences = [conf for _, _, conf in ocr_results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
 
        return OCRResult(
            raw_text=raw_text,
            meta=_extract_meta(lines),
            results=[_parse_line(line) for line in lines if _looks_like_data_row(line)],
            confidence=round(avg_confidence, 2),
        )