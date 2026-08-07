import time

from adapters.ocr.base import LabResultLine, OCRProvider, OCRResult


class MockOCRProvider(OCRProvider):
    """Returns a canned, structured lab-report extraction instead of running
    a real OCR model. Default provider — zero setup, zero credentials,
    deterministic for tests.
    """

    def extract(self, image_bytes: bytes) -> OCRResult:
        time.sleep(0.05)  # mimics real OCR's non-trivial processing time

        if len(image_bytes) == 0:
            # Edge case: empty/corrupt upload — mirrors what a real adapter
            # must also handle gracefully rather than crashing.
            return OCRResult(raw_text="", meta={}, results=[], confidence=0.0)

        raw_text = (
            "Patient: John Doe\n"
            "Date: 2026-03-14\n"
            "Hemoglobin  13.5  g/dL  (13.0-17.0)\n"
            "WBC Count   l2.S  x10^3/uL  (4.0-11.0)  High\n"
            "Platelet    <0.5  x10^3/uL  (150-410)\n"
        )

        results = [
            LabResultLine(
                raw_line="Hemoglobin  13.5  g/dL  (13.0-17.0)",
                test_name="Hemoglobin",
                value="13.5",
                unit="g/dL",
                reference_range="13.0-17.0",
                flag=None,
            ),
            LabResultLine(
                # Deliberately OCR-garbled: "12.5" misread as "l2.S" (lowercase L, capital S).
                # This is here on purpose — it proves the "never guess" rule end-to-end:
                # raw_line preserves exactly what was read, value stays None because
                # we genuinely can't be sure "l2.S" means "12.5".
                raw_line="WBC Count   l2.S  x10^3/uL  (4.0-11.0)  High",
                test_name="WBC Count",
                value=None,
                unit="x10^3/uL",
                reference_range="4.0-11.0",
                flag="High",
            ),
            LabResultLine(
                # Deliberately a "less-than" value — tests that normalizer.py
                # (coming soon) handles comparison operators, not just plain numbers.
                raw_line="Platelet    <0.5  x10^3/uL  (150-410)",
                test_name="Platelet",
                value="<0.5",
                unit="x10^3/uL",
                reference_range="150-410",
                flag=None,
            ),
        ]

        return OCRResult(
            raw_text=raw_text,
            meta={"patient_name": "John Doe", "report_date": "2026-03-14"},
            results=results,
            confidence=0.88,
        )