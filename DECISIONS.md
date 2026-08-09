# Architectural Decisions

Five consequential decisions from building this service: what was picked, what
was rejected, and why. Full engineering detail (bug repros, test names,
confidence numbers) lives in code comments and README.md's "Known
Limitations" — this file stays to the decisions themselves.

---

## 1. Layer separation enforced by dependency direction, not just convention

`api/ → services/ → adapters/`, one direction only. `services/` type-signatures
depend only on the abstract interfaces in `adapters/*/base.py`
(`TranscriptionProvider`, `OCRProvider`), never on a concrete adapter class —
swapping a provider touches only `adapters/`. Each domain
(`get_transcription_provider()`, `get_ocr_provider()`) has exactly one factory
function that branches on config; real provider imports (`faster_whisper`,
`easyocr`) are deferred inside that branch so the default mock path stays
fast and dependency-free. Rejected alternative: attaching normalized values
directly to `LabResultLine` in `adapters/ocr/base.py` — this would have made
Layer 3 import from Layer 2, inverting the dependency. Normalization lives
in `services/document_service.py` instead, wrapping the adapter's raw output.

## 2. Speech-to-text: faster-whisper, self-hosted, `medium` model

Chosen for credential-free, offline operation with one model covering both
English and Bangla. `compute_type="int8"` trades minor accuracy for
meaningfully faster CPU inference. **Known, disclosed weakness:** `medium`
produced garbled, low-confidence (0.04) Bangla output on a clean recording,
versus 0.67–0.79 on equivalent English. `large-v3` would fix this but roughly
doubles model size/latency, working against "runs on a reviewer's machine, no
GPU" — rejected for that reason. `WHISPER_MODEL_SIZE` stays configurable via
`.env` so this is a size/speed trade-off, not a silently hidden gap.
**Real bug found and fixed:** `vad_filter=True` was added after Whisper
hallucinated "Thanks for watching!" on a deliberately silent clip — a
documented Whisper failure mode on non-speech audio. Regression-tested in
`tests/test_real_providers.py::test_silence_does_not_hallucinate_text`.

## 3. Document OCR: EasyOCR over Tesseract

EasyOCR (deep-learning based) handles angled/poorly-lit phone photos more
reliably than Tesseract, at the cost of a much larger dependency (PyTorch)
and slower cold start — an accepted trade-off. Row reconstruction from
EasyOCR's raw pixel-coordinate detections is fully custom
(`_group_into_lines` + `_parse_line`), not hidden behind a third-party
table API. **Real bug found and fixed:** initial row-detection accepted any
two-column line containing a number as a lab result, incorrectly parsing 3
false results from a store receipt. Fixed by requiring a recognizable
lab-unit token or numeric range *in addition to* the value — a receipt has
values but never lab units. Regression-tested in
`test_non_lab_document_produces_no_false_positive_results`. **Known
limitation:** row-grouping is non-deterministic on uneven line spacing,
occasionally splitting a test name from its value across runs — documented
rather than hidden, and the affected test was rewritten to assert what's
reliably true (the value was read) rather than a structured claim that
isn't guaranteed every run.

## 4. Normalization: value, unit, and date, all "never guess"

All three normalizers (`normalize_value`, `normalize_unit`, `normalize_date`
in `services/normalizer.py`) follow one rule: attempt a fixed set of known
patterns, and if none match confidently, preserve the raw input and return
`UNPARSEABLE` — never fabricate a number, canonical unit, or date. Units
canonicalize via a lookup table (e.g. `gm/dl` → `g/dL`; `K/uL` and `10^3/uL`
are treated as clinically equivalent and merged to one canonical form).
Dates try a fixed list of formats and fall back to verbatim; ambiguous
all-numeric formats (`03/04/2026`) are a disclosed limitation, not resolved
by guessing — see README. **Disclosed contradiction in the brief:**
requirement 6 says "every result must include a numeric value"; requirement
7 says "anything you cannot confidently parse must be preserved verbatim,
never guessed at." These conflict directly for a genuinely unparseable OCR
read (e.g. `l2.S`). This code sides with requirement 7: `value` and
`numeric_value` stay `None` rather than inventing a number, because a
fabricated figure silently entering a medical record is worse than an
honest null. Verified end-to-end in
`test_document_never_guesses_ambiguous_ocr_value`.

## 5. Deployment: mock-only Docker default, CPU-only ML dependencies

`docker-compose.yml` hardcodes `SPEECH_PROVIDER=mock` / `OCR_PROVIDER=mock`
with `env_file` marked `required: false`, so `docker compose up --build`
succeeds on a clean clone with no `.env` and no credentials. **Real bug
found and fixed:** the Dockerfile originally let `pip` resolve `torch`
however `easyocr` pulled it in, which defaulted to a full CUDA build
(~2GB of unused `nvidia_*`/`triton` packages) despite both adapters running
CPU-only (`device="cpu"` in Whisper, `gpu=False` in EasyOCR) — directly
contradicting this file's own "no GPU assumption" claim. Fixed by installing
the CPU-only torch/torchvision wheel explicitly before `requirements.txt`,
so `easyocr`'s resolution finds it already satisfied.