# Speech_Document_Extraction

An AI service exposing two capabilities behind a FastAPI HTTP API:

- **Transcription** — audio in English or Bangla → text
- **Document extraction** — a photographed English-language lab report → structured data

Both endpoints run against pluggable provider adapters (real model or mock),
selected entirely by configuration — see `DECISIONS.md` for why each real
provider was chosen and what was rejected.

## How to run it

### Docker (recommended — matches the reviewer's path)

```bash
docker compose up --build
```

No `.env` file and no credentials are required. `docker-compose.yml`
hardcodes both providers to `mock`, so this brings the service up fully
functional, offline, with zero setup. Confirm it's up:

```bash
curl http://localhost:8000/health
# {"status":"ok","speech_provider":"mock","ocr_provider":"mock"}
```

To use the real providers (faster-whisper / EasyOCR) instead of mocks,
copy `.env.example` to `.env`, set `SPEECH_PROVIDER=whisper` and/or
`OCR_PROVIDER=easyocr`, and re-run `docker compose up --build`. Real
providers are opt-in only — never on the default path.

### Local (for development / running tests)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

(On macOS/Linux, activate with `source .venv/bin/activate` instead.)

### Tests

```bash
pytest tests/
```

Runs the fast suite only (54 tests, pure-function + HTTP-level tests
against mock providers, under 1 second) — this is the default via
`pytest.ini`'s `addopts = -m "not slow"`. To run the 7 slow tests that load
the actual Whisper/EasyOCR models against real files in `testdata/`
(several minutes, meaningful RAM/disk required):

```bash
pytest tests/ -m slow
```

## Architecture

Three layers, dependencies pointing inward only:
api/ HTTP routing, request/response models, status codes — no business logic
services/ orchestration: validation, provider selection, normalization
adapters/ provider/model integration — the only place a provider SDK is imported


`services/` depends only on the abstract interfaces in `adapters/*/base.py`
(`TranscriptionProvider`, `OCRProvider`) — never a concrete adapter class —
so swapping a provider touches only `adapters/`. Every service raises one of
three domain exceptions (`UnsupportedFormatError`, `FileTooLargeError`,
`ProviderError`, defined in `core/exceptions.py`); a single handler in
`main.py` maps each to an HTTP status code, so no route file contains a
`try/except`. Full rationale for the more consequential calls in
`DECISIONS.md`.

## Endpoints

### `POST /api/v1/transcribe`

Multipart upload: an audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, ≤25MB) plus
a `language` field (`en`, `bn`, or `auto`). Returns transcript, resolved
language, duration, confidence, and which provider produced it.

### `POST /api/v1/documents/extract`

Multipart upload: a lab report image (`.jpg`, `.jpeg`, `.png`, `.webp`,
≤15MB). Returns:

```json
{
  "raw_text": "...",
  "meta": {
    "patient_name": "...", "age": "...", "sex": "...",
    "report_date": "...", "lab_name": "...", "reference_no": "..."
  },
  "results": [
    {
      "raw_line": "...", "test_name": "...", "value": "...",
      "unit": "...", "reference_range": "...", "flag": "...",
      "normalized": { "raw": "...", "value_type": "...", "numeric_value": ... },
      "normalized_unit": { "raw": "...", "unit_type": "...", "canonical": "..." }
    }
  ]
}
```

`meta` fields are populated only when a clear label was found in the source
(e.g. `"Age: 45"`) — an unlabeled or ambiguous field is simply omitted, not
guessed. `report_date` is replaced with canonical ISO form (`YYYY-MM-DD`)
when confidently parsed, with the original preserved alongside it as
`report_date_raw`.

## Normalized formats

**Values** (`normalize_value`, `services/normalizer.py`): parsed into one
of `exact`, `less_than`, `greater_than`, `range`, or `unparseable`. Handles
plain numbers (`13.5`), comparators (`<0.5`, `>100`), ranges (`0.8-1.2`),
lab-style scientific notation (`1.2 x 10^3`), and comma-thousands
separators (`12,500` → `12500.0`, but only when the comma sits directly
between two digits — a trailing or orphaned comma is left to fail honestly
rather than being "repaired").

**Units** (`normalize_unit`): known spelling variants map to one canonical
form via a lookup table — e.g. `gm/dl`, `g/dl`, `G/DL` → `g/dL`; `K/µL` and
`10^3/uL` are treated as clinically equivalent and merged to `10^3/uL`.
Anything not in the table is returned `unparseable`, preserved verbatim —
this list is a starting set based on our test data's units, not exhaustive.

**Dates** (`normalize_date`): tried against a fixed list of known formats
(ISO, `DD-MM-YYYY`, `DD/MM/YYYY`, `DD Mon YYYY`, `Mon DD, YYYY`, etc.),
normalized to ISO `YYYY-MM-DD` on a confident match. **Known ambiguity,
deliberately not resolved by guessing:** an all-numeric slash format like
`03/04/2026` is genuinely ambiguous (day-first vs month-first) — this
service only accepts day-first (`%d/%m/%Y`), matching the convention on our
test data. A month-first date will fail to parse and fall back to verbatim
rather than being guessed at.

**Every normalizer follows the same rule:** attempt known patterns; if none
match confidently, preserve the raw input untouched and mark it
unparseable. This applies at the raw-line level (`LabResultLine.raw_line`,
always kept regardless of parse success), the value level, the unit level,
and the date level — see `DECISIONS.md` §4 for the one place this
collides with another stated requirement in the brief, and why we sided
the way we did.

## Test data

`testdata/audio/` — four real recordings (clean English, clean Bangla,
English with background noise, deliberate silence), each with a
manually-verified ground-truth transcript in `transcripts.json`, chosen to
exercise the noisy/silent edge cases the brief calls out specifically.

`testdata/lab_reports/` — six real images: clean, angled, poor-light,
cropped, and a dense multi-section report, plus one deliberately unrelated
document (a store receipt) to test graceful degradation on non-lab input.
Several images originally contained real patient names/ages/record
numbers — these were manually cropped to remove all identifying
information before being committed to this public repository.

## Known limitations

- **Bangla transcription accuracy is weak** on the default `medium` Whisper
  model (confirmed via a clean test recording); `WHISPER_MODEL_SIZE` is
  configurable to trade size/speed for accuracy. See `DECISIONS.md` §2.
- **Digital screenshots score worse than real photos** in EasyOCR —
  counter-intuitively, our cleanest-looking image (`report_clean.jpg`, a
  screenshot) produced the weakest OCR result of the set, while genuinely
  angled/poor-light phone photos scored higher. Preprocessing
  (upscaling + contrast) improved but didn't fully resolve this.
- **Row-grouping can non-deterministically split a test name from its
  value** when source line spacing is uneven — observed and documented in
  `DECISIONS.md` §3; the affected test was rewritten to assert what's
  reliably true rather than a guarantee that doesn't hold every run.
- **Meta-field extraction is label-based and best-effort** — a field is
  only populated when the source has an explicit label (`"Age:"`,
  `"Sex:"`, etc.); reports that present this information without a label
  (e.g. inline in a sentence) will have that field simply absent from
  `meta`, not guessed at.
- **Date parsing only accepts day-first numeric formats** (not
  month-first) — a genuine, disclosed ambiguity rather than a guess; see
  "Normalized formats" above.
- **The slow test suite** (`pytest tests/ -m slow`) loads real Whisper/
  EasyOCR models and needs meaningful free RAM/disk — it may fail with an
  allocation error on constrained machines. This doesn't affect the fast
  suite, which is what runs by default and what CI/reviewers are expected
  to run.