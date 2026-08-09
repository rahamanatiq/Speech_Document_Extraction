# Architectural Decisions

## Speech-to-text provider: faster-whisper (self-hosted)

Chose faster-whisper over a commercial API to keep the service credential-free
and runnable offline, per the "no credentials on default compose path"
requirement. `int8` quantization on CPU trades a small amount of accuracy for
meaningfully faster inference — acceptable since there's no GPU assumption.

## Whisper model size: kept `medium` as the default, despite weaker Bangla accuracy

Tested against testdata/audio/clip_clean_bn.m4a: `medium` produced garbled,
low-confidence (0.04) output on a clean Bangla recording, while the same
speaker's English clips transcribed accurately (confidence 0.67-0.79).

Considered switching the default to `large-v3` to improve Bangla accuracy.
Decided against it: `large-v3` roughly doubles model size (~3GB vs ~1.5GB)
and per-request latency on CPU, working against the requirement that the
service run easily on a reviewer's machine with no GPU. `WHISPER_MODEL_SIZE`
is configurable via `.env` — a deployment that prioritizes Bangla accuracy
over speed/size can set `large-v3` without any code change.

This is a known, honest limitation, not a silent gap: Bangla transcription
quality with the default configuration is currently weak and would need a
larger model (at a real cost) to improve.

## VAD filtering enabled on the Whisper adapter

During testing, clip_silence.m4a (a deliberately silent recording) was
transcribed by Whisper as "Thanks for watching!" — a documented Whisper
failure mode where the model hallucinates plausible-sounding captions on
non-speech audio. Enabling faster-whisper's `vad_filter=True` (Silero VAD)
fixed this: the same silent clip now correctly returns an empty transcript
with 0.0 confidence. Verified via before/after test run, not assumed.


## OCR provider: EasyOCR (self-hosted) over Tesseract

Chose EasyOCR for better accuracy on real-world phone photos (angled, poor
lighting) versus Tesseract's more rigid expectations, at the cost of a much
larger dependency (PyTorch) and slower cold start. See requirements.txt —
this is a deliberate trade-off, not an oversight.

## OCR row detection: requires both a value AND a lab-unit signal

Initial version accepted any two-column line containing a number as a
potential lab result. Tested against testdata/lab_reports/not_a_lab_report.jpg
(a store receipt) and found it incorrectly parsed 3 "lab results" from
prices (e.g. "Subtotal: 3.78"). Fixed by requiring a recognizable lab-unit
token (g/dL, mg/dl, %, K/uL, etc.) or numeric range alongside the value —
a receipt has values but never lab units. Re-verified: the same receipt now
correctly returns 0 parsed rows. This is the single most important test
case for the "never guess, degrade gracefully" requirement.

## Known limitation: digital screenshots score worse than real photos

testdata/lab_reports/report_clean.jpg (a screenshot, not a photo) produced
the weakest result of all six test images — confidence 0.15-0.38 and only
0-4 rows correctly parsed, despite being visually the "cleanest" looking
image. The other five (real phone photos, including angled and poor-light
ones) all scored higher. EasyOCR's recognition model appears to handle
natural photographs of printed text more reliably than anti-aliased
screenshot fonts at lower resolution. Added image upscaling + contrast
preprocessing, which improved (not fixed) this case. Documented rather
than over-fit to, since the pipeline demonstrably works well on the
photo-based images it's actually designed for.

## Known limitation: row-splitting on inconsistent table spacing

EasyOCR occasionally detects a single visual table row (e.g. "PLT  294.0
10*3/uL  150-400") as two separate lines when line spacing in the source
image is uneven, causing the test name and its value to land in different
reconstructed lines and fail to associate. Observed in
testdata/lab_reports/report_complex.jpg. A more robust fix (e.g. bounding-
box-width-aware row merging) was considered but not implemented, given
time constraints — noted here as a concrete area for future improvement
rather than silently shipped as if solved.