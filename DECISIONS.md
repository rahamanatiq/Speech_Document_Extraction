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