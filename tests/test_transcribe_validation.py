import io

def test_transcribe_rejects_unsupported_format(client):
    file = io.BytesIO(b"fake content")
    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("clip.png", file, "image/png")},
        data={"language": "auto"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "UnsupportedFormatError"
    assert ".png" in body["detail"]


def test_transcribe_rejects_oversized_file(client):
    huge_content = b"x" * (21 * 1024 * 1024)  # exceeds MAX_AUDIO_BYTES (20 MB)
    file = io.BytesIO(huge_content)
    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("clip.wav", file, "audio/wav")},
        data={"language": "auto"},
    )

    assert response.status_code == 413
    body = response.json()
    assert body["error"] == "FileTooLargeError"


def test_transcribe_happy_path_returns_expected_shape(client):
    file = io.BytesIO(b"fake audio bytes")
    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("clip.wav", file, "audio/wav")},
        data={"language": "en"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "en"
    assert isinstance(body["raw_text"], str)
    assert isinstance(body["duration_seconds"], float)
    assert "confidence" in body


def test_transcribe_handles_silent_audio_edge_case(client):
    # Empty bytes is how our mock represents "no speech detected" — the
    # brief explicitly calls out silence as an edge case to handle.
    file = io.BytesIO(b"")
    response = client.post(
        "/api/v1/transcribe",
        files={"file": ("silence.wav", file, "audio/wav")},
        data={"language": "en"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["raw_text"] == ""
    assert body["confidence"] == 0.0