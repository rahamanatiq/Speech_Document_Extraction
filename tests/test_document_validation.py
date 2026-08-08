import io


def test_document_rejects_unsupported_format(client):
    file = io.BytesIO(b"fake content")
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("report.pdf", file, "application/pdf")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "UnsupportedFormatError"
    assert ".pdf" in body["detail"]


def test_document_rejects_oversized_file(client):
    huge_content = b"x" * (16 * 1024 * 1024)  # exceeds MAX_IMAGE_BYTES (15 MB)
    file = io.BytesIO(huge_content)
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("report.jpg", file, "image/jpeg")},
    )

    assert response.status_code == 413
    body = response.json()
    assert body["error"] == "FileTooLargeError"


def test_document_happy_path_returns_expected_shape(client):
    file = io.BytesIO(b"fake image bytes")
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("report.jpg", file, "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["raw_text"], str)
    assert isinstance(body["meta"], dict)
    assert isinstance(body["results"], list)
    assert len(body["results"]) > 0


def test_document_never_guesses_ambiguous_ocr_value(client):
    """The single most important behavioral test in this project: proves,
    through a real HTTP request, that an OCR-garbled value ('l2.S') never
    gets silently resolved into a fabricated number, all the way out to
    the API response.
    """
    file = io.BytesIO(b"fake image bytes")
    response = client.post(
        "/api/v1/documents/extract",
        files={"file": ("report.jpg", file, "image/jpeg")},
    )

    body = response.json()
    wbc_line = next(line for line in body["results"] if line["test_name"] == "WBC Count")

    assert wbc_line["value"] is None
    assert wbc_line["normalized"]["value_type"] == "unparseable"
    assert wbc_line["normalized"]["numeric_value"] is None
    # The raw OCR text must still be there, untouched, even though we
    # couldn't confidently interpret it.
    assert "l2.S" in wbc_line["raw_line"]