import os

# Force mock providers for the entire test session, before anything
# imports main.py (which imports core.config, which builds the `settings`
# singleton at import time). This must happen before any other import in
# this file — import order is load-bearing here, not just style.
os.environ["SPEECH_PROVIDER"] = "mock"
os.environ["OCR_PROVIDER"] = "mock"

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client() -> TestClient:
    """A FastAPI test client — makes real-shaped HTTP requests against the
    app in-memory, no running server required. Shared across every test
    file in tests/, via pytest's automatic conftest.py discovery.
    """
    return TestClient(app)