from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.v1.router import router as v1_router
from core.config import settings
from core.exceptions import DomainError, FileTooLargeError, ProviderError, UnsupportedFormatError

app = FastAPI(title="Speech & Document Extraction Service")

app.include_router(v1_router)


# Maps each specific DomainError subclass to the HTTP status code that
# actually fits it. This dict is the single source of truth for that
# mapping — nowhere else in the codebase should be deciding status codes.
_STATUS_CODE_MAP: dict[type[DomainError], int] = {
    UnsupportedFormatError: 400,
    FileTooLargeError: 413,
    ProviderError: 502,
}


@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    """Catches every DomainError raised anywhere in services/ or adapters/
    and converts it into a consistent JSON error response. This is the
    ONLY place in the whole codebase that maps our domain exceptions to
    HTTP concepts — routes themselves never do this.
    """
    status_code = _STATUS_CODE_MAP.get(type(exc), 500)  # unrecognized DomainError subclass -> 500, not a guess at 400
    return JSONResponse(
        status_code=status_code,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — also confirms config loaded correctly."""
    return {
        "status": "ok",
        "speech_provider": settings.speech_provider,
        "ocr_provider": settings.ocr_provider,
    }