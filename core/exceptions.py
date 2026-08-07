class DomainError(Exception):
    """Base class for all errors raised by services/ or adapters/.

    api/ catches these and translates them into HTTP responses — this is
    the mechanism that keeps HTTP concerns out of the business logic layer.
    """


class UnsupportedFormatError(DomainError):
    """Raised when an uploaded file's type isn't one we support."""


class FileTooLargeError(DomainError):
    """Raised when an uploaded file exceeds the configured size limit."""


class ProviderError(DomainError):
    """Raised when a provider (real or mock) fails internally.

    Adapters wrap raw provider/library exceptions in this, so services/
    and api/ only ever need to know about one error type, not every
    possible exception a specific SDK might throw.
    """