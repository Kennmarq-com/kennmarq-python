"""What the SDK raises. API errors carry the API's code and request ID (docs/errors)."""
from __future__ import annotations

from typing import Any


class KennmarqError(Exception):
    """An API error, or local mode failing closed."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None,
                 request_id: str | None = None, body: Any = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.request_id = request_id
        self.body = body


class ContentBlocked(KennmarqError):
    """The safety check blocked a picture: nothing was marked. `safety` holds the check's answer."""

    def __init__(self, message: str, safety: dict, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.safety = safety


class MissingApiKey(KennmarqError):
    """No API key: set KENNMARQ_API_KEY, or pass api_key=. A free key marks 1,000 files a month."""


class LocalModeUnavailable(KennmarqError):
    """Local mode needs `pip install "kennmarq[local]"` (and "kennmarq[safety]" for a local check)."""


class InvalidSignedFile(KennmarqError):
    """The signed file didn't validate with Kennmarq.com's roots: it was not written (fail closed)."""
