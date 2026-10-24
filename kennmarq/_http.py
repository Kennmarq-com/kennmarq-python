"""HTTP for the SDK, with the standard library only: the base install depends on nothing."""
from __future__ import annotations

import json
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ._version import __version__
from .errors import ContentBlocked, KennmarqError

USER_AGENT = f"kennmarq-python/{__version__}"


@dataclass(frozen=True)
class Reply:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


def multipart(fields: dict[str, Any]) -> tuple[bytes, str]:
    """multipart/form-data: a str value is a field, a (filename, bytes, content type) tuple a file."""
    boundary = "kennmarq-" + secrets.token_hex(12)
    parts: list[bytes] = []
    for name, value in fields.items():
        if value is None:
            continue
        if isinstance(value, tuple):
            filename, data, content_type = value
            head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n")
            parts.append(head.encode() + data + b"\r\n")
        else:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # an API call never follows a redirect elsewhere
        return None


# A 429 that says to come back within a minute (the plan's calls a minute, for all of an organisation's keys
# together; a clip waiting for a free slot) is waited out, at most this many times. A longer wait (a day's
# cap) raises at once.
RETRIES_429 = 3
RETRY_429_MAX_S = 61


def _retry_after(reply: "Reply") -> int | None:
    value = reply.headers.get("retry-after", "")
    return int(value) if value.isdigit() else None


class Http:
    def __init__(self, base_url: str, api_key: str | None, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect)

    def request(self, method: str, path: str, *, body: bytes | None = None, content_type: str | None = None,
                query: dict[str, str] | None = None, auth: bool = True) -> Reply:
        url = self.base_url + path + ("?" + urllib.parse.urlencode(query) if query else "")
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json, */*"}
        if content_type:
            headers["Content-Type"] = content_type
        if auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(RETRIES_429 + 1):
            request = urllib.request.Request(url, data=body, method=method, headers=headers)
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    reply = Reply(response.status, {k.lower(): v for k, v in response.headers.items()}, response.read())
            except urllib.error.HTTPError as error:
                reply = Reply(error.code, {k.lower(): v for k, v in error.headers.items()}, error.read())
            except (urllib.error.URLError, OSError) as error:
                raise KennmarqError(f"couldn't reach {self.base_url}: {error}") from error
            wait = _retry_after(reply) if reply.status == 429 else None
            if wait is None or wait > RETRY_429_MAX_S or attempt == RETRIES_429:
                break
            time.sleep(wait)
        if reply.status >= 400:
            raise _error(reply)
        return reply

    def get_url(self, url: str) -> bytes:
        """GET a download link (a clip's marked file, presigned). Never with our key: the link is the permission."""
        request = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            raise KennmarqError(f"the download was refused (HTTP {error.code})", status=error.code) from error
        except (urllib.error.URLError, OSError) as error:
            raise KennmarqError(f"couldn't download the file: {error}") from error

    def put_upload(self, url: str, body: bytes, headers: dict[str, str]) -> None:
        """PUT a file to an upload's presigned URL (S3). Never with our key: the URL is the permission."""
        request = urllib.request.Request(url, data=body, method="PUT", headers={"User-Agent": USER_AGENT, **headers})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
        except (urllib.error.URLError, OSError) as error:
            raise KennmarqError(f"couldn't upload the file: {error}") from error
        if status >= 300:
            raise KennmarqError(f"the upload was refused (HTTP {status})", status=status)


def _error(reply: Reply) -> KennmarqError:
    try:
        body = reply.json()
    except ValueError:
        body = None
    detail = body.get("error") if isinstance(body, dict) else None
    detail = detail if isinstance(detail, dict) else {}
    message = str(detail.get("message") or f"HTTP {reply.status}")
    kwargs = {"status": reply.status, "code": detail.get("code"), "request_id": detail.get("request_id") or reply.headers.get("x-request-id"),
              "body": body}
    if detail.get("code") == "content_blocked" and isinstance(body.get("safety"), dict):
        return ContentBlocked(message, body["safety"], **kwargs)
    return KennmarqError(message, **kwargs)
