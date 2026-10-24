"""The Kennmarq.com client: the cloud API, and local mode, where files never leave this machine."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Union

from ._http import Http, multipart
from ._version import __version__
from .errors import ContentBlocked, InvalidSignedFile, KennmarqError, LocalModeUnavailable, MissingApiKey

DEFAULT_BASE_URL = "https://api.kennmarq.com"
BRAND = "Kennmarq.com"
BRAND_SLUG = "kennmarq"
SOURCES = ("created", "composite", "edited")
#: Pictures: keep, png, jpeg, webp. Songs: keep, mp3, wav, m4a. Clips: keep, mp4.
OUTPUT_FORMATS = ("keep", "png", "jpeg", "webp", "mp3", "wav", "m4a", "mp4")
PICTURE_FORMATS = ("keep", "png", "jpeg", "webp")
CHECKS = ("off", "ai-act", "safe-only")
#: The visible label (release C3): the word beside "AI", the corner, the size and the style.
LABEL_LANGUAGES = ("en", "ko", "zh")
LABEL_POSITIONS = ("top-right", "top-left", "bottom-right", "bottom-left")
LABEL_SIZES = ("small", "medium", "large")
LABEL_STYLES = ("auto", "dark", "light", "dark-translucent", "light-translucent")
#: A request carries at most 4 MB; larger files (up to 25 MB, a clip to mark up to 1 GB) go through
#: POST /v1/uploads first.
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_CLIP_BYTES = 1024 * 1024 * 1024
#: How long mark() waits for a clip's job, and how often it asks.
CLIP_TIMEOUT = 40 * 60
POLL_SECONDS = 5
#: Room left in a request for the options and the multipart framing.
_FRAMING_BYTES = 64 * 1024

FileLike = Union[str, os.PathLike, bytes]


@dataclass(frozen=True)
class MarkResult:
    """A marked file: its bytes, and what the API said about it."""

    data: bytes
    media_type: str
    mark_id: str | None
    watermark_id: str | None
    log_index: int | None
    marks_used: int | None
    marks_limit: int | None
    #: With a safety check: {"verdict": "allow" | "flag", "reason": ...}.
    safety: dict | None
    #: "cloud" (POST /v1/mark), "local" (made here, signed by POST /v1/sign), "hybrid" (watermarked here,
    #: signed in the cloud, where c2pa has no build), or "watermark-only" (no key: no manifest).
    mode: str
    path: Path | None = None
    #: "picture", "song" or "clip".
    kind: str = "picture"
    #: A clip's job: "completed", or "running" when mark(wait=False) returned before it was done.
    status: str = "completed"
    #: The marks this file counted: a song five per started minute, a clip five per started second.
    marks_counted: int | None = None
    #: A song's or clip's mark in the sound: "claimed" (it reads back, and the manifest says so),
    #: "unclaimed" (too quiet or short to read back) or "off".
    audio_watermark: str | None = None


def default_models_dir() -> Path:
    return Path(os.environ.get("KENNMARQ_MODELS_DIR") or Path.home() / ".cache" / "kennmarq" / "models")


def media_kind(data: bytes) -> str | None:
    """A file's kind from its first bytes, as the API reads it: "picture" (PNG, JPEG, WebP), "song"
    (MP3, WAV, M4A) or "clip" (MP4 or QuickTime), or None."""
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff") or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return "picture"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and head[1] & 0x06):
        return "song"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "song"
    if head[4:8] == b"ftyp":
        return "song" if head[8:12] in (b"M4A ", b"M4B ") else "clip"
    return None


def _read(file: FileLike) -> tuple[bytes, str]:
    if isinstance(file, (bytes, bytearray)):
        return bytes(file), "upload"
    path = Path(file)
    return path.read_bytes(), path.name


def with_png_text(png: bytes, text: dict[str, str]) -> bytes:
    """tEXt chunks right after IHDR (ComfyUI's prompt and workflow, as its Save Image writes them).
    Added before signing, so the manifest's hard binding covers them."""
    import struct
    import zlib

    if not text or not png.startswith(b"\x89PNG\r\n\x1a\n"):
        return png
    end = 8 + 12 + struct.unpack(">I", png[8:12])[0]
    chunks = b""
    for key, value in text.items():
        data = key.encode("latin-1") + b"\x00" + value.encode("latin-1", errors="replace")
        chunks += struct.pack(">I", len(data)) + b"tEXt" + data + struct.pack(">I", zlib.crc32(b"tEXt" + data) & 0xFFFFFFFF)
    return png[:end] + chunks + png[end:]


def _int(value: str | None) -> int | None:
    return int(value) if value and value.isdigit() else None


def _app_version_needs_app_name(app_name: str | None, app_version: str | None) -> None:
    if app_version and not app_name:
        raise KennmarqError("app_version needs app_name: it is that app's version")


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    return value


class Kennmarq:
    """Marks AI pictures, songs and clips for the EU AI Act: an invisible watermark and signed,
    time-stamped C2PA Content Credentials.

    - Cloud: ``mark()`` sends the file to the API and returns it marked. A clip is marked as a job:
      ``mark()`` waits for it and downloads the marked clip (``wait=False`` returns at once, and
      ``wait_for()`` finishes the job later).
    - Local: ``mark_file()`` makes the watermark and the manifest on this machine; only the manifest
      (about 13 KB, never pixels) goes to the API, to be signed. Needs ``pip install "kennmarq[local]"``.
    - ``check()``: the safety check, in the cloud, or here with ``local=True`` (``kennmarq[safety]``).
    - ``verify()``: what a file's Content Credentials and watermark say; free, no key needed.

    The key comes from ``api_key=`` or the KENNMARQ_API_KEY environment variable.
    """

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None, models_dir: str | os.PathLike | None = None,
                 timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("KENNMARQ_API_KEY") or None
        self.base_url = (base_url or os.environ.get("KENNMARQ_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self._http = Http(self.base_url, self.api_key, timeout)
        self._lock = threading.Lock()
        self._certificates: dict | None = None
        self._watermarker = None
        self._safety = None

    # -- the cloud API ---------------------------------------------------------------------------------

    def _need_key(self) -> None:
        if not self.api_key:
            raise MissingApiKey("no API key: set KENNMARQ_API_KEY or pass api_key=. A free key marks 1,000 files a month.")

    def upload(self, data: bytes) -> str:
        """POST /v1/uploads, then the file PUT to the URL it answers: the upload's ID, for one call
        within the hour, with this key. The cloud calls do this themselves for files over 4 MB."""
        self._need_key()
        if not 0 < len(data) <= MAX_CLIP_BYTES:
            raise KennmarqError("files are 1 byte to 25 MB, and a clip to mark up to 1 GB")
        import json

        made = self._http.request("POST", "/v1/uploads", body=json.dumps({"bytes": len(data)}).encode(),
                                  content_type="application/json").json()
        self._http.put_upload(made["url"], data, made["headers"])
        return made["id"]

    def _pictures(self, files: dict[str, tuple[bytes, str] | None]) -> dict[str, Any]:
        """Multipart fields for a call's pictures: each sent in the request, or, when they wouldn't fit
        in its 4 MB, uploaded first and named by ID ("file" → "upload", "input" → "input_upload")."""
        present = {role: f for role, f in files.items() if f is not None}
        total = sum(len(data) for data, _name in present.values()) + _FRAMING_BYTES
        fields: dict[str, Any] = {}
        for role, (data, name) in present.items():
            if total > MAX_REQUEST_BYTES:
                fields["upload" if role == "file" else f"{role}_upload"] = self.upload(data)
            else:
                fields[role] = (name, data, "application/octet-stream")
        return fields

    def mark(self, file: FileLike, dest: str | os.PathLike | None = None, *, source: str = "created", model: str | None = None,
             model_version: str | None = None, app_name: str | None = None, title: str | None = None, watermark: bool = True,
             provider: str | None = None, app_version: str | None = None, markets: str | Sequence[str] | None = None,
             iptc: bool = True, output_format: str = "keep", keep_input_credentials: bool = True, input: FileLike | None = None,
             check: str = "off", deepfake: bool = False, label: bool | None = None, label_language: str | None = None,
             label_position: str | None = None, label_size: str | None = None, label_style: str | None = None,
             wait: bool = True, timeout: float = CLIP_TIMEOUT) -> MarkResult:
        """POST /v1/mark: the picture, song or clip marked in the cloud. A safety block raises
        ContentBlocked. A clip is a job: this waits for it (up to `timeout` seconds) and downloads the
        marked clip; with wait=False it returns at once with status "running" and no data.

        `markets` ("eu", "us-ca", "us-wa", "us-ut" or "us" for all three states, "kr", "in") makes the API
        refuse a call missing what those laws need, and gives the mark a receipt: receipt(mark_id).

        `label=True` draws the EU's visible label ("AI GENERATED", or "AI MODIFIED" for edits) into a
        picture or every frame of a clip; `deepfake=True` turns it on, and India ("in") always needs it.
        `label_language` ("en", "ko", "zh"), `label_position`, `label_size` and `label_style` shape it."""
        self._need_key()
        _app_version_needs_app_name(app_name, app_version)
        fields: dict[str, Any] = {
            "source": _choice("source", source, SOURCES),
            "provider": provider, "model": model, "model_version": model_version, "app_name": app_name,
            "app_version": app_version, "title": title,
            "markets": markets if markets is None or isinstance(markets, str) else ",".join(markets),
            "watermark": "true" if watermark else "false",
            "iptc": "true" if iptc else "false",
            "output_format": _choice("output_format", output_format, OUTPUT_FORMATS),
            "keep_input_credentials": "true" if keep_input_credentials else "false",
            "check": _choice("check", check, CHECKS),
            "deepfake": "true" if deepfake else None,
            "label": None if label is None else "true" if label else "false",
            "label_language": label_language and _choice("label_language", label_language, LABEL_LANGUAGES),
            "label_position": label_position and _choice("label_position", label_position, LABEL_POSITIONS),
            "label_size": label_size and _choice("label_size", label_size, LABEL_SIZES),
            "label_style": label_style and _choice("label_style", label_style, LABEL_STYLES),
        }
        data, name = _read(file)
        fields.update(self._pictures({"file": (data, name), "input": _read(input) if input is not None else None}))
        body, content_type = multipart(fields)
        reply = self._http.request("POST", "/v1/mark", body=body, content_type=content_type)
        h = reply.headers
        if reply.status == 202:
            job = reply.json()
            if not wait:
                return MarkResult(b"", "video/mp4", job["id"], None, None, _int(h.get("x-marks-used")), _int(h.get("x-marks-limit")),
                                  None, "cloud", kind="clip", status="running")
            return self.wait_for(job["id"], dest, timeout=timeout)
        safety = {"verdict": h["x-safety-verdict"], "reason": h.get("x-safety-reason")} if h.get("x-safety-verdict") else None
        media_type = h.get("content-type", "").split(";")[0]
        result = MarkResult(reply.body, media_type, h.get("x-mark-id"), h.get("x-watermark-id"),
                            _int(h.get("x-log-index")), _int(h.get("x-marks-used")), _int(h.get("x-marks-limit")), safety, "cloud",
                            kind="song" if media_type.startswith("audio/") else "picture",
                            marks_counted=_int(h.get("x-marks-counted")) or 1, audio_watermark=h.get("x-audio-watermark"))
        return self._write(result, dest)

    def get_mark(self, mark_id: str) -> dict:
        """GET /v1/marks/{id}: a mark's status; a clip's job, with a download link once it's done."""
        self._need_key()
        return self._http.request("GET", f"/v1/marks/{mark_id}").json()

    def receipt(self, mark_id: str) -> dict:
        """GET /v1/marks/{id}/receipt: the receipt of a mark made with `markets`, signed with the checker
        key: {"receipt", "jws", "key_id", "public_key"}."""
        self._need_key()
        return self._http.request("GET", f"/v1/marks/{mark_id}/receipt").json()

    def wait_for(self, mark_id: str, dest: str | os.PathLike | None = None, *, timeout: float = CLIP_TIMEOUT) -> MarkResult:
        """Waits for a clip's job and downloads the marked clip. A job that failed raises its error
        (ContentBlocked for a blocked frame, with the frame in `safety`)."""
        deadline = time.monotonic() + timeout
        while True:
            mark = self.get_mark(mark_id)
            if mark["status"] != "running":
                break
            if time.monotonic() > deadline:
                raise KennmarqError(f"the clip's job {mark_id} is still running; ask again with wait_for()", code="still_running")
            time.sleep(POLL_SECONDS)
        if mark["status"] == "failed":
            error = mark.get("error") or {}
            if error.get("code") == "content_blocked":
                raise ContentBlocked(error.get("message", "blocked"), mark.get("safety") or {}, code="content_blocked")
            raise KennmarqError(error.get("message", "the clip's job failed"), code=error.get("code"))
        download = mark.get("download") or {}
        if not download.get("url"):
            # Why there's no link: gone (kept a day), download_limit (5 links), quota_exceeded (the month's data).
            error = mark.get("download_error") or {}
            raise KennmarqError(error.get("message", f"{mark_id}'s file can't be downloaded"), code=error.get("code", "not_found"))
        data = self._http.get_url(download["url"])
        file = mark.get("file") or {}
        if len(data) != file.get("bytes"):
            raise KennmarqError("the downloaded clip isn't the size its mark says; download it again")
        audio = ((mark.get("watermarks") or {}).get("audio") or None)
        result = MarkResult(data, file.get("type", "video/mp4"), mark_id, mark.get("watermark_id"), mark.get("log_index"),
                            None, None, mark.get("safety"), "cloud", kind=mark.get("kind", "clip"), marks_counted=mark.get("marks"),
                            audio_watermark=("claimed" if audio["claimed"] else "unclaimed") if audio else "off")
        return self._write(result, dest)

    def check(self, file: FileLike, policy: str = "ai-act", *, local: bool = False) -> dict:
        """The safety check of a picture: POST /v1/check (two marks), or here with local=True (nothing
        sent, nothing counted). A clip (up to 25 MB) is checked in the cloud on 12 frames, two marks each."""
        _choice("policy", policy, ("ai-act", "safe-only"))
        data, name = _read(file)
        if local:
            if media_kind(data) != "picture":
                raise KennmarqError("the local safety check reads pictures; check a clip in the cloud (local=False)")
            return self._check_locally(data, policy)
        self._need_key()
        body, content_type = multipart({**self._pictures({"file": (data, name)}), "policy": policy})
        return self._http.request("POST", "/v1/check", body=body, content_type=content_type).json()

    def verify(self, file: FileLike, *, signed: bool = False) -> dict:
        """POST /v1/verify: what the file's Content Credentials and watermarks say. Free; a picture up to
        4 MB needs no key. A larger file goes through an upload, and a song or clip is read on the
        server: both take a key (counting nothing)."""
        data, name = _read(file)
        pictures = self._pictures({"file": (data, name)}) if len(data) + _FRAMING_BYTES > MAX_REQUEST_BYTES else \
            {"file": (name, data, "application/octet-stream")}
        body, content_type = multipart({**pictures, "signed": "true" if signed else None})
        keyed = "upload" in pictures or media_kind(data) in ("song", "clip")
        return self._http.request("POST", "/v1/verify", body=body, content_type=content_type, auth=keyed).json()

    def certificate(self) -> dict:
        """GET /v1/certificate: Kennmarq.com's public chains (live, test, TSA), roots included."""
        with self._lock:
            if self._certificates is None:
                self._certificates = self._http.request("GET", "/v1/certificate", auth=False).json()
            return self._certificates

    def usage(self) -> dict:
        """GET /v1/usage: marks and data used this month."""
        self._need_key()
        return self._http.request("GET", "/v1/usage").json()

    def sign_store(self, store: bytes) -> tuple[bytes, dict[str, str]]:
        """POST /v1/sign: a local manifest store, signed by Kennmarq.com (what mark_file() uses)."""
        self._need_key()
        reply = self._http.request("POST", "/v1/sign", body=store, content_type="application/c2pa")
        return reply.body, reply.headers

    # -- local mode ----------------------------------------------------------------------------------------------

    def _local_modules(self):
        try:
            from .local import imaging, manifest, throwaway, watermark
        except ImportError as exc:
            raise LocalModeUnavailable(f'local mode needs `pip install "kennmarq[local]"`: {exc}') from exc
        return imaging, manifest, throwaway, watermark

    def _watermarker_for(self, watermark_module):
        with self._lock:
            if self._watermarker is None:
                self._watermarker = watermark_module.Watermarker(self.models_dir)
            return self._watermarker

    def _check_locally(self, data: bytes, policy: str) -> dict:
        imaging = self._local_modules()[0]
        try:
            from .local.safety import SafetyChecker, upright
        except ImportError as exc:
            raise LocalModeUnavailable(f'a local safety check needs `pip install "kennmarq[safety]"`: {exc}') from exc
        with self._lock:
            if self._safety is None:
                self._safety = SafetyChecker(self.models_dir, brand=BRAND, version=__version__)
        return self._check_picture(imaging, upright, data, policy)

    def _check_picture(self, imaging, upright, data: bytes, policy: str) -> dict:
        import hashlib

        media_type = imaging.sniff(data)
        if media_type is None:
            raise KennmarqError("the file isn't a PNG, JPEG or WebP picture")
        picture = imaging.decode(data, media_type)
        file = {"type": media_type, "bytes": len(data), "width": picture.image.width, "height": picture.image.height,
                "sha256": hashlib.sha256(data).hexdigest()}
        return self._safety.check(upright(picture.rgb(), picture.exif), policy, file)

    def watermark_file(self, file: FileLike, dest: str | os.PathLike | None = None, *, source: str = "created",
                       iptc: bool = True, output_format: str = "keep", png_text: dict[str, str] | None = None) -> MarkResult:
        """The invisible watermark and the IPTC AI label only, made here, with no manifest: what works
        without a key. Sign up free for signed Content Credentials (mark_file)."""
        imaging, _manifest, _throwaway, watermark = self._local_modules()
        data, _name = _read(file)
        media_type = imaging.sniff(data)
        if media_type is None:
            raise KennmarqError("the file isn't a PNG, JPEG or WebP picture")
        picture = imaging.decode(data, media_type)
        wm = watermark.WatermarkId.random()
        picture = picture.with_rgb(self._watermarker_for(watermark).embed(picture.rgb(), wm))
        out_type = media_type if output_format == "keep" else imaging.OUTPUT_FORMATS[_choice("output_format", output_format, OUTPUT_FORMATS)]
        from .local.manifest import SOURCE_TYPES

        encoded = imaging.encode(picture, out_type, xmp=imaging.xmp_packet(SOURCE_TYPES[_choice("source", source, SOURCES)]) if iptc else None)
        if png_text and out_type == "image/png":
            encoded = with_png_text(encoded, png_text)
        return self._write(MarkResult(encoded, out_type, None, wm.hex, None, None, None, None, "watermark-only"), dest)

    def mark_file(self, file: FileLike, dest: str | os.PathLike | None = None, *, source: str = "created", model: str | None = None,
                  model_version: str | None = None, app_name: str | None = None, title: str | None = None, watermark: bool = True,
                  provider: str | None = None, app_version: str | None = None,
                  iptc: bool = True, output_format: str = "keep", keep_input_credentials: bool = True, input: FileLike | None = None,
                  check: str = "off", png_text: dict[str, str] | None = None) -> MarkResult:
        """Local mode: the watermark and the manifest made here, the manifest signed by Kennmarq.com.

        Only the manifest store (about 13 KB, never pixels) is sent, to POST /v1/sign; one mark. With
        `check`, the safety check runs here first and a block raises ContentBlocked (nothing marked or
        counted). The file is validated with Kennmarq.com's roots before it is written: a file that isn't
        Trusted, with a validated time-stamp and its watermark reading back, raises InvalidSignedFile.

        Credentials the file (or `input`) already carries travel with the manifest as ingredients,
        thumbnails included if their signer put any in; keep_input_credentials=False sends none of
        the file's. Where c2pa has no build (Windows), the watermark is made here and the manifest in
        the cloud: the watermarked picture is sent to POST /v1/mark ("hybrid").

        `png_text` adds tEXt chunks to a PNG before it's signed (ComfyUI's prompt and workflow).
        Local mode marks pictures; mark() marks songs and clips, in the cloud.
        """
        self._need_key()
        _app_version_needs_app_name(app_name, app_version)
        _choice("source", source, SOURCES)
        _choice("output_format", output_format, PICTURE_FORMATS)
        _choice("check", check, CHECKS)
        if media_kind(_read(file)[0]) in ("song", "clip"):
            raise KennmarqError("local mode marks pictures; mark() marks songs and clips in the cloud")
        imaging, manifest, throwaway, watermark_module = self._local_modules()
        try:
            import c2pa  # noqa: F401
        except ImportError:
            return self._hybrid(file, dest, source=source, provider=provider, model=model, model_version=model_version,
                                app_name=app_name, app_version=app_version, title=title, watermark=watermark, iptc=iptc, output_format=output_format,
                                keep_input_credentials=keep_input_credentials, input=input, check=check)  # the cloud re-encodes: no png_text
        data, _name = _read(file)
        media_type = imaging.sniff(data)
        if media_type is None:
            raise KennmarqError("the file isn't a PNG, JPEG or WebP picture")
        picture = imaging.decode(data, media_type)
        input_data = _read(input)[0] if input is not None else None
        input_type = imaging.sniff(input_data) if input_data is not None else None
        if input_data is not None and input_type is None:
            raise KennmarqError("the input isn't a PNG, JPEG or WebP picture")

        safety = None
        if check != "off":
            answers = [("file", self._check_locally(data, check))]
            if input_data is not None:
                answers.append(("input", self._check_locally(input_data, check)))
            for picture_name, answer in answers:
                if answer["verdict"] == "block":
                    raise ContentBlocked("the safety check blocked this picture; nothing was marked",
                                         {**answer, "picture": picture_name}, code="content_blocked")
            flagged = next((a for _, a in answers if a["verdict"] == "flag"), None)
            safety = {"verdict": "flag" if flagged else "allow", "reason": flagged["reason"] if flagged else None}

        watermarker = self._watermarker_for(watermark_module)
        wm = watermark_module.WatermarkId.random() if watermark else None
        if wm is not None:
            picture = picture.with_rgb(watermarker.embed(picture.rgb(), wm))
        out_type = media_type if output_format == "keep" else imaging.OUTPUT_FORMATS[output_format]
        encoded = imaging.encode(picture, out_type, xmp=imaging.xmp_packet(manifest.SOURCE_TYPES[source]) if iptc else None)
        if png_text and out_type == "image/png":
            encoded = with_png_text(encoded, png_text)
        parent = (data, media_type) if keep_input_credentials and manifest.has_credentials(data, media_type) else None
        input_credentials = (input_data, input_type) if input_data is not None and manifest.has_credentials(input_data, input_type) else None
        definition = manifest.definition(
            brand=BRAND, version=__version__, brand_slug=BRAND_SLUG, media_type=out_type, source=source,
            watermark_soft_binding=wm.soft_binding_value if wm else None, parent=parent is not None,
            has_input=input_credentials is not None, provider=provider, model=model, model_version=model_version,
            app_name=app_name, app_version=app_version, title=title)
        local_file, store = manifest.sign_locally(encoded, out_type, definition, throwaway.current(), parent=parent,
                                                  input_file=input_credentials)
        signed_store, headers = self.sign_store(store)
        final = manifest.replace_store(local_file, out_type, store, signed_store)

        # The read-back, as the cloud's: nothing is written unless all of it holds.
        state, success, failure = manifest.validate(final, out_type, self.certificate())
        if state != "Trusted" or failure or "timeStamp.validated" not in success:
            raise InvalidSignedFile(f"the signed file reads {state} with Kennmarq.com's roots ({', '.join(sorted(failure)) or 'no time-stamp'}); it was not written")
        if wm is not None and watermarker.read(imaging.decode(final, out_type).rgb()) != wm:
            raise InvalidSignedFile("the watermark doesn't read back from the signed file; it was not written")
        result = MarkResult(final, out_type, headers.get("x-mark-id"), wm.hex if wm else None, _int(headers.get("x-log-index")),
                            _int(headers.get("x-marks-used")), _int(headers.get("x-marks-limit")), safety, "local")
        return self._write(result, dest)

    def _hybrid(self, file: FileLike, dest, *, watermark: bool, iptc: bool, output_format: str, source: str, **options) -> MarkResult:
        """No c2pa build here: the watermark made here, the manifest by POST /v1/mark (watermark=false)."""
        if watermark:
            marked = self.watermark_file(file, source=source, iptc=iptc, output_format=output_format)
            data, wm = marked.data, marked.watermark_id
        else:
            data, wm = _read(file)[0], None
        result = self.mark(data, source=source, watermark=False, iptc=iptc, output_format="keep", **options)
        return self._write(MarkResult(result.data, result.media_type, result.mark_id, wm, result.log_index, result.marks_used,
                                      result.marks_limit, result.safety, "hybrid"), dest)

    @staticmethod
    def _write(result: MarkResult, dest: str | os.PathLike | None) -> MarkResult:
        if dest is None:
            return result
        path = Path(dest)
        path.write_bytes(result.data)
        return MarkResult(**{**result.__dict__, "path": path})
