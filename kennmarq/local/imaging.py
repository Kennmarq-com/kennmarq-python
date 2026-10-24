"""Pictures in and out: the type read from the bytes, decoding with guards, encoding, and IPTC's
digital source type in XMP.

Encoding (the brief's order, step 2): PNG stays lossless; JPEG at quality 95 with 4:4:4 chroma;
WebP lossy at quality 95. The ICC profile, EXIF (re-serialised, which drops its embedded thumbnail,
an unwatermarked copy of the picture) and DPI carry over; nothing else of the original's metadata
does. JPEG has no alpha, so a transparent picture going to JPEG is flattened onto white.

XMP (step 3) carries one field, Iptc4xmpExt:DigitalSourceType, which platforms that skip C2PA read.
Pillow 13 writes XMP for JPEG (an APP1 segment) and WebP (an "XMP " chunk) when it saves, so those
two get it at encode time; it doesn't for PNG, so a PNG gets an iTXt chunk right after IHDR. Any XMP
the original carried is not kept: ours replaces it.
"""
from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass

from PIL import Image

PNG = "image/png"
JPEG = "image/jpeg"
WEBP = "image/webp"
MEDIA_TYPES = (PNG, JPEG, WEBP)
OUTPUT_FORMATS = {"png": PNG, "jpeg": JPEG, "webp": WEBP}
PIL_FORMATS = {PNG: "PNG", JPEG: "JPEG", WEBP: "WEBP"}
EXTENSIONS = {PNG: "png", JPEG: "jpg", WEBP: "webp"}

JPEG_QUALITY = 95
WEBP_QUALITY = 95
# A 25 MB upload can still decode to an enormous canvas (a flat PNG compresses ~1000:1). TrustMark
# works on a 256-pixel copy anyway; 50 megapixels covers 8K and keeps memory bounded.
MAX_PIXELS = 50_000_000

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
XMP_KEYWORD = b"XML:com.adobe.xmp"

# Modes whose ICC profile still fits after conversion to RGB(A); any other profile (CMYK, grey)
# would describe the wrong colour space and is dropped.
RGB_MODES = {"RGB", "RGBA", "RGBX", "RGBa", "P", "PA"}


class ImageError(ValueError):
    """The bytes can't be decoded as the picture they claim to be, or the picture can't be marked."""


def sniff(data: bytes) -> str | None:
    """The media type from the magic bytes, or None: never from a filename or a header."""
    if data.startswith(PNG_SIGNATURE):
        return PNG
    if data.startswith(b"\xff\xd8\xff"):
        return JPEG
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return WEBP
    return None


@dataclass
class Picture:
    """A decoded picture, normalised to RGB or RGBA, and the metadata that carries over."""

    image: Image.Image
    icc_profile: bytes | None
    exif: bytes | None
    dpi: tuple[float, float] | None

    @property
    def has_alpha(self) -> bool:
        return self.image.mode == "RGBA"

    def rgb(self) -> Image.Image:
        return self.image.convert("RGB") if self.has_alpha else self.image

    def with_rgb(self, rgb: Image.Image) -> "Picture":
        """The same picture with its colour channels replaced (the alpha channel is kept)."""
        image = rgb
        if self.has_alpha:
            image = rgb.convert("RGBA")
            image.putalpha(self.image.getchannel("A"))
        return Picture(image, self.icc_profile, self.exif, self.dpi)


def _to_8bit(image: Image.Image) -> Image.Image:
    """Greyscale with more than 8 bits (I;16, I, F) scaled to L, not clipped at 255."""
    if image.mode.startswith("I;16") or image.mode in ("I", "F"):
        extrema = image.getextrema()
        top = 65535 if image.mode.startswith("I;16") or (extrema[1] or 0) > 255 else 255
        scale = 255.0 / top
        return image.convert("F").point(lambda v: v * scale).convert("L")
    return image


def decode(data: bytes, media_type: str) -> Picture:
    """Decode a PNG, JPEG or WebP and normalise it to RGB(A), keeping what should carry over."""
    try:
        with Image.open(io.BytesIO(data), formats=[PIL_FORMATS[media_type]]) as opened:
            width, height = opened.size
            if width * height > MAX_PIXELS:
                raise ImageError(f"{width}x{height} is more than {MAX_PIXELS // 1_000_000} megapixels")
            if getattr(opened, "n_frames", 1) > 1:
                raise ImageError("animated pictures are not supported")
            opened.load()
            mode = opened.mode
            icc = opened.info.get("icc_profile") if mode in RGB_MODES else None
            dpi = opened.info.get("dpi")
            exif = None
            try:
                parsed = opened.getexif()
                if len(parsed):
                    exif = parsed.tobytes()
            except Exception:  # malformed EXIF: drop it rather than fail the mark
                exif = None
            image = _to_8bit(opened)
            has_alpha = image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
                image.mode in ("P", "L", "RGB") and "transparency" in image.info
            )
            image = image.convert("RGBA" if has_alpha else "RGB")
    except ImageError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageError(f"picture too large: {exc}") from exc
    except Exception as exc:
        raise ImageError(f"the file can't be decoded as {media_type}: {type(exc).__name__}") from exc
    return Picture(image, icc or None, exif, tuple(dpi) if dpi else None)


def xmp_packet(source_type_uri: str) -> bytes:
    """An XMP packet holding IPTC's digital source type, and nothing else."""
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"'
        f' Iptc4xmpExt:DigitalSourceType="{source_type_uri}"/>'
        "</rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    ).encode("utf-8")


def _png_chunks(data: bytes):
    """(type, body, raw) for each chunk of a PNG."""
    if not data.startswith(PNG_SIGNATURE):
        raise ImageError("not a PNG")
    at = len(PNG_SIGNATURE)
    while at < len(data):
        if at + 12 > len(data):
            raise ImageError("truncated PNG chunk")
        (length,) = struct.unpack(">I", data[at:at + 4])
        kind = data[at + 4:at + 8]
        end = at + 12 + length
        if end > len(data):
            raise ImageError("truncated PNG chunk")
        yield kind, data[at + 8:at + 8 + length], data[at:end]
        at = end


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def add_png_xmp(data: bytes, packet: bytes) -> bytes:
    """`data` with `packet` as an iTXt chunk (keyword XML:com.adobe.xmp, uncompressed) right after
    IHDR, replacing any XMP chunk it had."""
    body = XMP_KEYWORD + b"\x00\x00\x00\x00\x00" + packet
    out = [PNG_SIGNATURE]
    for kind, chunk_body, raw in _png_chunks(data):
        if kind in (b"iTXt", b"tEXt", b"zTXt") and chunk_body.split(b"\x00", 1)[0] == XMP_KEYWORD:
            continue
        out.append(raw)
        if kind == b"IHDR":
            out.append(_png_chunk(b"iTXt", body))
    return b"".join(out)


def read_xmp(data: bytes, media_type: str) -> bytes | None:
    """The XMP packet a file carries, as Pillow reads it (for tests and checks)."""
    with Image.open(io.BytesIO(data), formats=[PIL_FORMATS[media_type]]) as image:
        xmp = image.info.get("xmp")
        if isinstance(xmp, str):
            xmp = xmp.encode("utf-8")
        return xmp


def encode(picture: Picture, media_type: str, *, xmp: bytes | None) -> bytes:
    """`picture` encoded as `media_type`, with `xmp` written in when given."""
    image = picture.image
    params: dict = {}
    if picture.icc_profile:
        params["icc_profile"] = picture.icc_profile
    if picture.exif:
        params["exif"] = picture.exif
    out = io.BytesIO()
    if media_type == PNG:
        if picture.dpi:
            params["dpi"] = picture.dpi
        image.save(out, format="PNG", compress_level=6, **params)
        data = out.getvalue()
        return add_png_xmp(data, xmp) if xmp else data
    if media_type == JPEG:
        if image.mode == "RGBA":
            flat = Image.new("RGB", image.size, (255, 255, 255))
            flat.paste(image, mask=image.getchannel("A"))
            image = flat
        if picture.dpi:
            params["dpi"] = picture.dpi
        if xmp:
            params["xmp"] = xmp
        try:
            image.save(out, format="JPEG", quality=JPEG_QUALITY, subsampling=0, optimize=True, **params)
        except OSError:
            # Optimising needs the whole JPEG in one buffer, which Pillow sizes at 2 bytes a pixel at this
            # quality: a very noisy picture compresses to more ("broken data stream"). Unoptimised, the
            # encoder streams, at a few percent more bytes.
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=JPEG_QUALITY, subsampling=0, **params)
        return out.getvalue()
    if media_type == WEBP:
        if xmp:
            params["xmp"] = xmp
        image.save(out, format="WEBP", quality=WEBP_QUALITY, method=6, **params)
        return out.getvalue()
    raise ImageError(f"can't encode {media_type}")
