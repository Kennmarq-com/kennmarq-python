"""The invisible watermark: Adobe TrustMark (MIT), variant Q, on CPU.

Payload. TrustMark carries 100 bits; its encodings split them between data and error correction:
BCH_SUPER 40 data bits, BCH_5 61, BCH_4 68, BCH_3 75 (trustmark/datalayer.py). BCH_5 is the most
robust one that still carries at least 48 bits (it corrects 5 flipped bits). We fill all 61 with a
random ID: at 48 bits, ten million marks would hold about an 18% chance that two share an ID
(n²/2^(b+1)); at 61 bits it is about 2 in 100,000, and the ID is what the durable-credential lookup
keys on.

How the ID is written down:
- `watermark_id` in X-Mark-Meta: the 61 bits as 16 hex digits (the first digit is at most 1);
- the C2PA soft-binding value: "<schema>*<bits>", as TrustMark's own C2PA example
  (c2pa/c2pa_watermark_example.py) writes it: the encoding's schema code (BCH_5 = 1), "*", and the
  payload as a string of 0s and 1s.

Reading. `inspect` is TrustMark's own decode at one orientation: the centre square of a picture over
twice as wide as high (or high as wide), resized to 245 px. TrustMark's decode(ROTATION=True) tries
the quarter turns in TURNS' order; Kennmarq's checker does that itself, and also reads the centre
(`centre`) for a second look.

Weights. TrustMark downloads them on first use from Adobe's CDN,
https://cai-watermark.adobe.net/watermarking/trustmark-models/ (trustmark_Q.yaml, encoder_Q.ckpt,
decoder_Q.ckpt), into its own package folder; here they go to the models folder given (the
service's MODELS_DIR, or the SDK's own cache in local mode). Each file is checked against TrustMark's
published MD5 and against the SHA-256 pinned
below, both after a download and before every load: a file that doesn't match is never loaded,
and marking fails closed rather than skipping the watermark.
"""
from __future__ import annotations

import hashlib
import math
import os
import secrets
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ALG = "com.adobe.trustmark.Q"
MODEL_TYPE = "Q"
SCHEMA_BCH_5 = 1  # TrustMark.Encoding.BCH_5
ID_BITS = 61

MODEL_HOST = "https://cai-watermark.adobe.net/watermarking/trustmark-models/"
# SHA-256 of the files TrustMark 0.9.2 fetches for variant Q, taken on 2026-10-22 from the files
# whose MD5 matched TrustMark's own MODEL_CHECKSUMS.
PINNED_SHA256 = {
    "trustmark_Q.yaml": "43f37103f92efa8bd6b1c5902bb537cc12a981dc699fca19d9bb7de8c62d03d9",
    "encoder_Q.ckpt": "dc382c3f6b4fd568b27d6fbb763d6ffc2d2587126d84afe9d5ee95b4c5d99826",
    "decoder_Q.ckpt": "e3d9cea5406a26590735719f8f15cb10802b11852ae69047eaf4cf17214df781",
}
DOWNLOAD_TIMEOUT = 120

# The quarter turns a picture can be read at, in the order TrustMark's decode(ROTATION=True) tries
# them: as it is, then Pillow's ROTATE_90 (a quarter turn anticlockwise), ROTATE_180 and ROTATE_270.
TURNS = (0, 90, 180, 270)
_TRANSPOSE = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}
# The share of each side the checker's second look reads, as kennmarq.com/check reads it.
CENTRE = 0.9


class WatermarkUnavailable(RuntimeError):
    """TrustMark or its weights could not be loaded."""


@dataclass(frozen=True)
class WatermarkId:
    """A watermark payload: ID_BITS bits."""

    bits: str

    def __post_init__(self) -> None:
        if len(self.bits) != ID_BITS or set(self.bits) - {"0", "1"}:
            raise ValueError(f"a watermark ID is {ID_BITS} bits")

    @classmethod
    def random(cls) -> "WatermarkId":
        return cls(format(secrets.randbits(ID_BITS), f"0{ID_BITS}b"))

    @classmethod
    def from_hex(cls, value: str) -> "WatermarkId":
        number = int(value, 16)
        if number >> ID_BITS:
            raise ValueError(f"a watermark ID is {ID_BITS} bits")
        return cls(format(number, f"0{ID_BITS}b"))

    @property
    def hex(self) -> str:
        return format(int(self.bits, 2), "016x")

    @property
    def soft_binding_value(self) -> str:
        return f"{SCHEMA_BCH_5}*{self.bits}"


@dataclass(frozen=True)
class Reading:
    """What TrustMark reads at one quarter turn: whether error correction accepts the packet, its
    schema (TrustMark.Encoding) and its corrected payload bits."""

    found: bool
    schema: int
    bits: str
    turn: int = 0

    @property
    def watermark(self) -> WatermarkId | None:
        """The payload as one of our IDs when the packet is in our schema (BCH_5). Only our records
        tell whether it is ours: anyone can mark with BCH_5, and chance makes such packets too."""
        if self.found and self.schema == SCHEMA_BCH_5 and len(self.bits) == ID_BITS:
            return WatermarkId(self.bits)
        return None

    def same_packet(self, other: "Reading") -> bool:
        return self.found and other.found and (self.schema, self.bits) == (other.schema, other.bits)


NOT_FOUND = Reading(False, -1, "")


def turned(image: Image.Image, turn: int) -> Image.Image:
    """`image` turned anticlockwise by `turn` degrees (one of TURNS), as TrustMark turns it."""
    return image if turn == 0 else image.transpose(_TRANSPOSE[turn])


def centre(image: Image.Image) -> Image.Image:
    """The centre CENTRE of each side, rounded as the checker page rounds it (Math.round)."""
    width, height = math.floor(image.width * CENTRE + 0.5), math.floor(image.height * CENTRE + 0.5)
    left, top = (image.width - width) // 2, (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def _digests(path: Path) -> tuple[str, str]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _matches(path: Path, name: str, md5_list: dict[str, str]) -> bool:
    if not path.is_file():
        return False
    md5, sha256 = _digests(path)
    return md5 == md5_list.get(name) and sha256 == PINNED_SHA256[name]


def fetch_weights(models_dir: Path, md5_list: dict[str, str]) -> None:
    """Make sure every pinned file is in `models_dir` and matches; download what's missing.

    A download goes to a temporary file in the same folder and is renamed into place only once both
    digests match, so a partial or tampered file is never left where the loader looks."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in PINNED_SHA256:
        target = models_dir / name
        if _matches(target, name, md5_list):
            continue
        fd, tmp_name = tempfile.mkstemp(dir=models_dir, prefix=f".{name}.")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with urllib.request.urlopen(MODEL_HOST + name, timeout=DOWNLOAD_TIMEOUT) as response, open(tmp, "wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
            if not _matches(tmp, name, md5_list):
                raise WatermarkUnavailable(f"{name} from {MODEL_HOST} does not match its pinned digests")
            os.replace(tmp, target)
        except WatermarkUnavailable:
            raise
        except Exception as exc:
            raise WatermarkUnavailable(f"could not fetch {name}: {type(exc).__name__}: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)


class Watermarker:
    """TrustMark Q with BCH_5, loaded on first use, CPU only, one inference at a time."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from trustmark import TrustMark
                from trustmark import trustmark as trustmark_module
            except Exception as exc:  # torch or trustmark missing or broken
                raise WatermarkUnavailable(f"TrustMark is not importable: {exc}") from exc
            fetch_weights(self.models_dir, trustmark_module.MODEL_CHECKSUMS)
            models_dir = self.models_dir.resolve()

            class _TrustMark(TrustMark):
                """TrustMark reading its weights from MODELS_DIR instead of its package folder."""

                def load_model(self, config_path, weight_path, device, secret_len, part="all"):
                    config_path = models_dir / Path(config_path).name
                    weight_path = models_dir / Path(weight_path).name
                    for path in (config_path, weight_path):
                        if not _matches(path, path.name, trustmark_module.MODEL_CHECKSUMS):
                            raise WatermarkUnavailable(f"{path} does not match its pinned digests")
                    return super().load_model(str(config_path), str(weight_path), device, secret_len, part)

            try:
                model = _TrustMark(
                    verbose=False,
                    model_type=MODEL_TYPE,
                    encoding_type=TrustMark.Encoding.BCH_5,
                    device="cpu",
                    loadRemover=False,
                )
            except WatermarkUnavailable:
                raise
            except Exception as exc:
                raise WatermarkUnavailable(f"TrustMark failed to load: {type(exc).__name__}: {exc}") from exc
            if model.encoder is None or model.decoder is None or model.schemaCapacity() != ID_BITS:
                raise WatermarkUnavailable("TrustMark loaded without its encoder or decoder")
            self._model = model
            return model

    def load(self) -> None:
        """Load the model now (at start-up or before a test), instead of on the first mark."""
        self._load()

    def embed(self, image: Image.Image, watermark: WatermarkId) -> Image.Image:
        """`image` (RGB) with `watermark` in it, same size, RGB."""
        if image.mode != "RGB":
            raise ValueError("TrustMark embeds into RGB pixels")
        model = self._load()
        with self._lock:
            return model.encode(image, watermark.bits, MODE="binary")

    def inspect(self, image: Image.Image) -> tuple[bool, int, str]:
        """What TrustMark reads in `image`, whatever the schema: (found, schema, payload bits)."""
        model = self._load()
        with self._lock:
            bits, found, schema = model.decode(image.convert("RGB"), MODE="binary")
        return bool(found), int(schema), str(bits)

    def reading(self, image: Image.Image, turn: int = 0) -> Reading:
        """What TrustMark reads in `image` turned by `turn` (one of TURNS), whatever the schema."""
        found, schema, bits = self.inspect(turned(image, turn))
        return Reading(found, schema, bits, turn) if found else Reading(False, -1, "", turn)

    def read(self, image: Image.Image) -> WatermarkId | None:
        """The watermark in `image`, or None when TrustMark finds none (or one of another schema)."""
        found, schema, bits = self.inspect(image)
        if not found or schema != SCHEMA_BCH_5 or len(bits) != ID_BITS:
            return None
        return WatermarkId(bits)
