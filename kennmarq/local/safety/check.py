"""Kennmarq's safety check: the gates (nudity.py, age.py, faces.py) read a picture, and this decides
what the reading means under a customer's policy and under our own floor.

A picture is read in steps, each only when the one before calls for it, so a picture without nudity
costs one ViT pass:
1. nudity: "explicit" (the NSFW screen's verdict), "suggestive" (its score at 0.3 or more, not
   explicit) or "none";
2. for a suggestive or explicit picture (an explicit one only, under the floor alone): whether it's a
   photo (SigLIP's photo share at 0.5 or more), then
   - a photo: its faces, and whether one reads as under 18, on age.py's explicit lines when the
     picture is explicit and on its standard lines when it is suggestive;
   - a drawing: whether the tagger reads a child.

So `apparent_minor` and `realistic_face` are null (not read) for a picture without nudity: the check
never estimates ages for their own sake. It is offered only as a block/allow filter, since age
estimation sold on its own could be high-risk biometric categorisation (AI Act Annex III 1(b)).

Policies:
- ai-act: blocks sexual or suggestive content with an apparent minor (Art. 5(1)(bb)), and flags
  explicit content with a realistic face for the customer's consent check (Art. 5(1)(ba));
- safe-only: blocks an apparent minor as above, and anything explicit;
- our floor, on every cloud mark whatever the policy: blocks explicit content with an apparent
  minor.
Under every policy, then, the floor holds.

Nothing about a check is kept: no picture, no score, no hash. Results go only into the answer, never
into the file's marks. No ages and no identities are returned.
"""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .age import DRAWN_MINOR_THRESHOLD, REALISM_THRESHOLD, AgeReader
from .faces import FaceFinder
from .models import SafetyUnavailable, fetch
from .nudity import NudityReader

POLICIES = ("ai-act", "safe-only")
FLOOR = "floor"
# A picture's longest side before anything reads it: the readers see 224 to 640 px, and a 50 MP
# upload in age.py's grey border would hold about 380 MB.
MAX_SIDE = 2048

# EXIF orientation -> the transpose that makes the picture upright, as the person who took it saw it
# (faces turned on their side aren't found).
_UPRIGHT = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
            5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90}


def upright(rgb: Image.Image, exif: bytes | None) -> Image.Image:
    """The picture as its EXIF orientation shows it, at most MAX_SIDE on its longest side."""
    image = rgb.convert("RGB")
    if exif:
        try:
            parsed = Image.Exif()
            parsed.load(exif)
            method = _UPRIGHT.get(parsed.get(0x0112))
        except Exception:  # malformed EXIF: read the pixels as they are
            method = None
        if method is not None:
            image = image.transpose(method)
    if max(image.size) > MAX_SIDE:
        image = image.copy()
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return image


@dataclass(frozen=True)
class Reading:
    nudity: str  # "none" | "suggestive" | "explicit"
    score: float  # the NSFW classifier's P(nsfw)
    realistic_face: bool | None  # a photo with a face the gate could read; None when not read
    apparent_minor: bool | None  # None when not read


def decide(reading: Reading, policy: str) -> tuple[str, str | None]:
    """(verdict, reason) under a policy, or under the floor alone."""
    if policy == FLOOR:
        if reading.nudity == "explicit" and reading.apparent_minor:
            return "block", "sexual_apparent_minor"
        return "allow", None
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if reading.apparent_minor:
        return "block", "sexual_apparent_minor"
    if policy == "safe-only" and reading.nudity == "explicit":
        return "block", "explicit"
    if policy == "ai-act" and reading.nudity == "explicit" and reading.realistic_face:
        return "flag", "sexual_realistic_face"
    return "allow", None


class SafetyChecker:
    """The models, loaded on first use (after their files are checked against their pins), CPU only,
    one check at a time. Any failure to fetch, check, load or run them raises SafetyUnavailable:
    the call fails closed and never skips the check."""

    def __init__(self, models_dir: Path, *, brand: str, version: str):
        self.models_dir = Path(models_dir)
        self.brand = brand
        self.version = version
        self.faces = FaceFinder()
        self.nudity = NudityReader(self.models_dir)
        self.age = AgeReader(self.models_dir, self.faces)
        self._checked = False
        self._lock = threading.Lock()

    def _ready(self) -> None:
        if not self._checked:
            fetch(self.models_dir)
            self._checked = True

    def load(self) -> None:
        """Everything, ahead of the first check (tests and a warm start)."""
        def work() -> None:
            self._ready()
            self.nudity.load()
            self.faces.load()
            self.age.load()
            self.age.load_tagger()

        with self._lock:
            self._guarded(work)

    def _guarded(self, work):
        try:
            return work()
        except SafetyUnavailable:
            raise
        except Exception as exc:
            raise SafetyUnavailable(f"the safety check can't run: {type(exc).__name__}: {exc}") from exc

    def read(self, image: Image.Image, policy: str) -> Reading:
        """The steps of the module docstring, for a picture already made upright()."""
        def work() -> Reading:
            self._ready()
            nudity, score = self.nudity.read(image)
            if nudity == "none" or (policy == FLOOR and nudity != "explicit"):
                return Reading(nudity, score, None, None)
            explicit = nudity == "explicit"
            [real] = self.age.realism([image])
            if real >= REALISM_THRESHOLD:
                face, minor = self.age.photo_minor(image, explicit)
                return Reading(nudity, score, face, minor)
            return Reading(nudity, score, False, self.age.drawn_child(image) >= DRAWN_MINOR_THRESHOLD)

        with self._lock:
            return self._guarded(work)

    def check(self, image: Image.Image, policy: str, file: dict) -> dict:
        """The answer for one picture, as the service's safety check returns it."""
        reading = self.read(image, policy)
        verdict, reason = decide(reading, policy)
        return {
            "object": "safety_check",
            "checked_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "checker": {"name": self.brand, "version": self.version},
            "file": file,
            "policy": policy,
            "verdict": verdict,
            "reason": reason,
            "nudity": {"label": reading.nudity, "score": round(reading.score, 3)},
            "apparent_minor": reading.apparent_minor,
            "realistic_face": reading.realistic_face,
        }
