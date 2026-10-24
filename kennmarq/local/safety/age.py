"""Apparent minors: the under-18 gate.

A photo:
  1. It is realistic when google/siglip-base-patch16-224 (Apache-2.0), zero-shot against photo
     prompts and drawn, painted and rendered ones (REAL, DRAWN), puts at least REALISM_THRESHOLD
     of its share on the photo side. Drawn faces never reach the age models: age-vit reads them as
     children's (an anime picture scored 0.99).
  2. Each face YuNet finds (faces.py; on the picture as it is and inside a grey border, so a close-up
     that fills the frame is found too), MIN_FACE_PX tall or more, is cut to its face and shoulders
     at three sizes, each also mirrored, and two models read the six cuts, their scores averaged:
       - prithivMLmods/Age-Classification-SigLIP2 (Apache-2.0, five age bands): its P(child 0-12)
         at or above the child line reads as a minor;
       - Civitai/age-vit (Apache-2.0, minor or adult): its P(minor), with SigLIP2-age's P(child or
         teen 13-20), both at or above the minor line reads as a minor. SigLIP2-age's teen band runs
         to 20, so a teenager reads as a minor only when both models agree.
     Two pairs of lines: CHILD_THRESHOLD and MINOR_THRESHOLD, and the stricter EXPLICIT_ ones
     ("any person appearing to be a child"). check.py uses the explicit lines on an explicit
     picture and the others on a suggestive one.
A drawing: SmilingWolf's wd-vit-tagger-v3 (Apache-2.0, ONNX) reads its Danbooru tags, and a child
tag (loli, shota, child, aged_down) at DRAWN_MINOR_THRESHOLD or more reads as a minor.

Measured on 2026-10-22 (5,980 FairFace validation faces, share read as a minor): the standard
lines, 0-2 97%, 3-9 85%, 10-19 34%, 20-29 2.3%, 30 and over 0.6%; the explicit lines, 0-2
98.5%, 3-9 92%, 10-19 47%, 20-29 5.9% (women 8.0%, men 3.2%; Black and Southeast Asian faces about
12.5%, White 1.7%), 30 and over 0.9%. FairFace is age-vit's own training set, so real pictures will
do worse. The docs publish these (kennmarq.com/docs/api/check). Known limits: a face YuNet can't find (turned away, hidden, under MIN_FACE_PX) gets no
read, so a photo with no face passes; apparent age is a guess near 18, both ways; a near-photoreal 3D
render counts as a photo; the tagger reads Danbooru's idea of a child, which a style far from anime
may hide.

The lines are constants, the models come from models.py's pinned files, and the prompts'
embeddings are checked by SHA-256 too.
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .faces import FaceFinder, cut_box, overlap
from .models import (AGE_DIR, CHILD_DIR, PROMPTS, PROMPTS_SHA256, REAL_DIR, TAGGER_DIR, SafetyUnavailable, check_file,
                     quiet_transformers)
from .nudity import BATCH, vit_pixels

REAL = (
    "a photo", "a photograph of a person", "a realistic photo", "a photorealistic image", "a selfie",
    "a candid snapshot", "a professional photoshoot", "a movie still", "a real photo",
    "a photograph taken with a camera", "a smartphone photo", "a DSLR photograph", "a film photograph",
    "a close-up photo of a face",
)
DRAWN = (
    "an anime illustration", "a manga drawing", "a cartoon", "a 3D cartoon render", "a drawing", "a painting",
    "an oil painting", "a watercolor painting", "a pencil sketch", "a comic book panel", "pixel art",
    "a digital illustration", "a woodblock print", "a video game character render", "an anime girl", "an anime boy",
    "an anime character", "an anime screenshot", "an anime key visual", "a visual novel illustration",
    "a cel-shaded drawing", "a 2D illustration", "digital anime art", "a comic book illustration",
    "a watercolor illustration", "a semi-realistic anime illustration",
)

CHILD_THRESHOLD = 0.6
MINOR_THRESHOLD = 0.8
EXPLICIT_CHILD_THRESHOLD = 0.2
EXPLICIT_MINOR_THRESHOLD = 0.4
REALISM_THRESHOLD = 0.5
DRAWN_MINOR_THRESHOLD = 0.35
DRAWN_MINOR_TAGS = ("loli", "shota", "child", "aged_down")

# Smaller faces carry too little to read an age from.
MIN_FACE_PX = 32
# The grey border round the picture YuNet looks at a second time, as a share of each side: a face that
# fills the frame isn't found without it (FairFace's close-ups: 60% found, 99% with it).
BORDER = 0.3
# Each face is read on these scales of its box, each also mirrored, and its scores are the means: on
# one cut, age-vit read a five-year-old as 0.02 and, the box moved 7 px, 0.99.
CUT_SCALES = (0.85, 1.0, 1.2)


def face_is_minor(scores: list[tuple[float, float, float]], explicit: bool) -> bool:
    """One face's cuts, each (age-vit's P(minor), SigLIP2-age's P(child), its P(child or teen)),
    averaged: SigLIP2-age reads a child, or both models read someone young."""
    n = len(scores)
    minor, child, young = (sum(s[i] for s in scores) / n for i in range(3))
    child_line, minor_line = (EXPLICIT_CHILD_THRESHOLD, EXPLICIT_MINOR_THRESHOLD) if explicit else (CHILD_THRESHOLD, MINOR_THRESHOLD)
    return child >= child_line or (minor >= minor_line and young >= minor_line)


def _unit(e):
    return e / e.norm(dim=-1, keepdim=True)


class AgeReader:
    def __init__(self, models_dir: Path, faces: FaceFinder):
        self.models_dir = Path(models_dir)
        self.finder = faces
        self._models: dict | None = None
        self._tagger: dict | None = None
        self._lock = threading.Lock()

    def load(self) -> dict:
        """SigLIP's vision half, both age models and the prompts' embeddings, on the CPU, from the
        checked files only (local_files_only)."""
        with self._lock:
            if self._models is not None:
                return self._models
            import torch
            from transformers import SiglipForImageClassification, SiglipVisionModel, ViTForImageClassification

            check_file(PROMPTS, PROMPTS_SHA256)
            with np.load(PROMPTS, allow_pickle=False) as saved:
                if [str(p) for p in saved["prompts"]] != list(REAL + DRAWN):
                    raise SafetyUnavailable(f"{PROMPTS} is for other prompts")
                prompts = torch.from_numpy(saved["embeds"].astype(np.float32))
                scale = float(saved["scale"])
            # SigLIP's checkpoint holds its text half too, left unloaded on purpose.
            with quiet_transformers():
                vision = SiglipVisionModel.from_pretrained(self.models_dir / REAL_DIR, local_files_only=True).eval()
                age = ViTForImageClassification.from_pretrained(self.models_dir / AGE_DIR, local_files_only=True).eval()
                child = SiglipForImageClassification.from_pretrained(self.models_dir / CHILD_DIR, local_files_only=True).eval()
            if age.config.id2label.get(0) != "minor":
                raise SafetyUnavailable(f"age-vit's labels aren't the pinned ones: {age.config.id2label}")
            labels = child.config.id2label
            if not (str(labels.get(0, "")).startswith("Child") and str(labels.get(1, "")).startswith("Teen")):
                raise SafetyUnavailable(f"SigLIP2-age's labels aren't the pinned ones: {labels}")
            self._models = {"vision": vision, "age": age, "child": child, "prompts": prompts, "scale": scale}
            return self._models

    def load_tagger(self) -> dict:
        with self._lock:
            if self._tagger is not None:
                return self._tagger
            import onnxruntime

            folder = self.models_dir / TAGGER_DIR
            with open(folder / "selected_tags.csv", newline="") as fh:
                names = [row["name"] for row in csv.DictReader(fh)]
            index = {name: i for i, name in enumerate(names)}
            missing = [t for t in DRAWN_MINOR_TAGS if t not in index]
            if missing:
                raise SafetyUnavailable(f"the tagger's tag list lacks {missing}")
            session = onnxruntime.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
            self._tagger = {"session": session, "side": int(session.get_inputs()[0].shape[1]),
                            "minor": [index[t] for t in DRAWN_MINOR_TAGS]}
            return self._tagger

    def realism(self, images: list[Image.Image]) -> list[float]:
        """Each picture's photo share, 0-1."""
        import torch

        m = self.load()
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(images), BATCH):
                e = _unit(m["vision"](pixel_values=vit_pixels(images[start:start + BATCH], Image.BICUBIC)).pooler_output)
                probs = torch.softmax(e @ m["prompts"].T * m["scale"], dim=-1)
                out.extend(probs[:, :len(REAL)].sum(dim=-1).tolist())
        return out

    def face_cuts(self, image: Image.Image) -> list[list[Image.Image]]:
        """Each face's cuts, the faces MIN_FACE_PX tall or more: found on the picture as it is and
        inside a grey border, each face once, and cut to their face and shoulders from the bordered
        picture, so a face at its edge keeps its framing."""
        w, h = image.size
        bx, by = round(w * BORDER), round(h * BORDER)
        canvas = Image.new("RGB", (w + 2 * bx, h + 2 * by), (127, 127, 127))
        canvas.paste(image, (bx, by))
        boxes = [(x1 + bx, y1 + by, x2 + bx, y2 + by) for (x1, y1, x2, y2), _score in self.finder.faces(image)]
        for box, _score in self.finder.faces(canvas):
            if all(overlap(box, seen) <= 0.3 for seen in boxes):
                boxes.append(box)
        faces = []
        for x1, y1, x2, y2 in boxes:
            if y2 - y1 < MIN_FACE_PX:
                continue
            cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
            cuts = []
            for s in CUT_SCALES:
                box = (cx - hw * s, cy - hh * s, cx + hw * s, cy + hh * s)
                cut = canvas.crop(cut_box(canvas.width, canvas.height, box))
                cuts += [cut, ImageOps.mirror(cut)]
            faces.append(cuts)
        return faces

    def age_scores(self, cuts: list[Image.Image]) -> list[tuple[float, float, float]]:
        """Each cut's (age-vit's P(minor), SigLIP2-age's P(child 0-12), its P(child or teen 13-20))."""
        import torch

        m = self.load()
        out: list[tuple[float, float, float]] = []
        with torch.no_grad():
            for start in range(0, len(cuts), BATCH):
                px = vit_pixels(cuts[start:start + BATCH], Image.BILINEAR)
                minor = torch.softmax(m["age"](pixel_values=px).logits, dim=-1)[:, 0]
                classes = torch.softmax(m["child"](pixel_values=px).logits, dim=-1)
                out.extend(zip(minor.tolist(), classes[:, 0].tolist(), (classes[:, 0] + classes[:, 1]).tolist()))
        return out

    def photo_minor(self, image: Image.Image, explicit: bool) -> tuple[bool, bool]:
        """(a face was found, one reads as a minor) for a photo."""
        per_face = self.face_cuts(image)
        if not per_face:
            return False, False
        scores = self.age_scores([cut for cuts in per_face for cut in cuts])
        at, minor = 0, False
        for cuts in per_face:
            minor = minor or face_is_minor(scores[at:at + len(cuts)], explicit)
            at += len(cuts)
        return True, minor

    def drawn_child(self, image: Image.Image) -> float:
        """A drawing's child read: the highest of DRAWN_MINOR_TAGS, 0-1. The tagger's own
        preprocessing: on white, padded to a square, bicubic to its side, BGR, 0-255, NHWC."""
        t = self.load_tagger()
        session = t["session"]
        dim = max(image.size)
        square = Image.new("RGB", (dim, dim), (255, 255, 255))
        square.paste(image.convert("RGB"), ((dim - image.width) // 2, (dim - image.height) // 2))
        if dim != t["side"]:
            square = square.resize((t["side"], t["side"]), Image.BICUBIC)
        pixels = np.asarray(square, dtype=np.float32)[:, :, ::-1][None]
        probs = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: np.ascontiguousarray(pixels)})[0][0]
        return float(max(probs[i] for i in t["minor"]))
