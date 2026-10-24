"""Nudity: the NSFW screen.

Falconsai/nsfw_image_detection (a ViT-B/16 at 224 px with two labels, normal and nsfw) scores each
picture. A score at or above HARD is explicit on its own. One between THRESHOLD and HARD is explicit
only when NudeNet's 320n detector finds an exposed breast, genitals, buttocks or anus scoring
EXPOSED_MIN or more: the ViT reads clothed but revealing pictures as unsafe (seen on 2026-10-10: a
woman asleep in a sports top scored 0.72-0.93), and NudeNet found no exposed part on them.

What check.py makes of it: "explicit" is that verdict; "suggestive" is a score of SUGGESTIVE or more
that isn't explicit (MINOR_SUGGESTIVE_THRESHOLD, the line used for a drawn child);
"none" is below it.

Two choices, neither changing a number:
- the ViT's input is computed here (squashed to 224x224, bilinear, scaled to [-1, 1]) instead of by
  transformers' ViTImageProcessor, so it doesn't depend on which backend a transformers version
  picks. That is what the pinned preprocessor_config.json asks for (checked at load), and
  the tests hold it against the processor;
- NudeNet's file comes out of the pinned nudenet 3.4.2 wheel (models.py), not an installed package.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .models import NSFW_DIR, NUDENET_DIR, SafetyUnavailable, quiet_transformers

THRESHOLD = 0.75
HARD = 0.97
EXPOSED_MIN = 0.3
SUGGESTIVE = 0.3
SIDE = 224
BATCH = 8
# NudeNet's 18 labels in output order; these five are exposed parts: buttocks, female breast, female
# genitalia, anus, male genitalia.
EXPOSED = (2, 3, 4, 6, 14)
# What the pinned preprocessor_config.json must say for the input computed here to be its input.
PREPROCESSING = {"do_resize": True, "size": {"height": SIDE, "width": SIDE}, "resample": 2, "do_rescale": True,
                 "rescale_factor": 1 / 255, "do_normalize": True, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}


def vit_pixels(images: list[Image.Image], resample=Image.BILINEAR):
    """224x224, scaled to [-1, 1], NCHW: the ViTs' and SigLIPs' input."""
    import torch

    batch = [np.asarray(img.convert("RGB").resize((SIDE, SIDE), resample), np.float32) / 127.5 - 1.0 for img in images]
    return torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2).copy())


def _linear_resize(img: np.ndarray, size: int) -> np.ndarray:
    """OpenCV's INTER_LINEAR resize (no antialiasing), which NudeNet's preprocessing uses, in numpy:
    within 0.01 of its scores without OpenCV."""
    h, w = img.shape[:2]

    def axis(n_in):
        src = np.clip((np.arange(size) + 0.5) * (n_in / size) - 0.5, 0, n_in - 1)
        lo = np.floor(src).astype(int)
        return lo, np.minimum(lo + 1, n_in - 1), (src - lo).astype(np.float32)

    y0, y1, wy = axis(h)
    x0, x1, wx = axis(w)
    a = img.astype(np.float32)
    top = a[y0][:, x0] * (1 - wx)[None, :, None] + a[y0][:, x1] * wx[None, :, None]
    bottom = a[y1][:, x0] * (1 - wx)[None, :, None] + a[y1][:, x1] * wx[None, :, None]
    return top * (1 - wy)[:, None, None] + bottom * wy[:, None, None]


class NudityReader:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._vit = None
        self._detector = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._vit is None:
                folder = self.models_dir / NSFW_DIR
                config = json.loads((folder / "preprocessor_config.json").read_text())
                if any(config.get(k) != v for k, v in PREPROCESSING.items() if k != "rescale_factor") or \
                        abs(config.get("rescale_factor", 0) - 1 / 255) > 1e-9:
                    raise SafetyUnavailable("the NSFW classifier's preprocessing isn't the pinned one")
                from transformers import ViTForImageClassification

                with quiet_transformers():
                    vit = ViTForImageClassification.from_pretrained(folder, local_files_only=True).eval()
                if vit.config.id2label.get(1) != "nsfw":
                    raise SafetyUnavailable(f"the NSFW classifier's labels aren't the pinned ones: {vit.config.id2label}")
                self._vit = vit
            if self._detector is None:
                import onnxruntime

                path = self.models_dir / NUDENET_DIR / "320n.onnx"
                session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                self._detector = (session, session.get_inputs()[0].name)

    def scores(self, images: list[Image.Image]) -> list[float]:
        """P(nsfw) per picture."""
        import torch

        self.load()
        out: list[float] = []
        with torch.no_grad():
            for start in range(0, len(images), BATCH):
                logits = self._vit(pixel_values=vit_pixels(images[start:start + BATCH])).logits
                out.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
        return out

    def exposed(self, image: Image.Image) -> float:
        """NudeNet's highest score for an exposed part (0 when none of its detections is one), each
        detection counting as the class it scores highest: padded square at the bottom and right,
        BGR, 320x320, as its own preprocessing."""
        self.load()
        session, name = self._detector
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        side = max(h, w)
        padded = np.zeros((side, side, 3), np.uint8)
        padded[:h, :w] = rgb[..., ::-1]
        blob = (_linear_resize(padded, 320) / 255.0).transpose(2, 0, 1)[None].astype(np.float32)
        out = session.run(None, {name: np.ascontiguousarray(blob)})[0][0]
        classes = out[4:]
        best, top = classes.argmax(0), classes.max(0)
        exposed = np.isin(best, EXPOSED)
        return float(top[exposed].max()) if exposed.any() else 0.0

    def read(self, image: Image.Image) -> tuple[str, float]:
        """("explicit" | "suggestive" | "none", the ViT's P(nsfw))."""
        [score] = self.scores([image])
        if score >= HARD or (score >= THRESHOLD and self.exposed(image) >= EXPOSED_MIN):
            return "explicit", score
        return ("suggestive" if score >= SUGGESTIVE else "none"), score
