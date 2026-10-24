"""Faces: OpenCV Zoo's YuNet (face_detection_yunet_2023mar.onnx, MIT, 232 KB, in assets/) on
onnxruntime's CPU build, about 12 ms a picture, decoded as OpenCV's FaceDetectorYN does
(modules/objdetect/src/face_detect.cpp).

The model is checked against its pinned SHA-256 before it loads.
"""
from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from .models import YUNET, YUNET_SHA256, SafetyUnavailable, check_file

# YuNet's input, fixed in the 2023mar file.
SIZE = 640
STRIDES = (8, 16, 32)
MIN_SCORE = 0.75
# The cut, in face heights from the face's centre, and its shape.
ABOVE, BELOW = 1.3, 1.9
ASPECT = 3 / 4


def overlap(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


class FaceFinder:
    def __init__(self) -> None:
        self._session = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._session is None:
                check_file(YUNET, YUNET_SHA256)
                import onnxruntime

                try:
                    self._session = onnxruntime.InferenceSession(str(YUNET), providers=["CPUExecutionProvider"])
                except Exception as exc:
                    raise SafetyUnavailable(f"YuNet won't load: {exc}") from exc
            return self._session

    def faces(self, image: Image.Image) -> list[tuple[tuple[float, float, float, float], float]]:
        """The faces in a picture as ((x1, y1, x2, y2), score), largest first: YuNet on the picture
        fitted into its 640x640 input (BGR, unscaled), each stride's cells decoded as face_detect.cpp
        does, overlaps above 0.3 dropped."""
        rgb = image.convert("RGB")
        w, h = rgb.size
        scale = min(SIZE / w, SIZE / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = np.zeros((SIZE, SIZE, 3), np.float32)
        canvas[:nh, :nw] = np.asarray(rgb.resize((nw, nh), Image.BILINEAR), np.float32)[:, :, ::-1]
        session = self.load()
        names = [o.name for o in session.get_outputs()]
        outs = dict(zip(names, session.run(None, {session.get_inputs()[0].name: canvas.transpose(2, 0, 1)[None]})))
        found = []
        for stride in STRIDES:
            cols = SIZE // stride
            score = np.sqrt(np.clip(outs[f"cls_{stride}"][0, :, 0], 0, 1) * np.clip(outs[f"obj_{stride}"][0, :, 0], 0, 1))
            for i in np.where(score >= MIN_SCORE)[0]:
                dx, dy, dw, dh = outs[f"bbox_{stride}"][0, i]
                cx, cy = (i % cols + dx) * stride, (i // cols + dy) * stride
                bw, bh = np.exp(dw) * stride, np.exp(dh) * stride
                box = tuple(float(v) / scale for v in (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
                found.append((box, float(score[i])))
        kept: list = []
        for face in sorted(found, key=lambda f: -f[1]):
            if all(overlap(face[0], k[0]) <= 0.3 for k in kept):
                kept.append(face)
        return sorted(kept, key=lambda f: -(f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))


def cut_box(width: int, height: int, face: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """The face and shoulders around a face box: ABOVE and BELOW face heights from its centre, at
    ASPECT, centred on it, moved into the picture where it overhangs and shrunk where it can't fit."""
    x1, y1, x2, y2 = face
    fh = y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = (ABOVE + BELOW) * fh
    w = ASPECT * h
    if w > width:
        w, h = width, width / ASPECT
    if h > height:
        w, h = height * ASPECT, height
    left = min(max(cx - w / 2, 0.0), width - w)
    top = min(max(cy - ABOVE * fh, 0.0), height - h)
    return round(left), round(top), round(left + w), round(top + h)
