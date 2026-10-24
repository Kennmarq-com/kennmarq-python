"""The safety check's models, each pinned to a commit and checked against its SHA-256.

| Folder under MODELS_DIR       | Model                                              | Licence    | Reads                      |
|-------------------------------|----------------------------------------------------|------------|----------------------------|
| safety/falconsai_nsfw         | Falconsai/nsfw_image_detection (ViT-B/16)          | Apache-2.0 | nudity                     |
| safety/nudenet                | NudeNet 3.4.2's 320n detector                      | MIT        | exposed parts, a 2nd read  |
| safety/siglip_base_224        | google/siglip-base-patch16-224 (its vision half)   | Apache-2.0 | photo or drawing           |
| safety/civitai_age_vit        | Civitai/age-vit (ViT-B/16)                         | Apache-2.0 | a face: minor or adult     |
| safety/siglip2_age            | prithivMLmods/Age-Classification-SigLIP2           | Apache-2.0 | a face: five age bands     |
| safety/wd_vit_tagger_v3       | SmilingWolf/wd-vit-tagger-v3 (ONNX)                | Apache-2.0 | a drawn character's age    |

YuNet (OpenCV Zoo's face finder, MIT) and SigLIP's prompt embeddings are small and ship next to
this file, in assets/, pinned below too.

Kennmarq.com's service and its SDK share this package, file for file. In the service's image the
models are baked in under MODELS_DIR. Elsewhere `fetch()`
downloads what's missing (about 2.3 GB), each file to a temporary name and renamed into place only
once its SHA-256 matches, so a partial or tampered file is never where the loaders look.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DOWNLOAD_TIMEOUT = 120

_FALCONSAI = "https://huggingface.co/Falconsai/nsfw_image_detection/resolve/04367978d3474804ab1a00a9bd6548b741764069"
_SIGLIP = "https://huggingface.co/google/siglip-base-patch16-224/resolve/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
_AGE_VIT = "https://huggingface.co/Civitai/age-vit/resolve/1d32ab709857fcdb8ee56c26dff9353694894958"
_SIGLIP2_AGE = "https://huggingface.co/prithivMLmods/Age-Classification-SigLIP2/resolve/b41e62c0f7b338910c3139c76c183f8dba8a8aca"
_TAGGER = "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f"
# The pinned NudeNet file is the 320n file inside PyPI's nudenet 3.4.2 wheel (the file on NudeNet's
# GitHub release, v3.4-weights, is a different one), so it's taken out of that wheel, itself pinned.
_NUDENET_WHEEL = ("https://files.pythonhosted.org/packages/1c/ee/1aa02d44ba958cc77e16ff1e41a0aac5e721037db7bf62b9c9d124917f87/"
                  "nudenet-3.4.2-py3-none-any.whl")
_NUDENET_WHEEL_SHA256 = "5937dbd84e5d8e5de038f08ffea5a1bb50a08475776bf2b4795914ce0eaf0331"

NSFW_DIR = "safety/falconsai_nsfw"
NUDENET_DIR = "safety/nudenet"
REAL_DIR = "safety/siglip_base_224"
AGE_DIR = "safety/civitai_age_vit"
CHILD_DIR = "safety/siglip2_age"
TAGGER_DIR = "safety/wd_vit_tagger_v3"


@dataclass(frozen=True)
class Asset:
    path: str  # under MODELS_DIR
    url: str
    sha256: str
    # For a file inside a zip (a wheel): the zip's own SHA-256, and the file's name in it.
    archive_sha256: str | None = None
    member: str | None = None


ASSETS = (
    Asset(f"{NSFW_DIR}/model.safetensors", f"{_FALCONSAI}/model.safetensors",
          "97b2ce64ec146884b37f98ee7944ca4891aa72f6827dc0cb10684a1cbecd5830"),
    Asset(f"{NSFW_DIR}/config.json", f"{_FALCONSAI}/config.json",
          "cd8c0a19566f199b5c64da402f1d6a9daf80d094fe6536befb060a2cc59449c2"),
    Asset(f"{NSFW_DIR}/preprocessor_config.json", f"{_FALCONSAI}/preprocessor_config.json",
          "a34861a24e781942424ad790b82bc99348404a2e64ea882de88c40905851698d"),
    Asset(f"{NUDENET_DIR}/320n.onnx", _NUDENET_WHEEL,
          "c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f",
          archive_sha256=_NUDENET_WHEEL_SHA256, member="nudenet/320n.onnx"),
    Asset(f"{REAL_DIR}/model.safetensors", f"{_SIGLIP}/model.safetensors",
          "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8"),
    Asset(f"{REAL_DIR}/config.json", f"{_SIGLIP}/config.json",
          "cd85b3d28829722820bcb89a2cfbb4160e55fd359249a3044da724166a8d9688"),
    Asset(f"{AGE_DIR}/model.safetensors", f"{_AGE_VIT}/model.safetensors",
          "56930ccd798c12618b2bd0b2cff559576dc5e505bea1569d86961e0dd81f7f12"),
    Asset(f"{AGE_DIR}/config.json", f"{_AGE_VIT}/config.json",
          "57093d192693de4acf5e914c468130d6a2de6aa3a61ee836037b559a6d5a6f0b"),
    Asset(f"{CHILD_DIR}/model.safetensors", f"{_SIGLIP2_AGE}/model.safetensors",
          "7e0205ae229b38817996d56b5106c6843cfa6c93eb460b43e85f8a8d2fabf90a"),
    Asset(f"{CHILD_DIR}/config.json", f"{_SIGLIP2_AGE}/config.json",
          "043cfa3d3ee98b2fe26db507503376dc4b336845ab2431d2712ec6360915f6e3"),
    Asset(f"{TAGGER_DIR}/model.onnx", f"{_TAGGER}/model.onnx",
          "35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6"),
    Asset(f"{TAGGER_DIR}/selected_tags.csv", f"{_TAGGER}/selected_tags.csv",
          "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217"),
)

# The files in assets/. YuNet's is the file OpenCV Zoo publishes; the prompts are age.py's REAL and
# DRAWN prompts through the pinned SigLIP's text half.
YUNET = ASSETS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
PROMPTS = ASSETS_DIR / "age_gate_prompts.npz"
PROMPTS_SHA256 = "53973a16c24b7df5d4cc03f892a51ad8bb8d91a6f8cc6d4a55afc2276219283c"


class SafetyUnavailable(RuntimeError):
    """The safety check's models could not be fetched, checked or loaded: the call fails (closed)."""


@contextlib.contextmanager
def quiet_transformers():
    """transformers' loaders print progress bars and config notes (SigLIP's checkpoints carry a text
    half and token IDs this check never uses): while they load, the log keeps errors only."""
    from transformers.utils import logging as hf_logging

    verbosity, bars = hf_logging.get_verbosity(), hf_logging.is_progress_bar_enabled()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.set_verbosity(verbosity)
        if bars:
            hf_logging.enable_progress_bar()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path: Path, sha256: str) -> None:
    if not path.is_file() or sha256_of(path) != sha256:
        raise SafetyUnavailable(f"{path} is missing or isn't the pinned file")


def _download(asset: Asset, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(asset.url, timeout=DOWNLOAD_TIMEOUT) as response, open(tmp, "wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
        if asset.member:
            if sha256_of(tmp) != asset.archive_sha256:
                raise SafetyUnavailable(f"{asset.url} does not match its pinned SHA-256")
            with zipfile.ZipFile(tmp) as archive:
                data = archive.read(asset.member)
            tmp.write_bytes(data)
        if sha256_of(tmp) != asset.sha256:
            raise SafetyUnavailable(f"{asset.url} does not match its pinned SHA-256")
        os.replace(tmp, target)
    except SafetyUnavailable:
        raise
    except Exception as exc:
        raise SafetyUnavailable(f"could not fetch {asset.path}: {type(exc).__name__}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def fetch(models_dir: Path, *, download: bool = True) -> None:
    """Make sure every pinned file is in `models_dir` and matches; download what's missing (unless
    `download` is off, as in a baked image, where a missing file is an error)."""
    for asset in ASSETS:
        target = Path(models_dir) / asset.path
        if target.is_file() and sha256_of(target) == asset.sha256:
            continue
        if not download:
            raise SafetyUnavailable(f"{target} is missing or isn't the pinned file")
        _download(asset, target)
    check_file(YUNET, YUNET_SHA256)
    check_file(PROMPTS, PROMPTS_SHA256)
