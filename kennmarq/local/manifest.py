"""Local mode's manifest: made on this machine, signed by Kennmarq.

1. The manifest is the same as a cloud mark's: the AI source
   type, the watermark's soft binding, the model, the app, the generator; no thumbnail.
2. c2pa signs it here with a throwaway chain (throwaway.py), with the network closed.
3. The manifest store c2pa embedded goes to POST /v1/sign: about 13 KB of claim and assertions,
   never pixels. Kennmarq checks it, signs the same claim with its own key, time-stamps it, and
   sends the store back with its signature in place of the throwaway one, at the same size.
4. replace_store() puts the signed store in the file where the throwaway one was. The file's hard
   binding leaves the store out, so it still holds.
5. validate() reads the file with Kennmarq's roots as the only trust anchors; a file that isn't
   Trusted with a validated time-stamp is never written (fail closed, as the cloud's read-back).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import struct
import zlib
from typing import Any

from .throwaway import Throwaway

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
SOURCE_TYPES = {
    "created": IPTC + "trainedAlgorithmicMedia",
    "composite": IPTC + "compositeWithTrainedAlgorithmicMedia",
    "edited": IPTC + "compositeWithTrainedAlgorithmicMedia",
}
WATERMARK_ALG = "com.adobe.trustmark.Q"
PARENT_ID = "parent"
INPUT_ID = "input"

# c2pa never gets the network here: no remote manifests, no OCSP, no redirects, as when the service reads uploads.
NO_NETWORK = {
    "core": {"allowed_network_hosts": [], "allow_redirects": False},
    "verify": {"remote_manifest_fetch": False, "ocsp_fetch": False},
}
SIGN_SETTINGS = {"builder": {"thumbnail": {"enabled": False}}, **NO_NETWORK}


class ManifestError(RuntimeError):
    """The file or its manifest isn't what local mode expects."""


def definition(*, brand: str, version: str, brand_slug: str, media_type: str, source: str,
               watermark_soft_binding: str | None, parent: bool, has_input: bool, model: str | None = None,
               model_version: str | None = None, app_name: str | None = None, title: str | None = None,
               provider: str | None = None, app_version: str | None = None) -> dict:
    """The manifest a cloud mark carries (marking.py's manifest_definition), for local mode: the AI
    system (app_name, app_version) as the created action's agent, never our own name, and the provider,
    model and model version in com.<brand>.generator (California's latent disclosure, SB 1000)."""
    inception: dict[str, Any] = {
        "action": "c2pa.opened" if parent else "c2pa.created",
        "digitalSourceType": SOURCE_TYPES[source],
        "when": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if app_name:
        inception["softwareAgent"] = {"name": app_name, **({"version": app_version} if app_version else {})}
    if parent:
        inception["parameters"] = {"ingredientIds": [PARENT_ID]}
    elif has_input:
        inception["parameters"] = {"ingredientIds": [INPUT_ID]}
    actions = [inception]
    assertions: list[dict] = [{"label": "c2pa.actions.v2", "data": {"actions": actions}}]
    if watermark_soft_binding is not None:
        actions.append({"action": "c2pa.watermarked.bound", "softwareAgent": {"name": brand, "version": version}})
        assertions.append({"label": "c2pa.soft-binding",
                           "data": {"alg": WATERMARK_ALG, "blocks": [{"scope": {}, "value": watermark_soft_binding}]}})
    generator = {k: v for k, v in (("provider", provider), ("model", model), ("model_version", model_version)) if v}
    if generator:
        assertions.append({"label": f"com.{brand_slug}.generator", "data": generator})
    out: dict[str, Any] = {
        "claim_generator_info": [{"name": brand, "version": version}],
        "format": media_type,
        "assertions": assertions,
    }
    if title:
        out["title"] = title
    return out


def has_credentials(data: bytes, media_type: str) -> bool:
    """Whether a file carries C2PA credentials c2pa can read (offline), to keep as an ingredient."""
    import c2pa

    try:
        with c2pa.Context.from_dict(NO_NETWORK) as context:
            reader = c2pa.Reader.try_create(media_type, io.BytesIO(data), None, context)
            if reader is None:
                return False
            reader.close()
            return True
    except c2pa.C2paError:
        return False


def sign_locally(encoded: bytes, media_type: str, manifest: dict, chain: Throwaway, *,
                 parent: tuple[bytes, str] | None = None, input_file: tuple[bytes, str] | None = None,
                 settings: dict | None = None) -> tuple[bytes, bytes]:
    """The file with its manifest signed by the throwaway chain, and the manifest store embedded."""
    import c2pa

    dest = io.BytesIO()
    with c2pa.Context.from_dict(settings or SIGN_SETTINGS) as context, \
            c2pa.Builder.from_json(json.dumps(manifest), context) as builder:
        if parent is not None:
            builder.add_ingredient({"title": "original", "relationship": "parentOf", "label": PARENT_ID},
                                   parent[1], io.BytesIO(parent[0]))
        if input_file is not None:
            builder.add_ingredient({"title": "input", "relationship": "inputTo", "label": INPUT_ID},
                                   input_file[1], io.BytesIO(input_file[0]))
        with c2pa.Signer.from_callback(chain.sign, c2pa.C2paSigningAlg.ES256, chain.chain_pem, None) as signer:
            store = builder.sign(signer, media_type, io.BytesIO(encoded), dest)
    return dest.getvalue(), bytes(store)


# -- putting the signed store back -------------------------------------------------------------------------

def replace_store(data: bytes, media_type: str, old: bytes, new: bytes) -> bytes:
    """The file with the manifest store `old` replaced by `new`, byte for byte where it was."""
    if len(old) != len(new):
        raise ManifestError("the signed store isn't the size of the one it replaces")
    if media_type == "image/png":
        return _replace_png(data, old, new)
    if media_type == "image/webp":
        return _replace_webp(data, old, new)
    if media_type == "image/jpeg":
        return _replace_jpeg(data, old, new)
    raise ManifestError(f"local mode doesn't handle {media_type}")


def _replace_png(data: bytes, old: bytes, new: bytes) -> bytes:
    at = 8
    while at + 12 <= len(data):
        (length,) = struct.unpack(">I", data[at:at + 4])
        kind = data[at + 4:at + 8]
        body = data[at + 8:at + 8 + length]
        if kind == b"caBX" and body == old:
            crc = struct.pack(">I", zlib.crc32(kind + new) & 0xFFFFFFFF)
            return data[:at + 8] + new + crc + data[at + 12 + length:]
        at += 12 + length
    raise ManifestError("no caBX chunk holds the manifest store")


def _replace_webp(data: bytes, old: bytes, new: bytes) -> bytes:
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ManifestError("not a RIFF WebP file")
    at = 12
    while at + 8 <= len(data):
        kind = data[at:at + 4]
        (length,) = struct.unpack("<I", data[at + 4:at + 8])
        if kind == b"C2PA" and data[at + 8:at + 8 + length] == old:
            return data[:at + 8] + new + data[at + 8 + length:]
        at += 8 + length + (length & 1)
    raise ManifestError("no C2PA chunk holds the manifest store")


def _replace_jpeg(data: bytes, old: bytes, new: bytes) -> bytes:
    """JPEG carries JUMBF in APP11 segments: "JP", its instance (En), its sequence (Z), then the box;
    a box split across segments repeats its LBox and TBox (and XLBox) at the start of every one
    after the first (ISO/IEC 19566-5 Annex B)."""
    if data[:2] != b"\xff\xd8":
        raise ManifestError("not a JPEG file")
    instances: dict[int, list[tuple[int, int, int]]] = {}
    at = 2
    while at + 4 <= len(data):
        if data[at] != 0xFF:
            raise ManifestError("a JPEG marker was expected")
        marker = data[at + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            at += 2
            continue
        if marker in (0xDA, 0xD9):  # the scan's data follows: no more metadata segments
            break
        (length,) = struct.unpack(">H", data[at + 2:at + 4])
        start, end = at + 4, at + 2 + length
        if marker == 0xEB and data[start:start + 2] == b"JP" and end - start >= 8:
            en = struct.unpack(">H", data[start + 2:start + 4])[0]
            z = struct.unpack(">I", data[start + 4:start + 8])[0]
            instances.setdefault(en, []).append((z, start + 8, end))
        at = end
    for parts in instances.values():
        parts.sort()
        spans: list[tuple[int, int]] = []
        for i, (_, s, e) in enumerate(parts):
            if i > 0:  # the repeated box header
                lbox = struct.unpack(">I", data[s:s + 4])[0]
                s += 16 if lbox == 1 else 8
            spans.append((s, e))
        if b"".join(data[s:e] for s, e in spans) != old:
            continue
        out = bytearray(data)
        done = 0
        for s, e in spans:
            out[s:e] = new[done:done + (e - s)]
            done += e - s
        return bytes(out)
    raise ManifestError("no APP11 segments hold the manifest store")


# -- the check before writing -----------------------------------------------------------------------------

def last_certificate(pem_chain: str) -> str:
    """The root at the end of one of Kennmarq's chains (GET /v1/certificate)."""
    begin = "-----BEGIN CERTIFICATE-----"
    blocks = [begin + b for b in pem_chain.split(begin)[1:]]
    if not blocks:
        raise ManifestError("a certificate chain without certificates")
    return blocks[-1].strip() + "\n"


def validate(data: bytes, media_type: str, certificates: dict) -> tuple[str, set[str], set[str]]:
    """The file as a validator with only Kennmarq's roots sees it: (state, success codes, failure codes)."""
    import c2pa

    manifest_roots = last_certificate(certificates["signing"]["live"]) + last_certificate(certificates["signing"]["test"])
    tsa_root = last_certificate(certificates["tsa"])
    settings = {
        "trust": {"anchors": [
            {"trust_anchors": manifest_roots, "trust_kind": "manifest", "trust_uri": "urn:kennmarq:roots:manifest"},
            {"trust_anchors": tsa_root, "trust_kind": "tsa", "trust_uri": "urn:kennmarq:roots:tsa"},
        ]},
        "core": dict(NO_NETWORK["core"]),
        "verify": {**NO_NETWORK["verify"], "verify_trust": True, "verify_timestamp_trust": True},
    }
    with c2pa.Context.from_dict(settings) as context, c2pa.Reader(media_type, io.BytesIO(data), None, context) as reader:
        state = str(reader.get_validation_state())
        results = reader.get_validation_results() or {}
    active = results.get("activeManifest") or {}
    codes = lambda key: {str(e.get("code")) for e in active.get(key) or [] if isinstance(e, dict)}  # noqa: E731
    return state, codes("success"), codes("failure")
