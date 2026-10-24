"""The throwaway chain local mode signs with first (manifest.py): a root and a claim-signing leaf
made in memory for this process, trusted by nobody, never saved. c2pa checks every signature against
the certificate it's given, and the certificate against C2PA's profile, so the manifest is signed
with this chain on this machine; Kennmarq then signs the same claim with its own key and puts its
signature in place of this one (POST /v1/sign). Nothing signed with this chain ever leaves the
process as a finished file.

The profile is the one Kennmarq.com's development certificates use, which c2pa accepts.
"""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID, ObjectIdentifier

C2PA_CLAIM_SIGNING = ObjectIdentifier("1.3.6.1.4.1.62558.2.1")
DOCUMENT_SIGNING = ObjectIdentifier("1.3.6.1.5.5.7.3.36")
CLAIM_SIGNING_EKUS = (C2PA_CLAIM_SIGNING, DOCUMENT_SIGNING, ExtendedKeyUsageOID.EMAIL_PROTECTION)
BACKDATE = dt.timedelta(minutes=5)
LIFETIME = dt.timedelta(days=1)


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _key_usage(*, digital_signature: bool = False, cert_sign: bool = False) -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=digital_signature, content_commitment=False, key_encipherment=False,
        data_encipherment=False, key_agreement=False, key_cert_sign=cert_sign, crl_sign=cert_sign,
        encipher_only=False, decipher_only=False,
    )


@dataclass(frozen=True)
class Throwaway:
    key: ec.EllipticCurvePrivateKey
    chain_pem: str

    def sign(self, data: bytes) -> bytes:
        """What c2pa's callback signer asks for: a DER ECDSA P-256 / SHA-256 signature."""
        return self.key.sign(bytes(data), ec.ECDSA(hashes.SHA256()))


def make() -> Throwaway:
    now = dt.datetime.now(dt.timezone.utc)
    start, end = now - BACKDATE, now + LIFETIME
    root_key = ec.generate_private_key(ec.SECP384R1())
    root_name = _name("Kennmarq local throwaway root")
    root = (
        x509.CertificateBuilder().subject_name(root_name).issuer_name(root_name).public_key(root_key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(start).not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(_key_usage(cert_sign=True), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
        .sign(root_key, hashes.SHA384())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder().subject_name(_name("Kennmarq local throwaway")).issuer_name(root_name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(start).not_valid_after(end)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(_key_usage(digital_signature=True), critical=True)
        .add_extension(x509.ExtendedKeyUsage(list(CLAIM_SIGNING_EKUS)), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
        .sign(root_key, hashes.SHA384())
    )
    pem = (leaf.public_bytes(serialization.Encoding.PEM) + root.public_bytes(serialization.Encoding.PEM)).decode("ascii")
    return Throwaway(key, pem)


_current: Throwaway | None = None
_lock = threading.Lock()


def current() -> Throwaway:
    """One chain per process, made on first use and renewed before its day runs out."""
    global _current
    with _lock:
        if _current is None or _expires_soon(_current):
            _current = make()
        return _current


def _expires_soon(t: Throwaway) -> bool:
    leaf = x509.load_pem_x509_certificates(t.chain_pem.encode("ascii"))[0]
    return leaf.not_valid_after_utc - dt.datetime.now(dt.timezone.utc) < dt.timedelta(hours=1)
