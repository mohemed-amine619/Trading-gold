"""
core/accounts/security.py — password hashing and secret handling.

Standard library only (hashlib.scrypt, hmac, secrets), so the account system adds
no dependency to the Linux or Docker build.

* Passwords are stored as scrypt hashes with a per-user salt.
* Licence keys, session tokens and machine fingerprints are stored only as
  SHA-256 digests. A database leak then reveals no usable secret. The raw value
  is shown to the user once, at creation, and never again.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

# scrypt cost: about 16 MiB per hash. Fast enough for a login, slow enough to
# make offline guessing expensive. Parameters are stored with each hash, so
# they can be raised later without invalidating existing passwords.
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32

MIN_PASSWORD_LENGTH = 12


def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(digest_hex) // 2,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


# A throwaway hash checked when the email is unknown, so an attacker cannot
# tell "no such account" from "wrong password" by timing the response.
DUMMY_PASSWORD_HASH = hash_password("dummy-password-for-timing-only")


def new_secret(nbytes: int = 32) -> str:
    """A URL-safe random secret. Shown to the user once; store only its hash."""
    return secrets.token_urlsafe(nbytes)


def hash_secret(value: str) -> str:
    """SHA-256 digest for storage and lookup of a high-entropy secret.

    Unsalted on purpose: the input is 256 bits of randomness, not a password, so
    a salt buys nothing and lookup by digest needs it to be deterministic.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
