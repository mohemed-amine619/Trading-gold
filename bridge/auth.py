"""
bridge/auth.py — Authentication for the MT5 bridge.

The bridge exposes order-submission endpoints. It is reachable across the
network, so authentication is mandatory and the token comparison must not leak
timing information.

Rules enforced here:

* A missing or empty configured token puts the bridge in **refuse-all** mode
  rather than open mode. A misconfigured bridge must not become an unauthenticated
  trading endpoint.
* Token comparison is constant-time.
* Tokens are never logged, echoed in errors, or returned in responses.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

# Reasons surfaced to callers. Deliberately coarse — a caller must not be able
# to distinguish "wrong token" from "no token" by timing or message.
_AUTH_FAILED = "authentication failed"


def _constant_time_equals(a: str, b: str) -> bool:
    """Compare two strings without leaking length-independent timing."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


class BridgeAuth:
    """Bearer-token gate for protected bridge endpoints."""

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token if token else ""
        if not self._token:
            # Fail closed. Logged as a warning only — never the token itself.
            logger.error(
                "MT5_BRIDGE_TOKEN is not configured: all protected endpoints "
                "will refuse requests. Generate one with: openssl rand -hex 32"
            )

    @property
    def configured(self) -> bool:
        return bool(self._token)

    async def require_token(
        self,
        authorization: Optional[str] = Header(default=None),
        x_bridge_token: Optional[str] = Header(default=None, alias="X-Bridge-Token"),
    ) -> None:
        """
        FastAPI dependency enforcing a valid bearer token.

        Raises 401 when the token is absent, malformed or wrong. If no token is
        configured, every request is refused — never allowed through.
        """
        if not self._token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="bridge authentication is not configured on this host",
            )

        presented = ""
        if authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer" and value:
                presented = value.strip()
        if not presented and x_bridge_token:
            presented = x_bridge_token.strip()

        if not presented or not _constant_time_equals(presented, self._token):
            logger.warning("Rejected bridge request: invalid or missing token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=_AUTH_FAILED,
            )


def generate_token() -> str:
    """
    Generate a new shared secret.

    Used by the setup tooling so operators do not invent weak tokens.
    """
    return secrets.token_hex(32)


def token_from_env() -> Optional[str]:
    """Read the configured token from the environment."""
    return os.getenv("MT5_BRIDGE_TOKEN") or None