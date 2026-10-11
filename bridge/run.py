"""
bridge/run.py — Entry point for the bridge service on the Windows VM.

    py bridge\run.py

Binds to the private/VPN interface by default rather than 0.0.0.0, so the
bridge is not exposed to every network the host happens to be attached to.

Configuration comes from the environment (or a .env file on the VM):
    MT5_BRIDGE_TOKEN           shared secret (REQUIRED — service refuses without it)
    MT5_BRIDGE_HOST            default 127.0.0.1
    MT5_BRIDGE_PORT            default 8100
    MT5_BRIDGE_ALLOW_WRITES   "1" to enable order submission (default: read-only)
    MT5_BRIDGE_EXPECTED_LOGIN  account number the terminal must be logged into
    MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / MT5_PATH   broker credentials
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow running as `py bridge\run.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional; env vars still work
    pass

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bridge")


def main() -> int:
    from bridge.app import create_app

    token = os.getenv("MT5_BRIDGE_TOKEN") or ""
    allow_writes = os.getenv("MT5_BRIDGE_ALLOW_WRITES", "0") == "1"
    expected = os.getenv("MT5_BRIDGE_EXPECTED_LOGIN")
    host = os.getenv("MT5_BRIDGE_HOST", "127.0.0.1")
    port = int(os.getenv("MT5_BRIDGE_PORT", "8100"))

    if not token:
        logger.error(
            "MT5_BRIDGE_TOKEN is not set. Generate one with 'openssl rand -hex 32' "
            "and set the SAME value on the Ubuntu host. Refusing to start: an "
            "unauthenticated trading endpoint must never run."
        )
        return 2

    logger.info("=" * 64)
    logger.info("MT5 Bridge %s starting", "1.0.0")
    logger.info("  bind            : %s:%s", host, port)
    logger.info("  expected login  : %s", expected or "(NOT SET — writes will refuse)")
    logger.info("  writes enabled  : %s", allow_writes)
    logger.info("  token           : configured (%d chars, value not logged)",
                len(token))
    if allow_writes:
        logger.warning("  !! ORDER SUBMISSION IS ENABLED — this bridge can place real orders")
    logger.info("=" * 64)

    app = create_app(allow_writes=allow_writes)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())