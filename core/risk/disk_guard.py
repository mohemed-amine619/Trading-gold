"""
core/risk/disk_guard.py — refuse NEW entries while the data disk is nearly full.

A full disk does not crash the bot (a failed log write is reported and
swallowed), but it silently stops the things that protect the account from
being saved: the risk state (``data/state/risk_<symbol>.json``: daily start,
loss streak, halt), the trade history in SQLite, and trained models. A position
opened in that state may not survive a restart with its risk accounting intact.

So below ``min_free_mb`` no new position is opened. Open positions are still
managed — this only stops entries, it never closes or loosens anything.

If free space cannot be measured the answer is UNKNOWN and is reported as such,
but it does not block: a filesystem that cannot report its size is not evidence
of a full one, and blocking forever on it would be a silent outage. Pure apart
from the injected ``usage`` call; no broker, no clock.
"""

from __future__ import annotations

import shutil
from typing import Callable, Optional, Tuple

_MB = 1024 * 1024


def free_mb(path: str, usage: Callable = shutil.disk_usage) -> Optional[float]:
    """Free space at ``path`` in MiB, or ``None`` when it cannot be measured."""
    try:
        return usage(path).free / _MB
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def low_disk_reason(path: str, min_free_mb: float,
                    usage: Callable = shutil.disk_usage) -> Tuple[str, Optional[float]]:
    """``(reason, free)``: ``reason`` is non-empty when entries must stop.

    ``min_free_mb <= 0`` disables the guard. ``free`` is ``None`` when unknown.
    """
    if not min_free_mb or min_free_mb <= 0:
        return "", None
    free = free_mb(path, usage)
    if free is None or free >= min_free_mb:
        return "", free
    return (f"low disk space: {free:.0f} MB free < {min_free_mb:.0f} MB "
            "(risk state and trade history may not be saved)"), free
