"""
core/ai/validation.py — leak-free train / validation splitting for time series.

Why a purge gap exists
----------------------
A label on bar *t* is decided by bars *t+1 … t+h*. If the training set ends at
bar *T* and the validation set starts at bar *T+1*, the last training labels
were decided by prices that sit inside the validation window. The model is
then scored on outcomes it has effectively already seen, and the out-of-sample
accuracy used by the promotion gate is optimistic.

Dropping the last ``gap`` training rows, where ``gap`` is the label horizon,
removes that overlap. Pure functions, no pandas or broker SDK: safe to import
anywhere and trivial to test.
"""

from __future__ import annotations

from typing import Tuple


def purged_split(n_rows: int, n_val: int, gap: int) -> Tuple[slice, slice]:
    """Return ``(train, validation)`` slices for a time-ordered sample.

    The validation block is the last ``n_val`` rows. Training is everything
    before it, minus the last ``gap`` rows so no training label reaches into
    the validation window.

    ``gap`` is counted in *rows*. Rows are one per bar, and rows that were
    dropped for a missing feature or an ambiguous label only widen the real
    bar distance between train and validation, so a row gap of ``h`` always
    guarantees at least ``h`` bars of separation.

    Raises ``ValueError`` when the split would leave no training rows or no
    validation rows, rather than returning an empty slice that fails later.
    """
    if n_val <= 0:
        raise ValueError("n_val must be positive")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    train_end = n_rows - n_val - gap
    if train_end <= 0:
        raise ValueError(
            f"no training rows left: n_rows={n_rows} n_val={n_val} gap={gap}")
    return slice(0, train_end), slice(n_rows - n_val, n_rows)
