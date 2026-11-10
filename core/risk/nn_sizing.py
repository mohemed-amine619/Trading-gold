"""
core/risk/nn_sizing.py — how far the neural network's confidence may lift size,
and whether it is required to agree before an entry.

Two rules, both pure:

* ``nn_size_factor`` rewards confidence ABOVE the network's own threshold, up
  to ``1 + max_boost`` (never more, never below 1). The network must be voting
  the same direction as the entry. Nothing here looks at the day's P&L or at how
  much of a profit target is left: sizing up because a target is still far away
  is martingale-style loss chasing and is deliberately impossible.
* ``nn_entry_ok`` is the optional hard requirement: with it on, an entry needs
  the network to call the same direction at or above the threshold.

The caller must not apply the boost while a risk-reduction rung is active (the
drawdown ladder or a loss streak already shrank the size); ``apply_nn_boost``
enforces that.
"""

from __future__ import annotations


def nn_size_factor(nn_direction: int, direction: int, confidence: float,
                   threshold: float, max_boost: float) -> float:
    if max_boost <= 0 or direction == 0 or nn_direction != direction:
        return 1.0
    if not (0.0 < threshold < 1.0) or confidence is None or confidence < threshold:
        return 1.0
    span = 1.0 - threshold
    return 1.0 + max_boost * min(max((confidence - threshold) / span, 0.0), 1.0)


def apply_nn_boost(size_multiplier: float, factor: float) -> float:
    """Boost only a full-size entry: a reduced size stays reduced."""
    if size_multiplier < 1.0 or factor <= 1.0:
        return size_multiplier
    return size_multiplier * factor


def nn_entry_ok(nn_direction: int, direction: int, confidence: float,
                threshold: float) -> bool:
    return (direction != 0 and nn_direction == direction
            and confidence is not None and confidence >= threshold)
