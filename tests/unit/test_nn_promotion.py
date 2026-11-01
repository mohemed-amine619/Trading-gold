"""
tests/unit/test_nn_promotion.py — the champion/challenger rule. Pure, torch-free.

An unmeasured edge (None) is never treated as zero, and a challenger must beat
the champion by a margin on the same bars before it replaces it.
"""

from __future__ import annotations

import pytest

from core.ai.nn_trainer import should_promote


def m(edge, trades=40):
    return {"edge": edge, "trades": trades}


def test_unmeasured_challenger_is_never_promoted():
    ok, why = should_promote(None, m(None))
    assert ok is False and "unmeasured" in why


def test_unmeasured_challenger_does_not_beat_even_a_bad_champion():
    assert should_promote(m(-0.20), m(None))[0] is False


@pytest.mark.parametrize("edge", [0.0, -0.01, -0.5])
def test_challenger_at_or_below_the_floor_is_refused(edge):
    assert should_promote(None, m(edge), floor=0.0)[0] is False


def test_floor_is_respected_when_set_above_zero():
    assert should_promote(None, m(0.04), floor=0.05)[0] is False
    assert should_promote(None, m(0.06), floor=0.05)[0] is True


def test_first_model_is_promoted_when_measured_and_above_the_floor():
    ok, why = should_promote(None, m(0.08))
    assert ok is True and "no champion" in why


def test_champion_with_unmeasured_edge_is_replaced_by_a_measured_challenger():
    ok, why = should_promote(m(None), m(0.05))
    assert ok is True and "unmeasured" in why


def test_challenger_must_beat_the_champion_by_the_tolerance():
    assert should_promote(m(0.05), m(0.06), tolerance=0.02)[0] is False   # +0.01 only
    assert should_promote(m(0.05), m(0.071), tolerance=0.02)[0] is True   # +0.021


def test_equal_edge_keeps_the_champion():
    assert should_promote(m(0.05), m(0.05), tolerance=0.0)[0] is False


def test_a_worse_challenger_is_refused():
    assert should_promote(m(0.10), m(0.04))[0] is False


def test_a_negative_champion_is_replaced_by_a_positive_challenger():
    assert should_promote(m(-0.05), m(0.03), tolerance=0.02)[0] is True


def test_reasons_are_always_given():
    for champion, challenger in [(None, m(None)), (m(0.1), m(0.2)), (m(0.1), m(0.0))]:
        assert should_promote(champion, challenger)[1]
