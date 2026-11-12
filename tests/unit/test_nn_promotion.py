"""
tests/unit/test_nn_promotion.py — the champion/challenger rule. Pure, torch-free.

An unmeasured edge (None) is never treated as zero, a challenger's edge must be
positive WITH CONFIDENCE (lower bound above the floor, never below 0), better
than the naive always-one-direction baseline, and show timing skill beyond its
direction; and it must beat the champion by a margin on the same bars before it
replaces it.
"""

from __future__ import annotations

import pytest

from core.ai.nn_trainer import should_promote

_SAME = object()


def m(edge, trades=40, lcb=_SAME, baseline=-0.10, excess=_SAME):
    """An ``edge_report``-shaped result. By default a measured edge also has a
    lower bound equal to it, a naive baseline well below it and positive timing
    skill, so the older tests exercise only the rule they are about."""
    return {"edge": edge, "trades": trades,
            "edge_lcb": edge if lcb is _SAME else lcb,
            "baseline_edge": baseline,
            "excess_lcb": edge if excess is _SAME else excess}


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


# ── The confidence, baseline and timing-skill gates ──────────────────────────

def test_a_positive_point_edge_without_confidence_is_refused():
    ok, why = should_promote(None, m(0.12, lcb=-0.03))
    assert ok is False and "not confidently above the floor" in why


def test_the_lower_bound_must_clear_a_positive_floor_not_just_the_edge():
    assert should_promote(None, m(0.20, lcb=0.04), floor=0.05)[0] is False
    assert should_promote(None, m(0.20, lcb=0.06), floor=0.05)[0] is True


def test_a_missing_lower_bound_is_unmeasured_and_refused():
    assert should_promote(None, m(0.30, lcb=None))[0] is False
    assert should_promote(None, {"edge": 0.30, "trades": 40})[0] is False   # old-shape result


def test_a_negative_floor_cannot_open_the_gate():
    # "Positive with confidence" is not negotiable: a negative floor is treated as 0.
    assert should_promote(None, m(0.10, lcb=-0.01), floor=-0.5)[0] is False


def test_no_better_than_the_naive_baseline_is_refused():
    ok, why = should_promote(None, m(0.30, lcb=0.12, baseline=0.30))
    assert ok is False and "naive" in why
    assert should_promote(None, m(0.30, lcb=0.12, baseline=0.35))[0] is False


def test_an_unmeasured_baseline_refuses():
    ok, why = should_promote(None, m(0.30, lcb=0.12, baseline=None))
    assert ok is False and "baseline" in why


def test_riding_the_trend_without_timing_skill_is_refused():
    ok, why = should_promote(None, m(0.30, lcb=0.12, baseline=0.25, excess=-0.05))
    assert ok is False and "timing skill" in why
    assert should_promote(None, m(0.30, lcb=0.12, baseline=0.25, excess=None))[0] is False


def test_a_clear_edge_is_promoted_and_the_reason_says_why():
    ok, why = should_promote(None, m(0.40, trades=60, lcb=0.18, baseline=0.02, excess=0.15))
    assert ok is True and "+0.400" in why and "60 trades" in why


def test_the_champion_tolerance_rule_is_unchanged_once_the_gates_pass():
    good = m(0.40, lcb=0.20, baseline=0.0, excess=0.20)
    assert should_promote(m(0.39), good, tolerance=0.02)[0] is False
    assert should_promote(m(0.37), good, tolerance=0.02)[0] is True
    # ...and a champion never lets a challenger skip the gates.
    assert should_promote(m(-0.5), m(0.30, lcb=0.12, baseline=0.30))[0] is False
