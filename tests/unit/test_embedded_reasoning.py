"""Tests for the built-in (embedded) reasoning provider's position management.

These pin the routing and the RR logic. They do not show the confidence values
are calibrated; those are still fixed constants and are unmeasured.
"""

import json

from core.ai.provider import EmbeddedReasoningProvider


def _pm(text: str) -> dict:
    raw = EmbeddedReasoningProvider().generate(text, "", json_mode=True)
    return json.loads(raw)


def test_losing_position_is_never_secured_even_with_a_pinbar():
    out = _pm("Position management. RR: -0.3, floating_pnl -2.0, opposite pinbar")
    assert out["action"] != "SECURE_PROFIT"


def test_profitable_position_with_opposite_pinbar_secures_profit():
    out = _pm("Position management. RR: 0.8, opposite pinbar formed")
    assert out["action"] == "SECURE_PROFIT"


def test_profitable_position_moves_to_breakeven_when_not_yet_there():
    out = _pm("Position management. RR: 0.6, price trending, structure intact")
    assert out["action"] == "MOVE_TO_BREAKEVEN"


def test_profitable_position_already_at_breakeven_holds():
    out = _pm("Position management. RR: 0.6, stop already at breakeven")
    assert out["action"] == "HOLD"


def test_losing_position_holds_without_other_signals():
    out = _pm("Position management. RR: -0.5, drifting, no signal")
    assert out["action"] == "HOLD"


def test_giveback_from_a_strong_profit_tightens_stop():
    out = _pm("Position management. RR: 2.1, profit giveback from peak")
    assert out["action"] == "TIGHTEN_SL"
    assert out["confidence"] == 88.0


def test_giveback_from_a_weak_profit_tightens_stop_at_lower_confidence():
    out = _pm("Position management. RR: 0.4, retracement from peak")
    assert out["action"] == "TIGHTEN_SL"
    assert out["confidence"] == 82.0


def test_the_word_reaction_does_not_route_to_position_management():
    out = json.loads(EmbeddedReasoningProvider().generate(
        "Symbol: XAUUSDm\nReaction to the last swing was weak. Output JSON.",
        "", json_mode=True,
    ))
    assert "action" not in out
    assert out["decision"] == "PASS"


def test_the_word_transaction_does_not_route_to_position_management():
    out = json.loads(EmbeddedReasoningProvider().generate(
        "Transaction log for XAUUSDm. Output JSON.", "", json_mode=True,
    ))
    assert "action" not in out


def test_an_explicit_action_field_still_routes_to_position_management():
    out = _pm('Position management. action? RR: 0.9')
    assert "action" in out
