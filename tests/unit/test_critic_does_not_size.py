"""
tests/unit/test_critic_does_not_size.py — the LLM critic cannot raise position size.

The pre-trade critic returns a confidence. Before this fix, `_get_signal`
replaced the model's confidence with the critic's whenever the critic's was
higher. That confidence drives the AI size tiers in `_size_lot`, so an opinion
the model never validated could upsize a trade.

The embedded provider is a rule table that returns a fixed 83 or 86 for any
direction-matching setup. Under the old code that constant became the live
confidence, and a position was sized as if the validated model were 83% sure.

The critic may still approve or veto. It may not change the size.
"""

from __future__ import annotations

import types

from core.ai.dynamic_sltp import TradeCritiqueResult

from tests.unit.test_run_py_wiring import build_bot, run_and_stub, uptrend_frame  # noqa: F401


class _ModestAI:
    """A model that voted with real but modest confidence, above the 0.55 gate."""

    def update(self, frame): pass

    def predict(self, frame):
        return types.SimpleNamespace(direction=1, confidence=0.58, abstained=False,
                                     reason="AI LONG modest")


class _ApprovingCritic:
    """Approves every candidate with a high fixed confidence, like the embedded rule table."""

    def evaluate_trade_candidate(self, **kwargs):
        return TradeCritiqueResult(approved=True, confidence=83.0, decision="BUY",
                                   reasoning="rule table", risk_assessment="n/a")


def _bot(run_module):
    bot = build_bot(run_module, location_gate=False, use_session_filter=False,
                    min_signal_score=0, adx_min=0, fib_enabled=False,
                    ai_confidence_threshold=0.55)
    bot.ai = _ModestAI()
    bot._get_sltp_reasoner = lambda: _ApprovingCritic()
    return bot


def test_critic_confidence_does_not_replace_model_confidence(run_and_stub):
    run_module, _ = run_and_stub
    bot = _bot(run_module)
    frame = run_module._prepare(uptrend_frame(), bot.cfg)

    direction, score, conf, reason = bot._get_signal(frame, frame, frame)

    assert direction == 1, f"setup should have survived the gates: {reason}"
    assert reason == "signal"
    assert conf == 0.58, f"critic's 83 leaked into the sizing confidence: {conf}"


def test_critic_veto_still_stops_the_trade(run_and_stub):
    """The critic keeps its veto. Only its upward influence on size is removed."""
    run_module, _ = run_and_stub
    bot = _bot(run_module)

    class _Veto:
        def evaluate_trade_candidate(self, **kwargs):
            return TradeCritiqueResult(approved=False, confidence=90.0, decision="PASS",
                                       reasoning="trap", risk_assessment="n/a")

    bot._get_sltp_reasoner = lambda: _Veto()
    frame = run_module._prepare(uptrend_frame(), bot.cfg)

    direction, _, _, reason = bot._get_signal(frame, frame, frame)

    assert direction == 0
    assert "LLM critic veto" in reason
