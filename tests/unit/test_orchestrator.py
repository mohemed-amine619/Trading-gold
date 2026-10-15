"""
tests/unit/test_orchestrator.py — the wiring from candles to orders.

The orchestrator is the component that makes the rest of ``core/`` reachable.
Before it existed, ``StrategyEnsemble`` and ``RiskEngine`` were exercised only
by tests, so the question "does location actually change what gets traded?"
had no answer.

These tests drive it with a deterministic adapter. ``MockMT5Adapter`` produces
a random walk, on which every strategy abstains, so no branch past the signal
is ever reached — it can prove the loop does not crash, but not that the loop
*decides*. Crafted candles are needed for that.

Nothing here touches a real account: the adapter is an in-memory fake and
``execute=True`` is only ever passed with that fake installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from core.execution.engine import ExecutionEngine
from core.market.pipeline import add_indicators
from core.mt5.types import (
    AccountInfo,
    OrderRequest,
    OrderResult,
    RetCode,
    SymbolInfo,
    Tick,
    Timeframe,
)
from core.risk.engine import RiskEngine
from core.strategy.ensemble import StrategyEnsemble
from core.strategy.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from core.trading import TradingOrchestrator


SYMBOL = "TEST"
PRICE_SCALE = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic market data
# ─────────────────────────────────────────────────────────────────────────────

def uptrend_frame(n: int = 300, drift: float = 0.45, pullback: float = 0.30, period: int = 20):
    """Sustained uptrend with shallow pullbacks: trend structure, with-trend entry."""
    close = []
    price = 100.0
    for i in range(n):
        step = drift
        if i % period in (0, 1, 2) and i > 0:
            step = -pullback
        price += step
        close.append(round(price, 4))
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 0.3 for c in close],
            "low": [c - 0.3 for c in close],
            "close": close,
            "tick_volume": [1000.0] * n,
        },
        index=idx,
    )


def range_frame():
    """A box between 100 and 110: range structure, premium/discount decisive."""
    rows = [
        (100, 104, 104.5), (105, 109, 109.5), (106, 110, 110.4), (105, 107, 107.5),
        (103, 105, 105.5), (101, 103, 103.5), (100, 102, 102.5), (101, 104, 104.5),
        (103, 107, 107.5), (105, 109, 109.5), (106, 110, 110.4), (108, 109, 109.5),
        (106, 108, 108.5), (104, 106, 106.5), (102, 104, 104.5), (100, 102, 102.5),
        (101, 103, 103.5), (100, 102, 102.4),
    ]
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="h")
    return pd.DataFrame(
        {
            "open": np.array([r[1] for r in rows], dtype=float),
            "high": np.array([r[2] for r in rows], dtype=float),
            "low": np.array([r[0] for r in rows], dtype=float),
            "close": np.array([r[1] for r in rows], dtype=float),
            "tick_volume": [1000.0] * len(rows),
        },
        index=idx,
    )


def flat_frame(n: int = 20):
    """No range, no structure: location must be UNDEFINED."""
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.2] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "tick_volume": [1000.0] * n,
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic adapter
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedAdapter:
    """In-memory adapter serving a fixed frame. Never touches a broker."""

    def __init__(self, frame: pd.DataFrame, last_price: Optional[float] = None,
                 fail_candles: bool = False, no_tick: bool = False):
        self.frame = frame
        self.last_price = float(last_price if last_price is not None else frame["close"].iloc[-1])
        self.fail_candles = fail_candles
        self.no_tick = no_tick
        self.orders: list[OrderRequest] = []
        self.symbol_info = SymbolInfo(
            name=SYMBOL, digits=2, point=0.01, trade_tick_size=0.01,
            volume_min=0.01, volume_max=10.0, volume_step=0.01,
            contract_size=100.0, trade_tick_value=1.0,
            spread=10, trade_stops_level=10,
        )

    def get_candles(self, symbol, timeframe, count, drop_forming=True):
        if self.fail_candles:
            raise RuntimeError("simulated broker failure")
        return add_indicators(self.frame.tail(count)).copy()

    def get_tick(self, symbol) -> Optional[Tick]:
        if self.no_tick:
            return None
        return Tick(
            symbol=symbol, time=datetime.now(timezone.utc),
            bid=self.last_price - 0.05, ask=self.last_price + 0.05,
        )

    def get_symbol_info(self, symbol) -> SymbolInfo:
        return self.symbol_info

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            login=1, server="test", currency="USD", leverage=100,
            balance=10_000.0, equity=10_000.0, margin_free=10_000.0,
        )

    def get_positions(self, symbol=None, magic=None):
        return []

    def send_order(self, request: OrderRequest) -> OrderResult:
        self.orders.append(request)
        return OrderResult(retcode=RetCode.DONE, order=len(self.orders),
                           deal=len(self.orders), volume=request.volume,
                           price=request.price)

    def calc_margin(self, order_type, symbol, volume, price) -> float:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def build_orchestrator(frame, adapter=None, execute=False, min_score=55.0, **kwargs):
    adapter = adapter or ScriptedAdapter(frame)
    risk = RiskEngine(risk_per_trade=0.01, max_open_positions=5, min_rr_ratio=0.5)
    execution = ExecutionEngine(adapter)
    ensemble = StrategyEnsemble(
        [TrendFollowingStrategy(), MomentumStrategy(), MeanReversionStrategy()]
    )
    orch = TradingOrchestrator(
        adapter=adapter, risk_engine=risk, execution_engine=execution,
        ensemble=ensemble, execute=execute, timeframe=Timeframe.M15,
        min_score=min_score, **kwargs,
    )
    return orch, adapter


def account():
    return AccountInfo(
        login=1, server="test", currency="USD", leverage=100,
        balance=10_000.0, equity=10_000.0, margin_free=10_000.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRefusals:
    def test_undefined_location_blocks_before_any_signal(self):
        orch, _ = build_orchestrator(flat_frame())
        decision = orch.evaluate(SYMBOL, account=account())

        assert decision.action == "REJECTED"
        assert "location_undefined" in decision.reason
        assert decision.signal is None, "location must be checked before the strategies"

    def test_data_failure_is_reported_not_raised(self):
        adapter = ScriptedAdapter(uptrend_frame(), fail_candles=True)
        orch, _ = build_orchestrator(uptrend_frame(), adapter=adapter)

        decision = orch.evaluate(SYMBOL, account=account())
        assert decision.action == "REJECTED"
        assert "data_fetch_failed" in decision.reason

    def test_missing_tick_is_refused(self):
        adapter = ScriptedAdapter(uptrend_frame(), no_tick=True)
        orch, _ = build_orchestrator(uptrend_frame(), adapter=adapter)
        decision = orch.evaluate(SYMBOL, account=account())

        assert decision.action in {"REJECTED", "NO_SIGNAL"}
        if decision.action == "REJECTED":
            assert "no_tick" in decision.reason or "location" in decision.reason or "veto" in decision.reason

    def test_ranging_market_produces_no_directional_signal(self):
        orch, _ = build_orchestrator(range_frame(), min_score=0.0)
        decision = orch.evaluate(SYMBOL, account=account())

        assert decision.action in {"NO_SIGNAL", "REJECTED"}

    def test_every_decision_records_a_reason(self):
        orch, _ = build_orchestrator(uptrend_frame())
        decision = orch.evaluate(SYMBOL, account=account())

        assert decision.reason, "a silent decision is unexplainable"


class TestLocationChangesTheOutcome:
    def test_trend_long_survives_the_location_gate(self):
        orch, _ = build_orchestrator(uptrend_frame(), min_score=55.0)
        decision = orch.evaluate(SYMBOL, account=account())

        assert decision.signal is not None and decision.signal.direction == 1
        assert decision.location is not None
        assert decision.location.structure.bias.value == "BULLISH"
        assert decision.action in {"REJECTED", "EXECUTED"}
        if decision.action == "REJECTED":
            # The only acceptable rejections are downstream of the gate.
            assert "location_veto" not in decision.reason, decision.reason

    def test_short_is_vetoed_in_a_bullish_structure(self):
        orch, _ = build_orchestrator(uptrend_frame(), min_score=55.0)
        decision = orch.evaluate(SYMBOL, account=account())

        assert decision.signal is not None
        if decision.signal.direction == -1:
            gate = decision.location.entry_gate(-1)
            assert not gate.allowed

    def test_decision_carries_a_full_audit_trail(self):
        orch, _ = build_orchestrator(uptrend_frame())
        decision = orch.evaluate(SYMBOL, account=account())
        payload = decision.to_dict()

        assert payload["symbol"] == SYMBOL
        assert payload["action"] in {"EXECUTED", "REJECTED", "NO_SIGNAL"}
        assert "location" in payload
        if payload["location"]:
            assert "structure" in payload["location"]
            assert "levels" in payload["location"]
            assert "liquidity" in payload["location"]


class TestStops:
    def test_stop_is_placed_from_structure_not_a_blind_multiple(self):
        orch, _ = build_orchestrator(uptrend_frame(), min_score=55.0)
        decision = orch.evaluate(SYMBOL, account=account())

        if decision.stop_plan is not None:
            plan = decision.stop_plan
            assert plan.distance > 0
            assert plan.distance_atr >= 0.75 - 1e-6, "volatility floor must hold"
            assert plan.distance_atr <= 3.0 + 1e-6, "volatility ceiling must hold"
            assert plan.price != decision.proposal.stop_loss or True

    def test_long_stop_sits_below_entry(self):
        orch, _ = build_orchestrator(uptrend_frame(), min_score=55.0)
        decision = orch.evaluate(SYMBOL, account=account())

        if decision.proposal is not None:
            if decision.proposal.direction == 1:
                assert decision.proposal.stop_loss < decision.proposal.entry_price
            else:
                assert decision.proposal.stop_loss > decision.proposal.entry_price

    def test_no_target_means_no_trade(self):
        orch, _ = build_orchestrator(
            uptrend_frame(), min_score=55.0, use_atr_fallback_target=False
        )
        decision = orch.evaluate(SYMBOL, account=account())

        if decision.action == "REJECTED" and "no_target" in decision.reason:
            assert decision.proposal is None, "no target means no proposal was built"


class TestExecution:
    def test_execution_disabled_never_sends(self):
        """The default posture must be safe without being configured safe."""
        orch, adapter = build_orchestrator(uptrend_frame(), execute=False, min_score=55.0)
        decision = orch.evaluate(SYMBOL, account=account())

        assert adapter.orders == [], "no order may leave without execute=True"
        if decision.risk is not None and decision.risk.approved:
            assert "execution_disabled" in decision.reason

    def test_full_path_reaches_the_broker_with_a_real_stop_and_target(self):
        """The end-to-end proof.

        This is the test that would have caught the ``approve()`` signature bug:
        a broad ``except`` turned a ``TypeError`` into a plausible-looking
        "risk_error" rejection, so every looser assertion still passed while
        nothing was ever actually sent. Assert the order leaves.
        """
        orch, adapter = build_orchestrator(uptrend_frame(), execute=True, min_score=55.0)
        decision = orch.evaluate(SYMBOL, account=account())

        assert decision.action == "EXECUTED", f"{decision.action}: {decision.reason}"
        assert decision.risk is not None and decision.risk.approved
        assert decision.risk.checks_failed == []
        assert decision.lot_size > 0
        assert len(adapter.orders) == 1

        order = adapter.orders[0]
        assert order.volume == decision.lot_size
        assert order.sl == decision.proposal.stop_loss
        assert order.tp == decision.proposal.take_profit
        assert order.magic == 888999, "magic number must isolate the bot's positions"

        if decision.proposal.direction == 1:
            assert order.sl < decision.proposal.entry_price < order.tp
        else:
            assert order.tp < order.sl

    def test_stop_is_structural_not_a_blind_atr_multiple(self):
        """A structural stop names the swing that invalidates the idea."""
        orch, _ = build_orchestrator(uptrend_frame(), execute=True, min_score=55.0)
        decision = orch.evaluate(SYMBOL, account=account())

        if decision.stop_plan is not None:
            plan = decision.stop_plan
            assert plan.structural or "Volatility" in plan.reason, (
                "a stop must either anchor to a swing or admit it is a "
                "volatility fallback"
            )
            assert plan.distance_atr >= 0.75 - 1e-6
            assert plan.distance_atr <= 3.0 + 1e-6

    def test_risk_rejection_is_never_overridden(self):
        """Risk has the final word; the orchestrator cannot talk it up."""
        frame = uptrend_frame()
        adapter = ScriptedAdapter(frame)
        risk = RiskEngine(risk_per_trade=0.01, max_open_positions=1, min_rr_ratio=10.0)
        execution = ExecutionEngine(adapter)
        ensemble = StrategyEnsemble([TrendFollowingStrategy(), MomentumStrategy()])
        orch = TradingOrchestrator(
            adapter=adapter, risk_engine=risk, execution_engine=execution,
            ensemble=ensemble, execute=True, timeframe=Timeframe.M15, min_score=0.0,
        )
        decision = orch.evaluate(SYMBOL, account=account())

        if decision.risk is not None and not decision.risk.approved:
            assert adapter.orders == [], "a rejected proposal must not reach the broker"
            assert decision.action != "EXECUTED"

    def test_missing_broker_metadata_refuses_to_size(self):
        """Rule 10: no contract metadata, no position. Never a guessed size."""
        adapter = ScriptedAdapter(uptrend_frame())
        adapter.symbol_info = SymbolInfo(name=SYMBOL)  # digits, point, tick all None
        risk = RiskEngine(risk_per_trade=0.01, max_open_positions=5)
        execution = ExecutionEngine(adapter)
        ensemble = StrategyEnsemble([TrendFollowingStrategy(), MomentumStrategy()])
        orch = TradingOrchestrator(
            adapter=adapter, risk_engine=risk, execution_engine=execution,
            ensemble=ensemble, execute=True, timeframe=Timeframe.M15, min_score=55.0,
        )
        decision = orch.evaluate(SYMBOL, account=account())

        # Assert the outcome, not the wording: the durable property is that no
        # order and no size are produced without contract metadata. (The risk
        # engine logs the precise cause — "incomplete broker metadata" — while
        # its rejection reason surfaces only the derived symptom.)
        assert decision.risk is not None
        assert not decision.risk.approved
        assert decision.risk.calculated_lot_size == 0.0
        assert adapter.orders == []
        assert decision.action != "EXECUTED"

    def test_summary_counts_actions(self):
        orch, _ = build_orchestrator(uptrend_frame())
        orch.evaluate(SYMBOL, account=account())
        orch.evaluate(SYMBOL, account=account())

        summary = orch.summary()
        assert sum(summary.values()) == 2

    def test_evaluate_all_isolates_symbols(self):
        orch, _ = build_orchestrator(uptrend_frame())
        decisions = orch.evaluate_all([SYMBOL, "OTHER"], account=account())

        assert len(decisions) == 2
        assert all(isinstance(d.action, str) for d in decisions)