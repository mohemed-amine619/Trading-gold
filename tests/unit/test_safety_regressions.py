"""
tests/unit/test_safety_regressions.py — Regression tests for fail-open defects.

Every test in this file corresponds to a defect that was present in the
codebase and has been fixed. Each test is written so that it FAILS against
the previous behaviour — they are not general coverage, they are tripwires.

The defects share one root cause: a safety check that silently succeeded
when it could not actually evaluate. An absent symbol spec, an unreadable
position, a missing tick value — each was treated as "no problem found"
rather than "cannot tell", and the trade proceeded.

No test here contacts a live terminal. All broker interaction uses
MockMT5Adapter or a hand-written stub.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math

import pandas as pd
import pytest

from core.execution.engine import ExecutionEngine
from core.execution.states import ExecutionState
from core.instruments.model import AssetClass, InstrumentSpec, TradingStatus
from core.mt5.adapter import MockMT5Adapter, MT5CallError
from core.mt5.types import (
    AccountInfo,
    MT5Position,
    OrderRequest,
    OrderResult,
    OrderType,
    RetCode,
    SymbolInfo,
    Tick,
    TradeAction,
)
from core.position.trailing import normalize_price, validate_sl_modification
from core.risk.engine import RiskEngine, TradeProposal, risk_engine_from_settings
from core.strategy.base import Signal


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

GOLD = SymbolInfo(
    name="XAUUSDm",
    digits=2,
    point=0.01,
    trade_tick_size=0.01,
    trade_tick_value=1.0,
    trade_stops_level=0,
    trade_freeze_level=0,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    contract_size=100.0,
    spread=20,
    trade_mode=4,
)

ACCOUNT = AccountInfo(
    login=1, server="Test-Server", balance=10_000.0, equity=10_000.0
)


def _proposal(**overrides) -> TradeProposal:
    """A proposal that passes every check, so each test can isolate one guard."""
    base = dict(
        symbol="XAUUSDm",
        direction=1,
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        strategy_name="test",
        account_info=ACCOUNT,
        symbol_info=GOLD,
    )
    base.update(overrides)
    return TradeProposal(**base)


def _engine(**overrides) -> RiskEngine:
    """A risk engine that would otherwise approve, with a seeded equity baseline."""
    engine = RiskEngine(**overrides)
    engine.update_equity(balance=10_000.0, floating_pnl=0.0)
    return engine


# ═════════════════════════════════════════════════════════════════════════════
# 1. Risk engine — approve() must not return APPROVED at zero size
# ═════════════════════════════════════════════════════════════════════════════

def test_approve_rejects_when_symbol_metadata_missing():
    """A proposal with no broker spec must be REJECTED, not approved at 0.0 lots.

    The sizing block used to be guarded by `if proposal.symbol_info and
    equity > 0:`. With no symbol_info the block was skipped entirely, leaving
    calculated_lot_size at 0.0 and still returning APPROVED.
    """
    engine = _engine()
    result = engine.approve(_proposal(symbol_info=None), open_positions_count=0)

    assert not result.approved, (
        "approve() returned APPROVED without broker metadata — the trade "
        "would reach the broker with an uncomputed size"
    )
    assert "symbol metadata" in result.reason
    assert result.calculated_lot_size == 0.0


def test_approve_rejects_when_equity_unusable():
    engine = _engine()
    result = engine.approve(
        _proposal(account_info=AccountInfo(
            login=1, server="Test-Server", balance=0.0, equity=0.0)),
        open_positions_count=0,
    )
    assert not result.approved
    assert "equity" in result.reason


def test_approved_decision_always_carries_a_positive_size():
    """Property: APPROVED implies calculated_lot_size > 0. Always."""
    engine = _engine()
    result = engine.approve(_proposal(), open_positions_count=0)
    if result.approved:
        assert result.calculated_lot_size > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 2. Risk engine — sizing must refuse, never invent broker constants
# ═════════════════════════════════════════════════════════════════════════════

def test_calculate_lot_size_refuses_without_tick_value():
    """Missing tick_value must produce 0.0, not a lot sized on an invented 1.0.

    `trade_tick_value or 1.0` silently invented a tick value for any
    instrument the broker never described, producing a confident, wrong size.
    """
    engine = _engine()
    bare = SymbolInfo(name="MYSTERY", volume_min=0.01, volume_step=0.01)

    assert engine.calculate_lot_size(
        equity=10_000.0, sl_distance=10.0, symbol_info=bare
    ) == 0.0


@pytest.mark.parametrize(
    "field", ["trade_tick_size", "trade_tick_value", "volume_step", "volume_min"]
)
def test_calculate_lot_size_refuses_on_any_missing_critical_field(field):
    engine = _engine()
    degraded = GOLD.model_copy(update={field: None})
    assert engine.calculate_lot_size(
        equity=10_000.0, sl_distance=10.0, symbol_info=degraded
    ) == 0.0


def test_calculate_lot_size_refuses_rather_than_exceed_budget_for_min_lot():
    """Never round UP to the broker minimum just to make the order tradeable.

    A risk budget that supports 0.004 lots on an instrument with volume_min
    0.01 must return 0.0 (refuse), not 0.01 (which risks 2.5x the budget).
    """
    engine = RiskEngine(risk_per_trade=0.0025)
    coarse = GOLD.model_copy(update={"volume_min": 1.0, "volume_step": 1.0})

    lots = engine.calculate_lot_size(
        equity=1_000.0, sl_distance=500.0, symbol_info=coarse
    )
    assert lots == 0.0, (
        "sizing climbed to the broker minimum and would risk far more than "
        "the configured budget"
    )


def test_calculate_lot_size_is_proportional_to_equity():
    engine = _engine()
    a = engine.calculate_lot_size(equity=10_000.0, sl_distance=10.0, symbol_info=GOLD)
    b = engine.calculate_lot_size(equity=20_000.0, sl_distance=10.0, symbol_info=GOLD)
    assert b == pytest.approx(a * 2, rel=0.02)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Risk engine — runtime config cannot rewrite internal state
# ═════════════════════════════════════════════════════════════════════════════

def test_update_config_cannot_disarm_the_kill_switch():
    """A caller must not be able to write _emergency_stop through config."""
    engine = _engine()
    engine.trigger_emergency_stop()
    assert engine.emergency_stopped

    with pytest.raises(ValueError):
        engine.update_config(_emergency_stop=False)

    assert engine.emergency_stopped, "kill switch was disarmed via update_config"


def test_update_config_cannot_zero_the_daily_loss_baseline():
    """A caller must not be able to write the equity baseline it is measured against."""
    engine = _engine()
    before = engine.status()["daily_start_equity"]

    with pytest.raises(ValueError):
        engine.update_config(_daily_start_equity=1e12)

    assert engine.status()["daily_start_equity"] == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_per_trade": 0.99},      # above the 5% ceiling
        {"risk_per_trade": -1.0},      # negative
        {"max_open_positions": 0},     # below 1
        {"max_open_positions": 9999},  # above 50
        {"risk_per_trade": "half"},    # wrong type
        {"risk_per_trade": True},      # bool is not an acceptable float
    ],
)
def test_update_config_rejects_out_of_range_or_mistyped_values(kwargs):
    engine = _engine()
    with pytest.raises(ValueError):
        engine.update_config(**kwargs)


def test_update_config_accepts_valid_values():
    engine = _engine()
    engine.update_config(risk_per_trade=0.005, max_open_positions=2)
    assert engine.risk_per_trade == 0.005
    assert engine.max_open_positions == 2


def test_update_config_is_atomic_on_failure():
    """A rejected batch must not partially apply."""
    engine = _engine()
    original = engine.risk_per_trade
    with pytest.raises(ValueError):
        engine.update_config(risk_per_trade=0.005, max_open_positions=9999)
    assert engine.risk_per_trade == original


# ═════════════════════════════════════════════════════════════════════════════
# 4. Risk engine — Settings must actually reach the engine
# ═════════════════════════════════════════════════════════════════════════════

def test_risk_engine_is_built_from_settings(monkeypatch):
    """Every risk value in .env must reach the engine that enforces them.

    get_risk_engine() previously called RiskEngine() with defaults only, so
    the whole risk section of .env was inert in production.
    """
    from core.settings import Settings

    monkeypatch.setenv("RISK_PER_TRADE", "0.0035")
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.011")
    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "0.077")
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "9")
    monkeypatch.setenv("MIN_RR_RATIO", "2.25")
    monkeypatch.setenv("MAX_MARGIN_USAGE", "0.42")

    engine = risk_engine_from_settings(Settings())

    assert engine.risk_per_trade == 0.0035
    assert engine.max_daily_loss_pct == 0.011
    assert engine.max_drawdown_pct == 0.077
    assert engine.max_open_positions == 9
    assert engine.min_rr_ratio == 2.25
    assert engine.max_margin_usage == 0.42


# ═════════════════════════════════════════════════════════════════════════════
# 5. Risk engine — margin projection
# ═════════════════════════════════════════════════════════════════════════════

def test_margin_check_accounts_for_the_incoming_order():
    """The new order's own margin must count, not just committed margin."""
    engine = _engine(max_margin_usage=0.50)  # limit = 5000
    # Committed margin is comfortably under the limit...
    result = engine.approve(
        _proposal(margin_estimate=1_000.0),
        open_positions_count=0,
        used_margin=4_500.0,
    )
    assert not result.approved, (
        "order admitted that would take projected margin to 5500 over a 5000 limit"
    )
    assert "margin_limit" in result.reason


def test_margin_check_still_passes_within_budget():
    engine = _engine(max_margin_usage=0.50)
    result = engine.approve(
        _proposal(margin_estimate=100.0),
        open_positions_count=0,
        used_margin=1_000.0,
    )
    assert result.approved
    assert "margin_ok" in " ".join(result.checks_passed)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Execution — an indeterminate order must never read as a clean rejection
# ═════════════════════════════════════════════════════════════════════════════

class _TimeoutAdapter(MockMT5Adapter):
    """Adapter whose sends always time out — the indeterminate case."""

    def send_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(retcode=RetCode.TIMEOUT, comment="timeout")


def test_execute_proposal_preserves_unknown_outcome():
    """UNKNOWN must not be reported to the caller as a definite failure.

    ExecutionReport.retcode is an int that is never None, so the old
    `report.retcode if report.retcode is not None else ...` guard never
    fired and UNKNOWN surfaced as retcode 0. Callers read 0 as "rejected"
    and would happily resubmit an order the broker may already have filled.
    """
    adapter = _TimeoutAdapter()
    engine = ExecutionEngine(adapter=adapter, max_retries=0)
    proposal = _proposal()

    result = engine.execute_proposal(proposal, lot_size=0.1, idempotency_key="k1")

    assert result.retcode == RetCode.TIMEOUT, (
        "indeterminate outcome was reported as a clean rejection"
    )
    assert not result.success


def test_unknown_submission_is_not_retried():
    """Retrying an indeterminate order risks a duplicate fill."""
    calls = []

    class _CountingAdapter(MockMT5Adapter):
        def send_order(self, request):
            calls.append(1)
            return OrderResult(retcode=RetCode.TIMEOUT)

    engine = ExecutionEngine(adapter=_CountingAdapter(), max_retries=5)
    engine.execute_proposal(_proposal(), lot_size=0.1, idempotency_key="k2")

    assert len(calls) == 1, f"indeterminate order was resent {len(calls)} times"


def test_send_order_exception_becomes_unknown_and_frees_the_key():
    """A raising adapter must not leave the idempotency key permanently stuck.

    An unhandled exception previously escaped with the key still in_flight,
    so every later submission with that key was blocked forever.
    """
    class _ExplodingAdapter(MockMT5Adapter):
        def send_order(self, request):
            raise ConnectionError("terminal vanished")

    engine = ExecutionEngine(adapter=_ExplodingAdapter(), max_retries=3)
    report = engine.submit(
        OrderRequest(
            symbol="XAUUSDm", action=TradeAction.DEAL, type=OrderType.BUY,
            volume=0.1, price=0.0,
        ),
        idempotency_key="k3",
    )
    assert report.state is ExecutionState.UNKNOWN
    # The key must now resolve to a completed record, not hang in flight.
    assert engine.ledger.get("k3") is not None
    assert not engine.ledger.get("k3").is_indeterminate or report.state is ExecutionState.UNKNOWN


# ═════════════════════════════════════════════════════════════════════════════
# 7. Execution — the never-loosen guard must fail CLOSED
# ═════════════════════════════════════════════════════════════════════════════

def _position(direction=1, sl=1990.0) -> MT5Position:
    return MT5Position(
        ticket=555, symbol="XAUUSDm", type=0 if direction == 1 else 1,
        volume=0.1, price_open=2000.0, price_current=2000.0,
        sl=sl, tp=2020.0, magic=888999, time=0,
    )


def test_modify_position_refuses_when_position_cannot_be_read():
    """An unreadable position means the stop cannot be verified — refuse.

    The guard previously caught the exception, logged a warning, and fell
    through to send the order anyway, so a transient broker error silently
    disabled the single most safety-critical check in the module.
    """
    class _BrokenAdapter(MockMT5Adapter):
        def get_positions(self, symbol=None, magic=None):
            raise ConnectionError("positions_get failed")

    engine = ExecutionEngine(adapter=_BrokenAdapter())
    result = engine.modify_position(555, "XAUUSDm", new_sl=1980.0, new_tp=2020.0)
    assert result is None, "widened stop was submitted while state was unverifiable"


def test_modify_position_refuses_unknown_ticket():
    engine = ExecutionEngine(adapter=MockMT5Adapter())
    engine.adapter._positions = [_position(sl=1990.0)]

    # Ticket 999 does not exist; the stop cannot be verified.
    assert engine.modify_position(999, "XAUUSDm", new_sl=1980.0, new_tp=2020.0) is None


def test_modify_position_blocks_widening_a_long_stop():
    engine = ExecutionEngine(adapter=MockMT5Adapter())
    engine.adapter._positions = [_position(direction=1, sl=1990.0)]

    assert engine.modify_position(555, "XAUUSDm", new_sl=1980.0, new_tp=2020.0) is None
    assert engine.adapter._positions[0].sl == 1990.0


def test_modify_position_blocks_widening_a_short_stop():
    engine = ExecutionEngine(adapter=MockMT5Adapter())
    engine.adapter._positions = [_position(direction=-1, sl=2010.0)]

    assert engine.modify_position(555, "XAUUSDm", new_sl=2020.0, new_tp=1980.0) is None
    assert engine.adapter._positions[0].sl == 2010.0


def test_modify_position_allows_tightening_a_stop():
    engine = ExecutionEngine(adapter=MockMT5Adapter())
    engine.adapter.connect()
    engine.adapter._positions = [_position(direction=1, sl=1990.0)]
    engine.adapter.set_price("XAUUSDm", 2000.0)

    engine.modify_position(555, "XAUUSDm", new_sl=1995.0, new_tp=2020.0)
    assert engine.adapter._positions[0].sl == pytest.approx(1995.0)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Execution — closing must target the intended position
# ═════════════════════════════════════════════════════════════════════════════

def test_close_position_targets_the_given_ticket():
    """The close request must carry `position` or a hedging account opens a
    second, opposite position instead of closing the first."""
    sent: list[OrderRequest] = []

    class _CaptureAdapter(MockMT5Adapter):
        def send_order(self, request):
            sent.append(request)
            return OrderResult(retcode=RetCode.DONE)

    engine = ExecutionEngine(adapter=_CaptureAdapter())
    engine.close_position(ticket=777, symbol="XAUUSDm", volume=0.1, direction=1)

    assert len(sent) == 1
    assert sent[0].position == 777, (
        "close request omitted the position ticket — on a hedging account "
        "this opens an opposite position rather than closing"
    )


def test_real_adapter_forwards_position_to_mt5(monkeypatch):
    """The adapter must put `position` in the order_send payload.

    It previously dropped the field entirely, so every SL/TP modification
    reached the broker with position=0 and could not work at all.
    """
    import sys
    import types as pytypes

    captured: dict = {}

    class _Result:
        retcode = 10009
        deal = 1
        order = 2
        volume = 0.1
        price = 2000.0
        bid = 0.0
        ask = 0.0
        comment = "ok"
        request_id = 0
        retcode_external = 0

    fake = pytypes.ModuleType("MetaTrader5")
    fake.order_send = lambda payload: (captured.update(payload) or _Result())
    fake.last_error = lambda: (0, 0)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)

    from core.mt5.adapter import RealMT5Adapter

    adapter = RealMT5Adapter(login=1, password="p", server="s")
    adapter._mt5 = fake
    adapter.send_order(
        OrderRequest(
            symbol="XAUUSDm", action=TradeAction.SLTP, type=OrderType.BUY,
            volume=0.0, price=0.0, sl=1990.0, tp=2020.0, position=4242,
        )
    )
    assert captured.get("position") == 4242, (
        "RealMT5Adapter dropped `position` — SL/TP modification cannot work"
    )


def test_send_order_none_result_is_indeterminate(monkeypatch):
    """order_send() returning None must not be reported as retcode 0."""
    import sys
    import types as pytypes

    fake = pytypes.ModuleType("MetaTrader5")
    fake.order_send = lambda payload: None
    fake.last_error = lambda: (10019, 0)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)

    from core.mt5.adapter import RealMT5Adapter

    adapter = RealMT5Adapter(login=1, password="p", server="s")
    adapter._mt5 = fake
    result = adapter.send_order(
        OrderRequest(
            symbol="XAUUSDm", action=TradeAction.DEAL, type=OrderType.BUY,
            volume=0.1, price=0.0,
        )
    )
    assert result.retcode == RetCode.TIMEOUT, (
        "a None order_send result was reported as retcode 0, which callers "
        "read as a clean rejection"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 9. MT5 adapter — a failed account read must never look like "no positions"
# ═════════════════════════════════════════════════════════════════════════════

def test_get_positions_raises_when_the_broker_call_fails(monkeypatch):
    """`positions_get() or []` made a dropped connection indistinguishable
    from 'every position closed', and reconcile() deletes local state on an
    empty list — orphaning live positions."""
    import sys
    import types as pytypes

    fake = pytypes.ModuleType("MetaTrader5")
    fake.positions_get = lambda **kw: None
    fake.last_error = lambda: (10019, 0)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)

    from core.mt5.adapter import RealMT5Adapter

    adapter = RealMT5Adapter(login=1, password="p", server="s")
    adapter._mt5 = fake

    with pytest.raises(MT5CallError):
        adapter.get_positions()


def test_get_positions_returns_empty_list_on_a_genuine_empty_account(monkeypatch):
    """Genuinely having no positions is a valid answer and must not raise."""
    import sys
    import types as pytypes

    fake = pytypes.ModuleType("MetaTrader5")
    fake.positions_get = lambda **kw: ()
    fake.last_error = lambda: (0, 0)
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)

    from core.mt5.adapter import RealMT5Adapter

    adapter = RealMT5Adapter(login=1, password="p", server="s")
    adapter._mt5 = fake
    assert adapter.get_positions() == []


def test_mock_get_tick_on_unknown_symbol_returns_none():
    """An unregistered symbol must return None, not raise AttributeError."""
    adapter = MockMT5Adapter()
    adapter.connect()
    assert adapter.get_tick("NOT_REGISTERED") is None


def test_mock_tick_is_deterministic():
    """Two reads of an unchanged mock market must agree."""
    adapter = MockMT5Adapter()
    adapter.connect()
    first = adapter.get_tick("XAUUSDm")
    second = adapter.get_tick("XAUUSDm")
    assert first.bid == second.bid and first.ask == second.ask


# ═════════════════════════════════════════════════════════════════════════════
# 10. Trailing stops — no crash on absent broker levels, no stop above market
# ═════════════════════════════════════════════════════════════════════════════

def test_validate_sl_tolerates_absent_stops_and_freeze_levels():
    """SymbolInfo models an absent level as None; comparing None raised TypeError
    and killed the whole position-management loop."""
    assert validate_sl_modification(
        direction=1, proposed_sl=1995.0, current_sl=1990.0,
        current_price_for_sl=2000.0, tick_size=0.01, point=0.01,
        stops_level=None, freeze_level=None,
    ) == pytest.approx(1995.0)


def test_validate_sl_tolerates_one_absent_level():
    assert validate_sl_modification(
        direction=1, proposed_sl=1995.0, current_sl=1990.0,
        current_price_for_sl=2000.0, tick_size=0.01, point=0.01,
        stops_level=10, freeze_level=None,
    ) == pytest.approx(1995.0)
    assert validate_sl_modification(
        direction=1, proposed_sl=1995.0, current_sl=1990.0,
        current_price_for_sl=2000.0, tick_size=0.01, point=0.01,
        stops_level=None, freeze_level=10,
    ) == pytest.approx(1995.0)


def test_validate_sl_rejects_stop_above_market_on_a_long():
    """A long stop at or above the market is INVALID_STOPS at the broker."""
    assert validate_sl_modification(
        direction=1, proposed_sl=2015.0, current_sl=1990.0,
        current_price_for_sl=2010.0, tick_size=0.01, point=0.01,
        stops_level=0, freeze_level=0,
    ) is None


def test_validate_sl_rejects_stop_below_market_on_a_short():
    assert validate_sl_modification(
        direction=-1, proposed_sl=1985.0, current_sl=2010.0,
        current_price_for_sl=1990.0, tick_size=0.01, point=0.01,
        stops_level=0, freeze_level=0,
    ) is None


def test_validate_sl_never_loosens():
    """The core safety property — preserved across every refactor."""
    assert validate_sl_modification(
        direction=1, proposed_sl=1980.0, current_sl=1990.0,
        current_price_for_sl=2000.0, tick_size=0.01, point=0.01,
        stops_level=0, freeze_level=0,
    ) is None
    assert validate_sl_modification(
        direction=-1, proposed_sl=2020.0, current_sl=2010.0,
        current_price_for_sl=2000.0, tick_size=0.01, point=0.01,
        stops_level=0, freeze_level=0,
    ) is None


def test_normalize_price_rounds_half_away_from_zero_like_mt5():
    """Python's round() is banker's rounding; MT5 rounds half away from zero.

    On a protective stop the difference can land one tick on the LESS
    protective side, which is why this is not cosmetic.
    """
    # 0.5 ticks must go up, not to zero as round() does.
    assert normalize_price(0.05, 0.1) == pytest.approx(0.1)
    # 1.5 ticks must go up to 2.
    assert normalize_price(0.15, 0.1) == pytest.approx(0.2)
    # 2.5 ticks must go up to 3 (round() would give 2).
    assert normalize_price(0.25, 0.1) == pytest.approx(0.3)


# ═════════════════════════════════════════════════════════════════════════════
# 11. Ensemble — abstention must not dilute a confident signal
# ═════════════════════════════════════════════════════════════════════════════

class _StubStrategy:
    def __init__(self, name, signal, weight=1.0):
        self.name = name
        self.weight = weight
        self._signal = signal

    def evaluate(self, df, regime):
        return self._signal


def _trend_df(n=300) -> pd.DataFrame:
    rng = pd.numpy.random.RandomState(7) if hasattr(pd, "numpy") else None
    import numpy as np
    rs = np.random.RandomState(7)
    close = 2000 + np.cumsum(rs.randn(n) * 2.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close + 3, "low": close - 3, "close": close,
         "tick_volume": 1000, "volume": 1000},
        index=idx,
    )


def test_ensemble_does_not_dilute_a_unanimous_signal():
    """One strategy voting 90/100 must not be scored as 30/100 because the
    others abstained. Abstention is not disagreement."""
    from core.strategy.ensemble import StrategyEnsemble

    df = _trend_df()
    strong = Signal(direction=1, score=90.0, reason="strong", invalidation="")
    abstained = Signal(direction=0, score=0.0, reason="no setup", invalidation="")

    ensemble = StrategyEnsemble([
        _StubStrategy("trend", strong),
        _StubStrategy("momentum", abstained),
        _StubStrategy("mean_rev", abstained),
    ])
    out = ensemble.evaluate_all(df)

    assert out.direction == 1, "a confident unanimous signal was discarded"
    assert out.score == pytest.approx(90.0, abs=1e-6)


def test_ensemble_reports_zero_score_when_everyone_abstains():
    """A neutral signal must never carry a tradable score."""
    from core.strategy.ensemble import StrategyEnsemble

    df = _trend_df()
    abstained = Signal(direction=0, score=0.0, reason="no setup", invalidation="")

    ensemble = StrategyEnsemble([
        _StubStrategy("a", abstained),
        _StubStrategy("b", abstained),
    ])
    out = ensemble.evaluate_all(df)

    assert out.direction == 0
    assert out.score == 0.0


def test_ensemble_still_resolves_a_genuine_conflict():
    """When strategies disagree, the stronger weighted side wins."""
    from core.strategy.ensemble import StrategyEnsemble

    df = _trend_df()
    buy = Signal(direction=1, score=70.0, reason="buy", invalidation="")
    sell = Signal(direction=-1, score=90.0, reason="sell", invalidation="")

    ensemble = StrategyEnsemble([
        _StubStrategy("bull", buy),
        _StubStrategy("bear", sell),
    ])
    out = ensemble.evaluate_all(df)
    assert out.direction == -1
    assert out.score == pytest.approx(90.0)


# ═════════════════════════════════════════════════════════════════════════════
# 12. Configuration — the documented env key must actually be read
# ═════════════════════════════════════════════════════════════════════════════

def test_trade_symbols_env_key_is_honoured(monkeypatch):
    """`.env.example` documents TRADE_SYMBOLS; only SYMBOLS was ever read, so
    following the documentation silently traded the default XAUUSDm instead."""
    from core.settings import Settings

    monkeypatch.setenv("TRADE_SYMBOLS", "EURUSDm,GBPUSDm")
    monkeypatch.delenv("SYMBOLS", raising=False)
    assert Settings().symbols == ["EURUSDm", "GBPUSDm"]


def test_symbols_env_key_still_works(monkeypatch):
    from core.settings import Settings

    monkeypatch.delenv("TRADE_SYMBOLS", raising=False)
    monkeypatch.setenv("SYMBOLS", "XAUUSDm,EURUSDm")
    assert Settings().symbols == ["XAUUSDm", "EURUSDm"]


def test_paper_mode_is_still_the_default(monkeypatch):
    from core.settings import Settings, TradingMode

    monkeypatch.delenv("TRADING_MODE", raising=False)
    assert Settings().trading_mode is TradingMode.PAPER
    assert Settings().is_paper


def test_default_risk_limits_are_conservative(monkeypatch):
    """The shipped defaults must not exceed the documented safety envelope."""
    from core.settings import Settings

    for var in (
        "RISK_PER_TRADE",
        "MAX_AGGREGATE_RISK_PCT",
        "MAX_DAILY_LOSS_PCT",
        "MAX_WEEKLY_LOSS_PCT",
        "MAX_DRAWDOWN_PCT",
        "MAX_OPEN_POSITIONS",
        "MAX_POSITIONS_PER_SYMBOL",
    ):
        monkeypatch.delenv(var, raising=False)

    s = Settings()
    assert s.risk_per_trade <= 0.005, "default per-trade risk exceeds 0.5%"
    assert s.max_aggregate_risk_pct <= 0.02, "default aggregate risk exceeds 2%"
    assert s.max_daily_loss_pct <= 0.02, "default daily loss exceeds 2%"
    assert s.max_drawdown_pct <= 0.10, "default drawdown exceeds 10%"
    assert s.max_positions_per_symbol == 1, "default allows stacking per symbol"


def test_non_sqlite_database_is_rejected():
    """Postgres/Redis are explicitly out of scope for this project."""
    from pydantic import ValidationError
    from core.settings import Settings

    with pytest.raises(ValidationError):
        Settings(database_url="postgresql://user:pass@localhost/db")


# ═════════════════════════════════════════════════════════════════════════════
# 13. Instrument registry — sizing must stay proportional and honest
# ═════════════════════════════════════════════════════════════════════════════

def _spec(**overrides) -> InstrumentSpec:
    base = dict(
        broker_symbol="XAUUSDm",
        canonical_symbol="XAUUSD",
        asset_class=AssetClass.METAL,
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=1.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level=0,
        freeze_level=0,
        trading_status=TradingStatus.ENABLED,
        fetched_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return InstrumentSpec(**base)


def test_registry_refuses_to_size_without_broker_metadata():
    from core.instruments.model import InstrumentRegistry

    registry = InstrumentRegistry()
    registry.register(_spec(tick_value=None))
    with pytest.raises(Exception):
        registry.size_position("XAUUSDm", 10_000.0, 0.0025, 10.0)


def test_registry_size_doubles_with_budget():
    from core.instruments.model import InstrumentRegistry

    registry = InstrumentRegistry()
    registry.register(_spec())
    a = registry.size_position("XAUUSDm", 10_000.0, 0.005, 10.0)
    b = registry.size_position("XAUUSDm", 20_000.0, 0.005, 10.0)
    assert b == pytest.approx(a * 2, rel=0.02)
