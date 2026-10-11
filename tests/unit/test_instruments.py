"""
tests/unit/test_instruments.py — Broker-neutral instrument model and registry.

These tests encode the two guarantees the platform depends on:

1. **Position sizing is genuinely risk-proportional.** Before the instrument
   registry existed, the mock adapter returned gold-shaped constants for every
   symbol (contract_size=100000, tick_size=0.00001, volume_min=0.01), which
   floored every computed size to 0.01 and made sizing inert. These tests fail
   if that regresses.

2. **Missing broker metadata is never replaced by a default.** An instrument
   lacking critical fields raises rather than being traded on invented values.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.instruments import (
    AssetClass,
    InstrumentRegistry,
    InstrumentSpec,
    InstrumentType,
    OrderTypeSupport,
    ResolutionError,
    TradingStatus,
    canonical_symbol,
    infer_asset_class,
)
from core.instruments.adapter import build_registry, to_instrument_spec
from core.mt5.adapter import MockMT5Adapter


def _spec(**overrides) -> InstrumentSpec:
    """A complete, fresh, verified instrument. Override to make it defective."""
    base = dict(
        broker_symbol="TEST",
        canonical_symbol="TEST",
        asset_class=AssetClass.FOREX,
        instrument_type=InstrumentType.SPOT,
        digits=5,
        point=0.00001,
        contract_size=100_000.0,
        tick_size=0.00001,
        tick_value=1.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level=0,
        supported_order_types=(OrderTypeSupport.MARKET, OrderTypeSupport.LIMIT),
        trading_status=TradingStatus.ENABLED,
        fetched_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return InstrumentSpec(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical symbol normalisation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "broker_symbol,expected",
    [
        ("XAUUSDm", "XAUUSD"),      # micro suffix
        ("XAUUSD.m", "XAUUSD"),      # dot decoration
        ("GBPUSD.pro", "GBPUSD"),    # account-type suffix
        ("XAUUSD-ECN", "XAUUSD"),    # hyphenated ECN
        ("EURUSD", "EURUSD"),        # already clean
        ("BTCUSD", "BTCUSD"),        # crypto must NOT lose a real trailing char
        ("US30.cash", "US30"),       # cash index
    ],
)
def test_canonical_symbol_strips_broker_decoration(broker_symbol, expected):
    assert canonical_symbol(broker_symbol) == expected


@pytest.mark.parametrize(
    "symbol,asset_class",
    [
        ("XAUUSD", AssetClass.METAL),
        ("EURUSD", AssetClass.FOREX),
        ("BTCUSD", AssetClass.CRYPTO),
        ("US30", AssetClass.INDEX),
        ("WTI", AssetClass.ENERGY),
    ],
)
def test_infer_asset_class_for_known_instruments(symbol, asset_class):
    got, _ = infer_asset_class(symbol)
    assert got == asset_class


def test_unknown_symbol_stays_unknown():
    """Never guess an asset class we have no evidence for."""
    got, _ = infer_asset_class("ZZZZQQQ")
    assert got == AssetClass.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Metadata completeness — the "no invented defaults" guarantee
# ─────────────────────────────────────────────────────────────────────────────

def test_complete_spec_is_verified():
    assert _spec().is_verified is True
    assert _spec().missing_critical_fields() == []


@pytest.mark.parametrize(
    "field",
    ["digits", "point", "contract_size", "tick_size", "tick_value",
     "volume_min", "volume_max", "volume_step", "stops_level"],
)
def test_missing_critical_field_marks_spec_unverified(field):
    spec = _spec(**{field: None})
    assert spec.is_verified is False
    assert field in spec.missing_critical_fields()


def test_zero_is_not_treated_as_missing():
    """
    A broker value of 0 is real data, not absence.
    stops_level=0 legitimately means 'no minimum stop distance'.
    """
    spec = _spec(stops_level=0)
    assert spec.is_verified is True
    assert spec.min_stop_distance() == 0.0


def test_spec_without_timestamp_is_stale():
    spec = _spec(fetched_at=None)
    assert spec.is_stale(timedelta(hours=1)) is True


def test_fresh_spec_is_not_stale():
    spec = _spec(fetched_at=datetime.now(timezone.utc))
    assert spec.is_stale(timedelta(hours=1)) is False


def test_old_spec_is_stale():
    spec = _spec(fetched_at=datetime.now(timezone.utc) - timedelta(days=2))
    assert spec.is_stale(timedelta(hours=1)) is True


# ─────────────────────────────────────────────────────────────────────────────
# Risk maths
# ─────────────────────────────────────────────────────────────────────────────

def test_risk_per_lot_uses_real_spec():
    # 20 pips on a 5-digit EURUSD: 20 * 0.00001 * 1.0 = $20 per lot
    spec = _spec()
    assert spec.risk_per_lot(0.00020) == pytest.approx(20.0)


def test_risk_per_lot_is_none_without_tick_value():
    """Unknown tick value must yield None, never a silent 0 or 1."""
    assert _spec(tick_value=None).risk_per_lot(0.001) is None


def test_risk_per_lot_rejects_non_positive_distance():
    assert _spec().risk_per_lot(0) is None
    assert _spec().risk_per_lot(-1) is None


@pytest.mark.parametrize("volume,valid", [(0.01, True), (0.5, True), (100.0, True),
                                         (0.005, False), (150.0, False), (0, False)])
def test_volume_validation(volume, valid):
    assert _spec().volume_is_valid(volume) is valid


def test_volume_step_is_enforced():
    """0.015 is between steps and must be rejected."""
    assert _spec().volume_is_valid(0.015) is False
    assert _spec().volume_is_valid(0.02) is True


# ─────────────────────────────────────────────────────────────────────────────
# Registry lookup and fail-safe behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_lookup_by_broker_and_canonical_symbol():
    reg = InstrumentRegistry([_spec(broker_symbol="EURUSDm", canonical_symbol="EURUSD")])
    assert reg.get("EURUSDm") is not None      # exact broker symbol
    assert reg.get("EURUSD") is not None       # canonical form
    assert "EURUSD" in reg
    assert len(reg) == 1


def test_require_raises_for_unknown_symbol():
    reg = InstrumentRegistry([_spec()])
    with pytest.raises(ResolutionError, match="Unknown instrument"):
        reg.require("NOSUCH")


def test_require_raises_when_metadata_incomplete():
    """Must refuse rather than assume contract values (mandate rule 10)."""
    reg = InstrumentRegistry([_spec(tick_value=None)])
    with pytest.raises(ResolutionError, match="missing critical broker metadata"):
        reg.require("TEST")


def test_require_raises_when_stale():
    reg = InstrumentRegistry([_spec(fetched_at=datetime.now(timezone.utc) - timedelta(days=3))])
    with pytest.raises(ResolutionError, match="stale"):
        reg.require("TEST", max_age=timedelta(hours=1))


def test_require_raises_when_not_tradable():
    reg = InstrumentRegistry([_spec(trading_status=TradingStatus.CLOSED)])
    with pytest.raises(ResolutionError, match="not tradable"):
        reg.require("TEST")


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing — the regression this whole module exists to prevent
# ─────────────────────────────────────────────────────────────────────────────

def test_size_position_is_proportional_to_risk_budget():
    """
    The core guarantee. Doubling the risk budget must double the size.

    With the old gold-default mock every answer collapsed to volume_min=0.01,
    so this test could not have passed.
    """
    spec = _spec()
    reg = InstrumentRegistry([spec])

    small = reg.size_position("TEST", 10_000.0, 0.005, 0.00020)
    large = reg.size_position("TEST", 10_000.0, 0.010, 0.00020)

    assert small > spec.volume_min, "sizing must exceed the broker minimum"
    assert large == pytest.approx(small * 2, rel=0.01)


def test_size_position_shrinks_with_wider_stop():
    """A wider stop must mean fewer lots for the same risk."""
    reg = InstrumentRegistry([_spec()])
    tight = reg.size_position("TEST", 10_000.0, 0.01, 0.00010)
    wide = reg.size_position("TEST", 10_000.0, 0.01, 0.00100)
    assert wide < tight


def test_size_position_respects_broker_volume_limits():
    reg = InstrumentRegistry([_spec(volume_max=1.0)])
    lots = reg.size_position("TEST", 10_000.0, 0.01, 0.00010)
    assert lots <= 1.0


def test_size_position_refuses_when_budget_below_minimum():
    """
    Must refuse rather than silently exceed the risk budget.

    A 50-pip stop costs $50 per lot, so the 0.01 minimum lot already risks
    $0.50 — far beyond the $0.10 budget below.
    """
    reg = InstrumentRegistry([_spec()])
    with pytest.raises(ResolutionError, match="below the minimum lot"):
        reg.size_position("TEST", 100.0, 0.001, 0.00050)


def test_size_position_refuses_invalid_inputs():
    reg = InstrumentRegistry([_spec()])
    with pytest.raises(ResolutionError):
        reg.size_position("TEST", 0.0, 0.01, 0.0001)          # no equity
    with pytest.raises(ResolutionError):
        reg.size_position("TEST", 10_000.0, 0.0, 0.0001)      # zero risk
    with pytest.raises(ResolutionError):
        reg.size_position("TEST", 10_000.0, 0.01, 0.0)        # no stop
    with pytest.raises(ResolutionError):
        reg.size_position("TEST", 10_000.0, 0.5, 0.0001)      # absurd risk


def test_size_position_enforces_minimum_stop_distance():
    reg = InstrumentRegistry([_spec(stops_level=50, point=0.00001)])  # 0.0005 min
    with pytest.raises(ResolutionError, match="below broker minimum"):
        reg.size_position("TEST", 10_000.0, 0.01, 0.00010)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter mapping
# ─────────────────────────────────────────────────────────────────────────────

def test_to_instrument_spec_maps_without_inventing():
    adapter = MockMT5Adapter()
    adapter.connect()
    spec = to_instrument_spec(adapter.get_symbol_info("XAUUSDm"))

    assert spec.broker_symbol == "XAUUSDm"
    assert spec.canonical_symbol == "XAUUSD"
    assert spec.asset_class == AssetClass.METAL
    assert spec.contract_size == 100.0      # gold CFD, not the old 100000
    assert spec.tick_size == 0.01
    assert spec.is_verified is True
    assert spec.fetched_at is not None


def test_adapter_does_not_fill_absent_fields():
    """A SymbolInfo without a value must produce None, not a default."""
    from core.mt5.types import SymbolInfo

    partial = SymbolInfo(name="PARTIAL", trade_tick_size=None, trade_tick_value=None)
    spec = to_instrument_spec(partial)
    assert spec.tick_size is None
    assert spec.tick_value is None
    assert spec.is_verified is False


def test_build_registry_from_mock_adapter():
    adapter = MockMT5Adapter()
    adapter.connect()
    # The mock describes only registered symbols, as a real broker would.
    for symbol in ("EURUSD", "BTCUSD", "US30.cash"):
        adapter.add_symbol(symbol)

    reg = build_registry(adapter, ["XAUUSDm", "EURUSD", "BTCUSD", "US30.cash"])

    assert len(reg) == 4
    assert reg.get("XAUUSDm").contract_size == 100.0
    assert reg.get("EURUSD").digits == 5
    assert reg.get("BTCUSD").contract_size == 1.0


def test_build_registry_skips_instruments_without_metadata():
    """An unreadable instrument must be skipped, never entered with defaults."""
    adapter = MockMT5Adapter()
    adapter.connect()
    reg = build_registry(adapter, ["XAUUSDm", "NEVER_REGISTERED"])

    assert len(reg) == 1
    assert reg.get("NEVER_REGISTERED") is None


def test_capability_matrix_is_honest():
    """Unverified instruments must be visible as unverified, not hidden."""
    reg = InstrumentRegistry([_spec(broker_symbol="GOOD"), _spec(broker_symbol="BAD", tick_value=None)])
    matrix = reg.capability_matrix()

    assert matrix["GOOD"]["verified"] is True
    assert matrix["BAD"]["verified"] is False
    assert "tick_value" in matrix["BAD"]["missing_critical_fields"]
    assert matrix["GOOD"]["supported_order_types"]


def test_instrument_types_have_no_broker_dependency():
    """core/instruments must stay MT5-free so it runs on Linux/CI."""
    import core.instruments.model as m
    import core.instruments.adapter as a

    for module in (m, a):
        source = open(module.__file__, encoding="utf-8").read()
        assert "import MetaTrader5" not in source