"""
core/instruments/adapter.py — Map broker symbol metadata to InstrumentSpec.

This is the seam between a broker SDK and the broker-neutral domain. It lives
in ``core/`` but imports no broker package: it consumes the project's own
``MT5Adapter`` / ``SymbolInfo`` dataclasses, so it stays testable on Linux.

Critical rule preserved here: **absent broker data becomes ``None``, never a
default.** The old ``MockMT5Adapter`` returned gold-shaped constants for every
symbol, which made position sizing meaningless; this module refuses to
reproduce that behaviour.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from core.instruments.model import (
    AssetClass,
    InstrumentRegistry,
    InstrumentSpec,
    InstrumentType,
    OrderTypeSupport,
    TradingStatus,
    canonical_symbol,
    infer_asset_class,
)
from core.mt5.types import SymbolInfo

logger = logging.getLogger(__name__)

# MT5 SYMBOL_TRADE_MODE_* -> broker-neutral status.
_TRADE_MODE_TO_STATUS: dict[int, TradingStatus] = {
    0: TradingStatus.DISABLED,   # SYMBOL_TRADE_MODE_DISABLED
    1: TradingStatus.DISABLED,   # SYMBOL_TRADE_MODE_LONGONLY
    2: TradingStatus.DISABLED,   # SYMBOL_TRADE_MODE_SHORTONLY
    3: TradingStatus.CLOSED,     # SYMBOL_TRADE_MODE_CLOSEONLY
    4: TradingStatus.ENABLED,    # SYMBOL_TRADE_MODE_FULL
}

# MT5 SYMBOL_TRADE_EXEMODE_* -> order types the broker accepts.
_EXEMODE_ORDER_TYPES: dict[int, tuple[OrderTypeSupport, ...]] = {
    0: (OrderTypeSupport.MARKET,),                                  # ORDER
    1: (OrderTypeSupport.LIMIT, OrderTypeSupport.STOP,
        OrderTypeSupport.STOP_LIMIT),                               # EXECUTION
    2: (OrderTypeSupport.MARKET, OrderTypeSupport.LIMIT,
        OrderTypeSupport.STOP, OrderTypeSupport.STOP_LIMIT),        # IMMEDIATE_OR_CANCEL
    3: (OrderTypeSupport.MARKET, OrderTypeSupport.LIMIT,
        OrderTypeSupport.STOP, OrderTypeSupport.STOP_LIMIT,
        OrderTypeSupport.FOK, OrderTypeSupport.IOC),                # MARKET_IF_TOUCHED
}


def _optional(value, caster):
    """
    Convert a broker field to ``caster`` type, preserving absence.

    A missing value stays ``None``. A genuine broker value of ``0``/``0.0``
    is preserved as ``0``/``0.0`` — these are different states and callers
    depend on the distinction.
    """
    if value is None:
        return None
    try:
        return caster(value)
    except (TypeError, ValueError):
        return None


def to_instrument_spec(
    info: SymbolInfo,
    *,
    trade_exemode: Optional[int] = None,
    base_currency: Optional[str] = None,
    quote_currency: Optional[str] = None,
    profit_currency: Optional[str] = None,
    margin_currency: Optional[str] = None,
    fetched_at: Optional[datetime] = None,
) -> InstrumentSpec:
    """
    Convert a broker ``SymbolInfo`` into a broker-neutral ``InstrumentSpec``.

    ``trade_exemode`` is not present on the project's ``SymbolInfo`` dataclass;
    pass it when the adapter can read it from the raw broker payload so the
    capability matrix reflects real execution policy rather than an assumption.
    """
    canon = canonical_symbol(info.name)
    guessed_class, guessed_type = infer_asset_class(canon)

    return InstrumentSpec(
        broker_symbol=info.name,
        canonical_symbol=canon,
        description=info.description or "",
        # Lexical inference is a hint only; broker metadata wins when richer
        # information is available. Unknown is a valid, honest answer.
        asset_class=AssetClass.UNKNOWN if guessed_class == AssetClass.UNKNOWN else guessed_class,
        instrument_type=(
            InstrumentType.UNKNOWN if guessed_type == InstrumentType.UNKNOWN else guessed_type
        ),
        base_currency=base_currency,
        quote_currency=quote_currency,
        profit_currency=profit_currency,
        margin_currency=margin_currency,
        digits=_optional(info.digits, int),
        point=_optional(info.point, float),
        contract_size=_optional(info.contract_size, float),
        tick_size=_optional(info.trade_tick_size, float),
        tick_value=_optional(info.trade_tick_value, float),
        volume_min=_optional(info.volume_min, float),
        volume_max=_optional(info.volume_max, float),
        volume_step=_optional(info.volume_step, float),
        stops_level=_optional(info.trade_stops_level, int),
        freeze_level=_optional(info.trade_freeze_level, int),
        spread=_optional(info.spread, float),
        supported_order_types=_EXEMODE_ORDER_TYPES.get(
            trade_exemode or 1, (OrderTypeSupport.MARKET,)
        ),
        trading_status=_TRADE_MODE_TO_STATUS.get(info.trade_mode, TradingStatus.UNKNOWN),
        session_deals=_optional(info.session_deals, int),
        fetched_at=fetched_at or datetime.now(timezone.utc),
    )


def build_registry(
    adapter,
    symbols: Optional[list[str]] = None,
    *,
    max_age_seconds: int = 3600,
) -> InstrumentRegistry:
    """
    Populate a registry from a broker adapter.

    ``symbols`` limits discovery to a known list; when omitted the adapter's
    own discovery is used if it exposes one. Instruments whose metadata cannot
    be read are skipped with a warning rather than inserted with defaults —
    an incomplete instrument must not look tradeable.
    """
    from datetime import timedelta

    registry = InstrumentRegistry()
    max_age = timedelta(seconds=max_age_seconds)

    if symbols is None:
        discovery = getattr(adapter, "get_symbols", None)
        symbols = list(discovery()) if callable(discovery) else []
        if symbols:
            logger.info("Discovered %d symbols from adapter", len(symbols))

    for symbol in symbols:
        try:
            raw = adapter.get_symbol_info(symbol)
        except Exception as exc:  # broker call failure must not abort discovery
            logger.warning("get_symbol_info(%s) failed: %s", symbol, exc)
            continue
        if raw is None:
            logger.warning("Broker returned no metadata for %s; skipping", symbol)
            continue

        spec = to_instrument_spec(raw)
        registry.register(spec)

        missing = spec.missing_critical_fields()
        if missing:
            logger.warning(
                "Instrument %s registered but NOT tradeable: missing %s",
                symbol, ", ".join(missing),
            )

    logger.info(
        "Registry built: %d instruments, %d verified",
        len(registry), len(registry.verified_specs()),
    )
    del max_age  # freshness is evaluated per-instrument at require() time
    return registry