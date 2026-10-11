"""Broker-neutral instrument model and registry.

See :mod:`core.instruments.model` for the full contract. This package is the
multi-asset foundation: it lets the platform reason about instruments without
importing any broker SDK.
"""

from core.instruments.model import (
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

__all__ = [
    "AssetClass",
    "InstrumentRegistry",
    "InstrumentSpec",
    "InstrumentType",
    "OrderTypeSupport",
    "ResolutionError",
    "TradingStatus",
    "build_registry",
    "canonical_symbol",
    "infer_asset_class",
    "to_instrument_spec",
]