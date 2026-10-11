"""
core/instruments/model.py — Broker-neutral instrument model and registry.

This module is the single source of truth for *what an instrument is* and
*what the platform is allowed to assume about it*.

Design rules (enforced here, not by convention):

1. **Broker-neutral.** Nothing in this module imports MetaTrader5 or any other
   broker SDK. Broker concepts are mapped onto these fields by an adapter.

2. **No silent defaults for critical contract specifications.** When a broker
   does not supply contract size, tick size, tick value or volume limits, the
   field is ``None`` and the instrument is NOT ``verified``. Position sizing
   and order validation must refuse to trade an unverified instrument rather
   than substitute a plausible-looking constant. Inventing a tick value turns
   a risk limit into a guess.

3. **Metadata freshness is explicit.** Every record carries ``fetched_at`` so a
   caller can decide whether it is too old to trade on.

4. **Unknown != zero.** Missing metadata is ``None``; a genuine broker value of
   zero is ``0.0``. Code that treats these interchangeably is a bug.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Asset classification
# ─────────────────────────────────────────────────────────────────────────────

class AssetClass(str, Enum):
    """Broker-neutral asset classification."""

    FOREX = "forex"
    METAL = "metal"
    ENERGY = "energy"
    AGRICULTURAL = "agricultural"
    INDEX = "index"
    EQUITY = "equity"
    CFD = "cfd"
    CRYPTO = "crypto"
    FUTURE = "future"
    BOND = "bond"
    UNKNOWN = "unknown"


class InstrumentType(str, Enum):
    SPOT = "spot"
    CFD = "cfd"
    FUTURE = "future"
    OPTION = "option"
    UNKNOWN = "unknown"


class OrderTypeSupport(str, Enum):
    """Order types an adapter has verified the instrument accepts."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    FOK = "fill_or_kill"
    IOC = "immediate_or_cancel"


class TradingStatus(str, Enum):
    """Broker-reported tradability state."""

    ENABLED = "enabled"           # fully tradable
    CLOSED = "closed"             # market closed right now (session/holiday)
    DISABLED = "disabled"         # broker-disabled (e.g. pending news)
    UNAVAILABLE = "unavailable"   # not offered on this account
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Instrument specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InstrumentSpec:
    """
    Broker-neutral contract specification for one tradable instrument.

    Optional fields are ``None`` when the broker did not supply them. They are
    NOT defaulted — see the module docstring.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    broker_symbol: str          # symbol exactly as the broker knows it
    canonical_symbol: str       # normalised platform-wide identity
    description: str = ""
    asset_class: AssetClass = AssetClass.UNKNOWN
    instrument_type: InstrumentType = InstrumentType.UNKNOWN

    # ── Currency ─────────────────────────────────────────────────────────────
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    profit_currency: Optional[str] = None
    margin_currency: Optional[str] = None

    # ── Price precision ──────────────────────────────────────────────────────
    digits: Optional[int] = None
    point: Optional[float] = None            # smallest price increment

    # ── Contract economics ───────────────────────────────────────────────────
    contract_size: Optional[float] = None    # units per 1.0 lot
    tick_size: Optional[float] = None        # minimum price movement
    tick_value: Optional[float] = None       # account-currency P/L per tick, per lot
    point_value: Optional[float] = None      # account-currency P/L per point, per lot

    # ── Volume constraints ───────────────────────────────────────────────────
    volume_min: Optional[float] = None
    volume_max: Optional[float] = None
    volume_step: Optional[float] = None
    volume_limit: Optional[float] = None     # max aggregate volume per direction

    # ── Execution constraints ────────────────────────────────────────────────
    stops_level: Optional[int] = None        # min distance SL/TP, in points
    freeze_level: Optional[int] = None       # distance from market within which
                                             # modification/closing is frozen
    spread: Optional[float] = None
    supported_order_types: tuple[OrderTypeSupport, ...] = ()

    # ── State ────────────────────────────────────────────────────────────────
    trading_status: TradingStatus = TradingStatus.UNKNOWN
    session_deals: Optional[int] = None
    margin_so_percent: Optional[float] = None

    # ── Provenance ───────────────────────────────────────────────────────────
    fetched_at: Optional[datetime] = None    # when the broker gave us this

    # ── Critical-field policy ────────────────────────────────────────────────
    # Fields without which a trade cannot be sized or validated safely.
    CRITICAL_FIELDS: tuple[str, ...] = field(
        default=(
            "digits",
            "point",
            "contract_size",
            "tick_size",
            "tick_value",
            "volume_min",
            "volume_max",
            "volume_step",
            "stops_level",
        ),
        init=False,
        repr=False,
        compare=False,
    )

    # ── Derived ──────────────────────────────────────────────────────────────

    def missing_critical_fields(self) -> list[str]:
        """Names of critical fields the broker did not supply."""
        return [f for f in self.CRITICAL_FIELDS if getattr(self, f, None) is None]

    @property
    def is_verified(self) -> bool:
        """
        True when the spec is complete enough to size and validate a trade.

        A verified spec still says nothing about whether trading is permitted
        right now — check ``trading_status`` separately.
        """
        return not self.missing_critical_fields()

    def is_stale(self, max_age: timedelta) -> bool:
        """
        True when metadata is older than ``max_age``.

        A spec with no ``fetched_at`` is treated as stale: we cannot prove it
        is current, so we must not assume it is.
        """
        if self.fetched_at is None:
            return True
        if self.fetched_at.tzinfo is None:
            self_fetched = self.fetched_at.replace(tzinfo=timezone.utc)
        else:
            self_fetched = self.fetched_at
        return datetime.now(timezone.utc) - self_fetched > max_age

    # ── Validation helpers ───────────────────────────────────────────────────

    def volume_is_valid(self, volume: float) -> bool:
        """Whether ``volume`` sits inside broker limits AND on a valid step."""
        if volume is None or volume <= 0:
            return False
        if not self.is_verified:
            return False
        if volume < (self.volume_min or 0.0) - 1e-9:
            return False
        if volume > (self.volume_max or math.inf) + 1e-9:
            return False
        step = self.volume_step or 0.0
        if step > 0:
            # Compare in step units to avoid float modulo error.
            steps = round(volume / step)
            if abs(volume - steps * step) > 1e-6:
                return False
        return True

    def min_stop_distance(self) -> Optional[float]:
        """Minimum SL/TP distance expressed in price terms."""
        if self.stops_level is None or self.point is None:
            return None
        return self.stops_level * self.point

    def stop_distance_is_valid(self, distance: float) -> bool:
        """
        Whether a stop sits far enough from the market.

        Returns False when the constraint is unknown, so callers fail safe.
        """
        required = self.min_stop_distance()
        if required is None:
            return False
        return distance >= required - 1e-9

    def normalise_price(self, price: float) -> Optional[float]:
        """Round a price to the instrument's tick grid. None if not verified."""
        if self.tick_size is None or self.tick_size <= 0:
            return None
        return round(price / self.tick_size) * self.tick_size

    def risk_per_lot(self, stop_distance: float) -> Optional[float]:
        """
        Account-currency loss for 1.0 lot at ``stop_distance``.

        Returns None when the spec cannot support the calculation — callers
        must treat None as "cannot size", never as zero.
        """
        if self.tick_size is None or self.tick_value is None:
            return None
        if self.tick_size <= 0:
            return None
        if stop_distance <= 0:
            return None
        ticks = stop_distance / self.tick_size
        return ticks * self.tick_value


# ─────────────────────────────────────────────────────────────────────────────
# Canonical symbol normalisation
# ─────────────────────────────────────────────────────────────────────────────

# Broker suffixes that qualify the same underlying instrument.
_BROKER_SUFFIXES = (
    "m", "micro", "mini", "nano", "cent", "c", "ecn", "raw", "pro",
    "idx", "index", "fut", "future", "cash", "spot", "swap", "stp", "ecn",
)

# Canonical base -> asset class. Deliberately conservative: anything not
# recognised stays UNKNOWN rather than being guessed into a class.
_KNOWN_BASES: dict[str, AssetClass] = {
    "XAU": AssetClass.METAL, "XAG": AssetClass.METAL, "XPT": AssetClass.METAL,
    "XPD": AssetClass.METAL,
    "BTC": AssetClass.CRYPTO, "ETH": AssetClass.CRYPTO, "LTC": AssetClass.CRYPTO,
    "WTI": AssetClass.ENERGY, "BRENT": AssetClass.ENERGY, "NATGAS": AssetClass.ENERGY,
    "COPPER": AssetClass.METAL,
    # Equity indices, commonly brokered as CFDs.
    "US30": AssetClass.INDEX, "US500": AssetClass.INDEX, "NAS100": AssetClass.INDEX,
    "SPX": AssetClass.INDEX, "UK100": AssetClass.INDEX, "DE40": AssetClass.INDEX,
    "FRA40": AssetClass.INDEX, "JP225": AssetClass.INDEX, "AUS200": AssetClass.INDEX,
}

# Three-letter majors -> fiat. Forex pairs are detected structurally.
_FIAT = {
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK",
    "DKK", "PLN", "CZK", "HUF", "TRY", "ZAR", "MXN", "SGD", "HKD", "CNY",
    "CNH", "INR", "KRW", "RUB", "BRL",
}


def canonical_symbol(broker_symbol: str) -> str:
    """
    Normalise a broker symbol to a platform-wide identity.

    Strips broker decoration (``.m``, ``-ECN``, ``_f``, ``.cash``) and
    uppercases. Purely lexical — it never asserts an asset class.
    """
    if not broker_symbol:
        return ""
    s = broker_symbol.strip().upper()

    # Split symbol from any provider decoration: XAUUSDm, XAUUSD.m, XAUUSD-pro
    s = re.sub(r"[._-](PRO|ECN|RAW|STP|MICRO|MINI|NANO|CENT|INDEX|FUT)\b", "", s)
    s = re.sub(r"[._](M|CASH|SPOT|FUT|INDEX|IDX|SWAP)$", "", s)
    s = s.strip("-_.")

    # Trailing single letters are broker lot-size suffixes: XAUUSDm (micro),
    # EURUSDc (cent). Only strip when what remains is still a plausible symbol,
    # so a genuine suffix-less symbol like "BTC" is left untouched.
    while len(s) > 3 and s[-1] in {"M", "C"} and _looks_like_base(s[:-1]):
        s = s[:-1]

    return s


def _looks_like_base(candidate: str) -> bool:
    """Whether a candidate stem reads as a currency pair or known base."""
    if len(candidate) == 6 and candidate[:3] in _FIAT and candidate[3:] in _FIAT:
        return True
    return candidate in _KNOWN_BASES or candidate[:3] in _KNOWN_BASES


def infer_asset_class(canon: str) -> tuple[AssetClass, InstrumentType]:
    """
    Best-effort asset classification from a canonical symbol.

    This is a *hint for display and routing*, never a substitute for broker
    metadata. When the result is UNKNOWN, downstream code must obtain the real
    specification from the broker before trading.
    """
    if not canon:
        return AssetClass.UNKNOWN, InstrumentType.UNKNOWN

    # Six-letter currency pair -> spot forex.
    if len(canon) == 6 and canon[:3] in _FIAT and canon[3:] in _FIAT:
        return AssetClass.FOREX, InstrumentType.SPOT

    for length in (6, 5, 4, 3):
        if len(canon) >= length and canon[:length] in _KNOWN_BASES:
            return _KNOWN_BASES[canon[:length]], InstrumentType.CFD

    if canon in _FIAT:
        return AssetClass.UNKNOWN, InstrumentType.SPOT

    return AssetClass.UNKNOWN, InstrumentType.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResolutionError(Exception):
    """Raised when an instrument cannot be safely resolved for trading."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class InstrumentRegistry:
    """
    Broker-neutral lookup over a set of instrument specifications.

    Holds metadata only — no broker calls. A broker adapter populates it via
    :meth:`register`, and application code resolves symbols through it.

    The registry's job is to make it *hard* to trade something unverified.
    """

    def __init__(self, specs: Optional[Iterable[InstrumentSpec]] = None) -> None:
        self._by_broker: dict[str, InstrumentSpec] = {}
        self._by_canonical: dict[str, InstrumentSpec] = {}
        for spec in specs or ():
            self.register(spec)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def register(self, spec: InstrumentSpec) -> None:
        self._by_broker[spec.broker_symbol] = spec
        if spec.canonical_symbol:
            self._by_canonical[spec.canonical_symbol] = spec

    def update(self, spec: InstrumentSpec) -> None:
        """Alias for :meth:`register`, kept for readability at call sites."""
        self.register(spec)

    # ── Lookup (non-raising) ─────────────────────────────────────────────────

    def get(self, symbol: str) -> Optional[InstrumentSpec]:
        """Resolve by broker symbol first, then canonical. None if unknown."""
        if symbol in self._by_broker:
            return self._by_broker[symbol]
        canon = canonical_symbol(symbol)
        return self._by_canonical.get(canon)

    def __contains__(self, symbol: str) -> bool:
        return self.get(symbol) is not None

    def __len__(self) -> int:
        return len(self._by_broker)

    def all_specs(self) -> list[InstrumentSpec]:
        return list(self._by_broker.values())

    def by_asset_class(self, asset_class: AssetClass) -> list[InstrumentSpec]:
        return [s for s in self._by_broker.values() if s.asset_class == asset_class]

    def verified_specs(self) -> list[InstrumentSpec]:
        return [s for s in self._by_broker.values() if s.is_verified]

    def capability_matrix(self) -> dict[str, dict[str, object]]:
        """
        Explicit statement of what is known per instrument.

        Used by operators and by the API so that supported assets are
        reported honestly rather than implied.
        """
        matrix: dict[str, dict[str, object]] = {}
        for spec in self._by_broker.values():
            matrix[spec.broker_symbol] = {
                "canonical_symbol": spec.canonical_symbol,
                "asset_class": spec.asset_class.value,
                "instrument_type": spec.instrument_type.value,
                "verified": spec.is_verified,
                "missing_critical_fields": spec.missing_critical_fields(),
                "trading_status": spec.trading_status.value,
                "supported_order_types": [o.value for o in spec.supported_order_types],
                "fetched_at": spec.fetched_at.isoformat() if spec.fetched_at else None,
            }
        return matrix

    # ── Lookup (raising) ─────────────────────────────────────────────────────

    def require(
        self,
        symbol: str,
        max_age: Optional[timedelta] = timedelta(hours=1),
    ) -> InstrumentSpec:
        """
        Resolve an instrument for trading, or raise :class:`ResolutionError`.

        Fails safe in three distinct ways, each with a distinct message so the
        caller can log precisely why a trade was refused:

        * unknown symbol,
        * known but broker metadata incomplete (never substitute defaults),
        * metadata present but stale.
        """
        spec = self.get(symbol)
        if spec is None:
            raise ResolutionError(f"Unknown instrument: {symbol!r}")

        missing = spec.missing_critical_fields()
        if missing:
            raise ResolutionError(
                f"Instrument {symbol!r} is missing critical broker metadata: "
                f"{', '.join(missing)}. Refusing to trade rather than assume values."
            )

        if max_age is not None and spec.is_stale(max_age):
            age = "unknown age" if spec.fetched_at is None else f"fetched_at={spec.fetched_at}"
            raise ResolutionError(
                f"Instrument {symbol!r} metadata is stale ({age}). "
                f"Refusing to trade on unverified specs."
            )

        if spec.trading_status not in (TradingStatus.ENABLED, TradingStatus.UNKNOWN):
            raise ResolutionError(
                f"Instrument {symbol!r} is not tradable: status={spec.trading_status.value}"
            )

        return spec

    def size_position(
        self,
        symbol: str,
        account_equity: float,
        risk_fraction: float,
        stop_distance: float,
        max_age: Optional[timedelta] = timedelta(hours=1),
    ) -> float:
        """
        Compute a risk-bounded lot size for ``symbol``.

        Raises :class:`ResolutionError` rather than returning a fallback. The
        caller decides how to handle refusal; this function never guesses.
        """
        spec = self.require(symbol, max_age=max_age)

        if account_equity <= 0:
            raise ResolutionError(f"Cannot size {symbol!r}: account equity is {account_equity}")
        if not 0 < risk_fraction <= 0.1:
            raise ResolutionError(
                f"Cannot size {symbol!r}: risk_fraction={risk_fraction} outside (0, 0.1]"
            )
        if stop_distance <= 0:
            raise ResolutionError(
                f"Cannot size {symbol!r}: stop_distance must be positive, got {stop_distance}"
            )

        risk_per_lot = spec.risk_per_lot(stop_distance)
        if risk_per_lot is None or risk_per_lot <= 0:
            raise ResolutionError(
                f"Cannot size {symbol!r}: tick value unavailable or non-positive"
            )

        if not spec.stop_distance_is_valid(stop_distance):
            required = spec.min_stop_distance()
            raise ResolutionError(
                f"Cannot size {symbol!r}: stop distance {stop_distance} below "
                f"broker minimum {required}"
            )

        risk_budget = account_equity * risk_fraction
        raw_lots = risk_budget / risk_per_lot

        step = spec.volume_step or 0.01
        lots = (int(raw_lots / step)) * step
        lots = max(spec.volume_min or 0.01, min(lots, spec.volume_max or 0.0))

        # Rounding down can still leave us above the broker minimum; in that
        # case we cannot respect the risk budget, so refuse instead of
        # silently exceeding it.
        if lots <= (spec.volume_min or 0.0):
            raise ResolutionError(
                f"Cannot size {symbol!r}: risk budget {risk_budget:.2f} is below the "
                f"minimum lot {spec.volume_min} (needs "
                f"{(spec.volume_min or 0) * risk_per_lot:.2f} at this stop distance)."
            )

        return round(lots, 8)