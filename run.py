"""
run.py – Self-contained AI trading bot for MetaTrader 5 (XAUUSDm).

A single-file "powerful" bot engine:
  • Multi-timeframe confluence scoring (M15 entry / H1 + H4 trend filters)
  • AI ensemble (RandomForest + GradientBoosting + ExtraTrees + HistGradientBoosting
    soft voting, ~47 engineered features, recency-weighted rolling retrain) –
    self-contained, no core/ deps
  • Dynamic risk sizing (ADX-scaled), margin guard, daily drawdown protection
  • ATR trailing stop with acceleration (tightens as profit grows)
  • Breakeven lock + multi-stage partial close (2 stages)
  • Hard per-trade stop-loss (% of equity) as a backstop under the ATR stop
  • Profit-lock / giveback close (locks gains if price retraces from peak)
  • Losing-position timeout cut (exits stale losers early, independent of SL)
  • Reversal close on strong opposite signal, H1 trend-flip close,
    position-age timeout close, daily-DD equity protection close
  • Telegram + sound notifications

Your account CONFIG below mirrors config.py but is edited per account here.
Run with:  python run.py

--------------------------------------------------------------------------
PERFORMANCE / RELIABILITY CHANGES vs the original version (logic unchanged,
CONFIG untouched):
  1. Indicators (_prepare) are only recomputed when a NEW closed M15/H1 bar
     actually arrives, instead of every 0.5s loop tick. Since _fetch() always
     drops the still-forming candle, the closed-bar data is identical between
     two bar closes — recomputing it every tick was pure wasted CPU (up to
     ~1800x redundant work per 15-minute bar at a 0.5s sleep interval).
  2. build_features() no longer recomputes RSI / MACD / Stoch / ADX when
     those columns already exist on the incoming (already-_prepare'd) frame
     — it reuses them instead of calling pandas_ta twice for the same thing.
  3. Position management (trailing stop / breakeven / partial close) now
     reacts to the LIVE tick price (bid/ask) instead of the last closed bar's
     close, so stops react in real time instead of being up to one bar late.
  4. _send_order() reuses the already-prepared M15 frame from the main loop
     instead of doing its own extra MT5 fetch + full indicator recompute.
  5. Added MT5 auto-reconnect with backoff if the terminal/session drops.
  6. Main loop iterations are now wrapped in try/except so a single
     transient error (network blip, MT5 hiccup) can't kill the whole bot.
  7. Small structure-score caching to avoid recomputing rolling(20) high/low
     twice per bar (once per direction).
--------------------------------------------------------------------------
POWER-UP CHANGES (new in this version):
  A. Hard per-trade stop-loss as a % of account equity — closes a position
     immediately if its floating loss exceeds this, as a backstop under the
     ATR-based SL (protects against gaps/slippage where the broker SL is
     honoured late or not at all).
  B. Profit-lock / giveback close — tracks each position's peak favourable
     excursion; once profit has run far enough, closes the trade if price
     gives back a configurable fraction of that peak, locking in gains
     instead of riding a full reversal back to breakeven/loss.
  C. Losing-position timeout cut — closes a position that has been
     underwater for longer than a configurable number of hours, instead of
     only ever timing out profitable positions (the old age-timeout only
     fired on profit).
  D. AI ensemble upgraded with a 4th model (HistGradientBoostingClassifier)
     in the soft-voting ensemble, plus recency-weighted training (recent
     bars matter more than old ones) via sample_weight.
  E. 7 new engineered features: ATR slope, EMA50 slope, Keltner channel
     width, OBV slope, higher-high / lower-low counts, and momentum
     acceleration (2nd derivative of returns) — all reused automatically
     because AIEnsemble consumes FEATURE_COLS dynamically.
--------------------------------------------------------------------------
"""

import logging
import math
import sys
import threading
import time
import os
from dotenv import load_dotenv
env_file = os.getenv("ENV_FILE")
if env_file and os.path.exists(env_file):
    load_dotenv(dotenv_path=env_file, override=True)
else:
    load_dotenv()
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Windows consoles default to a legacy codepage (cp1252) that can't encode
# emoji used in log/notification messages (✅ 🔒 🛡️ ...). Force UTF-8 on
# stdout/stderr so logging never crashes trying to print them.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

# Market-location layer (core/market/*). Pure pandas, no broker SDK, so it is
# safe to import here even though it lives under core/. It answers the
# questions the indicator stack cannot: where is price inside its range, which
# levels matter, which stops are resting, and what price invalidates the idea.
#
# Imported defensively. If this ever fails to import, the bot must still trade
# exactly as it did before rather than refuse to start — a live bot that dies
# at import is worse than one that trades without this refinement.
try:
    from core.market.location import analyse_location as _analyse_location
    from core.market.location import LocationBias as _LocationBias
    _LOCATION_AVAILABLE = True
except Exception as _loc_import_error:  # pragma: no cover - defensive
    _analyse_location = None
    _LocationBias = None
    _LOCATION_AVAILABLE = False
    print(
        f"WARNING: market location layer unavailable ({_loc_import_error!r}); "
        f"falling back to ATR-only stops and no location gate."
    )

try:
    from core.ai.dynamic_sltp import DynamicSLTPReasoner as _DynamicSLTPReasoner
    from core.ai.provider import get_llm_provider as _get_llm_provider
    _DYNAMIC_SLTP_AVAILABLE = True
except Exception as _sltp_import_error:  # pragma: no cover - defensive
    _DynamicSLTPReasoner = None
    _get_llm_provider = None
    _DYNAMIC_SLTP_AVAILABLE = False
    print(
        f"WARNING: dynamic SL/TP reasoning layer unavailable ({_sltp_import_error!r}); "
        f"falling back to standard stops."
    )

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg_module  # shared defaults (kept for reference)

# ---------------------------------------------------------------------------
# Trading mode resolution — single source of truth for paper/live
# ---------------------------------------------------------------------------
# Accepted values: paper (default, SAFE) | live | backtest
# Anything unrecognised falls back to paper rather than failing OPEN to live.
_TRADING_MODE = os.getenv("TRADING_MODE", "paper").strip().lower()
if _TRADING_MODE not in ("paper", "live", "backtest"):
    print(f"WARNING: unknown TRADING_MODE={_TRADING_MODE!r}; falling back to 'paper'.")
    _TRADING_MODE = "paper"

if _TRADING_MODE == "live":
    print("=" * 70)
    print("  TRADING_MODE=live — REAL ORDERS WILL BE SENT TO THE BROKER.")
    print("  Set TRADING_MODE=paper (or unset it) for simulation.")
    print("=" * 70)

# ---------------------------------------------------------------------------
# Trading Profile resolution (balanced vs high-performance money_printer)
# ---------------------------------------------------------------------------
_PROFILE = os.getenv("TRADING_PROFILE", "balanced").strip().lower()
_IS_SCALPER = _PROFILE in ("scalper", "ultra_scalper")
_IS_AGGRESSIVE = (
    _IS_SCALPER
    or _PROFILE in ("money_printer", "aggressive", "high_performance")
    or os.getenv("AGGRESSIVE_MODE", "false").strip().lower() in ("true", "1", "yes")
)

if _IS_AGGRESSIVE:
    print("=" * 70)
    if _IS_SCALPER:
        print("  TRADING_PROFILE=scalper (Ultra-Fast Microstructure Scalper)")
        print("  - Microstructure Confluence: EMA 9/21 Ribbon + Pinbars + Volume Surge")
        print("  - Fast responsive entries (Score >= 45, Dynamic Spread Guard <= 25% ATR)")
        print("  - Sharp Scalp R:R: 1.5 ATR SL / 2.0 RR TP (Target: 3.0 ATR)")
        print("  - Rapid Partial TP @ 0.8 ATR + Fast Breakeven @ 0.3 ATR")
        print("  - Micro-trailing ratchet @ 0.7 ATR + Rejection Wick Profit Securing")
    else:
        print("  TRADING_PROFILE=money_printer (High-Performance Scalper / Compounding)")
        print("  - Multi-Symbol scanning active (Gold + Major FX pairs)")
        print("  - High-frequency responsive entries (Confluence threshold: 45)")
        print("  - Compound R:R risk structure: 1.8 ATR SL / 2.0 RR TP")
        print("  - Early partial TP @ 0.9 ATR + Progressive Breakeven ratchet @ 55%")
        print("  - Dynamic AI lot scaling: +50% on high confidence setups")
    print("=" * 70)

def _profile_cfg(env_var: str, balanced_val, aggressive_val, val_type=float, scalper_val=None):
    raw = os.getenv(env_var)
    if raw is None or raw == "":
        if _IS_SCALPER and scalper_val is not None:
            return val_type(scalper_val)
        return val_type(aggressive_val if _IS_AGGRESSIVE else balanced_val)
    if _IS_AGGRESSIVE:
        try:
            if val_type(raw) == val_type(balanced_val):
                if _IS_SCALPER and scalper_val is not None:
                    return val_type(scalper_val)
                return val_type(aggressive_val)
        except (ValueError, TypeError):
            pass
    return val_type(raw)


def _env_login() -> int | None:
    """Read MT5_LOGIN from the environment. Returns None when unset/invalid.

    Returns None rather than 0 so that a misconfigured login is detectable:
    0 is not a valid MT5 account number.
    """
    raw = (os.getenv("MT5_LOGIN") or "").strip()
    if not raw or raw == "0":
        return None
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: MT5_LOGIN={raw!r} is not an integer; ignoring.")
        return None


# ---------------------------------------------------------------------------
# Per-account configuration (override anything from config.py here)
# NOTE: the original keys here are left 100% identical to the original —
# nothing was changed, removed, or reordered. New keys for the power-up
# features are appended at the end of the dict, clearly marked.
# ---------------------------------------------------------------------------

def _parse_timeframe(tf_str: Optional[str], default_val: int) -> int:
    if not tf_str:
        return default_val
    s = str(tf_str).strip().upper()
    mapping = {
        "M1": getattr(mt5, "TIMEFRAME_M1", 1) or 1,
        "M2": getattr(mt5, "TIMEFRAME_M2", 2) or 2,
        "M3": getattr(mt5, "TIMEFRAME_M3", 3) or 3,
        "M4": getattr(mt5, "TIMEFRAME_M4", 4) or 4,
        "M5": getattr(mt5, "TIMEFRAME_M5", 5) or 5,
        "M6": getattr(mt5, "TIMEFRAME_M6", 6) or 6,
        "M10": getattr(mt5, "TIMEFRAME_M10", 10) or 10,
        "M12": getattr(mt5, "TIMEFRAME_M12", 12) or 12,
        "M15": getattr(mt5, "TIMEFRAME_M15", 15) or 15,
        "M20": getattr(mt5, "TIMEFRAME_M20", 20) or 20,
        "M30": getattr(mt5, "TIMEFRAME_M30", 30) or 30,
        "H1": getattr(mt5, "TIMEFRAME_H1", 16385) or 16385,
        "H2": getattr(mt5, "TIMEFRAME_H2", 16386) or 16386,
        "H3": getattr(mt5, "TIMEFRAME_H3", 16387) or 16387,
        "H4": getattr(mt5, "TIMEFRAME_H4", 16388) or 16388,
        "H6": getattr(mt5, "TIMEFRAME_H6", 16390) or 16390,
        "H8": getattr(mt5, "TIMEFRAME_H8", 16392) or 16392,
        "H12": getattr(mt5, "TIMEFRAME_H12", 16396) or 16396,
        "D1": getattr(mt5, "TIMEFRAME_D1", 16408) or 16408,
        "W1": getattr(mt5, "TIMEFRAME_W1", 32769) or 32769,
        "MN1": getattr(mt5, "TIMEFRAME_MN1", 49153) or 49153,
    }
    return mapping.get(s, default_val)


def _default_trend_tfs(primary_tf: int) -> tuple[int, int]:
    """Derive appropriate intermediate and macro trend timeframes from primary execution timeframe."""
    m1 = getattr(mt5, "TIMEFRAME_M1", 1) or 1
    m5 = getattr(mt5, "TIMEFRAME_M5", 5) or 5
    m15 = getattr(mt5, "TIMEFRAME_M15", 15) or 15
    h1 = getattr(mt5, "TIMEFRAME_H1", 16385) or 16385
    h4 = getattr(mt5, "TIMEFRAME_H4", 16388) or 16388

    if primary_tf == m1:
        return m5, m15
    elif primary_tf == m5:
        return m15, h1
    elif primary_tf == m15:
        return h1, h4
    elif primary_tf == getattr(mt5, "TIMEFRAME_M30", 30):
        return h1, h4
    return h1, h4


_primary_tf_str = _profile_cfg("TIMEFRAME", "M15", "M1", val_type=str)
_primary_tf = _parse_timeframe(_primary_tf_str, getattr(mt5, "TIMEFRAME_M15", 15))
_def_h1_tf, _def_h4_tf = _default_trend_tfs(_primary_tf)
_h1_tf = _parse_timeframe(os.getenv("TREND_TF"), _def_h1_tf)
_h4_tf = _parse_timeframe(os.getenv("MACRO_TREND_TF"), _def_h4_tf)

# Multi-timeframe trend protection:
# Higher timeframes MUST be higher than the primary execution timeframe.
if _h1_tf <= _primary_tf:
    print(f"WARNING: TREND_TF ({_h1_tf}) <= primary timeframe ({_primary_tf}). Enforcing {_def_h1_tf} for trend analysis.")
    _h1_tf = _def_h1_tf
if _h4_tf <= _h1_tf:
    print(f"WARNING: MACRO_TREND_TF ({_h4_tf}) <= intermediate trend ({_h1_tf}). Enforcing {_def_h4_tf} for macro trend.")
    _h4_tf = _def_h4_tf

CONFIG = {
    # --- connection ---
    # Credentials come from the environment ONLY. There are deliberately no
    # fallback literals here: a default of `None` fails loudly at connect
    # time instead of silently logging into whatever account was once pasted
    # into this file. See docs/SECURITY.md for credential rotation.
    "account":  _env_login(),
    "password": os.getenv("MT5_PASSWORD") or None,
    "server":   os.getenv("MT5_SERVER") or None,
    "mt5_path": os.getenv("MT5_PATH") or "",
    
    # --- instrument ---
    "symbol":    os.getenv("TRADE_SYMBOL", "XAUUSDm"),
    "timeframe": _primary_tf,
    "h1_tf":     _h1_tf,
    "h4_tf":     _h4_tf,
    "n_candles": int(os.getenv("HISTORY_CANDLES", 600)),
    
    # --- AI Settings ---
    "ai_confidence_threshold": _profile_cfg("AI_CONFIDENCE_THRESHOLD", 0.55, 0.50),
    "ai_train_bars":  int(os.getenv("AI_TRAIN_BARS", "800")),
    "ai_retrain_bars": int(os.getenv("AI_RETRAIN_BARS", "50")),

    # --- AI model rigor ---
    # The ensemble is trained on a rolling window, but a model that is only
    # ever FIT is not a model that has been shown to work. These govern a
    # time-ordered train/validation split and the promotion gate that decides
    # whether a freshly-trained challenger is allowed to replace the live
    # model. See AIEnsemble.train().
    "ai_validation_fraction":  float(os.getenv("AI_VALIDATION_FRACTION", "0.20")),
    # Minimum out-of-sample accuracy for a model to be promoted at all.
    # 0.50 is a coin flip; anything at or below that has no tradeable edge.
    "ai_min_oos_accuracy":     float(os.getenv("AI_MIN_OOS_ACCURACY", "0.52")),
    # Ranking (AUC) gate. Accuracy alone is fooled by class imbalance: a model
    # that always answers UP scores 0.58 on a window that is 58% UP. AUC
    # measures whether the model can ORDER outcomes at all. 0.50 == random.
    "ai_min_oos_auc":          float(os.getenv("AI_MIN_OOS_AUC", "0.50")),
    # Lift over the trivial baseline. The best naive predictor is "always
    # answer the majority class", which scores exactly the base rate. A model
    # below it has negative expectancy and must not trade. Set to -1.0 to
    # disable this gate entirely.
    "ai_min_oos_edge":         float(os.getenv("AI_MIN_OOS_EDGE", "0.0")),
    # A challenger may be this much WORSE than the incumbent and still be
    # promoted (avoids churning models when two are statistically tied).
    "ai_promote_tolerance":    float(os.getenv("AI_PROMOTE_TOLERANCE", "0.01")),
    # Platt-scaling calibration of raw predict_proba output. Raw soft-voting
    # probabilities are systematically overconfident; calibrating them is what
    # makes "confidence 0.62" mean something before it is used to size a trade.
    "ai_calibrate":            os.getenv("AI_CALIBRATE", "true").lower() in ("true", "1", "yes"),
    # Expected-edge floor for AI-driven position upsizing, as a probability
    # margin over a coin flip. Size scales with edge, not with raw confidence.
    "ai_min_edge_for_size_up": float(os.getenv("AI_MIN_EDGE_FOR_SIZE_UP", "0.10")),
    "ai_size_high_conf": _profile_cfg("AI_SIZE_HIGH_CONF", 0.85, 0.70),
    "ai_size_low_conf":  _profile_cfg("AI_SIZE_LOW_CONF",  0.60, 0.60),
    
    # --- strategy scoring ---
    "min_signal_score": _profile_cfg("MIN_SIGNAL_SCORE", 50.0, 45.0),
    "adx_min": _profile_cfg("ADX_MIN", 18.0, 15.0),
    "adx_strong": _profile_cfg("ADX_STRONG", 30.0, 25.0),
    "use_session_filter": os.getenv("USE_SESSION_FILTER", "false").lower() in ("true", "1", "yes") if os.getenv("USE_SESSION_FILTER") is not None else False,

    # --- market location ("where is the market") ---
    # The indicator stack says WHAT to trade; this says WHERE. Both features
    # default ON but are independently switchable, and both are fail-safe:
    # if the location layer cannot produce an answer the bot falls back to its
    # previous ATR-based behaviour instead of stopping.
    #
    # location_gate  : veto entries whose location is indefensible (e.g. buying
    #                  a range top). Only ever *removes* trades, never adds one.
    # location_stop  : place the stop beyond the swing that invalidates the
    #                  idea instead of a blind sl_atr_mult multiple.
    # location_target: aim the take-profit at resting liquidity when one exists
    #                  in the trade's favour, else keep the fixed RR.
    "location_gate": os.getenv("LOCATION_GATE", "true").lower() in ("true", "1", "yes"),
    "location_stop": os.getenv("LOCATION_STOP", "true").lower() in ("true", "1", "yes"),
    "location_target": os.getenv("LOCATION_TARGET", "true").lower() in ("true", "1", "yes"),
    # Bounds on a structural stop, as ATR multiples. The floor keeps the stop
    # out of bar noise; the ceiling stops a distant swing producing an
    # un-sizably small position.
    "location_stop_min_atr": _profile_cfg("LOCATION_STOP_MIN_ATR", 0.75, 0.75),
    "location_stop_max_atr": _profile_cfg("LOCATION_STOP_MAX_ATR", 3.0, 2.5, scalper_val=2.0),
    # How close to entry untaken opposing liquidity must be before it is worth
    # flagging, as a fraction of the pool's distance in ATR.
    "location_min_bars": int(os.getenv("LOCATION_MIN_BARS", "60")),
    
    # trade window (UTC)
    "london_start_utc": 7,  "london_end_utc": 16,
    "ny_start_utc":     12, "ny_end_utc":     21,
    
    # --- risk & money management ---
    "risk_per_trade":   _profile_cfg("RISK_PER_TRADE", 0.01, 0.015),
    "sl_atr_mult":      _profile_cfg("SL_ATR_MULT", 2.5, 1.8, scalper_val=1.5),
    "tp_rr_ratio":      _profile_cfg("TP_RR_RATIO", 2.5, 2.0, scalper_val=2.0),
    
    # --- dynamic trade management ("never lose" once in profit) ---
    "trail_atr_mult":          _profile_cfg("TRAIL_ATR_MULT", 1.5, 1.2, scalper_val=1.0),
    "trail_activate_atr":      _profile_cfg("TRAIL_ACTIVATE_ATR", 1.0, 0.8, scalper_val=0.7),
    "trail_tighten1_atr":      _profile_cfg("TRAIL_TIGHTEN1_ATR", 2.0, 1.5),
    "trail_tighten1_mult":     _profile_cfg("TRAIL_TIGHTEN1_MULT", 1.0, 0.8),
    "trail_tighten2_atr":      _profile_cfg("TRAIL_TIGHTEN2_ATR", 3.0, 2.2),
    "trail_tighten2_mult":     _profile_cfg("TRAIL_TIGHTEN2_MULT", 0.6, 0.5),
    "dynamic_risk":            True,
    "min_risk_scale":          0.5,
    "max_risk_scale":          _profile_cfg("MAX_RISK_SCALE", 1.2, 1.5),
    "max_margin_usage":        _profile_cfg("MAX_MARGIN_USAGE", 0.30, 0.45),
    "max_open_pos":            _profile_cfg("MAX_OPEN_POS", 3, 5, val_type=int),
    "max_daily_dd":            _profile_cfg("MAX_DAILY_DD", 0.03, 0.05),
    
    # --- breakeven / progressive profit lock ---
    "enable_breakeven":        True,
    "breakeven_trigger_atr":   _profile_cfg("BREAKEVEN_TRIGGER_ATR", 0.4, 0.3, scalper_val=0.3),
    "breakeven_buffer_atr":    float(os.getenv("BREAKEVEN_BUFFER_ATR", "0.15")),
    "profit_lock_ratio":       _profile_cfg("PROFIT_LOCK_RATIO", 0.50, 0.55),
    "enable_partial_close":    True,
    "partial_close_trigger_atr": _profile_cfg("PARTIAL_CLOSE_TRIGGER_ATR", 1.2, 0.9, scalper_val=0.8),
    "profit_lock_activate_atr": _profile_cfg("PROFIT_LOCK_ACTIVATE_ATR", 1.0, 0.8, scalper_val=0.7),
    "profit_giveback_pct":     _profile_cfg("PROFIT_GIVEBACK_PCT", 0.30, 0.25),
    "max_spread_atr_ratio":    _profile_cfg("MAX_SPREAD_ATR_RATIO", 0.40, 0.25, scalper_val=0.20),
    
    # --- system ---
    "sleep_interval":   1.0,
    "max_trades_per_day": _profile_cfg("MAX_TRADES_PER_DAY", 15, 60, val_type=int),
    "magic_number":     int(os.getenv("MAGIC_NUMBER", 888999)),
    "deviation":        20,

    # --- dynamic AI reasoning & SL/TP ---
    "dynamic_sltp_enabled":    os.getenv("DYNAMIC_SLTP_ENABLED", "true").lower() in ("1", "true", "yes"),
    "llm_provider":            os.getenv("LLM_PROVIDER", "embedded"),
    "local_llm_url":           os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1"),
    "local_llm_model":         os.getenv("LOCAL_LLM_MODEL", "local-model"),
    "local_llm_timeout":       float(os.getenv("LOCAL_LLM_TIMEOUT", "3.0")),
    "scalp_micro_trailing":    os.getenv("SCALP_MICRO_TRAILING", "true").lower() in ("1", "true", "yes"),

    # --- manual trade management ---
    "ai_protect_manual_trades":  True,
    "manual_trade_sl_atr":       2.0,
    "manage_manual_trades_full": False,

    # --- safety ---
    # SAFETY: the bot MUST default to paper mode. Every mutating method
    # (_send_order, _close_position, _close_all, _modify_sltp, _modify_sl,
    # _partial_close) gates on cfg["dry_run"] and returns early when True.
    # This key was previously absent from CONFIG, so cfg.get("dry_run")
    # returned None -> falsy -> REAL ORDERS were sent by default.
    "dry_run": _TRADING_MODE != "live",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def _get_sqlite_db_path() -> str:
    """Resolve unified SQLite database path from DATABASE_URL or default to data/trading.db."""
    db_url = os.getenv("DATABASE_URL", "sqlite:///./data/trading.db")
    if db_url.startswith("sqlite:///"):
        path = db_url.replace("sqlite:///", "")
        if path.startswith("./"):
            path = path[2:]
        return path
    return os.path.join("data", "trading.db")


def _update_env_file(updates: dict[str, Any], env_path: str = ".env") -> bool:
    """Safely updates or appends KEY=VALUE lines in .env, preserving comments and formatting."""
    try:
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        updated_keys = set()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                new_lines.append(line)
                continue
            k = stripped.split("=", 1)[0].strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}\n")
                updated_keys.add(k)
            else:
                new_lines.append(line)

        # Append keys that were not already in the file
        for k, v in updates.items():
            if k not in updated_keys and v is not None:
                new_lines.append(f"{k}={v}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        return True
    except Exception as exc:
        logger.error("Failed to update env file %s: %s", env_path, exc)
        return False



# ---------------------------------------------------------------------------
# Market location (core/market/*)
# ---------------------------------------------------------------------------

def _valid_tick(value) -> Optional[float]:
    """Coerce a broker increment to a positive float, or ``None``.

    MT5 client objects are untrusted here: a stub, a mock, or a partially
    initialised terminal can return something that is not a number, and that
    value would otherwise flow into the price-snapping arithmetic. ``None``
    means "unknown increment" and simply leaves prices unsnapped.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

class Notifier:
    """Sends Telegram + sound alerts without ever risking the trading loop.

    Telegram delivery is now resilient: a single timeout/network blip no
    longer permanently disables notifications for the rest of the session.
    Each message gets a couple of quick retries, and if Telegram is briefly
    unreachable we back off for a short cooldown rather than giving up for
    good — so alerts recover automatically once connectivity returns.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._telegram_cooldown_until = 0.0   # epoch seconds; retry after this
        self._lock = threading.Lock()

    def send(self, msg: str) -> None:
        """Non-blocking send – a network hang must never kill the bot."""
        threading.Thread(target=self._send, args=(msg,), daemon=True).start()

    def _send(self, msg: str) -> None:
        try:
            self._send_telegram(msg)
            try:
                import winsound
                if self.cfg.get("enable_sound"):
                    winsound.Beep(1000, 200)
            except Exception:
                pass
        except BaseException:
            pass

    def _send_telegram(self, msg: str) -> None:
        token   = self.cfg.get("telegram_token", "")
        chat_id = self.cfg.get("telegram_chat_id", "")
        if not (token and chat_id):
            return

        with self._lock:
            if time.time() < self._telegram_cooldown_until:
                return  # still cooling down from a recent failure, skip quietly

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        attempts = 3
        for attempt in range(attempts):
            try:
                resp = requests.post(
                    url, data={"chat_id": chat_id, "text": msg}, timeout=(5, 10)
                )
                if resp.status_code == 200:
                    return
                logger.warning("Telegram HTTP %s on attempt %d/%d",
                               resp.status_code, attempt + 1, attempts)
            except Exception as exc:
                logger.warning("Telegram send failed (attempt %d/%d): %s",
                               attempt + 1, attempts, exc)
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))

        # All attempts failed: back off briefly instead of disabling forever.
        with self._lock:
            self._telegram_cooldown_until = time.time() + 60
        logger.warning(
            "Telegram unreachable after %d attempts – pausing alerts for 60s "
            "(will auto-resume, sounds still active).", attempts)


# ---------------------------------------------------------------------------
# Data + indicator preparation
# ---------------------------------------------------------------------------

def _fetch(symbol: str, tf: int, n: int) -> pd.DataFrame:
    """Fetch `n+1` bars and drop the forming candle → n closed bars."""
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n + 1)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates).iloc[:-1].copy()
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    return df[["open", "high", "low", "close", "tick_volume", "spread"]].astype(float)


def _prepare(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach all indicators needed by the scoring system and AI features."""
    df = df.copy()
    cl, hi, lo, op = df["close"], df["high"], df["low"], df["open"]

    # Scalping & Trend EMAs
    df["ema9"]   = ta.ema(cl, length=9)
    df["ema21"]  = ta.ema(cl, length=21)
    df["ema50"]  = ta.ema(cl, length=50)
    df["ema200"] = ta.ema(cl, length=200)
    df["rsi"]    = ta.rsi(cl, length=14)
    df["atr"]    = ta.atr(hi, lo, cl, length=14)

    try:
        macd = ta.macd(cl, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            h = next((c for c in macd.columns if "MACDh" in c), None)
            s = next((c for c in macd.columns if "MACDs" in c), None)
            m = next((c for c in macd.columns if c.startswith("MACD_")), None)
            df["macd_hist"]   = macd[h] if h else 0.0
            df["macd_signal"] = macd[s] if s else 0.0
            df["macd_line"]   = macd[m] if m else 0.0
            df["macd_hist_prev"] = df["macd_hist"].shift(1)
    except Exception:
        df["macd_hist"] = df["macd_signal"] = df["macd_line"] = 0.0
        df["macd_hist_prev"] = 0.0

    try:
        stoch = ta.stoch(hi, lo, cl, k=14, d=3)
        if stoch is not None and not stoch.empty:
            kc = next((c for c in stoch.columns if "STOCHk" in c), None)
            dc = next((c for c in stoch.columns if "STOCHd" in c), None)
            df["stoch_k"] = stoch[kc] if kc else 50.0
            df["stoch_d"] = stoch[dc] if dc else 50.0
    except Exception:
        df["stoch_k"] = df["stoch_d"] = 50.0

    try:
        adx_df = ta.adx(hi, lo, cl, length=14)
        if adx_df is not None and not adx_df.empty:
            ac = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
            pc = next((c for c in adx_df.columns if c.startswith("DMP_")), None)
            nc = next((c for c in adx_df.columns if c.startswith("DMN_")), None)
            df["adx"]      = adx_df[ac] if ac else 0.0
            df["di_plus"]  = adx_df[pc] if pc else 0.0
            df["di_minus"] = adx_df[nc] if nc else 0.0
    except Exception:
        df["adx"] = df["di_plus"] = df["di_minus"] = 0.0

    # Bollinger Bands
    try:
        bb_df = ta.bbands(cl, length=20, std=2.0)
        if bb_df is not None and not bb_df.empty:
            p_col = next((c for c in bb_df.columns if "BBP" in c), None)
            w_col = next((c for c in bb_df.columns if "BBB" in c), None)
            df["bb_pct_b"] = bb_df[p_col] if p_col else 0.5
            df["bb_width"] = bb_df[w_col] if w_col else 0.0
        else:
            df["bb_pct_b"] = 0.5
            df["bb_width"] = 0.0
    except Exception:
        df["bb_pct_b"] = 0.5
        df["bb_width"] = 0.0

    # Candlestick Microstructure for Scalping (Pinbars, Rejection Wicks, Engulfing)
    hl = (hi - lo).clip(lower=1e-10)
    upper_wick = hi - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - lo

    # Rejection wicks (>= 45% wick and closed opposite to rejection)
    df["bull_pinbar"] = (lower_wick / hl >= 0.45) & (cl > op)
    df["bear_pinbar"] = (upper_wick / hl >= 0.45) & (cl < op)

    # Momentum Engulfing
    df["bull_engulf"] = (cl > op) & (cl > hi.shift(1)) & (op <= cl.shift(1))
    df["bear_engulf"] = (cl < op) & (cl < lo.shift(1)) & (op >= cl.shift(1))

    df["vol_sma20"] = df["tick_volume"].rolling(20).mean()
    df["vol_ratio"] = df["tick_volume"] / (df["vol_sma20"] + 1e-10)

    # cache rolling 20-bar high/low once here so downstream scoring
    # (structure score) doesn't recompute the same rolling window twice
    # per bar (once per direction).
    df["_high20"] = df["high"].rolling(20).max()
    df["_low20"]  = df["low"].rolling(20).min()

    return df


# ---------------------------------------------------------------------------
# AI ENSEMBLE (self-contained – ported from core/ai_model.py)
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "body_ratio", "upper_wick", "lower_wick", "range_atr",
    "dist_ema20", "dist_ema50", "dist_ema100", "dist_ema200",
    "ema20_50_spread", "ema50_200_spread",
    "macd_hist", "macd_hist_delta", "macd_vs_signal",
    "rsi", "rsi_delta",
    "stoch_k", "stoch_d", "stoch_spread",
    "adx", "di_spread",
    "bb_pct_b", "bb_width",
    "atr_pct",
    "vol_ratio",
    "dist_high20_pct", "dist_low20_pct",
    "cci", "williams_r",
    "up_bars_5", "up_bars_10",
    # --- POWER-UP: new engineered features (D/E) ---
    "atr_slope", "ema50_slope", "keltner_width",
    "obv_slope", "hh_count_5", "ll_count_5", "accel",
    # --- SCALPER POWER-UP: fast microstructure features ---
    "dist_ema9", "dist_ema21", "ema9_21_spread",
    "pinbar_wick", "bb_squeeze",
    # --- PRO TRADER FEATURES ---
    # Regime: captures whether market is trending or ranging
    "adx_slope",            # ADX acceleration (trend strengthening?)
    "regime_score",         # composite trend/chop score normalised -1..+1
    "kc_squeeze",           # Keltner-Bollinger squeeze (1=compressed, 0=open)
    # Pullback quality: is price near value area (EMA21/50)?
    "dist_ema21_atr",       # distance from EMA21 in ATR units
    "dist_ema50_atr",       # distance from EMA50 in ATR units
    "at_value_area",        # 1 if price within 0.75 ATR of EMA21 or EMA50
    # Multi-bar momentum confirmation (avoids single-bar false signals)
    "rsi_trending_2",       # RSI directional consistency over last 2 bars
    "macd_hist_trending_2", # MACD histogram expanding for 2 bars
    "consecutive_dir_3",    # net directional count over last 3 bars (-3..+3)
    # Volatility regime
    "atr_vs_20d",           # current ATR / 20-bar mean ATR
    "bb_vs_kc_ratio",       # BB width / KC width (squeeze compression)
]

_EP = 1e-10


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all AI feature columns. Requires `atr` column (from _prepare).

    If the frame was already run through `_prepare()` (which is always the
    case in this bot — AIEnsemble only ever receives m15p), the base
    indicators (rsi, macd_hist, stoch_k/d, adx, di_plus/minus) are already
    present and are reused instead of being recomputed with pandas_ta a
    second time — this was previously done twice per call for no reason.
    """
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    open_ = df["open"]

    has_rsi   = "rsi" in df.columns and df["rsi"].notna().any()
    has_macd  = "macd_hist" in df.columns and df["macd_hist"].notna().any()
    has_stoch = "stoch_k" in df.columns and df["stoch_k"].notna().any()
    has_adx   = "adx" in df.columns and df["adx"].notna().any()

    for p in [1, 2, 3, 5, 10]:
        df[f"ret_{p}"] = close.pct_change(p)

    hl        = (high - low).clip(lower=_EP)
    body_top  = df[["open", "close"]].max(axis=1)
    body_bot  = df[["open", "close"]].min(axis=1)
    df["body_ratio"] = (close - open_) / hl
    df["upper_wick"] = (high - body_top) / hl
    df["lower_wick"] = (body_bot - low) / hl
    atr_col          = df.get("atr", pd.Series(dtype=float))
    df["range_atr"]  = hl / (atr_col + _EP)

    for period in [20, 50, 100, 200]:
        ema = ta.ema(close, length=period)
        df[f"_ema{period}"] = ema
        df[f"dist_ema{period}"] = (close - ema) / (ema + _EP)
    df["ema20_50_spread"]  = (df["_ema20"]  - df["_ema50"])  / (df["_ema50"]  + _EP)
    df["ema50_200_spread"] = (df["_ema50"]  - df["_ema200"]) / (df["_ema200"] + _EP)

    if has_macd:
        # already computed by _prepare() — just derive the delta
        df["macd_hist_delta"] = df["macd_hist"] - df["macd_hist"].shift(1)
        if "macd_line" in df.columns and "macd_signal" in df.columns:
            df["macd_vs_signal"] = df["macd_line"] - df["macd_signal"]
    else:
        try:
            macd_df = ta.macd(close, fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                h_col = next((c for c in macd_df.columns if "MACDh" in c), None)
                s_col = next((c for c in macd_df.columns if "MACDs" in c), None)
                m_col = next((c for c in macd_df.columns if c.startswith("MACD_")), None)
                if h_col:
                    df["macd_hist"]       = macd_df[h_col]
                    df["macd_hist_delta"] = df["macd_hist"] - df["macd_hist"].shift(1)
                if m_col and s_col:
                    df["macd_vs_signal"] = macd_df[m_col] - macd_df[s_col]
        except Exception:
            pass

    if has_rsi:
        df["rsi_delta"] = df["rsi"] - df["rsi"].shift(1)
    else:
        df["rsi"]       = ta.rsi(close, length=14)
        df["rsi_delta"] = df["rsi"] - df["rsi"].shift(1)

    if has_stoch:
        df["stoch_spread"] = df["stoch_k"] - df["stoch_d"]
    else:
        try:
            stoch_df = ta.stoch(high, low, close, k=14, d=3)
            if stoch_df is not None and not stoch_df.empty:
                k_col = next((c for c in stoch_df.columns if "STOCHk" in c), None)
                d_col = next((c for c in stoch_df.columns if "STOCHd" in c), None)
                if k_col:
                    df["stoch_k"] = stoch_df[k_col]
                if d_col:
                    df["stoch_d"] = stoch_df[d_col]
                df["stoch_spread"] = df.get("stoch_k", pd.Series(dtype=float)) \
                                     - df.get("stoch_d", pd.Series(dtype=float))
        except Exception:
            pass

    if has_adx:
        if "di_plus" in df.columns and "di_minus" in df.columns:
            df["di_spread"] = df["di_plus"] - df["di_minus"]
    else:
        try:
            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None and not adx_df.empty:
                adx_c = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
                dmp_c = next((c for c in adx_df.columns if c.startswith("DMP_")), None)
                dmn_c = next((c for c in adx_df.columns if c.startswith("DMN_")), None)
                if adx_c:
                    df["adx"] = adx_df[adx_c]
                if dmp_c and dmn_c:
                    df["di_spread"] = adx_df[dmp_c] - adx_df[dmn_c]
        except Exception:
            pass

    try:
        bb_df = ta.bbands(close, length=20, std=2.0)
        if bb_df is not None and not bb_df.empty:
            p_col = next((c for c in bb_df.columns if "BBP" in c), None)
            w_col = next((c for c in bb_df.columns if "BBB" in c), None)
            if p_col:
                df["bb_pct_b"] = bb_df[p_col]
            if w_col:
                df["bb_width"] = bb_df[w_col]
    except Exception:
        pass

    df["atr_pct"] = atr_col / (close + _EP)

    if "vol_ratio" not in df.columns or df["vol_ratio"].isna().all():
        vol_sma = df["tick_volume"].rolling(20).mean()
        df["vol_ratio"] = df["tick_volume"] / (vol_sma + _EP)

    if "_high20" in df.columns and "_low20" in df.columns:
        h20, l20 = df["_high20"], df["_low20"]
    else:
        h20 = high.rolling(20).max()
        l20 = low.rolling(20).min()
    df["dist_high20_pct"] = (h20 - close) / (close + _EP)
    df["dist_low20_pct"]  = (close - l20) / (close + _EP)

    try:
        df["cci"]        = ta.cci(high, low, close, length=14)
        df["williams_r"] = ta.willr(high, low, close, length=14)
    except Exception:
        pass

    up_bar = (close > close.shift(1)).astype(float)
    df["up_bars_5"]  = up_bar.rolling(5).sum()
    df["up_bars_10"] = up_bar.rolling(10).sum()

    # --- POWER-UP: new engineered features -------------------------------
    # ATR slope: is volatility expanding or contracting over the last 5 bars?
    df["atr_slope"] = (atr_col - atr_col.shift(5)) / (atr_col.shift(5) + _EP)

    # EMA50 slope: trend acceleration, independent of price-vs-EMA distance
    ema50_local = df.get("_ema50", ta.ema(close, length=50))
    df["ema50_slope"] = (ema50_local - ema50_local.shift(5)) / (ema50_local.shift(5) + _EP)

    # Keltner channel width (ATR-based) as a normalised volatility regime feature
    try:
        kc_df = ta.kc(high, low, close, length=20, scalar=2.0)
        if kc_df is not None and not kc_df.empty:
            lo_col = next((c for c in kc_df.columns if c.startswith("KCL")), None)
            up_col = next((c for c in kc_df.columns if c.startswith("KCU")), None)
            if lo_col and up_col:
                df["keltner_width"] = (kc_df[up_col] - kc_df[lo_col]) / (close + _EP)
    except Exception:
        pass
    if "keltner_width" not in df.columns:
        df["keltner_width"] = (4.0 * atr_col) / (close + _EP)

    # OBV slope: is volume confirming the recent price move?
    try:
        obv = ta.obv(close, df["tick_volume"])
        df["obv_slope"] = obv.diff(10) / (obv.rolling(20).std() + _EP)
    except Exception:
        df["obv_slope"] = 0.0

    # Higher-high / lower-low counts over the last 5 bars – simple structure momentum
    df["hh_count_5"] = (high > high.shift(1)).rolling(5).sum()
    df["ll_count_5"] = (low  < low.shift(1)).rolling(5).sum()

    # Momentum acceleration: 2nd derivative of returns (is momentum speeding up?)
    df["accel"] = df["ret_1"] - df["ret_1"].shift(1)

    # --- SCALPER POWER-UP: fast microstructure features ---
    ema9_local = df.get("ema9", ta.ema(close, length=9))
    ema21_local = df.get("ema21", ta.ema(close, length=21))
    df["dist_ema9"] = (close - ema9_local) / (ema9_local + _EP)
    df["dist_ema21"] = (close - ema21_local) / (ema21_local + _EP)
    df["ema9_21_spread"] = (ema9_local - ema21_local) / (ema21_local + _EP)

    # Pinbar wick delta (positive = buyers rejected low, negative = sellers rejected high)
    df["pinbar_wick"] = df["lower_wick"] - df["upper_wick"]

    # Bollinger Bands squeeze ratio
    bb_w = df.get("bb_width", pd.Series(0.0, index=df.index))
    df["bb_squeeze"] = bb_w / (bb_w.rolling(20).mean() + _EP)

    # -----------------------------------------------------------------------
    # PRO TRADER FEATURES
    # -----------------------------------------------------------------------
    # 1. ADX slope — is the trend accelerating or decelerating?
    adx_s = df.get("adx", pd.Series(0.0, index=df.index))
    df["adx_slope"] = (adx_s - adx_s.shift(3)) / (adx_s.shift(3) + _EP)

    # 2. Regime score: EMA50 slope direction × ADX magnitude → -1..+1
    #    Positive = bullish trend, negative = bearish, near-zero = ranging
    ema50_s = df.get("_ema50", df.get("ema50", ta.ema(close, length=50)))
    ema50_slope_5 = (ema50_s - ema50_s.shift(5)) / (ema50_s.shift(5) + _EP)
    adx_norm = (adx_s / 50.0).clip(0.0, 1.0)  # normalise ADX 0-50 → 0-1
    # sign of EMA slope drives direction; ADX amplitude drives magnitude
    df["regime_score"] = ema50_slope_5.apply(
        lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
    ) * adx_norm

    # 3. Keltner-Bollinger squeeze flag (Lazy Bear squeeze concept)
    #    BB is inside KC → range compression → explosive breakout pending
    bb_w_s = df.get("bb_width", pd.Series(pd.NA, index=df.index))
    kc_w_s = df.get("keltner_width", (4.0 * atr_col) / (close + _EP))
    df["kc_squeeze"] = (bb_w_s < kc_w_s).astype(float).fillna(0.0)
    df["bb_vs_kc_ratio"] = bb_w_s / (kc_w_s + _EP)

    # 4. Pullback-to-value distance (in ATR units — broker-independent)
    ema21_s = df.get("ema21", ta.ema(close, length=21))
    ema50_v = df.get("_ema50", ema50_s)
    df["dist_ema21_atr"] = (close - ema21_s) / (atr_col + _EP)
    df["dist_ema50_atr"] = (close - ema50_v) / (atr_col + _EP)
    #    "At value area" = price is within 0.75 ATR of EMA21 or EMA50 (pullback zone)
    df["at_value_area"] = (
        (df["dist_ema21_atr"].abs() < 0.75) | (df["dist_ema50_atr"].abs() < 0.75)
    ).astype(float)

    # 5. Multi-bar momentum confirmation
    rsi_s = df.get("rsi", pd.Series(50.0, index=df.index))
    macd_h = df.get("macd_hist", pd.Series(0.0, index=df.index))
    #    RSI trending: difference over 2 bars (positive = rising momentum)
    df["rsi_trending_2"] = rsi_s - rsi_s.shift(2)
    #    MACD histogram: expanding (same sign and larger) over 2 bars
    macd_delta2 = macd_h - macd_h.shift(2)
    same_sign = (macd_h * macd_h.shift(2) > 0).astype(float)
    df["macd_hist_trending_2"] = macd_delta2 * same_sign
    #    Consecutive directional closes: net -3..+3 (e.g. +3 = all 3 bars up)
    dir_bar = (close > close.shift(1)).astype(float) * 2 - 1  # +1 or -1
    df["consecutive_dir_3"] = (
        dir_bar + dir_bar.shift(1) + dir_bar.shift(2)
    )

    # 6. Volatility regime: current ATR relative to its own 20-bar mean
    atr_mean20 = atr_col.rolling(20).mean()
    df["atr_vs_20d"] = atr_col / (atr_mean20 + _EP)

    return df


@dataclass
class AISignal:
    direction: int    = 0      # +1 long | -1 short | 0 uncertain
    confidence: float = 0.5
    reason: str       = ""


def _make_pipeline() -> Pipeline:
    # NOTE on n_jobs: every estimator below is intentionally single-process
    # (n_jobs=1). n_jobs=-1 tells scikit-learn/joblib to fan training out
    # across separate OS processes, and doing that from a background
    # (non-main) thread inside a Tkinter GUI — or from a frozen PyInstaller
    # .exe — is exactly the scenario that produces intermittent
    # "worker process was unexpectedly terminated / segmentation fault"
    # crashes. Training 500 bars with ~150 trees is fast enough
    # single-process (well under a couple of seconds), and it only runs
    # once every `ai_retrain_bars` bars, so the stability trade-off is
    # clearly worth it for a live trading bot.
    rf  = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=20,
        random_state=42, n_jobs=1)
    gbt = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    et  = ExtraTreesClassifier(
        n_estimators=150, max_depth=8, min_samples_split=20,
        random_state=42, n_jobs=1)
    # POWER-UP (D): 4th model — HistGradientBoostingClassifier. It's fast,
    # handles feature interactions well, and adds diversity to the vote
    # since it builds trees on binned features rather than raw splits.
    hgb = HistGradientBoostingClassifier(
        max_depth=6, max_iter=150, learning_rate=0.08,
        l2_regularization=0.1, random_state=42)
    voter = VotingClassifier(
        estimators=[("rf", rf), ("gbt", gbt), ("et", et), ("hgb", hgb)],
        voting="soft", weights=[1.0, 1.0, 1.0, 1.2], n_jobs=1)
    return Pipeline([("scaler", StandardScaler()), ("model", voter)])


_MIN_TRAIN = 150
_MIN_PRED  = 50
# Smallest out-of-sample split we will accept. Below this the measured
# accuracy is noise, so we train without a promotion gate rather than promote
# a model on 12 samples.
_MIN_VALIDATION = 40
# Calibration cannot produce a probability outside this band. Purely
# empirical hit rates can reach 0.0/1.0 on a small validation split, and a
# reported confidence of exactly 1.0 would be a lie — and would also make
# any confidence-based sizing threshold trivially pass.
_CALIBRATION_FLOOR = 0.02


def _platt_scale(raw_prob, y_val) -> Optional[LogisticRegression]:
    """Fit a Platt (sigmoid) calibrator mapping raw P(UP) -> empirical P(UP).

    Why sigmoid and not isotonic: the validation split is only ~20% of a
    few hundred labelled bars. Isotonic regression is non-parametric and will
    happily memorise that split, producing a step function that looks
    perfectly calibrated on the very data it was fitted to and misfires on
    the next bar. Platt scaling fits exactly two parameters (slope and
    intercept), which is the right model complexity for this sample size.

    Returns None when there is not enough data, or when the split contains a
    single class (there is nothing to calibrate *towards*).
    """
    p = np.asarray(raw_prob, dtype=float).ravel()
    y = np.asarray(y_val, dtype=int).ravel()
    if len(p) < _MIN_VALIDATION or len(set(y.tolist())) < 2:
        return None
    # logit transform, clipped so 0.0 and 1.0 do not produce infinities
    eps = 1e-6
    p = np.clip(p, eps, 1.0 - eps)
    z = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        cal = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
        cal.fit(z, y)
    except Exception as exc:
        logger.warning("[AI] calibration fit failed (%s) — using raw probability", exc)
        return None
    return cal


def _calibrated_prob(cal: Optional[LogisticRegression], prob: float) -> float:
    """Apply a fitted calibrator to a raw probability."""
    if cal is None:
        return prob
    eps = 1e-6
    p = float(np.clip(prob, eps, 1.0 - eps))
    z = np.array([[math.log(p / (1.0 - p))]])
    try:
        out = float(cal.predict_proba(z)[0][1])
    except Exception:
        return p
    return min(1.0 - _CALIBRATION_FLOOR, max(_CALIBRATION_FLOOR, out))


def _mean_feature_importance(pipeline: Pipeline, feature_cols: List[str]) -> dict:
    """Average feature importance across the tree sub-estimators.

    Diagnostic only — used for logging so it is possible to see what the
    model is actually keying on. HistGradientBoosting has no
    ``feature_importances_`` and is skipped rather than guessed at.
    """
    try:
        voter = pipeline.named_steps["model"]
        acc = np.zeros(len(feature_cols), dtype=float)
        used = 0
        for est in voter.estimators_:
            fi = getattr(est, "feature_importances_", None)
            if fi is None or len(fi) != len(feature_cols):
                continue
            acc += np.asarray(fi, dtype=float)
            used += 1
        if used == 0:
            return {}
        acc /= used
        return dict(sorted(
            zip(feature_cols, acc.tolist()), key=lambda kv: kv[1], reverse=True
        ))
    except Exception:
        return {}


class AIEnsemble:
    """Stateful per-symbol ensemble: trains on roll, predicts direction.

    Champion/challenger design. Every retrain produces a *challenger* that
    is fitted on the older part of the rolling window and scored on a
    strictly later, never-trained-on validation split. The challenger only
    replaces the live *champion* if its out-of-sample accuracy clears the
    configured floor and is not meaningfully worse than the incumbent. This
    is what stops a retrain that has silently decayed into a coin flip from
    going live and quietly driving the account.
    """

    def __init__(self, symbol: str, config):
        self.symbol          = symbol
        self.config          = config
        self._pipeline: Optional[Pipeline] = None
        self._calibrator: Optional[LogisticRegression] = None
        self._feature_cols: List[str] = FEATURE_COLS[:]
        self._bars_since_train: int = 0
        self._n_trains: int = 0
        self._n_attempts: int = 0
        self._n_promotions: int = 0
        self._n_rejections: int = 0
        self._last_oos_accuracy: Optional[float] = None
        self._last_oos_auc: Optional[float] = None
        self._last_oos_base_rate: Optional[float] = None
        self._last_oos_edge: Optional[float] = None
        self._last_reject_reason: str = ""
        self._feature_importance: dict = {}
        self._trained: bool = False
        self._lock = threading.Lock()   # guards pipeline swap during training

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Highest-weighted features of the live model, for logging/audit."""
        with self._lock:
            items = list(self._feature_importance.items())
        return items[:n]

    def _build_xy(self, df):
        """Build feature matrix X and label vector y.

        Label definition (pro-grade):
          - 1 (UP)   if the *best* close in the next 3 bars exceeds
                     current close by ≥ 0.4 × ATR
          - 0 (DOWN) if the *worst* close in the next 3 bars falls
                     below current close by ≥ 0.4 × ATR
          - NaN      if the move is below the ATR threshold in both
                     directions — ambiguous, dropped from training.

        This teaches the model to distinguish *meaningful* directional
        moves from noise, the way a professional trader would frame it:
        "did this setup produce a move worth trading?"
        """
        df = build_features(df)
        cl  = df["close"]
        atr = df.get("atr", pd.Series(0.0, index=df.index))
        threshold = 0.40  # ATR multiples — tune via ATR_LABEL_MULT env var if needed

        # Best & worst close over next 3 bars (forward-looking)
        fwd_max = cl.shift(-1).combine(cl.shift(-2), max).combine(cl.shift(-3), max)
        fwd_min = cl.shift(-1).combine(cl.shift(-2), min).combine(cl.shift(-3), min)

        move_up   = (fwd_max - cl) / (atr + _EP)
        move_down = (cl - fwd_min) / (atr + _EP)

        target = pd.Series(float("nan"), index=df.index)
        target[move_up   >= threshold] = 1.0
        target[move_down >= threshold] = 0.0
        # Conflicted bars (both sides move ≥ threshold) → drop (chop)
        target[(move_up >= threshold) & (move_down >= threshold)] = float("nan")

        df["_target"] = target
        df = df.iloc[:-3].copy()   # drop last 3 rows (no forward data)
        avail = [c for c in FEATURE_COLS if c in df.columns]
        df_clean = df[avail + ["_target"]].dropna()
        if len(df_clean) < _MIN_TRAIN:
            return None, None, avail
        return df_clean[avail].values, df_clean["_target"].values.astype(int), avail

    def train(self, df: pd.DataFrame) -> bool:
        """Fit a challenger and promote it only if it validates out-of-sample.

        The rolling window is split by TIME, never shuffled: the older
        portion trains, the newer portion validates. Because the validation
        bars are strictly later than anything the model was fitted on, the
        measured accuracy is an honest out-of-sample estimate rather than a
        training score dressed up as one.
        """
        n = min(len(df), self.config.AI_TRAIN_BARS)
        try:
            X, y, avail = self._build_xy(df.iloc[-n:].copy())
        except Exception as exc:
            # Feature engineering on a short or malformed frame raises rather
            # than returning None. train() must never propagate: update() calls
            # it from the main trading loop.
            logger.warning("[AI] %s feature build failed: %s", self.symbol, exc)
            return False
        if X is None:
            logger.warning("[AI] %s: need >=%d samples, skipping train",
                           self.symbol, _MIN_TRAIN)
            return False
        up_rate = y.mean()
        if up_rate < 0.05 or up_rate > 0.95:
            logger.warning("[AI] %s: degenerate labels (%.0f%% up) – skipping",
                           self.symbol, up_rate * 100)
            return False

        # --- time-ordered train / validation split -------------------------
        val_frac = min(max(float(self.config.AI_VALIDATION_FRACTION), 0.0), 0.5)
        n_val = int(len(X) * val_frac)
        y_va = None
        if n_val >= _MIN_VALIDATION and (len(X) - n_val) >= _MIN_TRAIN:
            X_tr, y_tr = X[:-n_val], y[:-n_val]
            X_va, y_va = X[-n_val:], y[-n_val:]
        else:
            # Window too small to validate honestly. Still train, but record
            # that there is no measured edge rather than implying one.
            logger.warning("[AI] %s: %d samples too small for a %d-bar "
                           "validation split — training ungated",
                           self.symbol, len(X), _MIN_VALIDATION)
            X_tr, y_tr = X, y

        try:
            pipe = _make_pipeline()
            # POWER-UP (D): recency-weighted training — recent bars get up
            # to ~2.5x the weight of the oldest bars in the window, so the
            # model adapts faster to the current regime instead of treating
            # a bar from 500 candles ago the same as the latest one.
            sample_weight = np.linspace(0.4, 1.0, len(X_tr))
            pipe.fit(X_tr, y_tr, model__sample_weight=sample_weight)
        except Exception as exc:
            logger.error("[AI] %s training error: %s", self.symbol, exc)
            return False

        if y_va is None:
            # No validation split was possible: fit as before, but do not
            # pretend we measured anything.
            self._promote(pipe, None, avail, len(X_tr))
            return True

        # --- out-of-sample measurement -------------------------------------
        up_idx = self._up_index(pipe)
        if up_idx is None:
            self._reject("fitted model has no UP class")
            return False
        try:
            raw_up = pipe.predict_proba(X_va)[:, up_idx]
            # Predict-then-measure on the validation split, using the same
            # 0.5 decision boundary that predict() applies live.
            oos_acc = float(np.mean((raw_up >= 0.5).astype(int) == y_va))
            try:
                oos_auc = float(roc_auc_score(y_va, raw_up))
            except Exception:
                oos_auc = None
        except Exception as exc:
            logger.error("[AI] %s validation error: %s", self.symbol, exc)
            return False

        # Accuracy alone is not evidence of skill. The best trivial predictor
        # is "always answer the majority class", and whichever class that is,
        # it scores max(base_rate, 1 - base_rate). A model that cannot match
        # that has negative expectancy against the do-nothing alternative.
        # NOTE: comparing against the raw UP rate instead would be wrong --
        # on a 38% UP window the trivial baseline scores 0.62, not 0.38.
        base_rate = float(np.mean(y_va))
        baseline = max(base_rate, 1.0 - base_rate)
        oos_edge = oos_acc - baseline

        # --- promotion gate ------------------------------------------------
        gate_ok, gate_reason = self._passes_gate(oos_acc, oos_auc, base_rate)
        if not gate_ok:
            self._reject(gate_reason)
            return False

        if oos_auc is not None and oos_auc < 0.55:
            # Not blocking — just honest. Near-chance ranking means most of the
            # accuracy above is the base rate, not the model.
            logger.warning(
                "[AI] %s ranking skill is marginal (auc=%.3f, base_rate=%.3f, "
                "acc=%.3f, lift=%+.3f) — promoted, but treat confidence sceptically",
                self.symbol, oos_auc, base_rate, oos_acc, oos_edge)

        calibrator = _platt_scale(raw_up, y_va) if self.config.AI_CALIBRATE else None
        self._promote(pipe, calibrator, avail, len(X_tr),
                      oos_acc=oos_acc, oos_auc=oos_auc,
                      oos_base_rate=base_rate, oos_edge=oos_edge)
        return True

    def _passes_gate(self, oos_acc: float, oos_auc: Optional[float],
                     base_rate: float) -> Tuple[bool, str]:
        """Decide whether a challenger is fit to trade. Pure: (ok, reason).

        ``base_rate`` is the fraction of UP labels in the validation window.
        The trivial baseline is the majority class, which scores
        ``max(base_rate, 1 - base_rate)``.

        Kept separate from train() so the promotion policy can be exercised
        directly, without fitting a model to test it.
        """
        floor = float(self.config.AI_MIN_OOS_ACCURACY)
        if oos_acc < floor:
            return False, ("oos accuracy %.3f below floor %.3f" % (oos_acc, floor))

        if oos_auc is not None:
            auc_floor = float(self.config.AI_MIN_OOS_AUC)
            if oos_auc < auc_floor:
                return False, ("oos auc %.3f below floor %.3f" % (oos_auc, auc_floor))

        # The decisive check. A model that cannot match "always answer the
        # majority class" has negative expectancy and is not allowed to trade,
        # no matter how confident it looks. Defaults to 0.0; set
        # AI_MIN_OOS_EDGE to a negative value to disable.
        majority = "UP" if base_rate >= 0.5 else "DOWN"
        baseline = max(base_rate, 1.0 - base_rate)
        lift = oos_acc - baseline
        min_edge_gate = float(self.config.AI_MIN_OOS_EDGE)
        if lift < min_edge_gate:
            return False, (
                "oos lift %+.3f below floor %+.3f (acc %.3f vs baseline %.3f) "
                "- model is worse than always answering %s"
                % (lift, min_edge_gate, oos_acc, baseline, majority))

        with self._lock:
            incumbent_acc = self._last_oos_accuracy
        tol = float(self.config.AI_PROMOTE_TOLERANCE)
        if incumbent_acc is not None and oos_acc < incumbent_acc - tol:
            return False, ("oos accuracy %.3f worse than incumbent %.3f"
                           % (oos_acc, incumbent_acc))

        return True, ""

    @staticmethod
    def _up_index(pipe: Pipeline) -> Optional[int]:
        """Column index of the UP (label 1) probability, from the fitted model.

        Never assume it is index 1 — the column order follows whatever classes
        the estimator happened to observe.
        """
        try:
            classes = np.asarray(pipe.named_steps["model"].classes_)
        except Exception:
            return None
        hits = np.where(classes == 1)[0]
        return int(hits[0]) if len(hits) else None

    def _reject(self, reason: str) -> None:
        """Record a rejected challenger. The live model is left untouched."""
        with self._lock:
            self._n_rejections += 1
            self._last_reject_reason = reason
        logger.warning("[AI] %s challenger rejected (%s) — keeping live model "
                       "if one exists", self.symbol, reason)

    def _promote(self, pipe, calibrator, avail, n_train,
                 oos_acc: Optional[float] = None,
                 oos_auc: Optional[float] = None,
                 oos_base_rate: Optional[float] = None,
                 oos_edge: Optional[float] = None) -> None:
        """Atomically install a validated challenger as the live model."""
        importance = _mean_feature_importance(pipe, avail)
        with self._lock:
            self._pipeline     = pipe
            self._calibrator   = calibrator
            self._feature_cols = avail
            self._feature_importance = importance
            self._trained      = True
            self._n_trains    += 1
            self._n_promotions += 1
            self._last_oos_accuracy = oos_acc
            self._last_oos_auc = oos_auc
            self._last_oos_base_rate = oos_base_rate
            self._last_oos_edge = oos_edge
            self._last_reject_reason = ""
        top = ", ".join("%s=%.3f" % (k, v) for k, v in list(importance.items())[:5])
        logger.info(
            "[AI] %s promoted run #%d | train=%d | oos_acc=%s | auc=%s | "
            "base=%s | lift=%s | calibrated=%s | top: %s",
            self.symbol, self._n_trains, n_train,
            ("%.3f" % oos_acc) if oos_acc is not None else "n/a",
            ("%.3f" % oos_auc) if oos_auc is not None else "n/a",
            ("%.3f" % oos_base_rate) if oos_base_rate is not None else "n/a",
            ("%+.3f" % oos_edge) if oos_edge is not None else "n/a",
            calibrator is not None,
            top or "n/a")

    def update(self, df: pd.DataFrame) -> None:
        """Retrain when due. Runs in a background thread so a slow fit
        (four ensembles, hundreds of trees) never blocks order management
        or the main loop. predict() reads self._pipeline under a lock so
        it always sees either the old or the fully-trained new model,
        never a half-written one."""
        self._bars_since_train += 1
        # The first attempt is immediate; after that the cadence governs
        # REGARDLESS of whether a model was ever promoted. Testing
        # `not self._trained` here would re-arm every single bar for a model
        # that never clears the gate -- a full four-ensemble fit per candle,
        # forever, in exchange for a model that still may not trade.
        first_attempt = self._n_attempts == 0
        if not (first_attempt
                or self._bars_since_train >= self.config.AI_RETRAIN_BARS):
            return
        self._bars_since_train = 0
        # Count the ATTEMPT here, not inside train(). train() can raise
        # before it reaches its own bookkeeping, and a counter that misses
        # those raises leaves the model looking brand new forever -- which is
        # exactly the spin-on-every-bar pathology this guard prevents.
        self._n_attempts += 1
        if first_attempt:
            # Synchronous: we have nothing to trade with yet, and the caller
            # is the main loop.
            try:
                self.train(df)
            except Exception as exc:
                logger.error("[AI] %s initial training error: %s", self.symbol, exc)
            return
        snapshot = df.copy()
        threading.Thread(target=self.train, args=(snapshot,), daemon=True).start()

    def predict(self, df: pd.DataFrame) -> AISignal:
        with self._lock:
            trained, pipeline, feature_cols = (
                self._trained, self._pipeline, self._feature_cols)
            calibrator = self._calibrator
            oos_acc = self._last_oos_accuracy
            oos_edge = self._last_oos_edge
            reject_reason = self._last_reject_reason
        if not trained or pipeline is None:
            if reject_reason:
                return AISignal(0, 0.5, "model not promoted: %s" % reject_reason)
            return AISignal(0, 0.5, "model not trained yet")
        if len(df) < _MIN_PRED:
            return AISignal(0, 0.5, "not enough bars for prediction")
        try:
            feat_df = build_features(df)
            avail   = [c for c in feature_cols if c in feat_df.columns]
            row     = feat_df[avail].iloc[[-1]]
            if row.isnull().any(axis=1).values[0]:
                return AISignal(0, 0.5, "NaN in feature row")
            up_idx = self._up_index(pipeline)
            if up_idx is None:
                return AISignal(0, 0.5, "model has no UP class")
            raw_up = float(pipeline.predict_proba(row.values)[0][up_idx])
            # Calibrated probability, not the raw soft-vote output. This is
            # the number that drives the confluence score and the position
            # size, so it has to mean what it says.
            prob_up = _calibrated_prob(calibrator, raw_up)
            thr     = self.config.AI_CONFIDENCE_THRESHOLD
            oos_txt = (" oos=%.3f" % oos_acc) if oos_acc is not None else ""
            if oos_edge is not None:
                oos_txt += " lift=%+.3f" % oos_edge
            if prob_up >= thr:
                return AISignal(+1, prob_up,
                                "AI LONG  conf=%.3f (raw %.3f)%s" % (prob_up, raw_up, oos_txt))
            if prob_up <= 1.0 - thr:
                conf = 1.0 - prob_up
                return AISignal(-1, conf,
                                "AI SHORT conf=%.3f (raw %.3f)%s" % (conf, raw_up, oos_txt))
            return AISignal(0, max(prob_up, 1.0 - prob_up),
                            "AI uncertain p=%.3f%s" % (prob_up, oos_txt))
        except Exception as exc:
            logger.error("[AI] %s predict error: %s", self.symbol, exc)
            return AISignal(0, 0.5, f"predict error: {exc}")


# ---------------------------------------------------------------------------
# Signal scoring (multi-layer confluence 0-100)
# ---------------------------------------------------------------------------

def _session_ok(cfg: dict) -> bool:
    if not cfg.get("use_session_filter", True):
        return True
    h = datetime.now(timezone.utc).hour
    return ((cfg.get("london_start_utc", 7)  <= h < cfg.get("london_end_utc", 16)) or
            (cfg.get("ny_start_utc", 12)     <= h < cfg.get("ny_end_utc", 21)))


def _trend_score(df, direction: int) -> float:
    if df is None or df.empty:
        return 0.0
    last = df.iloc[-1]
    ef   = last.get("ema50",  float("nan"))
    es   = last.get("ema200", float("nan"))
    e9   = last.get("ema9",   float("nan"))
    e21  = last.get("ema21",  float("nan"))
    rsi  = last.get("rsi", 50.0)
    if pd.isna(ef) or pd.isna(es):
        if not pd.isna(e9) and not pd.isna(e21):
            return 0.7 if (e9 > e21 if direction == 1 else e9 < e21) else 0.2
        return 0.0
    if direction == 1:
        ema_ok = 1.0 if ef > es else 0.0
        # Fast scalp ribbon alignment bonus (e9 > e21 > ef)
        if not pd.isna(e9) and not pd.isna(e21) and e9 > e21 and e21 > ef:
            ema_ok = 1.15
        rsi_ok = min(1.0, max(0.0, (rsi - 45.0) / 20.0))
    else:
        ema_ok = 1.0 if ef < es else 0.0
        # Fast scalp ribbon alignment bonus (e9 < e21 < ef)
        if not pd.isna(e9) and not pd.isna(e21) and e9 < e21 and e21 < ef:
            ema_ok = 1.15
        rsi_ok = min(1.0, max(0.0, (55.0 - rsi) / 20.0))
    return min(1.0, 0.65 * ema_ok + 0.35 * rsi_ok)


def _momentum_score(last: pd.Series, direction: int) -> float:
    rsi  = float(last.get("rsi", 50.0))
    mh   = float(last.get("macd_hist", 0.0))
    mhp  = float(last.get("macd_hist_prev", 0.0))
    sk   = float(last.get("stoch_k", 50.0))
    sd   = float(last.get("stoch_d", 50.0))
    e9   = float(last.get("ema9", 0.0))
    e21  = float(last.get("ema21", 0.0))
    cl   = float(last.get("close", 0.0))

    score = 0.0
    if direction == 1:
        score += min(1.0, max(0.0, (rsi - 45.0) / 25.0)) * 0.35
        score += (1.0 if mh > 0 and mh > mhp else 0.5 if mh > 0 else 0.0) * 0.35
        # Reward both oversold bounce and upward momentum continuation
        score += (1.0 if sk > sd and (sk < 35 or 50 < sk < 80) else 0.5 if sk > sd else 0.0) * 0.30
        # Scalper EMA 9/21 ribbon & price action bonus
        if e9 > 0 and e21 > 0 and cl >= e9 and e9 > e21:
            score += 0.10
        if bool(last.get("bull_pinbar", False)) or bool(last.get("bull_engulf", False)):
            score += 0.10
    else:
        score += min(1.0, max(0.0, (55.0 - rsi) / 25.0)) * 0.35
        score += (1.0 if mh < 0 and mh < mhp else 0.5 if mh < 0 else 0.0) * 0.35
        # Reward both overbought rejection and downward momentum continuation
        score += (1.0 if sk < sd and (sk > 65 or 20 < sk < 50) else 0.5 if sk < sd else 0.0) * 0.30
        # Scalper EMA 9/21 ribbon & price action bonus
        if e9 > 0 and e21 > 0 and cl <= e9 and e9 < e21:
            score += 0.10
        if bool(last.get("bear_pinbar", False)) or bool(last.get("bear_engulf", False)):
            score += 0.10

    return min(1.0, score)


def _volume_score(last: pd.Series) -> float:
    vr = float(last.get("vol_ratio", 1.0))
    if pd.isna(vr):
        return 0.5
    return min(1.0, max(0.0, (vr - 0.5) / 1.5))


def _structure_score(df: pd.DataFrame, direction: int) -> float:
    if len(df) < 20:
        return 0.5
    last = df.iloc[-1]
    cl   = float(last["close"])
    atr  = float(last.get("atr", 0.0))
    if atr <= 0:
        return 0.5
    # reuse the rolling 20-bar high/low cached by _prepare() instead of
    # recomputing df["high"].rolling(20).max() again for every direction
    if "_high20" in df.columns and "_low20" in df.columns:
        h20 = float(last["_high20"])
        l20 = float(last["_low20"])
    else:
        h20 = float(df["high"].rolling(20).max().iloc[-1])
        l20 = float(df["low"].rolling(20).min().iloc[-1])
    room = ((h20 - cl) / atr) if direction == 1 else ((cl - l20) / atr)
    return min(1.0, max(0.0, room / 5.0))


def _entry_quality_score(last: pd.Series) -> float:
    """Score how well the entry aligns with value area (0-1).

    A professional trader waits for price to pull back into EMA21/EMA50
    before entering — they don't chase extended moves. This score is highest
    when price is near a key EMA and at_value_area is true.

    Returns a 0-1 score that feeds into _score_direction as 10 points.
    """
    at_val   = float(last.get("at_value_area", 0.5))
    d21_atr  = abs(float(last.get("dist_ema21_atr", 1.0)))
    d50_atr  = abs(float(last.get("dist_ema50_atr", 1.5)))
    pinbar   = float(last.get("pinbar_wick", 0.0))

    # Base: at value area gives good score
    base = at_val * 0.6

    # Proximity bonus: closer to EMA21 → higher quality pullback
    prox21 = max(0.0, 1.0 - d21_atr / 2.0)  # 1.0 at EMA21, 0 at 2×ATR away
    prox50 = max(0.0, 1.0 - d50_atr / 3.0)  # 1.0 at EMA50, 0 at 3×ATR away

    # Pinbar bonus: strong rejection candle at value area = high quality
    pinbar_bonus = min(0.2, abs(pinbar) * 0.3) if at_val > 0 else 0.0

    return min(1.0, base + max(prox21, prox50) * 0.4 + pinbar_bonus)


def _market_regime(last: pd.Series, cfg: dict) -> str:
    """Classify the current market regime: TREND_UP, TREND_DOWN, or RANGING.

    Uses ADX, regime_score (EMA50 slope × ADX), and Keltner squeeze.
    A professional trader adapts strategy to regime — they don't trade
    range-bound markets with trend-following entries.

    Returns: 'TREND_UP' | 'TREND_DOWN' | 'RANGING'
    """
    adx        = float(last.get("adx", 0.0))
    regime_s   = float(last.get("regime_score", 0.0))
    kc_squeeze = float(last.get("kc_squeeze", 0.0))
    adx_min    = cfg.get("adx_min", 18.0)

    # During a squeeze, market is coiling — treat as ranging until breakout
    if kc_squeeze > 0.5 and adx < adx_min:
        return "RANGING"

    if adx < adx_min:
        return "RANGING"

    if regime_s > 0.15:
        return "TREND_UP"
    if regime_s < -0.15:
        return "TREND_DOWN"
    return "RANGING"


def _score_direction(ai_direction: int, ai_conf: float,
                     m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame,
                     direction: int, cfg: dict) -> float:
    """Total confluence score (0-100) for the given direction.

    Professional weight allocation:
      AI ensemble  45 pts  — the most predictive single component
      H4 trend     20 pts  — higher timeframe = higher reliability
      H1 trend     10 pts  — intermediate trend confirmation
      Momentum     15 pts  — RSI / MACD / Stoch confluence
      Entry quality 10 pts — pullback-to-value (EMA21/50), pinbar
      ─────────────────────
      Total        100 pts

    Volume is folded into the regime context rather than scored separately;
    it was previously too loosely correlated with direction to merit 10pts.
    """
    last       = m15.iloc[-1]
    adx_val    = float(last.get("adx", 0.0))
    adx_strong = cfg.get("adx_strong", 30)

    # AI component: 0-45 pts
    if ai_direction == direction:
        ai_pts = ai_conf * 45.0
    elif ai_direction == -direction:
        ai_pts = 0.0
    else:
        ai_pts = 13.5  # uncertain (0.5 × 27 midpoint)

    # Trend component: H4 carries 20 pts, H1 carries 10 pts (total 30)
    h1_s  = _trend_score(h1, direction)
    h4_s  = _trend_score(h4, direction)
    trend = h1_s * 10.0 + h4_s * 20.0

    # ADX boost: strong trends get up to +25% on the trend component
    if adx_val >= adx_strong:
        trend = min(30.0, trend * (1.0 + (adx_val - adx_strong) / 50.0))

    # Momentum: 15 pts
    mom = _momentum_score(last, direction) * 15.0

    # Entry quality: 10 pts (pullback-to-value, pinbar at level)
    eq = _entry_quality_score(last) * 10.0

    return ai_pts + trend + mom + eq


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------

def _risk_scale(adx: float, cfg: dict) -> float:
    lo     = cfg.get("min_risk_scale", 0.5)
    hi     = cfg.get("max_risk_scale", 1.5)
    lo_adx = cfg.get("adx_min", 20)
    hi_adx = cfg.get("adx_strong", 30) + 20
    if adx <= lo_adx:
        return lo
    if adx >= hi_adx:
        return hi
    return lo + (adx - lo_adx) / (hi_adx - lo_adx) * (hi - lo)


def _size_lot(account_equity, sl_dist_price, tick_size, tick_value,
              vol_min, vol_max, vol_step, cfg, adx=0.0, ai_conf=0.0):
    scale    = _risk_scale(adx, cfg) if cfg.get("dynamic_risk") else 1.0
    # --- AI CONFIDENCE SIZING -------------------------------------------
    # `ai_conf` is now a CALIBRATED probability (see AIEnsemble.predict),
    # so it can be compared against a confidence level directly. Size is
    # driven by the model's EDGE over a coin flip, not by how sure the model
    # sounds: a model that is confidently wrong is the expensive case, and an
    # unvalidated confidence must never be allowed to upsize a position.
    high_tier = cfg.get("ai_size_high_conf", 0.85 if _IS_AGGRESSIVE else 0.70)
    low_tier  = cfg.get("ai_size_low_conf", 0.60)
    edge      = ai_conf - 0.5
    min_edge  = cfg.get("ai_min_edge_for_size_up", 0.10)

    if ai_conf >= high_tier and edge >= min_edge:
        scale *= 1.5
    elif 0.0 < ai_conf <= low_tier:
        scale *= 0.5
    risk_amt = account_equity * cfg["risk_per_trade"] * scale
    sl_ticks = sl_dist_price / (tick_size + 1e-10)
    rpl      = sl_ticks * tick_value
    if rpl <= 0:
        return vol_min
    raw = risk_amt / rpl
    vol = math.floor(raw / vol_step) * vol_step
    return max(vol_min, min(vol, vol_max))


# ---------------------------------------------------------------------------
# Main bot class
# ---------------------------------------------------------------------------

class StandaloneAIBot:
    """Single-symbol AI trading bot with close + trailing management."""

    def __init__(self, cfg: dict):
        self.cfg            = cfg
        self.notifier       = Notifier(cfg)
        self.ai: Optional[AIEnsemble] = None
        self.running        = False
        self.last_bar_time  = None
        self.start_balance  = 0.0
        self.daily_start    = 0.0
        self.daily_date: Optional[date] = None
        self.trades_today   = 0
        self.trade_date: Optional[date] = None
        self._partial_stages: dict = {}   # ticket -> set of stage numbers done
        self._peak_profit: dict = {}      # ticket -> peak favourable move (price units), POWER-UP (B)
        self._manual_sltp_failed: dict = {} # ticket -> epoch timestamp of last failed modification
        self._dd_positions_closed = False # one-shot equity protect close

        # --- GUI hook: thread-safe status snapshot. Only this bot thread
        # ever writes to it; any other thread (e.g. a GUI) should only read
        # it via get_status() and must never call MT5 functions itself. ---
        self._status_lock = threading.Lock()
        self.status: dict = {
            "connected": False, "balance": 0.0, "equity": 0.0,
            "open_positions": 0, "trades_today": 0,
            "last_bar": None, "last_direction": 0,
            "last_score": 0.0, "last_ai_conf": 0.0, "last_signal": "",
            # --- RESULTS / performance tracking (session, since bot start) ---
            "closed_trades": 0, "wins": 0, "losses": 0, "breakeven_trades": 0,
            "total_pnl": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
            "win_rate": 0.0, "start_equity": 0.0,
        }

        # --- indicator caches: only recomputed when a new closed bar
        # actually appears, instead of on every 0.5s loop tick ---
        self._m15_bar_time  = None
        self._m15p_cache: Optional[pd.DataFrame] = None
        self._h1_bar_time   = None
        self._h1p_cache: Optional[pd.DataFrame] = None
        self._reconnect_backoff = 5  # seconds, grows on repeated failures

        # --- market location cache. Same rule as the indicator caches: the
        # location can only change when a new bar closes, and recomputing it on
        # every 0.5s loop tick would be pure waste. ---
        self._location_cache = None
        self._location_bar_time = None
        self._sltp_reasoner = None

    # --------------------------------------------------------------
    def _location_for(self, m15p: pd.DataFrame):
        """
        Market location for the latest closed bar, cached per bar.

        Returns ``None`` whenever a location cannot be produced — unavailable
        layer, insufficient history, malformed frame, or an internal error.
        Callers must treat ``None`` as "no opinion" and fall back to their
        previous behaviour. It is deliberately never a fabricated default:
        guessing a location would veto real trades on invented data.

        This method must never raise into the main loop. A live bot must not
        die because an analysis helper had a bad bar.
        """
        if not _LOCATION_AVAILABLE or _analyse_location is None:
            return None
        if m15p is None or getattr(m15p, "empty", True):
            return None

        try:
            bar_time = m15p.index[-1]
        except Exception:
            return None

        if bar_time == self._location_bar_time and self._location_cache is not None:
            return self._location_cache

        min_bars = self.cfg.get("location_min_bars", 60)
        if len(m15p) < min_bars:
            return None

        tick_size = None
        try:
            info = mt5.symbol_info(self.cfg["symbol"])
            if info is not None:
                raw = (getattr(info, "trade_tick_size", None)
                       or getattr(info, "point", None))
                # Broker metadata is untrusted: a stub or a partially
                # initialised client can return a non-numeric object here, which
                # would otherwise propagate into the snapping arithmetic and
                # silently cost us the whole location read.
                try:
                    candidate = float(raw)
                    tick_size = candidate if candidate > 0 else None
                except (TypeError, ValueError):
                    tick_size = None
        except Exception:
            tick_size = None

        try:
            location = _analyse_location(m15p, tick_size=tick_size)
        except Exception as exc:
            logger.warning("Location analysis failed for %s: %r — continuing without it",
                           self.cfg["symbol"], exc)
            return None

        self._location_bar_time = bar_time
        self._location_cache = location
        return location

    def _get_sltp_reasoner(self):
        """Lazy initializer for the DynamicSLTPReasoner engine."""
        if self._sltp_reasoner is None and _DYNAMIC_SLTP_AVAILABLE:
            try:
                provider = _get_llm_provider(
                    provider_name=self.cfg.get("llm_provider", "embedded"),
                    base_url=self.cfg.get("local_llm_url", "http://localhost:1234/v1"),
                    model=self.cfg.get("local_llm_model", ""),
                    timeout=float(self.cfg.get("local_llm_timeout", 3.0)),
                )
                self._sltp_reasoner = _DynamicSLTPReasoner(provider=provider, config=self.cfg)
                logger.info("[%s] Dynamic AI SL/TP Reasoner initialized (provider=%s)",
                            self.cfg.get("symbol", ""), self.cfg.get("llm_provider", "embedded"))
            except Exception as exc:
                logger.warning("[%s] Failed to initialize DynamicSLTPReasoner: %r",
                               self.cfg.get("symbol", ""), exc)
                self._sltp_reasoner = None
        return self._sltp_reasoner

    # --------------------------------------------------------------
    def get_status(self) -> dict:
        """Thread-safe snapshot of the bot's current state, for GUIs/monitors."""
        with self._status_lock:
            return dict(self.status)

    def _update_status(self, **kwargs) -> None:
        with self._status_lock:
            self.status.update(kwargs)

    def _record_trade_result(self, pnl: float) -> None:
        with self._status_lock:
            self.status["closed_trades"] += 1
            self.status["total_pnl"] += pnl
            if pnl > 1e-9:
                self.status["wins"] += 1
            elif pnl < -1e-9:
                self.status["losses"] += 1
            else:
                self.status["breakeven_trades"] += 1
            self.status["best_trade"] = max(self.status["best_trade"], pnl)
            self.status["worst_trade"] = min(self.status["worst_trade"], pnl)
            
        # 💾 SAVE TO DATABASE
        try:
            import sqlite3
            from datetime import datetime
            db_path = _get_sqlite_db_path()
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS trades
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                             timestamp TEXT, 
                             symbol TEXT, 
                             pnl REAL,
                             balance REAL)''')
            cursor.execute("INSERT INTO trades (timestamp, symbol, pnl, balance) VALUES (?, ?, ?, ?)",
                         (datetime.now().isoformat(), self.cfg["symbol"], pnl, self.status.get("balance", 0.0)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to save trade to DB: %s", e)
            
        with self._status_lock:
            decided = self.status["wins"] + self.status["losses"]
            self.status["win_rate"] = (self.status["wins"] / decided * 100.0) if decided else 0.0

    # --------------------------------------------------------------
    def connect(self) -> bool:
        # FORCE EMPTY kwargs to force connection to the currently running terminal
        kwargs   = {"timeout": 60_000}
        
        logger.info("Initialising MT5...")
        if not mt5.initialize(**kwargs):
            logger.error("MT5 init failed: %s", mt5.last_error())
            return False
            
        acc = mt5.account_info()
        # _env_login() returns None (not 0) when unset, so normalise to 0 here.
        target_account = self.cfg.get("account") or 0
        
        if acc is not None and (target_account <= 0 or acc.login == target_account):
            logger.info("Already logged in to terminal account %s. Bypassing login step.", acc.login)
            self.cfg["account"] = acc.login
            ok = True
        elif target_account <= 0:
            logger.error(
                "MT5 login failed: no MT5_LOGIN configured and terminal is not logged into an active account. "
                "Please configure MT5_LOGIN/MT5_PASSWORD/MT5_SERVER in .env or log into MetaTrader 5 manually."
            )
            ok = False
        elif not self.cfg.get("password") or not self.cfg.get("server"):
            # Never call mt5.login() with a None credential: it produces a
            # confusing retcode instead of a clear configuration error.
            logger.error(
                "MT5 login failed: MT5_LOGIN is set (%s) but MT5_PASSWORD/MT5_SERVER are not. "
                "Refusing to attempt login with incomplete credentials.",
                target_account,
            )
            ok = False
        else:
            logger.info("Logging into %s...", target_account)
            ok = mt5.login(
                target_account,
                password=self.cfg["password"],
                server=self.cfg["server"],
            )
            
        if ok:
            acc = mt5.account_info()
            self.start_balance = acc.balance if acc else 0.0
            self.daily_start   = acc.equity  if acc else 0.0
            self.daily_date    = date.today()
            logger.info("Connected: balance=%.2f equity=%.2f",
                        self.start_balance, self.daily_start)
            self._update_status(connected=True, balance=self.start_balance,
                                 equity=self.daily_start, start_equity=self.daily_start)
        else:
            logger.error("MT5 login failed: %s", mt5.last_error())
            self._update_status(connected=False)
        return ok

    def _ensure_connected(self) -> bool:
        """Detect a dropped MT5 session/terminal and reconnect with backoff.
        Returns True once (re)connected, False if it should keep retrying."""
        term = mt5.terminal_info()
        acc  = mt5.account_info()
        if term is not None and acc is not None:
            self._reconnect_backoff = 5
            return True
        logger.warning("MT5 session appears down – attempting reconnect…")
        mt5.shutdown()
        if self.connect():
            self._reconnect_backoff = 5
            return True
        logger.warning("Reconnect failed, retrying in %ss", self._reconnect_backoff)
        time.sleep(self._reconnect_backoff)
        self._reconnect_backoff = min(self._reconnect_backoff * 2, 120)
        return False

    # --------------------------------------------------------------
    def _daily_dd_ok(self, equity: float) -> bool:
        today = date.today()
        if self.daily_date != today:
            self.daily_date  = today
            self.daily_start = equity
            self._dd_positions_closed = False
        if self.daily_start > 0:
            dd = (self.daily_start - equity) / self.daily_start
            if dd >= self.cfg["max_daily_dd"]:
                logger.warning("Daily DD limit hit (%.1f%%) – no new trades today",
                               dd * 100)
                if (self.cfg.get("equity_protect_close_all", True) and
                        not self._dd_positions_closed):
                    self._close_all("daily DD limit")
                    self._dd_positions_closed = True
                    self.notifier.send(
                        f"🛡️ Daily DD {dd*100:.1f}% hit – all positions closed.")
                return False
        return True

    # --------------------------------------------------------------
    def _get_data(self) -> tuple:
        sym = self.cfg["symbol"]
        n   = self.cfg["n_candles"]
        m15 = _fetch(sym, self.cfg["timeframe"], n)
        h1  = _fetch(sym, self.cfg.get("h1_tf", getattr(mt5, "TIMEFRAME_H1", 16385)), 300)
        h4  = _fetch(sym, self.cfg.get("h4_tf", getattr(mt5, "TIMEFRAME_H4", 16388)), 300)
        return m15, h1, h4

    def _prepared_m15(self, m15: pd.DataFrame) -> pd.DataFrame:
        """Return cached indicators unless a new M15 bar has closed."""
        bar_time = m15.index[-1] if not m15.empty else None
        if bar_time is not None and bar_time == self._m15_bar_time and self._m15p_cache is not None:
            return self._m15p_cache
        m15p = _prepare(m15, self.cfg)
        self._m15_bar_time  = bar_time
        self._m15p_cache    = m15p
        return m15p

    def _prepared_h1(self, h1: pd.DataFrame) -> Optional[pd.DataFrame]:
        if h1.empty:
            return None
        bar_time = h1.index[-1]
        if bar_time == self._h1_bar_time and self._h1p_cache is not None:
            return self._h1p_cache
        h1p = _prepare(h1, self.cfg)
        self._h1_bar_time = bar_time
        self._h1p_cache   = h1p
        return h1p

    # --------------------------------------------------------------
    def _get_signal(self, m15p, h1p, h4p) -> tuple:
        """Return (direction, score, ai_conf, reason).  direction 0 = no trade.

        The AI model is always updated/predicted (even off-session or in a
        low-ADX regime) so it keeps learning; the ADX/session gates only
        block *entries*, never training.
        """
        if m15p.empty or len(m15p) < 50:
            return 0, 0.0, 0.5, "no data"

        # AI signal – trained & predicted every new bar
        if self.ai is None:
            self.ai = AIEnsemble(self.cfg["symbol"], type("C", (), {
                "AI_CONFIDENCE_THRESHOLD": self.cfg["ai_confidence_threshold"],
                "AI_TRAIN_BARS":           self.cfg["ai_train_bars"],
                "AI_RETRAIN_BARS":         self.cfg["ai_retrain_bars"],
                # Model-validation + calibration knobs. These MUST be present
                # here: AIEnsemble.train() reads them outside any try/except,
                # so a missing key raises AttributeError straight into the
                # main loop. Keep this in sync with CONFIG and core/settings.py.
                "AI_VALIDATION_FRACTION":  self.cfg["ai_validation_fraction"],
                "AI_MIN_OOS_ACCURACY":     self.cfg["ai_min_oos_accuracy"],
                "AI_MIN_OOS_AUC":          self.cfg["ai_min_oos_auc"],
                "AI_MIN_OOS_EDGE":         self.cfg["ai_min_oos_edge"],
                "AI_PROMOTE_TOLERANCE":    self.cfg["ai_promote_tolerance"],
                "AI_CALIBRATE":            self.cfg["ai_calibrate"],
                "AI_MIN_EDGE_FOR_SIZE_UP": self.cfg["ai_min_edge_for_size_up"],
            })())
        self.ai.update(m15p)
        ai_sig       = self.ai.predict(m15p)
        ai_direction = ai_sig.direction
        ai_conf      = ai_sig.confidence

        # Score both directions (confluence, independent of gates)
        best_dir, best_score = 0, 0.0
        for d in [1, -1]:
            score = _score_direction(ai_direction, ai_conf, m15p, h1p, h4p,
                                     d, self.cfg)
            if score > best_score:
                best_score, best_dir = score, d

        # Entry gates (training already done above)
        adx_val = float(m15p.iloc[-1].get("adx", 0.0))
        if adx_val < self.cfg.get("adx_min", 20):
            return 0, best_score, ai_conf, f"low ADX ({adx_val:.1f})"

        if not _session_ok(self.cfg):
            return 0, best_score, ai_conf, "off-session"

        if best_score < self.cfg.get("min_signal_score", 55):
            return 0, best_score, ai_conf, f"score {best_score:.0f} < threshold"

        # --- Gold (XAUUSD) specific volatility and macro trend protection ---
        sym_upper = str(self.cfg.get("symbol", "")).upper()
        is_gold = "XAU" in sym_upper or "GOLD" in sym_upper
        if is_gold:
            # 1. Macro Trend Veto: Never scalp counter to established H4 trend
            if h4p is not None and not h4p.empty and len(h4p) >= 20:
                h4_opp = _trend_score(h4p, -best_dir)
                if h4_opp >= 0.65:
                    logger.info("[%s] Gold macro trend VETO: H4 opposes %s (score %.2f >= 0.65)",
                                self.cfg["symbol"], "BUY" if best_dir == 1 else "SELL", h4_opp)
                    return 0, best_score, ai_conf, f"gold macro veto: H4 opposes {'BUY' if best_dir == 1 else 'SELL'}"

            # 2. Gold High-Confluence Minimum Score: Gold demands minimum 58 confluence
            gold_min_score = float(self.cfg.get("gold_min_signal_score", 58.0))
            if best_score < gold_min_score:
                return 0, best_score, ai_conf, f"gold score {best_score:.0f} < threshold ({gold_min_score:.0f})"

            # 3. Gold AI Confidence Gate: 0.50 is pure noise; Gold requires >= 0.55
            gold_min_conf = float(self.cfg.get("gold_min_ai_conf", 0.55))
            if ai_conf < gold_min_conf:
                return 0, best_score, ai_conf, f"gold AI conf {ai_conf:.2f} < threshold ({gold_min_conf:.2f})"

            # 4. Gold Spread Gate: Prevent entries during wide spread / roll-over
            atr_val = float(m15p.iloc[-1].get("atr", 0.0))
            if atr_val > 0:
                sym_info = mt5.symbol_info(self.cfg["symbol"])
                if sym_info:
                    spread_pts = getattr(sym_info, "spread", 0) or 0
                    point_val = getattr(sym_info, "point", 0.01) or 0.01
                    spread_price = spread_pts * point_val
                    max_ratio = float(self.cfg.get("max_spread_atr_ratio", 0.18))
                    if spread_price / atr_val > max_ratio:
                        return 0, best_score, ai_conf, f"gold spread ({spread_price:.2f}) > {max_ratio*100:.0f}% ATR"

        # --- market location gate -------------------------------------
        # The scoring above answers "what". This answers "where", and can only
        # ever *remove* a trade, never introduce one. It runs after the score
        # gate so it never costs anything on a signal that was already weak.
        if self.cfg.get("location_gate", True):
            location = self._location_for(m15p)
            if location is None:
                # No location available. Deliberately NOT a veto: this layer is
                # an enhancement on top of a working signal stack, and refusing
                # to trade whenever it cannot form an opinion would turn a
                # cold start or a missing dependency into a silent kill switch.
                # The existing ADX/session/score gates remain in force.
                logger.debug("[%s] Location unavailable — trading on indicator gates only",
                             self.cfg["symbol"])
            else:
                self._publish_location_status(location)
                if location.is_valid:
                    gate = location.entry_gate(best_dir)
                    for caution in gate.cautions:
                        logger.info("[%s] Location caution: %s", self.cfg["symbol"], caution)
                    if not gate.allowed:
                        logger.info("[%s] Location VETO: %s", self.cfg["symbol"], gate.reason)
                        return 0, best_score, ai_conf, f"location veto: {gate.reason}"
                else:
                    logger.debug("[%s] Location UNDEFINED (no confirmed range) — not vetoing",
                                 self.cfg["symbol"])

        return best_dir, best_score, ai_conf, "signal"

    # --------------------------------------------------------------
    def _publish_location_status(self, location) -> None:
        """Expose location on the status dict for the dashboard/GUI.

        Purely additive: every key is new, and nothing existing is overwritten,
        so a consumer that ignores location keeps working unchanged.

        Wrapped defensively for the same reason as ``_location_for`` — this is
        a cosmetic feature, and it must never be the thing that raises into the
        main loop. On any problem the status is simply left untouched.
        """
        try:
            payload = {
                "loc_bias": location.bias.value,
                "loc_position": location.range_position,
                "loc_range_low": location.range_low,
                "loc_range_high": location.range_high,
                "loc_structure": location.structure.bias.value,
                "loc_confidence": location.confidence,
            }
            res = location.nearest_resistance()
            sup = location.nearest_support()
            payload["loc_resistance"] = res.price if res else None
            payload["loc_support"] = sup.price if sup else None
            payload["loc_below_range"] = location.below_range
            payload["loc_above_range"] = location.above_range
            self._update_status(**payload)
        except Exception as exc:
            logger.debug("Location status publication skipped: %r", exc)

    # --------------------------------------------------------------
    def _open_positions(self):
        return mt5.positions_get(symbol=self.cfg["symbol"]) or []

    def _count_positions(self) -> int:
        bot_magic = self.cfg.get("magic_number", 888999)
        return len([p for p in self._open_positions() if getattr(p, "magic", 0) == bot_magic])

    def _trades_today_ok(self) -> bool:
        today = date.today()
        if self.trade_date != today:
            self.trade_date  = today
            self.trades_today = 0
        limit = self.cfg.get("max_trades_per_day", 60 if _IS_AGGRESSIVE else 15)
        return self.trades_today < limit

    # --------------------------------------------------------------
    def _margin_ok(self, price: float, volume: float, equity: float) -> bool:
        sym = self.cfg["symbol"]
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, sym, volume, price)
        if margin is None:
            # API failure — fail safe: block the order rather than silently pass
            logger.warning(
                "Margin guard: order_calc_margin(%s, vol=%.2f, price=%.5f) returned None "
                "(MT5 error: %s) – blocking order",
                sym, volume, price,
                mt5.last_error() if hasattr(mt5, "last_error") else "N/A",
            )
            return False
        margin = float(margin)
        acc = mt5.account_info()
        used      = float(acc.margin)       if acc else 0.0
        free      = float(acc.margin_free)  if acc else 0.0
        limit     = equity * self.cfg.get("max_margin_usage", 0.5)

        # Guard 1 – configured margin-usage ceiling
        if margin + used > limit:
            logger.warning(
                "Margin guard: need=%.2f + used=%.2f > limit=%.2f (%.0f%% of equity=%.2f)",
                margin, used, limit,
                self.cfg.get("max_margin_usage", 0.5) * 100, equity,
            )
            return False

        # Guard 2 – broker's actual free margin (prevents retcode=10026)
        if margin > free:
            logger.warning(
                "Margin guard: required=%.2f > broker free_margin=%.2f (vol=%.2f, price=%.5f) "
                "– would cause NO_MONEY (10026)",
                margin, free, volume, price,
            )
            return False

        return True

    def _spread_ok(self, info, tick, atr: float = 0.0) -> bool:
        """Spread guard with both fixed-point threshold and dynamic ATR-relative
        ratio (vital for scalping to prevent entering when spread friction is high)."""
        spread_price = tick.ask - tick.bid
        max_spread = self.cfg.get("max_spread_points")
        if max_spread:
            spread_points = spread_price / info.point if info.point else 0
            if spread_points > max_spread:
                logger.info("Spread guard: %.1f pts > max %.1f – skipping entry",
                            spread_points, max_spread)
                return False

        # Relative spread-to-ATR guard for scalping (prevents trading when friction is too high)
        max_spread_atr_ratio = self.cfg.get("max_spread_atr_ratio", 0.25 if _IS_AGGRESSIVE else 0.40)
        if atr > 0 and (spread_price / atr) > max_spread_atr_ratio:
            logger.info("Scalp Spread guard: spread (%.5f) is %.1f%% of ATR (%.5f) > max %.1f%% – skipping entry",
                        spread_price, (spread_price / atr) * 100.0, atr, max_spread_atr_ratio * 100.0)
            return False
        return True

    def _get_filling_type(self, symbol: str) -> int:
        """Resolve a supported filling type for the broker symbol."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return getattr(mt5, "ORDER_FILLING_IOC", 1)
        filling_mode = getattr(info, "filling_mode", 0)
        # Bit 1 (value 2) = SYMBOL_FILLING_IOC
        if filling_mode & 2:
            return getattr(mt5, "ORDER_FILLING_IOC", 1)
        # Bit 0 (value 1) = SYMBOL_FILLING_FOK
        elif filling_mode & 1:
            return getattr(mt5, "ORDER_FILLING_FOK", 0)
        return getattr(mt5, "ORDER_FILLING_RETURN", 2)

    # --------------------------------------------------------------
    def _send_order(self, direction: int, equity: float, m15p: pd.DataFrame, ai_conf: float = 0.0) -> bool:
        """`m15p` is the already-prepared frame from the main loop —
        avoids an extra MT5 fetch + full indicator recompute per entry."""
        sym  = self.cfg["symbol"]
        info = mt5.symbol_info(sym)
        if info is None:
            logger.error("symbol_info(%s) failed", sym)
            return False
        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            return False

        last = m15p.iloc[-1]
        atr  = float(last.get("atr", info.point * 100))
        adx  = float(last.get("adx", 0.0))

        if not self._spread_ok(info, tick, atr):
            return False

        price = tick.ask if direction == 1 else tick.bid

        # --- stop placement -------------------------------------------
        # Default remains the ATR multiple this bot has always used. When the
        # location layer can name the swing that invalidates the idea, that
        # price is used instead: it is where the market itself says the thesis
        # is wrong, which is a far better place than an arbitrary multiple.
        #
        # `structural_stop` returns None when it cannot place a defensible
        # stop. That falls back to the ATR multiple rather than blocking the
        # trade — but a None is never silently turned into a zero-distance or
        # default stop.
        sl_dist = self.cfg["sl_atr_mult"] * atr
        stop_source = "atr"
        stop_note = ""
        sltp_done = False

        # --- DYNAMIC AI REASONING FOR ENTRY SL/TP ---
        if self.cfg.get("dynamic_sltp_enabled", True) and self.cfg.get("location_stop", True) and _DYNAMIC_SLTP_AVAILABLE and atr > 0:
            reasoner = self._get_sltp_reasoner()
            if reasoner is not None:
                location = self._location_for(m15p) if _LOCATION_AVAILABLE else None
                plan = reasoner.calculate_entry_sltp(
                    sym, direction, price, m15p, atr,
                    location=location, symbol_info=info
                )
                if plan is not None and plan.valid:
                    sl = plan.sl_price
                    tp = plan.tp_price
                    sl_dist = plan.sl_distance
                    stop_source = plan.sl_source
                    stop_note = plan.reasoning
                    sltp_done = True
                    logger.info("[%s] AI Dynamic SL/TP: SL=%.5f TP=%.5f (R:R=%.2f) | %s",
                                sym, sl, tp, plan.rr_ratio, plan.reasoning)

        if not sltp_done:
            if self.cfg.get("location_stop", True) and atr > 0:
                location = self._location_for(m15p)
                if location is not None and location.is_valid:
                    plan = location.structural_stop(
                        price,
                        direction,
                        min_atr_mult=self.cfg.get("location_stop_min_atr", 0.75),
                        max_atr_mult=self.cfg.get("location_stop_max_atr", 3.0),
                        tick_size=_valid_tick(getattr(info, "trade_tick_size", None)
                                              or getattr(info, "point", None)),
                    )
                    if plan is not None:
                        stop_source = "structure" if plan.structural else "volatility"
                        stop_note = plan.reason
                        sl = plan.price
                        sl_dist = abs(price - sl)
                    # plan is None -> keep the ATR stop below

            if direction == 1:
                if stop_source == "atr":
                    sl = price - sl_dist
                tp = price + sl_dist * self.cfg["tp_rr_ratio"]
            else:
                if stop_source == "atr":
                    sl = price + sl_dist
                tp = price - sl_dist * self.cfg["tp_rr_ratio"]

            # --- take profit: prefer resting liquidity in the trade's favour ---
            if self.cfg.get("location_target", True):
                location = self._location_for(m15p)
                if location is not None and location.is_valid and location.atr:
                    pool = location.target_above() if direction == 1 else location.target_below()
                    if pool is not None:
                        candidate = pool.price - direction * location.atr * 0.10
                        rr_from_pool = abs(candidate - price) / max(sl_dist, 1e-10)
                        if (candidate - price) * direction > 0 and rr_from_pool >= self.cfg["tp_rr_ratio"]:
                            tp = candidate
                            stop_note = (stop_note + "; " if stop_note else "") + \
                                        f"TP at {pool.kind.value} liquidity pool {pool.price:.5f}"

        if stop_note:
            logger.info("[%s] Stop via %s: %s", self.cfg["symbol"], stop_source, stop_note)

        min_d = info.trade_stops_level * info.point
        if min_d > 0:
            if abs(price - sl) < min_d:
                sl = price - min_d if direction == 1 else price + min_d
            if abs(tp - price) < min_d:
                tp = price + min_d if direction == 1 else price - min_d

        volume = _size_lot(
            equity, abs(price - sl),
            info.trade_tick_size, info.trade_tick_value,
            info.volume_min, info.volume_max, info.volume_step,
            self.cfg, adx=adx, ai_conf=ai_conf,
        )

        if not self._margin_ok(price, volume, equity):
            return False

        digits = getattr(info, "digits", 2) or 2
        price = round(float(price), digits)
        sl = round(float(sl), digits)
        tp = round(float(tp), digits)

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       sym,
            "volume":       float(volume),
            "type":         mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    self.cfg.get("deviation", 20),
            "magic":        self.cfg.get("magic_number", 888999),
            "comment":      f"AI {'BUY' if direction == 1 else 'SELL'}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(sym),
        }
        if self.cfg.get("dry_run"):
            logger.info("[DRY-RUN] %s vol=%.2f sl=%.5f tp=%.5f",
                        "BUY" if direction == 1 else "SELL", volume, sl, tp)
            return True

        retries = self.cfg.get("max_slippage_retries", 3)
        for attempt in range(retries):
            result = mt5.order_send(req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.trades_today += 1
                msg = (f"✅ {'BUY' if direction==1 else 'SELL'} {sym} "
                       f"vol={volume:.2f} @ {price:.5f}  SL={sl:.5f}  TP={tp:.5f}")
                logger.info(msg)
                self.notifier.send(msg)
                return True
            if result and result.retcode in (mt5.TRADE_RETCODE_REQUOTE,
                                             mt5.TRADE_RETCODE_PRICE_CHANGED,
                                             mt5.TRADE_RETCODE_PRICE_OFF):
                tick = mt5.symbol_info_tick(sym)
                if tick:
                    price = round(float(tick.ask if direction == 1 else tick.bid), digits)
                    req["price"] = price
                time.sleep(0.3)
                continue

            # Fallback filling modes on invalid fill or None
            invalid_fill_code = getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030)
            if (result and result.retcode == invalid_fill_code) or (result is None and attempt == 0):
                fallback_success = False
                for alt_fill in (getattr(mt5, "ORDER_FILLING_RETURN", 2),
                                 getattr(mt5, "ORDER_FILLING_FOK", 0),
                                 getattr(mt5, "ORDER_FILLING_IOC", 1)):
                    if alt_fill != req["type_filling"]:
                        req["type_filling"] = alt_fill
                        alt_res = mt5.order_send(req)
                        if alt_res and alt_res.retcode == mt5.TRADE_RETCODE_DONE:
                            self.trades_today += 1
                            msg = (f"✅ {'BUY' if direction==1 else 'SELL'} {sym} "
                                   f"vol={volume:.2f} @ {price:.5f}  SL={sl:.5f}  TP={tp:.5f}")
                            logger.info(msg)
                            self.notifier.send(msg)
                            return True

            retcode_str = result.retcode if result else "None"
            last_err = mt5.last_error() if hasattr(mt5, "last_error") else "N/A"
            NO_MONEY = getattr(mt5, "TRADE_RETCODE_NO_MONEY", 10026)
            if result and result.retcode == NO_MONEY:
                acc = mt5.account_info()
                free = float(acc.margin_free) if acc else -1.0
                logger.error(
                    "Order failed: NO_MONEY (10026) – vol=%.2f price=%.5f free_margin=%.2f "
                    "equity=%.2f mt5_error=%s  "
                    "(Hint: _margin_ok should have blocked this — check order_calc_margin availability)",
                    volume, price, free, equity, last_err,
                )
            else:
                logger.error("Order failed: retcode=%s mt5_error=%s",
                             retcode_str, last_err)
            break
        return False

    # --------------------------------------------------------------
    def _close_position(self, pos, reason: str, pnl: Optional[float] = None) -> bool:
        calc_pnl = float(getattr(pos, "profit", 0.0)) + float(getattr(pos, "swap", 0.0)) if pnl is None else float(pnl)
        if self.cfg.get("dry_run"):
            logger.info("[DRY-RUN] CLOSE ticket=%s (%s) pnl=%+.2f reason=%s",
                        pos.ticket, pos.symbol, calc_pnl, reason)
            self._peak_profit.pop(pos.ticket, None)
            self._partial_stages.pop(pos.ticket, None)
            self._record_trade_result(calc_pnl)
            return True
        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        tick   = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False
        price = tick.bid if is_buy else tick.ask
        info = mt5.symbol_info(pos.symbol)
        digits = getattr(info, "digits", 2) or 2
        price = round(float(price), digits)

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       float(pos.volume),
            "type":         mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position":     pos.ticket,
            "price":        price,
            "deviation":    self.cfg.get("deviation", 20),
            "magic":        getattr(pos, "magic", self.cfg.get("magic_number", 888999)),
            "comment":      "close:" + reason[:24],
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(pos.symbol),
        }
        for attempt in range(self.cfg.get("max_slippage_retries", 3)):
            result = mt5.order_send(req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                pnl = pos.profit + pos.swap
                msg = (f"🔒 CLOSE {pos.symbol} ticket={pos.ticket} "
                       f"pnl={pnl:+.2f} reason={reason}")
                logger.info(msg)
                self.notifier.send(msg)
                self._peak_profit.pop(pos.ticket, None)
                self._partial_stages.pop(pos.ticket, None)
                self._record_trade_result(pnl)
                return True
            if result and result.retcode in (mt5.TRADE_RETCODE_REQUOTE,
                                             mt5.TRADE_RETCODE_PRICE_CHANGED,
                                             mt5.TRADE_RETCODE_PRICE_OFF):
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick:
                    price = round(float(tick.bid if is_buy else tick.ask), digits)
                    req["price"] = price
                time.sleep(0.3)
                continue

            # Fallback filling modes on invalid fill or None
            invalid_fill_code = getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030)
            if (result and result.retcode == invalid_fill_code) or (result is None and attempt == 0):
                fallback_success = False
                for alt_fill in (getattr(mt5, "ORDER_FILLING_RETURN", 2),
                                 getattr(mt5, "ORDER_FILLING_FOK", 0),
                                 getattr(mt5, "ORDER_FILLING_IOC", 1)):
                    if alt_fill != req["type_filling"]:
                        req["type_filling"] = alt_fill
                        alt_res = mt5.order_send(req)
                        if alt_res and alt_res.retcode == mt5.TRADE_RETCODE_DONE:
                            pnl = pos.profit + pos.swap
                            msg = (f"🔒 CLOSE {pos.symbol} ticket={pos.ticket} "
                                   f"pnl={pnl:+.2f} reason={reason}")
                            logger.info(msg)
                            self.notifier.send(msg)
                            self._peak_profit.pop(pos.ticket, None)
                            self._partial_stages.pop(pos.ticket, None)
                            self._record_trade_result(pnl)
                            return True

            retcode_str = result.retcode if result else "None"
            last_err = mt5.last_error() if hasattr(mt5, "last_error") else "N/A"
            logger.error("Close failed ticket=%s retcode=%s mt5_error=%s reason=%s",
                         pos.ticket, retcode_str, last_err, reason)
            break
        return False

    def _close_all(self, reason: str) -> None:
        for pos in self._open_positions():
            self._close_position(pos, reason)

    # --------------------------------------------------------------
    def _modify_sltp(self, pos, new_sl: float, new_tp: float) -> bool:
        if self.cfg.get("dry_run"):
            logger.info("[DRY-RUN] Modify SLTP ticket=%s → SL:%.5f TP:%.5f", pos.ticket, new_sl, new_tp)
            return True
        info = mt5.symbol_info(pos.symbol)
        digits = getattr(info, "digits", 2) or 2
        sl_val = round(float(new_sl), digits) if new_sl > 0 else 0.0
        tp_val = round(float(new_tp), digits) if new_tp > 0 else 0.0

        stops_level = getattr(info, "trade_stops_level", 0) or 0
        point = getattr(info, "point", 0.01) or 0.01
        min_distance = stops_level * point
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick and min_distance > 0:
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            cur_price = tick.bid if is_buy else tick.ask
            if sl_val > 0 and abs(cur_price - sl_val) < min_distance:
                sl_val = round(cur_price - min_distance if is_buy else cur_price + min_distance, digits)
            if tp_val > 0 and abs(cur_price - tp_val) < min_distance:
                tp_val = round(cur_price + min_distance if is_buy else cur_price - min_distance, digits)

        req = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": pos.ticket,
            "sl":       sl_val,
            "tp":       tp_val,
            "magic":    getattr(pos, "magic", self.cfg.get("magic_number", 888999)),
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Modified SLTP ticket=%s → SL:%.5f TP:%.5f", pos.ticket, sl_val, tp_val)
            return True
        retcode_str = result.retcode if result else "None"
        last_err = mt5.last_error() if hasattr(mt5, "last_error") else "N/A"
        logger.error("Modify SLTP failed ticket=%s retcode=%s mt5_error=%s (SL=%.5f TP=%.5f)",
                     pos.ticket, retcode_str, last_err, sl_val, tp_val)
        return False

    def _modify_sl(self, pos, new_sl: float) -> bool:
        if self.cfg.get("dry_run"):
            logger.info("[DRY-RUN] Modify SL ticket=%s → %.5f", pos.ticket, new_sl)
            return True
        info = mt5.symbol_info(pos.symbol)
        digits = getattr(info, "digits", 2) or 2
        sl_val = round(float(new_sl), digits)

        req = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": pos.ticket,
            "sl":       sl_val,
            "tp":       round(float(pos.tp), digits),
            "magic":    getattr(pos, "magic", self.cfg.get("magic_number", 888999)),
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Modified SL ticket=%s → %.5f", pos.ticket, sl_val)
            return True
        retcode_str = result.retcode if result else "None"
        last_err = mt5.last_error() if hasattr(mt5, "last_error") else "N/A"
        logger.error("Modify SL failed ticket=%s retcode=%s mt5_error=%s (SL=%.5f)",
                     pos.ticket, retcode_str, last_err, sl_val)
        return False

    def _partial_close(self, pos, volume: float, stage: int) -> None:
        if self.cfg.get("dry_run"):
            logger.info("[DRY-RUN] Partial close ticket=%s vol=%.2f stage=%d",
                        pos.ticket, volume, stage)
            self._partial_stages.setdefault(pos.ticket, set()).add(stage)
            return
        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        tick   = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return
        price = tick.bid if is_buy else tick.ask
        info = mt5.symbol_info(pos.symbol)
        digits = getattr(info, "digits", 2) or 2
        price = round(float(price), digits)
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       float(volume),
            "type":         mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position":     pos.ticket,
            "price":        price,
            "deviation":    self.cfg.get("deviation", 20),
            "magic":        getattr(pos, "magic", self.cfg.get("magic_number", 888999)),
            "comment":      f"partial{stage}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_type(pos.symbol),
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self._partial_stages.setdefault(pos.ticket, set()).add(stage)
            logger.info("Partial close ticket=%s vol=%.2f stage=%d executed",
                        pos.ticket, volume, stage)
        else:
            retcode_str = result.retcode if result else "None"
            last_err = mt5.last_error() if hasattr(mt5, "last_error") else "N/A"
            logger.error("Partial close failed ticket=%s: retcode=%s mt5_error=%s",
                         pos.ticket, retcode_str, last_err)

    # --------------------------------------------------------------
    def _h1_trend_flipped(self, h1p, direction: int) -> bool:
        """True when H1 EMA50/EMA200 crossed against the position."""
        if not self.cfg.get("enable_trend_flip_close", True):
            return False
        if h1p is None or len(h1p) < 3:
            return False
        e50_p = float(h1p["ema50"].iloc[-2])
        e200_p = float(h1p["ema200"].iloc[-2])
        e50 = float(h1p["ema50"].iloc[-1])
        e200 = float(h1p["ema200"].iloc[-1])
        if any(math.isnan(x) for x in (e50_p, e200_p, e50, e200)):
            return False
        if direction == 1:   # long position – death cross is bad
            return e50_p > e200_p and e50 < e200
        return e50_p < e200_p and e50 > e200   # short – golden cross is bad

    def _position_age(self, pos) -> float:
        off = self.cfg.get("server_utc_offset_hours", 0) * 3600.0
        return (time.time() + off - float(pos.time)) / 3600.0

    # --------------------------------------------------------------
    def _manage_positions(self, m15p: pd.DataFrame, h1p=None,
                           tick=None, equity: float = 0.0) -> None:
        """Trailing + breakeven + partial close + flip/timeout closes, plus
        the POWER-UP profit/loss decision layer (hard %-equity stop,
        profit-lock giveback, losing-position timeout cut).

        Uses the LIVE tick (bid/ask) for the current price when available,
        falling back to the last closed bar's close otherwise. ATR (for
        stop distances) still comes from the closed-bar indicators, which
        is correct — only the "where is price right now" part should be
        live."""
        positions = self._open_positions()
        if not positions:
            return
        if m15p is None or m15p.empty:
            return
        last = m15p.iloc[-1]
        atr  = float(last.get("atr", 0.0))
        if atr <= 0:
            return

        # Prune tracking dicts for tickets that are no longer open, so they
        # don't grow unbounded over a long-running session.
        open_tickets = {pos.ticket for pos in positions}
        for _tracked in (self._peak_profit, self._partial_stages, self._manual_sltp_failed):
            for ticket in list(_tracked.keys()):
                if ticket not in open_tickets:
                    _tracked.pop(ticket, None)

        for pos in positions:
            direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            if tick is not None:
                close_price = tick.bid if direction == 1 else tick.ask
            else:
                close_price = float(last["close"])
            profit_move = ((close_price - pos.price_open) if direction == 1
                           else (pos.price_open - close_price))
            floating_pnl = pos.profit + pos.swap

            # --- MANUAL TRADE HANDLING ---
            bot_magic = self.cfg.get("magic_number", 888999)
            is_manual = (getattr(pos, "magic", 0) != bot_magic)
            if is_manual:
                if self.cfg.get("ai_protect_manual_trades", True):
                    if pos.sl == 0.0 or pos.tp == 0.0:
                        now = time.time()
                        last_attempt = self._manual_sltp_failed.get(pos.ticket, 0)
                        if now - last_attempt >= 30.0:  # Throttled retry on failure
                            sl_dist = self.cfg.get("manual_trade_sl_atr", 2.0) * atr
                            new_sl = pos.sl if pos.sl > 0 else (pos.price_open - sl_dist if direction == 1 else pos.price_open + sl_dist)
                            tp_rr = self.cfg.get("tp_rr_ratio", 2.0)
                            new_tp = pos.tp if pos.tp > 0 else (pos.price_open + (sl_dist * tp_rr) if direction == 1 else pos.price_open - (sl_dist * tp_rr))
                            if self._modify_sltp(pos, new_sl, new_tp):
                                logger.info("[%s] 🧠 Applied AI SL (%.5f) and TP (%.5f) to manual trade #%d",
                                            self.cfg["symbol"], new_sl, new_tp, pos.ticket)
                                self._manual_sltp_failed.pop(pos.ticket, None)
                            else:
                                self._manual_sltp_failed[pos.ticket] = now

                # Protect Only / Observe Only: Never close manual positions with bot strategy signals!
                # The manual trade is protected by the broker SL/TP set above.
                if not self.cfg.get("manage_manual_trades_full", False):
                    continue

            # --- INTELLIGENT AI EARLY EXIT (Don't wait for hard DD) ---
            last_dir = self.status.get("last_direction", 0)
            last_score = self.status.get("last_score", 0.0)
            
            # Scenario A: Moderate Opposite Signal (AI sees incoming reversal)
            if last_dir == -direction and last_score >= 45:
                if floating_pnl <= (equity * 0.002): # Let big runners run, but cut struggling trades
                    self._close_position(pos, "AI Early Exit (Incoming Reversal)", floating_pnl)
                    continue
                    
            # Scenario B: Momentum Exhaustion (AI lost faith in trend)
            if last_dir == direction and last_score < 35:
                if floating_pnl <= 0: # Cut losers before they hit hard DD
                    self._close_position(pos, "AI Early Exit (Momentum Dead)", floating_pnl)
                    continue

            # Scenario C: Scalp Rejection Exit (Lock in profit when opposite pinbar forms)
            if profit_move >= 0.35 * atr and floating_pnl > 0:
                opp_pinbar = bool(last.get("bear_pinbar", False)) if direction == 1 else bool(last.get("bull_pinbar", False))
                if opp_pinbar:
                    self._close_position(pos, "Scalp Rejection Exit (Opposite Pinbar)", floating_pnl)
                    continue
                    
            # --- TIME DECAY (Cut losers early) ---
            current_time = time.time()
            if getattr(pos, "time", 0) > 0 and (current_time - pos.time) > self.cfg.get("cut_losing_after_hours", 12) * 3600:
                if floating_pnl <= 0:
                    self._close_position(pos, "TIME_DECAY", floating_pnl)
                    continue

            # --- DYNAMIC AI SL/TP RATCHET (One-Way Market Reasoner) ---
            if self.cfg.get("dynamic_sltp_enabled", True) and _DYNAMIC_SLTP_AVAILABLE and floating_pnl > 0:
                reasoner = self._get_sltp_reasoner()
                if reasoner is not None:
                    loc = self._location_for(m15p) if _LOCATION_AVAILABLE else None
                    sym_info = mt5.symbol_info(pos.symbol)
                    adj = reasoner.evaluate_position_sltp(
                        pos, m15p, atr, location=loc, symbol_info=sym_info,
                        peak_profit=self._peak_profit.get(pos.ticket, 0.0)
                    )
                    if adj.new_sl is not None:
                        if direction == 1 and adj.new_sl > pos.sl + 1e-9 and adj.new_sl < close_price:
                            logger.info("[%s] AI Dynamic SL (%s): %.5f -> %.5f (%s)",
                                        pos.symbol, adj.action, pos.sl, adj.new_sl, adj.reasoning)
                            self._modify_sl(pos, adj.new_sl)
                        elif direction == -1 and (pos.sl == 0 or adj.new_sl < pos.sl - 1e-9) and adj.new_sl > close_price:
                            logger.info("[%s] AI Dynamic SL (%s): %.5f -> %.5f (%s)",
                                        pos.symbol, adj.action, pos.sl, adj.new_sl, adj.reasoning)
                            self._modify_sl(pos, adj.new_sl)

            # --- MARKET STRUCTURE TRAILING STOP ---
            if floating_pnl > 0 and self.cfg.get("structure_trailing", True):
                if direction == 1:  # BUY
                    recent_low = m15p['low'].iloc[-5:].min()
                    if recent_low > pos.sl and close_price > recent_low + (atr * 0.2):
                        new_sl = recent_low - (atr * 0.1)
                        if new_sl > pos.sl and new_sl < close_price:
                            self._modify_sl(pos, new_sl)
                else:  # SELL
                    recent_high = m15p['high'].iloc[-5:].max()
                    if pos.sl == 0 or recent_high < pos.sl:
                        if close_price < recent_high - (atr * 0.2):
                            new_sl = recent_high + (atr * 0.1)
                            if (pos.sl == 0 or new_sl < pos.sl) and new_sl > close_price:
                                self._modify_sl(pos, new_sl)

            # ---- POWER-UP (A): hard per-trade stop-loss, % of equity ---
            max_loss_pct = self.cfg.get("max_loss_per_trade_pct")
            if max_loss_pct and equity > 0 and floating_pnl <= -(equity * max_loss_pct):
                self._close_position(pos, "hard stop-loss (%.1f%% equity)" % (max_loss_pct * 100))
                continue

            # ---- POWER-UP (C): losing-position timeout cut -------------
            cut_after = self.cfg.get("cut_losing_after_hours")
            if cut_after and floating_pnl < 0 and self._position_age(pos) >= cut_after:
                self._close_position(pos, "losing timeout cut (%dh)" % cut_after)
                continue

            # ---- H1 trend-flip close ---------------------------------
            if self._h1_trend_flipped(h1p, direction):
                self._close_position(pos, "h1 trend flip")
                continue

            # ---- age timeout close (only if profitable) --------------
            if self.cfg.get("close_old_profitable", True):
                max_age = self.cfg.get("max_position_age_hours", 24)
                if self._position_age(pos) >= max_age:
                    if floating_pnl > 0:
                        self._close_position(pos, "age timeout")
                    continue

            # ---- POWER-UP (B): profit-lock / giveback close -------------
            peak = self._peak_profit.get(pos.ticket, 0.0)
            if profit_move > peak:
                peak = profit_move
                self._peak_profit[pos.ticket] = peak
            lock_activate = self.cfg.get("profit_lock_activate_atr", 1.2) * atr
            giveback_pct  = self.cfg.get("profit_giveback_pct", 0.35)
            if peak >= lock_activate and profit_move <= peak * (1.0 - giveback_pct):
                self._close_position(pos, "profit lock giveback")
                continue

            # ---- PROGRESSIVE BREAKEVEN & PROFIT LOCKING (Advances SL as profit grows) ----
            be_buffer = self.cfg.get("breakeven_buffer_atr", 0.15) * atr
            be_sl = (pos.price_open + be_buffer) if direction == 1 else (pos.price_open - be_buffer)
            if self.cfg.get("enable_breakeven", True):
                be_trigger = self.cfg.get("breakeven_trigger_atr", 0.4) * atr
                if profit_move >= be_trigger:
                    target_sl = be_sl

                    # Progressive profit locking: as profit increases, advance SL forward into profit
                    extra_profit = profit_move - be_trigger
                    lock_ratio = self.cfg.get("profit_lock_ratio", 0.50)
                    if extra_profit > 0 and lock_ratio > 0:
                        locked_gain = be_buffer + (extra_profit * lock_ratio)
                        progressive_sl = (pos.price_open + locked_gain) if direction == 1 else (pos.price_open - locked_gain)
                        target_sl = max(target_sl, progressive_sl) if direction == 1 else min(target_sl, progressive_sl)

                    # Ensure minimum distance from current price to respect broker stops level
                    info = mt5.symbol_info(pos.symbol)
                    min_dist = (info.trade_stops_level * info.point) if info and info.trade_stops_level else (atr * 0.1)
                    min_dist = max(min_dist, atr * 0.05)

                    if direction == 1:
                        target_sl = min(target_sl, close_price - min_dist)
                        if target_sl > pos.sl + 1e-9 and target_sl >= be_sl:
                            self._modify_sl(pos, target_sl)
                    else:
                        target_sl = max(target_sl, close_price + min_dist)
                        if (pos.sl == 0 or target_sl < pos.sl - 1e-9) and target_sl <= be_sl:
                            self._modify_sl(pos, target_sl)

            # ---- trailing stop (accelerating, never falls below breakeven)
            trail_atr  = self.cfg.get("trail_atr_mult", 1.5) * atr
            activate   = self.cfg.get("trail_activate_atr", 1.0) * atr
            if profit_move >= activate:
                if profit_move >= self.cfg.get("trail_tighten2_atr", 3.0) * atr:
                    trail_atr = self.cfg.get("trail_tighten2_mult", 0.6) * atr
                elif profit_move >= self.cfg.get("trail_tighten1_atr", 2.0) * atr:
                    trail_atr = self.cfg.get("trail_tighten1_mult", 1.0) * atr
                new_sl = (close_price - trail_atr if direction == 1
                          else close_price + trail_atr)

                # Floor at breakeven + buffer: once trailing activates, trade can NEVER turn into a loss
                if direction == 1:
                    new_sl = max(new_sl, be_sl)
                else:
                    new_sl = min(new_sl, be_sl)

                if ((direction == 1  and new_sl > pos.sl + 1e-9) or
                        (direction == -1 and (pos.sl == 0 or new_sl < pos.sl - 1e-9))):
                    self._modify_sl(pos, new_sl)

            # ---- multi-stage partial close ---------------------------
            if self.cfg.get("enable_partial_close", True):
                info = mt5.symbol_info(pos.symbol)
                if info is None:
                    continue
                step = info.volume_step or 0.01
                vmin = info.volume_min
                stages = self._partial_stages.setdefault(pos.ticket, set())

                if 1 not in stages:
                    trig = self.cfg.get("partial_close_trigger_atr", 1.5) * atr
                    if profit_move >= trig:
                        frac = self.cfg.get("partial_close_fraction", 0.5)
                        part = math.floor(pos.volume * frac / step) * step
                        if part >= vmin and pos.volume - part >= vmin:
                            self._partial_close(pos, part, 1)

                if 2 not in stages:
                    trig2 = self.cfg.get("partial_close2_trigger_atr", 3.0) * atr
                    if profit_move >= trig2:
                        frac2 = self.cfg.get("partial_close2_fraction", 0.25)
                        part2 = math.floor(pos.volume * frac2 / step) * step
                        if part2 >= vmin and pos.volume - part2 >= vmin:
                            self._partial_close(pos, part2, 2)

    # --------------------------------------------------------------

    def _close_profitable(self, reason="SECURE_PROFITS") -> None:
        positions = self._open_positions()
        if not positions:
            return
        bot_magic = self.cfg.get("magic_number", 888999)
        full_mgmt = self.cfg.get("manage_manual_trades_full", False)
        for p in positions:
            if not full_mgmt and getattr(p, "magic", 0) != bot_magic:
                continue
            if p.profit > 0:
                self._close_position(p, reason, p.profit)

    def _close_opposite_positions(self, direction: int, score: float) -> None:
        """Close positions running against a strong new signal."""
        if not self.cfg.get("close_on_opposite_signal", True):
            return
        threshold = self.cfg.get("opposite_signal_min_score", 65)
        if score < threshold:
            return
        bot_magic = self.cfg.get("magic_number", 888999)
        full_mgmt = self.cfg.get("manage_manual_trades_full", False)
        for pos in self._open_positions():
            if not full_mgmt and getattr(pos, "magic", 0) != bot_magic:
                continue
            pos_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            if pos_dir == -direction:
                self._close_position(pos, "opposite signal")

    # --------------------------------------------------------------
    def step(self) -> None:
        if getattr(self, "paused", False):
            return
            
        if not self._ensure_connected():
            return

        m15, h1, h4 = self._get_data()
        if m15.empty:
            return

        acc    = mt5.account_info()
        equity = acc.equity if acc else 0.0
        self._update_status(
            connected=True,
            balance=acc.balance if acc else 0.0,
            equity=equity,
            open_positions=self._count_positions(),
            trades_today=self.trades_today,
        )

        if not self._daily_dd_ok(equity):
            return

        current_bar = m15.index[-1] if not m15.empty else None

        # Indicators only recomputed when a new bar actually closed
        m15p = self._prepared_m15(m15)
        h1p  = self._prepared_h1(h1)

        # Manage open positions every cycle, but on the live tick
        tick = mt5.symbol_info_tick(self.cfg["symbol"])
        if tick is None:
            return
        self._manage_positions(m15p, h1p, tick=tick, equity=equity)

        # Entries / reversal closes only on new closed bar
        if current_bar != self.last_bar_time:
            self.last_bar_time = current_bar
            h4p = _prepare(h4, self.cfg) if not h4.empty else None
            direction, score, ai_conf, reason = self._get_signal(m15p, h1p, h4p)
            logger.info("[%s] Bar %s | direction=%+d score=%.1f ai_conf=%.3f (%s)",
                        self.cfg["symbol"], current_bar, direction, score, ai_conf, reason)
            self._update_status(
                last_bar=str(current_bar), last_direction=direction,
                last_score=score, last_ai_conf=ai_conf, last_signal=reason,
            )

            if direction != 0:
                self._close_opposite_positions(direction, score)

            if (direction != 0 and
                    self._count_positions() < self.cfg.get("max_open_pos", 3) and
                    self._trades_today_ok()):
                self._send_order(direction, equity, m15p, ai_conf)

    def run(self) -> None:
        if not self.connect():
            return
        self.running = True
        logger.info("=== AI Bot STARTED | symbol=%s | dry_run=%s ===",
                    self.cfg["symbol"], self.cfg.get("dry_run", False))
        if hasattr(self, 'notifier') and hasattr(self.notifier, 'send'):
            self.notifier.send(f"🤖 AI Bot started | {self.cfg['symbol']}")

        try:
            while self.running:
                try:
                    self.step()
                    import time
                    time.sleep(self.cfg.get("sleep_interval", 1.0))
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    logger.exception("Unhandled error in main loop: %s", exc)
                    import time
                    time.sleep(min(self.cfg.get("sleep_interval", 1.0) * 4, 5))
        except KeyboardInterrupt:
            logger.info("Bot stopping…")
        finally:
            import MetaTrader5 as mt5
            mt5.shutdown()
            logger.info("MT5 closed.")
            self._update_status(connected=False)
            if hasattr(self, 'notifier') and hasattr(self.notifier, 'send'):
                self.notifier.send("🛑 AI Bot stopped.")



# ---------------------------------------------------------------------------
# Entry point with FastAPI Dashboard
# ---------------------------------------------------------------------------
import uvicorn
from fastapi import FastAPI, Response, Request

def create_app(bots: Optional[List[Any]] = None) -> FastAPI:
    """Factory to create the FastAPI dashboard app, testable and decoupled."""
    if bots is None:
        bots = []

    app = FastAPI(title="AI Quant Platform Dashboard")
    
    @app.get("/")
    def serve_dashboard():
        try:
            with open("dashboard.html", "r", encoding="utf-8") as f:
                html = f.read()
            from fastapi import Response
            return Response(content=html, media_type="text/html")
        except FileNotFoundError:
            from fastapi import Response
            return Response(content="<h1>Dashboard HTML not found!</h1>", status_code=404)
            
    @app.get("/api/status")
    def api_status():
        if not bots:
            return {}
        acc_balance = next((b.status.get("balance", 0) for b in bots if b.status.get("balance", 0) > 0), 0)
        acc_equity = next((b.status.get("equity", 0) for b in bots if b.status.get("equity", 0) > 0), 0)
        wins = sum(b.status.get("wins", 0) for b in bots)
        losses = sum(b.status.get("losses", 0) for b in bots)
        total_trades = wins + losses + sum(b.status.get("breakeven_trades", 0) for b in bots)
        
        status_obj = {
            "balance": acc_balance,
            "equity": acc_equity,
            "sessionPnl": sum(b.status.get("total_pnl", 0.0) for b in bots),
            "winRate": (wins / total_trades * 100) if total_trades > 0 else 0.0,
            "wins": wins,
            "losses": losses,
            "maxWin": 0,
            "maxLoss": 0,
            "tradesToday": sum(b.status.get("trades_today", 0) for b in bots)
        }
        
        markets = []
        for b in bots:
            st = b.get_status()
            # `loc_*` keys are additive; dashboard.html reads the keys it knows
            # and ignores the rest, so this is safe for the existing UI.
            markets.append({
                "sym": b.cfg["symbol"],
                "regime": getattr(b, "regime", "UNKNOWN") if hasattr(b, "regime") else "UNKNOWN",
                "signal": "BUY" if st.get("last_direction", 0) == 1 else "SELL" if st.get("last_direction", 0) == -1 else "WAIT",
                "score": st.get("last_score", 0),
                "conf": st.get("last_ai_conf", 0),
                "pos": st.get("open_positions", 0),
                "pnl": 0,
                "unc": 0.1,
                "state": "PAUSED" if getattr(b, "paused", False) else "ACTIVE",
                "locBias": st.get("loc_bias"),
                "locPosition": st.get("loc_position"),
                "locRangeLow": st.get("loc_range_low"),
                "locRangeHigh": st.get("loc_range_high"),
                "locStructure": st.get("loc_structure"),
                "locSupport": st.get("loc_support"),
                "locResistance": st.get("loc_resistance"),
            })
            
        # Report the mode from the key that actually gates execution ("dry_run").
        # Previously read cfg.get("paper_trading"), a key that exists nowhere, so
        # the badge always showed LIVE regardless of the real trading state.
        mode = "PAPER" if bots[0].cfg.get("dry_run") else "LIVE"
        if any(getattr(b, "paused", False) for b in bots):
            mode = "PAUSED"
            
        return {"status": status_obj, "markets": markets, "mode": mode}

    @app.get("/api/positions")
    def api_positions():
        all_positions = []
        for b in bots:
            sym = b.cfg["symbol"]
            try:
                import MetaTrader5 as mt5
                pos_list = mt5.positions_get(symbol=sym) or []
                for p in pos_list:
                    age_hours = 0
                    is_manual = (p.magic == 0)
                    urg = 0.5
                    all_positions.append({
                        "id": p.ticket,
                        "sym": sym,
                        "type": "BUY" if p.type == 0 else "SELL",
                        "vol": p.volume,
                        "open": p.price_open,
                        "cur": p.price_current,
                        "sl": p.sl,
                        "tp": p.tp,
                        "pnl": p.profit,
                        "age": age_hours,
                        "urg": urg,
                        "regime": "UNKNOWN",
                        "manual": is_manual
                    })
            except Exception:
                pass
        return {"positions": all_positions}

    @app.get("/api/intelligence")
    def api_intelligence():
        data = []
        for b in bots:
            data.append({
                "sym": b.cfg["symbol"],
                "tf": "M15",
                "regime": "UNKNOWN",
                "conf": b.status.get("last_ai_conf", 0.0),
                "signal": "BUY" if b.status.get("last_direction", 0) == 1 else "SELL" if b.status.get("last_direction", 0) == -1 else "WAIT",
                "unc": 0.1,
                "pats": [],
                "sup": [], "res": [],
                "ema": {"m15":1, "h1":1, "h4":1, "d1":1},
                "scores": [b.status.get("last_score", 0), 20, 15, 10, 10]
            })
        return {"data": data}

    @app.get("/api/performance")
    def api_performance():
        return {"dailyPnl": [0]*30}

    @app.get("/api/logs")
    def api_logs():
        logs = []
        try:
            with open("trading_bot.log", "r") as logf:
                lines = logf.readlines()[-20:]
                for line in lines:
                    logs.append({"ts": "", "lvl": "INFO", "msg": line.strip()})
        except:
            pass
        return {"logs": logs}

    @app.get("/api/settings")
    def api_get_settings():
        cfg = bots[0].cfg if bots else CONFIG
        sym_list = ", ".join([b.cfg.get("symbol", "") for b in bots if b.cfg.get("symbol")]) if bots else cfg.get("symbol", "XAUUSDm")
        if not sym_list:
            sym_list = cfg.get("symbol", "XAUUSDm")

        tf_val = cfg.get("timeframe", 15)
        tf_map_rev = {
            1: "M1", 2: "M2", 3: "M3", 5: "M5", 10: "M10", 15: "M15", 30: "M30",
            16385: "H1", 16388: "H4", 16408: "D1",
        }
        tf_str = tf_map_rev.get(tf_val, "M15")

        risk_pct = round(float(cfg.get("risk_per_trade", 0.01)) * 100, 2)
        dd_pct = round(float(cfg.get("max_daily_dd", 0.05)) * 100, 2)
        margin_pct = round(float(cfg.get("max_margin_usage", 0.35)) * 100, 1)

        be_val = float(cfg.get("breakeven_trigger_atr", 0.0) or 0.0)

        res = {
            **cfg,
            "mt5_server": cfg.get("server", "") or "",
            "mt5_login": cfg.get("account", 0) or 0,
            "mt5_password": cfg.get("password", "") or "",
            "server_utc_offset_hours": cfg.get("server_utc_offset_hours", 2),
            "raw_symbols": sym_list,
            "trading_mode": "paper" if cfg.get("dry_run", True) else "live",
            "entry_timeframe": tf_str,
            "risk_per_trade": risk_pct,
            "max_daily_drawdown_pct": dd_pct,
            "max_open_positions": cfg.get("max_open_pos", 3),
            "max_margin_usage": margin_pct,
            "sl_atr_mult": cfg.get("sl_atr_mult", 1.8),
            "trail_atr_mult": cfg.get("trail_atr_mult", 1.0),
            "enable_breakeven": be_val > 0,
            "default_manual_policy": cfg.get("default_manual_policy", "observe_only"),
            "ai_enabled": cfg.get("ai_enabled", True),
            "ai_confidence_threshold": cfg.get("ai_confidence_threshold", 0.55),
            "min_signal_score": cfg.get("min_signal_score", 58),
            "adx_trending_min": cfg.get("adx_min", 18),
            "deviation_points": cfg.get("max_slippage", 30),
            "max_requote_retries": cfg.get("max_requote_retries", 5),
            "magic_number": cfg.get("magic_number", 888992),
            "telegram_bot_token": cfg.get("telegram_token", "") or cfg.get("telegram_bot_token", ""),
        }
        return res

    @app.post("/api/settings")
    async def api_post_settings(request: Request):
        new_cfg = await request.json()
        env_updates = {}

        # 1. Trading Mode
        if "trading_mode" in new_cfg:
            tm = str(new_cfg["trading_mode"]).lower().strip()
            if tm in ("paper", "live", "backtest"):
                is_dry = (tm != "live")
                CONFIG["dry_run"] = is_dry
                for b in bots:
                    b.cfg["dry_run"] = is_dry
                env_updates["TRADING_MODE"] = tm

        # 2. Risk per trade
        if "risk_per_trade" in new_cfg and new_cfg["risk_per_trade"] is not None:
            r = float(new_cfg["risk_per_trade"])
            if r > 0.05:
                r = r / 100.0
            CONFIG["risk_per_trade"] = r
            for b in bots:
                b.cfg["risk_per_trade"] = r
            env_updates["RISK_PER_TRADE"] = str(r)

        # 3. Max daily DD
        raw_dd = new_cfg.get("max_daily_drawdown_pct") if "max_daily_drawdown_pct" in new_cfg else new_cfg.get("max_daily_dd")
        if raw_dd is not None:
            dd = float(raw_dd)
            if dd > 0.05:
                dd = dd / 100.0
            CONFIG["max_daily_dd"] = dd
            for b in bots:
                b.cfg["max_daily_dd"] = dd
            env_updates["MAX_DAILY_DD"] = str(dd)

        # 4. Max open positions
        raw_pos = new_cfg.get("max_open_positions") if "max_open_positions" in new_cfg else new_cfg.get("max_open_pos")
        if raw_pos is not None:
            p = int(raw_pos)
            CONFIG["max_open_pos"] = p
            for b in bots:
                b.cfg["max_open_pos"] = p
            env_updates["MAX_OPEN_POS"] = str(p)

        # 5. Max margin usage
        if "max_margin_usage" in new_cfg and new_cfg["max_margin_usage"] is not None:
            m = float(new_cfg["max_margin_usage"])
            if m > 1.0:
                m = m / 100.0
            CONFIG["max_margin_usage"] = m
            for b in bots:
                b.cfg["max_margin_usage"] = m
            env_updates["MAX_MARGIN_USAGE"] = str(m)

        # 6. SL ATR Mult
        if "sl_atr_mult" in new_cfg and new_cfg["sl_atr_mult"] is not None:
            sl = float(new_cfg["sl_atr_mult"])
            CONFIG["sl_atr_mult"] = sl
            for b in bots:
                b.cfg["sl_atr_mult"] = sl
            env_updates["SL_ATR_MULT"] = str(sl)

        # 7. Trail ATR Mult
        raw_trail = new_cfg.get("trail_atr_mult") if "trail_atr_mult" in new_cfg else new_cfg.get("trail_atr")
        if raw_trail is not None:
            tr = float(raw_trail)
            CONFIG["trail_atr_mult"] = tr
            for b in bots:
                b.cfg["trail_atr_mult"] = tr
            env_updates["TRAIL_ATR_MULT"] = str(tr)

        # 8. Breakeven
        if "enable_breakeven" in new_cfg and new_cfg["enable_breakeven"] is not None:
            be_trig = 0.50 if bool(new_cfg["enable_breakeven"]) else 0.0
            CONFIG["breakeven_trigger_atr"] = be_trig
            for b in bots:
                b.cfg["breakeven_trigger_atr"] = be_trig
            env_updates["BREAKEVEN_TRIGGER_ATR"] = str(be_trig)

        # 9. Timeframe
        raw_tf = new_cfg.get("entry_timeframe") or new_cfg.get("timeframe")
        if raw_tf:
            tf_int = _parse_timeframe(str(raw_tf), 15)
            CONFIG["timeframe"] = tf_int
            for b in bots:
                b.cfg["timeframe"] = tf_int
            env_updates["TIMEFRAME"] = str(raw_tf).upper()

        # 10. AI Knobs
        if "ai_confidence_threshold" in new_cfg and new_cfg["ai_confidence_threshold"] is not None:
            ai_c = float(new_cfg["ai_confidence_threshold"])
            CONFIG["ai_confidence_threshold"] = ai_c
            for b in bots:
                b.cfg["ai_confidence_threshold"] = ai_c
            env_updates["AI_CONFIDENCE_THRESHOLD"] = str(ai_c)

        if "min_signal_score" in new_cfg and new_cfg["min_signal_score"] is not None:
            score = int(new_cfg["min_signal_score"])
            CONFIG["min_signal_score"] = score
            for b in bots:
                b.cfg["min_signal_score"] = score
            env_updates["MIN_SIGNAL_SCORE"] = str(score)

        raw_adx = new_cfg.get("adx_trending_min") if "adx_trending_min" in new_cfg else new_cfg.get("adx_min")
        if raw_adx is not None:
            adx = int(raw_adx)
            CONFIG["adx_min"] = adx
            for b in bots:
                b.cfg["adx_min"] = adx
            env_updates["ADX_MIN"] = str(adx)

        # 11. Slippage / Deviation
        raw_dev = new_cfg.get("deviation_points") if "deviation_points" in new_cfg else new_cfg.get("max_slippage")
        if raw_dev is not None:
            dev = int(raw_dev)
            CONFIG["max_slippage"] = dev
            for b in bots:
                b.cfg["max_slippage"] = dev

        # 12. Retries & Magic
        if "max_requote_retries" in new_cfg and new_cfg["max_requote_retries"] is not None:
            CONFIG["max_requote_retries"] = int(new_cfg["max_requote_retries"])
            for b in bots:
                b.cfg["max_requote_retries"] = int(new_cfg["max_requote_retries"])

        if "magic_number" in new_cfg and new_cfg["magic_number"] is not None:
            m_num = int(new_cfg["magic_number"])
            CONFIG["magic_number"] = m_num
            for b in bots:
                b.cfg["magic_number"] = m_num
            env_updates["MAGIC_NUMBER"] = str(m_num)

        if "telegram_bot_token" in new_cfg and new_cfg["telegram_bot_token"] is not None:
            t_tok = str(new_cfg["telegram_bot_token"])
            CONFIG["telegram_token"] = t_tok
            for b in bots:
                b.cfg["telegram_token"] = t_tok
            env_updates["TELEGRAM_TOKEN"] = t_tok

        # 13. MT5 Credentials
        if new_cfg.get("mt5_server"):
            srv = str(new_cfg["mt5_server"])
            CONFIG["server"] = srv
            for b in bots:
                b.cfg["server"] = srv
            env_updates["MT5_SERVER"] = srv

        if new_cfg.get("mt5_login"):
            log_num = int(new_cfg["mt5_login"])
            CONFIG["account"] = log_num
            for b in bots:
                b.cfg["account"] = log_num
            env_updates["MT5_LOGIN"] = str(log_num)

        if new_cfg.get("mt5_password"):
            pw = str(new_cfg["mt5_password"])
            CONFIG["password"] = pw
            for b in bots:
                b.cfg["password"] = pw
            env_updates["MT5_PASSWORD"] = pw

        if new_cfg.get("raw_symbols"):
            syms = str(new_cfg["raw_symbols"]).strip()
            env_updates["TRADE_SYMBOLS"] = syms

        # 14. Update Reasoners
        for b in bots:
            if hasattr(b, "reasoner") and b.reasoner is not None:
                b.reasoner.cfg.update(b.cfg)

        # 15. Persist to .env
        if env_updates:
            _update_env_file(env_updates)

        logger.info("Bot Settings Updated from Web UI & persisted to .env! Applied to all %d bot instances.", len(bots))
        return {"status": "ok", "message": "Settings saved and applied successfully."}

    @app.get("/api/config/{section}")
    def api_get_config_section(section: str):
        cfg = bots[0].cfg if bots else CONFIG
        if section == "risk":
            return {
                "max_daily_loss": round(float(cfg.get("max_daily_dd", 0.05)) * 100, 2),
                "max_dd": round(float(cfg.get("max_daily_dd", 0.05)) * 100, 2),
                "risk_per_trade": round(float(cfg.get("risk_per_trade", 0.01)) * 100, 2),
                "max_pos": cfg.get("max_open_pos", 3),
                "manual_policy": cfg.get("default_manual_policy", "observe_only"),
                "trail_atr": cfg.get("trail_atr_mult", 1.0),
                "enable_be": float(cfg.get("breakeven_trigger_atr", 0.0) or 0.0) > 0,
            }
        elif section == "strategy":
            return {
                "trend_enabled": cfg.get("trend_strategy_enabled", True),
                "trend_fast_ma": cfg.get("fast_ema", 10),
                "trend_slow_ma": cfg.get("slow_ema", 50),
                "trend_atr": cfg.get("sl_atr_mult", 1.8),
                "trend_min_score": cfg.get("min_signal_score", 58),
                "mr_enabled": cfg.get("mr_strategy_enabled", False),
                "mr_rsi": cfg.get("rsi_period", 14),
            }
        return cfg

    @app.post("/api/config/{section}")
    async def api_post_config_section(section: str, request: Request):
        payload = await request.json()
        env_updates = {}
        if section == "risk":
            if "risk_per_trade" in payload and payload["risk_per_trade"] is not None:
                r = float(payload["risk_per_trade"])
                if r > 0.05:
                    r = r / 100.0
                CONFIG["risk_per_trade"] = r
                env_updates["RISK_PER_TRADE"] = str(r)
                for b in bots:
                    b.cfg["risk_per_trade"] = r
            if "max_daily_loss" in payload or "max_dd" in payload:
                raw_dd = payload.get("max_daily_loss") if "max_daily_loss" in payload else payload.get("max_dd")
                if raw_dd is not None:
                    dd = float(raw_dd)
                    if dd > 0.05:
                        dd = dd / 100.0
                    CONFIG["max_daily_dd"] = dd
                    env_updates["MAX_DAILY_DD"] = str(dd)
                    for b in bots:
                        b.cfg["max_daily_dd"] = dd
            if "max_pos" in payload and payload["max_pos"] is not None:
                p = int(payload["max_pos"])
                CONFIG["max_open_pos"] = p
                env_updates["MAX_OPEN_POS"] = str(p)
                for b in bots:
                    b.cfg["max_open_pos"] = p
            if "trail_atr" in payload and payload["trail_atr"] is not None:
                tr = float(payload["trail_atr"])
                CONFIG["trail_atr_mult"] = tr
                env_updates["TRAIL_ATR_MULT"] = str(tr)
                for b in bots:
                    b.cfg["trail_atr_mult"] = tr
            if "enable_be" in payload and payload["enable_be"] is not None:
                be_val = 0.50 if bool(payload["enable_be"]) else 0.0
                CONFIG["breakeven_trigger_atr"] = be_val
                env_updates["BREAKEVEN_TRIGGER_ATR"] = str(be_val)
                for b in bots:
                    b.cfg["breakeven_trigger_atr"] = be_val
            if "manual_policy" in payload:
                CONFIG["default_manual_policy"] = str(payload["manual_policy"])
                for b in bots:
                    b.cfg["default_manual_policy"] = str(payload["manual_policy"])
        elif section == "strategy":
            for k, v in payload.items():
                CONFIG[k] = v
                for b in bots:
                    b.cfg[k] = v
        for b in bots:
            if hasattr(b, "reasoner") and b.reasoner is not None:
                b.reasoner.cfg.update(b.cfg)
        if env_updates:
            _update_env_file(env_updates)
        logger.info("Section '%s' configuration updated from Web UI & saved to .env", section)
        return {"status": "success", "section": section}
        
    @app.post("/api/control")
    async def api_control(request: Request):
        data = await request.json()
        action = data.get("action")
        if action == "pause":
            for b in bots:
                b.paused = True
            logger.info("Trading PAUSED via Web UI.")
        elif action == "resume":
            for b in bots:
                b.paused = False
            logger.info("Trading RESUMED via Web UI.")
        elif action == "emergency_stop":
            logger.warning("EMERGENCY STOP Triggered via Web UI!")
            for b in bots:
                b._close_all("EMERGENCY_STOP")
                b.paused = True
        elif action == "close_profitable":
            logger.info("SECURE PROFITS Triggered via Web UI!")
            for b in bots:
                b._close_profitable("USER_SECURE_PROFITS")
        return {"status": "ok"}

    @app.post("/api/emergency-stop")
    def api_emergency_stop():
        logger.warning("EMERGENCY STOP Triggered via /api/emergency-stop!")
        for b in bots:
            b._close_all("EMERGENCY_STOP")
            b.paused = True
        return {"status": "emergency_stop_activated", "message": "All trading stopped and positions closing."}

    @app.post("/api/resume")
    def api_resume():
        logger.info("Trading RESUMED via /api/resume.")
        for b in bots:
            b.paused = False
        return {"status": "trading_resumed", "message": "Trading resumed."}
        
    @app.get("/api/history")
    def api_history():
        import sqlite3
        try:
            db_path = _get_sqlite_db_path()
            if not os.path.exists(db_path):
                return {"history": []}
            conn = sqlite3.connect(db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 50")
            rows = cursor.fetchall()
            conn.close()
            return {"history": [dict(ix) for ix in rows]}
        except Exception:
            return {"history": []}

    return app


def _find_available_dashboard_port() -> int:
    explicit = os.getenv("DASHBOARD_PORT")
    if explicit:
        return int(explicit)
    import socket
    for p in list(range(8000, 8015)) + [8080, 8081, 8082]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    return 8000


if __name__ == "__main__":
    default_symbols = "XAUUSDm,EURUSDm,GBPUSDm" if _IS_AGGRESSIVE else CONFIG.get("symbol", "XAUUSDm")
    symbols_str = os.getenv("TRADE_SYMBOLS", default_symbols)
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    if not symbols:
        symbols = ["XAUUSDm"]

    bots = []
    for sym in symbols:
        sym_cfg = CONFIG.copy()
        sym_cfg["symbol"] = sym
        b = StandaloneAIBot(sym_cfg)
        b.paused = False
        bots.append(b)

    app = create_app(bots)
    dashboard_port = _find_available_dashboard_port()

    def run_dashboard():
        import logging
        import uvicorn
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        try:
            uvicorn.run(app, host="127.0.0.1", port=dashboard_port, log_level="warning")
        except Exception as e:
            logger.error("Failed to start Dashboard on port %d: %s", dashboard_port, e)

    # Start dashboard in background
    import threading
    t = threading.Thread(target=run_dashboard, daemon=True)
    t.start()
    
    import time
    print("\n" + "="*60)
    print(f"🚀 DASHBOARD RUNNING AT: http://127.0.0.1:{dashboard_port}")
    print("="*60 + "\n")
    time.sleep(1)
    
    # Manager loop
    logger.info("=== Bot Manager STARTED | %d symbols ===", len(bots))
    import MetaTrader5 as mt5
    for b in bots:
        if not b.connect():
            logger.error("Failed to connect %s", b.cfg["symbol"])
        else:
            b.running = True
            if hasattr(b, 'notifier') and hasattr(b.notifier, 'send'):
                b.notifier.send(f"🤖 AI Bot started | {b.cfg['symbol']}")
    
    try:
        while True:
            for b in bots:
                if getattr(b, "running", False):
                    try:
                        b.step()
                    except Exception as exc:
                        logger.exception("Error in bot step (%s): %s", b.cfg["symbol"], exc)
            time.sleep(CONFIG.get("sleep_interval", 1.0))
    except KeyboardInterrupt:
        logger.info("Manager stopping…")
    finally:
        mt5.shutdown()
