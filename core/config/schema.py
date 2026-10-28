"""
core/config/schema.py — the declarative contract for every runtime parameter.

Why this file exists
--------------------
``run.py`` had grown two parallel sets of parameters (its own ``CONFIG`` and
``core/settings.py``) with different defaults, and the dashboard could only
edit whichever subset someone remembered to add an ``if`` branch for. Worse,
``cfg.get("some_key")`` silently returns a call-site default when the key is
missing, so a parameter that was "added to the dashboard" could be persisted to
``.env`` and still have no effect.

This registry fixes the class of bug:

* A parameter that is not declared here **cannot** be written through the API.
* A declared parameter carries its own bounds, so an out-of-range value is
  rejected with a useful message instead of corrupting a live setting.
* The dashboard renders this schema, so the UI and the validator cannot drift.
* ``dangerous`` marks settings that change the blast radius (paper -> live).
  The API refuses those unless the caller passes ``confirm: true``.

Nothing here imports the broker SDK or reads the environment at import time; it
is a plain data structure, safe to import on Linux.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# MT5 timeframe constants. Duplicated intentionally: this module must not import
# MetaTrader5 (Windows-only) so the schema stays importable on any platform.
TIMEFRAME_LABELS: Dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}
TIMEFRAME_NAMES: Dict[int, str] = {v: k for k, v in TIMEFRAME_LABELS.items()}


@dataclass(frozen=True)
class ParamSpec:
    """One editable parameter.

    ``kind`` is one of: ``float``, ``int``, ``bool``, ``str``, ``enum``,
    ``timeframe``. The type is enforced on write; booleans are deliberately NOT
    accepted as numbers (``True`` would otherwise validate as ``1``).
    """

    key: str
    label: str
    group: str
    kind: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Tuple[str, ...] = ()
    env: Optional[str] = None
    requires_restart: bool = False
    safe_live: bool = True
    dangerous: bool = False
    percent: bool = False
    help: str = ""

    def to_dict(self, current: Any = None) -> Dict[str, Any]:
        value = self.default if current is None else current
        if self.kind == "timeframe":
            value = TIMEFRAME_NAMES.get(int(value), str(value)) if isinstance(value, (int, float)) else value
        return {
            "key": self.key,
            "label": self.label,
            "group": self.group,
            "kind": self.kind,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "choices": list(self.choices),
            "env": self.env,
            "requires_restart": self.requires_restart,
            "safe_live": self.safe_live,
            "dangerous": self.dangerous,
            "percent": self.percent,
            "help": self.help,
            "value": value,
        }


def _p(*args, **kwargs) -> ParamSpec:
    return ParamSpec(*args, **kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Order matters: GROUP_ORDER drives the dashboard, and insertion order drives
# rendering inside a group.

PARAMS: Dict[str, ParamSpec] = {}


def _add(spec: ParamSpec) -> None:
    if spec.key in PARAMS:  # pragma: no cover - guards a copy/paste mistake
        raise RuntimeError(f"duplicate parameter declared: {spec.key}")
    PARAMS[spec.key] = spec


# ── Connection / instrument ────────────────────────────────────────────────
_add(_p("symbol", "Primary symbol", "Instrument", "str", "XAUUSDm",
        env="TRADE_SYMBOL", requires_restart=True, safe_live=False,
        help="Instrument the bot trades. Changing this requires a restart."))
_add(_p("timeframe", "Entry timeframe", "Instrument", "timeframe", 15,
        choices=tuple(TIMEFRAME_LABELS), env="TIMEFRAME", requires_restart=True,
        safe_live=False))
_add(_p("h1_tf", "Trend timeframe", "Instrument", "timeframe", 16385,
        choices=tuple(TIMEFRAME_LABELS), env="TREND_TF", requires_restart=True,
        safe_live=False))
_add(_p("h4_tf", "Macro timeframe", "Instrument", "timeframe", 16388,
        choices=tuple(TIMEFRAME_LABELS), env="MACRO_TREND_TF", requires_restart=True,
        safe_live=False))
_add(_p("n_candles", "History candles", "Instrument", "int", 600, 100, 5000, 50,
        env="HISTORY_CANDLES", requires_restart=True))
_add(_p("magic_number", "Magic number", "Instrument", "int", 888999, 1, 2_000_000_000, 1,
        env="MAGIC_NUMBER", requires_restart=True, safe_live=False,
        help="Separates bot positions from manual ones. Never change while in a trade."))
_add(_p("dry_run", "Paper mode (no real orders)", "Instrument", "bool", True,
        env="TRADING_MODE", requires_restart=False, safe_live=False, dangerous=True,
        help="True = simulated orders. Turning this OFF arms live trading."))

# ── Risk ───────────────────────────────────────────────────────────────────
_add(_p("risk_per_trade", "Risk per trade", "Risk", "float", 0.01, 0.001, 0.05, 0.001,
        env="RISK_PER_TRADE", percent=True,
        help="Fraction of equity risked if the stop is hit (0.01 = 1%)."))
_add(_p("max_daily_dd", "Max daily drawdown", "Risk", "float", 0.03, 0.005, 0.20, 0.005,
        env="MAX_DAILY_DD", percent=True,
        help="Hard stop for the day when equity falls this far below the day's start."))
_add(_p("max_weekly_dd", "Max weekly drawdown", "Risk", "float", 0.08, 0.01, 0.40, 0.01,
        env="MAX_WEEKLY_DD", percent=True))
_add(_p("max_total_dd", "Max total drawdown", "Risk", "float", 0.20, 0.02, 0.60, 0.01,
        env="MAX_TOTAL_DD", percent=True,
        help="Drawdown from the equity peak that stops the bot entirely."))
_add(_p("max_open_pos", "Max open positions", "Risk", "int", 3, 1, 20, 1,
        env="MAX_OPEN_POS"))
_add(_p("max_pos_per_symbol", "Max positions per symbol", "Risk", "int", 1, 1, 20, 1,
        env="MAX_POS_PER_SYMBOL",
        help="One idea, one position: blocks re-firing the same symbol until max_open_pos would stack risk on a single thesis."))
_add(_p("max_margin_usage", "Max margin usage", "Risk", "float", 0.30, 0.05, 0.95, 0.01,
        env="MAX_MARGIN_USAGE", percent=True))
_add(_p("max_trades_per_day", "Max trades per day", "Risk", "int", 15, 1, 500, 1,
        env="MAX_TRADES_PER_DAY"))
_add(_p("max_consecutive_losses", "Loss-streak cooldown trigger", "Risk", "int", 4, 1, 50, 1,
        env="MAX_CONSECUTIVE_LOSSES"))
_add(_p("loss_streak_cooldown_minutes", "Cooldown after loss streak (min)", "Risk", "int", 60, 0, 1440, 5,
        env="LOSS_STREAK_COOLDOWN_MIN"))
_add(_p("dynamic_risk", "Scale risk with ADX", "Risk", "bool", True,
        env="DYNAMIC_RISK"))
_add(_p("min_risk_scale", "Minimum risk scale", "Risk", "float", 0.5, 0.0, 1.0, 0.05,
        env="MIN_RISK_SCALE"))
_add(_p("max_risk_scale", "Maximum risk scale", "Risk", "float", 1.2, 0.0, 3.0, 0.05,
        env="MAX_RISK_SCALE"))
_add(_p("equity_protect_close_all", "Close all at daily limit", "Risk", "bool", True,
        env="EQUITY_PROTECT_CLOSE_ALL"))
_add(_p("reduce_size_at_dd", "Halve size at this share of daily cap", "Risk", "float", 0.50, 0.0, 1.0, 0.05,
        env="REDUCE_SIZE_AT_DD", percent=True,
        help="At this fraction of the daily drawdown cap, new positions are sized at 50%."))
_add(_p("halt_at_dd", "Stop new trades at this share of daily cap", "Risk", "float", 0.80, 0.0, 1.0, 0.05,
        env="HALT_AT_DD", percent=True))

# ── Stops / targets ────────────────────────────────────────────────────────
_add(_p("sl_atr_mult", "Stop distance (ATR multiple)", "Stops & Targets", "float", 2.5, 0.3, 8.0, 0.1,
        env="SL_ATR_MULT"))
_add(_p("tp_rr_ratio", "Take-profit R:R", "Stops & Targets", "float", 2.5, 0.5, 10.0, 0.1,
        env="TP_RR_RATIO"))
_add(_p("location_stop", "Structural stop placement", "Stops & Targets", "bool", True,
        env="LOCATION_STOP",
        help="Place the stop beyond the swing that invalidates the idea."))
_add(_p("location_stop_min_atr", "Structural stop min (ATR)", "Stops & Targets", "float", 0.75, 0.1, 5.0, 0.05,
        env="LOCATION_STOP_MIN_ATR"))
_add(_p("location_stop_max_atr", "Structural stop max (ATR)", "Stops & Targets", "float", 3.0, 0.3, 10.0, 0.1,
        env="LOCATION_STOP_MAX_ATR"))
_add(_p("location_target", "Liquidity/Fib take-profit", "Stops & Targets", "bool", True,
        env="LOCATION_TARGET"))
_add(_p("max_spread_atr_ratio", "Max spread as share of ATR", "Stops & Targets", "float", 0.40, 0.01, 1.0, 0.01,
        env="MAX_SPREAD_ATR_RATIO", percent=True))
_add(_p("max_spread_points", "Max spread (points, 0 = off)", "Stops & Targets", "float", 0.0, 0.0, 1000.0, 1.0,
        env="MAX_SPREAD_POINTS"))

# ── Trade management ───────────────────────────────────────────────────────
_add(_p("trail_atr_mult", "Trail distance (ATR)", "Trade Management", "float", 1.5, 0.1, 10.0, 0.1,
        env="TRAIL_ATR_MULT"))
_add(_p("trail_activate_atr", "Trail activation (ATR profit)", "Trade Management", "float", 1.0, 0.1, 10.0, 0.1,
        env="TRAIL_ACTIVATE_ATR"))
_add(_p("early_protect_atr", "Early protection trigger (ATR)", "Trade Management", "float", 0.3, 0.1, 5.0, 0.1,
        env="EARLY_PROTECT_ATR",
        help="Move SL toward breakeven at this profit level to survive corrections before trailing activates."))
_add(_p("trail_tighten1_atr", "Tighten stage 1 at (ATR)", "Trade Management", "float", 2.0, 0.2, 20.0, 0.1,
        env="TRAIL_TIGHTEN1_ATR"))
_add(_p("trail_tighten1_mult", "Stage 1 trail multiplier", "Trade Management", "float", 1.0, 0.1, 5.0, 0.05,
        env="TRAIL_TIGHTEN1_MULT"))
_add(_p("trail_tighten2_atr", "Tighten stage 2 at (ATR)", "Trade Management", "float", 3.0, 0.3, 30.0, 0.1,
        env="TRAIL_TIGHTEN2_ATR"))
_add(_p("trail_tighten2_mult", "Stage 2 trail multiplier", "Trade Management", "float", 0.6, 0.05, 5.0, 0.05,
        env="TRAIL_TIGHTEN2_MULT"))
_add(_p("structure_trailing", "Structure-based trailing", "Trade Management", "bool", True,
        env="STRUCTURE_TRAILING"))
_add(_p("min_sl_separation_atr", "Min stop separation (ATR)", "Trade Management", "float", 0.55, 0.05, 3.0, 0.05,
        env="MIN_SL_SEPARATION_ATR",
        help="Keeps the ratchet from choking a position on a normal pullback."))
_add(_p("min_sl_step_atr", "Min stop improvement (ATR)", "Trade Management", "float", 0.05, 0.0, 0.50, 0.01,
        env="MIN_SL_STEP_ATR",
        help="Do not send a stop modification unless it improves the stop by at least this much. 0 = every move."))
_add(_p("enable_breakeven", "Move stop to breakeven", "Trade Management", "bool", True,
        env="ENABLE_BREAKEVEN"))
_add(_p("breakeven_trigger_atr", "Breakeven trigger (ATR)", "Trade Management", "float", 0.4, 0.0, 5.0, 0.05,
        env="BREAKEVEN_TRIGGER_ATR"))
_add(_p("breakeven_buffer_atr", "Breakeven buffer (ATR)", "Trade Management", "float", 0.15, 0.0, 2.0, 0.05,
        env="BREAKEVEN_BUFFER_ATR"))
_add(_p("profit_lock_ratio", "Profit lock ratio", "Trade Management", "float", 0.50, 0.0, 1.0, 0.05,
        env="PROFIT_LOCK_RATIO", percent=True))
_add(_p("profit_lock_activate_atr", "Profit lock activation (ATR)", "Trade Management", "float", 1.0, 0.0, 10.0, 0.1,
        env="PROFIT_LOCK_ACTIVATE_ATR"))
_add(_p("profit_giveback_pct", "Close on giveback from peak", "Trade Management", "float", 0.45, 0.05, 1.0, 0.05,
        env="PROFIT_GIVEBACK_PCT", percent=True))
_add(_p("enable_partial_close", "Scale out in stages", "Trade Management", "bool", True,
        env="ENABLE_PARTIAL_CLOSE"))
_add(_p("partial_close_trigger_atr", "First scale-out at (ATR)", "Trade Management", "float", 1.2, 0.2, 10.0, 0.1,
        env="PARTIAL_CLOSE_TRIGGER_ATR"))
_add(_p("scalp_micro_trailing", "Micro trailing", "Trade Management", "bool", True,
        env="SCALP_MICRO_TRAILING"))

# ── Signal quality ─────────────────────────────────────────────────────────
_add(_p("min_signal_score", "Minimum confluence score", "Signals", "float", 50.0, 0.0, 100.0, 1.0,
        env="MIN_SIGNAL_SCORE"))
_add(_p("gold_min_signal_score", "Minimum score (gold)", "Signals", "float", 58.0, 0.0, 100.0, 1.0,
        env="GOLD_MIN_SIGNAL_SCORE"))
_add(_p("adx_min", "Minimum ADX", "Signals", "float", 18.0, 0.0, 100.0, 1.0,
        env="ADX_MIN"))
_add(_p("adx_strong", "Strong-trend ADX", "Signals", "float", 30.0, 0.0, 100.0, 1.0,
        env="ADX_STRONG"))
_add(_p("use_session_filter", "Restrict to London/NY", "Signals", "bool", False,
        env="USE_SESSION_FILTER"))
_add(_p("london_start_utc", "London start (UTC)", "Signals", "int", 7, 0, 23, 1,
        env="LONDON_START_UTC"))
_add(_p("london_end_utc", "London end (UTC)", "Signals", "int", 16, 0, 23, 1,
        env="LONDON_END_UTC"))
_add(_p("ny_start_utc", "New York start (UTC)", "Signals", "int", 12, 0, 23, 1,
        env="NY_START_UTC"))
_add(_p("ny_end_utc", "New York end (UTC)", "Signals", "int", 21, 0, 23, 1,
        env="NY_END_UTC"))
_add(_p("location_gate", "Location veto gate", "Signals", "bool", True,
        env="LOCATION_GATE"))
_add(_p("strict_location_veto", "Strict trap vetoes", "Signals", "bool", True,
        env="STRICT_LOCATION_VETO"))

# ── Strategies ─────────────────────────────────────────────────────────────
_add(_p("fib_enabled", "Fibonacci strategy", "Strategies", "bool", True,
        env="FIB_ENABLED",
        help="Score entries at Fibonacci retracement/extension confluence."))
_add(_p("fib_weight", "Fibonacci weight (pts)", "Strategies", "float", 10.0, 0.0, 40.0, 1.0,
        env="FIB_WEIGHT"))
_add(_p("fib_zone_low", "Fib discount zone start", "Strategies", "float", 0.382, 0.0, 1.0, 0.001,
        env="FIB_ZONE_LOW"))
_add(_p("fib_zone_high", "Fib discount zone end", "Strategies", "float", 0.786, 0.0, 1.0, 0.001,
        env="FIB_ZONE_HIGH"))
_add(_p("fib_extension_target", "Fib extension for TP", "Strategies", "float", 1.618, 1.0, 2.618, 0.001,
        env="FIB_EXTENSION_TARGET"))
_add(_p("trend_weight", "Trend weight (pts)", "Strategies", "float", 30.0, 0.0, 60.0, 1.0,
        env="TREND_WEIGHT"))
_add(_p("momentum_weight", "Momentum weight (pts)", "Strategies", "float", 15.0, 0.0, 40.0, 1.0,
        env="MOMENTUM_WEIGHT"))

# ── AI ─────────────────────────────────────────────────────────────────────
_add(_p("ai_enabled", "Enable ML model", "AI", "bool", True,
        env="AI_ENABLED"))
_add(_p("ai_confidence_threshold", "AI confidence threshold", "AI", "float", 0.55, 0.50, 0.99, 0.01,
        env="AI_CONFIDENCE_THRESHOLD"))
_add(_p("gold_min_ai_conf", "AI confidence threshold (gold)", "AI", "float", 0.55, 0.50, 0.99, 0.01,
        env="GOLD_MIN_AI_CONF"))
_add(_p("gold_macro_veto_enabled", "Gold H4 macro trend veto", "AI", "bool", True,
        env="GOLD_MACRO_VETO",
        help="Veto gold trades when H4 trend opposes. Disable for testing."))
_add(_p("ai_train_bars", "Training window (bars)", "AI", "int", 800, 200, 5000, 50,
        env="AI_TRAIN_BARS"))
_add(_p("ai_retrain_bars", "Retrain every N bars", "AI", "int", 50, 5, 1000, 5,
        env="AI_RETRAIN_BARS"))
_add(_p("ai_validation_fraction", "Validation split fraction", "AI", "float", 0.20, 0.05, 0.50, 0.01,
        env="AI_VALIDATION_FRACTION", percent=True))
_add(_p("ai_min_oos_accuracy", "Min out-of-sample accuracy", "AI", "float", 0.52, 0.40, 0.90, 0.01,
        env="AI_MIN_OOS_ACCURACY"))
_add(_p("ai_min_oos_auc", "Min out-of-sample AUC", "AI", "float", 0.50, 0.40, 0.95, 0.01,
        env="AI_MIN_OOS_AUC"))
_add(_p("ai_min_oos_edge", "Min edge over baseline", "AI", "float", 0.0, -0.10, 0.50, 0.01,
        env="AI_MIN_OOS_EDGE"))
_add(_p("ai_promote_tolerance", "Promotion tolerance", "AI", "float", 0.01, 0.0, 0.20, 0.01,
        env="AI_PROMOTE_TOLERANCE"))
_add(_p("ai_calibrate", "Calibrate probabilities", "AI", "bool", True,
        env="AI_CALIBRATE"))
_add(_p("ai_min_edge_for_size_up", "Min edge to upsize", "AI", "float", 0.10, 0.0, 0.50, 0.01,
        env="AI_MIN_EDGE_FOR_SIZE_UP"))
_add(_p("ai_size_high_conf", "Upsize above confidence", "AI", "float", 0.85, 0.50, 0.99, 0.01,
        env="AI_SIZE_HIGH_CONF"))
_add(_p("ai_size_low_conf", "Downsize below confidence", "AI", "float", 0.60, 0.50, 0.99, 0.01,
        env="AI_SIZE_LOW_CONF"))
_add(_p("ai_model_persist", "Persist trained model to disk", "AI", "bool", True,
        env="AI_MODEL_PERSIST",
        help="Save/load data/models so a restart does not discard a validated model."))
_add(_p("llm_critic_enabled", "Pre-trade AI critic", "AI", "bool", True,
        env="LLM_CRITIC_ENABLED"))
_add(_p("dynamic_sltp_enabled", "Dynamic SL/TP planner", "AI", "bool", True,
        env="DYNAMIC_SLTP_ENABLED"))
_add(_p("llm_provider", "Reasoning provider", "AI", "enum", "embedded",
        choices=("embedded", "local_endpoint", "openai_compatible"),
        env="LLM_PROVIDER",
        help="'embedded' is the built-in rule engine. No Ollama, no external server."))
_add(_p("local_llm_url", "Local endpoint URL", "AI", "str", "http://localhost:1234/v1",
        env="LOCAL_LLM_URL"))
_add(_p("local_llm_model", "Local endpoint model", "AI", "str", "local-model",
        env="LOCAL_LLM_MODEL"))
_add(_p("local_llm_timeout", "Local endpoint timeout (s)", "AI", "float", 3.0, 0.5, 120.0, 0.5,
        env="LOCAL_LLM_TIMEOUT"))

# ── Execution ──────────────────────────────────────────────────────────────
_add(_p("deviation", "Max slippage (points)", "Execution", "int", 20, 1, 500, 1,
        env="DEVIATION"))
_add(_p("sleep_interval", "Loop interval (s)", "Execution", "float", 1.0, 0.1, 60.0, 0.1,
        env="SLEEP_INTERVAL"))
_add(_p("default_manual_policy", "Manual position policy", "Execution", "enum", "observe_only",
        choices=("observe_only", "protect_only", "full_management"),
        env="DEFAULT_MANUAL_POLICY"))
_add(_p("ai_protect_manual_trades", "Protect manual trades with a stop", "Execution", "bool", True,
        env="AI_PROTECT_MANUAL_TRADES"))
_add(_p("manual_trade_sl_atr", "Manual stop (ATR)", "Execution", "float", 2.0, 0.3, 10.0, 0.1,
        env="MANUAL_TRADE_SL_ATR"))
_add(_p("manage_manual_trades_full", "Full management of manual trades", "Execution", "bool", False,
        help="If true, manual trades are managed like bot trades (trailing/partials/opposite closes). If false, only protected."))
_add(_p("smart_manage_manual_trades", "Smart intelligent management", "Execution", "bool", True,
        help="If true and full management off, use AI/structure analysis to manage manual trades intelligently."))
_add(_p("manual_intel_enabled", "Manual trade AI intelligence", "Execution", "bool", True,
        help="Enable analysis-based prediction for manual trades (flip detection, H1 confluence, regime-aware)."))
_add(_p("manual_intel_min_flip_conf", "Min AI flip confidence", "Execution", "float", 0.62, 0.5, 0.95, 0.01))
_add(_p("manual_intel_allow_early_exit_on_flip", "Allow early exit on strong flip", "Execution", "bool", True))
_add(_p("manual_intel_cut_on_momentum_death", "Cut on momentum exhaustion", "Execution", "bool", True))
_add(_p("manual_intel_require_pnl_filter", "Require PnL filter for exits", "Execution", "bool", True))
_add(_p("manual_intel_use_h1_confluence", "Use H1 structure confluence", "Execution", "bool", True))
_add(_p("manual_intel_use_location", "Use market location analysis", "Execution", "bool", True))
_add(_p("manual_intel_tighten_on_exhaustion", "Tighten on exhaustion", "Execution", "bool", True))
_add(_p("manual_intel_runner_protect_atr", "Runner protection threshold (ATR)", "Execution", "float", 1.8, 0.5, 5.0, 0.1))
_add(_p("manual_intel_strong_conf", "Strong flip confidence", "Execution", "float", 0.75, 0.6, 0.99, 0.01))
_add(_p("manual_intel_mid_conf", "Mid flip confidence", "Execution", "float", 0.70, 0.55, 0.95, 0.01))
_add(_p("manual_intel_log_verbose", "Verbose intel logging", "Execution", "bool", True))

GROUP_ORDER: Tuple[str, ...] = (
    "Instrument",
    "Risk",
    "Stops & Targets",
    "Trade Management",
    "Signals",
    "Strategies",
    "AI",
    "Execution",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def coerce_value(spec: ParamSpec, raw: Any) -> Any:
    """Coerce *raw* to the declared type or raise ValueError.

    Deliberately strict: ``bool`` is not a number, numeric strings are allowed
    (they come from JSON forms), and NaN/inf are rejected because they poison
    every downstream comparison.
    """
    if spec.kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in ("true", "1", "yes", "on"):
            return True
        if isinstance(raw, str) and raw.strip().lower() in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{spec.key}: expected a boolean, got {raw!r}")

    if spec.kind == "str":
        if not isinstance(raw, str):
            raise ValueError(f"{spec.key}: expected a string, got {type(raw).__name__}")
        return raw.strip()

    if spec.kind == "enum":
        value = str(raw).strip().lower()
        if value not in spec.choices:
            raise ValueError(
                f"{spec.key}: {value!r} is not one of {list(spec.choices)}"
            )
        return value

    if spec.kind == "timeframe":
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            value = int(raw)
            if value in TIMEFRAME_NAMES:
                return value
            raise ValueError(
                f"{spec.key}: unknown timeframe value {value}; "
                f"expected one of {list(TIMEFRAME_LABELS)}"
            )
        label = str(raw).strip().upper()
        if label in TIMEFRAME_LABELS:
            return TIMEFRAME_LABELS[label]
        raise ValueError(
            f"{spec.key}: unknown timeframe {raw!r}; expected one of {list(TIMEFRAME_LABELS)}"
        )

    # numeric kinds
    if isinstance(raw, bool):
        raise ValueError(f"{spec.key}: expected a number, got a boolean")
    try:
        number = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{spec.key}: expected a number, got {raw!r}") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{spec.key}: value must be finite")
    if spec.kind == "int":
        if not number.is_integer():
            raise ValueError(f"{spec.key}: expected an integer, got {number}")
        number = int(number)
    if spec.min is not None and number < spec.min:
        raise ValueError(f"{spec.key}={number} is below the minimum {spec.min}")
    if spec.max is not None and number > spec.max:
        raise ValueError(f"{spec.key}={number} is above the maximum {spec.max}")
    return number


def validate_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a dashboard write.

    Returns ``(values, errors)``. ``values`` is safe to apply only when
    ``errors`` is empty; partial application is never attempted because a
    half-applied risk change is worse than a rejected one.
    """
    values: Dict[str, Any] = {}
    errors: List[str] = []

    if not isinstance(payload, dict):
        return {}, ["payload must be a JSON object"]

    # `confirm` is a control flag, not a parameter.
    for key, raw in payload.items():
        if key in ("confirm", "_source"):
            continue
        spec = PARAMS.get(key)
        if spec is None:
            errors.append(f"unknown parameter {key!r}")
            continue
        try:
            values[key] = coerce_value(spec, raw)
        except ValueError as exc:
            errors.append(str(exc))

    # Cross-field invariants. Checked only on values that are present in this
    # payload, because a partial update inherits the rest from the live config.
    def _get(name):
        return values.get(name)

    risk = _get("risk_per_trade")
    daily = _get("max_daily_dd")
    if risk is not None and daily is not None and risk >= daily:
        errors.append(
            f"risk_per_trade ({risk}) must be smaller than max_daily_dd ({daily})"
        )

    lo = _get("min_risk_scale")
    hi = _get("max_risk_scale")
    if lo is not None and hi is not None and lo > hi:
        errors.append(f"min_risk_scale ({lo}) must not exceed max_risk_scale ({hi})")

    min_stop = _get("location_stop_min_atr")
    max_stop = _get("location_stop_max_atr")
    if min_stop is not None and max_stop is not None and min_stop > max_stop:
        errors.append(
            f"location_stop_min_atr ({min_stop}) must not exceed location_stop_max_atr ({max_stop})"
        )

    zone_lo = _get("fib_zone_low")
    zone_hi = _get("fib_zone_high")
    if zone_lo is not None and zone_hi is not None and zone_lo >= zone_hi:
        errors.append(f"fib_zone_low ({zone_lo}) must be smaller than fib_zone_high ({zone_hi})")

    low_conf = _get("ai_size_low_conf")
    high_conf = _get("ai_size_high_conf")
    if low_conf is not None and high_conf is not None and low_conf >= high_conf:
        errors.append(
            f"ai_size_low_conf ({low_conf}) must be smaller than ai_size_high_conf ({high_conf})"
        )

    return values, errors


def schema_payload(current: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render the registry plus live values for the dashboard."""
    current = current or {}
    groups: Dict[str, List[Dict[str, Any]]] = {name: [] for name in GROUP_ORDER}
    for spec in PARAMS.values():
        groups.setdefault(spec.group, []).append(spec.to_dict(current.get(spec.key)))
    return {
        "groups": [
            {"name": name, "params": groups.get(name, [])}
            for name in GROUP_ORDER
            if groups.get(name)
        ],
        "count": len(PARAMS),
    }
