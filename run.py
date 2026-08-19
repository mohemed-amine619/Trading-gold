"""
run.py – Self-contained AI trading bot for MetaTrader 5 (XAUUSDm).

A single-file "powerful" bot engine:
  • Multi-timeframe confluence scoring (M15 entry / H1 + H4 trend filters)
  • AI ensemble (RandomForest + GradientBoosting + ExtraTrees soft voting,
    ~40 engineered features, rolling retrain) – self-contained, no core/ deps
  • Dynamic risk sizing (ADX-scaled), margin guard, daily drawdown protection
  • ATR trailing stop with acceleration (tightens as profit grows)
  • Breakeven lock + multi-stage partial close (2 stages)
  • Reversal close on strong opposite signal, H1 trend-flip close,
    position-age timeout close, daily-DD equity protection close
  • Telegram + sound notifications

Your account CONFIG below mirrors config.py but is edited per account here.
Run with:  python run.py
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg_module  # shared defaults (kept for reference)

# ---------------------------------------------------------------------------
# Per-account configuration (override anything from config.py here)
# ---------------------------------------------------------------------------
CONFIG = {
    # --- connection ---
    "account":  261009880,
    "password": "Amine2002@",
    "server":   "Exness-MT5Trial16",
    "mt5_path": r"C:\Program Files\MetaTrader 5\terminal64.exe",
    # --- instrument ---
    "symbol":    "XAUUSDm",
    "timeframe": mt5.TIMEFRAME_M15,
    "h1_tf":     mt5.TIMEFRAME_H1,
    "h4_tf":     mt5.TIMEFRAME_H4,
    "n_candles": 600,
    # --- AI ---
    "ai_confidence_threshold": 0.58,
    "ai_train_bars":  500,
    "ai_retrain_bars": 50,
    # --- strategy scoring ---
    "min_signal_score": 55,
    "adx_min": 20,
    "adx_strong": 30,
    "use_session_filter": True,
    # trade window (UTC) – London + New York overlap by default
    "london_start_utc": 7, "london_end_utc": 16,
    "ny_start_utc":     12, "ny_end_utc":     21,
    # --- risk ---
    "risk_per_trade":   0.01,
    "sl_atr_mult":      2.0,
    "tp_rr_ratio":      2.5,
    "trail_atr_mult":   3.0,
    "trail_activate_atr": 1.0,
    "dynamic_risk":     True,
    "min_risk_scale":   0.5,
    "max_risk_scale":   1.5,
    "max_margin_usage": 0.50,
    "max_open_pos":     5,
    "max_daily_dd":     0.03,
    # --- breakeven / partial close ---
    "enable_breakeven":        True,
    "breakeven_trigger_atr":   1.0,
    "enable_partial_close":    True,
    "partial_close_trigger_atr": 1.5,
    "partial_close_fraction":  0.50,
    "partial_close2_trigger_atr": 3.0,
    "partial_close2_fraction": 0.25,
    # --- trailing acceleration (tightens as profit grows) ---
    "trail_tighten1_atr":  2.5,
    "trail_tighten1_mult": 1.5,
    "trail_tighten2_atr":  4.0,
    "trail_tighten2_mult": 1.0,
    # --- exit / close management ---
    "close_on_opposite_signal": True,
    "opposite_signal_min_score": 65,
    "enable_trend_flip_close":   True,
    "close_old_profitable":      True,
    "max_position_age_hours":    24,
    "max_trades_per_day":        8,
    "equity_protect_close_all":  True,
    "server_utc_offset_hours":   2,
    # --- execution ---
    "lot_size":      0.01,         # fallback minimum lot
    "magic_number":  888999,
    "deviation":     30,
    "dry_run":       False,
    "sleep_interval": 0.5,
    "max_slippage_retries": 3,
    # --- notifications ---
    "enable_sound":      True,
    "telegram_token":    "8423946950:AAF0Ja88p_52coyDsh48nQD1yNZW7NwAgck",
    "telegram_chat_id":  "6476316022",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

class Notifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._telegram_dead = False   # set after first connect failure

    def send(self, msg: str) -> None:
        """Non-blocking send – a network hang must never kill the bot."""
        threading.Thread(target=self._send, args=(msg,), daemon=True).start()

    def _send(self, msg: str) -> None:
        try:
            token   = self.cfg.get("telegram_token", "")
            chat_id = self.cfg.get("telegram_chat_id", "")
            if token and chat_id and not self._telegram_dead:
                try:
                    url  = f"https://api.telegram.org/bot{token}/sendMessage"
                    requests.post(url, data={"chat_id": chat_id, "text": msg},
                                  timeout=(3, 5))
                except Exception as exc:
                    self._telegram_dead = True
                    logger.warning(
                        "Telegram unreachable (%s) – notifications disabled "
                        "for this session (sounds still active).", exc)
            try:
                import winsound
                if self.cfg.get("enable_sound"):
                    winsound.Beep(1000, 200)
            except Exception:
                pass
        except BaseException:
            pass


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
    cl, hi, lo = df["close"], df["high"], df["low"]

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

    df["vol_sma20"] = df["tick_volume"].rolling(20).mean()
    df["vol_ratio"] = df["tick_volume"] / (df["vol_sma20"] + 1e-10)

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
]

_EP = 1e-10


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all AI feature columns. Requires `atr` column (from _prepare)."""
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    open_ = df["open"]

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

    df["rsi"]       = ta.rsi(close, length=14)
    df["rsi_delta"] = df["rsi"] - df["rsi"].shift(1)

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

    vol_sma = df["tick_volume"].rolling(20).mean()
    df["vol_ratio"] = df["tick_volume"] / (vol_sma + _EP)

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

    return df


@dataclass
class AISignal:
    direction: int    = 0      # +1 long | -1 short | 0 uncertain
    confidence: float = 0.5
    reason: str       = ""


def _make_pipeline() -> Pipeline:
    rf  = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=20,
        random_state=42, n_jobs=-1)
    gbt = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    et  = ExtraTreesClassifier(
        n_estimators=150, max_depth=8, min_samples_split=20,
        random_state=42, n_jobs=-1)
    voter = VotingClassifier(
        estimators=[("rf", rf), ("gbt", gbt), ("et", et)],
        voting="soft", n_jobs=-1)
    return Pipeline([("scaler", StandardScaler()), ("model", voter)])


_MIN_TRAIN = 150
_MIN_PRED  = 50


class AIEnsemble:
    """Stateful per-symbol ensemble: trains on roll, predicts direction."""

    def __init__(self, symbol: str, config):
        self.symbol          = symbol
        self.config          = config
        self._pipeline: Optional[Pipeline] = None
        self._feature_cols: List[str] = FEATURE_COLS[:]
        self._bars_since_train: int = 0
        self._n_trains: int = 0
        self._trained: bool = False

    def _build_xy(self, df):
        df = build_features(df)
        df["_target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.iloc[:-1].copy()
        avail = [c for c in FEATURE_COLS if c in df.columns]
        df_clean = df[avail + ["_target"]].dropna()
        if len(df_clean) < _MIN_TRAIN:
            return None, None, avail
        return df_clean[avail].values, df_clean["_target"].values, avail

    def train(self, df: pd.DataFrame) -> bool:
        n = min(len(df), self.config.AI_TRAIN_BARS)
        X, y, avail = self._build_xy(df.iloc[-n:].copy())
        if X is None:
            logger.warning("[AI] %s: need >=%d samples, skipping train",
                           self.symbol, _MIN_TRAIN)
            return False
        up_rate = y.mean()
        if up_rate < 0.05 or up_rate > 0.95:
            logger.warning("[AI] %s: degenerate labels (%.0f%% up) – skipping",
                           self.symbol, up_rate * 100)
            return False
        try:
            pipe = _make_pipeline()
            pipe.fit(X, y)
            self._pipeline     = pipe
            self._feature_cols = avail
            self._trained      = True
            self._n_trains    += 1
            logger.info("[AI] %s trained (run #%d | samples=%d | up_rate=%.1f%%)",
                        self.symbol, self._n_trains, len(X), up_rate * 100)
            return True
        except Exception as exc:
            logger.error("[AI] %s training error: %s", self.symbol, exc)
            return False

    def update(self, df: pd.DataFrame) -> None:
        self._bars_since_train += 1
        if not self._trained or self._bars_since_train >= self.config.AI_RETRAIN_BARS:
            if self.train(df):
                self._bars_since_train = 0

    def predict(self, df: pd.DataFrame) -> AISignal:
        if not self._trained or self._pipeline is None:
            return AISignal(0, 0.5, "model not trained yet")
        if len(df) < _MIN_PRED:
            return AISignal(0, 0.5, "not enough bars for prediction")
        try:
            feat_df = build_features(df)
            avail   = [c for c in self._feature_cols if c in feat_df.columns]
            row     = feat_df[avail].iloc[[-1]]
            if row.isnull().any(axis=1).values[0]:
                return AISignal(0, 0.5, "NaN in feature row")
            proba   = self._pipeline.predict_proba(row.values)[0]
            prob_up = float(proba[1])
            thr     = self.config.AI_CONFIDENCE_THRESHOLD
            if prob_up >= thr:
                return AISignal(+1, prob_up, f"AI LONG  conf={prob_up:.3f}")
            if prob_up <= 1.0 - thr:
                conf = 1.0 - prob_up
                return AISignal(-1, conf, f"AI SHORT conf={conf:.3f}")
            return AISignal(0, max(prob_up, 1.0 - prob_up), "AI uncertain")
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
    rsi  = last.get("rsi", 50.0)
    if pd.isna(ef) or pd.isna(es):
        return 0.0
    if direction == 1:
        ema_ok = 1.0 if ef > es else 0.0
        rsi_ok = min(1.0, max(0.0, (rsi - 45.0) / 20.0))
    else:
        ema_ok = 1.0 if ef < es else 0.0
        rsi_ok = min(1.0, max(0.0, (55.0 - rsi) / 20.0))
    return 0.65 * ema_ok + 0.35 * rsi_ok


def _momentum_score(last: pd.Series, direction: int) -> float:
    rsi  = float(last.get("rsi", 50.0))
    mh   = float(last.get("macd_hist", 0.0))
    mhp  = float(last.get("macd_hist_prev", 0.0))
    sk   = float(last.get("stoch_k", 50.0))
    sd   = float(last.get("stoch_d", 50.0))
    score = 0.0
    if direction == 1:
        score += min(1.0, max(0.0, (rsi - 45.0) / 25.0)) * 0.35
        score += (1.0 if mh > 0 and mh > mhp else 0.5 if mh > 0 else 0.0) * 0.35
        score += (1.0 if sk > sd and 50 < sk < 80 else 0.5 if sk > sd else 0.0) * 0.30
    else:
        score += min(1.0, max(0.0, (55.0 - rsi) / 25.0)) * 0.35
        score += (1.0 if mh < 0 and mh < mhp else 0.5 if mh < 0 else 0.0) * 0.35
        score += (1.0 if sk < sd and 20 < sk < 50 else 0.5 if sk < sd else 0.0) * 0.30
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
    h20  = float(df["high"].rolling(20).max().iloc[-1])
    l20  = float(df["low"].rolling(20).min().iloc[-1])
    room = ((h20 - cl) / atr) if direction == 1 else ((cl - l20) / atr)
    return min(1.0, max(0.0, room / 5.0))


def _score_direction(ai_direction: int, ai_conf: float,
                     m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame,
                     direction: int, cfg: dict) -> float:
    """Total confluence score (0-100) for the given direction."""
    adx_val   = float(m15.iloc[-1].get("adx", 0.0))
    adx_strong = cfg.get("adx_strong", 30)

    if ai_direction == direction:
        ai_pts = ai_conf * 40.0
    elif ai_direction == -direction:
        ai_pts = 0.0
    else:
        ai_pts = 12.0

    h1_s  = _trend_score(h1, direction)
    h4_s  = _trend_score(h4, direction)
    trend = h1_s * 12.0 + h4_s * 13.0
    if adx_val >= adx_strong:
        trend = min(25.0, trend * (1.0 + (adx_val - adx_strong) / 50.0))

    mom  = _momentum_score(m15.iloc[-1], direction) * 20.0
    vol  = _volume_score(m15.iloc[-1]) * 10.0
    stru = _structure_score(m15, direction) * 5.0

    return ai_pts + trend + mom + vol + stru


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
              vol_min, vol_max, vol_step, cfg, adx=0.0):
    scale    = _risk_scale(adx, cfg) if cfg.get("dynamic_risk") else 1.0
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
        self._dd_positions_closed = False # one-shot equity protect close

    # --------------------------------------------------------------
    def connect(self) -> bool:
        mt5_path = self.cfg.get("mt5_path", "")
        kwargs   = {}
        if mt5_path:
            kwargs["path"] = mt5_path
        logger.info("Initialising MT5…")
        if not mt5.initialize(**kwargs, timeout=120_000):
            logger.error("MT5 init failed: %s", mt5.last_error())
            return False
        ok = mt5.login(
            self.cfg["account"],
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
        else:
            logger.error("MT5 login failed: %s", mt5.last_error())
        return ok

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
        h1  = _fetch(sym, self.cfg.get("h1_tf", mt5.TIMEFRAME_H1), 300)
        h4  = _fetch(sym, self.cfg.get("h4_tf", mt5.TIMEFRAME_H4), 300)
        return m15, h1, h4

    # --------------------------------------------------------------
    def _get_signal(self, m15, h1, h4) -> tuple:
        """Return (direction, score, ai_conf, reason).  direction 0 = no trade.

        The AI model is always updated/predicted (even off-session or in a
        low-ADX regime) so it keeps learning; the ADX/session gates only
        block *entries*, never training.
        """
        if m15.empty or len(m15) < 50:
            return 0, 0.0, 0.5, "no data"

        m15p = _prepare(m15, self.cfg)
        h1p  = _prepare(h1, self.cfg) if not h1.empty else None
        h4p  = _prepare(h4, self.cfg) if not h4.empty else None

        # AI signal – trained & predicted every new bar
        if self.ai is None:
            self.ai = AIEnsemble(self.cfg["symbol"], type("C", (), {
                "AI_CONFIDENCE_THRESHOLD": self.cfg["ai_confidence_threshold"],
                "AI_TRAIN_BARS":           self.cfg["ai_train_bars"],
                "AI_RETRAIN_BARS":         self.cfg["ai_retrain_bars"],
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

        return best_dir, best_score, ai_conf, "signal"

    # --------------------------------------------------------------
    def _open_positions(self):
        return mt5.positions_get(symbol=self.cfg["symbol"],
                                 magic=self.cfg["magic_number"]) or []

    def _count_positions(self) -> int:
        return len(self._open_positions())

    def _trades_today_ok(self) -> bool:
        today = date.today()
        if self.trade_date != today:
            self.trade_date  = today
            self.trades_today = 0
        return self.trades_today < self.cfg.get("max_trades_per_day", 8)

    # --------------------------------------------------------------
    def _margin_ok(self, price: float, volume: float, equity: float) -> bool:
        sym = self.cfg["symbol"]
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, sym, volume, price)
        margin = float(margin) if margin else 0.0
        acc = mt5.account_info()
        used = float(acc.margin) if acc else 0.0
        limit = equity * self.cfg.get("max_margin_usage", 0.5)
        if margin + used > limit:
            logger.warning("Margin guard: need %.2f + used %.2f > limit %.2f",
                           margin, used, limit)
            return False
        return True

    # --------------------------------------------------------------
    def _send_order(self, direction: int, equity: float) -> bool:
        sym  = self.cfg["symbol"]
        info = mt5.symbol_info(sym)
        if info is None:
            logger.error("symbol_info(%s) failed", sym)
            return False
        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            return False
        price = tick.ask if direction == 1 else tick.bid

        m15p = _prepare(_fetch(sym, self.cfg["timeframe"], 50), self.cfg)
        atr  = float(m15p.iloc[-1].get("atr", info.point * 100))
        adx  = float(m15p.iloc[-1].get("adx", 0.0))

        sl_dist = self.cfg["sl_atr_mult"] * atr
        if direction == 1:
            sl = price - sl_dist
            tp = price + sl_dist * self.cfg["tp_rr_ratio"]
        else:
            sl = price + sl_dist
            tp = price - sl_dist * self.cfg["tp_rr_ratio"]

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
            self.cfg, adx=adx,
        )

        if not self._margin_ok(price, volume, equity):
            return False

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       sym,
            "volume":       float(volume),
            "type":         mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL,
            "price":        float(price),
            "sl":           float(sl),
            "tp":           float(tp),
            "deviation":    self.cfg["deviation"],
            "magic":        self.cfg["magic_number"],
            "comment":      f"AI {'BUY' if direction == 1 else 'SELL'}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
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
                    price = tick.ask if direction == 1 else tick.bid
                    req["price"] = float(price)
                time.sleep(0.3)
                continue
            logger.error("Order failed: retcode=%s",
                         result.retcode if result else "None")
            break
        return False

    # --------------------------------------------------------------
    def _close_position(self, pos, reason: str) -> bool:
        if self.cfg.get("dry_run"):
            logger.info("[DRY-RUN] CLOSE ticket=%s (%s) reason=%s",
                        pos.ticket, pos.symbol, reason)
            return True
        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        tick   = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return False
        price = tick.bid if is_buy else tick.ask
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       float(pos.volume),
            "type":         mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position":     pos.ticket,
            "price":        float(price),
            "deviation":    self.cfg["deviation"],
            "magic":        self.cfg["magic_number"],
            "comment":      "close:" + reason[:24],
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        for attempt in range(self.cfg.get("max_slippage_retries", 3)):
            result = mt5.order_send(req)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                pnl = pos.profit + pos.swap
                msg = (f"🔒 CLOSE {pos.symbol} ticket={pos.ticket} "
                       f"pnl={pnl:+.2f} reason={reason}")
                logger.info(msg)
                self.notifier.send(msg)
                return True
            if result and result.retcode in (mt5.TRADE_RETCODE_REQUOTE,
                                             mt5.TRADE_RETCODE_PRICE_CHANGED,
                                             mt5.TRADE_RETCODE_PRICE_OFF):
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick:
                    price = tick.bid if is_buy else tick.ask
                    req["price"] = float(price)
                time.sleep(0.3)
                continue
            logger.error("Close failed ticket=%s retcode=%s",
                         pos.ticket, result.retcode if result else "None")
            break
        return False

    def _close_all(self, reason: str) -> None:
        for pos in self._open_positions():
            self._close_position(pos, reason)

    # --------------------------------------------------------------
    def _modify_sl(self, pos, new_sl: float) -> None:
        if self.cfg.get("dry_run"):
            logger.info("[DRY-RUN] Modify SL ticket=%s → %.5f", pos.ticket, new_sl)
            return
        req = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   pos.symbol,
            "position": pos.ticket,
            "sl":       float(new_sl),
            "tp":       float(pos.tp),
            "magic":    self.cfg["magic_number"],
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Modified SL ticket=%s → %.5f", pos.ticket, new_sl)

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
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       float(volume),
            "type":         mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position":     pos.ticket,
            "price":        float(price),
            "deviation":    self.cfg["deviation"],
            "magic":        self.cfg["magic_number"],
            "comment":      f"partial{stage}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self._partial_stages.setdefault(pos.ticket, set()).add(stage)
            logger.info("Partial close ticket=%s vol=%.2f stage=%d executed",
                        pos.ticket, volume, stage)
        else:
            logger.error("Partial close failed: %s",
                         result.retcode if result else "None")

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
    def _manage_positions(self, m15p: pd.DataFrame, h1p=None) -> None:
        """Trailing + breakeven + partial close + flip/timeout closes."""
        positions = self._open_positions()
        if not positions:
            return
        if m15p is None or m15p.empty:
            return
        last = m15p.iloc[-1]
        atr  = float(last.get("atr", 0.0))
        if atr <= 0:
            return

        for pos in positions:
            close_price = float(last["close"])
            direction   = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            profit_move = ((close_price - pos.price_open) if direction == 1
                           else (pos.price_open - close_price))

            # ---- H1 trend-flip close ---------------------------------
            if self._h1_trend_flipped(h1p, direction):
                self._close_position(pos, "h1 trend flip")
                continue

            # ---- age timeout close (only if profitable) --------------
            if self.cfg.get("close_old_profitable", True):
                max_age = self.cfg.get("max_position_age_hours", 24)
                if self._position_age(pos) >= max_age:
                    if pos.profit + pos.swap > 0:
                        self._close_position(pos, "age timeout")
                    continue

            # ---- trailing stop (accelerating) ------------------------
            trail_atr  = self.cfg["trail_atr_mult"] * atr
            activate   = self.cfg["trail_activate_atr"] * atr
            if profit_move >= activate:
                if profit_move >= self.cfg.get("trail_tighten2_atr", 4.0) * atr:
                    trail_atr = self.cfg.get("trail_tighten2_mult", 1.0) * atr
                elif profit_move >= self.cfg.get("trail_tighten1_atr", 2.5) * atr:
                    trail_atr = self.cfg.get("trail_tighten1_mult", 1.5) * atr
                new_sl = (close_price - trail_atr if direction == 1
                          else close_price + trail_atr)
                if ((direction == 1  and new_sl > pos.sl + 1e-9) or
                        (direction == -1 and new_sl < pos.sl - 1e-9)):
                    self._modify_sl(pos, new_sl)

            # ---- breakeven --------------------------------------------
            if self.cfg.get("enable_breakeven", True):
                be_trigger = self.cfg.get("breakeven_trigger_atr", 1.0) * atr
                if profit_move >= be_trigger:
                    if ((direction == 1  and pos.sl < pos.price_open) or
                            (direction == -1 and pos.sl > pos.price_open)):
                        self._modify_sl(pos, pos.price_open)

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
    def _close_opposite_positions(self, direction: int, score: float) -> None:
        """Close positions running against a strong new signal."""
        if not self.cfg.get("close_on_opposite_signal", True):
            return
        threshold = self.cfg.get("opposite_signal_min_score", 65)
        if score < threshold:
            return
        for pos in self._open_positions():
            pos_dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
            if pos_dir == -direction:
                self._close_position(pos, "opposite signal")

    # --------------------------------------------------------------
    def run(self) -> None:
        if not self.connect():
            return
        self.running = True
        logger.info("=== AI Bot STARTED | symbol=%s | dry_run=%s ===",
                    self.cfg["symbol"], self.cfg["dry_run"])
        self.notifier.send(f"🤖 AI Bot started | {self.cfg['symbol']}")

        try:
            while self.running:
                m15, h1, h4 = self._get_data()
                if m15.empty:
                    time.sleep(self.cfg["sleep_interval"])
                    continue

                acc    = mt5.account_info()
                equity = acc.equity if acc else 0.0

                if not self._daily_dd_ok(equity):
                    time.sleep(60)
                    continue

                current_bar = m15.index[-1] if not m15.empty else None

                # Manage open positions every cycle
                m15p = _prepare(m15, self.cfg)
                h1p  = _prepare(h1, self.cfg) if not h1.empty else None
                self._manage_positions(m15p, h1p)

                # Entries / reversal closes only on new closed bar
                if current_bar != self.last_bar_time:
                    self.last_bar_time = current_bar
                    direction, score, ai_conf, reason = self._get_signal(m15, h1, h4)
                    logger.info("Bar %s | direction=%+d score=%.1f ai_conf=%.3f (%s)",
                                current_bar, direction, score, ai_conf, reason)

                    if direction != 0:
                        self._close_opposite_positions(direction, score)

                    if (direction != 0 and
                            self._count_positions() < self.cfg["max_open_pos"] and
                            self._trades_today_ok()):
                        self._send_order(direction, equity)

                time.sleep(self.cfg["sleep_interval"])

        except KeyboardInterrupt:
            logger.info("Bot stopping…")
        finally:
            mt5.shutdown()
            logger.info("MT5 closed.")
            self.notifier.send("🛑 AI Bot stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bot = StandaloneAIBot(CONFIG)
    bot.run()