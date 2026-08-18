"""
run.py – standalone AI trading bot (no GUI required).

Differences from the modular bot.py:
  • Self-contained single-file script for quick launch / testing.
  • Uses the shared core.ai_model.AIEnsemble for signal generation.
  • Integrates breakeven, partial-close and daily drawdown protection.
  • CONFIG dict mirrors config.py but can be edited independently per account.
"""

import logging
import math
import time
from datetime import date, datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

from core.ai_model import AIEnsemble, build_features
import config as cfg_module   # reuse the shared config defaults

# ---------------------------------------------------------------------------
# Per-account configuration (override anything from config.py here)
# ---------------------------------------------------------------------------
CONFIG = {
    # --- connection ---
    "account":  261009880,
    "password": "Amine2002@",
    "server":   "Exness-MT5Trial16",
    "mt5_path": r"C:\Users\mohamed.bougrioua\Trading\terminal64.exe",
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
    # --- execution ---
    "lot_size":      0.01,         # fallback minimum lot
    "magic_number":  888999,
    "deviation":     30,
    "dry_run":       False,
    "sleep_interval": 0.5,
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

    def send(self, msg: str) -> None:
        token   = self.cfg.get("telegram_token", "")
        chat_id = self.cfg.get("telegram_chat_id", "")
        if token and chat_id:
            try:
                url  = f"https://api.telegram.org/bot{token}/sendMessage"
                requests.post(url, data={"chat_id": chat_id, "text": msg}, timeout=10)
            except Exception as exc:
                logger.warning("Telegram send failed: %s", exc)
        try:
            import winsound
            if self.cfg.get("enable_sound"):
                winsound.Beep(1000, 200)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Indicator preparation (compatible with AIEnsemble)
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach all indicators needed by the scoring system."""
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


def _fetch(symbol: str, tf: int, n: int) -> pd.DataFrame:
    """Fetch `n+1` bars and drop the forming candle → n closed bars."""
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n + 1)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates).iloc[:-1].copy()
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    return df[["open", "high", "low", "close", "tick_volume", "spread"]].astype(float)


# ---------------------------------------------------------------------------
# Signal scoring (mirrors strategy.py logic)
# ---------------------------------------------------------------------------

def _session_ok(cfg: dict) -> bool:
    if not cfg.get("use_session_filter", True):
        return True
    h = datetime.now(timezone.utc).hour
    return (7 <= h < 16) or (12 <= h < 21)


def _trend_score(df, direction: int) -> float:
    if df is None or df.empty:
        return 0.0
    last  = df.iloc[-1]
    ef    = last.get("ema50",  float("nan"))
    es    = last.get("ema200", float("nan"))
    rsi   = last.get("rsi", 50.0)
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
    rsi   = float(last.get("rsi", 50.0))
    mh    = float(last.get("macd_hist", 0.0))
    mhp   = float(last.get("macd_hist_prev", 0.0))
    sk    = float(last.get("stoch_k", 50.0))
    sd    = float(last.get("stoch_d", 50.0))
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
    last  = df.iloc[-1]
    cl    = float(last["close"])
    atr   = float(last.get("atr", 0.0))
    if atr <= 0:
        return 0.5
    h20   = float(df["high"].rolling(20).max().iloc[-1])
    l20   = float(df["low"].rolling(20).min().iloc[-1])
    room  = ((h20 - cl) / atr) if direction == 1 else ((cl - l20) / atr)
    return min(1.0, max(0.0, room / 5.0))


def _score_direction(
    ai_direction: int, ai_conf: float,
    m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame,
    direction: int, cfg: dict,
) -> float:
    """Return total confluence score (0-100) for the given direction."""
    adx_val   = float(m15.iloc[-1].get("adx", 0.0))
    adx_strong = cfg.get("adx_strong", 30)

    # Layer 1: AI (0-40)
    if ai_direction == direction:
        ai_pts = ai_conf * 40.0
    elif ai_direction == -direction:
        ai_pts = 0.0
    else:
        ai_pts = 12.0

    # Layer 2: Trend (0-25)
    h1_s  = _trend_score(h1,  direction)
    h4_s  = _trend_score(h4,  direction)
    trend = h1_s * 12.0 + h4_s * 13.0
    if adx_val >= adx_strong:
        trend = min(25.0, trend * (1.0 + (adx_val - adx_strong) / 50.0))

    # Layer 3: Momentum (0-20)
    mom = _momentum_score(m15.iloc[-1], direction) * 20.0

    # Layer 4: Volume (0-10)
    vol = _volume_score(m15.iloc[-1]) * 10.0

    # Layer 5: Structure (0-5)
    struct = _structure_score(m15, direction) * 5.0

    return ai_pts + trend + mom + vol + struct


# ---------------------------------------------------------------------------
# Risk helpers (standalone, duplicating risk_manager logic without MT5 async)
# ---------------------------------------------------------------------------

def _risk_scale(adx: float, cfg: dict) -> float:
    lo  = cfg.get("min_risk_scale", 0.5)
    hi  = cfg.get("max_risk_scale", 1.5)
    lo_adx = cfg.get("adx_min", 20)
    hi_adx = cfg.get("adx_strong", 30) + 20
    if adx <= lo_adx:
        return lo
    if adx >= hi_adx:
        return hi
    return lo + (adx - lo_adx) / (hi_adx - lo_adx) * (hi - lo)


def _size_lot(account_equity, sl_dist_price, tick_size, tick_value,
              vol_min, vol_max, vol_step, cfg, adx=0.0):
    scale      = _risk_scale(adx, cfg) if cfg.get("dynamic_risk") else 1.0
    risk_amt   = account_equity * cfg["risk_per_trade"] * scale
    sl_ticks   = sl_dist_price / (tick_size + 1e-10)
    rpl        = sl_ticks * tick_value
    if rpl <= 0:
        return vol_min
    raw    = risk_amt / rpl
    vol    = math.floor(raw / vol_step) * vol_step
    return max(vol_min, min(vol, vol_max))


# ---------------------------------------------------------------------------
# Main bot class
# ---------------------------------------------------------------------------

class StandaloneAIBot:
    """Single-symbol AI trading bot that runs in a blocking loop."""

    def __init__(self, cfg: dict):
        self.cfg           = cfg
        self.notifier      = Notifier(cfg)
        self.ai            = None      # initialised after first data fetch
        self.running       = False
        self.last_bar_time = None
        self.start_balance = 0.0
        self.daily_start   = 0.0
        self.daily_date: Optional[date] = None
        self._partial_closed: set = set()   # position tickets partially closed

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _daily_dd_ok(self, equity: float) -> bool:
        today = date.today()
        if self.daily_date != today:
            self.daily_date  = today
            self.daily_start = equity
        if self.daily_start > 0:
            dd = (self.daily_start - equity) / self.daily_start
            if dd >= self.cfg["max_daily_dd"]:
                logger.warning("Daily DD limit hit (%.1f%%) – no new trades today", dd * 100)
                return False
        return True

    # ------------------------------------------------------------------
    def _get_data(self) -> tuple:
        sym  = self.cfg["symbol"]
        n    = self.cfg["n_candles"]
        m15  = _fetch(sym, self.cfg["timeframe"], n)
        h1   = _fetch(sym, self.cfg.get("h1_tf", mt5.TIMEFRAME_H1), 300)
        h4   = _fetch(sym, self.cfg.get("h4_tf", mt5.TIMEFRAME_H4), 300)
        return m15, h1, h4

    # ------------------------------------------------------------------
    def _get_signal(self, m15, h1, h4) -> tuple:
        """Return (direction, score, ai_conf).  direction 0 = no trade."""
        if m15.empty or len(m15) < 50:
            return 0, 0.0, 0.5

        m15p = _prepare(m15, self.cfg)
        h1p  = _prepare(h1,  self.cfg) if not h1.empty else None
        h4p  = _prepare(h4,  self.cfg) if not h4.empty else None

        # ADX gate
        adx_val = float(m15p.iloc[-1].get("adx", 0.0))
        if adx_val < self.cfg.get("adx_min", 20):
            return 0, 0.0, 0.5

        # Session gate
        if not _session_ok(self.cfg):
            return 0, 0.0, 0.5

        # AI signal
        if self.ai is None:
            self.ai = AIEnsemble(self.cfg["symbol"], type("C", (), {
                "AI_CONFIDENCE_THRESHOLD": self.cfg["ai_confidence_threshold"],
                "AI_TRAIN_BARS":           self.cfg["ai_train_bars"],
                "AI_RETRAIN_BARS":         self.cfg["ai_retrain_bars"],
                "EMA_FAST":  50, "EMA_SLOW": 200, "RSI_LENGTH": 14,
                "ATR_PERIOD": 14,
            })())
        self.ai.update(m15p)
        ai_sig       = self.ai.predict(m15p)
        ai_direction = ai_sig.direction
        ai_conf      = ai_sig.confidence

        # Score both directions
        best_dir   = 0
        best_score = 0.0
        for d in [1, -1]:
            score = _score_direction(ai_direction, ai_conf, m15p, h1p, h4p, d, self.cfg)
            if score > best_score:
                best_score = score
                best_dir   = d

        threshold = self.cfg.get("min_signal_score", 55)
        if best_score < threshold:
            return 0, best_score, ai_conf
        return best_dir, best_score, ai_conf

    # ------------------------------------------------------------------
    def _open_positions(self):
        return mt5.positions_get(symbol=self.cfg["symbol"],
                                 magic=self.cfg["magic_number"]) or []

    def _count_positions(self) -> int:
        return len(self._open_positions())

    # ------------------------------------------------------------------
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

        # ATR-based SL/TP
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

        # Enforce min stop distance
        min_d = info.trade_stops_level * info.point
        if min_d > 0:
            if abs(price - sl) < min_d:
                sl = price - min_d if direction == 1 else price + min_d
            if abs(tp - price) < min_d:
                tp = price + min_d if direction == 1 else price - min_d

        # Dynamic lot sizing
        volume = _size_lot(
            equity, abs(price - sl),
            info.trade_tick_size, info.trade_tick_value,
            info.volume_min, info.volume_max, info.volume_step,
            self.cfg, adx=adx,
        )

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

        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            msg = (f"✅ {'BUY' if direction==1 else 'SELL'} {sym} "
                   f"vol={volume:.2f} @ {price:.5f}  SL={sl:.5f}  TP={tp:.5f}")
            logger.info(msg)
            self.notifier.send(msg)
            return True
        logger.error("Order failed: retcode=%s", result.retcode if result else "None")
        return False

    # ------------------------------------------------------------------
    def _manage_positions(self, m15p: pd.DataFrame) -> None:
        """Trailing stop + breakeven + partial close for open positions."""
        positions = self._open_positions()
        if not positions:
            return
        if m15p.empty:
            return
        last = m15p.iloc[-1]
        atr  = float(last.get("atr", 0.0))
        if atr <= 0:
            return

        for pos in positions:
            close_price = float(last["close"])
            direction   = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1

            # --- trailing stop ---
            profit_move = ((close_price - pos.price_open) if direction == 1
                           else (pos.price_open - close_price))
            trail_atr   = self.cfg["trail_atr_mult"] * atr
            activate    = self.cfg["trail_activate_atr"] * atr
            if profit_move >= activate:
                new_sl = (close_price - trail_atr if direction == 1
                          else close_price + trail_atr)
                if ((direction == 1  and new_sl > pos.sl + 1e-9) or
                        (direction == -1 and new_sl < pos.sl - 1e-9)):
                    self._modify_sl(pos, new_sl)

            # --- breakeven ---
            if self.cfg.get("enable_breakeven", True):
                be_trigger = self.cfg.get("breakeven_trigger_atr", 1.0) * atr
                if profit_move >= be_trigger:
                    if ((direction == 1  and pos.sl < pos.price_open) or
                            (direction == -1 and pos.sl > pos.price_open)):
                        self._modify_sl(pos, pos.price_open)

            # --- partial close ---
            if (self.cfg.get("enable_partial_close", True) and
                    pos.ticket not in self._partial_closed):
                pc_trigger = self.cfg.get("partial_close_trigger_atr", 1.5) * atr
                if profit_move >= pc_trigger:
                    info = mt5.symbol_info(pos.symbol)
                    if info:
                        frac    = self.cfg.get("partial_close_fraction", 0.5)
                        step    = info.volume_step or 0.01
                        partial = math.floor(pos.volume * frac / step) * step
                        if (partial >= info.volume_min and
                                pos.volume - partial >= info.volume_min):
                            self._partial_close(pos, partial)

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

    def _partial_close(self, pos, volume: float) -> None:
        if self.cfg.get("dry_run"):
            logger.info("[DRY-RUN] Partial close ticket=%s vol=%.2f", pos.ticket, volume)
            self._partial_closed.add(pos.ticket)
            return
        is_buy = pos.type == mt5.ORDER_TYPE_BUY
        tick   = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return
        price  = tick.bid if is_buy else tick.ask
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       pos.symbol,
            "volume":       float(volume),
            "type":         mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position":     pos.ticket,
            "price":        float(price),
            "deviation":    self.cfg["deviation"],
            "magic":        self.cfg["magic_number"],
            "comment":      "partial",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self._partial_closed.add(pos.ticket)
            logger.info("Partial close ticket=%s vol=%.2f executed", pos.ticket, volume)
        else:
            logger.error("Partial close failed: %s",
                         result.retcode if result else "None")

    # ------------------------------------------------------------------
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

                # Daily drawdown check
                if not self._daily_dd_ok(equity):
                    time.sleep(60)
                    continue

                # Current bar detection
                current_bar = m15.index[-1] if not m15.empty else None

                # Manage open positions on every cycle
                m15p = _prepare(m15, self.cfg)
                self._manage_positions(m15p)

                # Entry signals only on new closed bar
                if current_bar != self.last_bar_time:
                    self.last_bar_time = current_bar
                    direction, score, ai_conf = self._get_signal(m15, h1, h4)
                    logger.info(
                        "Bar %s | direction=%+d score=%.1f ai_conf=%.3f",
                        current_bar, direction, score, ai_conf)

                    if (direction != 0 and
                            self._count_positions() < self.cfg["max_open_pos"]):
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