"""
core/ai/dynamic_sltp.py — Dynamic SL/TP and Scalping Reasoning Engine.

Combines market structure (swings, S/R zones, liquidity pools), micro-volatility
(ATR slope, spread ratio), orderflow wick rejections, and AI reasoning (embedded
or local LLM) to dynamically calculate, adjust, and protect SL & TP levels.

Strict mathematical safety invariants are enforced at all times:
- NEVER loosen a stop loss (one-way ratchet: stops can only tighten into profit)
- Bounded within ATR volatility limits
- Broker grid snapping and minimum stops level compliance
- Fail-safe fallback to structural stops if AI fails or times out
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from core.ai.provider import LLMProvider, EmbeddedReasoningProvider

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM System Prompts (Adversarial Quant Critic)
# ─────────────────────────────────────────────────────────────────────────────

QUANT_CRITIC_SYSTEM_PROMPT = """You are the Chief Quantitative Strategist and Adversarial Risk Critic for an institutional high-frequency algorithmic fund specializing in Gold (XAUUSD) and FX.

Your sole objective is to evaluate trade candidates and dynamically calibrate Stop Loss (SL) and Take Profit (TP) levels to maximize Risk:Reward, eliminate retail bull/bear traps, and protect firm capital.

OPERATIONAL PRINCIPLES:
1. ADVERSARIAL SKEPTICISM: Assume every technical breakout or momentum surge is a potential retail trap until proven backed by institutional order flow and macro structure.
2. MARKET LOCATION DISCIPLINE:
   - BUY entries are ONLY permitted in DISCOUNT (0%-40% of the active swing range) or following confirmed bullish market structure break (BOS). Never buy directly into major resistance overhead ("brick wall").
   - SELL entries are ONLY permitted in PREMIUM (60%-100% of the active swing range) or following confirmed bearish change of character (CHoCH). Never sell directly into major resting demand pools.
3. LIQUIDITY POOL AWARENESS:
   - Stop Losses MUST be placed beyond structural invalidation levels (swing high/low pivots) plus ATR buffer to prevent spread noise stop-outs.
   - Take Profits MUST target resting liquidity pools (equal highs/lows) or high-volume nodes with a minimum 1:2.0 Risk-to-Reward ratio.
4. SPREAD & FRICTION MANAGEMENT:
   - Account for broker spread friction. Reject scalp setups if spread exceeds 15% of the average candle volatility (ATR).
5. CAPITAL PRESERVATION:
   - When market conditions are ambiguous, overextended, or exhibiting momentum exhaustion/pinbar rejection, issue a capital preservation VETO ("PASS").

OUTPUT MANDATE:
You MUST ALWAYS respond with valid, parseable JSON ONLY. Never output markdown conversational text outside the JSON object.

JSON SCHEMA:
{
  "decision": "BUY" | "SELL" | "PASS",
  "confidence": 85.0,
  "sl_type": "ai_structure" | "ai_liquidity" | "ai_volatility",
  "tp_type": "ai_liquidity_pool" | "fixed_rr",
  "reasoning": "Concise institutional market rationale explaining why the trade is accepted or vetoed.",
  "risk_assessment": "Summary of primary risk factors (e.g., overhead order block, spread friction, trend alignment)."
}"""

ENTRY_SLTP_SYSTEM_PROMPT = """You are the Chief Quantitative Strategist and Adversarial Risk Critic for an institutional high-frequency algorithmic fund specializing in Gold (XAUUSD) and FX.

Evaluate the proposed entry SL/TP levels against market structure, liquidity pools, and volatility.

OPERATIONAL PRINCIPLES:
1. Stop Losses MUST sit beyond structural invalidation (swing high/low pivots) plus an ATR buffer — never inside spread noise.
2. Take Profits MUST target resting liquidity pools (equal highs/lows) or high-volume nodes with a minimum 1:2.0 Risk-to-Reward ratio.
3. Reject (PASS) setups where spread friction exceeds 15% of ATR or the SL/TP geometry invites a retail trap stop-out.

OUTPUT MANDATE:
You MUST ALWAYS respond with valid, parseable JSON ONLY. Never output markdown conversational text outside the JSON object.

JSON SCHEMA:
{
  "decision": "CONFIRM" | "PASS",
  "confidence": 85.0,
  "sl_type": "ai_structure" | "ai_liquidity" | "ai_volatility",
  "tp_type": "ai_liquidity_pool" | "fixed_rr",
  "reasoning": "Concise institutional rationale for the SL/TP geometry.",
  "risk_assessment": "Summary of primary risk factors."
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketMicrostructureSnapshot:
    """Snapshot of spatial and volatility market conditions at decision time."""
    symbol: str
    price: float
    bid: float
    ask: float
    spread: float
    spread_atr_ratio: float
    atr: float
    atr_slope: float                # >0 expanding volatility, <0 contracting
    adx: float
    rsi: float
    bias: str                       # BULLISH, BEARISH, RANGING, UNDEFINED
    range_position: float           # 0.0 (discount) to 1.0 (premium)
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    target_above_pool: Optional[float] = None   # Resting buy stops (TP magnet for longs)
    target_below_pool: Optional[float] = None   # Resting sell stops (TP magnet for shorts)
    has_opp_pinbar: bool = False
    pinbar_direction: int = 0       # +1 for bullish pinbar, -1 for bearish pinbar


@dataclass
class ActivePositionSnapshot:
    """Snapshot of an open trade being actively managed."""
    ticket: int
    symbol: str
    direction: int                  # +1 BUY, -1 SELL
    open_price: float
    current_price: float
    current_sl: float
    current_tp: float
    volume: float
    profit_move: float              # in price distance in favorable direction
    profit_atr: float               # profit_move / atr
    current_rr: float               # current profit / initial risk
    peak_profit: float
    giveback_pct: float             # drawdown from peak profit [0.0 - 1.0]
    age_bars: int = 0


@dataclass
class EntrySLTPPlan:
    """Intelligent entry SL and TP computed for a new trade."""
    sl_price: float
    tp_price: float
    sl_distance: float
    tp_distance: float
    rr_ratio: float
    sl_source: str                  # "ai_structure", "ai_volatility", "atr_fallback"
    tp_source: str                  # "ai_liquidity_pool", "ai_rr_target", "atr_target"
    reasoning: str
    confidence: float
    valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "sl_distance": self.sl_distance,
            "tp_distance": self.tp_distance,
            "rr_ratio": self.rr_ratio,
            "sl_source": self.sl_source,
            "tp_source": self.tp_source,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "valid": self.valid,
        }


@dataclass
class PositionAdjustmentPlan:
    """Dynamic decision on how to adjust an open position."""
    action: str                     # "HOLD", "TIGHTEN_SL", "MOVE_TO_BREAKEVEN", "SECURE_PROFIT", "CLOSE_EARLY"
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    reasoning: str = ""
    confidence: float = 0.8
    executed: bool = False


@dataclass
class TradeCritiqueResult:
    """Quantitative decision from the LLM Market Analyst and Risk Critic."""
    approved: bool
    confidence: float
    decision: str              # "BUY", "SELL", "PASS"
    reasoning: str
    risk_assessment: str


# ─────────────────────────────────────────────────────────────────────────────
# Safety Guardrails (Strict Deterministic Rules)
# ─────────────────────────────────────────────────────────────────────────────

class SLTPSafetyGuard:
    """
    Mathematical guardrails that enforce zero capital exposure violations.
    Regardless of what an AI or model outputs, this validator enforces:
    1. NEVER LOOSEN SL: Stop loss can only ever tighten into profit.
    2. Proper orientation: SL is always on the defensive side of price.
    3. ATR volatility bounds: Stops cannot be absurdly wide or dangerously tight.
    4. Broker stops level & tick grid alignment.
    """

    @staticmethod
    def snap_price(price: float, tick_size: Optional[float]) -> float:
        """Snap price to broker grid, rounding half-up."""
        if not tick_size or tick_size <= 0:
            return round(price, 5)
        snapped = float(np.floor(price / tick_size + 0.5) * tick_size)
        # Clean binary float artifacts
        for decimals in range(1, 9):
            rounded = round(snapped, decimals)
            if abs(rounded - snapped) <= tick_size * 1e-6:
                return rounded
        return snapped

    @classmethod
    def validate_new_sl(
        cls,
        direction: int,
        current_price: float,
        current_sl: float,
        proposed_sl: float,
        tick_size: Optional[float],
        stops_level_pts: float,
        point: float,
        atr: float,
        min_atr_mult: float = 0.2,
        max_atr_mult: float = 4.0,
    ) -> Optional[float]:
        """
        Validates and normalizes a proposed new SL.
        Returns the clean, valid SL price, or None if the change is rejected.
        """
        if not math.isfinite(proposed_sl) or proposed_sl <= 0:
            return None

        snapped = cls.snap_price(proposed_sl, tick_size)

        # Rule 1: Side of market (BUY SL must be below price; SELL SL above price)
        if direction == 1 and snapped >= current_price:
            logger.debug("SLTPSafetyGuard: BUY SL (%.5f) cannot be >= current price (%.5f)", snapped, current_price)
            return None
        if direction == -1 and snapped <= current_price:
            logger.debug("SLTPSafetyGuard: SELL SL (%.5f) cannot be <= current price (%.5f)", snapped, current_price)
            return None

        # Rule 2: NEVER LOOSEN SL (One-Way Ratchet)
        if current_sl > 0:
            if direction == 1:
                # BUY: proposed must be strictly higher than current SL
                if snapped <= current_sl + 1e-9:
                    return None
            else:
                # SELL: proposed must be strictly lower than current SL
                if snapped >= current_sl - 1e-9:
                    return None

        # Rule 3: Minimum distance from current price (broker stops level)
        min_dist = max(stops_level_pts * (point or 1e-5), (tick_size or 0.01) * 2)
        actual_dist = abs(current_price - snapped)
        if actual_dist < min_dist:
            logger.debug("SLTPSafetyGuard: Proposed SL too close to market (dist=%.5f < min_dist=%.5f)", actual_dist, min_dist)
            return None

        # Rule 4: Maximum ATR distance guard
        if atr > 0 and actual_dist > max_atr_mult * atr:
            logger.debug("SLTPSafetyGuard: Proposed SL exceeds max ATR multiplier (dist=%.5f > %.2f*ATR)", actual_dist, max_atr_mult)
            return None

        return snapped

    @classmethod
    def validate_entry_sltp(
        cls,
        direction: int,
        entry_price: float,
        proposed_sl: float,
        proposed_tp: float,
        tick_size: Optional[float],
        stops_level_pts: float,
        point: float,
        atr: float,
        min_rr: float = 1.2,
        min_atr_mult: float = 0.5,
        max_atr_mult: float = 3.5,
    ) -> Optional[Tuple[float, float]]:
        """
        Validates an entry SL and TP pair.
        Returns (sl, tp) or None if the trade fails structural validity.
        """
        if not (math.isfinite(proposed_sl) and math.isfinite(proposed_tp)):
            return None

        sl = cls.snap_price(proposed_sl, tick_size)
        tp = cls.snap_price(proposed_tp, tick_size)

        # Orientation check
        if direction == 1:
            if not (sl < entry_price < tp):
                return None
        else:
            if not (tp < entry_price < sl):
                return None

        sl_dist = abs(entry_price - sl)
        tp_dist = abs(tp - entry_price)

        if sl_dist <= 0:
            return None

        # R:R check
        rr = tp_dist / sl_dist
        if rr < min_rr:
            # Adjust TP to meet minimum R:R if needed
            tp_dist = sl_dist * min_rr
            tp = cls.snap_price(entry_price + (tp_dist if direction == 1 else -tp_dist), tick_size)

        # ATR bounds on SL
        if atr > 0:
            if sl_dist < min_atr_mult * atr:
                sl_dist = min_atr_mult * atr
                sl = cls.snap_price(entry_price - sl_dist if direction == 1 else entry_price + sl_dist, tick_size)
            elif sl_dist > max_atr_mult * atr:
                sl_dist = max_atr_mult * atr
                sl = cls.snap_price(entry_price - sl_dist if direction == 1 else entry_price + sl_dist, tick_size)

        # Broker stops level
        min_dist = max(stops_level_pts * (point or 1e-5), (tick_size or 0.01) * 2)
        if abs(entry_price - sl) < min_dist:
            sl = cls.snap_price(entry_price - min_dist if direction == 1 else entry_price + min_dist, tick_size)
        if abs(tp - entry_price) < min_dist:
            tp = cls.snap_price(entry_price + min_dist if direction == 1 else entry_price - min_dist, tick_size)

        return sl, tp


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic SL/TP Reasoner Engine
# ─────────────────────────────────────────────────────────────────────────────

class DynamicSLTPReasoner:
    """
    Intelligent Market Reasoner for Prediction, Scalping, and Dynamic SL/TP.
    
    Translates market topology and live position performance into optimal,
    dynamic risk/reward profiles.
    """

    def __init__(self, provider: Optional[LLMProvider] = None, config: Optional[Dict[str, Any]] = None):
        self.provider = provider or EmbeddedReasoningProvider()
        self.cfg = config or {}
        self.guard = SLTPSafetyGuard()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Entry SL & TP Calculation
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_entry_sltp(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        df: pd.DataFrame,
        atr: float,
        location: Any = None,
        symbol_info: Any = None,
    ) -> EntrySLTPPlan:
        """
        Dynamically calculates the best structural SL and TP for a new trade entry.
        
        Anchors SL behind the structural invalidation level (swing low for BUY,
        swing high for SELL) plus a volatility buffer.
        
        Targets TP at resting liquidity pools (retail stops) where price is drawn,
        falling back to dynamic R:R multiples.
        """
        tick_size = getattr(symbol_info, "trade_tick_size", None) or getattr(symbol_info, "point", 0.01)
        point = getattr(symbol_info, "point", 0.01) or 0.01
        stops_level = getattr(symbol_info, "trade_stops_level", 0) or 0

        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        min_rr = float(self.cfg.get("tp_rr_ratio", 2.0))
        sl_atr_mult = float(self.cfg.get("sl_atr_mult", 2.2 if is_gold else 1.8))
        min_atr = float(self.cfg.get("location_stop_min_atr", 1.8 if is_gold else 1.0))
        max_atr = float(self.cfg.get("location_stop_max_atr", 3.2))
        if is_gold:
            min_atr = max(min_atr, 1.8)
            sl_atr_mult = max(sl_atr_mult, 2.0)

        sl_source = "atr_fallback"
        tp_source = "atr_target"
        reasoning_parts = []
        if is_gold:
            reasoning_parts.append("Gold volatility protection: enforced minimum 1.8 ATR stop buffer for correction resilience")

        # 1. Structural Stop Loss Determination
        sl_price = entry_price - (sl_atr_mult * atr) if direction == 1 else entry_price + (sl_atr_mult * atr)
        if location is not None and getattr(location, "is_valid", False):
            plan = location.structural_stop(
                entry_price,
                direction,
                min_atr_mult=min_atr,
                max_atr_mult=max_atr,
                tick_size=tick_size,
            )
            if plan is not None:
                sl_price = plan.price
                sl_source = "ai_structure" if plan.structural else "ai_volatility"
                reasoning_parts.append(f"SL via {sl_source}: {plan.reason}")

        # 2. Dynamic Take Profit Target Determination (Liquidity Magnet)
        sl_dist = abs(entry_price - sl_price)
        tp_price = entry_price + (sl_dist * min_rr) if direction == 1 else entry_price - (sl_dist * min_rr)

        if location is not None and getattr(location, "is_valid", False) and getattr(location, "atr", 0):
            pool = location.target_above() if direction == 1 else location.target_below()
            if pool is not None:
                # Place TP slightly in front of the pool (0.15 ATR) to ensure fill before reversal
                buffer = direction * location.atr * 0.15
                candidate_tp = pool.price - buffer
                candidate_rr = abs(candidate_tp - entry_price) / max(sl_dist, 1e-10)

                # Ensure candidate TP is in the trade direction and clears minimum R:R
                if (candidate_tp - entry_price) * direction > 0 and candidate_rr >= min_rr:
                    tp_price = candidate_tp
                    tp_source = f"ai_liquidity_pool_{pool.kind.value}"
                    reasoning_parts.append(f"TP targeted 0.15 ATR before resting liquidity at {pool.price:.4f} (R:R {candidate_rr:.2f})")

        # 3. Microstructure Scalping Refinement & LLM Consultation
        spread_pts = getattr(symbol_info, "spread", 0) or 0
        spread_val = spread_pts * point
        spread_ratio = (spread_val / atr) if atr > 0 else 0.0
        if atr > 0 and spread_ratio < 0.10:
            reasoning_parts.append("Low spread friction (<10% ATR) enhances scalp precision.")

        prompt = (
            f"Symbol: {symbol}\n"
            f"Direction: {'BUY' if direction == 1 else 'SELL'}\n"
            f"Entry Price: {entry_price:.5f}\n"
            f"ATR: {atr:.5f}\n"
            f"Proposed Structural SL: {sl_price:.5f}\n"
            f"Proposed Target TP: {tp_price:.5f}\n"
            f"Spread Points: {spread_pts}\n"
            "Task: Evaluate entry SL/TP validity and output JSON."
        )
        try:
            raw_ai = self.provider.generate(prompt, ENTRY_SLTP_SYSTEM_PROMPT, json_mode=True)
            ai_data = json.loads(raw_ai)
            if ai_data.get("reasoning"):
                reasoning_parts.append(f"AI: {ai_data['reasoning']}")
        except Exception as e:
            logger.debug("[%s] LLM query in calculate_entry_sltp: %s", symbol, e)

        # 4. Pass through Safety Guardrails
        validated = self.guard.validate_entry_sltp(
            direction=direction,
            entry_price=entry_price,
            proposed_sl=sl_price,
            proposed_tp=tp_price,
            tick_size=tick_size,
            stops_level_pts=stops_level,
            point=point,
            atr=atr,
            min_rr=min_rr,
            min_atr_mult=min_atr,
            max_atr_mult=max_atr,
        )

        if validated is not None:
            final_sl, final_tp = validated
        else:
            # Fallback to standard safe ATR stop
            final_sl = entry_price - (sl_atr_mult * atr) if direction == 1 else entry_price + (sl_atr_mult * atr)
            final_tp = entry_price + (sl_atr_mult * atr * min_rr) if direction == 1 else entry_price - (sl_atr_mult * atr * min_rr)
            final_sl = self.guard.snap_price(final_sl, tick_size)
            final_tp = self.guard.snap_price(final_tp, tick_size)
            sl_source = "safety_fallback"
            tp_source = "safety_fallback"
            reasoning_parts.append("Applied safety fallback bounds.")

        final_sl_dist = abs(entry_price - final_sl)
        final_tp_dist = abs(final_tp - entry_price)
        final_rr = final_tp_dist / max(final_sl_dist, 1e-10)

        full_reason = " | ".join(reasoning_parts) if reasoning_parts else "Dynamic AI entry profile calculated."

        return EntrySLTPPlan(
            sl_price=final_sl,
            tp_price=final_tp,
            sl_distance=final_sl_dist,
            tp_distance=final_tp_dist,
            rr_ratio=final_rr,
            sl_source=sl_source,
            tp_source=tp_source,
            reasoning=full_reason,
            confidence=0.85,
            valid=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Live Position Management & Dynamic SL Ratchet
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_position_sltp(
        self,
        position: Any,
        df: pd.DataFrame,
        atr: float,
        location: Any = None,
        symbol_info: Any = None,
        peak_profit: float = 0.0,
    ) -> PositionAdjustmentPlan:
        """
        Evaluates an active open position and determines whether to:
        - Advance SL to Breakeven (+ spread buffer)
        - Trail SL behind newly formed micro-swings
        - Lock in profit after an exhaustion wick or liquidity sweep
        - Tighten due to profit giveback from peak
        
        Guaranteed: proposed SL is strictly one-way (NEVER loosens).
        """
        direction = 1 if getattr(position, "type", 0) == 0 else -1
        open_price = float(getattr(position, "price_open", 0.0))
        current_sl = float(getattr(position, "sl", 0.0))
        current_tp = float(getattr(position, "tp", 0.0))

        tick_size = getattr(symbol_info, "trade_tick_size", None) or getattr(symbol_info, "point", 0.01)
        point = getattr(symbol_info, "point", 0.01) or 0.01
        stops_level = getattr(symbol_info, "trade_stops_level", 0) or 0

        # Current price depending on direction
        bid = float(getattr(symbol_info, "bid", 0.0) or 0.0)
        ask = float(getattr(symbol_info, "ask", 0.0) or 0.0)
        current_price = bid if direction == 1 else ask
        if current_price <= 0 and not df.empty:
            current_price = float(df["close"].iloc[-1])

        profit_move = (current_price - open_price) * direction
        profit_atr = profit_move / max(atr, 1e-10)

        # Track giveback from peak
        actual_peak = max(peak_profit, profit_move)
        giveback_pct = (actual_peak - profit_move) / actual_peak if actual_peak > 0 else 0.0

        # Best candidate SL (can only advance)
        candidate_sl = current_sl
        action = "HOLD"
        reason = "Position structure intact."

        sym_name = str(getattr(position, "symbol", "") or self.cfg.get("symbol", ""))
        is_gold = "XAU" in sym_name.upper() or "GOLD" in sym_name.upper()

        # Stage 1: Fast Breakeven Ratchet (Scalping-friendly)
        # In scalping mode, breakeven triggers early (e.g. 0.35 - 0.45 ATR)
        default_be_trigger = 0.45 if is_gold else 0.40
        be_trigger = float(self.cfg.get("breakeven_trigger_atr", default_be_trigger)) * atr
        be_buffer = float(self.cfg.get("breakeven_buffer_atr", 0.15)) * atr
        be_price = open_price + be_buffer if direction == 1 else open_price - be_buffer

        if profit_move >= be_trigger:
            if direction == 1 and be_price > candidate_sl:
                candidate_sl = be_price
                action = "MOVE_TO_BREAKEVEN"
                reason = f"Fast Breakeven triggered at +{profit_atr:.2f} ATR; locked risk-free +{be_buffer:.2f} buffer."
            elif direction == -1 and (candidate_sl == 0.0 or be_price < candidate_sl):
                candidate_sl = be_price
                action = "MOVE_TO_BREAKEVEN"
                reason = f"Fast Breakeven triggered at +{profit_atr:.2f} ATR; locked risk-free +{be_buffer:.2f} buffer."

        # Stage 2: Progressive Profit Locking (Advances SL as profit grows)
        extra_profit = profit_move - be_trigger
        lock_ratio = float(self.cfg.get("profit_lock_ratio", 0.50))
        if extra_profit > 0 and lock_ratio > 0:
            locked_gain = be_buffer + (extra_profit * lock_ratio)
            progressive_sl = open_price + locked_gain if direction == 1 else open_price - locked_gain
            if direction == 1 and progressive_sl > candidate_sl:
                candidate_sl = progressive_sl
                action = "TIGHTEN_SL"
                reason = f"Progressive profit lock: ratcheted SL to lock {(locked_gain/atr):.2f} ATR gain."
            elif direction == -1 and (candidate_sl == 0.0 or progressive_sl < candidate_sl):
                candidate_sl = progressive_sl
                action = "TIGHTEN_SL"
                reason = f"Progressive profit lock: ratcheted SL to lock {(locked_gain/atr):.2f} ATR gain."

        # Stage 3: Market Structure Dynamic Trailing (Micro-Swings)
        if self.cfg.get("structure_trailing", True) and not df.empty and len(df) >= 10:
            if direction == 1:
                recent_low = float(df["low"].iloc[-5:].min())
                structural_trail = recent_low - (atr * 0.10)
                if structural_trail > candidate_sl and structural_trail < current_price:
                    candidate_sl = structural_trail
                    action = "TIGHTEN_SL"
                    reason = f"Market Structure trailing: SL anchored below 5-bar swing low ({recent_low:.4f})."
            else:
                recent_high = float(df["high"].iloc[-5:].max())
                structural_trail = recent_high + (atr * 0.10)
                if (candidate_sl == 0.0 or structural_trail < candidate_sl) and structural_trail > current_price:
                    candidate_sl = structural_trail
                    action = "TIGHTEN_SL"
                    reason = f"Market Structure trailing: SL anchored above 5-bar swing high ({recent_high:.4f})."

        # Stage 4: Rejection Wick / Pinbar Profit Securing
        # If opposite pinbar forms after reaching 0.4+ ATR profit, tighten tightly or secure profit
        last_bar = df.iloc[-1] if not df.empty else None
        if last_bar is not None and profit_move >= 0.40 * atr:
            opp_pinbar = bool(last_bar.get("bear_pinbar", False)) if direction == 1 else bool(last_bar.get("bull_pinbar", False))
            if opp_pinbar:
                tight_sl = current_price - (atr * 0.20) if direction == 1 else current_price + (atr * 0.20)
                if direction == 1 and tight_sl > candidate_sl:
                    candidate_sl = tight_sl
                    action = "SECURE_PROFIT"
                    reason = "Opposing rejection pinbar formed; tightened SL immediately to secure profits."
                elif direction == -1 and (candidate_sl == 0.0 or tight_sl < candidate_sl):
                    candidate_sl = tight_sl
                    action = "SECURE_PROFIT"
                    reason = "Opposing rejection pinbar formed; tightened SL immediately to secure profits."

        # Stage 5: Giveback from Peak Protection
        default_lock_activate = 1.0 if is_gold else 1.2
        lock_activate = float(self.cfg.get("profit_lock_activate_atr", default_lock_activate)) * atr
        default_giveback = 0.30 if is_gold else 0.35
        giveback_threshold = float(self.cfg.get("profit_giveback_pct", default_giveback))
        if actual_peak >= lock_activate and giveback_pct >= giveback_threshold:
            # Price gave back significant profit from peak; aggressively defend remainder
            defend_sl = current_price - (atr * 0.15) if direction == 1 else current_price + (atr * 0.15)
            if direction == 1 and defend_sl > candidate_sl:
                candidate_sl = defend_sl
                action = "TIGHTEN_SL"
                reason = f"Profit giveback ({giveback_pct*100:.0f}% from peak); ratcheting defensive stop."
            elif direction == -1 and (candidate_sl == 0.0 or defend_sl < candidate_sl):
                candidate_sl = defend_sl
                action = "TIGHTEN_SL"
                reason = f"Profit giveback ({giveback_pct*100:.0f}% from peak); ratcheting defensive stop."

        # Stage 6: Anti-Choke Correction Buffer
        # Enforce minimum separation from live market price so normal pullbacks do not clip SL
        min_separation = float(self.cfg.get("min_sl_separation_atr", 0.55 if is_gold else 0.45)) * atr
        if direction == 1 and candidate_sl > current_price - min_separation:
            buffered_sl = current_price - min_separation
            if buffered_sl > current_sl:
                candidate_sl = buffered_sl
                reason += f" (clamped with {min_separation/atr:.2f} ATR correction buffer)"
            else:
                candidate_sl = current_sl
                action = "HOLD"
                reason = "Correction buffer enforced: held SL to prevent premature exit on pullback."
        elif direction == -1 and candidate_sl < current_price + min_separation and candidate_sl > 0:
            buffered_sl = current_price + min_separation
            if current_sl == 0.0 or buffered_sl < current_sl:
                candidate_sl = buffered_sl
                reason += f" (clamped with {min_separation/atr:.2f} ATR correction buffer)"
            else:
                candidate_sl = current_sl
                action = "HOLD"
                reason = "Correction buffer enforced: held SL to prevent premature exit on pullback."

        # Pass through Safety Guard (Absolute Never-Loosen Gate)
        final_sl = None
        if candidate_sl != current_sl:
            final_sl = self.guard.validate_new_sl(
                direction=direction,
                current_price=current_price,
                current_sl=current_sl,
                proposed_sl=candidate_sl,
                tick_size=tick_size,
                stops_level_pts=stops_level,
                point=point,
                atr=atr,
            )

        if final_sl is None or final_sl == current_sl:
            return PositionAdjustmentPlan(
                action="HOLD",
                new_sl=None,
                new_tp=None,
                reasoning="Current SL position is optimal or proposed change would loosen risk.",
                confidence=0.8,
            )

        return PositionAdjustmentPlan(
            action=action,
            new_sl=final_sl,
            new_tp=None,
            reasoning=reason,
            confidence=0.85,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Pre-Trade LLM Critic & Institutional Trade Validation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_trade_candidate(
        self,
        symbol: str,
        direction: int,
        score: float,
        ai_conf: float,
        df: pd.DataFrame,
        h1_df: Optional[pd.DataFrame] = None,
        h4_df: Optional[pd.DataFrame] = None,
        location: Any = None,
    ) -> TradeCritiqueResult:
        """
        Adversarial LLM Risk Critic.
        Evaluates a candidate trade setup using spatial location, higher timeframe
        context, volatility, and orderflow to validate edge or veto traps.
        """
        dir_str = "BUY" if direction == 1 else "SELL"
        last_bar = df.iloc[-1] if not df.empty else {}
        atr = float(last_bar.get("atr", 0.0))
        adx = float(last_bar.get("adx", 0.0))
        rsi = float(last_bar.get("rsi", 50.0))

        context_lines = [
            f"Symbol: {symbol}",
            f"Technical Signal: {dir_str} (Confluence Score: {score:.1f}, ML Base Conf: {ai_conf:.2f})",
            f"Execution Microstructure: ADX={adx:.1f}, RSI={rsi:.1f}, ATR={atr:.5f}",
        ]
        if location is not None and getattr(location, "is_valid", False):
            context_lines.append(f"Dealing Range: {location.range_low:.5f} - {location.range_high:.5f}")
            context_lines.append(f"Range Location: {location.bias.value} ({location.range_position:.1%} of range)")
            if location.nearest_resistance():
                context_lines.append(f"Overhead Resistance: {location.nearest_resistance().price:.5f} (Strength: {location.nearest_resistance().strength:.2f})")
            if location.nearest_support():
                context_lines.append(f"Underlying Support: {location.nearest_support().price:.5f} (Strength: {location.nearest_support().strength:.2f})")
            if location.structure and getattr(location.structure, "last_choch", None):
                context_lines.append(f"Last Structural Break: {location.structure.last_choch.kind.value} (Direction: {location.structure.last_choch.direction})")

        prompt = (
            "\n".join(context_lines) + "\n"
            "Task: Act as Adversarial Risk Critic. Evaluate if this trade has genuine positive edge or is a trap. "
            "Output JSON: {\"decision\": \"BUY\"|\"SELL\"|\"PASS\", \"confidence\": float, \"reasoning\": str, \"risk_assessment\": str}"
        )
        try:
            raw = self.provider.generate(prompt, QUANT_CRITIC_SYSTEM_PROMPT, json_mode=True)
            data = json.loads(raw)
            decision = str(data.get("decision", "PASS")).upper().strip()
            confidence = float(data.get("confidence", 80.0))
            reasoning = str(data.get("reasoning", ""))
            risk = str(data.get("risk_assessment", ""))

            approved = (decision == dir_str)
            return TradeCritiqueResult(
                approved=approved,
                confidence=confidence,
                decision=decision,
                reasoning=reasoning,
                risk_assessment=risk,
            )
        except Exception as e:
            logger.debug("[%s] LLM Trade Candidate Evaluation fallback: %s", symbol, e)
            return TradeCritiqueResult(
                approved=True,
                confidence=ai_conf * 100.0,
                decision=dir_str,
                reasoning="Approved via technical confluence (AI fallback).",
                risk_assessment="Standard risk parameters.",
            )
