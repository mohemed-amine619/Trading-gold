import math
from typing import Optional
from dataclasses import dataclass

@dataclass
class TrailingConfig:
    mode: str = "fixed"  # "fixed", "atr", "profit_locking"
    # Fixed distance
    activation_distance: float = 0.0  # In price distance
    trailing_distance: float = 0.0
    # Breakeven
    breakeven_activation: float = 0.0
    breakeven_offset: float = 0.0
    # Profit Locking
    profit_locking_ratio: float = 0.0  # e.g., 0.5 to lock 50% of profit

def normalize_price(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size

def calculate_proposed_sl(
    direction: int,
    price_open: float,
    price_current: float,
    current_sl: float,
    config: TrailingConfig,
    atr_value: Optional[float] = None
) -> float:
    """Pure calculation of proposed SL based on trailing rules."""
    profit = (price_current - price_open) * direction
    proposed_sl = current_sl

    # 1. Breakeven & Progressive Profit Locking
    if config.breakeven_activation > 0 and profit >= config.breakeven_activation:
        extra_profit = max(0.0, profit - config.breakeven_activation)
        locked_offset = config.breakeven_offset + (extra_profit * max(0.0, config.profit_locking_ratio))
        be_price = price_open + (locked_offset * direction)
        if _is_better_sl(direction, be_price, proposed_sl):
            proposed_sl = be_price

    # 2. Fixed distance trailing
    if config.mode == "fixed" and config.activation_distance > 0 and config.trailing_distance > 0:
        if profit >= config.activation_distance:
            trail_sl = price_current - (config.trailing_distance * direction)
            if _is_better_sl(direction, trail_sl, proposed_sl):
                proposed_sl = trail_sl

    # 3. ATR trailing
    elif config.mode == "atr" and atr_value is not None and atr_value > 0 and config.trailing_distance > 0:
        # trailing_distance acts as a multiplier here
        trail_dist = config.trailing_distance * atr_value
        if profit >= trail_dist:
            trail_sl = price_current - (trail_dist * direction)
            if _is_better_sl(direction, trail_sl, proposed_sl):
                proposed_sl = trail_sl
                
    # 4. Profit Locking
    elif config.mode == "profit_locking" and config.profit_locking_ratio > 0:
        if profit > 0:
            locked_profit = profit * config.profit_locking_ratio
            lock_sl = price_open + (locked_profit * direction)
            if _is_better_sl(direction, lock_sl, proposed_sl):
                proposed_sl = lock_sl

    return proposed_sl

def validate_sl_modification(
    direction: int,
    proposed_sl: float,
    current_sl: float,
    current_price_for_sl: float,
    tick_size: float,
    point: float,
    stops_level: int,
    freeze_level: int
) -> Optional[float]:
    """
    Validates and normalizes the proposed SL against MT5 constraints.
    Returns the normalized valid SL, or None if the modification is invalid/rejected.
    """
    # 1. Normalize
    norm_sl = normalize_price(proposed_sl, tick_size)
    
    # Check if anything changed after normalization
    if abs(norm_sl - current_sl) < (tick_size / 2.0):
        return None

    # 2. Never loosen an existing stop loss
    if not _is_better_sl(direction, norm_sl, current_sl):
        return None

    # 3. Stops level & Freeze level check
    # SL distance must be >= max(stops_level, freeze_level) * point
    min_dist = max(stops_level, freeze_level) * point
    actual_dist = abs(current_price_for_sl - norm_sl)
    
    # Strict MT5 rule: distance from execution price to stop must be >= min_dist
    # Note: For BUY position, SL is triggered by BID. So we must use BID to check distance.
    # For SELL position, SL is triggered by ASK.
    # We assume current_price_for_sl is the correct trigger price.
    
    # Due to floating point imprecision, we add a tiny tolerance
    if actual_dist < (min_dist - point/10.0):
        return None

    return norm_sl

def _is_better_sl(direction: int, proposed: float, current: float) -> bool:
    """Returns True if 'proposed' is a tighter/better stop than 'current'."""
    if current == 0.0:
        return True # any SL is better than no SL
    if direction == 1:
        return proposed > current
    else:
        return proposed < current
