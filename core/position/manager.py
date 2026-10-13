"""
core/position/manager.py — Position Management Engine.

Monitors open positions, reconciles local state with MT5, applies trailing stops, 
moves stops to breakeven, and protects/closes positions based on configured rules.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass, field

from core.mt5.adapter import MT5Adapter
from core.mt5.types import MT5Position
from core.execution.engine import ExecutionEngine
from core.strategy.base import Signal

from core.position.trailing import TrailingConfig, calculate_proposed_sl, validate_sl_modification
from core.position.decision import PNLDecisionEngine, PositionAction
from core.market.intelligence import MarketRegime

logger = logging.getLogger(__name__)

class ManualPolicy(str, Enum):
    OBSERVE_ONLY = "observe_only"
    PROTECT_ONLY = "protect_only"
    FULL_MANAGEMENT = "full_management"

@dataclass
class PositionState:
    ticket: int
    symbol: str
    direction: int
    volume: float
    magic: int
    policy: ManualPolicy
    invalidation_reason: Optional[str] = None
    is_invalidated: bool = False
    
class PositionManager:
    def __init__(self, adapter: MT5Adapter, execution_engine: ExecutionEngine, 
                 bot_magic: int = 888999, default_manual_policy: ManualPolicy = ManualPolicy.OBSERVE_ONLY,
                 dry_run: bool = False):
        self.adapter = adapter
        self.execution = execution_engine
        self.bot_magic = bot_magic
        self.default_manual_policy = default_manual_policy
        self.dry_run = dry_run
        
        # Local state cache mapping ticket -> PositionState
        self._state: Dict[int, PositionState] = {}
        
        # Configuration for trailing
        self.trail_activate_rr = 1.0  # Activate trailing stop when 1R in profit
        self.trail_distance_rr = 0.5  # Trail by 0.5R
        self.breakeven_rr = 0.5       # Move to breakeven when 0.5R in profit
        
        self.pnl_engine = PNLDecisionEngine(
            be_activation_rr=self.breakeven_rr,
            trail_activation_rr=self.trail_activate_rr,
            partial_tp_rr=2.0
        )
        
        from core.market.intelligence import MarketIntelligenceEngine
        self.intel_engine = MarketIntelligenceEngine()


    def set_manual_policy(self, ticket: int, policy: ManualPolicy):
        """Override the policy for a specific manual position."""
        if ticket in self._state:
            self._state[ticket].policy = policy
            logger.info(f"Manual policy for position {ticket} set to {policy.value}")

    def register_position(self, ticket: int, signal: Signal):
        """Link a newly opened bot position to its trading thesis."""
        # Wait for the next reconcile loop to pick it up, or add it manually.
        pass

    def reconcile(self) -> List[MT5Position]:
        """
        Reconcile local state with the actual broker state.
        Detects closed, partially closed, externally modified, and new positions.

        If the broker cannot be read, the account state is UNKNOWN — not
        empty. Propagating the error here is deliberate: the alternative was
        to delete every tracked position because a call timed out, which
        orphans live positions with no stop management.
        """
        broker_positions = self.adapter.get_positions()
        broker_tickets = {pos.ticket: pos for pos in broker_positions}
        
        # 1. Detect externally closed positions
        closed_tickets = [t for t in self._state if t not in broker_tickets]
        for t in closed_tickets:
            logger.info(f"Position {t} detected as closed externally.")
            del self._state[t]
            
        # 2. Detect new or modified positions
        for pos in broker_positions:
            if pos.ticket not in self._state:
                # New position detected
                is_bot = (pos.magic == self.bot_magic)
                policy = ManualPolicy.FULL_MANAGEMENT if is_bot else self.default_manual_policy
                
                self._state[pos.ticket] = PositionState(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    volume=pos.volume,
                    magic=pos.magic,
                    policy=policy
                )
                creator = "Bot" if is_bot else "Manual/External"
                logger.info(f"New {creator} position tracked: {pos.ticket} ({pos.symbol}, vol={pos.volume}, policy={policy.value})")
            else:
                # Modified position (partial close check)
                state = self._state[pos.ticket]
                if pos.volume != state.volume:
                    logger.info(f"Position {pos.ticket} volume changed externally from {state.volume} to {pos.volume}")
                    state.volume = pos.volume
                    
        return broker_positions

    def manage_positions(self, current_signal: Optional[Signal] = None, df: Optional[pd.DataFrame] = None):
        """
        Main loop to manage open positions safely.
        """
        broker_positions = self.reconcile()
        
        for pos in broker_positions:
            state = self._state.get(pos.ticket)
            if not state:
                continue
                
            if state.policy == ManualPolicy.OBSERVE_ONLY:
                continue
                
            self._manage_single_position(pos, state, current_signal, df)
            

    def _manage_single_position(self, pos: MT5Position, state: PositionState, current_signal: Optional[Signal], df: Optional[pd.DataFrame]):
        # 1. Check thesis invalidation (Only if FULL_MANAGEMENT)
        if state.policy == ManualPolicy.FULL_MANAGEMENT and current_signal:
            if current_signal.direction != 0 and current_signal.direction != pos.direction:
                if current_signal.score >= 60.0:
                    logger.info(f"Position {pos.ticket} thesis invalidated. Closing.")
                    if not self.dry_run:
                        self.execution.close_position(pos.ticket, pos.symbol, pos.volume, pos.direction)
                    else:
                        logger.info(f"[DRY RUN] Would close position {pos.ticket}")
                    state.is_invalidated = True
                    return
                    
        # 2. Apply Protect Rules (Trailing Stop and Breakeven)
        self._update_stops(pos, df)
            
    def _update_stops(self, pos: MT5Position, df: Optional[pd.DataFrame]):
        # Fetch symbol info and tick data
        info = self.adapter.get_symbol_info(pos.symbol)
        tick = self.adapter.get_tick(pos.symbol)
        
        if not info or not tick:
            logger.warning(f"Skipping SL update for {pos.ticket}. Missing symbol info or tick data.")
            return
            
        current_price_for_sl = tick.bid if pos.direction == 1 else tick.ask
        
        # PNL Engine Integration
        historical_high = pos.price_current # Mock historical high
        historical_low = pos.price_current  # Mock historical low
        if df is not None and not df.empty:
            historical_high = df['high'].max()
            historical_low = df['low'].min()
            
        metrics = self.pnl_engine.calculate_metrics(
            ticket=pos.ticket,
            open_price=pos.price_open,
            # Same snapshot the stop logic uses (see note at calculate_proposed_sl).
            current_price=current_price_for_sl,
            sl=pos.sl,
            initial_sl=pos.sl, # Fallback, ideally we store initial_sl in PositionState
            direction=pos.direction,
            volume=pos.volume,
            tick_value=info.trade_tick_value or 1.0,
            tick_size=info.trade_tick_size or 1e-5,
            historical_high=historical_high,
            historical_low=historical_low
        )
        
        regime = MarketRegime.UNCERTAIN
        if df is not None and not df.empty:
            regime_result = self.intel_engine.classify_regime(df)
            regime = regime_result.regime
            logger.info(f"Detected Market Regime: {regime.value} for {pos.symbol}")
            
        decision = self.pnl_engine.evaluate(metrics, regime)
        logger.info(f"Position {pos.ticket} recommendation: {decision.action.value} ({decision.rule_triggered})")
        
        if decision.action == PositionAction.CLOSE_POSITION:
            logger.info(f"Closing position {pos.ticket} based on PNL Decision Engine.")
            if not self.dry_run:
                self.execution.close_position(pos.ticket, pos.symbol, pos.volume, pos.direction)
            else:
                logger.info(f"[DRY RUN] Would close {pos.ticket}")
            return

        
        # Calculate ATR if available and mode is ATR
        atr_val = None
        if df is not None and not df.empty and "atr" in df.columns:
            atr_val = df.iloc[-1]["atr"]
            
        # Dynamically build TrailingConfig based on original RR preferences
        sl_dist = abs(pos.price_open - pos.sl)
        if sl_dist == 0:
            return # Can't RR trail without initial SL
            
        # Example configuration logic. Could be made configurable per strategy.
        config = TrailingConfig(
            mode="fixed", 
            activation_distance=sl_dist * self.trail_activate_rr,
            trailing_distance=sl_dist * self.trail_distance_rr,
            breakeven_activation=sl_dist * self.breakeven_rr,
            breakeven_offset=info.trade_tick_size * 2 # Slight offset
        )
        
        proposed_sl = calculate_proposed_sl(
            direction=pos.direction,
            price_open=pos.price_open,
            # Use the SAME price snapshot that the validation below will check
            # against. These previously came from different sources (the tick
            # vs pos.price_current), so a stop could be proposed against one
            # price and judged against another — which is how a stop above the
            # market can pass a "never loosen" test.
            price_current=current_price_for_sl,
            current_sl=pos.sl,
            config=config,
            atr_value=atr_val
        )
        
        valid_sl = validate_sl_modification(
            direction=pos.direction,
            proposed_sl=proposed_sl,
            current_sl=pos.sl,
            current_price_for_sl=current_price_for_sl,
            tick_size=info.trade_tick_size,
            point=info.point,
            stops_level=info.trade_stops_level,
            freeze_level=getattr(info, "trade_freeze_level", 0)
        )
        
        if valid_sl is not None:
            logger.info(f"Proposing valid SL update for {pos.ticket}. Old: {pos.sl}, New: {valid_sl}")
            if not self.dry_run:
                res = self.execution.modify_position(pos.ticket, pos.symbol, valid_sl, pos.tp)
                if res and res.success:
                    logger.info(f"Successfully modified SL for {pos.ticket}")
                else:
                    logger.error(f"Failed to modify SL for {pos.ticket}: {res.comment if res else 'Unknown error'}")
            else:
                logger.info(f"[DRY RUN] Would modify SL for {pos.ticket} to {valid_sl}")

    def get_total_floating_pnl(self, df: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate real-time floating PNL for all managed positions 
        using the PNLDecisionEngine.
        """
        total_pnl = 0.0
        broker_positions = self.adapter.get_positions()
        for pos in broker_positions:
            state = self._state.get(pos.ticket)
            if not state or state.policy == ManualPolicy.OBSERVE_ONLY:
                continue
                
            info = self.adapter.get_symbol_info(pos.symbol)
            if not info:
                continue
                
            historical_high = pos.price_current
            historical_low = pos.price_current
            if df is not None and not df.empty:
                historical_high = df['high'].max()
                historical_low = df['low'].min()
                
            metrics = self.pnl_engine.calculate_metrics(
                ticket=pos.ticket,
                open_price=pos.price_open,
                current_price=pos.price_current,
                sl=pos.sl,
                initial_sl=pos.sl,
                direction=pos.direction,
                volume=pos.volume,
                tick_value=info.trade_tick_value or 1.0,
                tick_size=info.trade_tick_size or 1e-5,
                historical_high=historical_high,
                historical_low=historical_low
            )
            total_pnl += metrics.floating_pnl
            
        return total_pnl
