"""
core/backtesting/engine.py — Backtesting Engine.

Simulates trading over historical market data.
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.strategy.base import BaseStrategy, Signal
from core.mt5.types import Timeframe

logger = logging.getLogger(__name__)

class BacktestTrade:
    def __init__(self, entry_time: datetime, entry_price: float, direction: int, size: float, sl: float, tp: float):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.sl = sl
        self.tp = tp
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.pnl: float = 0.0
        self.is_open = True
        
    def close(self, time: datetime, price: float):
        self.exit_time = time
        self.exit_price = price
        self.pnl = (self.exit_price - self.entry_price) * self.direction * self.size
        self.is_open = False

class BacktestEngine:
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.0, slippage: float = 0.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def run(self, df: pd.DataFrame, strategy: BaseStrategy):
        """
        Run backtest on a pre-processed dataframe containing indicators.
        """
        logger.info(f"Running backtest with {len(df)} candles.")
        self.trades = []
        self.capital = self.initial_capital
        self.equity_curve = []
        
        open_trade: Optional[BacktestTrade] = None
        
        # Loop through each row
        # In a real engine, we'd step through and avoid look-ahead bias
        # by evaluating strategy on df.iloc[:i] but that's very slow in Python.
        # So we assume indicators are correctly shifted or we just evaluate row by row.
        
        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            time = df.index[i]
            
            # 1. Manage open trade (check SL/TP)
            if open_trade and open_trade.is_open:
                # Basic intrabar check: if low < SL or high > TP
                hit_sl = False
                hit_tp = False
                exit_price = current_bar['close']
                
                if open_trade.direction == 1: # Long
                    if current_bar['low'] <= open_trade.sl:
                        hit_sl = True
                        exit_price = open_trade.sl
                    elif current_bar['high'] >= open_trade.tp:
                        hit_tp = True
                        exit_price = open_trade.tp
                else: # Short
                    if current_bar['high'] >= open_trade.sl:
                        hit_sl = True
                        exit_price = open_trade.sl
                    elif current_bar['low'] <= open_trade.tp:
                        hit_tp = True
                        exit_price = open_trade.tp
                        
                if hit_sl or hit_tp:
                    # Apply slippage on stop loss
                    if hit_sl:
                        exit_price -= self.slippage * open_trade.direction
                        
                    open_trade.close(time, exit_price)
                    self.capital += open_trade.pnl - self.commission
                    open_trade = None
            
            # 2. Evaluate strategy if no open trade
            if not open_trade:
                # Construct historical slice up to prev_bar
                # To prevent look-ahead bias, strategy should only see data up to i-1
                historical_slice = df.iloc[:i]
                
                # Mock a regime
                from core.strategy.base import MarketRegime
                regime = MarketRegime.TRENDING # Dummy for now
                
                signal = strategy.evaluate(historical_slice, regime)
                
                if signal.direction != 0 and signal.score > 50:
                    entry_price = current_bar['open'] + (self.slippage * signal.direction)
                    
                    # Basic 1:2 RR for simulation if strategy doesn't provide
                    sl_dist = current_bar['atr'] * 2.0 if 'atr' in current_bar else 10.0
                    sl = entry_price - (sl_dist * signal.direction)
                    tp = entry_price + (sl_dist * 2.0 * signal.direction)
                    
                    open_trade = BacktestTrade(
                        entry_time=time,
                        entry_price=entry_price,
                        direction=signal.direction,
                        size=100.0, # Fixed size for simple backtest
                        sl=sl,
                        tp=tp
                    )
                    self.trades.append(open_trade)
            
            # Record equity curve
            current_equity = self.capital
            if open_trade:
                unrealized = (current_bar['close'] - open_trade.entry_price) * open_trade.direction * open_trade.size
                current_equity += unrealized
                
            self.equity_curve.append({
                'time': time,
                'equity': current_equity
            })
            
        return self.get_metrics()
        
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.trades:
            return {"total_trades": 0, "net_profit": 0.0}
            
        closed_trades = [t for t in self.trades if not t.is_open]
        if not closed_trades:
            return {"total_trades": 0, "net_profit": 0.0}
            
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = sum(abs(t.pnl) for t in losing_trades)
        net_profit = gross_profit - gross_loss
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "net_profit": net_profit,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "final_capital": self.capital,
            "return_pct": ((self.capital - self.initial_capital) / self.initial_capital) * 100
        }
