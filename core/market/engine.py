"""
core/market/engine.py — Market Data Engine.

Responsible for fetching multi-timeframe candles, caching them,
applying the technical indicator pipeline, and streaming ticks.
"""

import logging
import threading
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone
import pandas as pd

from core.mt5.adapter import MT5Adapter
from core.mt5.types import Timeframe, Tick
from core.market.pipeline import add_indicators

logger = logging.getLogger(__name__)

class MarketEngine:
    def __init__(self, adapter: MT5Adapter, symbols: List[str]):
        self.adapter = adapter
        self.symbols = symbols
        self._lock = threading.Lock()
        
        # Cache for indicator-enriched dataframes
        # Structure: _cache[symbol][timeframe] = DataFrame
        self._cache: Dict[str, Dict[Timeframe, pd.DataFrame]] = {
            s: {} for s in symbols
        }
        
        # Keep track of last bar fetch time to avoid spamming the broker
        self._last_fetch_time: Dict[str, Dict[Timeframe, float]] = {
            s: {} for s in symbols
        }

    def get_data(self, symbol: str, timeframe: Timeframe, count: int = 1000, 
                 force_fetch: bool = False, apply_pipeline: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol and timeframe.
        Will use cache if available and not force_fetch.
        """
        now = time.time()
        
        with self._lock:
            cached_df = self._cache.get(symbol, {}).get(timeframe)
            last_fetch = self._last_fetch_time.get(symbol, {}).get(timeframe, 0.0)
            
            # Simple caching: fetch if we don't have it, or if it's been more than 60s
            # In a real system, we might only fetch when a new bar closes, but this is a good start
            needs_fetch = force_fetch or cached_df is None or (now - last_fetch > 60.0)
            
            if not needs_fetch:
                return cached_df.copy()

        # Fetch from adapter (releases lock during network call)
        logger.debug(f"Fetching {count} {timeframe.to_string()} candles for {symbol}")
        df = self.adapter.get_candles(symbol, timeframe, count, drop_forming=True)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol} {timeframe.to_string()}")
            return df
            
        if apply_pipeline:
            df = add_indicators(df)
            
        with self._lock:
            if symbol not in self._cache:
                self._cache[symbol] = {}
            if symbol not in self._last_fetch_time:
                self._last_fetch_time[symbol] = {}
                
            self._cache[symbol][timeframe] = df
            self._last_fetch_time[symbol][timeframe] = time.time()
            
            return df.copy()

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Fetch the latest tick for a symbol."""
        return self.adapter.get_tick(symbol)

    def pre_warm_cache(self, timeframes: List[Timeframe], count: int = 1000):
        """Fetch data for all symbols and timeframes to warm up the cache."""
        logger.info("Warming up market data cache...")
        for symbol in self.symbols:
            for tf in timeframes:
                self.get_data(symbol, tf, count=count, force_fetch=True)
        logger.info("Cache warmup complete.")
