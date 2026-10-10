"""
core/market/pipeline.py — Technical indicator pipeline.

This module ports the indicator calculations from the old run.py script
into a robust pipeline that processes OHLCV DataFrames.
"""

import logging
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach core technical indicators to an OHLCV DataFrame.
    
    Required columns: time (index), open, high, low, close, tick_volume
    """
    df = df.copy()
    cl, hi, lo = df["close"], df["high"], df["low"]
    
    # 1. Moving Averages
    df["ema50"] = ta.ema(cl, length=50)
    df["ema200"] = ta.ema(cl, length=200)
    
    # 2. Oscillators & Volatility
    df["rsi"] = ta.rsi(cl, length=14)
    df["atr"] = ta.atr(hi, lo, cl, length=14)
    
    # 3. MACD
    try:
        macd = ta.macd(cl, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            h = next((c for c in macd.columns if "MACDh" in c), None)
            s = next((c for c in macd.columns if "MACDs" in c), None)
            m = next((c for c in macd.columns if c.startswith("MACD_")), None)
            
            df["macd_hist"] = macd[h] if h else 0.0
            df["macd_signal"] = macd[s] if s else 0.0
            df["macd_line"] = macd[m] if m else 0.0
            df["macd_hist_prev"] = df["macd_hist"].shift(1)
        else:
            _fallback_macd(df)
    except Exception as e:
        logger.warning(f"MACD calculation failed: {e}")
        _fallback_macd(df)
        
    # 4. Stochastic
    try:
        stoch = ta.stoch(hi, lo, cl, k=14, d=3)
        if stoch is not None and not stoch.empty:
            kc = next((c for c in stoch.columns if "STOCHk" in c), None)
            dc = next((c for c in stoch.columns if "STOCHd" in c), None)
            df["stoch_k"] = stoch[kc] if kc else 50.0
            df["stoch_d"] = stoch[dc] if dc else 50.0
        else:
            df["stoch_k"] = df["stoch_d"] = 50.0
    except Exception as e:
        logger.warning(f"Stochastic calculation failed: {e}")
        df["stoch_k"] = df["stoch_d"] = 50.0
        
    # 5. ADX
    try:
        adx_df = ta.adx(hi, lo, cl, length=14)
        if adx_df is not None and not adx_df.empty:
            ac = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
            pc = next((c for c in adx_df.columns if c.startswith("DMP_")), None)
            nc = next((c for c in adx_df.columns if c.startswith("DMN_")), None)
            
            df["adx"] = adx_df[ac] if ac else 0.0
            df["di_plus"] = adx_df[pc] if pc else 0.0
            df["di_minus"] = adx_df[nc] if nc else 0.0
        else:
            _fallback_adx(df)
    except Exception as e:
        logger.warning(f"ADX calculation failed: {e}")
        _fallback_adx(df)
        
    # 6. Volume
    df["vol_sma20"] = df.get("tick_volume", pd.Series([0.0]*len(df))).rolling(20).mean()
    df["vol_ratio"] = df.get("tick_volume", pd.Series([0.0]*len(df))) / (df["vol_sma20"] + 1e-10)
    
    # 7. Structure
    df["_high20"] = df["high"].rolling(20).max()
    df["_low20"] = df["low"].rolling(20).min()
    
    return df

def _fallback_macd(df: pd.DataFrame):
    df["macd_hist"] = 0.0
    df["macd_signal"] = 0.0
    df["macd_line"] = 0.0
    df["macd_hist_prev"] = 0.0

def _fallback_adx(df: pd.DataFrame):
    df["adx"] = 0.0
    df["di_plus"] = 0.0
    df["di_minus"] = 0.0
