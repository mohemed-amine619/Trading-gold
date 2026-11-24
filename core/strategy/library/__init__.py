"""
core/strategy/library — the full strategy catalogue.

Every buildable strategy is a CAUSAL signal generator ``(df, atr, cfg) -> int8``
(+1 long, -1 short, 0 nothing) registered in ``LIBRARY_GENERATORS``. The scalping
engine (core/strategy/scalp.py) runs them through the same proof gate as its own
twelve: a strategy votes only while its replay over the recent bars shows a
statistically clear edge, with the bar raised for the number of strategies
tested at once.

``CATALOGUE`` maps every strategy name from the project brief to what it became:

  signal       a generator in this library (or one of the original twelve)
  sizing       a position-sizing / position-adding rule (core/risk/progression.py),
               not an entry signal
  existing     already covered by another part of the bot (the AI / NN votes)
  unavailable  needs data or infrastructure MT5 gold does not provide; listed
               with the reason instead of being faked

Names may be used in ``SCALP_STRATEGIES`` either as canonical names, as the
aliases below, or as group tokens: ``all``, ``classic`` (the original twelve) and
the category tokens in ``CATEGORIES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from core.strategy.library import events, indicators, patterns, price_action, sessions, smc, volume

CLASSIC = ("orb", "vwap_fade", "squeeze", "rsi2", "burst", "ema_pullback", "sweep",
           "breakout", "pinbar", "macd_cross", "stoch_cross", "inside_break")

LIBRARY_GENERATORS: Dict[str, object] = {}
for _mod in (price_action, smc, sessions, indicators, patterns, volume, events):
    for _name, _fn in _mod.GENERATORS.items():
        if _name in LIBRARY_GENERATORS or _name in CLASSIC:   # pragma: no cover - guard
            raise RuntimeError(f"duplicate strategy name: {_name}")
        LIBRARY_GENERATORS[_name] = _fn


@dataclass(frozen=True)
class Entry:
    category: str
    label: str            # the name as given in the brief
    kind: str             # signal / sizing / existing / unavailable
    target: Optional[str]  # canonical strategy (signal), rule (sizing) or component
    note: str = ""


def _e(cat, label, kind, target=None, note=""):
    return Entry(cat, label, kind, target, note)


_PROXY = "tick-volume proxy: MT5 gold has no exchange volume or trade prints"

CATALOGUE: Tuple[Entry, ...] = (
    # 1. price action
    _e("price_action", "Support & Resistance Trading", "signal", "sr_bounce"),
    _e("price_action", "Supply & Demand", "signal", "supply_demand"),
    _e("price_action", "Breakout Trading", "signal", "breakout"),
    _e("price_action", "Fakeout / False Breakout", "signal", "fakeout"),
    _e("price_action", "Retest Trading", "signal", "retest"),
    _e("price_action", "Trendline Trading", "signal", "trendline"),
    _e("price_action", "Channel Trading", "signal", "channel"),
    _e("price_action", "Range Trading", "signal", "range_trading"),
    _e("price_action", "Swing High / Swing Low", "signal", "swing_hl"),
    _e("price_action", "Candlestick Patterns", "signal", "candles_3bar"),
    _e("price_action", "Engulfing Strategy", "signal", "engulfing"),
    _e("price_action", "Pin Bar Strategy", "signal", "pinbar"),
    _e("price_action", "Inside Bar Strategy", "signal", "inside_break"),
    _e("price_action", "Outside Bar Strategy", "signal", "outside_bar"),
    _e("price_action", "Reversal Trading", "signal", "reversal"),
    _e("price_action", "Momentum Trading", "signal", "momentum"),
    # 2. SMC
    _e("smc", "Smart Money Concepts (SMC)", "signal", "smc", "composite: structure + discount + FVG"),
    _e("smc", "Inner Circle Trader (ICT)", "signal", "ict_2022", "umbrella term; mapped to the ICT 2022 model"),
    _e("smc", "Break of Structure (BOS)", "signal", "bos"),
    _e("smc", "Change of Character (CHOCH)", "signal", "choch"),
    _e("smc", "Market Structure Shift (MSS)", "signal", "mss"),
    _e("smc", "Order Blocks (OB)", "signal", "order_block"),
    _e("smc", "Breaker Blocks", "signal", "breaker"),
    _e("smc", "Mitigation Blocks", "signal", "mitigation"),
    _e("smc", "Fair Value Gap (FVG)", "signal", "fvg"),
    _e("smc", "Inverse Fair Value Gap (IFVG)", "signal", "ifvg"),
    _e("smc", "Liquidity Sweep", "signal", "sweep"),
    _e("smc", "Buy Side Liquidity (BSL)", "signal", "bsl_sweep"),
    _e("smc", "Sell Side Liquidity (SSL)", "signal", "ssl_sweep"),
    _e("smc", "Inducement", "signal", "inducement"),
    _e("smc", "Premium & Discount Zones", "signal", "premium_discount"),
    _e("smc", "Optimal Trade Entry (OTE)", "signal", "ote"),
    _e("smc", "Power of Three (PO3)", "signal", "po3"),
    _e("smc", "Judas Swing", "signal", "judas"),
    _e("smc", "Turtle Soup", "signal", "turtle_soup"),
    _e("smc", "Silver Bullet", "signal", "silver_bullet"),
    _e("smc", "Unicorn Model", "signal", "unicorn"),
    # 3. ICT time-based
    _e("ict", "London Kill Zone", "signal", "london_killzone"),
    _e("ict", "New York Kill Zone", "signal", "ny_killzone"),
    _e("ict", "Asian Session Range", "signal", "asian_range"),
    _e("ict", "London Open Strategy", "signal", "london_open"),
    _e("ict", "New York Open Strategy", "signal", "ny_open"),
    _e("ict", "London Close Reversal", "signal", "london_close_reversal"),
    _e("ict", "ICT 2022 Model", "signal", "ict_2022"),
    _e("ict", "ICT Macros", "signal", "ict_macros"),
    _e("ict", "ICT Weekly Profile", "signal", "weekly_profile"),
    _e("ict", "ICT Daily Bias", "signal", "daily_bias"),
    _e("ict", "ICT Midnight Open", "signal", "midnight_open"),
    _e("ict", "ICT Silver Bullet", "signal", "silver_bullet"),
    # 4. indicators
    _e("indicators", "RSI Strategy", "signal", "rsi"),
    _e("indicators", "MACD Crossover", "signal", "macd_cross"),
    _e("indicators", "Moving Average Crossover", "signal", "ma_cross"),
    _e("indicators", "EMA 9/21", "signal", "ema_9_21"),
    _e("indicators", "EMA 50/200", "signal", "ema_50_200"),
    _e("indicators", "SMA 20/50/200", "signal", "sma_stack"),
    _e("indicators", "Bollinger Bands", "signal", "bollinger"),
    _e("indicators", "Stochastic Oscillator", "signal", "stoch_cross"),
    _e("indicators", "ADX Trend Strategy", "signal", "adx_trend"),
    _e("indicators", "Ichimoku Cloud", "signal", "ichimoku"),
    _e("indicators", "Parabolic SAR", "signal", "psar"),
    _e("indicators", "ATR Breakout", "signal", "atr_breakout"),
    _e("indicators", "VWAP Strategy", "signal", "vwap_trend"),
    _e("indicators", "Fibonacci Retracement", "signal", "fib_retracement"),
    _e("indicators", "Fibonacci Extension", "signal", "fib_extension"),
    _e("indicators", "Pivot Points", "signal", "pivot_points"),
    _e("indicators", "Supertrend", "signal", "supertrend"),
    _e("indicators", "Donchian Channels", "signal", "donchian"),
    _e("indicators", "Keltner Channels", "signal", "keltner"),
    _e("indicators", "Heikin Ashi", "signal", "heikin_ashi"),
    # 5. scalping
    _e("scalping", "1-Minute Scalping", "signal", "ribbon_scalp",
       "a timeframe, not a rule: EMA ribbon scalp; runs on whatever bars the bot is fed"),
    _e("scalping", "5-Minute Scalping", "signal", "bb_rsi_scalp",
       "a timeframe, not a rule: Bollinger + RSI(7) scalp"),
    _e("scalping", "EMA Scalping", "signal", "ema_pullback"),
    _e("scalping", "RSI Scalping", "signal", "rsi2"),
    _e("scalping", "VWAP Scalping", "signal", "vwap_fade"),
    _e("scalping", "Order Flow Scalping", "signal", "orderflow_scalp", _PROXY),
    _e("scalping", "DOM Scalping", "unavailable", None,
       "needs a depth-of-market feed; MT5 CFD gold exposes no real order book"),
    _e("scalping", "Tape Reading", "unavailable", None,
       "needs time & sales prints with aggressor side; MT5 CFD gold has none"),
    _e("scalping", "Liquidity Scalping", "signal", "sweep"),
    _e("scalping", "FVG Scalping", "signal", "fvg"),
    _e("scalping", "Breakout Scalping", "signal", "squeeze"),
    _e("scalping", "Session Open Scalping", "signal", "orb"),
    _e("scalping", "Gold Scalping", "signal", "burst", "umbrella term; the whole set runs on XAUUSD"),
    _e("scalping", "Momentum Scalping", "signal", "burst"),
    # 6. day trading
    _e("daytrading", "Opening Range Breakout (ORB)", "signal", "orb"),
    _e("daytrading", "Initial Balance Trading", "signal", "initial_balance"),
    _e("daytrading", "Previous Day High/Low", "signal", "pdh_pdl"),
    _e("daytrading", "Previous Week High/Low", "signal", "pwh_pwl"),
    _e("daytrading", "Asian Range Breakout", "signal", "london_open"),
    _e("daytrading", "London Breakout", "signal", "london_breakout"),
    _e("daytrading", "New York Breakout", "signal", "ny_breakout"),
    _e("daytrading", "Gap Trading", "signal", "gap"),
    _e("daytrading", "VWAP Reversion", "signal", "vwap_fade"),
    _e("daytrading", "Trend Following Intraday", "signal", "trend_intraday"),
    _e("daytrading", "News Trading", "signal", "news_breakout"),
    _e("daytrading", "Liquidity Grab Trading", "signal", "pd_sweep"),
    # 7. swing
    _e("swing", "Trend Following", "signal", "trend_following"),
    _e("swing", "Pullback Trading", "signal", "pullback"),
    _e("swing", "Fibonacci Swing Trading", "signal", "fib_retracement"),
    _e("swing", "Moving Average Swing Trading", "signal", "ma_swing"),
    _e("swing", "Support/Resistance Swing", "signal", "sr_swing"),
    _e("swing", "Breakout Swing Trading", "signal", "donchian"),
    _e("swing", "Elliott Wave", "signal", "elliott_w3", "wave-3 entry only; full wave counting is subjective"),
    _e("swing", "Harmonic Patterns", "signal", "harmonic", "Gartley / Bat family D-zone"),
    _e("swing", "Head and Shoulders", "signal", "head_shoulders"),
    _e("swing", "Double Top / Double Bottom", "signal", "double_top_bottom"),
    _e("swing", "Cup and Handle", "signal", "cup_handle"),
    _e("swing", "Flag and Pennant", "signal", "flag"),
    _e("swing", "Triangle Patterns", "signal", "triangle"),
    _e("swing", "Divergence Trading", "signal", "divergence"),
    # 8. quant
    _e("quant", "Mean Reversion", "signal", "mean_reversion"),
    _e("quant", "Statistical Arbitrage", "unavailable", None,
       "needs a basket of co-moving instruments traded together; the bot trades one symbol"),
    _e("quant", "Pairs Trading", "unavailable", None,
       "needs a second instrument (e.g. XAGUSD) and simultaneous two-leg orders"),
    _e("quant", "Momentum Factor", "signal", "momentum_factor", "time-series momentum (single asset)"),
    _e("quant", "Trend Following Algorithms", "signal", "trend_algo"),
    _e("quant", "Grid Trading", "sizing", "grid", "adds at fixed ATR steps; OFF by default"),
    _e("quant", "Martingale", "sizing", "martingale", "OFF by default; capped steps"),
    _e("quant", "Anti-Martingale", "sizing", "anti_martingale", "OFF by default"),
    _e("quant", "DCA (Dollar Cost Averaging)", "sizing", "dca", "OFF by default"),
    _e("quant", "Kelly Criterion", "sizing", "kelly", "fractional Kelly; OFF by default"),
    _e("quant", "Machine Learning Trading", "existing", "nn", "the neural-network vote (core/ai/nn_*)"),
    _e("quant", "AI-Based Trading", "existing", "ai", "the AI ensemble vote in run.py"),
    _e("quant", "Reinforcement Learning", "unavailable", None,
       "needs a trading simulator and a training pipeline; not built"),
    _e("quant", "Market Making", "unavailable", None,
       "needs two-sided quoting at exchange; a retail CFD account pays the spread instead"),
    _e("quant", "Arbitrage Trading", "unavailable", None, "needs several venues/brokers at once"),
    _e("quant", "High-Frequency Trading (HFT)", "unavailable", None,
       "needs co-location and sub-millisecond execution; the bot loops once a second"),
    _e("quant", "Volatility Breakout", "signal", "vol_breakout"),
    _e("quant", "Regime Detection", "signal", "regime_switch"),
    # 9. volume / order flow
    _e("volume", "Volume Profile", "signal", "volume_profile", _PROXY),
    _e("volume", "Fixed Range Volume Profile", "signal", "frvp", _PROXY),
    _e("volume", "Market Profile / TPO", "signal", "tpo_80", "TPO is time-based: exact"),
    _e("volume", "Footprint Charts", "unavailable", None,
       "needs bid/ask volume per price level; MT5 CFD gold has no trade prints"),
    _e("volume", "Delta Analysis", "signal", "delta", _PROXY),
    _e("volume", "Cumulative Volume Delta (CVD)", "signal", "cvd_divergence", _PROXY),
    _e("volume", "Absorption Trading", "signal", "absorption", _PROXY),
    _e("volume", "Volume Imbalance", "signal", "volume_imbalance", "ICT body-gap definition: exact"),
    _e("volume", "VWAP Bands", "signal", "vwap_bands", _PROXY),
    _e("volume", "Point of Control (POC)", "signal", "poc", _PROXY),
    _e("volume", "Value Area High / Low", "signal", "vah_val", _PROXY),
    _e("volume", "Auction Market Theory", "signal", "auction", _PROXY),
    _e("volume", "Liquidity Heatmap", "unavailable", None, "needs order-book history"),
    _e("volume", "Order Book Imbalance", "unavailable", None, "needs a live order book"),
    # 10. events
    _e("events", "NFP Trading", "signal", "nfp", "dates generated by rule + calendar"),
    _e("events", "CPI Trading", "signal", "cpi", "dates from the news calendar only"),
    _e("events", "FOMC Trading", "signal", "fomc", "dates from the news calendar only"),
    _e("events", "Interest Rate Decisions", "signal", "rate_decision", "dates from the news calendar only"),
    _e("events", "News Breakout", "signal", "news_breakout"),
    _e("events", "News Reversal", "signal", "news_reversal"),
    _e("events", "Fundamental Analysis", "unavailable", None,
       "needs macro data feeds (real yields, USD index, flows); not connected"),
    _e("events", "Economic Calendar Trading", "signal", "news_breakout"),
    _e("events", "Sentiment Analysis", "unavailable", None, "needs a sentiment data source; not connected"),
    _e("events", "Geopolitical Event Trading", "unavailable", None,
       "events are not scheduled; needs a news feed and judgement"),
)

CATEGORIES: Tuple[str, ...] = ("price_action", "smc", "ict", "indicators", "scalping",
                               "daytrading", "swing", "quant", "volume", "events")


def _slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


ALIASES: Dict[str, str] = {}
for _entry in CATALOGUE:
    if _entry.kind == "signal":
        ALIASES.setdefault(_slug(_entry.label), _entry.target)


def category_members(category: str) -> Tuple[str, ...]:
    seen: List[str] = []
    for e in CATALOGUE:
        if e.category == category and e.kind == "signal" and e.target not in seen:
            seen.append(e.target)
    return tuple(seen)


def all_signal_names() -> Tuple[str, ...]:
    return CLASSIC + tuple(LIBRARY_GENERATORS)


def resolve(names: Iterable[str]) -> Tuple[str, ...]:
    """Expand group tokens and aliases into canonical, de-duplicated strategy names,
    in first-seen order. Unknown names are dropped (the engine ignores them anyway)."""
    known = set(all_signal_names())
    out: List[str] = []

    def add(n):
        if n in known and n not in out:
            out.append(n)

    for raw in names:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name == "all":
            for n in all_signal_names():
                add(n)
        elif name == "classic":
            for n in CLASSIC:
                add(n)
        elif name in CATEGORIES:
            for n in category_members(name):
                add(n)
        elif name in known:
            add(name)
        elif name in ALIASES:
            add(ALIASES[name])
    return tuple(out)
