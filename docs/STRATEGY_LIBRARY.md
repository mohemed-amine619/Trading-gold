# Strategy library

All 151 strategies from the brief, and what each one became in the code.
Generated from `core/strategy/library/__init__.py` (`CATALOGUE`); the tests in
`tests/unit/test_strategy_library.py` check that this list and the code agree.

## How a strategy gets to trade

1. Every strategy is a **causal signal generator** (`+1` buy / `-1` sell / `0`), run on
   the bot's M15 frame. Swings are only used once confirmed, session ranges only once
   the session is over. Each one is tested for look-ahead.
2. It may **vote only once it is proven**: replayed over the last `SCALP_CHECK_BARS`
   bars, its calls must win more than they lose against a ±1 ATR first-touch barrier,
   over at least `SCALP_MIN_CALLS` independent calls, by a significance of `min_z`.
3. With `SCALP_AUTO_Z=true` (default) `min_z` rises with the number of strategies
   tested at once: 2.7σ for the original set, **3.52σ for all 118**.
   Measured on 40 random walks: one walk produced one false "proven" strategy (≈ 2.5%
   expected by design).
4. Proven strategies with a live signal feed the existing scalping score component
   (`SCALP_WEIGHT` points). Every other gate (risk, news blackout, spread, NN, location
   veto) still applies.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `SCALP_STRATEGIES` | `all` | names, aliases or groups: `all`, `classic`, `price_action`, `smc`, `ict`, `indicators`, `scalping`, `daytrading`, `swing`, `quant`, `volume`, `events` |
| `SCALP_AUTO_Z` | `true` | raise the significance bar for the number of strategies |
| `SCALP_SERVER_UTC_OFFSET` | `0` | hours the broker chart clock is ahead of UTC (Exness = 0) for New York session times |
| `SCALP_CHECK_BARS` | `900` | bars replayed to prove a strategy |
| `HISTORY_CANDLES` | `600` | bars the bot loads; **the proof window cannot be longer than this** |

**Once-a-day strategies need much more history.** Session strategies (kill zones,
opens, ORB, PDH/PDL, Judas, PO3, news...) fire at most once a day, so 600 M15 bars
(~6 days) can never reach 15 independent calls. To let them prove themselves set
`HISTORY_CANDLES=4500` and `SCALP_CHECK_BARS=4000` (≈ 2 months of M15), or test them
offline first:

```powershell
.venv\Scripts\python.exe scripts\export_bars.py --symbol XAUUSDm --timeframe M15 --days 365 --out data\history\XAUUSDm_M15.csv
.venv\Scripts\python.exe scripts\evaluate_scalp.py --csv data\history\XAUUSDm_M15.csv --strategies all --window 4000
.venv\Scripts\python.exe scripts\evaluate_scalp.py --csv data\history\XAUUSDm_M15.csv --strategies smc,ict --window 4000
```

## Sizing rules (not entry signals) — OFF by default

| Variable | Default | |
|---|---|---|
| `SIZING_PROGRESSION` | `off` | `martingale`, `anti_martingale` or `kelly` |
| `PROGRESSION_FACTOR` / `PROGRESSION_MAX_STEPS` / `PROGRESSION_MAX_MULT` | 2.0 / 3 / 4.0 | martingale x2 per loss, at most 3 steps, never above x4 |
| `KELLY_FRACTION` / `KELLY_MIN_TRADES` | 0.25 / 30 | quarter Kelly, after 30 closed trades |
| `GRID_MODE` | `off` | `grid` (equal adds) or `dca` (scaled adds) |
| `GRID_SPACING_ATR` / `GRID_MAX_LEVELS` / `DCA_SCALE` | 1.0 / 3 / 1.5 | add every 1 ATR against, at most 3 adds |
| `GRID_MAX_TOTAL_RISK` | 0.03 | refuse an add if all legs stopped together would lose > 3% of equity |

Every grid/DCA add **uses the first position's stop-loss and take-profit**: nothing is
opened without a stop and no stop is widened. Martingale with a stop on every trade
still loses `risk x (1 + 2 + 4 + 8)` = 15x the base risk over four straight losses.

## Catalogue

Kinds: **signal** = a generator that can vote · **sizing** = a position-size / add rule ·
**existing** = already another part of the bot · **unavailable** = needs data MT5 gold
does not provide (listed instead of faked).


### 1. Price action

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| Support & Resistance Trading | signal | `sr_bounce` | Support & Resistance: first rejection of the latest confirmed swing level. |
| Supply & Demand | signal | `supply_demand` | Supply & Demand: a tight base (2 small bars) left by a strong impulse; trade the first return into the base. |
| Breakout Trading | signal | `breakout` |  |
| Fakeout / False Breakout | signal | `fakeout` | False breakout: the previous bar CLOSED beyond a swing level, this bar closes back inside -> trade the failure. |
| Retest Trading | signal | `retest` | Break and retest: a close through a swing level, then the first pull-back to it that holds. |
| Trendline Trading | signal | `trendline` | Trendline: a line through the last two rising swing lows (falling swing highs); buy (sell) the first touch that holds. |
| Channel Trading | signal | `channel` | Channel: a rising (falling) 50-bar regression channel; buy the lower band (sell the upper band) with the slope. |
| Range Trading | signal | `range_trading` | Range trading: a flat 60-bar box 3-10 ATR tall; fade its edges. |
| Swing High / Swing Low | signal | `swing_hl` | Swing high / swing low: a higher low confirmed inside a higher-high structure (or a lower high in a lower-low structure). |
| Candlestick Patterns | signal | `candles_3bar` | Candlestick patterns: morning/evening star and three soldiers/crows. |
| Engulfing Strategy | signal | `engulfing` | Engulfing after a 3-bar move the other way. |
| Pin Bar Strategy | signal | `pinbar` |  |
| Inside Bar Strategy | signal | `inside_break` |  |
| Outside Bar Strategy | signal | `outside_bar` | Outside bar: engulfs the prior range, is at least 1 ATR tall and closes at an extreme. |
| Reversal Trading | signal | `reversal` | Reversal: price stretched 2.5 ATR from EMA20, then a bar that closes through the previous bar's extreme the other way. |
| Momentum Trading | signal | `momentum` | Momentum: 10-bar rate of change crosses 2 ATR with ADX above 25. |

### 2. Smart Money Concepts (SMC)

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| Smart Money Concepts (SMC) | signal | `smc` | SMC composite: bullish structure + price in the discount half of the last swing range + a fresh bullish FVG on this bar (mirrored for shorts). — composite: structure + discount + FVG |
| Inner Circle Trader (ICT) | signal | `ict_2022` | ICT 2022 model: a sweep of the last swing low (high), then within 12 bars a displacement that leaves an FVG the other way; trade the FVG retest. 02-12 NY. — umbrella term; mapped to the ICT 2022 model |
| Break of Structure (BOS) | signal | `bos` | Break of Structure: a close through the last swing in the direction of the trend. |
| Change of Character (CHOCH) | signal | `choch` | Change of Character: the first close through a swing AGAINST the prior trend. |
| Market Structure Shift (MSS) | signal | `mss` | Market Structure Shift: a CHOCH made by a displacement candle (>= 1.5 ATR) after the last swing on the other side swept the one before it. |
| Order Blocks (OB) | signal | `order_block` | Order block: the last opposite candle before a break of structure; trade its first retest. |
| Breaker Blocks | signal | `breaker` | Breaker block: an order block that price closed through flips sides. |
| Mitigation Blocks | signal | `mitigation` | Mitigation block: the order block of a CHOCH that came from a FAILURE swing (a lower high before a bearish break, a higher low before a bullish one). |
| Fair Value Gap (FVG) | signal | `fvg` | Fair Value Gap: trade the first retrace into a fresh 3-bar imbalance. |
| Inverse Fair Value Gap (IFVG) | signal | `ifvg` | Inverse FVG: a gap that price closed through, traded from the other side. |
| Liquidity Sweep | signal | `sweep` |  |
| Buy Side Liquidity (BSL) | signal | `bsl_sweep` | Buy-side liquidity: a wick above equal highs or the previous day's high that closes back below -> short. |
| Sell Side Liquidity (SSL) | signal | `ssl_sweep` | Sell-side liquidity: a wick below equal lows or the previous day's low that closes back above -> long. |
| Inducement | signal | `inducement` | Inducement: inside a bullish structure, price sweeps the latest MINOR swing low (1-bar pivot) but holds above the major one, then closes back up. |
| Premium & Discount Zones | signal | `premium_discount` | Premium & discount: in a bullish structure, buy a bar that closes above the prior high while in the discount half of the 50-bar dealing range (mirrored). |
| Optimal Trade Entry (OTE) | signal | `ote` | Optimal Trade Entry: a retrace into the 62-79% zone of the last impulse leg (>= 2 ATR) that closes back in the leg's direction. |
| Power of Three (PO3) | signal | `po3` | Power of Three: accumulation (Asian range), manipulation (London sweeps one side), distribution (close back inside -> trade toward the other side). Once a day. |
| Judas Swing | signal | `judas` | Judas swing: between 00:00 and 05:00 New York the market runs above the midnight open and the Asian high, then closes back below the midnight open. |
| Turtle Soup | signal | `turtle_soup` | Turtle soup: a new 20-bar low whose previous low is at least 4 bars old, closed back above that old low (mirrored). |
| Silver Bullet | signal | `silver_bullet` | ICT Silver Bullet: an FVG formed inside 03-04, 10-11 or 14-15 New York, traded on its retest within the hour. |
| Unicorn Model | signal | `unicorn` | Unicorn model: a breaker block overlapping a same-direction FVG formed within 10 bars of it; trade the retest of the overlap. |

### 3. ICT time-based

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| London Kill Zone | signal | `london_killzone` | London kill zone (02-05 NY): fade a sweep of the Asian range. |
| New York Kill Zone | signal | `ny_killzone` | New York kill zone (07-10 NY): fade a sweep of the London session range. |
| Asian Session Range | signal | `asian_range` | Asian range: before London (00-02 NY) fade a test of the finished Asian range edges. |
| London Open Strategy | signal | `london_open` | London open / Asian range breakout: first close beyond the Asian range, 02-05 NY. |
| New York Open Strategy | signal | `ny_open` | New York open: first close beyond the 09:30-10:00 opening range, 10:00-12:00 NY. |
| London Close Reversal | signal | `london_close_reversal` | London close (10-12 NY): after a day move of 1.5 ATR from the midnight open, fade it on a close through the previous bar's extreme. |
| ICT 2022 Model | signal | `ict_2022` | ICT 2022 model: a sweep of the last swing low (high), then within 12 bars a displacement that leaves an FVG the other way; trade the FVG retest. 02-12 NY. |
| ICT Macros | signal | `ict_macros` | ICT macros: a displacement close beyond the prior 4-bar range during one of the algorithmic 20-30 minute windows. |
| ICT Weekly Profile | signal | `weekly_profile` | ICT weekly profile: early in the week (Mon-Wed) price sweeps the previous week's low (high) and closes back inside -> the week's low (high) is likely in. |
| ICT Daily Bias | signal | `daily_bias` | ICT daily bias: with the previous day's bias, buy (sell) the first rejection of the previous day's midpoint inside the London or New York kill zone. |
| ICT Midnight Open | signal | `midnight_open` | ICT midnight open: with a rising one-day EMA, buy a 07-11 NY close back above the midnight open after trading below it (mirrored). |
| ICT Silver Bullet | signal | `silver_bullet` | ICT Silver Bullet: an FVG formed inside 03-04, 10-11 or 14-15 New York, traded on its retest within the hour. |

### 4. Technical indicators

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| RSI Strategy | signal | `rsi` | RSI(14) crosses back up through 30 (down through 70). |
| MACD Crossover | signal | `macd_cross` |  |
| Moving Average Crossover | signal | `ma_cross` | Moving-average crossover: SMA 10 / SMA 30. |
| EMA 9/21 | signal | `ema_9_21` | EMA 9 / EMA 21 crossover. |
| EMA 50/200 | signal | `ema_50_200` | EMA 50 / EMA 200 golden / death cross. |
| SMA 20/50/200 | signal | `sma_stack` | SMA 20/50/200 stacked; buy the pull-back to the SMA20 that closes back above it. |
| Bollinger Bands | signal | `bollinger` | Bollinger(20, 2) reversion: a close back inside after a close outside. |
| Stochastic Oscillator | signal | `stoch_cross` |  |
| ADX Trend Strategy | signal | `adx_trend` | ADX trend: +DI crosses -DI while ADX(14) is above 25. |
| Ichimoku Cloud | signal | `ichimoku` | Ichimoku: Tenkan/Kijun cross on the right side of the cloud (cloud shifted 26 bars forward, i.e. built only from data 26 bars old). |
| Parabolic SAR | signal | `psar` | Parabolic SAR flip. |
| ATR Breakout | signal | `atr_breakout` | ATR breakout: a close-to-close move larger than 1.2 x the previous bar's ATR. |
| VWAP Strategy | signal | `vwap_trend` | VWAP: a close crossing the session VWAP in the direction of its 4-bar slope. |
| Fibonacci Retracement | signal | `fib_retracement` | Fibonacci retracement: a pull-back into 50-61.8% of the last major leg (>= 3 ATR) that closes back in the leg's direction. |
| Fibonacci Extension | signal | `fib_extension` | Fibonacci extension: exhaustion at the 1.618 extension of the last leg (A->B, projected from C) on a rejection candle -> fade. |
| Pivot Points | signal | `pivot_points` | Classic daily pivots: rejection of S1 (buy) or R1 (sell). |
| Supertrend | signal | `supertrend` | Supertrend (ATR 14 x 3) flip. |
| Donchian Channels | signal | `donchian` | Donchian(55) breakout (the Turtle rule). |
| Keltner Channels | signal | `keltner` | Keltner (EMA20 +/- 2 ATR): a close breaking out of the channel. |
| Heikin Ashi | signal | `heikin_ashi` | Heikin Ashi: after 3+ opposite candles, a flat-bottomed (flat-topped) HA candle. |

### 5. Scalping

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| 1-Minute Scalping | signal | `ribbon_scalp` | 1-minute style ribbon scalp: EMA 5/8/13 fanned; buy a tag of EMA8 that closes above EMA5. — a timeframe, not a rule: EMA ribbon scalp; runs on whatever bars the bot is fed |
| 5-Minute Scalping | signal | `bb_rsi_scalp` | 5-minute style scalp: a wick outside Bollinger(20,2) with RSI(7) < 25 (> 75) closing in the other direction. — a timeframe, not a rule: Bollinger + RSI(7) scalp |
| EMA Scalping | signal | `ema_pullback` |  |
| RSI Scalping | signal | `rsi2` |  |
| VWAP Scalping | signal | `vwap_fade` |  |
| Order Flow Scalping | signal | `orderflow_scalp` | Order-flow scalping (proxy): three bars in a row whose delta and close agree, with rising tick volume, closing beyond the 3-bar range. — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| DOM Scalping | unavailable | — | needs a depth-of-market feed; MT5 CFD gold exposes no real order book |
| Tape Reading | unavailable | — | needs time & sales prints with aggressor side; MT5 CFD gold has none |
| Liquidity Scalping | signal | `sweep` |  |
| FVG Scalping | signal | `fvg` | Fair Value Gap: trade the first retrace into a fresh 3-bar imbalance. |
| Breakout Scalping | signal | `squeeze` |  |
| Session Open Scalping | signal | `orb` |  |
| Gold Scalping | signal | `burst` | umbrella term; the whole set runs on XAUUSD |
| Momentum Scalping | signal | `burst` |  |

### 6. Day trading

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| Opening Range Breakout (ORB) | signal | `orb` |  |
| Initial Balance Trading | signal | `initial_balance` | Initial balance: first close beyond the 09:30-10:30 NY range, until 14:00 NY. |
| Previous Day High/Low | signal | `pdh_pdl` | Previous day high/low breakout: first close beyond yesterday's high or low. |
| Previous Week High/Low | signal | `pwh_pwl` | Previous week high/low breakout: first close beyond last week's high or low. |
| Asian Range Breakout | signal | `london_open` | London open / Asian range breakout: first close beyond the Asian range, 02-05 NY. |
| London Breakout | signal | `london_breakout` | London breakout: first close beyond London's first hour (03-04 NY), until 08:00. |
| New York Breakout | signal | `ny_breakout` | New York breakout: first close beyond the 08:00-09:00 NY range, until 12:00. |
| Gap Trading | signal | `gap` | Gap trading: the week's opening gap (>= 0.5 ATR from Friday's close) is faded on the first bar that closes toward the fill within the first 8 bars. |
| VWAP Reversion | signal | `vwap_fade` |  |
| Trend Following Intraday | signal | `trend_intraday` | Intraday trend: above a rising session VWAP, buy a pull-back to EMA20 that closes back above it (mirrored). |
| News Trading | signal | `news_breakout` | News breakout: any high-impact release, first close beyond the pre-release hour. |
| Liquidity Grab Trading | signal | `pd_sweep` | Liquidity grab: a wick through yesterday's high/low that closes back inside -> fade. |

### 7. Swing trading

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| Trend Following | signal | `trend_following` | Trend following: EMA20 > EMA50 > EMA100 with ADX > 20; buy the close back above EMA20 after a close below it. |
| Pullback Trading | signal | `pullback` | Pull-back: above a rising EMA50, after three lower highs, a close above the previous bar's high (mirrored). |
| Fibonacci Swing Trading | signal | `fib_retracement` | Fibonacci retracement: a pull-back into 50-61.8% of the last major leg (>= 3 ATR) that closes back in the leg's direction. |
| Moving Average Swing Trading | signal | `ma_swing` | Moving-average swing: a close crossing SMA50 while SMA50 slopes the same way. |
| Support/Resistance Swing | signal | `sr_swing` | Support/Resistance swing: the same rejection on major (5-bar) swing levels. |
| Breakout Swing Trading | signal | `donchian` | Donchian(55) breakout (the Turtle rule). |
| Elliott Wave | signal | `elliott_w3` | Elliott wave 3 entry: wave 1 up (>= 2 ATR), wave 2 retraces 38-79% without breaking the start, then the close above the wave-1 top starts wave 3. — wave-3 entry only; full wave counting is subjective |
| Harmonic Patterns | signal | `harmonic` | Harmonic (Gartley / Bat family): X-A-B-C with AB = 0.382-0.886 XA and BC = 0.382-0.886 AB, then a close back out of the 0.786-0.886 XA zone (D). — Gartley / Bat family D-zone |
| Head and Shoulders | signal | `head_shoulders` | Head and shoulders (and inverse): H-L-H(head)-L-H with the head >= 0.5 ATR above both shoulders and the shoulders within 1 ATR; trade the neckline break. |
| Double Top / Double Bottom | signal | `double_top_bottom` | Double top / bottom: two swing extremes within 0.3 ATR, 1.5+ ATR above (below) the trough between them; trade the break of that trough (peak). |
| Cup and Handle | signal | `cup_handle` | Cup and handle: rim - bottom - rim (rims within 0.5 ATR, cup >= 3 ATR deep and >= 20 bars wide), then a handle low in the upper half of the cup; trade the break of the rims. Inverted for shorts. |
| Flag and Pennant | signal | `flag` | Flag / pennant: a pole of 3+ ATR within 8 bars, then 6 bars of consolidation narrower than half the pole that retrace less than half of it; trade the break. |
| Triangle Patterns | signal | `triangle` | Triangle: lower highs and higher lows over the last four pivots; trade the first close outside. |
| Divergence Trading | signal | `divergence` | RSI divergence: a lower swing low with a higher RSI (bullish), a higher swing high with a lower RSI (bearish), traded when the second swing is confirmed. |

### 8. Quantitative / algorithmic

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| Mean Reversion | signal | `mean_reversion` | Mean reversion (IBS): a bar closing in its bottom 15% above the SMA200 (top 15% below it). |
| Statistical Arbitrage | unavailable | — | needs a basket of co-moving instruments traded together; the bot trades one symbol |
| Pairs Trading | unavailable | — | needs a second instrument (e.g. XAGUSD) and simultaneous two-leg orders |
| Momentum Factor | signal | `momentum_factor` | Time-series momentum: the one-day return crosses +/- 2 ATR with the last quarter-day agreeing. — time-series momentum (single asset) |
| Trend Following Algorithms | signal | `trend_algo` | Trend-following algorithm: EMA 8/21/55 become aligned on this bar. |
| Grid Trading | sizing | `grid` | adds at fixed ATR steps; OFF by default |
| Martingale | sizing | `martingale` | OFF by default; capped steps |
| Anti-Martingale | sizing | `anti_martingale` | OFF by default |
| DCA (Dollar Cost Averaging) | sizing | `dca` | OFF by default |
| Kelly Criterion | sizing | `kelly` | fractional Kelly; OFF by default |
| Machine Learning Trading | existing | `nn` | the neural-network vote (core/ai/nn_*) |
| AI-Based Trading | existing | `ai` | the AI ensemble vote in run.py |
| Reinforcement Learning | unavailable | — | needs a trading simulator and a training pipeline; not built |
| Market Making | unavailable | — | needs two-sided quoting at exchange; a retail CFD account pays the spread instead |
| Arbitrage Trading | unavailable | — | needs several venues/brokers at once |
| High-Frequency Trading (HFT) | unavailable | — | needs co-location and sub-millisecond execution; the bot loops once a second |
| Volatility Breakout | signal | `vol_breakout` | Volatility breakout (Larry Williams): first close beyond today's open +/- half of yesterday's range. |
| Regime Detection | signal | `regime_switch` | Regime detection: Donchian(20) breakouts when ADX > 25, Bollinger reversion when ADX < 20, nothing in between. |

### 9. Volume & order flow

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| Volume Profile | signal | `volume_profile` | Volume profile (rolling 200 bars, tick-volume weighted): rejection of the value area low (buy) or high (sell). — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| Fixed Range Volume Profile | signal | `frvp` | Fixed-range volume profile (previous day): the first test of yesterday's POC from the side the day opened on, rejected. — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| Market Profile / TPO | signal | `tpo_80` | Market profile / TPO 80% rule: the day opens outside yesterday's TPO value area, then two closes back inside -> trade toward the other side. — TPO is time-based: exact |
| Footprint Charts | unavailable | — | needs bid/ask volume per price level; MT5 CFD gold has no trade prints |
| Delta Analysis | signal | `delta` | Delta (proxy): 5-bar delta more than 2 standard deviations from its 50-bar mean, with price closing at a 5-bar extreme in the same direction. — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| Cumulative Volume Delta (CVD) | signal | `cvd_divergence` | Cumulative volume delta divergence (proxy): a new 20-bar low while the session CVD stays above its own 20-bar low (mirrored). — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| Absorption Trading | signal | `absorption` | Absorption (proxy): twice the average tick volume on a bar narrower than 0.6 ATR at a 20-bar extreme -> the extreme holds. — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| Volume Imbalance | signal | `volume_imbalance` | Volume imbalance (ICT): a gap between the BODIES of two consecutive candles whose wicks overlap; trade its first retest. — ICT body-gap definition: exact |
| VWAP Bands | signal | `vwap_bands` | VWAP bands: a close back inside the 2-sigma volume-weighted band. — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| Point of Control (POC) | signal | `poc` | Point of control: a close crossing the rolling 200-bar POC on 1.5x average tick volume. — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| Value Area High / Low | signal | `vah_val` | Value-area acceptance: two consecutive closes above yesterday's VAH (below VAL). — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| Auction Market Theory | signal | `auction` | Auction market theory, failed auction: a look above yesterday's VAH that closes back inside value (look below VAL mirrored). — tick-volume proxy: MT5 gold has no exchange volume or trade prints |
| Liquidity Heatmap | unavailable | — | needs order-book history |
| Order Book Imbalance | unavailable | — | needs a live order book |

### 10. Economic events

| Strategy | Kind | Code name | How it is implemented |
|---|---|---|---|
| NFP Trading | signal | `nfp` | Non-Farm Payrolls breakout (dates generated by rule + calendar). — dates generated by rule + calendar |
| CPI Trading | signal | `cpi` | CPI breakout (dates from the news calendar). — dates from the news calendar only |
| FOMC Trading | signal | `fomc` | FOMC decision / press conference: fade the first spike (whipsaw is typical). — dates from the news calendar only |
| Interest Rate Decisions | signal | `rate_decision` | Interest-rate decisions (any central bank in the calendar): breakout. — dates from the news calendar only |
| News Breakout | signal | `news_breakout` | News breakout: any high-impact release, first close beyond the pre-release hour. |
| News Reversal | signal | `news_reversal` | News reversal: fade a 1.5 ATR release spike that closes back inside the pre-release range. |
| Fundamental Analysis | unavailable | — | needs macro data feeds (real yields, USD index, flows); not connected |
| Economic Calendar Trading | signal | `news_breakout` | News breakout: any high-impact release, first close beyond the pre-release hour. |
| Sentiment Analysis | unavailable | — | needs a sentiment data source; not connected |
| Geopolitical Event Trading | unavailable | — | events are not scheduled; needs a news feed and judgement |
