"""
Global configuration for the trading bot.

Every tunable parameter of the system lives in this single file so that no
magic numbers leak into the engine code.  See the inline comments for the
meaning of each setting.
"""

# --------------------------------------------------------------------------
# MetaTrader 5 terminal & account
# --------------------------------------------------------------------------

# Full path to terminal64.exe. Leave None for the default installation
# (auto-detected by the MT5 python package).
MT5_PATH = None  # e.g. r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Credentials. Leave None/empty to use the account currently logged into the
# terminal (recommended: log in manually so no password sits on disk).
MT5_LOGIN    = None   # int,  e.g. 12345678
MT5_PASSWORD = None   # str
MT5_SERVER   = None   # str,  e.g. "Exness-MT5Trial16"

# --------------------------------------------------------------------------
# Server timezone normalization
# --------------------------------------------------------------------------
# IMPORTANT: MT5 candle `time` values are UNIX timestamps expressed in the
# BROKER SERVER timezone (the clock shown at the bottom-right of the terminal).
# Set this to the UTC offset of your broker's server time:
#   * EET / EEST (most EU brokers) -> 2 or 3 (2 in winter, 3 in summer)
#   * UTC itself                    -> 0
#   * EST (US brokers, winter)      -> -5
SERVER_UTC_OFFSET_HOURS = 2

# --------------------------------------------------------------------------
# Instruments & timeframes
# --------------------------------------------------------------------------
SYMBOLS = ["XAUUSDm", "EURUSDm", "GBPUSDm"]  # must be quoted exactly as on the broker

ENTRY_TIMEFRAME  = "M15"   # timeframe on which entry signals are generated
FILTER_TIMEFRAME = "H1"    # first higher timeframe used as trend filter
H4_TIMEFRAME     = "H4"    # second higher timeframe (stronger trend filter)
D1_TIMEFRAME     = "D1"    # daily timeframe for regime detection
HISTORY_BARS     = 600     # bars fetched on ENTRY_TIMEFRAME (covers EMA-200 warm-up)
H4_HISTORY_BARS  = 300     # bars fetched on H4_TIMEFRAME

# --------------------------------------------------------------------------
# AI / Machine-learning model
# --------------------------------------------------------------------------
AI_ENABLED               = True   # master switch for the AI ensemble
AI_CONFIDENCE_THRESHOLD  = 0.58   # min directional probability to register as a signal
AI_TRAIN_BARS            = 500    # rolling window of bars used for training
AI_RETRAIN_BARS          = 50     # retrain the model every N new bars per symbol

# --------------------------------------------------------------------------
# Strategy scoring (0 – 100 confluence score)
# --------------------------------------------------------------------------
# A trade is entered ONLY when the total confluence score >= MIN_SIGNAL_SCORE.
# Score layers:
#   Layer 1  AI ensemble        0–40 pts
#   Layer 2  Trend alignment    0–25 pts  (H1 + H4 EMA direction + ADX)
#   Layer 3  Momentum           0–20 pts  (RSI + MACD + Stochastic)
#   Layer 4  Volume             0–10 pts  (above-average volume on signal bar)
#   Layer 5  Market structure   0–5  pts  (room to run from recent high/low)
MIN_SIGNAL_SCORE = 55

# Session filter – only trade during high-liquidity windows (UTC hours).
# Set USE_SESSION_FILTER = False to trade 24/5.
USE_SESSION_FILTER = True
LONDON_START_UTC   = 7    # 07:00 UTC – London open
LONDON_END_UTC     = 16   # 16:00 UTC – London close
NY_START_UTC       = 12   # 12:00 UTC – New York open
NY_END_UTC         = 21   # 21:00 UTC – New York close

# Market-regime filter: skip trades when the market is choppy / ranging.
ADX_TRENDING_MIN = 20   # ADX below this -> skip the bar
ADX_STRONG_TREND = 30   # ADX above this -> boost trend score

# --------------------------------------------------------------------------
# Legacy strategy parameters (still used as base indicator features)
# --------------------------------------------------------------------------
EMA_FAST        = 50
EMA_SLOW        = 200
RSI_LENGTH      = 14
RSI_LONG_LEVEL  = 50
RSI_SHORT_LEVEL = 50

# --------------------------------------------------------------------------
# Risk management (fixed fractional)
# --------------------------------------------------------------------------
RISK_PER_TRADE     = 0.01    # max 1% of CURRENT equity lost if SL is hit
MAX_OPEN_POSITIONS = 5       # hard cap on simultaneously open positions
MAX_MARGIN_USAGE   = 0.50    # reject trades if total margin > 50% of equity
ATR_PERIOD         = 14
SL_ATR_MULT        = 2.0     # initial SL distance  = ATR * SL_ATR_MULT
TP_RR_RATIO        = 2.5     # take-profit placed at this R:R (2.5 = 2.5R)
TRAIL_ATR_MULT     = 3.0     # trailing stop distance = ATR * TRAIL_ATR_MULT
TRAIL_ACTIVATE_ATR = 1.0     # start trailing only after price moved >= this ATRs

# Dynamic risk scaling: scale position size by trend strength (ADX).
# At ADX <= ADX_TRENDING_MIN -> risk * MIN_RISK_SCALE (conservative).
# At ADX >= ADX_STRONG_TREND + 20 -> risk * MAX_RISK_SCALE (aggressive).
DYNAMIC_RISK   = True
MIN_RISK_SCALE = 0.5    # floor scaling factor
MAX_RISK_SCALE = 1.5    # ceiling scaling factor

# Breakeven: move SL to entry once the position profits by BREAKEVEN_TRIGGER_ATR × ATR.
ENABLE_BREAKEVEN      = True
BREAKEVEN_TRIGGER_ATR = 1.0   # activate after 1 ATR of profit

# Partial close: lock in profits by closing PARTIAL_CLOSE_FRACTION of the
# position once it reaches PARTIAL_CLOSE_TRIGGER_ATR × ATR in profit.
ENABLE_PARTIAL_CLOSE      = True
PARTIAL_CLOSE_TRIGGER_ATR = 1.5
PARTIAL_CLOSE_FRACTION    = 0.50   # close 50% of the position

# Daily drawdown guard: stop trading if equity falls more than
# MAX_DAILY_DRAWDOWN_PCT × start-of-day balance within one session.
MAX_DAILY_DRAWDOWN_PCT = 0.03   # 3 % daily drawdown limit

# --------------------------------------------------------------------------
# Execution engine
# --------------------------------------------------------------------------
# DRY_RUN = True  → paper trading (simulated fills, zero real orders)
# DRY_RUN = False → LIVE trading – use at your own risk!
DRY_RUN = True

# Unique fingerprint of this bot instance.  Every order we place carries this
# magic number and we ONLY touch positions with it.
MAGIC_NUMBER = 20250101

# Maximum acceptable slippage for market orders, expressed in POINTS.
# 0 = no slippage allowed (constant requotes on volatile instruments).
# 20-50 points is a sane default for most CFD brokers.
DEVIATION_POINTS = 30

MAX_REQUOTE_RETRIES   = 5     # how many times to resubmit on requote/price-change
RETRY_BACKOFF_SECONDS = 0.5   # pause between retries

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 5.0   # wake interval of the async polling loop

# --------------------------------------------------------------------------
# Alerts (Telegram or Discord) – empty strings disable the channel
# --------------------------------------------------------------------------
ALERT_CHANNEL       = "telegram"   # "telegram" | "discord" | "none"
TELEGRAM_BOT_TOKEN  = ""           # from @BotFather
TELEGRAM_CHAT_ID    = ""           # your chat id (get from @userinfobot)
DISCORD_WEBHOOK_URL = ""           # https://discord.com/api/webhooks/...

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_DIR   = "logs"
LOG_FILE  = "trading_bot.log"
LOG_LEVEL = "INFO"   # DEBUG | INFO | WARNING | ERROR
