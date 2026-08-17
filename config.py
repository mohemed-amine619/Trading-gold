"""
Global configuration for the trading bot.

Every tunable parameter of the system lives in this single file so that no
magic numbers leak into the engine code. See the inline comments for the
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
MT5_LOGIN = None            # int, e.g. 12345678
MT5_PASSWORD = None         # str
MT5_SERVER = None           # str, e.g. "Exness-MT5Trial16"

# --------------------------------------------------------------------------
# Server timezone normalization
# --------------------------------------------------------------------------
# IMPORTANT: MT5 candle `time` values are UNIX timestamps expressed in the
# BROKER SERVER timezone (the clock shown at the bottom-right of the terminal),
# NOT your local time and NOT necessarily UTC. If we treated them as local
# time, new-bar detection and daily logic would drift by the offset.
# Set this to the UTC offset of your broker's server time:
#   * EET / EEST (most EU brokers)  -> 2 or 3 (2 in winter, 3 in summer)
#   * UTC itself                     -> 0
#   * EST (US brokers, winter)       -> -5
SERVER_UTC_OFFSET_HOURS = 2

# --------------------------------------------------------------------------
# Instruments & timeframes
# --------------------------------------------------------------------------
SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD"]  # must be quoted exactly as on the broker

ENTRY_TIMEFRAME = "M15"   # timeframe on which entry signals are generated
FILTER_TIMEFRAME = "H1"   # higher timeframe used as the trend filter
HISTORY_BARS = 350        # bars fetched per symbol/timeframe (must cover the
                          # EMA-200 warm-up: 200 + safety margin)

# --------------------------------------------------------------------------
# Strategy parameters (placeholder default strategy: EMA cross + RSI filter)
# --------------------------------------------------------------------------
EMA_FAST = 50
EMA_SLOW = 200
RSI_LENGTH = 14
RSI_LONG_LEVEL = 50    # RSI above this => long-friendly momentum
RSI_SHORT_LEVEL = 50   # RSI below this => short-friendly momentum

# --------------------------------------------------------------------------
# Risk management (fixed fractional)
# --------------------------------------------------------------------------
RISK_PER_TRADE = 0.01          # max 1% of CURRENT equity lost if SL is hit
MAX_OPEN_POSITIONS = 5         # hard cap on simultaneously open positions
MAX_MARGIN_USAGE = 0.50        # reject trades if total margin > 50% of equity
ATR_PERIOD = 14
SL_ATR_MULT = 2.0              # initial SL distance = ATR * SL_ATR_MULT
TP_RR_RATIO = 2.0              # take-profit placed at this R:R (2 = 2R)
TRAIL_ATR_MULT = 3.0           # trailing stop distance = ATR * TRAIL_ATR_MULT
TRAIL_ACTIVATE_ATR = 1.0       # start trailing only after price moved >= this
                               # many ATRs in our favour

# --------------------------------------------------------------------------
# Execution engine
# --------------------------------------------------------------------------
# Unique fingerprint of this bot instance. Every order we place carries this
# magic number and we ONLY touch positions with it, so manual trades and other
# bots are left alone. Change it per instance (e.g. YYYYMMDD + seq).
MAGIC_NUMBER = 20250101

# Maximum acceptable slippage for market orders, expressed in POINTS.
# A "point" is the smallest price increment the broker displays
# (symbol_info.point). On 5-digit quotes 1 point = 1/10 of a pip.
# deviation=0 means "no slippage allowed" and will produce constant requotes
# on volatile instruments such as XAUUSD. 20-50 points is a sane default.
DEVIATION_POINTS = 30

MAX_REQUOTE_RETRIES = 5        # how many times to resubmit on requote/price-change
RETRY_BACKOFF_SECONDS = 0.5    # pause between retries (also used on context busy)

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 5.0    # wake interval of the async polling loop

# --------------------------------------------------------------------------
# Alerts (Telegram or Discord) - empty strings disable the channel
# --------------------------------------------------------------------------
ALERT_CHANNEL = "telegram"     # "telegram" | "discord" | "none"
TELEGRAM_BOT_TOKEN = ""        # from @BotFather
TELEGRAM_CHAT_ID = ""          # your chat id (get from @userinfobot)
DISCORD_WEBHOOK_URL = ""       # https://discord.com/api/webhooks/...

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_DIR = "logs"
LOG_FILE = "trading_bot.log"
LOG_LEVEL = "INFO"             # DEBUG | INFO | WARNING | ERROR
