# Configuration

Every environment variable, its default, and what happens when it is wrong.
Configured in `.env` (git-ignored) at the project root.

---

## How configuration is read

`core/settings.py` (pydantic-settings v2). Priority, highest first:

1. Real environment variables
2. `.env` file
3. Built-in defaults

Validation runs at startup. Invalid values fail fast rather than being
silently coerced.

---

## Safety-critical settings

### `TRADING_MODE`

| Value | Effect |
|---|---|
| `paper` | **Default.** Orders are simulated; nothing reaches the broker |
| `backtest` | Historical simulation |
| `live` | Real orders. Requires credentials and a changed secret key |

An **unrecognised value falls back to `paper`**. The system fails closed.

```dotenv
TRADING_MODE=paper
```

Live mode additionally requires `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, and
a `DASHBOARD_SECRET_KEY` that is not `CHANGE_ME`. `start.ps1` refuses to start
otherwise.

### Credentials

```dotenv
MT5_LOGIN=12345678
MT5_PASSWORD=
MT5_SERVER=YourBroker-Server
MT5_PATH=
```

Never commit these. They are never logged — `core/logging_setup.py` masks
`password=`, `token=`, `Authorization:` and credentials inside URLs.

> **Note:** `run.py` still contains hardcoded credential *fallbacks* at
> `run.py:130-132` inherited from the original project. Setting these
> variables in `.env` ensures the fallbacks are never used. The password should
> be rotated at the broker regardless — it is in git history.

---

## Database

```dotenv
# SQLite — simplest, no service, fine for single-user research
DATABASE_URL=sqlite:///./trading.db

# PostgreSQL — recommended for anything long-running
DATABASE_URL=postgresql://trading:password@localhost:5432/trading
```

Supported drivers: `sqlite`, `postgresql` (psycopg3), `postgresql+asyncpg`
(async). See [DATABASE.md](./DATABASE.md).

Schema is owned by Alembic — never created implicitly:

```powershell
.\scripts\windows\migrate-database.ps1
```

---

## Instruments

```dotenv
# Comma-separated. Note the key is SYMBOLS, not TRADE_SYMBOLS.
SYMBOLS=XAUUSDm,EURUSD,BTCUSD
```

Multi-asset is supported: forex, metals, indices, energy, crypto. Availability
depends on your broker. Unrecognised symbols stay `UNKNOWN` rather than being
guessed into an asset class.

Instruments are resolved from **broker metadata**. If required metadata is
missing or stale, trading is refused for that instrument — the platform never
substitutes a default contract size.

---

## Risk limits

```dotenv
RISK_PER_TRADE=0.01        # 1% of equity per trade (0.001–0.05)
MAX_DAILY_LOSS_PCT=0.02    # 2% daily stop (0.005–0.20)
MAX_DRAWDOWN_PCT=0.10      # 10% max drawdown
MAX_OPEN_POSITIONS=5       # 1–50
MAX_MARGIN_USAGE=0.50      # 50% of equity in margin
MIN_RR_RATIO=1.5           # minimum reward:risk
MAGIC_NUMBER=888999        # isolates bot positions from manual ones
```

All bounded by pydantic; out-of-range values are rejected at startup.

---

## API and dashboard

```dotenv
DASHBOARD_HOST=127.0.0.1   # loopback only by default
DASHBOARD_PORT=8000
DASHBOARD_SECRET_KEY=CHANGE_ME
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=
```

> **The API has no authentication yet.** Loopback binding is a mitigation, not
> a security control. Do not expose port 8000 beyond `127.0.0.1` until auth is
> implemented.

---

## MT5 bridge (optional)

Only needed for a split deployment.

```dotenv
MT5_BRIDGE_URL=             # e.g. http://192.168.122.212:8100
MT5_BRIDGE_TOKEN=           # openssl rand -hex 32
MT5_BRIDGE_TIMEOUT_SECONDS=10
MT5_BRIDGE_TLS_VERIFY=true
```

On the bridge host itself:

```dotenv
MT5_BRIDGE_HOST=127.0.0.1
MT5_BRIDGE_PORT=8100
MT5_BRIDGE_ALLOW_WRITES=0
MT5_BRIDGE_EXPECTED_LOGIN=12345678
```

The bridge **refuses to start** without a token and refuses writes unless
`MT5_BRIDGE_ALLOW_WRITES=1`.

---

## Logging and notifications

```dotenv
LOG_LEVEL=INFO              # DEBUG | INFO | WARNING | ERROR
LOG_DIR=logs                # created automatically
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
LLM_PROVIDER=none           # none | embedded | local_endpoint | openai_compatible
```

Logs are **UTC**, rotated at 10 MB × 5, with credentials redacted.

---

## Changing configuration

```powershell
notepad .env
.\scripts\windows\doctor.ps1      # validate
.\scripts\windows\stop.ps1
.\scripts\windows\start.ps1
```

`doctor.ps1` reports every problem as PASS / WARN / FAIL — it never assumes a
setting is valid because a file exists.