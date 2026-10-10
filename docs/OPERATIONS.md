# OPERATIONS.md — AI Gold Trading Bot · Production Operations Manual

> **Audience:** System operators, DevOps engineers, and traders responsible for running and monitoring the AI Trading Bot in a production environment.

---

## Table of Contents

1. [Deployment Requirements](#1-deployment-requirements)
2. [Environment Setup](#2-environment-setup)
3. [Starting the System](#3-starting-the-system)
4. [Watchdog & Heartbeat](#4-watchdog--heartbeat)
5. [MT5 Connection Monitoring](#5-mt5-connection-monitoring)
6. [AI API Health Check](#6-ai-api-health-check)
7. [Database Health Check](#7-database-health-check)
8. [Log Rotation](#8-log-rotation)
9. [Backup Strategy](#9-backup-strategy)
10. [Alert Channels](#10-alert-channels)
11. [Alert Types & Trigger Conditions](#11-alert-types--trigger-conditions)
12. [Emergency Procedures](#12-emergency-procedures)
13. [Daily Startup Checklist](#13-daily-startup-checklist)
14. [Trading Modes](#14-trading-modes)
15. [Safe Mode Switching](#15-safe-mode-switching)
16. [Monitoring Dashboard](#16-monitoring-dashboard)
17. [Key Metrics to Watch Daily](#17-key-metrics-to-watch-daily)

---

## 1. Deployment Requirements

### 1.1 Operating System

> [!IMPORTANT]
> **Windows OS is mandatory.** The MetaTrader 5 Python API (`MetaTrader5` package) only works on Windows. It cannot run on Linux or macOS. All production deployments must target a Windows host.

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 (64-bit) | Windows 11 or Windows Server 2022 |
| **Python** | 3.10 | 3.11 or 3.12 |
| **RAM** | 4 GB | 8 GB+ |
| **CPU** | 2 cores | 4+ cores (AI retraining is CPU-bound) |
| **Disk** | 10 GB free | 50 GB SSD (logs + DB growth over months) |
| **Network** | Stable broadband | Low-latency connection to broker server |
| **Screen** | Not required (headless OK) | 1 monitor for Tkinter GUI |

### 1.2 Required Software

| Software | Version | Purpose | Download |
|---|---|---|---|
| **MetaTrader 5 Terminal** | Latest stable | Broker connection, order execution | Broker website |
| **Python** | 3.10 – 3.12 | Bot runtime | python.org |
| **Git** | Any | Source control | git-scm.com |
| **Visual C++ Redistributable** | 2019+ | scikit-learn native extensions | Microsoft |

### 1.3 Broker Account Requirements

- **Real or Demo account** with a broker supporting MetaTrader 5
- **XAUUSDm** (Gold Micro) symbol available on the account
- Account type: **ECN** or **STP** recommended (tighter spreads)
- **Leverage:** Minimum 1:20 (1:100 or higher recommended for micro lots)
- API trading must be **enabled** in the broker portal (some brokers disable it by default)
- The MetaTrader 5 terminal must be **logged in and running** before the bot starts

### 1.4 Python Package Dependencies

```
MetaTrader5       # MT5 Python bridge (Windows only)
pandas            # Data manipulation
pandas_ta         # Technical indicators
numpy             # Numerical computing
PyQt5             # GUI (Tkinter is used at runtime)
matplotlib        # Chart rendering
xgboost           # XGBoost model (optional, future use)
requests          # HTTP (Telegram, webhook alerts)
scikit-learn      # AI ensemble (RF, GBT, ET, HGB)
scipy             # Statistical utilities
pydantic>=2.0     # Settings validation
pydantic-settings # .env loading
python-dotenv     # .env parsing
```

---

## 2. Environment Setup

### 2.1 Clone and Prepare the Repository

```powershell
# 1. Clone the repository
git clone https://github.com/your-org/Trading-gold.git
cd Trading-gold

# 2. Create a Python virtual environment
python -m venv .venv

# 3. Activate the virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Copy the environment template
copy .env.example .env
```

> [!TIP]
> Use `python -m venv .venv` rather than a system-wide install. This isolates dependencies and makes it easy to reproduce the environment on another machine.

### 2.2 Full `.env` File Reference

Create `.env` in the project root. **Never commit this file to version control** — it contains secrets.

```dotenv
# ════════════════════════════════════════════════════════════════════
# .env — AI Gold Trading Bot · Production Configuration
# Copy from .env.example, then fill in all values marked <REQUIRED>
# NEVER commit this file. It is listed in .gitignore.
# ════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# MetaTrader 5 Credentials                                  REQUIRED
# ──────────────────────────────────────────────────────────────────
MT5_LOGIN=261009880
MT5_PASSWORD=YourSecureBrokerPassword
MT5_SERVER=Exness-MT5Trial16
# Full path to the terminal64.exe binary.
# Leave empty to let the MT5 Python API auto-detect the installation.
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

# ──────────────────────────────────────────────────────────────────
# Broker Server Clock Offset
# EET/EEST (most EU brokers): 2 or 3
# UTC brokers: 0  |  EST brokers: -5
# ──────────────────────────────────────────────────────────────────
SERVER_UTC_OFFSET_HOURS=2

# ──────────────────────────────────────────────────────────────────
# Trading Mode
# paper          → live market data, simulated execution (no real orders)
# live           → real money execution
# backtest       → offline historical data, no MT5 connection required
# ──────────────────────────────────────────────────────────────────
TRADING_MODE=paper

# ──────────────────────────────────────────────────────────────────
# Symbols (comma-separated)
# ──────────────────────────────────────────────────────────────────
SYMBOLS=XAUUSDm

# ──────────────────────────────────────────────────────────────────
# Database
# SQLite (development / single machine):
DATABASE_URL=sqlite:///./trading.db
# PostgreSQL (production / multi-machine):
# DATABASE_URL=postgresql+asyncpg://trading_user:dbpassword@localhost:5432/trading_gold
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# LLM Provider  (leave LLM_PROVIDER=none to disable AI reasoning)
# ──────────────────────────────────────────────────────────────────
LLM_PROVIDER=none
# Options: openai | anthropic | ollama | none

# OpenAI
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Local Ollama (no API key required)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# ──────────────────────────────────────────────────────────────────
# Notification Channels
# ──────────────────────────────────────────────────────────────────
# Telegram Bot
TELEGRAM_BOT_TOKEN=8423946950:AAF0Ja88p_52coyDsh48nQD1yNZW7NwAgck
TELEGRAM_CHAT_ID=6476316022

# Discord Webhook (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Generic HTTP Webhook (optional)
WEBHOOK_URL=https://your-ops-server.com/alerts/trading
WEBHOOK_SECRET=a_shared_hmac_secret_for_signature_verification

# Email / SMTP (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=alerts@yourcompany.com
SMTP_PASSWORD=your_app_password_here
ALERT_EMAIL_FROM=alerts@yourcompany.com
ALERT_EMAIL_TO=trader@yourcompany.com,ops@yourcompany.com

# ──────────────────────────────────────────────────────────────────
# Dashboard / REST API
# ──────────────────────────────────────────────────────────────────
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
DASHBOARD_SECRET_KEY=replace_with_64_char_random_hex_before_deploying
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=a_very_strong_password_here

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_DIR=logs

# ──────────────────────────────────────────────────────────────────
# Risk Parameters
# ──────────────────────────────────────────────────────────────────
RISK_PER_TRADE=0.01            # 1% of equity per trade
MAX_DAILY_LOSS_PCT=0.02        # 2% max daily drawdown
MAX_DRAWDOWN_PCT=0.10          # 10% max total drawdown
MAX_OPEN_POSITIONS=5
MIN_RR_RATIO=1.5
MAX_MARGIN_USAGE=0.50          # 50% margin cap
MAGIC_NUMBER=888999            # unique identifier for bot's orders

# ──────────────────────────────────────────────────────────────────
# Strategy & AI Thresholds
# ──────────────────────────────────────────────────────────────────
MIN_SIGNAL_SCORE=55.0
AI_CONFIDENCE_THRESHOLD=0.58
AI_TRAIN_BARS=500
AI_RETRAIN_BARS=50

# ──────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS=5.0
DEVIATION_POINTS=30
MAX_REQUOTE_RETRIES=5
MAX_SPREAD_POINTS=50           # skip orders if spread > this; blank = disabled

# ──────────────────────────────────────────────────────────────────
# Stop-Loss / Take-Profit / Trailing
# ──────────────────────────────────────────────────────────────────
SL_ATR_MULT=2.0
TP_RR_RATIO=2.5
TRAIL_ATR_MULT=3.0
TRAIL_ACTIVATE_ATR=1.0
TRAIL_TIGHTEN1_ATR=2.5
TRAIL_TIGHTEN1_MULT=1.5
TRAIL_TIGHTEN2_ATR=4.0
TRAIL_TIGHTEN2_MULT=1.0
ENABLE_BREAKEVEN=true
BREAKEVEN_TRIGGER_ATR=1.0
ENABLE_PARTIAL_CLOSE=true
PARTIAL_CLOSE_TRIGGER_ATR=1.5
PARTIAL_CLOSE_FRACTION=0.50
MAX_LOSS_PER_TRADE_PCT=0.01    # 1% equity hard backstop
PROFIT_LOCK_ACTIVATE_ATR=1.2
PROFIT_GIVEBACK_PCT=0.35
CUT_LOSING_AFTER_HOURS=3.0

# ──────────────────────────────────────────────────────────────────
# Session Filter (UTC hours)
# ──────────────────────────────────────────────────────────────────
USE_SESSION_FILTER=true
LONDON_START_UTC=7
LONDON_END_UTC=16
NY_START_UTC=12
NY_END_UTC=21

# ──────────────────────────────────────────────────────────────────
# Watchdog
# ──────────────────────────────────────────────────────────────────
WATCHDOG_INTERVAL_SECONDS=30
HEARTBEAT_MAX_AGE_SECONDS=120

# ──────────────────────────────────────────────────────────────────
# Consecutive-Loss Protection
# ──────────────────────────────────────────────────────────────────
MAX_CONSECUTIVE_LOSSES=5
LOSS_STREAK_COOLDOWN_MINUTES=60
```

### 2.3 Verify Settings Load

```powershell
# Confirm Pydantic can read all settings without errors
python -c "from core.settings import get_settings; s = get_settings(); print('Mode:', s.trading_mode); print('MT5 login:', s.mt5_login)"
```

Expected output (PAPER mode):
```
Mode: TradingMode.paper
MT5 login: 261009880
```

---

## 3. Starting the System

### 3.1 Prerequisites Checklist

Before issuing any start command, verify:

1. MetaTrader 5 terminal is **open and logged in**
2. The virtual environment is **activated** (`.\.venv\Scripts\Activate.ps1`)
3. The `.env` file is **complete** (no empty required fields)
4. `TRADING_MODE` is set to the **intended mode** (start with `paper`)

### 3.2 CLI Start Commands

```powershell
# ── Activate the virtual environment ──────────────────────────────
.\.venv\Scripts\Activate.ps1

# ── Paper mode (default, safest) ──────────────────────────────────
python run.py

# ── Override mode via environment without editing .env ────────────
$env:TRADING_MODE="paper"
python run.py

# ── Launch GUI (Tkinter config editor + log viewer) ───────────────
python gui.py

# ── Launch standalone position dashboard ──────────────────────────
python position_dashboard.py

# ── Launch web API / dashboard (FastAPI) ──────────────────────────
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Run backtests ─────────────────────────────────────────────────
$env:TRADING_MODE="backtest"
python run.py --from 2024-01-01 --to 2024-12-31
```

### 3.3 Startup Sequence

When `python run.py` is invoked, the following sequence executes in order:

```mermaid
flowchart TD
    A["python run.py"] --> B["Load .env → Settings validation\n(Pydantic v2)"]
    B --> C["Configure logging\n(trading_bot.log + stdout)"]
    C --> D["Connect to MetaTrader 5\nmt5.initialize()"]
    D --> E{Connection OK?}
    E -- No --> F["Retry with backoff\n(up to 5 attempts)"]
    F --> E
    E -- Yes --> G["Login to broker account\nmt5.login()"]
    G --> H["Verify symbol availability\nXAUUSDm"]
    H --> I["Fetch initial candles\nM15 / H1 / H4 × 600 bars"]
    I --> J["Compute all indicators\n_prepare()"]
    J --> K["Train AI Ensemble\n(synchronous first fit)"]
    K --> L["Start Watchdog thread\n(heartbeat monitor)"]
    L --> M["Start Notifier thread\n(Telegram / alerts)"]
    M --> N["Enter main trading loop\npoll_interval_seconds = 5.0"]
    N --> O["Emit heartbeat every loop"]
    O --> P["Check new bar → score →\nrisk → execute → manage"]
    P --> N
```

**Services started on launch:**

| Service | Thread | Purpose |
|---|---|---|
| **Main trading loop** | Main thread | Bar-by-bar signal generation & execution |
| **AI retraining thread** | Background daemon | Rolling model retrain every `AI_RETRAIN_BARS` |
| **Notifier** | Background daemon | Non-blocking Telegram/webhook delivery |
| **Watchdog** | Background daemon | Heartbeat monitoring & auto-restart |
| **Position monitor** | Main loop (tick-level) | Trailing stop, breakeven, partial close |

---

## 4. Watchdog & Heartbeat

### 4.1 How It Works

The watchdog is a lightweight background thread that monitors the health of the main trading loop. The main loop writes a **heartbeat timestamp** to a shared in-memory variable at the end of every iteration. The watchdog checks this timestamp every `WATCHDOG_INTERVAL_SECONDS` (default: 30 s). If the timestamp has not been updated within `HEARTBEAT_MAX_AGE_SECONDS` (default: 120 s), the main loop is considered **stale or dead**.

```mermaid
sequenceDiagram
    participant ML as Main Loop
    participant HB as Heartbeat Store
    participant WD as Watchdog Thread
    participant AL as Alert Channel

    loop Every loop iteration (5s)
        ML->>HB: Write timestamp (now)
    end

    loop Every 30s
        WD->>HB: Read last_heartbeat
        alt age < 120s
            WD->>WD: OK — continue
        else age >= 120s
            WD->>AL: SYSTEM_ERROR alert fired
            WD->>ML: Attempt graceful shutdown
            WD->>ML: Re-launch main loop
        end
    end
```

### 4.2 Configuration

```dotenv
WATCHDOG_INTERVAL_SECONDS=30      # how often the watchdog checks
HEARTBEAT_MAX_AGE_SECONDS=120     # stale threshold — 0 means disabled
```

### 4.3 Auto-Restart Behaviour

On detecting a stale heartbeat, the watchdog:

1. Sends a `SYSTEM_ERROR` alert to all configured channels
2. Logs `[WATCHDOG] Main loop stale — restarting` at ERROR level
3. Calls `sys.exit(1)` so a process supervisor (e.g. Windows Task Scheduler, NSSM, or PM2) can restart `run.py` automatically

### 4.4 Running as a Windows Service (NSSM)

```powershell
# Install NSSM (Non-Sucking Service Manager)
# Download from https://nssm.cc

# Register the bot as a Windows service
nssm install TradingGoldBot "C:\Trading-gold\.venv\Scripts\python.exe" "C:\Trading-gold\run.py"
nssm set TradingGoldBot AppDirectory "C:\Trading-gold"
nssm set TradingGoldBot AppStdout "C:\Trading-gold\logs\service_stdout.log"
nssm set TradingGoldBot AppStderr "C:\Trading-gold\logs\service_stderr.log"
nssm set TradingGoldBot AppRestartDelay 5000   # 5s before restart on crash

# Start / stop the service
nssm start TradingGoldBot
nssm stop TradingGoldBot
nssm status TradingGoldBot
```

---

## 5. MT5 Connection Monitoring

### 5.1 Periodic Ping

Every main-loop iteration the bot calls `mt5.terminal_info()` and `mt5.account_info()` as lightweight liveness probes. If either returns `None`, the connection is considered **lost**.

### 5.2 Reconnect Logic

```mermaid
stateDiagram-v2
    [*] --> Connected: mt5.initialize() success
    Connected --> Disconnected: terminal_info() returns None
    Disconnected --> Reconnecting: backoff attempt 1 (2s)
    Reconnecting --> Reconnecting: backoff attempt 2 (4s)
    Reconnecting --> Reconnecting: backoff attempt 3 (8s)
    Reconnecting --> Connected: mt5.initialize() success
    Reconnecting --> AlertSent: max retries (5) exhausted
    AlertSent --> Reconnecting: wait 60s, retry cycle
```

**Reconnect parameters (hardcoded defaults, configurable via code):**

| Parameter | Value | Description |
|---|---|---|
| Max retries per cycle | 5 | Attempts before pausing |
| Backoff base | 2 s | Doubles each attempt |
| Max backoff | 60 s | Cap per sleep |
| Pause after max retries | 60 s | Before next full cycle |

### 5.3 Alert on Disconnect

When the MT5 connection drops, the `MT5_DISCONNECT` alert fires:

- **Immediate** notification on Telegram / webhook
- **Trading halted** — no new orders placed while disconnected
- **Position management suspended** — trailing/breakeven updates paused
- Auto-reconnect runs silently in the background; a `MT5_RECONNECTED` info message is sent on recovery

---

## 6. AI API Health Check

### 6.1 LLM Provider Reachability

If `LLM_PROVIDER` is not `none`, the system performs a health check on startup and every 15 minutes during operation.

| Provider | Health Endpoint | Timeout |
|---|---|---|
| `openai` | `GET https://api.openai.com/v1/models` | 10 s |
| `anthropic` | `GET https://api.anthropic.com/v1/models` | 10 s |
| `ollama` | `GET {OLLAMA_BASE_URL}/api/tags` | 5 s |

### 6.2 Fallback Activation

```mermaid
flowchart LR
    A["LLM health check"] --> B{Reachable?}
    B -- Yes --> C["Normal operation\nLLM reasoning active"]
    B -- No --> D["LLM_UNAVAILABLE alert"]
    D --> E["Fallback: ML ensemble only\n(scikit-learn, no LLM reasoning)"]
    E --> F["Retry LLM every 5 min"]
    F --> B
```

> [!WARNING]
> When the LLM is unavailable, trades are still generated using the ML ensemble and multi-layer confluence score alone. Risk limits remain fully enforced. The alert message will note `[LLM_UNAVAILABLE] Falling back to ML-only mode`.

### 6.3 Disabling the LLM

To run without any LLM dependency:

```dotenv
LLM_PROVIDER=none
```

The bot will use only the scikit-learn ensemble (RandomForest + GBT + ExtraTrees + HistGBT) — this is the current production default.

---

## 7. Database Health Check

### 7.1 SQLite (Default)

On every startup, the system:

1. Opens the `trading.db` file with `PRAGMA integrity_check`
2. Enables **WAL mode** for safe concurrent reads: `PRAGMA journal_mode=WAL`
3. Sets `PRAGMA synchronous=NORMAL` (balance of safety and speed)
4. Verifies write permissions with a test `INSERT`/`DELETE`

```python
# Applied automatically by core/database/engine.py on connection
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA cache_size=-64000;   # 64 MB page cache
```

> [!TIP]
> WAL mode allows the dashboard (reader) and the trading loop (writer) to operate concurrently without blocking each other. Always use WAL for SQLite in production.

### 7.2 PostgreSQL (Production)

```dotenv
DATABASE_URL=postgresql+asyncpg://trading_user:secret@localhost:5432/trading_gold
```

PostgreSQL health check:

```sql
-- Run via psql or in the startup probe
SELECT 1;
SELECT pg_database_size('trading_gold');
SELECT count(*) FROM pg_stat_activity WHERE datname='trading_gold';
```

### 7.3 Database Health Probe

The system issues a health probe every 5 minutes. If the probe fails:

1. `SYSTEM_ERROR` alert fired with message `Database unreachable`
2. Trading **paused** (no new orders); existing positions continue to be managed from in-memory state
3. Write queue accumulates events in RAM for up to 60 s
4. On reconnect, the queue is flushed to disk

---

## 8. Log Rotation

### 8.1 Log Files

| File | Content | Rotation |
|---|---|---|
| `logs/trading_bot.log` | Main trading events, signals, orders | Daily |
| `logs/trading_bot.log.YYYY-MM-DD` | Rotated archived log | Retained 30 days |
| `logs/service_stdout.log` | NSSM service stdout | Manual |
| `logs/service_stderr.log` | NSSM service stderr | Manual |

### 8.2 Rotation Configuration (Python `logging`)

```python
import logging
from logging.handlers import TimedRotatingFileHandler

handler = TimedRotatingFileHandler(
    filename="logs/trading_bot.log",
    when="midnight",        # rotate at midnight local time
    interval=1,             # daily
    backupCount=30,         # keep 30 days of archives
    encoding="utf-8",
    errors="backslashreplace",
)
handler.suffix = "%Y-%m-%d"
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
))
```

### 8.3 Log Compression (PowerShell scheduled task)

Add this as a **Windows Scheduled Task** to compress archived logs older than 7 days:

```powershell
# compress_old_logs.ps1 — run daily at 01:00
$logsDir = "C:\Trading-gold\logs"
Get-ChildItem -Path $logsDir -Filter "trading_bot.log.*" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) -and $_.Extension -ne ".gz" } |
    ForEach-Object {
        $gz = "$($_.FullName).gz"
        $reader = [System.IO.File]::OpenRead($_.FullName)
        $writer = [System.IO.Compression.GZipStream]::new(
            [System.IO.File]::Create($gz),
            [System.IO.Compression.CompressionMode]::Compress
        )
        $reader.CopyTo($writer)
        $reader.Close(); $writer.Close()
        Remove-Item $_.FullName
        Write-Host "Compressed: $($_.Name)"
    }

# Delete logs older than 30 days
Get-ChildItem -Path $logsDir -Filter "*.gz" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item
```

### 8.4 Log Format

```
2026-10-08 09:15:02,341 [INFO] __main__ — [SIGNAL] XAUUSDm LONG score=72.4 AI_conf=0.631
2026-10-08 09:15:03,102 [INFO] __main__ — [ORDER] XAUUSDm BUY 0.10 @ 2685.50 SL=2673.20 TP=2716.00
2026-10-08 09:15:03,210 [INFO] __main__ — [TRADE_OPENED] ticket=12345678 lot=0.10 SL=2673.20 TP=2716.00
2026-10-08 09:45:12,891 [INFO] __main__ — [HEARTBEAT] loop_iteration=360 open_positions=1
2026-10-08 10:02:44,003 [WARNING] __main__ — [MT5] terminal_info returned None — reconnecting
2026-10-08 10:02:46,218 [INFO] __main__ — [MT5] Reconnected after 1 attempt (2.2s)
```

---

## 9. Backup Strategy

### 9.1 Database Backup (Daily)

```powershell
# backup_db.ps1 — run daily at 02:00
$src    = "C:\Trading-gold\trading.db"
$dest   = "C:\Trading-gold\backups\trading_$(Get-Date -Format 'yyyy-MM-dd').db"
$backupDir = "C:\Trading-gold\backups"
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null

# SQLite: use the official backup API for a consistent hot backup
# (safe even while the bot is running in WAL mode)
$python = "C:\Trading-gold\.venv\Scripts\python.exe"
& $python -c "
import sqlite3, shutil, sys
src  = r'$src'
dest = r'$dest'
src_con  = sqlite3.connect(src)
dest_con = sqlite3.connect(dest)
src_con.backup(dest_con)
src_con.close(); dest_con.close()
print('Backup OK:', dest)
"

# Delete backups older than 30 days
Get-ChildItem -Path $backupDir -Filter "trading_*.db" |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item
```

### 9.2 Configuration Backup

```powershell
# backup_config.ps1
$date    = Get-Date -Format 'yyyy-MM-dd'
$dest    = "C:\Trading-gold\backups\config_$date"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# Copy all config files (exclude .env — backed up separately, encrypted)
Copy-Item "C:\Trading-gold\core\settings.py"  $dest
Copy-Item "C:\Trading-gold\config.py"          $dest
Copy-Item "C:\Trading-gold\requirements.txt"   $dest
# .env backed up to a password-manager / secure vault — NOT plain-text
```

### 9.3 Backup Verification

Run after every backup to confirm integrity:

```powershell
# verify_backup.ps1
$backup = "C:\Trading-gold\backups\trading_$(Get-Date -Format 'yyyy-MM-dd').db"
& "C:\Trading-gold\.venv\Scripts\python.exe" -c "
import sqlite3
con = sqlite3.connect(r'$backup')
result = con.execute('PRAGMA integrity_check').fetchone()
count  = con.execute('SELECT count(*) FROM trades').fetchone()[0]
con.close()
if result[0] == 'ok':
    print(f'Backup VALID — {count} trade records')
else:
    print('BACKUP CORRUPT:', result[0])
    raise SystemExit(1)
"
```

> [!CAUTION]
> Never use `COPY` or file-copy tools on a live SQLite database — this can produce a corrupt backup. Always use `sqlite3.Connection.backup()` (Python) or the `.backup` shell command, which safely copies a WAL-mode database while it is running.

### 9.4 Off-site Backup

For production: replicate the `backups/` folder to a cloud storage bucket daily:

```powershell
# Using AWS CLI
aws s3 sync C:\Trading-gold\backups\ s3://my-trading-backups/Trading-gold/ --storage-class STANDARD_IA

# Using rclone (generic)
rclone sync C:\Trading-gold\backups\ remote:trading-gold-backups
```

---

## 10. Alert Channels

### 10.1 Telegram Bot

**Setup instructions:**

1. Open Telegram and message `@BotFather`
2. Send `/newbot`, follow the prompts → copy the **bot token**
3. Start a conversation with your new bot (send `/start`)
4. Fetch your **chat ID**:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
   Look for `"chat":{"id":...}` in the JSON response
5. Add to `.env`:
   ```dotenv
   TELEGRAM_BOT_TOKEN=8423946950:AAF0Ja88p_52coyDsh48nQD1yNZW7NwAgck
   TELEGRAM_CHAT_ID=6476316022
   ```

**Message format:**

```
🟢 TRADE_OPENED
Symbol:  XAUUSDm
Side:    BUY
Lot:     0.10
Entry:   2685.50
SL:      2673.20
TP:      2716.00
Score:   72.4
AI conf: 0.631
Time:    2026-10-08 09:15:03 UTC
```

```
🔴 DAILY_LOSS_LIMIT
Account:  261009880
Loss:     -$210.50 (-2.1% equity)
Limit:    2.0%
Action:   ALL POSITIONS CLOSED
Time:     2026-10-08 14:30:00 UTC
```

**Resilience:** The Telegram notifier retries 3 times with 1.5 s backoff per attempt. After 3 consecutive failures it enters a 60-second cooldown before auto-resuming. The trading loop is never blocked by network issues.

---

### 10.2 Email (SMTP)

**Setup for Gmail (App Password):**

1. Enable 2-Factor Authentication on the Gmail account
2. Go to **Google Account → Security → App Passwords**
3. Generate a password for "Mail / Windows Computer"
4. Add to `.env`:
   ```dotenv
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USE_TLS=true
   SMTP_USERNAME=alerts@yourcompany.com
   SMTP_PASSWORD=abcd efgh ijkl mnop    # 16-char app password
   ALERT_EMAIL_FROM=alerts@yourcompany.com
   ALERT_EMAIL_TO=trader@yourcompany.com
   ```

**Email subject/body format:**

```
Subject: [ALERT] DRAWDOWN_WARNING — Trading-gold Bot
Body:
Account:     261009880
Alert type:  DRAWDOWN_WARNING
Drawdown:    -7.4% (limit: 10%)
Open P&L:   -$740.00
Time (UTC):  2026-10-08 11:45:00
Bot version: run.py
```

> [!NOTE]
> Email alerts are rate-limited to **1 per alert type per 10 minutes** to prevent inbox flooding during volatile market events.

---

### 10.3 Webhook (Generic HTTP POST)

Configure a generic endpoint to receive all alerts as JSON:

```dotenv
WEBHOOK_URL=https://your-ops-server.com/alerts/trading
WEBHOOK_SECRET=a_shared_hmac_secret
```

**Payload format (application/json):**

```json
{
  "alert_type": "TRADE_OPENED",
  "timestamp": "2026-10-08T09:15:03Z",
  "account": 261009880,
  "symbol": "XAUUSDm",
  "data": {
    "side": "BUY",
    "lot": 0.10,
    "entry": 2685.50,
    "sl": 2673.20,
    "tp": 2716.00,
    "ticket": 12345678,
    "score": 72.4,
    "ai_confidence": 0.631
  },
  "signature": "sha256=abc123..."
}
```

Signature verification (Python receiver example):

```python
import hmac, hashlib

def verify_signature(body: bytes, secret: str, header_sig: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header_sig)
```

---

## 11. Alert Types & Trigger Conditions

| Alert Type | Severity | Trigger Condition | Action Taken | Channels |
|---|---|---|---|---|
| `TRADE_OPENED` | INFO | A new position is successfully placed with the broker | Log + notify | Telegram, Webhook, Email |
| `TRADE_CLOSED` | INFO | Any open position is closed (SL, TP, partial, manual, protection) | Log + notify with P&L | Telegram, Webhook, Email |
| `TRADE_PARTIAL_CLOSE` | INFO | A partial close (50% or 25%) is executed | Log + notify with remaining lot | Telegram, Webhook |
| `BREAKEVEN_MOVED` | INFO | Stop-loss moved to entry price | Log only | Log |
| `DAILY_LOSS_LIMIT` | CRITICAL | Daily P&L drawdown ≥ `MAX_DAILY_LOSS_PCT` of starting equity | All positions closed; trading suspended until next session | **All channels** |
| `DRAWDOWN_WARNING` | WARNING | Total drawdown reaches 75% of `MAX_DRAWDOWN_PCT` threshold | No new trades; alert sent; manual review required | Telegram, Email |
| `DRAWDOWN_LIMIT` | CRITICAL | Total drawdown ≥ `MAX_DRAWDOWN_PCT` | All positions closed; kill switch arms | **All channels** |
| `MAX_POSITIONS_REACHED` | WARNING | Open positions = `MAX_OPEN_POSITIONS` | New entry signals ignored | Log, Telegram |
| `MARGIN_GUARD` | WARNING | Used margin ≥ `MAX_MARGIN_USAGE` × account equity | New orders blocked | Log, Telegram |
| `SPREAD_GUARD` | INFO | Current spread > `MAX_SPREAD_POINTS` | Entry skipped for this bar | Log |
| `MT5_DISCONNECT` | CRITICAL | `mt5.terminal_info()` or `mt5.account_info()` returns `None` | Trading halted; reconnect loop starts | Telegram, Webhook, Email |
| `MT5_RECONNECTED` | INFO | MT5 connection restored after disconnect | Trading resumes | Telegram |
| `LLM_UNAVAILABLE` | WARNING | HTTP health check to LLM API times out or returns non-200 | Fallback to ML-only mode | Telegram, Log |
| `LLM_RECOVERED` | INFO | LLM API becomes reachable again | LLM reasoning re-enabled | Telegram |
| `DB_UNREACHABLE` | CRITICAL | Database probe fails (integrity check or connection refused) | New trades paused; write queue buffered | Telegram, Email |
| `DB_RECOVERED` | INFO | Database comes back online; write queue flushed | Normal operation resumes | Log |
| `CONSECUTIVE_LOSSES` | WARNING | Loss streak reaches `MAX_CONSECUTIVE_LOSSES` | Cooldown for `LOSS_STREAK_COOLDOWN_MINUTES` | Telegram, Email |
| `KILL_SWITCH_ACTIVATED` | CRITICAL | Manual kill switch triggered via dashboard or API | All positions closed immediately; bot halted | **All channels** |
| `KILL_SWITCH_DEACTIVATED` | WARNING | Kill switch released | Bot resumed in PAPER mode by default | All channels |
| `SYSTEM_ERROR` | ERROR | Watchdog detects stale heartbeat, or unhandled exception in main loop | Alert sent; watchdog attempts restart | **All channels** |
| `PROFIT_LOCK` | INFO | Profit-lock / giveback close triggered | Position closed with locked profit | Telegram |
| `LOSS_TIMEOUT` | INFO | Losing position cut after `CUT_LOSING_AFTER_HOURS` hours | Position force-closed | Telegram |
| `AGE_TIMEOUT` | INFO | Profitable position closed after `MAX_POSITION_AGE_HOURS` | Position force-closed | Telegram |
| `TREND_FLIP_CLOSE` | INFO | H1 trend reversal detected; position closed pre-emptively | Position closed | Log, Telegram |
| `OPPOSITE_SIGNAL_CLOSE` | INFO | Strong opposing signal score ≥ `OPPOSITE_SIGNAL_MIN_SCORE` | Position closed | Log, Telegram |
| `SESSION_FILTER_ACTIVE` | INFO | Current UTC hour outside London/NY windows | New entries blocked | Log |
| `AI_RETRAIN_COMPLETE` | INFO | Background ML ensemble retrain finished | New model version active | Log |
| `AI_RETRAIN_FAILED` | WARNING | Training raised an exception or had degenerate labels | Previous model kept | Log, Telegram |

---

## 12. Emergency Procedures

### 12.1 Kill Switch Activation

The kill switch immediately closes all open positions and halts the bot. Activate via **any** of these methods:

**Method A — Dashboard (recommended):**
1. Navigate to `http://localhost:8000/control` (or your dashboard URL)
2. Click **Emergency Stop** → confirm the prompt
3. All positions close at market; bot state changes to `HALTED`

**Method B — API endpoint:**
```powershell
# POST to the emergency stop endpoint
$token = "your_jwt_token"
Invoke-RestMethod -Method POST `
    -Uri "http://localhost:8000/api/v1/control/emergency-stop" `
    -Headers @{ Authorization = "Bearer $token" } `
    -ContentType "application/json" `
    -Body '{"reason": "Manual kill switch — operator request"}'
```

**Method C — Terminal (immediate):**
```powershell
# Find the bot process
Get-Process python | Where-Object { $_.CommandLine -like "*run.py*" }

# Send CTRL+C (graceful) — allows cleanup code to run
# The bot's SIGINT handler closes positions before exiting
[System.Console]::CancelKeyPress  # or press Ctrl+C in the terminal window
```

**Method D — Task Manager / NSSM:**
```powershell
nssm stop TradingGoldBot        # stops the service (positions stay open)
# THEN manually close positions in MT5 terminal
```

> [!CAUTION]
> Method D (service stop) does **not** close open positions automatically. Use only when the bot is completely unresponsive and you will manually close positions in the MT5 terminal.

---

### 12.2 Manual Position Close (MT5 Terminal)

If the bot cannot close positions programmatically:

1. Open **MetaTrader 5 Terminal**
2. Go to **View → Terminal** (or press `Ctrl+T`)
3. Click the **Trade** tab
4. Right-click on each open position → **Close Position**
5. Alternatively: Select all, right-click → **Close All Positions**

To close all positions for magic number `888999` only (bot positions):
- In **Terminal → Trade**, filter by the `Comment` column or `Magic` column
- Close only rows where Magic = `888999`

---

### 12.3 Broker Emergency Contact

> [!IMPORTANT]
> Save your broker's emergency contact before going live. Markets can move fast during outages.

| Method | Action |
|---|---|
| **Broker Live Chat** | Available 24/5 on most broker portals |
| **Phone support** | Request your broker's emergency trading desk number |
| **Broker Web Terminal** | If MT5 is unresponsive, use `https://[broker].com/trading` web trader |
| **MetaQuotes WebTerminal** | `https://trade.mql5.com` (requires broker server address) |

**Information to provide to broker support:**

- Account number: `261009880`
- Ticket numbers of all open positions
- Request: "Close all open positions at market — automated system failure"

---

### 12.4 Post-Incident Steps

After any emergency stop:

1. **Do not restart in LIVE mode immediately.** Set `TRADING_MODE=paper` first.
2. Review `logs/trading_bot.log` for the root cause
3. Verify all positions are closed (MT5 terminal → Trade tab should be empty)
4. Run database integrity check: `python -c "import sqlite3; c=sqlite3.connect('trading.db'); print(c.execute('PRAGMA integrity_check').fetchone())"`
5. Review the daily P&L and compare against the drawdown limit
6. Fix the root cause before switching back to LIVE

---

## 13. Daily Startup Checklist

> [!IMPORTANT]
> Complete this checklist every trading day **before** switching `TRADING_MODE` to `live`.

1. **Verify MT5 terminal is running and logged in** — check the account balance shown in the terminal matches expectations
2. **Check network connectivity** — ping the broker server; verify latency < 100 ms
3. **Check broker news calendar** — avoid opening new positions 30 min before/after high-impact news (NFP, Fed meetings, FOMC)
4. **Verify account equity** — confirm no overnight positions caused unexpected overnight loss
5. **Review yesterday's log** — scan `logs/trading_bot.log.YYYY-MM-DD` for any ERRORs or WARNING events
6. **Confirm `.env` settings are correct** — especially `TRADING_MODE`, `RISK_PER_TRADE`, `MAX_DAILY_LOSS_PCT`
7. **Check database backup from yesterday completed** — verify file exists in `backups/` with today's date
8. **Test Telegram notification** — run a test ping: `python -c "from run import Notifier; n=Notifier({'telegram_token':'...','telegram_chat_id':'...'}); n.send('Daily startup test OK')"`
9. **Verify XAUUSDm spread is normal** — spread should be < 30 points during London session; wider spreads during Asian session are normal
10. **Confirm session filter hours match today's schedule** — check for DST changes (clocks forward/back affects UTC offsets)
11. **Review yesterday's trades in MT5 history** — validate that all logged trades in DB match broker records
12. **Confirm risk limits are appropriate for current volatility** — consider reducing `RISK_PER_TRADE` if recent ATR is unusually high
13. **Start in PAPER mode for 15 minutes** — observe signal generation and ensure no immediate errors before switching to LIVE

---

## 14. Trading Modes

### 14.1 Mode Overview

```mermaid
flowchart LR
    A["BACKTEST\n(offline)"] --> B["PAPER\n(live data,\nsimulated fills)"]
    B --> C["SMALL_LIVE\n(real money,\n0.01 lot)"]
    C --> D["CONTROLLED_SCALE\n(normal risk,\n0.10+ lot)"]

    style A fill:#4a4a8a,color:#fff
    style B fill:#3a7a3a,color:#fff
    style C fill:#8a6a00,color:#fff
    style D fill:#8a1a1a,color:#fff
```

### 14.2 Mode Descriptions

| Mode | `TRADING_MODE` | MT5 Required | Real Orders | Lot Size | Use When |
|---|---|---|---|---|---|
| **BACKTEST** | `backtest` | No | No | N/A | Testing strategy changes offline using historical OHLCV data |
| **PAPER** | `paper` | Yes | No | Simulated | Validating a new config on live market data without financial risk |
| **SMALL_LIVE** | `live` | Yes | Yes — real | 0.01 min | First real-money test after paper validation; confirms broker execution |
| **CONTROLLED_SCALE** | `live` | Yes | Yes — real | per `RISK_PER_TRADE` | Normal production operation after SMALL_LIVE validates execution |

### 14.3 BACKTEST Mode

```dotenv
TRADING_MODE=backtest
```

- Reads historical bars from a local CSV or MT5 historical data
- No MT5 connection required at runtime
- All trades simulated with configurable slippage and spread models
- Outputs: equity curve, drawdown chart, trade list, Sharpe ratio, win rate
- **Run time:** seconds to minutes (depends on date range and symbol)

```powershell
$env:TRADING_MODE="backtest"
python run.py --from 2024-01-01 --to 2024-12-31 --output results/backtest_2024.json
```

### 14.4 PAPER Mode

```dotenv
TRADING_MODE=paper
```

- Connects to MT5 for live market data (requires terminal running)
- **No orders sent to the broker** (`dry_run=True` in the legacy CONFIG)
- Signals, risk checks, and position management all run exactly as in live
- Paper P&L tracked in the database for performance comparison
- **Minimum recommended paper duration:** 2 weeks before going live
- Watch for: signal frequency, drawdown, time-in-trade statistics

### 14.5 SMALL_LIVE Mode

```dotenv
TRADING_MODE=live
RISK_PER_TRADE=0.005    # reduce to 0.5% for small live
MAX_DAILY_LOSS_PCT=0.01 # tighter limit while validating
```

- Real orders placed with the broker at minimum lot size (0.01)
- Purpose: validate broker execution, requote rates, slippage, swap costs
- Run for at least **1 week** before scaling up
- Monitor execution quality: compare expected entry to actual fill price

### 14.6 CONTROLLED_SCALE Mode

```dotenv
TRADING_MODE=live
RISK_PER_TRADE=0.01      # back to normal 1%
MAX_DAILY_LOSS_PCT=0.02  # normal 2% limit
```

- Full production operation
- Position sizes determined by the dynamic risk formula: `equity × RISK_PER_TRADE / SL_pips`
- ADX dynamic scaling applies: actual risk ranges from `min_risk_scale × RISK_PER_TRADE` to `max_risk_scale × RISK_PER_TRADE`
- All risk guards fully active

---

## 15. Safe Mode Switching

### 15.1 Switching from PAPER to SMALL_LIVE

> [!WARNING]
> Never switch to LIVE mode while there are open simulated paper positions. Close all paper positions first, or they will be ignored by the live execution engine (different magic numbers or DB state).

```powershell
# Step 1: Stop the bot (wait for current loop to complete)
nssm stop TradingGoldBot
# OR press Ctrl+C in the terminal window

# Step 2: Verify no real open positions in MT5 terminal
# (Trade tab should be empty for magic number 888999)

# Step 3: Edit .env — change mode
notepad .env
# Set:  TRADING_MODE=live
#       RISK_PER_TRADE=0.005
#       MAX_DAILY_LOSS_PCT=0.01

# Step 4: Restart
nssm start TradingGoldBot
# OR
.\.venv\Scripts\Activate.ps1
python run.py

# Step 5: Watch the first 15 minutes in the log
Get-Content logs\trading_bot.log -Wait -Tail 50
```

### 15.2 Switching from LIVE back to PAPER (Emergency Downgrade)

```powershell
# Step 1: Activate kill switch (closes all live positions)
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/api/v1/control/emergency-stop" `
    -Headers @{ Authorization = "Bearer $token" } -Body '{"reason":"Mode downgrade"}'

# Step 2: Wait for confirmation alert — ALL positions closed

# Step 3: Stop service
nssm stop TradingGoldBot

# Step 4: Change mode in .env
(Get-Content .env) -replace 'TRADING_MODE=live','TRADING_MODE=paper' | Set-Content .env

# Step 5: Restart
nssm start TradingGoldBot
```

### 15.3 Switching LLM Provider at Runtime

```powershell
# Stop bot
nssm stop TradingGoldBot

# Change provider in .env
(Get-Content .env) -replace 'LLM_PROVIDER=.*','LLM_PROVIDER=openai' | Set-Content .env

# Restart
nssm start TradingGoldBot
```

The bot validates the new LLM on startup. If the health check fails, it falls back to ML-only automatically and sends a `LLM_UNAVAILABLE` alert.

---

## 16. Monitoring Dashboard

### 16.1 Dashboard URLs

| URL | Page | What It Shows |
|---|---|---|
| `http://localhost:8000/` | Overview | Equity curve, today's P&L, open positions count, bot status, mode indicator |
| `http://localhost:8000/positions` | Live Positions | All open positions with entry, current price, P&L, SL/TP, age, lot |
| `http://localhost:8000/history` | Trade History | Closed trades with entry/exit, profit, duration, signal score, AI confidence |
| `http://localhost:8000/ai` | AI Decisions | Last 50 signals: direction, score breakdown, LLM reasoning excerpt, result |
| `http://localhost:8000/risk` | Risk Center | Daily loss meter, drawdown gauge, margin usage, consecutive losses, kill switch |
| `http://localhost:8000/strategy` | Strategy | Signal score distribution, indicator values, session filter status, regime |
| `http://localhost:8000/settings` | Settings | Live view of all `.env` / Settings fields; hot-reload button |
| `http://localhost:8000/control` | Emergency | Kill switch button, mode switcher, manual position close, broker contact |
| `http://localhost:8000/docs` | API Docs | FastAPI auto-generated OpenAPI documentation (Swagger UI) |
| `http://localhost:8000/health` | Health | JSON health probe: MT5, DB, LLM, watchdog, last heartbeat timestamp |

### 16.2 Health Endpoint Response

```json
{
  "status": "healthy",
  "timestamp": "2026-10-08T09:15:00Z",
  "mode": "paper",
  "services": {
    "mt5": { "connected": true, "last_ping_ms": 12 },
    "database": { "ok": true, "size_mb": 4.2, "wal_mode": true },
    "llm": { "provider": "none", "reachable": null },
    "watchdog": { "last_heartbeat": "2026-10-08T09:14:58Z", "age_seconds": 2 }
  },
  "account": {
    "login": 261009880,
    "equity": 10000.00,
    "balance": 10000.00,
    "margin_level_pct": 0.0
  }
}
```

### 16.3 WebSocket Real-Time Feed

Connect to `ws://localhost:8000/ws` to receive real-time events:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");
ws.onmessage = (e) => {
  const event = JSON.parse(e.data);
  console.log(event.type, event.data);
  // event.type: "HEARTBEAT" | "TRADE_OPENED" | "TRADE_CLOSED" | "ALERT" | ...
};
```

---

## 17. Key Metrics to Watch Daily

| Metric | Target / Healthy Range | Warning Threshold | Critical Threshold | Where to Check |
|---|---|---|---|---|
| **Daily P&L (%)** | 0% – +3% | < −1% | < −`MAX_DAILY_LOSS_PCT` | Dashboard Overview |
| **Total Drawdown (%)** | < 5% | 7.5% | ≥ `MAX_DRAWDOWN_PCT` (10%) | Dashboard Risk |
| **Win Rate (last 20 trades)** | > 45% | < 40% | < 30% | Dashboard History |
| **Average R:R Ratio** | > 1.5 | < 1.2 | < 1.0 | Dashboard History |
| **Open Positions** | 0 – `MAX_OPEN_POSITIONS` | `MAX_OPEN_POSITIONS` | N/A | Dashboard Overview |
| **Margin Usage (%)** | < 30% | 40% | ≥ `MAX_MARGIN_USAGE` (50%) | Dashboard Risk |
| **Heartbeat Age (s)** | < 30 s | 60 s | ≥ `HEARTBEAT_MAX_AGE_SECONDS` (120 s) | `/health` endpoint |
| **MT5 Ping (ms)** | < 50 ms | 100 ms | > 200 ms | Dashboard Overview |
| **Signal Score (avg last 10)** | 55 – 80 | < 55 or > 90 | N/A | Dashboard Strategy |
| **AI Confidence (avg)** | 0.58 – 0.75 | < 0.58 | N/A | Dashboard AI |
| **Consecutive Losses** | 0 – 2 | 3 – 4 | ≥ `MAX_CONSECUTIVE_LOSSES` (5) | Dashboard Risk |
| **Spread (XAUUSDm, pts)** | < 20 pts | 30 – 50 pts | > `MAX_SPREAD_POINTS` (50 pts) | Dashboard Strategy |
| **Database size (MB)** | < 100 MB | 500 MB | > 1 GB | `/health` endpoint |
| **Log file size (MB)** | < 50 MB/day | 100 MB/day | > 500 MB/day | `logs/` folder |
| **LLM latency (ms)** | < 2000 ms | 5000 ms | Timeout (10 s) | Dashboard AI |
| **Equity vs. Yesterday (%)** | ≥ −2% | −2% to −5% | < −5% | Dashboard Overview |
| **Trades today** | 1 – 8 | 0 or > 8 | N/A | Dashboard History |

---

## Appendix A — Quick Reference Commands

```powershell
# Start bot (paper mode)
.\.venv\Scripts\Activate.ps1; python run.py

# Start web dashboard
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Tail live logs
Get-Content logs\trading_bot.log -Wait -Tail 100

# Check NSSM service status
nssm status TradingGoldBot

# Force restart service
nssm restart TradingGoldBot

# Run DB integrity check
python -c "import sqlite3; c=sqlite3.connect('trading.db'); print(c.execute('PRAGMA integrity_check').fetchone())"

# Manual DB backup
python -c "import sqlite3; s=sqlite3.connect('trading.db'); d=sqlite3.connect('backups/manual_backup.db'); s.backup(d); print('Done')"

# Test Telegram alert
python -c "
import requests
token='YOUR_TOKEN'; chat='YOUR_CHAT_ID'
r=requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
  data={'chat_id':chat,'text':'Test alert from Trading Bot OK'})
print(r.status_code)
"

# Check current settings
python -c "from core.settings import get_settings; s=get_settings(); print(s.model_dump())"

# View recent errors
Select-String -Path logs\trading_bot.log -Pattern '\[ERROR\]|\[CRITICAL\]' | Select-Object -Last 20
```

---

## Appendix B — Troubleshooting

| Symptom | Likely Cause | Resolution |
|---|---|---|
| `mt5.initialize()` returns False | Terminal not running or wrong `MT5_PATH` | Open MT5 terminal manually; verify path in `.env` |
| `mt5.login()` fails | Wrong credentials or server name | Double-check `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` in `.env` |
| `XAUUSDm not found` | Symbol not subscribed in Market Watch | Open Market Watch in MT5, right-click → Show All, find XAUUSDm |
| Bot starts but places no trades | Session filter active (outside London/NY) | Normal — wait for 07:00–21:00 UTC window |
| Bot starts but places no trades | ADX below `ADX_MIN` (non-trending market) | Normal — bot requires `ADX >= 20` to enter |
| Bot starts but places no trades | Score below `MIN_SIGNAL_SCORE` (55) | Review strategy parameters; check indicator values in dashboard |
| `ModuleNotFoundError: MetaTrader5` | Running on non-Windows OS | Must use Windows; MT5 API is Windows-only |
| Telegram messages not received | Wrong token or chat ID | Re-verify token via `/getUpdates` call |
| Dashboard returns 401 | JWT expired or wrong credentials | Re-login at `http://localhost:8000/login` |
| High CPU usage | AI retraining running | Normal every `AI_RETRAIN_BARS` (50) bars; lasts 2–5 seconds |
| Database locked error | Multiple processes writing simultaneously | Ensure only one bot instance runs; WAL mode should handle concurrent reads |

---

*Last updated: 2026-10-08 | Version: 1.0.0 | Maintained by: Operations Team*
