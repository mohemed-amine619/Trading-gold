# Troubleshooting

Start here:

```powershell
.\scripts\windows\doctor.ps1
```

It reports every check as PASS / WARN / FAIL. Exit code is non-zero when
something failed, so it works in automation.

---

## Installation

### `execution of scripts is disabled`

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Per-user; does not change machine policy.

### `No module named MetaTrader5`

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt
```

If the wheel is unavailable for your Python version, download the `cp312
win_amd64` wheel from <https://pypi.org/project/MetaTrader5/#files> and
install it manually.

### Python not found

Install from python.org and tick **"Add python.exe to PATH"**.

---

## Startup

### `Port 8000 is already in use`

```powershell
.\scripts\windows\start.ps1 -Port 8080
```

Or find the holder:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
```

### API did not become ready

```powershell
Get-Content .\logs\api.log -Tail 40
Get-Content .\logs\api.log.err -Tail 40
```

Common causes: database unreachable, migration not applied.

### `/api/ready` returns 503

The process is alive but the database is not. This is intentional — it
distinguishes "waiting for the database" from "broken process".

```powershell
.\scripts\windows\setup-database.ps1
```

---

## Database

### `could not connect to server`

```powershell
Get-Service postgresql*
```

Start it, or start it automatically:

```powershell
Set-Service postgresql* -StartupType Automatic
```

### `password authentication failed`

Check `DATABASE_URL`. Special characters must be URL-encoded.

### `database "trading" does not exist`

```sql
CREATE DATABASE trading OWNER trading;
```

The script deliberately will not create it for you.

### `table X already exists` during migration

The database has the schema but no Alembic revision — a legacy `create_all`
database.

```powershell
.\scripts\windows\migrate-database.ps1 -Stamp
```

### `database is locked` (SQLite)

Another process holds the file. Close the GUI, editors, or any tool using
`trading.db`. Windows locks SQLite harder than Linux; the Windows default
uses the DELETE journal for this reason.

---

## MetaTrader 5

### `initialize()` returns False

The terminal must be **open and logged in**, with Algo Trading enabled
(Tools → Options → Expert Advisors).

### `initialize()` hangs

Stale IPC state:

```powershell
Stop-Process -Name terminal64 -Force
# reopen the terminal, wait for charts, then retry
```

### Symbol returns nothing

Your broker may not offer it:

```powershell
.\.venv\Scripts\python.exe -c "import MetaTrader5 as mt5; mt5.initialize(); print(len(mt5.symbols_get()))"
```

### Prices are stale

The market is closed, or it is outside session hours. Check the broker's
trading schedule for that instrument.

---

## Bot

### Bot refuses to start

`start.ps1 -Bot` refuses when MetaTrader5 is not importable or the terminal is
not running. This is deliberate — it will not start a bot that cannot trade.

### Bot sends nothing in paper mode

That is correct. Paper mode simulates orders.

Check the mode actually in effect:

```powershell
curl http://127.0.0.1:8000/api/ready
```

```json
{"status": "ready", "trading_mode": "paper"}
```

### Dashboard shows the wrong mode

Fixed to read `dry_run`. If it disagrees with `/api/ready`, restart the bot so
it reloads `.env`.

---

## Configuration

### Changes to `.env` are ignored

Stop and restart:

```powershell
.\scripts\windows\stop.ps1
.\scripts\windows\start.ps1
```

### `TRADING_MODE=live` will not start

Live requires `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` and a
`DASHBOARD_SECRET_KEY` that is not `CHANGE_ME`. `doctor.ps1` names the missing
ones.

---

## PowerShell tooling

### `Get-PortListeners` cannot verify a port

Neither `Get-NetTCPConnection` nor `netstat`/`ss` was found. Rare; report it.

### Logs are not being written

```powershell
Test-Path .\logs
```

The application creates it at startup. If missing, permissions on the project
folder are the likely cause.

---

## Getting more detail

```powershell
.\scripts\windows\doctor.ps1
Get-Content .\logs\api.log -Tail 100
Get-Content .\logs\api.log.err -Tail 100
.\.venv\Scripts\python.exe -c "from core.settings import get_settings; print(get_settings())"
```

Logs redact credentials automatically, so they are safe to share.