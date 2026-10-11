# Windows Installation

Complete installation on **Windows 11 x64**. No Docker, WSL, Linux or VM.

---

## 1. Prerequisites

| Requirement | Version | Source |
|---|---|---|
| Windows | 11 x64 | — |
| Python | 3.12 x64 | https://www.python.org/downloads/windows/ |
| MetaTrader 5 | latest | https://www.metatrader5.com/en/download (or your broker) |
| PostgreSQL *(optional)* | 16.x | https://www.postgresql.org/download/windows/ |

During Python installation **tick "Add python.exe to PATH"**.

Verify:

```powershell
py -3.12 --version
```

---

## 2. Install

```powershell
git clone <your-repo-url> Trading-gold
cd Trading-gold
```

### Option A — guided (recommended)

```powershell
.\scripts\windows\doctor.ps1      # report current state
.\scripts\windows\install.ps1     # do everything
.\scripts\windows\start.ps1       # start the API
```

### Option B — step by step

```powershell
.\scripts\windows\setup-venv.ps1
.\scripts\windows\install-dependencies.ps1
Copy-Item .env.example .env
notepad .env
.\scripts\windows\setup-database.ps1
.\scripts\windows\migrate-database.ps1
.\scripts\windows\start.ps1
```

---

## 3. Configure

`install.ps1` creates `.env` from the template if absent. **It never
overwrites an existing `.env`.**

Minimum for safe operation:

```dotenv
TRADING_MODE=paper
DATABASE_URL=sqlite:///./trading.db
```

For PostgreSQL:

```dotenv
DATABASE_URL=postgresql://trading:YOUR_PASSWORD@localhost:5432/trading
```

Full reference: [CONFIGURATION.md](./CONFIGURATION.md).

> `.env` contains credentials. It is git-ignored — never commit it.

---

## 4. Start

```powershell
.\scripts\windows\start.ps1
```

| URL | Purpose |
|---|---|
| <http://127.0.0.1:8000> | Dashboard |
| <http://127.0.0.1:8000/docs> | Interactive API docs |
| <http://127.0.0.1:8000/api/health> | Liveness |
| <http://127.0.0.1:8000/api/ready> | Readiness (checks the database) |

Start the trading bot:

```powershell
.\scripts\windows\start.ps1 -Bot
```

Requires an open, logged-in MetaTrader 5 terminal. Without it, `start.ps1`
**refuses** rather than starting a bot that cannot trade.

Check state any time:

```powershell
.\scripts\windows\status.ps1
```

---

## 5. Stop

```powershell
.\scripts\windows\stop.ps1
.\scripts\windows\stop.ps1 -Target api
.\scripts\windows\stop.ps1 -WhatIf      # preview
```

PostgreSQL is a Windows service and is not touched:

```powershell
Stop-Service postgresql*
```

---

## 6. Test

```powershell
.\scripts\windows\test.ps1
.\scripts\windows\test.ps1 -Keyword "instrument"
```

Never bare `pytest` — always `pytest tests\`.

---

## 7. PostgreSQL (optional)

SQLite works out of the box. For PostgreSQL:

1. Install from postgresql.org and note the superuser password.
2. Create the role and database:

```sql
CREATE USER trading WITH PASSWORD 'your_password';
CREATE DATABASE trading OWNER trading;
```

3. Set `DATABASE_URL` in `.env`.
4. Apply the schema:

```powershell
.\scripts\windows\migrate-database.ps1
```

Details: [DATABASE.md](./DATABASE.md).

---

## 8. MetaTrader 5

1. Install the terminal and log in.
2. Tools → Options → Expert Advisors → enable **Algo Trading**.
3. Verify from Python:

```powershell
.\.venv\Scripts\python.exe -c "import MetaTrader5 as mt5; print(mt5.initialize())"
```

Setup detail: [MT5_SETUP.md](./MT5_SETUP.md).

> Without MT5 the API and paper research still work. Broker features do not.

---

## 9. Automatic startup (optional)

MT5 often needs an interactive session. Run `start.ps1 -Bot` from a desktop
session rather than as a headless service.

---

## 10. Manual installation steps

If you prefer not to use the scripts:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named MetaTrader5` | Install `requirements-windows.txt` in the activated venv |
| `execution of scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `psql not found` | Add `C:\Program Files\PostgreSQL\16\bin` to PATH |
| `table X already exists` during migrate | `.\scripts\windows\migrate-database.ps1 -Stamp` |
| Port 8000 already in use | `.\scripts\windows\start.ps1 -Port 8080` |
| `database is locked` (SQLite) | Close other tools using `trading.db` |

More: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).