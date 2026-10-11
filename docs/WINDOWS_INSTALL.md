# Running Everything on Windows

Single-machine setup: trading bot, API, database and MT5 bridge all on one
Windows PC. No Ubuntu, no Docker, no KVM required.

---

## 1. Prerequisites

| Requirement | Check |
|---|---|
| Windows 10/11 | — |
| Python 3.12 | `py -3.12 --version` |
| MetaTrader 5 terminal | Installed, logged in, **Algo Trading enabled** |
| PostgreSQL *(optional)* | Only if you want it instead of SQLite |

> **MT5 is Windows-only.** That is why this file exists. MetaTrader5 has no
> Linux wheel, which is the entire reason a bridge exists.

---

## 2. Install

Open **PowerShell** in the project folder:

```powershell
cd C:\path\to\Trading-gold

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements-windows.txt
```

`requirements-windows.txt` is the **only** file you need on Windows. It
includes MetaTrader5 plus everything else (API, database, bridge).

If `MetaTrader5` fails to install:

```powershell
# Install the wheel manually, then repeat the requirements file
pip install https://files.pythonhosted.org/packages/.../MetaTrader5-5.0.45-py3-none-win_amd64.whl
pip install -r requirements-windows.txt
```

Verify:

```powershell
python -c "import MetaTrader5; print('MT5 OK')"
python -c "import fastapi, sqlalchemy; print('core OK')"
```

---

## 3. Configure

Copy the template:

```powershell
Copy-Item .env.example .env
notepad .env
```

Minimum settings for **paper trading** (nothing is sent to the broker):

```
TRADING_MODE=paper
DATABASE_URL=sqlite:///./trading.db
SYMBOLS=XAUUSDm
```

For **PostgreSQL** instead of SQLite (if you installed it):

```
DATABASE_URL=postgresql://trading:YOUR_PASSWORD@localhost:5432/trading
```

To enable the bridge inside the same machine:

```
MT5_BRIDGE_HOST=127.0.0.1
MT5_BRIDGE_PORT=8100
MT5_BRIDGE_TOKEN=<generate with: python -c "import secrets;print(secrets.token_hex(32))">
MT5_BRIDGE_ALLOW_WRITES=0
MT5_BRIDGE_EXPECTED_LOGIN=<your MT5 account number>
```

---

## 4. Run

### Option A — the trading bot (what you most likely want)

```powershell
.\.venv\Scripts\Activate.ps1
py run.py
```

MT5 terminal must be **open and logged in** first. With `TRADING_MODE=paper`
the bot computes signals and simulates orders — nothing reaches the broker.

Dashboard: <http://127.0.0.1:8000>

### Option B — the API

```powershell
py -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>

### Option C — the MT5 bridge (only if the Linux core needs this PC)

```powershell
py bridge\run.py
```

Keep the window open. Read-only unless `MT5_BRIDGE_ALLOW_WRITES=1`.

### Option D — the GUI

```powershell
py gui.py
```

Tkinter ships with Windows Python. Note it imports `run.py`, so MT5 must be
installed.

---

## 5. Database

**SQLite (default)** needs no setup — the file is created automatically.

**PostgreSQL** requires one-time setup:

```powershell
# Create the database (run in psql)
CREATE USER trading WITH PASSWORD 'your_password';
CREATE DATABASE trading OWNER trading;

# Apply the schema
alembic upgrade head
```

Confirm:

```powershell
alembic current
```

---

## 6. Verify

```powershell
# Bot code is loadable
python -c "import ast; ast.parse(open('run.py').read()); print('run.py OK')"

# Database layer works
python -c "from core.settings import get_settings; print(get_settings().database_url)"

# Tests pass (never connects to a broker)
python -m pytest tests\ -q
```

> Use `pytest tests\` — **never bare `pytest`**. The repo-root `test_mt5.py`
> opens a live MT5 terminal at import time, and pytest collects it.

---

## 7. Safety

| Rule | Detail |
|---|---|
| Default is paper | `TRADING_MODE=paper` — real orders need explicit `live` |
| Unknown mode | Falls back to paper, never fails open to live |
| Writes gated | The bridge refuses order submission unless explicitly enabled |
| Credentials | Read from `.env`; never logged or returned by the API |

`run.py` still contains hardcoded credential *fallbacks* at `run.py:130-132`
for backwards compatibility. **Set `MT5_LOGIN`, `MT5_PASSWORD` and
`MT5_SERVER` in `.env` so those values are never used.**

---

## 8. Common problems

| Symptom | Fix |
|---|---|
| `No module named MetaTrader5` | `pip install -r requirements-windows.txt` inside the activated venv |
| `mt5.initialize()` fails | Open and log into the MT5 terminal first |
| MT5 initialize hangs | `fix_mt5.bat` restarts the terminal and tests the connection |
| `ImportError: greenlet` | Already pinned in the requirements file; reinstall |
| `execution of scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `sqlite3.OperationalError: database is locked` | Close other tools holding `trading.db` |
| Bare `pytest` aborts | Use `pytest tests\` |
| Nothing appears in the dashboard | Check the bot log; MT5 must be connected |

---

## 9. Stopping

```
Ctrl+C
```

Close the MT5 terminal. There are no background services or containers to
shut down on this setup.