# PROJECT AUDIT — TRADING-GOLD PLATFORM

> **Audit Date:** 2026-10-12  
> **Target Environment:** Native Windows 11 x64  
> **Database Mandate:** SQLite ONLY (Zero external database servers)  
> **Test Baseline:** 217 passed, 0 failed, 2 warnings (Python 3.12, pytest)  
> **Repository Branch:** `main` (clean working tree)

---

## 1. EXECUTIVE SUMMARY

The **Trading-gold** repository is an algorithmic quantitative trading system designed for Windows 11 with MetaTrader 5 (MT5). The codebase currently exists as two parallel layers:
1. **The Live Production Monolith (`run.py`, ~1,890 lines):** A self-contained, real-time trading engine running M15/H1/H4 multi-timeframe confluence scoring, an ML soft-voting ensemble (RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting), dynamic ADX risk scaling, breakeven/trailing stops with profit giveback protection, and an embedded FastAPI web dashboard (`dashboard.html`). It historically recorded trades into an ad-hoc SQLite file (`trading_bot.db`).
2. **The Modular Architecture (`core/` + `api/` + `bridge/`):** A clean domain-driven architecture with 18 SQLAlchemy ORM models, an abstract broker adapter (`MT5Adapter`, `MockMT5Adapter`, `RealMT5Adapter`), an idempotency ledger, a broker-neutral instrument registry (`core/instruments/`), a centralized risk gate (`RiskEngine`), and a standalone FastAPI app (`api/main.py`).

The primary challenge of this platform is **bridging the two layers** into a unified, bulletproof Windows 11 platform using **SQLite exclusively**, eliminating external database dependencies, preventing split-brain persistence, and ensuring rigorous risk governance and broker reconciliation.

---

## 2. COMPREHENSIVE AREA-BY-AREA AUDIT

### 2.1 Directory Structure & Entry Points
* **Live Bot Entry Point:** `run.py` — starts the multi-symbol trading manager (`StandaloneAIBot`) and runs an embedded Uvicorn/FastAPI server hosting `dashboard.html` at `http://127.0.0.1:8000`.
* **Modular API Entry Point:** `api/main.py` — standalone FastAPI app (`v2.0`) with `/api/health`, `/api/ready`, `/api/config`.
* **MT5 Bridge Entry Point:** `bridge/run.py` / `bridge/app.py` — HTTP bridge service on Windows for remote cores.
* **PowerShell Tooling:** `scripts/windows/` (15 scripts) — manages venv, dependencies, database, processes, testing, backup, and restore.

### 2.2 Specifications & Project State
* `AGENTS.md` and `README.md` clearly document the Windows 11 mandate, the dual-system divergence, and safety defaults.
* **Paper trading default:** `TRADING_MODE` defaults to `paper`. Real orders strictly require `TRADING_MODE=live`.
* **No Martingale / Grid:** Prohibited across the platform.

### 2.3 Git Status & Recent History
* Working directory is clean on `main`, tracked to `origin/main`.
* Dead code, PyInstaller specs, and legacy Tkinter GUIs (`gui.py`, `position_dashboard.py`, `b.bat`, `build.bat`) have been safely removed.
* Linux/Docker files (`Dockerfile`, `compose.yaml`) were purged to keep the repository dedicated to native Windows.

### 2.4 Dependencies & Python Environment
* Python 3.12 x64 is the target interpreter.
* `requirements-windows.txt` is the pinned requirements file, aliased via `requirements.txt`.
* **Defect identified:** `requirements-windows.txt` currently contains `asyncpg==0.32.0` and `psycopg[binary]==3.3.6` (PostgreSQL drivers), violating the SQLite-only mandate.

### 2.5 Startup & Diagnostics
* `scripts/windows/doctor.ps1` diagnoses Python, MT5, pathing, and environment.
* `scripts/windows/start.ps1` orchestrates startup.
* **Defect identified:** `start.ps1`, `doctor.ps1`, `setup-database.ps1`, `backup.ps1`, and `restore.ps1` contain residual PostgreSQL service checks and probing routines (`Get-PostgresService`, `psql`, `pg_dump`).

### 2.6 Strategy Engine & Signal Generation
* Multi-timeframe confluence: M15 entry bar + H1 trend filter + H4 higher-trend filter.
* Layered scoring:
  - Layer 1: ML Ensemble directional probability (0–40 pts).
  - Layer 2: Trend alignment (0–25 pts) — EMA-50/200 + ADX.
  - Layer 3: Momentum / Oscillators (0–20 pts) — RSI, MACD histogram, Stochastic.
  - Layer 4: Session filter (0–15 pts) — London & New York liquid hours.
* Minimum signal threshold required (`min_signal_score = 60`).

### 2.7 Market Data & Historical Storage
* Live bar fetching via `mt5.copy_rates_from_pos()`. Drops forming candle to avoid look-ahead bias on closed bars.
* Indicator calculation wrapped with `pandas_ta` (EMA, ATR, RSI, MACD, Stoch, ADX, Keltner, OBV).
* Data integrity check: detects missing candles, NaN feature rows, and zero ATR.

### 2.8 MetaTrader 5 Integration & Broker Abstraction
* `core/mt5/adapter.py`: defines `MT5Adapter` ABC, `MockMT5Adapter` (deterministic for CI tests), and `RealMT5Adapter` (lazy-imports `MetaTrader5` on Windows).
* `run.py`: uses direct MT5 API with auto-reconnection and active terminal session reuse.
* Handles broker errors: `TRADE_RETCODE_REQUOTE`, `TRADE_RETCODE_CONNECTION`, `TRADE_RETCODE_TIMEOUT`.

### 2.9 Order Execution & Reconciliation
* `core/execution/idempotency.py`: `IdempotencyLedger` prevents duplicate orders within configurable TTL windows.
* `core/execution/engine.py`: retries on requote/slippage, checks execution state.
* `run.py`: `_modify_sltp`, `_modify_sl`, `_close_position`, and partial fills.

### 2.10 Risk Controls & Account Protection
* `core/risk/engine.py`: comprehensive `RiskEngine.approve()` gate. Enforces max risk per trade, max daily loss, max weekly loss, max drawdown, margin limits, consecutive-loss cooldown, and emergency stop.
* `run.py`: dynamic risk sizing (ADX scaled), hard per-trade equity stop, losing-position timeout cut, and giveback protection (locks 70% of peak profit).
* Breakeven with buffer (+0.15 ATR) and accelerating trailing stop with breakeven floor.

### 2.11 Backtesting & Statistical Validation
* `core/backtesting/engine.py`: event-driven backtesting structure.
* Evaluates win rate, profit factor, max drawdown, Sharpe, and Sortino ratios.

### 2.12 Database Architecture & Persistence
* **Mandate:** SQLite ONLY (`data/trading.db` or `./trading.db`).
* `core/database/engine.py`: supports SQLite sync (`create_engine`) and async (`create_async_engine` with `aiosqlite`). Configures `PRAGMA busy_timeout=5000`, `PRAGMA foreign_keys=ON`, and journal mode (DELETE on Windows to avoid locking, WAL optional).
* Alembic migrations: `alembic/versions/aedbfe3f600c_initial_schema.py` defines 18 core tables.
* **Defect identified:** `run.py` writes closed trades to `trading_bot.db` via raw `sqlite3` instead of persisting to the unified `trading.db` schema.

### 2.13 API, Dashboard & Monitoring
* `dashboard.html`: single-page UI visualizing live accounts, symbols, active trades, floating P&L, historical trades, and equity curve.
* Communicates with `/api/status`, `/api/positions`, `/api/history`, `/api/control`.
* CORS enabled for local interface.

### 2.14 Windows Tooling & Scripts
* Complete PowerShell suite under `scripts/windows/`.
* No external administrative privilege required.

### 2.15 Testing & Verification Baseline
* 217 passing tests in `tests/`:
  - `test_instruments.py` (55 tests)
  - `test_execution_safety.py` (31 tests)
  - `test_bridge.py` (31 tests)
  - `test_safety_paper_default.py` (18 tests)
  - `test_windows_compat.py` (18 tests)
  - `test_risk_engine.py`, `test_position.py`, `test_mt5_adapter.py`, `test_ai.py`, etc. (64 tests)

---

## 3. FEATURE CLASSIFICATION MATRIX

| Component | Status | Details |
|---|---|---|
| **Windows MT5 Integration** | Working & Verified | Connects natively, reuses open session, fallback on disconnect |
| **Paper Trading Guard** | Working & Verified | Defaults to paper mode; tested by 18 unit tests |
| **Instrument Registry** | Working & Verified | 55 unit tests verify multi-asset sizing without gold assumptions |
| **Execution Idempotency** | Working & Verified | 31 unit tests guard against duplicate orders and requote loops |
| **Windows Logging & Redaction** | Working & Verified | UTC timestamps, auto-created `logs/`, credential masking |
| **Trailing Stop & Breakeven** | Working & Verified | Breakeven + buffer, trailing stop with inviolable profit floor |
| **SQLite Engine & Pragma** | Working & Verified | Busy timeout (5000ms), foreign keys enabled, sync & async support |
| **Alembic Migrations** | Implemented, Needs Alignment | 18 tables created, but `run.py` still bypasses it for `trading_bot.db` |
| **PowerShell Tooling** | Partially Implemented | Full set exists, but contains legacy PostgreSQL probes to be removed |
| **SQLite Backup & Restore** | Partially Implemented | Needs dedicated SQLite online backup (`sqlite3.backup` / `.backup`) |
| **Central Risk Engine Wiring** | Implemented in `core`, Unwired in `run.py` | `core/risk/engine.py` is comprehensive but `run.py` runs its own checks |
| **PostgreSQL Dependencies** | Deprecated / Broken | Must be purged from `requirements-windows.txt` and scripts |

---

## 4. HIGHEST-PRIORITY ACTIONS FOR PHASE 1 & 2

1. **Purge PostgreSQL Dependencies:**
   - Remove `asyncpg` and `psycopg` from `requirements-windows.txt`.
   - Purge PostgreSQL references from `scripts/windows/*.ps1` (`setup-database.ps1`, `start.ps1`, `backup.ps1`, `restore.ps1`, `doctor.ps1`, `common.ps1`).
2. **Standardize SQLite-Only Architecture:**
   - Standardize database path on `data/trading.db`.
   - Implement native SQLite backup and restore routines with integrity checks (`PRAGMA integrity_check`).
3. **Harmonize Persistence:**
   - Connect `run.py` trade recording to the unified SQLite schema or ensure seamless compatibility without split databases.
4. **Wire Central Risk & Idempotency into Execution Loop.**
