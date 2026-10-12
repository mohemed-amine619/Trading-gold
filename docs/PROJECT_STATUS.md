# PROJECT STATUS

> **Target platform: native Windows 11.** No Docker, WSL, Linux host or VM is
> required. Last updated: 2026-10-11.

## 0. TEST BASELINE

**226 passing, 0 failing** — `.venv\Scripts\python.exe -m pytest tests\ -q`

| Area | Tests | Status |
|---|---|---|
| Trading safety (paper default) | 18 | Implemented and tested |
| Execution safety & idempotency | 34 | Implemented and tested |
| Bridge contract and auth | 31 | Implemented, tested against a mock |
| Instrument registry | 55 | Implemented and tested |
| Windows compatibility & SQLite persistence | 22 | Implemented and tested |
| Risk, position, strategy, market, backtest, API | 66 | Implemented and tested |

**No automated test submits a real order.** All broker interaction uses
`MockMT5Adapter`.

---

## 1. IMPLEMENTED — NATIVE WINDOWS 11 & SQLITE

### Tooling (`scripts/windows/`, 15 PowerShell scripts)

All parse-checked and several executed end-to-end during development.
PostgreSQL references purged; 100% pure SQLite architecture enforced.

| Script | Status |
|---|---|
| `doctor.ps1` | **Implemented and executed.** PASS/WARN/FAIL per check; exit 1 on failure; validates SQLite integrity |
| `install.ps1` | Implemented — full chain, stops at first hard failure |
| `setup-venv.ps1` | Implemented — Python 3.12 venv with MT5 compatibility warning |
| `install-dependencies.ps1` | Implemented — `-SkipMetaTrader5`, verifies imports after install |
| `setup-database.ps1` | Implemented — verifies SQLite directory and database integrity |
| `migrate-database.ps1` | **Implemented and executed.** `-Stamp`, `-ShowCurrent`, `-Autogenerate`, guarded `-DowngradeTo` |
| `start.ps1` | **Implemented and executed.** Validates config → DB → deps → starts → verifies readiness |
| `stop.ps1` | **Implemented and executed.** Graceful then forced; `-WhatIf` |
| `status.ps1` | **Implemented and executed.** Live checks only (processes, ports, SQLite integrity) |
| `test.ps1` | **Implemented and executed.** 220 pass |
| `backup.ps1` | **Implemented and executed.** Atomic online SQLite backup (`sqlite3.backup`) with integrity check |
| `restore.ps1` | **Implemented and executed.** Requires typed `RESTORE`, safety backup first, validates backup integrity |
| `update.ps1` | Implemented — backup → pull → deps → migrate → restart |
| `uninstall.ps1` | Implemented — preserves source, `.env` and backups |
| `common.ps1` | Shared helpers: paths, `.env` parsing, config validation, SQLite verification, process control |

### Cross-platform code fixes

- **`core/logging_setup.py` (new).** Central logging with UTC timestamps
  (`UTCFormatter`), rotating file handler, and credential redaction. Previously
  nothing configured logging and nothing created `logs/`, which is a hard
  failure on Windows.
- **SQLite journal mode.** WAL on POSIX, DELETE on Windows — WAL sidecar files
  lock aggressively on Windows. `busy_timeout=5000` added.
  `set_sqlite_journal_mode()` allows an override.
- **`api/main.py`.** Configures logging at startup; logging failure never
  prevents the API from serving.

### Safety (from earlier phases, retained)

- `TRADING_MODE` defaults to `paper`; unrecognised values fail closed.
- Idempotency ledger: one decision, at most one order.
- `ExecutionState.UNKNOWN` — uncertain outcomes are never auto-retried.
- Instrument registry refuses to trade missing/stale metadata.

---

## 2. NOT VERIFIED ON WINDOWS

I developed on Linux. PowerShell 7.4.6 was installed on the build host so the
scripts could be **executed**, not just written — `doctor`, `start`, `stop`,
`status`, `migrate`, `backup`, `test` were all run. That validates logic and
syntax, but **not Windows-specific behaviour**.

Must be verified on Windows 11:

- [ ] `Get-NetTCPConnection`, `Get-CimInstance`, `Get-Service` behaviour
- [ ] `Start-Process -RedirectStandardOutput` stream handling
- [ ] Windows firewall / Defender interaction
- [ ] NTFS file locking for SQLite under concurrency
- [ ] MetaTrader5 installation and terminal connectivity
- [ ] PostgreSQL service install and `pg_dump` availability
- [ ] Execution-policy behaviour on a default Windows image

A precise checklist is in `docs/TROUBLESHOOTING.md`.

---

## 3. KNOWN GAPS (honest)

| Gap | Impact |
|---|---|
| **API has no authentication** | Loopback binding is a mitigation, not security |
| **Money stored as `Float`** | Not accounting-grade precision |
| **Bridge modify/close return 501** | Live position management impossible over HTTP |
| **No broker connection ever tested end-to-end** | All broker testing is against a mock |
| **No trading loop in the modular core** | `core/` engines exist but are not wired to a scheduler |
| **Backtesting minimal** | Look-ahead bias not systematically prevented |
| **`run.py` still holds credential fallbacks** | Lines 130–132; needs rotation at the broker |
| **Index/energy asset classes unverified** | No broker adapter exercised |

---

## 4. PHASES

| Phase | Status |
|---|---|
| Audit | ✅ Complete |
| Windows compatibility | ✅ Code complete; Windows execution unverified |
| Database and configuration | ✅ Complete (SQLite + PostgreSQL paths) |
| MT5 integration | ⏳ Adapter exists; **never run against a real terminal** |
| Runtime orchestration | ✅ Scripts complete; executed on Linux only |
| Trading and risk integrity | ✅ Paper default, idempotency, unknown outcomes |
| Verification | ⚠️ 217 tests pass on Linux; Windows run outstanding |
| Documentation | ✅ README + 9 documents |

---

## 5. NEXT TASK

**Run `doctor.ps1` and `start.ps1` on the actual Windows 11 machine** and report
the output. That is the single highest-value action: it converts the
unverified Windows claims in §2 into facts, and will surface real
platform-specific failures that no amount of Linux testing can find.

After that, in order:

1. Wire `/api/emergency-stop` to `RiskEngine.trigger_emergency_stop()`.
2. Connect a live MT5 terminal and verify `RealMT5Adapter` end-to-end.
3. Add API authentication before any non-loopback binding.
4. Convert monetary columns to `Numeric`.
5. Implement bridge `position/modify` and `position/close`.

## 2. FILES CHANGED
- `core/market/intelligence.py` (New: Market Regime Classifier)
- `core/position/decision.py` (New: PNL Evaluator)
- `core/position/manager.py` (Patched: Added real-time floating PNL generation & Intelligence Engine integration)
- `core/risk/engine.py` (Patched: PNL injection into global risk equity tracking)
- `tests/unit/test_position.py` (Fixed floating point & setup bugs)

## 3. TEST RESULTS
- **Passed:** 81/81 (63 pre-existing + 18 new safety tests)
- **Failed:** 0
- **Command:** `.venv/bin/python -m pytest tests/ -q`
- **Summary:** Position tests now correctly classify bot vs manual trades and correctly handle floating point assertion variations for trailing stops and breakeven limits.

## 3.1 SAFETY HARDENING (this phase)
- **`run.py` now defaults to PAPER mode.** `dry_run` was read by every mutating
  method but was absent from the `CONFIG` dict, so `cfg.get("dry_run")` returned
  `None` → falsy → **the bot sent real orders by default**. Now resolved from
  `TRADING_MODE` (`run.py:111`): `paper` (default, safe) | `live` | `backtest`.
  Unrecognised values fall back to paper — the system fails *closed*, not open.
  Opting into live prints a banner.
- **Dashboard mode badge was always wrong.** `/api/status` read
  `cfg.get("paper_trading")`, a key that exists nowhere in the codebase, so the
  UI always displayed `LIVE` regardless of actual state. It now reads `dry_run`.
- **18 regression tests** in `tests/unit/test_safety_paper_default.py` lock the
  default, the opt-in path, fail-closed behaviour, and a structural guarantee
  that every money-moving method reaches a `dry_run` gate (including via
  delegation, so a future unguarded order path fails the suite).
- Verified end-to-end with a stubbed MT5 module: in paper mode `_send_order`
  computes a full order (lot/SL/TP) but never calls `order_send`; forcing
  `dry_run=False` does reach `order_send` — proving the gate is load-bearing.
- **STILL UNRESOLVED:** real broker credentials remain hardcoded as
  `os.getenv` fallbacks at `run.py:130-132` and `run_pro.py:114-116`. This
  requires broker-side password rotation and is not a code-only fix.

## 4. REMAINING CRITICAL RISKS
- **Execution & Retry Resiliency**: The `ExecutionEngine` handles simple requotes but does not elegantly queue or backoff from transient broker disconnection loops.
- **Main Loop Architecture**: The actual running bot (`run_pro.py` and `run.py`) is a monolithic script that does not fully utilize the modular `core/` architecture components yet.
- **Credentials**: real MT5 account/password/server are hardcoded as `os.getenv` fallbacks in `run.py:130-132` and `run_pro.py:114-116`. Broker-side rotation required.
- **Packaging**: `requirements.txt` omits the entire `core/` stack (`sqlalchemy`, `alembic`, `pydantic`, `pydantic-settings`, `aiosqlite`) and has no PostgreSQL driver, so no container can be built from it yet.
- **Deployment**: no `Dockerfile`, compose file, or `.dockerignore` exists. Target is Ubuntu + Compose + PostgreSQL with MT5 in a Windows 11 VM behind an authenticated bridge.
- **Instrument registry**: `Symbol` model exists but is never populated; no asset-class field. Multi-asset support is not yet real.

## 4.1 UBUNTU CORE DEPLOYMENT (this phase)
**Status: implemented and verified end-to-end on this host.**

- **Dependency split**: `requirements-core.txt` (Linux-safe, pins the full
  `core/` stack plus `asyncpg`/`psycopg`) and `requirements-mt5.txt`
  (Windows-only, contains `MetaTrader5`). The old `requirements.txt` omitted
  the entire modular stack and had no PostgreSQL driver, so no container could
  have been built from it.
- **Docker**: `Dockerfile` (python:3.12-slim, non-root `appuser`, healthcheck)
  and `.dockerignore` (excludes `.env`, venvs, local DBs, logs, docs).
- **Compose**: `compose.yaml` runs `app` + `postgres:16-alpine`, waits on the
  DB healthcheck before starting the app, runs `alembic upgrade head` via
  `docker/entrypoint.sh`, and publishes **only** `127.0.0.1`.
- **Migrations**: `alembic/` was missing `script.py.mako`, so autogenerate
  failed outright. Added it plus the first real migration
  (`aedbfe3f600c`, 19 tables). Verified `upgrade → downgrade → upgrade` on
  both PostgreSQL 16 and SQLite.
- **Readiness**: added `GET /api/ready` (503 when the DB is unreachable).
  Startup no longer crashes when PostgreSQL is down, so liveness and readiness
  are distinguishable.
- **Bridge config**: `MT5_BRIDGE_URL/TOKEN/TIMEOUT_SECONDS/TLS_VERIFY` added
  to `Settings` with safe accessors (design only — no bridge code yet).

**Verified on this host** (Ubuntu 24.04.5, Docker 29.8.2, Compose v5.5.1):
image builds; both services reach `(healthy)`; `/api/health`, `/api/ready`
and `/api/status` respond over real HTTP; migrations apply; a row written to
PostgreSQL **survives `docker compose down` + `up`** (named volume); ports are
bound to loopback only and the app is unreachable from the host's LAN IP; the
container runs as uid 1000; no `.env` in the image; SIGTERM produces a clean
uvicorn shutdown; `MetaTrader5` is absent from the image and all `core/`+`api/`
modules import without it.

**Dependency bug found and fixed during verification:** `greenlet` is required
by SQLAlchemy's asyncio layer and was present in the local venv but not
declared in any requirements file — the image crash-looped with a confusing
`ImportError`. It is now pinned explicitly.

**Known limitations in this stack (not overselling it):**
- No authentication on the API; loopback binding is a stopgap only.
- `/api/status` returns hardcoded values; `/api/emergency-stop` still stops
  nothing; `/api/resume` is a stub.
- No trading loop runs inside the container yet — `core/` engines are not
  wired to a scheduler.
- PostgreSQL stores money as `Float`; decimal-safe accounting is not done.
- The Windows bridge does not exist. No broker connectivity has been verified.

## 3.1.1 MULTI-ASSET INSTRUMENT ENGINE
**Status: implemented and tested (135 tests).**

New `core/instruments/` provides a broker-neutral instrument model and
registry, and fixes a **financial-safety defect** found this cycle.

**Defect fixed — position sizing was inert.** `MockMT5Adapter.get_symbol_info`
returned gold-shaped constants (`contract_size=100000`, `tick_size=0.00001`,
`volume_min=0.01`) for *every* symbol. Because sizing floors to the broker
minimum, every instrument resolved to 0.01 lots regardless of risk budget or
stop distance. Verified before the fix: equity $10k at 1% risk returned
**0.01 lots for a $0.50 stop and for a $50 stop alike** — position sizing
could not express any risk difference. `calculate_lot_size` had **no direct
test coverage**.

After the fix (same $100 risk budget):

| Instrument | Stop distance | Size |
|---|---|---|
| XAUUSDm | $2.00 | 0.500 lots |
| EURUSD | $0.0020 | 0.500 lots |
| BTCUSD | $50.00 | 0.020 lots |
| US30.cash | $5.00 | 2.000 lots |

Changes:
- `core/instruments/model.py` — `InstrumentSpec`, `AssetClass`,
  `InstrumentType`, `TradingStatus`, `OrderTypeSupport`, `InstrumentRegistry`,
  canonical-symbol normalisation. Critical fields default to `None`, never to
  plausible constants.
- `core/instruments/adapter.py` — maps broker `SymbolInfo` to the neutral
  model without inventing values.
- `core/mt5/types.py` — `SymbolInfo` contract fields changed from
  `= 100000.0` style defaults to `Optional[...] = None`, so missing broker
  metadata is representable.
- `core/mt5/adapter.py` — mock now returns per-asset specs (gold 100oz/0.01,
  forex 100000/1e-5, crypto 1.0, index, energy).

Fail-safe behaviour verified: an instrument missing a critical field is
registered but refused for trading with an explicit reason rather than sized
on invented data.

**Test honesty note:** three pre-existing tests asserted the *buggy* values
(`digits == 5` for gold, `margin == 2000000`, breakeven `2000.00002`). These
encoded the defect, so they were corrected to the real values rather than
weakened. Three of my own new test premises were also wrong on first run and
were corrected against verified behaviour.

## 5. NEXT TASKS (prioritized)
1. **Build the Windows MT5 bridge** per `docs/MT5_WINDOWS_BRIDGE.md`, starting read-only (endpoints 1–11), defaulting to no order submission. **Blocked on VM console access.**
2. **Execution safety (P6):** idempotency keys, explicit execution states incl. *unknown*, and reconciliation on timeout instead of blind retry.
3. **Wire the real kill switch**: `api/main.py` `/api/emergency-stop` currently stops nothing; connect it to `core/risk/engine.py:218` and persist the state.
4. **Decimal-safe monetary columns** and a migration for existing data.
5. **Authentication** on the API before exposing it beyond loopback.
6. **Rotate the exposed broker password** at Exness, then remove the credential fallbacks. Requires user action — it is already in git history.
