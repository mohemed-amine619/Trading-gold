# PROJECT STATUS

## 1. IMPLEMENTED
- **Position Monitoring & Reconciliation**
  - Developed full state reconciliation engine for `PositionManager`.
  - Enforces three operational policies for manual positions.
- **SL/TP and Trailing-stop Management**
  - Extracted pure trailing calculations into `core/position/trailing.py`.
  - Implemented Fixed-distance, ATR-based, Break-even, and Profit-locking trailing.
- **Fully Configurable Trading Dashboard**
  - Introduced `SystemConfig` and `ConfigAudit` SQLAlchemy models.
  - Rebuilt `dashboard.html` with configuration tabs and Live Controls.
- **Market Intelligence & Real-Time Decision Engine (New!)**
  - Implemented `MarketIntelligenceEngine` (`core/market/intelligence.py`) to classify market regimes (Uptrend, Downtrend, Correction, Ranging).
  - Implemented `PNLDecisionEngine` (`core/position/decision.py`) to output management recommendations (HOLD, TRAIL_STOP, MOVE_TO_BREAKEVEN).
  - Integrated `MarketIntelligenceEngine` and `PNLDecisionEngine` into `PositionManager` to compute real-time trailing stops and regime evaluations.
  - Fixed `test_position.py` tests by correctly asserting normalized MT5 tick values (floating point precision).
  - Modified `RiskEngine` to natively ingest floating real-time PNL instead of relying exclusively on basic account balance.

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
