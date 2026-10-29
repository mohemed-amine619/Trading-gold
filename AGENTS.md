# AGENTS.md — Trading Platform

Python trading platform (MT5 broker adapter + modular `core/` + FastAPI). Read this before editing anything.

**Target platform is native Windows 11** — no Docker, WSL, Linux host or VM.
Use `.venv\Scripts\python.exe` there, or `.venv/bin/python` on Linux.
PowerShell tooling lives in `scripts/windows/`; see `README.md` and
`docs/WINDOWS_ARCHITECTURE.md`.

## Repo state — read this first

**All 152 files are git-tracked.** `run.py` is now ~4,680 lines. `git status`
is clean.

Note `trading_bot.log` is a **log file**, not a database — the old
"second shadow SQLite file" described below has been resolved: `run.py` now
writes its `trades` table into the single configured database via
`_get_sqlite_db_path()`, which is `data/trading.db`. There is one SQLite file.

### Two systems, do not confuse them

| | Legacy monolith | Modular engine |
|---|---|---|
| Files | `run.py` (~4,680 lines) | `core/`, `api/` |
| Imports `core/`? | **Yes, for market location only** — `core/market/location.py`, imported defensively at line ~108 | n/a |
| Tests? | Integration-tested via a stubbed MT5 module | 733 tests, all passing |
| Actually run today | Yes — this is the production bot | Not wired into any entrypoint |

- **`run.py` is the live bot.** `gui.py:226` does `import run`; `fix_mt5.bat:17` runs `py run.py`.
- `run.py` now consumes the market-location layer (structure, S/R zones,
  liquidity, range position) for its entry veto, stop placement and profit
  target. Everything else — scoring, AI, risk sizing, management — is still
  self-contained in `run.py`.
- `core/trading/orchestrator.py` wires the *entire* `core/` stack and is tested
  end-to-end, but is **not** an entrypoint. Two consumers of the same location
  engine, not two implementations.

## Commands

Run everything with `.venv/bin/python` — the repo has two virtualenvs and only `.venv` has pytest.

```bash
.venv/bin/python -m pytest tests/ -q          # 733 passed
.venv/bin/python -m pytest tests/unit/test_risk_engine.py -q   # single file
.venv/bin/python -m pytest tests/unit -k approve -q            # single test by name
```

**Never run bare `pytest` or `python -m pytest` with no path.** Root `test_mt5.py` is not a test — it calls `mt5.initialize()` at module import (`test_mt5.py:5`). Pytest collects it and aborts the whole run. It currently fails safe only because `MetaTrader5` isn't installed on Linux; **if that dependency is ever installed, bare `pytest` opens the live terminal at collection time.**

There is no `pyproject.toml`, `pytest.ini`, `conftest.py`, linter, or formatter config. No CI. Don't invent lint commands — none exist.

## Environment

- **MT5 / `MetaTrader5` is Windows-only and cannot be installed on this Linux box** (`pip install MetaTrader5` → no matching distribution). It is in `requirements.txt:1` anyway.
- `run.py:94` does `import MetaTrader5` at module level → **`run.py` cannot even be imported on Linux.** Never add a test that imports it.
- `core/mt5/adapter.py` imports MT5 lazily inside `RealMT5Adapter._import_mt5()` and raises `RuntimeError` — this is the only pattern that works cross-platform. Keep it.
- Use `MockMT5Adapter` (deterministic, seeded) for all tests. See `tests/unit/test_mt5_adapter.py:5-8` for the fixture pattern.
- No `.env` exists in the repo; `.env.example` is the template. `core/settings.py` reads `.env` via pydantic-settings.

## Safety-critical findings

Treat these as blockers for any live-trading work, not trivia:

1. **The live broker password is still in git history** — commits `fda97e0`, `3113e04`
   and `8bbfe0e` contain account `261009880` with its password and server
   `Exness-MT5Trail16` as `os.getenv()` fallbacks. It is **no longer on disk**:
   `run.py` now reads `os.getenv("MT5_PASSWORD") or None` and `config.py:20`
   defaults to `None`, matching `core/settings.py:53` ("NEVER add secrets as
   defaults"). **The password still must be rotated at the broker** — deleting a
   line does not remove it from history. Do not reintroduce a literal fallback.
2. **`run.py` now defaults to PAPER mode.** `dry_run` is a real `CONFIG` key
   (`run.py:566`) driven by `_TRADING_MODE` (`run.py:206`). `TRADING_MODE`
   defaults to `paper`; real orders require `TRADING_MODE=live`, which prints a
   loud banner. Unrecognised values fall back to paper, never open. Locked by
   `tests/unit/test_safety_paper_default.py`. **Do not reintroduce a live
   default** — that test will fail.
3. **The dashboard reports the real mode.** It previously read
   `cfg.get("paper_trading")`, a key that exists nowhere, so the badge always
   showed LIVE. It now reads `cfg.get("dry_run")` (`run.py:3994`).
4. **Two emergency stops exist; only one gates real orders.** `run.py:4664`
   serves `POST /api/emergency-stop` (closes all positions via `_close_all` and
   pauses), and `/api/control {"action":"emergency_stop"}` does the same.
   `api/main.py:132` wires the modular `RiskEngine.trigger_emergency_stop()`
   (checked at `core/risk/engine.py:327`), but that engine is only constructed
   by `core/trading/orchestrator.py`, which is **not an entrypoint**. Do not
   assume the modular kill switch stops the live bot.
5. **No martingale, grid, or automatic stop-widening.** Preserve this. `core/position/trailing.py` enforces "never loosen SL" — `tests/test_trailing.py::test_never_loosen_sl` guards it.

## Multi-asset

Do not hardcode gold assumptions. This is an explicit project mandate.

- `SYMBOLS` (env, comma-separated) drives the list — `core/settings.py:88-96`. **Watch out: `.env.example:24` documents `TRADE_SYMBOLS`, but `settings.py:90` uses `validation_alias="SYMBOLS"` — `TRADE_SYMBOLS` is silently ignored.** Verified: setting `TRADE_SYMBOLS` yields `['XAUUSDm']`; setting `SYMBOLS` yields the real value.
- `run.py` uses a *different* key: `TRADE_SYMBOL` (singular) for its `CONFIG["symbol"]` default (`run.py:120`) and `TRADE_SYMBOLS` for the manager loop (`run.py:1615`).
- `MockMT5Adapter` defaults to `symbol="XAUUSDm"`, `base_price=2000.0` — **pass these explicitly** for non-gold tests or P&L assertions will be nonsense.
- The legacy `core/database/models.py` `Symbol` table is still unpopulated — the new `core/instruments/` registry is the source of truth going forward. No consumer is wired into `run.py` yet.
- Never claim an instrument is supported until a test demonstrates it.

## Dead code — cleaned up

The one-shot, already-applied, **non-idempotent** patch scripts that used to sit
at the repo root have been **removed**, together with the `run_pro.py` fork,
`get-pip.py` and the dangerous root `test_mt5.py`:

`modify_html.py`, `update_dashboard_settings.py`, `update_manager.py`,
`update_status.py`, `patch_dashboard_controls.py`,
`patch_dashboard_controls_2.py`, `patch_position_manager.py`, `manual_protect.py`,
`run_pro.py`, `get-pip.py`, `test_mt5.py`

They rewrote source files in place and were actively dangerous:
`update_manager.py:93-94` sliced `content[:idx]`, truncating
`core/position/manager.py` and deleting ~154 lines including
`get_total_floating_pnl`. `test_mt5.py` called `mt5.initialize()` at import, so
on Windows (where MetaTrader5 *is* installed) collecting it would open the live
terminal.

Also removed: the empty `api/routers/` package (imported by nothing), the empty
`tests/e2e/` directory, and a stale byte-identical duplicate `trading.db` at the
repo root. **The live database is `data/trading.db`** — that is what
`_get_sqlite_db_path()` resolves and what `core/settings.py` defaults to. Only
one SQLite file should exist; do not recreate a second at the root.

`core/notifications/__init__.py` and `core/position/manager_orig.py` are empty
but still present. Do not confuse an empty **package marker** with dead code —
deleting any `__init__.py` breaks the package it anchors.

**If any of the removed patchers reappear, do not run them.** Make source edits
directly with the edit tool, never by running a script that rewrites files in
place.

## Config duplication trap

`run.py`'s `CONFIG` (starting around `:360`) and `core/settings.py Settings`
define the same knobs with **different defaults**. An edit to
`core/settings.py` does not affect the running bot, and vice versa. Within
`run.py`, many defaults are now chosen by `_profile_cfg(...)` from
`TRADING_PROFILE` (`balanced` default, `aggressive`, `scalper`,
`money_printer`); the table below shows the **balanced** value.

| Key | `run.py` (balanced) | `core/settings.py` |
|---|---|---|
| `min_signal_score` | 50.0 | 55.0 |
| `max_open_pos` | 3 | 5 |
| `max_daily_dd` | 0.03 | 0.02 |
| `max_margin_usage` | 0.30 | 0.50 |
| `sleep_interval` | 1.0 | 5.0 |

`core/config/schema.py` is the **write contract** for the dashboard: 93 declared
parameters, each with bounds, and `POST /api/config` refuses anything it does
not declare. That closes the "saved but ignored" hole for declared keys, but it
does not unify the two default sets — the dashboard writes `run.py CONFIG`
only. `core/settings.py` still has its own values, and `as_legacy_config()`
still never maps `max_spread_points`.

Two of those keys exist because of live failure modes, so do not "simplify"
them away:

- `max_pos_per_symbol` (default 1) — `max_open_pos` alone allowed the same
  signal to re-fire and open an identical position on the same symbol (two
  identical XAUUSDm shorts were observed). The guard is
  `_count_symbol_positions()` / `_entry_slots_ok()` in `run.py`.
- `min_sl_step_atr` (default 0.05) — a stop modify is only sent when it improves
  the stop by at least this ATR multiple, via the pure `_sl_modify_allowed()`.
  Without it the rolling 5-bar anchor caused one broker round-trip per bar for
  ~0.01 price units. `0` restores the old every-move behaviour.

Separately, `AIEnsemble.predict()` refuses to use a model whose recorded
out-of-sample edge is unmeasured or below `ai_min_oos_edge`: it returns neutral
`(0, 0.5)`. The promotion gate only guards *replacement*, so without this a
model promoted once kept voting forever. `AI_MIN_OOS_EDGE=-1.0` is the escape
hatch.

**When adding a config key, add it to `core/config/schema.py` and
`run.py CONFIG` (and `core/settings.py` if the modular engine should see it),
or the key will be silently ignored.**

## Docker & deployment

Ubuntu core (verified working on this host). The Linux core **never** imports MetaTrader5.

```bash
cp .env.example .env          # then set POSTGRES_PASSWORD (openssl rand -base64 32)
docker compose up -d --build  # migrations run automatically on start
docker compose ps             # both services must be (healthy)
docker compose logs -f app
docker compose down           # STOP, keeps data
docker compose down -v        # DESTROYS the database volume
```

- **Never `docker compose down -v`** unless you intend to delete all trading history.
- `docker compose exec app alembic current` / `upgrade head` / `revision --autogenerate -m "..."`.
- Ports bind **127.0.0.1 only** (8000 app, 5432 db). The app is unreachable from the host LAN IP — do not "fix" this by widening the binding; it is deliberate.
- `/api/health` = liveness, `/api/ready` = readiness (503 when Postgres is unreachable). Startup no longer crashes on DB failure, so orchestration can tell "waiting for DB" from "broken process".
- Container runs as non-root `appuser`; `.env` is excluded from the image; DB URLs are redacted before logging.
- Docs: `docs/DEPLOYMENT_UBUNTU.md`, `docs/MT5_WINDOWS_BRIDGE.md` (design only — bridge not implemented).
- The API has **no authentication**. Loopback binding is a stopgap, not security.

## Dependency split

- `requirements-core.txt` — Linux/Docker. Full `core/` stack + `asyncpg`/`psycopg`. No MetaTrader5.
- `requirements-mt5.txt` — **Windows only**. The only file that may contain `MetaTrader5`.
- `requirements.txt` is the old monolith list and is now misleading (it lacks the whole `core/` stack). Prefer the split files.
- `greenlet` is pinned explicitly: SQLAlchemy's asyncio layer needs it at import time and fails with a confusing `ImportError` if absent.

## Windows MT5 bridge (Side 2 — not yet implemented)

- Design contract: `docs/MT5_WINDOWS_BRIDGE.md`. **No bridge code exists.**
- Config already in `core/settings.py`: `MT5_BRIDGE_URL`, `MT5_BRIDGE_TOKEN`, `MT5_BRIDGE_TIMEOUT_SECONDS`, `MT5_BRIDGE_TLS_VERIFY`, plus `bridge_configured` / `mt5_bridge_http_url` / `mt5_bridge_auth_headers`.
- VM is at **192.168.122.212** on libvirt **NAT** (`virbr0`), *not* a bridged LAN network. Re-discover with `virsh domifaddr win11` — the address is DHCP and may change. Do not hardcode it.
- Docker containers use a different network namespace than the host; verify container→VM reachability with `docker compose exec app curl ...` rather than assuming a host ping implies it.

## Instrument registry (P5 — the multi-asset foundation)

**Rule 10 of the mandate: never substitute a default for missing broker metadata.** This is enforced in code, not by convention.

- `core/instruments/model.py` — `InstrumentSpec`, `InstrumentRegistry`, `AssetClass`, `TradingStatus`. Critical fields default to **`None`**, never to plausible constants.
- `core/instruments/adapter.py` — maps broker `SymbolInfo` → `InstrumentSpec`. Imports no broker SDK.
- `InstrumentRegistry.require()` raises `ResolutionError` for: unknown symbol, missing critical fields, stale metadata, non-tradable status.
- `InstrumentRegistry.size_position()` refuses rather than exceeding the risk budget. `None` from a calc means "cannot size" — **never treat it as 0**.
- `core/mt5/types.py SymbolInfo` contract fields are `Optional[...] = None`. A broker value of `0` is genuine data, distinct from absence — do not collapse them.
- **Never reintroduce gold-shaped defaults.** `MockMT5Adapter` returns per-asset specs (gold 100oz/0.01, forex 100000/1e-5, crypto 1.0, index, energy). Returning one global spec set made position sizing inert — every size floored to `volume_min`. `tests/unit/test_instruments.py::test_size_position_is_proportional_to_risk_budget` guards this.
- `AssetClass.UNKNOWN` is a valid, honest answer. Never guess an asset class.
- No instrument registry consumer exists in `run.py` yet; it is not wired into any entrypoint.

## Architecture reference

```
core/
  settings.py          Pydantic v2 Settings (.env) — single source of truth going forward
  database/            SQLAlchemy 2.x models.py (18 tables) + engine.py (sync + async)
  mt5/                 adapter.py (MT5Adapter ABC, RealMT5Adapter, MockMT5Adapter)
                       types.py (dataclasses), connection.py (reconnect/backoff)
  market/              engine.py, pipeline.py, intelligence.py (regime classifier)
                       swings.py (fractal pivots), structure.py (HH/HL, BOS/CHoCH)
                       levels.py (scored S/R zones), liquidity.py (stop pools, sweeps)
                       location.py (MarketLocation, entry_gate, structural_stop)
                       series.py (shared resolve_atr)
  instruments/         model.py (InstrumentSpec/Registry), adapter.py (broker mapping)
  strategy/            base.py, ensemble.py, strategies.py
  ai/                  decision.py (LLM), provider.py
  risk/engine.py       approve() gate, lot sizing, emergency stop, loss-streak
  execution/engine.py  send_order w/ requote retry, modify_position, close_position
  position/            manager.py (reconcile/trailing/breakeven), trailing.py (pure funcs),
                       decision.py (PNLDecisionEngine), manager_orig.py (empty, delete)
  trading/             orchestrator.py — candles → location → signal → gate → stop →
                       target → risk → execution. THE MISSING WIRE (see below)
  backtesting/engine.py  notifications/ (EMPTY — 0 bytes)  events/log.py
api/
  main.py              FastAPI app factory, lifespan inits async DB
  config_router.py     /api/config CRUD + ConfigAudit — the only real route module
```

Data flow (per `docs/ARCHITECTURE.md:275`): MarketEngine → StrategyEnsemble → RiskEngine.approve → ExecutionEngine → MT5Adapter → PositionManager → events/DB.

### Market location — the "where is the market" layer (NEW)

`core/market/location.py` is now the entry point for spatial reasoning.
`analyse_location(df)` returns a `MarketLocation` carrying range position
(premium/discount), Fibonacci, scored S/R zones, liquidity pools, and market
structure. Two methods decide things:

- **`entry_gate(direction)`** — a *veto*, not a scorer. Location can only remove
  trades, never add them, which is why the existing EMA/RSI/MACD strategies
  survive unchanged. **The rule is regime-aware**: strict premium/discount in a
  ranging market; permitted *with* the trend; counter-trend only at the
  counter-extreme. Do **not** make it strict everywhere — a confirmed uptrend
  sits at the top of its own swing range by construction, so a strict rule
  blocks nearly every trend trade and inverts the meaning of a new high.
- **`structural_stop(entry, direction)`** — places the stop beyond the swing
  that invalidates the idea, which is what `Signal.invalidation` could only
  describe in prose. Returns `None` when no defensible stop exists; callers
  must refuse rather than substitute.

Hard rules, each guarded by a test:
- No confirmed range ⇒ `UNDEFINED` ⇒ **block**. Never treat as neutral.
- Price outside the range is a breakdown/breakout, **not** an extreme; position
  is clamped to `[0,1]`.
- Every tolerance is an **ATR multiple**, never a fixed point count — the
  multi-asset mandate applies to all of this code.
- Level/stop prices snap to the broker grid only when `trade_tick_size`/`point`
  was supplied; otherwise they stay unrounded.

Notable: `core/position/trailing.py` holds **pure functions** with no broker dependency — keep it pure, it's the easiest thing to test and is the safety-critical "never widen a stop" rule.

### Wiring status — still not an entrypoint

`core/trading/orchestrator.py` wires the whole stack and is tested end-to-end
(candles → risk approval → order sent), but it is still not an entrypoint.
`run.py` imports only two things from `core/`: the market-location layer
(`core/market/location.py`) and the config plane (`core/config/`). The
orchestrator itself is never constructed by the live bot, so its risk/execution
graph still does not affect a real order. That gap is deliberate and documented,
not overlooked — see `docs/PROJECT_STATUS.md` §0.0.

## Dashboard

`dashboard.html` (128 KB, **git-tracked**) is served by `run.py` from a **nested FastAPI app defined inside `if __name__ == "__main__"`** — it is not importable, so you cannot test it directly. There is no JS test suite; after editing it, check that `<div>`/`<script>` tags balance and run `node --check` on the inline script block.

- The Settings tab renders a **"Master Parameters"** panel from `GET /api/config/schema` (`core/config/schema.py`), so new parameters appear without editing the HTML. Apply All posts the whole batch to `POST /api/config`; a rejected batch applies nothing. It also shows the `config_audit` trail and a Reset DD Halt button.
- Writes send `X-Dashboard-Token` when `DASHBOARD_TOKEN` is set; the JS prompts once and stores it in `localStorage.dashboardToken`.

- It expects `run.py`'s response shapes: `{status, markets, mode}`, `{positions}`, `{data}`, `{history}` (`dashboard.html:1050-1076`). The `markets` entries now also carry additive `locBias`, `locPosition`, `locRangeLow`, `locRangeHigh`, `locStructure`, `locSupport`, `locResistance` keys; the UI ignores unknown keys, so this is backward-compatible.
- `run.py` now serves the schema-driven config plane: `GET /api/config/schema`, `GET /api/config`, `POST /api/config`, `GET /api/config/audit`, `POST /api/config/reset-halt`, plus the legacy `/api/config/{section}`. `POST /api/config` validates the whole batch against `core/config/schema.py` and rejects it atomically; `dry_run` requires `confirm: true`.
- Set `DASHBOARD_TOKEN` in `.env` to require `X-Dashboard-Token` on every mutating `/api/*` call (reads stay open). Unset ⇒ unchanged, and loopback binding is the only control.
- `/api/emergency-stop` is now served by `run.py:4664` (closes all positions, pauses trading); it used to 404. `api/main.py:132` serves the same path on the *other* FastAPI app, wired to the modular `RiskEngine` — which does not gate `run.py`'s orders.
- `api/main.py` is a **second, separate FastAPI app** (`version="2.0"`) that also serves `dashboard.html`'s path space. Two apps, one port, no auth on either. `dashboard.html` effectively targets `run.py`.
- CORS is `allow_origins=["*"]` with `allow_credentials=True` in both apps (`api/main.py:25-30`) — insecure; note it if touching auth.
- `core/notifications/__init__.py` is empty; Telegram is dead in the monolith because `CONFIG` has no `telegram_token` key (`run.py:112-166`), so `Notifier` reads `""` and silently no-ops.

## Database

- `run.py` writes trade history to the same configured database via raw
  `sqlite3` (`CREATE TABLE IF NOT EXISTS trades`), bypassing SQLAlchemy and
  `core/database/models.py`. The table is created lazily on the first closed
  trade, so `/api/history` returns `{"history": []}` from a bare `except` until
  then. The table therefore exists outside the Alembic-managed schema — a
  known wart, but only one database file.
- The live DB is `data/trading.db`. `*.db`, `*.db-shm` and `*.db-wal` are all in
  `.gitignore`, so `git add .` will not commit trade data. Treat the file and
  its WAL sidecars as sensitive regardless.
- The same file holds `config_audit` (`core/config/audit.py`), one row per
  changed parameter. Audit writes are fail-safe: a failure is logged and
  swallowed so it can never abort an already-applied settings change.
- Persisted risk state lives **outside** SQLite in
  `data/state/risk_<symbol>.json` (`core/risk/state.py`), written atomically.
  Trained models go to `data/models/`. Both directories, plus `data/cache/`,
  are gitignored because they are machine-specific runtime state.
- Sync session for the trading loop (`get_sync_db`), async for FastAPI (`get_async_db`). `get_db_dependency` commits and then the handler may commit again (`api/config_router.py:66`).

## Conventions

- Python 3.12. `core/`/`api/` must stay Linux-testable and MT5-free (verified: the Docker image has no MetaTrader5 and every core/api module imports without it).
- Logging is stdlib `logging` with `logger = logging.getLogger(__name__)` per module — **not** the `structlog` the docs claim.
- Trading-mode enum is `TradingMode.PAPER | LIVE | BACKTEST` (`core/settings.py:18-21`), but `Settings.validate_live_requirements` (`core/settings.py:252`) only enforces extra checks for LIVE. Paper is the default and there is no separate paper simulator in `core/` — `MockMT5Adapter` is the closest thing.
- `ManualPolicy` (observe_only / protect_only / full_management) governs manual positions in `core/position/manager.py:25-28`; default is `OBSERVE_ONLY`. Preserve this default.
- Magic number isolates the bot's positions from manual ones (`888999` default). Never break this.
- Docstrings are English; some GUI strings are French. Match the file you're editing.

## Working agreement

- One coherent, testable phase at a time; run `.venv/bin/python -m pytest tests/ -q` after each change.
- Never report a test as passing unless you actually ran it. Never let tests touch a live account.
- Update `docs/PROJECT_STATUS.md` when behavior changes. The real count is now 733; keep that number honest.
- `docs/ROADMAP.md` and `docs/ARCHITECTURE.md` are **aspirational** — they describe the intended 12-phase platform, not current state. `docs/ARCHITECTURE.md:9` is already stale (claims `run.py` is 1,535 lines; it is 1840). Don't treat any of it as implemented without checking the code.