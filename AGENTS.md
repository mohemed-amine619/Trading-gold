# AGENTS.md — Trading Platform

Python trading platform (MT5 broker adapter + modular `core/` + FastAPI). Read this before editing anything.

## Repo state — read this first

**This repo is mid-migration and only 9 files are git-tracked.** `git ls-files` returns only:
`.gitignore`, `config.py`, `gui.py`, `position_dashboard.py`, `requirements.txt`, `run.py`, `build.bat`, `AI_Trading_Bot.spec`, `PositionDashboard.spec`.

Everything else — `core/`, `api/`, `tests/`, `alembic/`, `docs/`, `dashboard.html`, `.env.example`, `run_pro.py` — is **untracked**. Do not `git clean -fdx` or assume a fresh clone reproduces the app. `git status` shows ~35 entries; that is expected.

### Two parallel systems, do not confuse them

| | Legacy monolith | Modular engine |
|---|---|---|
| Files | `run.py` (1840 lines), `run_pro.py` (1608, untracked) | `core/`, `api/` |
| Imports `core/`? | **No** — zero `from core` imports | n/a |
| Tests? | **None** — no test references either file | 63 tests, all passing |
| Actually run today | Yes — this is the production bot | Not wired into any entrypoint |

- **`run.py` is the live bot.** `gui.py:226` does `import run`; `fix_mt5.bat:17` runs `py run.py`.
- **`run_pro.py` is a stale fork** — lines 1–782 are byte-identical to `run.py`, and `run.py` is a strict superset (`step()` extracted, `_modify_sltp`, multi-symbol manager, AI-confidence sizing). Never edit `run_pro.py`; port changes into `run.py`. Its docstring still says "Run with: `python run.py`".
- `core/` + `api/` is the intended target architecture (`docs/ARCHITECTURE.md`), well tested but **not reachable from any entrypoint**. `core/settings.py:264 as_legacy_config()` exists to bridge them and is currently called by nothing but a test.

## Commands

Run everything with `.venv/bin/python` — the repo has two virtualenvs and only `.venv` has pytest.

```bash
.venv/bin/python -m pytest tests/ -q          # 199 passed
.venv/bin/python -m pytest tests/unit/test_risk_engine.py -q   # single file
.venv/bin/python -m pytest tests/unit -k approve -q            # single test by name
```

**Never run bare `pytest` or `python -m pytest` with no path.** Root `test_mt5.py` is not a test — it calls `mt5.initialize()` at module import (`test_mt5.py:5`). Pytest collects it and aborts the whole run. It currently fails safe only because `MetaTrader5` isn't installed on Linux; **if that dependency is ever installed, bare `pytest` opens the live terminal at collection time.**

There is no `pyproject.toml`, `pytest.ini`, `conftest.py`, linter, or formatter config. No CI. Don't invent lint commands — none exist.

## Environment

- **MT5 / `MetaTrader5` is Windows-only and cannot be installed on this Linux box** (`pip install MetaTrader5` → no matching distribution). It is in `requirements.txt:1` anyway.
- `run.py:89` and `run_pro.py:89` do `import MetaTrader5` at module level → **`run.py` cannot even be imported on Linux.** Never add a test that imports it.
- `core/mt5/adapter.py` imports MT5 lazily inside `RealMT5Adapter._import_mt5()` and raises `RuntimeError` — this is the only pattern that works cross-platform. Keep it.
- Use `MockMT5Adapter` (deterministic, seeded) for all tests. See `tests/unit/test_mt5_adapter.py:5-8` for the fixture pattern.
- No `.env` exists in the repo; `.env.example` is the template. `core/settings.py` reads `.env` via pydantic-settings.

## Safety-critical findings

Treat these as blockers for any live-trading work, not trivia:

1. **Hardcoded live broker credentials are in git history and on disk** — `run.py:114-116` and `run_pro.py:114-116` have real account `261009880`, password, and server `Exness-MT5Trail16` as `os.getenv()` fallbacks. The password must be rotated at the broker regardless of any code change; removing it from the file does not remove it from history. This inverts both policies already stated in the repo: `core/settings.py:53` ("NEVER add secrets as defaults") and `config.py:19-23` (credentials default to `None`).
2. **`run.py` now defaults to PAPER mode.** This was fixed: `dry_run` is a real `CONFIG` key (`run.py:189`) driven by `_TRADING_MODE` (`run.py:111`). `TRADING_MODE` defaults to `paper`; real orders require `TRADING_MODE=live`, which prints a loud banner. Unrecognised values fall back to paper, never open. Locked by `tests/unit/test_safety_paper_default.py`. **Do not reintroduce a live default** — that test will fail.
3. **The dashboard now reports the real mode.** It previously read `cfg.get("paper_trading")`, a key that exists nowhere, so the badge always showed LIVE. It now reads `cfg.get("dry_run")` (`run.py:1705`).
4. **Hardcoded live broker credentials are still present** — `run.py:130-132` and `run_pro.py:114-116` have a real account number, password, and server as `os.getenv()` fallbacks. **This is unresolved and needs your action:** the password must be rotated at the broker regardless of any code change, and removing it from the file does not remove it from git history. It inverts both policies already stated in the repo: `core/settings.py:53` ("NEVER add secrets as defaults") and `config.py:19-23` (credentials default to `None`). Not changed in this phase because credential rotation requires confirmation.
5. **`/api/emergency-stop` in `api/main.py:54` is a stub** — it returns a success message and stops nothing. `core/risk/engine.py:218` has a real kill switch (`trigger_emergency_stop` / `reset_emergency_stop`, checked at `engine.py:276`) but no route or caller wires it up. `run.py` `/api/control` has a working `emergency_stop` action.
5. **No martingale, grid, or automatic stop-widening.** Preserve this. `core/position/trailing.py` enforces "never loosen SL" — `tests/test_trailing.py::test_never_loosen_sl` guards it.

## Multi-asset

Do not hardcode gold assumptions. This is an explicit project mandate.

- `SYMBOLS` (env, comma-separated) drives the list — `core/settings.py:88-96`. **Watch out: `.env.example:24` documents `TRADE_SYMBOLS`, but `settings.py:90` uses `validation_alias="SYMBOLS"` — `TRADE_SYMBOLS` is silently ignored.** Verified: setting `TRADE_SYMBOLS` yields `['XAUUSDm']`; setting `SYMBOLS` yields the real value.
- `run.py` uses a *different* key: `TRADE_SYMBOL` (singular) for its `CONFIG["symbol"]` default (`run.py:120`) and `TRADE_SYMBOLS` for the manager loop (`run.py:1615`).
- `MockMT5Adapter` defaults to `symbol="XAUUSDm"`, `base_price=2000.0` — **pass these explicitly** for non-gold tests or P&L assertions will be nonsense.
- The legacy `core/database/models.py` `Symbol` table is still unpopulated — the new `core/instruments/` registry is the source of truth going forward. No consumer is wired into `run.py` yet.
- Never claim an instrument is supported until a test demonstrates it.

## Dead code — do not run these

These are one-shot, already-applied, **non-idempotent** scripts that rewrite source files in place. Re-running corrupts the tree:

`modify_html.py`, `update_dashboard_settings.py`, `update_manager.py`, `update_status.py`, `patch_dashboard_controls.py`, `patch_dashboard_controls_2.py`, `patch_position_manager.py`, `manual_protect.py`

- **`update_manager.py:93-94` is the worst**: it slices `content[:idx]`, truncating `core/position/manager.py` and deleting ~154 lines including `get_total_floating_pnl`.
- `patch_position_manager.py`'s regex target sits inside its own replacement, so re-application duplicates lines — visible damage already exists at `core/position/manager.py:55-59` (`trail_activate_rr` / `trail_distance_rr` assigned twice). Harmless (idempotent values) but a marker for this failure mode.
- Make source edits directly with the edit tool, never by running a patcher.

Also dead: root `test_position_manager.py` (empty `unittest.TestCase`, collects 0 items — the real one is `tests/test_position_manager.py`), `dashboard/` (empty), `api/routers/` (empty `__init__.py`), `get-pip.py`. `create_tables.py` is superseded by Alembic but is harmless (`create_all`, idempotent).

## Config duplication trap

`run.py:112-166 CONFIG` and `core/settings.py Settings` define the same knobs with **different defaults**. An edit to `core/settings.py` does not affect the running bot, and vice versa.

| Key | `run.py` | `core/settings.py` |
|---|---|---|
| `min_signal_score` | 60 (`:132`) | 55.0 (`:164`) |
| `max_open_pos` | 3 (`:153`) | 5 (`:156`) |
| `max_daily_dd` | 0.03 (`:154`) | 0.02 (`:154`) |
| `max_margin_usage` | 0.30 (`:152`) | 0.50 (`:158`) |
| `sleep_interval` | 1.0 (`:163`) | 5.0 (`:175`) |

Also: `core/settings.py:176 max_spread_points` exists but `as_legacy_config()` never maps it. `run.py` reads 29 keys via `cfg.get()` that aren't in `CONFIG` (`telegram_token`, `dry_run`, `paper_trading`, `structure_trailing`, `max_spread_points`, …), so each silently takes its call-site default. **When adding a config key, add it to both sides or it will be silently ignored.**

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
  instruments/         model.py (InstrumentSpec/Registry), adapter.py (broker mapping)
  strategy/            base.py, ensemble.py, strategies.py
  ai/                  decision.py (LLM), provider.py
  risk/engine.py       approve() gate, lot sizing, emergency stop, loss-streak
  execution/engine.py  send_order w/ requote retry, modify_position, close_position
  position/            manager.py (reconcile/trailing/breakeven), trailing.py (pure funcs),
                       decision.py (PNLDecisionEngine), manager_orig.py (empty, delete)
  backtesting/engine.py  notifications/ (EMPTY — 0 bytes)  events/log.py
api/
  main.py              FastAPI app factory, lifespan inits async DB
  config_router.py     /api/config CRUD + ConfigAudit — the only real route module
```

Data flow (per `docs/ARCHITECTURE.md:275`): MarketEngine → StrategyEnsemble → RiskEngine.approve → ExecutionEngine → MT5Adapter → PositionManager → events/DB.

Notable: `core/position/trailing.py` holds **pure functions** with no broker dependency — keep it pure, it's the easiest thing to test and is the safety-critical "never widen a stop" rule.

## Dashboard

`dashboard.html` (94 KB, untracked) is served by `run.py:1633` from a **nested FastAPI app defined inside `if __name__ == "__main__"`** — it is not importable, so you cannot test it directly.

- It expects `run.py`'s response shapes: `{status, markets, mode}`, `{positions}`, `{data}`, `{history}` (`dashboard.html:1050-1076`). `run_pro.py` serves different shapes → **`dashboard.html` is incompatible with `run_pro.py`.**
- Routes `dashboard.html` calls that **nothing serves**: `/api/config/{section}` (only `api/config_router.py` has it, and that app is separate), `/api/emergency-stop` (it calls this but `run.py` exposes `/api/control`). Expect silent 404s.
- `api/main.py` is a **second, separate FastAPI app** (`version="2.0"`) that also serves `dashboard.html`'s path space. Two apps, one port, no auth on either. `dashboard.html` effectively targets `run.py`.
- CORS is `allow_origins=["*"]` with `allow_credentials=True` in both apps (`api/main.py:25-30`) — insecure; note it if touching auth.
- `core/notifications/__init__.py` is empty; Telegram is dead in the monolith because `CONFIG` has no `telegram_token` key (`run.py:112-166`), so `Notifier` reads `""` and silently no-ops.

## Database

- `trading.db` (SQLite, WAL) is the `core/` DB per `core/settings.py:102`. **`trading.db` is NOT in `.gitignore`** — `.env`, `venv/`, `.venv/`, `__pycache__`, `logs/` are. A `git add .` will commit live account/trade data. Add it before committing.
- Alembic is wired (`alembic.ini`, `alembic/env.py` reads `get_settings().database_url`). The first real migration now exists: `alembic/versions/aedbfe3f600c_initial_schema.py` (19 tables). Note `alembic/script.py.mako` was missing entirely and has been added — autogenerate failed before.
- `run.py` writes trade history to a **second, shadow SQLite file `trading_bot.db`** via raw `sqlite3` (`run.py:875` — `CREATE TABLE IF NOT EXISTS trades`), completely bypassing SQLAlchemy and `core/database/models.py`. It's created lazily on the first closed trade, so `/api/history` (`run.py:1790`) returns `{"history": []}` from a bare `except` until then. Different filename from `trading.db`, and also not gitignored. Decide on one DB and delete the other — do not leave two divergent schemas.
- On disk right now: `trading.db`, `trading.db-shm`, `trading.db-wal` are untracked and unignored. Treat all three as sensitive.
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
- Update `docs/PROJECT_STATUS.md` when behavior changes. It currently claims 63/63 passing; the real count is now 199. Keep that number honest.
- `docs/ROADMAP.md` and `docs/ARCHITECTURE.md` are **aspirational** — they describe the intended 12-phase platform, not current state. `docs/ARCHITECTURE.md:9` is already stale (claims `run.py` is 1,535 lines; it is 1840). Don't treat any of it as implemented without checking the code.