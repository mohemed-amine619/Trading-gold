# Windows Architecture

How the platform is put together on Windows 11, and why.

---

## 1. Why Windows-native

MetaTrader5 is a **Windows-only package** — there is no Linux wheel, so
`pip install MetaTrader5` fails outright on Linux and cannot run inside a
Linux container. The MT5 terminal is likewise a Windows application.

Rather than split the platform across two machines with an HTTP bridge, this
deployment keeps everything on one Windows host. The bridge code
(`bridge/`) remains available for anyone who prefers a split setup, but it is
not required.

---

## 2. Components

```
┌─ Windows 11 ───────────────────────────────────────────┐
│                                                       │
│  MetaTrader 5 terminal                                 │
│        ▲                                               │
│        │ MetaTrader5 Python package (in-process)        │
│  core/mt5/  adapter.py   MT5Adapter ABC                │
│        │      ├── RealMT5Adapter  (lazy import)         │
│        │      └── MockMT5Adapter  (tests, no terminal)  │
│        ▼                                               │
│  core/                                                 │
│    instruments/  broker-neutral specs + registry        │
│    strategy/     signal generation                     │
│    risk/         approve(), sizing, kill switch        │
│    execution/    idempotent submit + execution states  │
│    position/     trailing, breakeven, reconcile        │
│    database/     SQLAlchemy models + sessions          │
│    logging_setup UTC logs + credential redaction       │
│                                                       │
│  api/main.py  FastAPI  (:8000, loopback)               │
│  alembic/     migrations                               │
│  scripts/windows/  PowerShell tooling                  │
│                                                       │
│  PostgreSQL 16  (or SQLite for single-user setups)     │
└───────────────────────────────────────────────────────┘
```

---

## 3. Layering rules

| Layer | May import | Must not import |
|---|---|---|
| `core/strategy`, `core/risk`, `core/instruments` | stdlib, pandas, numpy | `MetaTrader5`, broker SDKs |
| `core/mt5/adapter.py` | `core.mt5.types` | — |
| `api/` | `core/*` | `MetaTrader5` |
| `bridge/` (optional) | `MetaTrader5` | domain logic |

**Verified:** `core/` and `api/` import successfully with `MetaTrader5` blocked
entirely. Only `RealMT5Adapter._import_mt5()` imports it, lazily, and raises a
clear `RuntimeError` off-Windows.

---

## 4. Data flow

```
MT5 terminal
    │  MetaTrader5 package
    ▼
RealMT5Adapter  ──►  InstrumentRegistry  ──►  InstrumentSpec
    │                     (never defaults unknown values)
    ▼
MarketEngine ──► StrategyEnsemble ──► RiskEngine.approve()
                                            │
                        ┌───────────────────┘
                        ▼
                 ExecutionEngine.submit(idempotency_key)
                        │
                        ▼
              ExecutionReport(state = filled | rejected | UNKNOWN)
                        │
                        ▼
                 PositionManager (trailing, breakeven)
                        │
                        ▼
            SQLite/PostgreSQL  +  redacted logs  +  events
```

---

## 5. Safety architecture

Three independent gates. A strategy cannot skip any of them.

| Gate | Module | Enforces |
|---|---|---|
| Instrument | `core/instruments` | Refuses to size an instrument with missing/stale metadata |
| Risk | `core/risk/engine.py` | Per-trade risk, daily loss, drawdown, exposure, kill switch |
| Execution | `core/execution/` | Idempotency, no retry on unknown outcome |

### Paper by default

`TRADING_MODE` defaults to `paper`. An unrecognised value falls back to
paper — the system fails *closed*, never open. Going live requires explicit
configuration plus credentials plus a changed `DASHBOARD_SECRET_KEY`.

### Unknown outcomes

A timeout is not a failure. `ExecutionState.UNKNOWN` means the broker may have
acted; the correct response is reconciliation, not a retry. Tested in
`tests/unit/test_execution_safety.py`.

### No invented instrument data

`InstrumentSpec` critical fields default to `None`. `require()` raises
`ResolutionError` rather than substituting a plausible constant — a guessed
tick value turns a risk limit into a guess.

---

## 6. Process model

| Process | Started by | Tracked in |
|---|---|---|
| API (uvicorn) | `start.ps1` | `run/api.pid` |
| Bot (`run.py`) | `start.ps1 -Bot` | `run/bot.pid` |
| Bridge (optional) | `start.ps1 -Bridge` | `run/bridge.pid` |
| PostgreSQL | Windows service | Services.msc |

`start.ps1` verifies `/api/ready` **and** that the PID it launched is still
alive — a health response from an unrelated listener must not be mistaken for
its own success.

Logs: `logs/api.log`, `logs/bot.log`, `logs/bridge.log` (rotating, 10 MB × 5).

---

## 7. Cross-platform decisions

| Concern | Decision | Reason |
|---|---|---|
| SQLite journal | WAL on POSIX, DELETE on Windows | WAL sidecars lock harder on Windows |
| `busy_timeout` | 5000 ms | Avoid immediate `database is locked` |
| Log timestamps | UTC always (`UTCFormatter`) | Engine must not depend on host timezone |
| Secrets in logs | Redacted by filter | Passwords never reach disk |
| Paths | `pathlib` | Identical behaviour on both platforms |
| `run/` dir for PIDs | Relative to project | No registry or Program Files writes |

---

## 8. Optional: split deployment

The `bridge/` package exposes the same `MT5Adapter` contract over HTTP for a
split setup. It is **read-only by default** and refuses to start without a
token. See [MT5_WINDOWS_BRIDGE.md](./MT5_WINDOWS_BRIDGE.md).

Docker files (`Dockerfile`, `compose.yaml`) remain for optional Linux
deployment. They are not part of the Windows workflow.