# IMPLEMENTATION ROADMAP — TRADING-GOLD

> **Autonomous Engineering Phases**  
> **Platform Target:** Native Windows 11 x64  
> **Database:** SQLite Only (`data/trading.db`)  
> **Broker Integration:** MetaTrader 5 Python SDK  
> **Last Updated:** 2026-10-12

---

## OVERVIEW OF PHASES

| Phase | Title | Focus Area | Status |
|---|---|---|---|
| **Phase 0** | Baseline & Audit | Repository audit, test baseline, defect mapping | ✅ Complete |
| **Phase 1** | Safety & Purge | Remove PostgreSQL dependencies, enforce SQLite-only, verify paper defaults | ✅ Complete |
| **Phase 2** | Correctness & Risk Gate | Idempotency integration, central risk engine wiring, stop widening prevention | ✅ Complete |
| **Phase 3** | SQLite Architecture | Unified schema (`data/trading.db`), WAL/busy_timeout, backup & restore scripts | 🔄 In Progress |
| **Phase 4** | MT5 Broker Resilience | Session discovery, disconnect recovery, order timeout reconciliation | ⏳ Queued |
| **Phase 5** | Research & Validation | Backtesting integrity, walk-forward splits, look-ahead bias elimination | ⏳ Queued |
| **Phase 6** | Portfolio Intelligence | Multi-asset exposure caps, correlation guard, strategy lifecycle | ⏳ Queued |
| **Phase 7** | Windows 11 Operations | Pure-PowerShell toolchain (`doctor`, `install`, `start`, `backup`, `restore`) | ⏳ Queued |
| **Phase 8** | Release Verification | End-to-end tests, documentation sync, operator acceptance | ⏳ Queued |

---

## DETAILED TASK BREAKDOWN

### Phase 1 — Safety & Database Purge (Completed)
- [x] Establish 217 passing tests baseline.
- [x] Create `docs/PROJECT_AUDIT.md`.
- [x] Remove `asyncpg` and `psycopg` from `requirements-windows.txt`.
- [x] Purge PostgreSQL service probes and `psql`/`pg_dump` dependencies from `scripts/windows/*.ps1`.
- [x] Verify pure SQLite configuration in `core/settings.py` and `core/database/engine.py`.
- [x] Enforce automated tests run safely without external database or live broker dependencies (220 passing).

### Phase 2 — Correctness & Risk Enforcement (Completed)
- [x] Implement Breakeven + buffer (+0.15 ATR) and Trailing stop floor.
- [x] Connect `core/execution/idempotency.py` to order execution pipeline (`execute_proposal`, `submit_proposal`).
- [x] Guard against stop-loss loosening across all position management paths (`modify_position`).
- [x] Wire real `RiskEngine` kill switch to API endpoints (`/api/emergency-stop`, `/api/resume`, `/api/control`).
- [x] Unify trade recording in `run.py` to use `data/trading.db` via `_get_sqlite_db_path()`.

### Phase 3 — SQLite-Only Architecture (In Progress)
- [x] Standardize local SQLite storage to `data/trading.db` with auto-creation of `data/` directory.
- [x] Align trade history persistence between `run.py` and the unified schema.
- [x] Implement robust SQLite backup via Python `sqlite3.Connection.backup()` and PowerShell.
- [x] Implement database integrity verification (`PRAGMA integrity_check`) before and after backups.
- [ ] Add explicit stress unit tests for SQLite concurrency and busy timeout handling.

### Phase 4 — MetaTrader 5 Integration & Reconnection
- [ ] Standardize terminal discovery and reuse of active Windows terminal session across all modules.
- [ ] Enhance reconnect backoff and network recovery routines.
- [ ] Implement broker state reconciliation for unresolved orders.

### Phase 5 — Quantitative Research & Backtesting
- [ ] Verify walk-forward splits and out-of-sample data separation in `core/backtesting/engine.py`.
- [ ] Incorporate realistic spread and commission modeling per asset class.
- [ ] Validate feature calculation against look-ahead bias.

### Phase 6 — Portfolio Intelligence & Multi-Asset Exposure
- [ ] Wire broker-neutral `InstrumentRegistry` into runtime trade sizing.
- [ ] Implement portfolio-level margin and asset-class concentration limits.
- [ ] Formalize strategy lifecycle transitions (`RESEARCH` → `PAPER` → `LIVE`).

### Phase 7 — Native Windows Operations & PowerShell Toolchain
- [ ] Test and streamline all 15 PowerShell scripts in `scripts/windows/`.
- [ ] Guarantee zero administrative privilege requirements for standard bot operations.
- [ ] Verify clean startup (`start.ps1`), status reporting (`status.ps1`), and shutdown (`stop.ps1`).

### Phase 8 — Regression & Release Readiness
- [ ] Run full test suite; report passing/failing tests.
- [ ] Update `docs/PROJECT_STATUS.md` and related technical guides.
- [ ] Provide operator runbook for Windows 11 paper-trading deployment.
