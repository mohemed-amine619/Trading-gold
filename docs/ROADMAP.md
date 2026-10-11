# ROADMAP — Implementation Status

> Persistent task list. Update at the end of every implementation cycle so a
> future session can resume without relying on conversation history.
>
> Last updated: 2026-10-10 · Branch: `main` · Working tree: clean

---

## Current state summary

| Phase | Name | Status |
|---|---|---|
| P0 | Baseline + safety hardening | ✅ Complete |
| P1 | Correctness / safety fixes | ✅ Complete |
| P2 | Data models + migrations + DB lifecycle | ✅ Complete |
| P3 | Docker Compose deployment | ✅ Complete (verified) |
| P4 | Windows MT5 bridge | ⛔ **Blocked** — needs VM console access |
| P5 | Instrument discovery & metadata | ✅ Complete (this cycle) |
| P6 | Execution safety / idempotency | ⏳ Next |
| P7 | Central risk engine + kill switch wiring | ⏳ Queued |
| P8 | Position management & broker-valid SL/TP | ⏳ Queued |
| P9 | Quant engine / backtesting hardening | ⏳ Queued |
| P10 | Monitoring & analytics | ⏳ Queued |
| P11 | Release hardening | ⏳ Queued |

**Test baseline: 135 passing** (was 81 at session start).

---

## ✅ Completed

### P0 — Baseline
- `AGENTS.md` written from verified repository inspection.
- Established `.venv` (Python 3.12.3) as the working interpreter.
- Baseline recorded before changes.

### P1 — Safety
- **Fixed: live trading by default.** `dry_run` was read by every money-moving
  method but was absent from `CONFIG`, so `cfg.get("dry_run")` returned `None`
  → falsy → real orders. Now driven by `TRADING_MODE`, defaulting to `paper`,
  failing *closed* on unrecognised values.
- Fixed dashboard mode reporting (`paper_trading` key existed nowhere).
- 18 regression tests lock this behaviour.
- Verified end-to-end with a stubbed MT5 module: paper mode computes a full
  order but never calls `order_send`; live mode does call it.

### P2 — Persistence
- First real Alembic migration (`aedbfe3f600c`, 19 tables).
- `alembic/script.py.mako` was **missing entirely** — autogenerate failed.
  Added.
- Verified `upgrade → downgrade → upgrade` on PostgreSQL 16 and SQLite.
- Fixed latent `greenlet` dependency omission that crash-looped the image.

### P3 — Docker
- `requirements-core.txt` (Linux) / `requirements-mt5.txt` (Windows) split.
- `Dockerfile` (non-root), `.dockerignore`, `compose.yaml`, `docker/entrypoint.sh`.
- `/api/ready` added (503 when DB down); startup no longer crash-loops.
- Verified: persistence across container recreate, loopback-only ports,
  clean SIGTERM shutdown, no secrets in image.
- See `docs/DEPLOYMENT_UBUNTU.md`.

### P5 — Instrument engine (this cycle)
- New `core/instruments/` — broker-neutral model + registry.
- **Fixed: position sizing was inert.** `MockMT5Adapter.get_symbol_info`
  returned gold-shaped constants (`contract_size=100000`,
  `tick_size=0.00001`, `volume_min=0.01`) for *every* symbol, so every
  computed size floored to 0.01 and never responded to stop distance or risk
  budget. Verified before the fix: a $0.50 and a $50 stop both returned 0.01.
- `core/mt5/types.py SymbolInfo` now uses `Optional[...] = None` for contract
  fields — absence is now representable instead of defaulted.
- Registry refuses to size/trade an instrument with missing or stale metadata
  (mandate rule 10). Verified: refuses with an explicit reason.
- 53 new tests. Three pre-existing tests were asserting the *buggy* values
  and were corrected, not weakened.

---

## ⛔ Blocked

### P4 — Windows MT5 bridge
**Blocked on:** console/SSH access to the Windows VM.

Design contract is complete in `docs/MT5_WINDOWS_BRIDGE.md` (16 endpoints,
auth model, idempotency and unknown-outcome handling, checklist). No bridge
code exists yet.

Known environment facts:
- VM `win11` reachable at **192.168.122.212** (libvirt **NAT**, `virbr0` —
  not bridged as the mandate specifies). DHCP; re-check with
  `virsh domifaddr win11` before relying on it.
- MT5 installed inside the VM: **unverified**.
- Bridge port 8100: closed (nothing built).

### Outstanding risk — requires user action
- **Live broker credentials remain in `run.py:130-132`** (account, password,
  broker server) as `os.getenv` fallbacks. The user chose to keep them for
  now. **The password must be rotated at the broker** — it is already in git
  history and cannot be removed by a code change.
- Two PostgreSQL databases now exist (Docker container + one the user
  installed in the Windows VM). Exactly one must own trading data.

---

## ⏳ Next up (in order)

**P6 — Execution safety / idempotency.** Highest remaining safety value.
- Idempotency keys on order submission.
- Explicit execution states: accepted / rejected / partial / filled /
  cancelled / **unknown**.
- Timeout must trigger reconciliation, never a blind retry.
- `core/execution/engine.py` currently retries only on requote.

**P7 — Risk engine wiring.** `/api/emergency-stop` in `api/main.py` returns
success and stops nothing. `core/risk/engine.py:218` has a real kill switch
that no route calls. Wire it, and persist the state.

**P8 — Position management.** Verify SL/TP modifications against instrument
tick size / freeze level using the new registry.

---

## Known limitations (honest status)

- `/api/status` returns hardcoded values; `/api/emergency-stop` and
  `/api/resume` are stubs.
- No trading loop runs in the container — `core/` engines are not yet wired
  to a scheduler.
- The API has **no authentication**; loopback binding is a stopgap.
- Money is stored as `Float`, not decimal-safe.
- Backtesting (`core/backtesting/engine.py`) is minimal and unverified.
- No broker connectivity has ever been tested end-to-end.
- Asset classes beyond forex/metal/crypto/index/energy have no verified
  broker adapter.