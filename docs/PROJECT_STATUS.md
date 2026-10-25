# PROJECT STATUS

> **Target platform: native Windows 11.** No Docker, WSL, Linux host or VM is
> required. Last updated: 2026-10-25.

## 0. TEST BASELINE

**587 passing, 0 failing** — `.venv/bin/python -m pytest tests/ -q`

| Area | Tests | Status |
|---|---|---|
| AI model validation, calibration & edge-gated sizing | 39 | Implemented and tested |
| Trading safety (paper default & credential hygiene) | 23 | Implemented and tested |
| Safety tripwires & fail-closed regressions | 54 | Implemented and tested |
| Execution safety & idempotency | 34 | Implemented and tested |
| Bridge contract and auth | 31 | Implemented, tested against a mock |
| Instrument registry | 55 | Implemented and tested |
| Windows compatibility & SQLite persistence | 22 | Implemented and tested |
| Risk, position, strategy, scalping & market | 78 | Implemented and tested |
| **Market structure & location** | **70** | Implemented and tested |
| **Trading orchestrator (candles → orders)** | **19** | Implemented and tested |
| **Dynamic AI SL/TP Reasoning & LLM Trader Council** | **19** | **Implemented and tested** |
| **Config plane, risk ladder & Fibonacci scoring** | **80** | **Implemented and tested** |
| Manual trades isolation & order execution | 7 | Implemented and tested |
| Other unit tests | 56 | Implemented and tested |

**No automated test submits a real order.** All broker interaction uses
`MockMT5Adapter` or an in-memory scripted fake.

---

## 0.000 NEW — CONFIG PLANE, RISK LADDER & FIBONACCI SCORING

The dashboard could previously edit only the handful of settings that someone
had remembered to hand-wire into an `if` branch, and `cfg.get("missing_key")`
silently fell back to a call-site default. A setting could therefore be saved to
`.env`, show as applied, and have no effect. This phase replaces that pattern
with a declared contract and adds the first real multi-stage risk ladder.

### What is real now

| Piece | File | Guarantee |
|---|---|---|
| Parameter registry | `core/config/schema.py` | 91 declared params, each with type, bounds and group. A key not declared here **cannot** be written through the API. |
| Atomic write | `POST /api/config` | Whole payload validated first; any error rejects the *entire* batch. No half-applied risk change. |
| Dangerous-setting gate | `POST /api/config` | `dry_run` (paper → live) returns `409` unless the body carries `confirm: true`. |
| Audit trail | `core/config/audit.py` | One row per changed key in `config_audit` inside `data/trading.db` — still exactly one SQLite file. Fail-safe: an audit error never blocks a settings change. |
| Persisted risk state | `core/risk/state.py` | Daily/weekly/peak equity, cooldown and loss streak survive a restart (`data/state/risk_<symbol>.json`). |
| Risk ladder | `run.py` | At `reduce_size_at_dd` of the daily cap new size is halved; at `halt_at_dd` entries stop; at `max_daily_dd` / `max_weekly_dd` / `max_total_dd` the hard stop fires. |
| Loss-streak breaker | `run.py` | `max_consecutive_losses` consecutive losses start a `loss_streak_cooldown_minutes` cooldown. |
| Fibonacci scoring | `run.py` `_fib_score` | Contributes `fib_weight` points at retracement confluence; with `fib_enabled=False` the scorer reproduces the legacy 45/30/15/10 formula exactly (guarded by a test). |
| Dashboard | `dashboard.html` | "Master Parameters" panel renders `GET /api/config/schema`, so the UI cannot drift from the validator. Includes Reset DD Halt and the audit list. |
| Optional write auth | `run.py` middleware | When `DASHBOARD_TOKEN` is set, every mutating `/api/*` call requires `X-Dashboard-Token`. Reads stay open. Unset ⇒ unchanged behaviour. |

### Invariants preserved

- **Never loosen a stop.** `core/position/trailing.py` is untouched and
  `test_never_loosen_sl` still guards it.
- **No martingale, grid, or automatic stop-widening.** The ladder only reduces
  size or refuses entries.
- **Paper is still the default.** `dry_run=True`; going live needs an explicit
  `confirm: true` over HTTP *and* writes `TRADING_MODE=live` to `.env`.
- **Renormalised scoring.** `_score_direction` now sums configurable weights and
  rescales to 100 points, so `min_signal_score` keeps its meaning.

### Honest gaps (not done in this phase)

- The registry describes `run.py`'s `CONFIG`; `core/settings.py` still has its
  own defaults. The two are not unified — the dashboard writes affect `run.py`
  only.
- `core/strategy/registry.py` (the modular engine consuming these weights) does
  not exist yet; the scoring lives in `run.py`.
- The backtest tab in the dashboard is still a mock (`runBacktest()` does
  nothing). Live and backtest code are not yet shared.
- No paper-soak promotion gate yet: nothing automatically promotes a validated
  model or blocks go-live on poor out-of-sample results.
- `DASHBOARD_TOKEN` is a shared secret in `.env`, not per-user auth. Loopback
  binding remains the real control.

### Files

New: `core/config/__init__.py`, `core/config/schema.py`, `core/config/audit.py`,
`core/risk/state.py`, `tests/unit/test_config_schema.py`,
`tests/unit/test_risk_ladder.py`.
Changed: `run.py`, `dashboard.html`, `core/ai/provider.py`,
`core/ai/dynamic_sltp.py`, `core/ai/decision.py`, `core/strategy/strategies.py`,
`core/strategy/ensemble.py`, `tests/unit/test_dashboard_settings_api.py`,
`tests/unit/test_dynamic_sltp.py`, `tests/unit/test_run_py_wiring.py`,
`.gitignore` (runtime state directories).


---

## 0.00 NEW — ANTI-CHOKE CORRECTION BUFFER & 5-BAR FORWARD ML RESILIENCE

### Problem Solved
Previously, trades with accurate directional conviction were frequently stopped out prematurely during standard market pullbacks and candle wicks:
1. **Choked Stop-Loss Tightening:** Progressive breakeven and market-structure trailing stops had `min_dist = max(min_dist, atr * 0.05)`, placing stops as close as 15 cents below current market price on Gold.
2. **Aggressive Giveback Cuts:** Profit giveback triggered at 25% after only 0.7 ATR ($2.00 on Gold), closing winning positions on routine 38.2% Fibonacci pullbacks.
3. **Premature Scalp Rejection Exits:** Opposite pinbar exits closed trades at only 0.35 ATR of profit movement.
4. **Short Forward Horizon in ML Training:** 3-bar training horizon penalized trades that retraced for 1-2 bars before expanding toward the target.

### Solution Implemented
1. **Stage 6 Anti-Choke Correction Buffer:** `core/ai/dynamic_sltp.py` and `run.py` enforce a minimum clearance of `0.55 * atr` (Gold) / `0.45 * atr` (FX) between live market price and any candidate trailing stop loss, preserving natural breathing room while strictly preserving the never-loosen invariant.
2. **5-Bar Forward ML Horizon:** Expanded training target in `AIEnsemble._build_xy` from 3 to 5 bars (25 minutes on M5), teaching the model to reward setups that absorb initial corrections.
3. **Institutional Spatial Features:** Added `range_pos_20` (discount/premium relative positioning), `ema_stack` (ribbon conviction), and `vol_thrust` (volume-weighted directional impulse) to `FEATURE_COLS`.
4. **Giveback & Rejection Resilience:** Increased giveback tolerance to 45% and raised scalp rejection pinbar exit threshold to 1.25 ATR so winning runners can mature.

---

## 0.0 NEW — MARKET STRUCTURE & LOCATION ("knowing where the market is")

The bot previously had **no concept of location**. Its entire spatial model was
`pipeline.py`'s `_high20` / `_low20` — a 20-bar rolling high/low, used nowhere.
Strategies read one bar and compared it to an average, so a long entered at the
top of a range into a wall of equal highs scored *identically* to one entered at
a discounted swing low. `Signal.invalidation` was a hardcoded English string
("Price drops below recent support…"), so the bot could describe why it was
wrong but never name the price.

### Added

| Module | Responsibility |
|---|---|
| `core/market/swings.py` | Fractal swing detection. Confirmation enforced (trailing pivots are `confirmed=False` and may repaint); deterministic tie-breaking on plateaus. |
| `core/market/structure.py` | Structural bias from HH/HL vs LH/LL; BOS vs CHoCH; breaks judged on **body close, not wick**. Causal bias derivation. |
| `core/market/levels.py` | S/R as scored **zones**: ATR-derived clustering (multi-asset safe), strength from touches + recency + freshness + tightness; broken zones flip polarity. |
| `core/market/liquidity.py` | Stop pools (equal highs/lows) as RESTING / SWEPT / MITIGATED. A wick that closes back beyond the zone is a stop run, not a breakout. |
| `core/market/series.py` | Shared `resolve_atr` (was triplicated). |
| `core/market/location.py` | `MarketLocation`: range position, premium/discount, Fibonacci, `entry_gate()`, `structural_stop()`. |
| `core/trading/orchestrator.py` | Wires candles → location → signal → gate → stop → target → risk → execution. Returns a full audit trail. |

### Defects found and fixed while building this

1. **Premature premium/discount veto.** Premium/discount is a *ranging*
   concept. Applied strictly to a trend it blocked nearly every trend trade —
   a confirmed uptrend sits at the top of its own swing range by construction —
   and it **inverted the meaning** of price above the range high, which in an
   uptrend is strength, not weakness. The gate is now regime-aware: strict in
   ranging markets, permitted with the trend, and counter-trend only at the
   counter-extreme.
2. **Latent crash in `core/strategy/strategies.py`.** `latest.get("ema200", close)`
   returns `None` — not the fallback — when `pandas_ta` emits `None` during
   warm-up. `close > ema50 > None` raised `TypeError`, so **any symbol with
   under 200 bars of history crashed the entire ensemble** instead of
   abstaining. Now routed through a `_num` helper.
3. **Degenerate ATR could size a maximal position.** `structural_stop` will
   return `None` rather than a near-zero stop; `calculate_lot_size` divides by
   stop distance, so a collapsed stop is maximally dangerous, not conservative.
4. **Snap could breach the ATR bounds.** Tick-snapping after the floor/ceiling
   were applied produced stops at `0.749` ATR against a `0.75` floor and
   `3.004` against a `3.0` ceiling. Both bounds are now re-asserted after
   rounding.
5. **Single-bar ATR.** Reading only the latest true range let one spike bar
   redefine volatility; because every tolerance is an ATR multiple, that could
   make a tradable range fail its width test and silently flip location to
   `UNDEFINED`. Now smoothed over 14 bars.
6. **Sweeps were never detected.** The stop-run bar forms its own pivot, so
   scanning only past the swing cluster skipped the very bar that took the
   stops; every pool read as RESTING forever. Also, the wick was dragging the
   resting level upward.

### Honesty guarantees (each has a test)

- No confirmed range ⇒ `bias=UNDEFINED` and **trading is blocked**. Unknown
  location is never treated as neutral.
- Price outside the range is reported as a breakdown/breakout, **not** scored
  as an extreme premium/discount, and position is clamped to `[0, 1]`.
- No defensible stop ⇒ `None` ⇒ **no trade**. Never a default.
- Missing broker metadata ⇒ **no position size** (never a guessed size).
- Level prices snap to the broker grid only when `trade_tick_size`/`point` is
  supplied; otherwise they stay unrounded.

### Wiring status

`run.py` **now imports `core/market/location.py`** (defensively — if the
import fails, the bot logs a warning and trades exactly as before) and uses it
for three things:

| Config key | Default | Effect |
|---|---|---|
| `LOCATION_GATE` | `true` | veto entries whose location is indefensible |
| `LOCATION_STOP` | `true` | stop beyond the invalidating swing instead of a blind ATR multiple |
| `LOCATION_TARGET` | `true` | aim the take-profit at resting liquidity when one exists |
| `LOCATION_STOP_MIN_ATR` | `0.75` | floor so the stop sits outside bar noise |
| `LOCATION_STOP_MAX_ATR` | `3.0` | ceiling so a distant swing cannot shrink the position |

Two deliberate differences from the standalone orchestrator:

1. **`UNDEFINED` location does not veto in `run.py`.** In the orchestrator
   location is the primary input, so unknown means abstain. In `run.py` it is an
   enhancement on a signal stack that already works; vetoing on every cold start
   would be a silent kill switch. The existing ADX/session/score gates stay in
   force regardless.
2. **Every failure path falls back to prior behaviour**, never to a stop. A
   missing import, a short history frame, a malformed bar, or non-numeric broker
   metadata all degrade to the original ATR-multiple stop.

`core/trading/orchestrator.py` wires the *whole* `core/` stack and is tested
end-to-end, but is still not an entrypoint.

### Known gaps

- The risk engine logs the *cause* of a sizing refusal ("incomplete broker
  metadata") but its `reason` string surfaces only the derived symptom, which
  costs diagnosability when a live trade is rejected.
- The `trades` table `run.py` creates via raw `sqlite3` still lives outside the
  Alembic-managed schema.
- No backtest has been run against the location layer; the regime-aware gate in
  particular should be validated on history before it is trusted with live
  capital.

---

## 0.0.1 NEW — DYNAMIC AI SL/TP & EMBEDDED REASONING MODEL

The bot previously relied on static ATR multipliers or fixed R:R multiples for
stop-loss and take-profit, with no understanding of live market conditions,
liquidity magnets, or dynamic trade adaptation.

### Added

| Module | Responsibility |
|---|---|
| `core/ai/provider.py` | Self-contained `EmbeddedReasoningProvider` (runs locally in pure Python with zero external model dependencies), plus `LocalEndpointProvider` (OpenAI-compatible) and `OllamaProvider`. |
| `core/ai/dynamic_sltp.py` | `DynamicSLTPReasoner`: dynamic entry SL/TP (anchored to structural invalidation swings and liquidity pool magnets) + dynamic position ratcheting. |
| `SLTPSafetyGuard` | Mathematical safety invariants: **Never Loosen SL** (one-way ratchet), ATR floor/ceiling bounds, broker stops level compliance, half-up tick grid price snapping. |
| `run.py` & `core/settings.py` | Defensively wired into live bot order submission (`_send_order`) and continuous position management (`_manage_positions`). |

### Guarded Invariants

1. **Never loosen SL**: A BUY stop can only ratchet upward; a SELL stop can only ratchet downward. Any loosening suggestion is rejected immediately.
2. **Deterministic execution**: The embedded model runs in sub-millisecond time and requires no external servers, API keys, or GPU setup.
3. **Fail-safe fallback**: If any AI layer is disabled or unavailable, the bot falls back seamlessly to structural stops and standard trailing without interrupting live trading.

---

## 0.1 IMPLEMENTED — AI MODEL VALIDATION (champion/challenger)

`run.py` → `AIEnsemble` previously promoted **any** model that finished
fitting, and reported the raw soft-vote probability as "confidence". Three
concrete defects, all fixed:

1. **Silent model decay.** A retrain that had collapsed into a coin flip went
   straight into production. Every retrain is now a *challenger*: fitted on
   the older part of the rolling window, scored on a strictly later,
   never-trained-on validation split, and promoted only if it clears the
   gates. The incumbent survives a rejected challenger untouched.
2. **Uncalibrated confidence.** Raw `predict_proba` from a soft-voting
   ensemble is systematically overconfident, and that number drove both the
   45-point confluence score *and* the position-size multiplier. Output is
   now Platt-scaled against the validation split, clipped to `[0.02, 0.98]`
   so it can never report certainty it has not earned.
3. **Sizing on bravado.** Upsizing was `ai_conf >= 0.70/0.85` — magic numbers
   compared against an uncalibrated probability. Size now scales with the
   model's **edge** over a coin flip (`AI_MIN_EDGE_FOR_SIZE_UP`, default
   0.10), so a confidently-wrong prediction cannot enlarge a position.

### The decisive gate: lift over the trivial baseline

Accuracy alone is not evidence of skill, because the best trivial predictor
is *"always answer the majority class"*, which scores
`max(UP_rate, 1 - UP_rate)`. Measured on a validation window that was 73.8%
UP, a model scoring 61.3% accuracy — which the old code promoted and reported
as "61% accurate" — is **12.5 points worse than doing nothing**. `AI_MIN_OOS_EDGE`
(default `0.0`) blocks it. AUC is gated separately (`AI_MIN_OOS_AUC`) because
a model can post above-baseline accuracy while being unable to *rank*
outcomes at all (observed AUC: 0.523).

On an AR(1) control series: `phi=0` (random walk) is promoted but its
calibrated confidence lands at ~0.52 and it abstains; `phi=±0.5` (genuine
autocorrelation) promotes at 0.66–0.69 accuracy / 0.69–0.76 AUC / +0.11–0.13
lift and trades.

### Also fixed in this phase

- **Live crash risk.** The `AIEnsemble` config object is built inline in
  `_get_signal`. New keys were added to `CONFIG`, `core/settings.py`,
  `as_legacy_config()`, **and** that construction site — `train()` reads them
  outside any `try`, so a missing key would have raised `AttributeError`
  straight into the main loop. Guarded by a test.
- **Retrain spin.** The anti-spin guard keyed on `_n_trains`, which only
  increments on *promotion*. A model that never cleared the gate therefore
  looked brand new forever and re-entered the synchronous first-fit path on
  every bar. Now keyed on `_n_attempts`, counted in `update()`, and the
  cadence no longer short-circuits on `not self._trained`.
- **`y_tr` unbound** on the too-small-to-validate path (caught by test).
- **`_build_xy` raising** on short/malformed frames now returns `False`
  instead of propagating into the trading loop.
- **Wrong UP column.** `proba[1]` assumed ascending class order; now resolved
  from `classes_`.

### Known limitation (stated, not hidden)

Calibration is fitted on the same validation split that gates promotion, so
the reported confidence carries some optimism bias. Proper nested
cross-validation is the fix; it needs more labelled history than the current
window provides. Until then the number is *directionally* honest rather than
strictly out-of-sample.

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
- **Passed:** 232/232
- **Failed:** 0
- **Command:** `.venv/bin/python -m pytest tests/ -q`
- **Summary:** All unit and integration test suites passing (232 tests). Includes full safety gates, instrument registry, trailing logic, and progressive profit locking ratchets.

## 3.2 PROGRESSIVE BREAKEVEN & ENTRY RESPONSIVENESS (this phase)
- **Responsive Trade Execution:**
  - `MIN_SIGNAL_SCORE`: Lowered to 50 (was hardcoded 60) configurable via env.
  - `AI_CONFIDENCE_THRESHOLD`: Lowered to 0.55 (was hardcoded 0.62) configurable via env.
  - `ADX_MIN`: Lowered to 18 (was hardcoded 25) allowing earlier entries before trend exhaustion.
  - `USE_SESSION_FILTER`: Configurable (default `false`) permitting 24/5 trading when markets are open.
- **Continuous Progressive Breakeven & Profit Ratchet:**
  - `BREAKEVEN_TRIGGER_ATR`: Defaulted to 0.4 ATR (was 0.8 ATR) to protect positions very early.
  - `BREAKEVEN_BUFFER_ATR`: Defaulted to 0.15 ATR to lock in spread, commission, and swap fees.
  - `PROFIT_LOCK_RATIO`: Defaulted to 0.50 (50%). As profit expands beyond the trigger, the stop loss ratchets forward into guaranteed profit rather than remaining stationary at the initial entry line.
  - Stop loss strictly obeys directional monotonicity: it never loosens or widens upon market retracements.
  - Respects broker minimum stop levels (`trade_stops_level * point`) to prevent order rejection.
  - Implemented across both `run.py` and `core/position/trailing.py` / `core/settings.py`.

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
