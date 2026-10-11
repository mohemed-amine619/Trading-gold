# Trading Safety

How the platform limits risk, and what stops it from trading when it should not.

---

## 1. Default posture

| Control | Default | Effect |
|---|---|---|
| `TRADING_MODE` | `paper` | Orders simulated; nothing reaches the broker |
| Unrecognised `TRADING_MODE` | → `paper` | Fails **closed**, never open |
| Manual positions | `observe_only` | Not touched |
| Magic number | `888999` | Isolates bot positions from yours |
| Missing instrument metadata | trade refused | Never guessed |

Going live requires **all** of: `TRADING_MODE=live`, `MT5_LOGIN`,
`MT5_PASSWORD`, `MT5_SERVER`, and a `DASHBOARD_SECRET_KEY` that is not
`CHANGE_ME`. `start.ps1` refuses to start otherwise.

---

## 2. Three independent gates

A strategy cannot bypass any of these.

### Gate 1 — Instrument (`core/instruments/`)

An instrument must have complete, fresh broker metadata before it can be
traded:

```python
spec = registry.require("XAUUSDm")   # raises ResolutionError if unusable
```

Refused when: symbol unknown, a critical field missing, metadata stale, or the
broker reports it untradable. The error names the exact missing field.

**Why this matters:** a defaulted tick value silently turns a risk limit into a
guess. Before this existed, every instrument returned gold-shaped constants and
position sizing collapsed to the broker minimum regardless of stop distance.

### Gate 2 — Risk (`core/risk/engine.py`)

`approve()` evaluates:

- risk per trade
- maximum open positions
- daily loss limit
- weekly loss limit
- maximum drawdown
- margin utilisation
- consecutive-loss streak
- minimum reward:risk
- spread limit
- **emergency stop state**

Every rejection records a reason. Rejections are persisted as risk events.

### Gate 3 — Execution (`core/execution/`)

Two guarantees, both tested:

**A decision produces at most one order.** Every submission carries an
idempotency key. Reusing a key returns the original report instead of sending
again.

**An uncertain outcome is never retried.** A timeout or dropped connection is
reported as `UNKNOWN`, because the broker may have filled the order. The
correct response is reconciliation, not a retry:

```
Submission signal-42 -> UNKNOWN (retcode 10035); not retrying
Duplicate suppressed for key=signal-42; reconcile with the broker
orders actually sent to broker: 1
```

---

## 3. Execution states

| State | Meaning | Safe to resubmit? |
|---|---|---|
| `filled` | Complete fill | No |
| `partially_filled` | Partial fill | No — reconcile |
| `rejected` | Broker refused | Yes, with a new key |
| `cancelled` | Cancelled | Yes, with a new key |
| `not_sent` | Requote; nothing pending | Yes, same key |
| `unknown` | **Outcome not known** | **No — reconcile** |

A successful HTTP response means the request was processed, **not** that the
trade filled.

---

## 4. Kill switch

```python
risk = RiskEngine()
risk.trigger_emergency_stop()   # blocks new orders immediately
risk.reset_emergency_stop()     # explicit, manual re-enable
```

`approve()` checks it first and rejects everything while engaged.

> **Known gap:** `api/main.py` `/api/emergency-stop` currently returns success
> without stopping anything, and nothing persists the state. `run.py`'s
> `/api/control` emergency action does call `_close_all`. Wiring the API route
> to the real switch is outstanding work.

Risk-reducing actions (closing positions) should follow a separately defined
emergency policy rather than being blocked alongside new orders.

---

## 5. Position management

Policies, conservative by default:

| Policy | Behaviour |
|---|---|
| `observe_only` | Track and report; never modify (**default**) |
| `protect_only` | Tighten stops; never close |
| `full_management` | Full trailing, breakeven, partial close |

### Stops are never widened

`core/position/trailing.py` enforces it: `validate_sl_modification` rejects any
change that would loosen a protective stop. Guarded by
`tests/test_trailing.py::test_never_loosen_sl`.

A protective stop only moves in your favour unless an explicitly approved
policy allows otherwise.

---

## 6. What is deliberately absent

- **No martingale** or loss-chasing position sizing.
- **No uncontrolled grid trading.**
- **No automatic risk escalation** after losses.
- **No martingale averaging down.**
- **No automatic live-mode enablement.**

---

## 7. Instrument and currency correctness

Position sizing uses real broker metadata:

```
lots = (equity × risk%) / (stop_ticks × tick_value)
```

Verified proportional behaviour — same $100 risk budget at different stops:

| Instrument | Stop | Size |
|---|---|---|
| XAUUSDm | $2.00 | 0.500 lots |
| EURUSD | $0.0020 | 0.500 lots |
| BTCUSD | $50.00 | 0.020 lots |
| US30.cash | $5.00 | 2.000 lots |

---

## 8. Credentials

Never logged. `core/logging_setup.py` masks:

- `password=`, `MT5_PASSWORD:`, `db_password=`
- `token=`, `secret=`, `api_key=`
- `Authorization: Bearer …`
- credentials inside database URLs

Verified end-to-end: a secret passed as a log argument is masked on disk.

---

## 9. Verification

```powershell
.\scripts\windows\test.ps1
.\scripts\windows\test.ps1 -Keyword safety
```

Relevant suites:

| File | Covers |
|---|---|
| `tests/unit/test_safety_paper_default.py` | Paper default, live opt-in, fail-closed |
| `tests/unit/test_execution_safety.py` | Idempotency, unknown outcomes |
| `tests/unit/test_instruments.py` | Metadata validation, proportional sizing |
| `tests/test_trailing.py` | Stop never widened |

No automated test submits a real order. Broker interactions use
`MockMT5Adapter`.

---

## 10. Limitations

Be aware:

- The API has **no authentication**. Loopback binding is a mitigation only.
- `/api/emergency-stop` does not currently stop anything (see §4).
- Money is stored as `Float`, not `Decimal`.
- Bridge position modify/close return **501**.
- **No broker connection has been verified end-to-end.**

No strategy here is presented as profitable. Backtest and demo results do not
predict live performance, and this platform makes no guarantee of returns.