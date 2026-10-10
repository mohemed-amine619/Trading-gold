# MT5 WINDOWS BRIDGE — Design Contract

> **Status: DESIGN ONLY — NOT YET IMPLEMENTED.**
> No bridge code exists in this repository. This document specifies the
> contract that will be built. Everything below is a design target, not
> current behaviour.

## 1. Why a bridge

MetaTrader5 is a **Windows-only** package — it has no Linux wheel, so
`pip install MetaTrader5` fails outright on Ubuntu and cannot run inside a
standard Linux container. The official MT5 terminal is likewise a Windows
application.

The platform therefore splits into two sides:

| | Side 1 — Ubuntu core | Side 2 — Windows bridge |
|---|---|---|
| Runs on | Ubuntu host, Docker Compose | Windows 11 VM |
| Contains | `core/`, `api/`, PostgreSQL, UI | MT5 terminal + bridge service |
| Knows about MT5 | **No** — never imports it | **Yes** — exclusively |
| Reaches broker | HTTP to the bridge | Direct MT5 API calls |

This split is already verified on Side 1: the Ubuntu image has no `MetaTrader5`
installed, and every module under `core/` and `api/` imports successfully
without it.

---

## 2. Network topology

```
┌─ Ubuntu host ──────────────────┐         ┌─ Windows 11 VM ──────────┐
│  app container  :8000          │         │  MT5 terminal             │
│  postgres       :5432          │  HTTP   │  bridge (FastAPI) :8100   │
│         │                      ├────────►│       │                  │
│         └── MT5_BRIDGE_URL     │  LAN/   │       └── MetaTrader5    │
└────────────────────────────────┘  NAT    └──────────────────────────┘
```

### 2.1 Discovering the VM address

**Do not hardcode a guessed IP.** The guest address is DHCP-assigned and can
change between boots.

```bash
# On the Ubuntu host
virsh domifaddr win11

# From inside the Windows VM
ipconfig
```

Current observed state on this host:

```
Name       MAC address          Protocol  Address
vnet9      52:54:00:36:56:15    ipv4      192.168.122.212/24
```

Note this is the libvirt **NAT** network (`virbr0`, `192.168.122.0/24`), not a
bridged LAN network. NAT is sufficient for host↔VM traffic but means the VM is
not directly reachable from other machines on the LAN. Switching to a bridged
interface is a deliberate network change — confirm before altering it.

For a stable address, either reserve the lease in the VM's DHCP settings or
assign a static IP inside Windows.

### 2.2 Testing connectivity

```bash
# 1. Network reachability
ping -c 3 192.168.122.212

# 2. Bridge liveness (once implemented)
curl http://192.168.122.212:8100/api/health

# 3. From inside the app container
docker compose exec app curl -v http://192.168.122.212:8100/api/health
```

> Docker's bridge network is a **separate** network namespace from the Ubuntu
> host. A container reaching the VM uses the host's routing; verify step 3
> explicitly rather than assuming step 2 implies step 3.

---

## 3. Configuration

Contract consumed by Side 1. Already implemented in `core/settings.py`:

| Variable | Default | Purpose |
|---|---|---|
| `MT5_BRIDGE_URL` | `""` (disabled) | Bridge base URL |
| `MT5_BRIDGE_TOKEN` | `""` | Bearer token for authentication |
| `MT5_BRIDGE_TIMEOUT_SECONDS` | `10` | Per-request timeout |
| `MT5_BRIDGE_TLS_VERIFY` | `true` | TLS certificate verification |

Accessors: `settings.bridge_configured`, `settings.mt5_bridge_http_url`
(trailing slash stripped), `settings.mt5_bridge_auth_headers`.

**Required safety behaviour:** when `MT5_BRIDGE_URL` is empty, the core must
treat every broker-dependent operation as unavailable and block it with a clear
reason. It must never silently no-op or pretend a broker call succeeded.

---

## 4. Bridge API surface

Versioned (`/api/v1/...`), typed request/response models, and **read-only by
default**. Order submission requires an explicit opt-in that is off by default.

| # | Endpoint | Method | Purpose | Write? |
|---|---|---|---|---|
| 1 | `/api/v1/health` | GET | Process liveness | no |
| 2 | `/api/v1/ready` | GET | Terminal connected + authenticated | no |
| 3 | `/api/v1/terminal/status` | GET | MT5 terminal state, build, last error | no |
| 4 | `/api/v1/account` | GET | Account id, currency, leverage, balance, equity, margin | no |
| 5 | `/api/v1/symbols` | GET | Dynamically discovered symbols | no |
| 6 | `/api/v1/symbol/{symbol}` | GET | Full instrument specification | no |
| 7 | `/api/v1/price/{symbol}` | GET | Current bid/ask | no |
| 8 | `/api/v1/rates/{symbol}` | GET | Historical candles | no |
| 9 | `/api/v1/positions` | GET | Open positions (filterable by magic) | no |
| 10 | `/api/v1/orders` | GET | Pending orders | no |
| 11 | `/api/v1/deals` | GET | Recent deals / trade history | no |
| 12 | `/api/v1/order/validate` | POST | Dry-run validation, no submission | no |
| 13 | `/api/v1/order` | POST | **Submit order — disabled unless opted in** | **yes** |
| 14 | `/api/v1/order/status` | GET | Execution state of a submitted order | no |
| 15 | `/api/v1/position/modify` | POST | Change SL/TP | **yes** |
| 16 | `/api/v1/position/close` | POST | Close / partial close | **yes** |

All endpoints mirror the existing `MT5Adapter` ABC in `core/mt5/adapter.py`
(10 methods). The bridge is the transport for that contract, not a redesign of
it.

---

## 5. Bridge security

- **Authentication on every protected endpoint.** `Authorization: Bearer
  <token>`, compared with a constant-time comparison.
- **Secrets never echoed.** Tokens and MT5 passwords must not appear in logs,
  responses or error messages.
- **Least privilege.** Bind to the private/VPN interface where possible, not
  `0.0.0.0` on a public network.
- **Restricted firewall.** Allow the bridge port only from the Ubuntu host's
  address. Never expose MT5 endpoints to the internet.
- **Strict request validation.** Pydantic schemas with bounded strings,
  symbols validated against discovered symbols, volumes bounded by instrument
  limits.
- **Rate limiting** on write endpoints.
- **Audit log** of every order-affecting call: timestamp, symbol, side,
  volume, request id, outcome.
- **TLS** for any deployment beyond a trusted lab network. Disable
  `MT5_BRIDGE_TLS_VERIFY` only deliberately — never as a workaround.
- **Versioned contract.** Breaking changes require a new path prefix.
- **Safe startup/shutdown.** Read-only mode must be the state after any
  failure or restart.

Token generation and distribution:

```bash
openssl rand -hex 32          # generate
# set identically in the Ubuntu .env and on the Windows VM
```

---

## 6. Execution safety (bridge side)

Before accepting a write request the bridge must verify:

1. Terminal is connected and the account is authenticated.
2. The connected account matches the **configured expected account** — never
   trust a caller-supplied `is_demo` boolean; read it from terminal metadata.
3. Trading mode permits writing (paper/live gate).
4. Symbol exists and is tradable; trading status permits it.
5. Volume respects instrument min/max/step.
6. Price, SL and TP satisfy stop-level and freeze-level constraints.
7. The kill switch permits the action.
8. The request carries a **unique idempotency key**, replayed requests must not
   create duplicate orders.
9. Risk checks have passed upstream.

### Timeout handling — critical

A timed-out HTTP request has an **unknown outcome**: the order may have been
accepted and filled. The correct behaviour is:

1. Do **not** blindly retry.
2. Query order/deal/position state by idempotency key.
3. Reconcile against broker state.
4. Only then decide whether a retry is safe.

### Execution states must be distinguished

`accepted`, `rejected`, `partially_filled`, `filled`, `cancelled`, `unknown`.

An HTTP 200 means "the bridge processed the request", **not** "the trade
filled".

---

## 7. Windows VM setup

```powershell
# In the Windows VM
cd C:\bridge
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-mt5.txt
```

`requirements-mt5.txt` is the **only** requirements file that may be installed
on Windows — it is the one that contains `MetaTrader5`.

Run the MT5 terminal, log in, enable API trading in the broker portal, then
start the bridge service. Register it to start automatically (Task Scheduler or
NSSM) so the terminal and bridge come up together after a reboot.

> The bridge service is **not yet implemented** — this section describes the
> intended install and will be updated when it exists.

---

## 8. Implementation checklist

- [ ] `bridge/` FastAPI service with typed schemas
- [ ] Token auth middleware with constant-time comparison
- [ ] Read-only default; writes gated behind explicit config
- [ ] Endpoints 1–11 (read-only surface)
- [ ] Order validate/submit/status with idempotency keys
- [ ] Unknown-outcome reconciliation path
- [ ] Structured audit log
- [ ] Graceful shutdown and reconnect/backoff
- [ ] Tests: auth, invalid input, terminal down, unsupported symbol,
      duplicate prevention, read-only enforcement, demo-mode safeguard
- [ ] Documented TLS guidance

---

## 9. Current verified state

| Claim | Status |
|---|---|
| Ubuntu image has no MetaTrader5 | **Verified** |
| All `core/`+`api/` modules import without it | **Verified** |
| Bridge settings exist in `core/settings.py` | **Verified** |
| Bridge service code | **Does not exist** |
| VM reachable at 192.168.122.212 | **Verified** (host ping, NAT) |
| MT5 installed inside the VM | **Unverified** — needs console access |
| Any end-to-end broker call | **Not possible yet** |

No broker connectivity, instrument discovery, or order submission has been
verified end-to-end. Do not describe the platform as broker-connected until the
bridge exists and has been exercised against a demo account.