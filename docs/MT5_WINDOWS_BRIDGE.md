# MT5 WINDOWS BRIDGE — Design & Setup

> **Status: implemented and tested (31 tests) — NOT yet run against a live terminal.**
> Bridge code lives in `bridge/`. It runs on Linux for tests, but has never been
> connected to a real MetaTrader 5 terminal. Treat broker connectivity as
> unverified until you have run the checks in §7.

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

Verified on Side 1: the Ubuntu image has no `MetaTrader5`, and every module
under `core/` and `api/` imports successfully without it.

---

## 2. What is implemented

| Endpoint | Purpose | Write? |
|---|---|---|
| `GET /api/health` | Process liveness. No auth, leaks nothing. | no |
| `GET /api/v1/ready` | Terminal connected + correct account. No auth. | no |
| `GET /api/v1/terminal/status` | Terminal state, last error | no |
| `GET /api/v1/account` | Balance, equity, margin, leverage | no |
| `GET /api/v1/symbols` | **Dynamic** symbol discovery | no |
| `GET /api/v1/symbol/{symbol}` | Full instrument specification | no |
| `GET /api/v1/price/{symbol}` | Current bid/ask | no |
| `GET /api/v1/rates/{symbol}` | Historical candles | no |
| `GET /api/v1/positions` | Open positions | no |
| `GET /api/v1/deals` | Recent deals | no |
| `POST /api/v1/order/validate` | Dry-run validation | no |
| `POST /api/v1/order` | **Submit order — disabled by default** | **yes** |
| `POST /api/v1/position/modify` | SL/TP change — **501 not implemented** | **yes** |
| `POST /api/v1/position/close` | Close position — **501 not implemented** | **yes** |

### Safety model (all test-enforced)

- **Fails closed.** No `MT5_BRIDGE_TOKEN` ⇒ every protected endpoint returns
  503. A misconfigured bridge never becomes an open trading endpoint.
- **Constant-time token comparison**; tokens are never logged or echoed.
- **Writes disabled by default.** Requires `MT5_BRIDGE_ALLOW_WRITES=1`.
- **Account identity comes from the terminal**, never the request body. The
  connected account must equal `MT5_BRIDGE_EXPECTED_LOGIN`.
- **Idempotency keys** — a repeat returns 409 instead of placing a second order.
- **Unknown outcomes stay unknown.** A timeout is reported as `unknown`, never
  as `rejected`, so the core reconciles instead of blindly resubmitting.

---

## 3. Windows VM setup

### 3.1 Connect to the VM

```bash
# VNC is already listening (port 5900 confirmed open)
virt-manager            # graphical, double-click win11
```

Or find the VNC port: `virsh domifaddr win11` / `virsh vncdisplay win11`.

### 3.2 Inside the VM: install MetaTrader 5

1. Download MT5 from your broker and install it.
2. Log in with the **demo** account first.
3. Enable **Algo Trading / API trading** (Tools → Options → Expert Advisors).
4. Confirm the terminal stays open — the bridge cannot connect without it.

### 3.3 Inside the VM: install the bridge

Open PowerShell in the repository folder (shared folder or clone):

```powershell
cd C:\path\to\Trading-gold
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-mt5.txt
```

`requirements-mt5.txt` is the **only** file that may be installed on Windows —
it is the one containing `MetaTrader5`.

### 3.4 Create the VM's `.env`

```
MT5_BRIDGE_TOKEN=<paste the token generated below>
MT5_BRIDGE_HOST=127.0.0.1
MT5_BRIDGE_PORT=8100
MT5_BRIDGE_ALLOW_WRITES=0
MT5_BRIDGE_EXPECTED_LOGIN=<your MT5 account number>

MT5_LOGIN=<account>
MT5_PASSWORD=<password>
MT5_SERVER=<broker server>
```

Generate the token **once**, then use the same value in both places:

```bash
# on Ubuntu — run this and copy the output
openssl rand -hex 32
```

### 3.5 Start the bridge

```powershell
py bridge\run.py
```

It prints its configuration (never the token value):

```
MT5 Bridge 1.0.0 starting
  bind            : 127.0.0.1:8100
  expected login  : 12345678
  writes enabled  : False
  token           : configured (64 chars, value not logged)
```

---

## 4. Network: allow Ubuntu to reach the bridge

The bridge binds to `127.0.0.1`, which is **not reachable from Ubuntu**. Pick
one:

**Option A — reverse proxy (recommended).** Keep the bridge on loopback and
expose it deliberately.

**Option B — bind to the private interface.** Set `MT5_BRIDGE_HOST` to the
VM's address (e.g. `192.168.122.212`) and open the Windows Firewall for that
port **restricted to the Ubuntu host address only**:

```powershell
New-NetFirewallRule -DisplayName "MT5 Bridge (Ubuntu only)" `
  -Direction Inbound -Protocol TCP -LocalPort 8100 -Action Allow `
  -RemoteAddress 192.168.122.1
```

Never open 8100 to `Any`. Never expose it to the public internet.

---

## 5. Configure Ubuntu to use the bridge

In the Ubuntu `.env`:

```
MT5_BRIDGE_URL=http://192.168.122.212:8100
MT5_BRIDGE_TOKEN=<the same token>
MT5_BRIDGE_TIMEOUT_SECONDS=10
MT5_BRIDGE_TLS_VERIFY=true
```

Restart the app so the new environment is applied:

```bash
docker compose up -d
```

---

## 6. Discover and verify the VM address

**Do not hardcode a guessed IP.** The guest address is DHCP-assigned and can
change between boots. The interface name has already changed once on this host
(`vnet9` → `vnet10`).

```bash
# Current observed address
virsh domifaddr win11
#  vnet10  52:54:00:36:56:15  ipv4  192.168.122.212/24
```

### Network topology note

This VM sits on the libvirt **NAT** network (`virbr0`, `192.168.122.0/24`),
**not** the bridged LAN network the mandate describes. NAT works for
host↔VM traffic and has the security benefit that the VM is unreachable from
other LAN machines. Switching to a bridged interface is a deliberate change to
your host network — make it deliberately, and update `docs/NETWORKING_BRIDGED_VM.md`.

---

## 7. Verification checklist

Run these in order. Anything that fails tells you where the problem is.

```bash
# 1. Host can reach the VM at all
ping -c 3 192.168.122.212

# 2. Bridge is listening on the VM
curl http://192.168.122.212:8100/api/health

# 3. Bridge process is alive (no auth needed)
curl http://192.168.122.212:8100/api/v1/ready

# 4. Auth works from the host (token required)
curl -H "Authorization: Bearer $MT5_BRIDGE_TOKEN" \
     http://192.168.122.212:8100/api/v1/account

# 5. Auth works from INSIDE the container (different network namespace!)
docker compose exec app python -c "
import os,urllib.request
r=urllib.request.Request('http://192.168.122.212:8100/api/v1/account',
  headers={'Authorization':'Bearer '+os.environ['MT5_BRIDGE_TOKEN']})
print(urllib.request.urlopen(r, timeout=10).status)"
```

Step 5 matters specifically: **Docker's bridge network is a different network
namespace from the Ubuntu host.** A successful host `curl` does not prove the
container can reach the VM. If step 5 fails but step 4 succeeds, the problem is
routing/firewall, not the bridge.

### Expected results

| Step | Healthy result |
|---|---|
| 2 | `{"status":"ok","service":"mt5-bridge"}` |
| 3 | `ready: true`, `terminal_state: logged_in` |
| 4 | 200 with balance/equity |
| 5 | `200` |

If step 3 says `"running in MOCK mode"`, MetaTrader5 is not installed in the
VM's Python environment — check step 3.3.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Connection refused | Bridge not running, or bound to loopback | Check `bridge\run.py` output; set `MT5_BRIDGE_HOST` |
| 401 unauthorized | Token mismatch | Both sides must hold the identical value |
| 503 on every endpoint | `MT5_BRIDGE_TOKEN` unset on the VM | Set it; the bridge refuses to start without it |
| 403 on order endpoints | Writes disabled | Expected. Set `MT5_BRIDGE_ALLOW_WRITES=1` |
| `ready: false, MOCK mode` | MetaTrader5 missing in VM venv | Reinstall `requirements-mt5.txt` |
| Works on host, fails in container | Firewall/routing | Open the firewall rule restricted to the host IP |
| 409 on order submit | Idempotency key reused | Use a new key per decision |

---

## 9. Honest status

| Claim | Status |
|---|---|
| Ubuntu image has no MetaTrader5 | **Verified** |
| All `core/`+`api/` modules import without it | **Verified** |
| Bridge settings in `core/settings.py` | **Verified** |
| Bridge service code | **Implemented, 31 tests passing** |
| Bridge auth / write-gating / idempotency | **Verified on Linux with a mock** |
| VM reachable at 192.168.122.212 | **Verified** (host ping, NAT) |
| Bridge running inside the VM | **Not yet — requires your setup (§3)** |
| MT5 installed inside the VM | **Unverified** |
| Any real broker call | **Not yet performed** |

No broker connectivity, instrument discovery, or order submission has been
verified end-to-end. Do not describe the platform as broker-connected until
§7 passes against a demo account.