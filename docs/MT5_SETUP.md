# MetaTrader 5 Setup

MetaTrader 5 is **Windows-only** — that constraint defines this platform's
architecture. There is no Linux wheel of the `MetaTrader5` package.

---

## 1. Install the terminal

1. Download from your broker, or <https://www.metatrader5.com/en/download>.
2. Install and **log in** with your account.
3. Enable trading automation:
   **Tools → Options → Expert Advisors → Allow automated trading → OK**

Leave the terminal open while the bot runs.

---

## 2. Install the Python package

Already handled by `requirements-windows.txt`. Verify:

```powershell
.\.venv\Scripts\python.exe -c "import MetaTrader5; print('installed')"
```

If installation fails, the wheel is not available for your Python version:

```powershell
# Download the cp312 win_amd64 wheel from
# https://pypi.org/project/MetaTrader5/#files
pip install .\MetaTrader5-5.0.x-py3-none-win_amd64.whl
pip install -r requirements-windows.txt
```

Python 3.12 has the best wheel availability. `doctor.ps1` warns if you are on
3.10 or 3.11.

---

## 3. Verify the connection

```powershell
.\.venv\Scripts\python.exe -c @"
import MetaTrader5 as mt5
if mt5.initialize():
    print('connected')
    acc = mt5.account_info()
    print('login   :', acc.login)
    print('server  :', acc.server)
    print('balance :', acc.balance)
    print('symbols :', len(mt5.symbols_get()))
    mt5.shutdown()
else:
    print('failed:', mt5.last_error())
"@
```

If `initialize()` returns `False`:

| Cause | Fix |
|---|---|
| Terminal not open | Start and log in first |
| Algo Trading disabled | Tools → Options → Expert Advisors |
| Timeout | Kill `terminal64.exe`, reopen, wait for charts |
| Wrong account | Re-login in the terminal |

An `initialize()` timeout usually means a stale IPC state in the terminal.

---

## 4. Configure credentials

In `.env`:

```dotenv
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server
```

`MT5_PATH` is only needed for a non-standard install location.

---

## 5. Discover what your broker offers

Do not assume an instrument exists. Ask the terminal:

```powershell
.\.venv\Scripts\python.exe -c @"
import MetaTrader5 as mt5
mt5.initialize()
syms = mt5.symbols_get()
print('total symbols:', len(syms))
for s in syms[:25]:
    print(' ', s.name, '-', s.description)
mt5.shutdown()
"@
```

Asset classes vary by broker. The platform supports forex, metals, indices,
energy and crypto — but **only instruments your broker actually offers**.
Anything unrecognised stays `UNKNOWN` rather than being guessed.

---

## 6. Inspect an instrument's contract

```powershell
.\.venv\Scripts\python.exe -c @"
import MetaTrader5 as mt5
mt5.initialize()
i = mt5.symbol_info('XAUUSDm')
for f in ('digits','point','trade_tick_size','trade_tick_value',
          'volume_min','volume_max','volume_step','contract_size',
          'trade_stops_level','trade_freeze_level'):
    print(f'{f:22}', getattr(i, f, None))
mt5.shutdown()
"@
```

These values drive position sizing. The platform reads them from the broker
rather than assuming — see `core/instruments/`.

---

## 7. Running the bot

```powershell
.\scripts\windows\start.ps1 -Bot
```

`start.ps1` refuses to start the bot if MetaTrader5 is not importable or the
terminal is not running, rather than starting something that cannot trade.

Without `-Bot`, only the API runs. That is useful for research and monitoring
and needs no terminal at all.

---

## 8. Health checks

```powershell
.\scripts\windows\doctor.ps1
```

Reports MT5 installation and terminal state as PASS / WARN / FAIL. A missing
terminal is reported as a broker-connectivity failure — **never** as a healthy
connection.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named MetaTrader5` | Not installed in this venv | `pip install -r requirements-windows.txt` |
| `initialize()` returns False | Terminal closed / not logged in | Open and log in |
| `initialize()` hangs | Stale IPC | Kill `terminal64.exe`, reopen |
| Symbol returns None | Broker does not offer it | Check `symbols_get()` |
| Prices stale | Market closed | Check trading status |

---

## 10. Optional: bridge mode

To run the engine elsewhere, expose this terminal over HTTP with
`bridge/run.py`. Read-only by default. See
[MT5_WINDOWS_BRIDGE.md](./MT5_WINDOWS_BRIDGE.md).