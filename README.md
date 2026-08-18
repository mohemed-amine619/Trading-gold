# Trading Gold Bot

Asynchronous algorithmic trading bot for MetaTrader 5: multi-timeframe
EMA/RSI strategy, fixed-fractional risk, ATR trailing stops, PyQt5 GUI,
dry-run (paper trading) mode, and a Windows .exe build.

## Quick start (development, on Windows)

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python gui_main.py
```

Run the CLI version instead (no GUI):

```bat
python main.py
```

## Build the .exe (Windows only)

PyInstaller cannot cross-compile a Windows exe from Linux, so build on the
Windows machine itself. Requires 64-bit Python matching your MT5 terminal
bitness and the MetaTrader 5 terminal installed.

```bat
pip install pyinstaller
pyinstaller trading_bot.spec --noconfirm
```

Output: `dist\GoldTradingBot.exe` (single file, no console).

Alternative one-click build:

```bat
build_exe.bat
```

## First run (GUI)

1. Start the MetaTrader 5 terminal and log into your account.
2. Launch `dist\GoldTradingBot.exe` (or `python gui_main.py`).
3. In the **MT5 connection** panel enter: terminal path (optional, empty =
   auto-detect), login, password, server. Click **Save credentials** (stored
   in `credentials.json` next to the app - plain text, keep it safe).
4. Tick the symbols to trade (e.g. XAUUSDm, EURUSDm).
5. Keep **Dry-run** checked and press **Start Bot** - watch the Signals tab.
6. When you are confident, uncheck Dry-run and restart the bot for live
   trading.

## Configuration

Edit `config.py` before building for new defaults: risk per trade,
ATR multipliers, deviations, magic number, timeframes, alert webhooks.
Most risk parameters can also be changed live from the GUI Dashboard.

## Project layout

```
config.py                  all tunable parameters
main.py                    CLI entry point
gui_main.py                PyQt5 dashboard (build target of the exe)
build_exe.bat              one-click Windows build
trading_bot.spec           PyInstaller spec
core/
  mt5_engine.py            MT5 lifecycle, serialized calls, reconnection
  data_handler.py          OHLCV, server-timezone -> UTC, closed candles
  strategy.py              BaseStrategy + EMA/RSI multi-timeframe
  risk_manager.py          1% fractional sizing, margin checks, ATR trail
  execution.py             order_send wrappers, retcode handling, dry-run
  state_manager.py         position mirror filtered by magic number
  alerts.py                Telegram / Discord webhooks
  credentials_store.py     GUI credential persistence
  bot.py                   orchestrator shared by CLI and GUI
```

## Warnings

- This software trades real money when dry-run is OFF. Test thoroughly first.
- `credentials.json` stores your password in plain text - do not share it.
- Logs are written to `logs/trading_bot.log`.