---
name: run-trading-gold
description: Run, start, launch, screenshot, or drive the Trading-gold dashboard on Linux. Use to run the dashboard, take a screenshot, click EMERGENCY STOP, or test the write gate. Paper mode only.
---

# Run Trading-gold (dashboard, paper mode, on Linux)

Paths are relative to the repo root (`Trading-gold/`).

The real bot, `python run.py`, needs a live MetaTrader5 terminal, which only
exists on Windows. This skill runs the same dashboard that `run.py` builds,
with MetaTrader5 stubbed and the bot in paper mode, and drives it in
system Chrome. No broker is contacted.

## Prerequisites (verified)

- `.venv` with the project requirements (`.venv/bin/python -c "import fastapi, uvicorn, pandas_ta"` succeeds).
- Google Chrome at `/usr/bin/google-chrome` (version 154 used). Override with `CHROME_PATH`.
- Playwright, outside `.venv`:
  ```bash
  python3 -m pip install --target ~/.cache/trading-gold-browser playwright
  ```
  Then set `PYTHONPATH=~/.cache/trading-gold-browser` when running `drive.py`.

## Run (agent path)

1. Start the dashboard. It runs in the foreground, so background it and wait for it to come up:
   ```bash
   DASHBOARD_TOKEN=driver-token .venv/bin/python .claude/skills/run-trading-gold/serve_dashboard.py &
   sleep 8
   curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8765/api/config/schema   # expect 200
   ```
   It listens on `127.0.0.1:8765`. Change the port with `DASHBOARD_PORT`.

2. Screenshot the Overview. Output goes to `$SHOT_DIR/dashboard.png`, or to the path you pass:
   ```bash
   export PYTHONPATH=~/.cache/trading-gold-browser DASHBOARD_TOKEN=driver-token
   .venv/bin/python .claude/skills/run-trading-gold/drive.py shot /tmp/dashboard.png
   ```

3. Click EMERGENCY STOP in the browser. The driver answers the confirm and token prompts and prints each dialog:
   ```bash
   .venv/bin/python .claude/skills/run-trading-gold/drive.py estop
   # dialogs: [..."DANGER: ...", "This dashboard is write-protected...", "EMERGENCY STOP ACTIVATED."]
   ```

4. Check the write gate over HTTP:
   ```bash
   .venv/bin/python .claude/skills/run-trading-gold/drive.py writes
   # GET /api/config/schema -> 200, POST no token -> 401, wrong token -> 401, good token -> 200
   ```

5. Stop the server. Use `pgrep -f "[s]erve_dashboard.py"` and kill the PIDs it prints.
   Do not use `pkill -f serve_dashboard.py` from the same shell: the pattern also matches
   the shell running the command, which kills it (exit 144).

## Run (human path)

`python run.py` starts the live bot and its dashboard on a free port from 8000 to 8015. It needs
Windows and a logged-in MT5 terminal. On Linux it fails at `import MetaTrader5`.

## Gotchas

- The dashboard is `run.py`'s `create_app(bots)`. The `api/main.py` app is a separate FastAPI app
  with no dashboard route. `/api/health` exists only there, so on this launcher `/api/health` returns 404.
  Use `/api/config/schema` for readiness.
- `serve_dashboard.py` does not read `.env`, on purpose. No broker password or real
  `DASHBOARD_TOKEN` reaches this process. Set `DASHBOARD_TOKEN` in the environment you start it from.
- The MT5 stub must return callables for functions and ints for constants. An int for everything
  makes `positions_get()` raise `TypeError: 'int' object is not callable`, which surfaces as a 500 from
  `/api/emergency-stop`.
- With `DASHBOARD_TOKEN` set, the EMERGENCY STOP button needs the token. The browser asks once and
  stores it in `localStorage`. An operator who has never entered it sees a token prompt during an emergency.
- The browser stores the token per origin. A new browser profile prompts again.
- Chrome runs with `--no-sandbox`, which the driver passes. Headless runs have no display.
- The screenshot shows known defects: the Account Equity card prints raw `<span>` HTML, and Today's P&L
  shows `NaN%` with no position. These are not fixed here.

## Troubleshooting

- `ModuleNotFoundError: No module named 'playwright'` → set `PYTHONPATH` to the `--target` dir.
- `curl` returns 000 → the server is not up yet, or it crashed. Read its output:
  the launcher prints `dashboard on http://127.0.0.1:8765/ (paper mode, MT5 stubbed)` when ready.
- `EMERGENCY STOP FAILED` after a token is entered → check the server log for a 500 from `_close_all`.
