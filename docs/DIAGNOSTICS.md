# Diagnostics: getting real data to review the bot

The bot's log does not show how trades really ended. A stop-loss or take-profit
hit at the broker never goes through the bot, and the bot's own `trades` table
only records trades the bot closed itself. The MT5 deal history has every exit
with the broker's reason. `scripts/collect_diagnostics.py` reads it, together
with the log and the saved state, into one zip.

## Run it

On the machine where the bot and MT5 run (Windows), from the bot folder:

```
.venv\Scripts\python.exe scripts\collect_diagnostics.py --days 7
```

Options: `--tail 20000` (log lines kept), `--magic N` (default: `MAGIC_NUMBER` from
`.env`), `--all-magics` (include manual trades), `--out DIR`.

It writes `diagnostics\diagnostics_<time>.zip` and `diagnostics\latest_summary.md`.
Send `latest_summary.md` first. It is short and holds the findings.

## What is in the zip

| File | Contents |
|---|---|
| `summary.md` | Findings, trade statistics, exit reasons, hours, execution health, why signals did or did not trade |
| `stats.json` | The same numbers, machine-readable |
| `closed_trades.csv` | Every closed round trip from the broker history (net of commission and swap) |
| `log_tail.txt` | The end of `trading_bot.log`, scrubbed |
| `config_redacted.env` | Your `.env` with secrets blanked and the login masked |
| `account.json` | demo / real, balance, equity, masked login |
| `trades_db.csv`, `risk_state/`, `nn/` | The bot's trade table, saved risk state, model-training history, if present |

## Safety

* **Read-only.** It reads MT5 history and local files. It never sends an order
  and never changes a bot file. The only thing it writes is the output folder.
* **Secrets.** Values of any `.env` key containing PASSWORD, TOKEN, SECRET, API
  KEY or PRIVATE are removed everywhere, by value, wherever they appear. The
  account login is masked to its last four digits. Password/token patterns in
  the log are removed even when the value is unknown.
* **It refuses to write the zip** if a secret value is still present in any file.
* The scrubbing is thorough, not a guarantee. Look inside before you send it.
* `diagnostics/` is in `.gitignore`: it holds balances and trade history.

## Reading the findings

Each finding states the numbers it rests on. They are thresholds, not verdicts:

* stop-loss modified many times on one ticket: the trailing/profit-lock paths
  are sending a broker round trip per tick;
* minimum lot risks Nx the per-trade budget: size reductions do nothing once the
  broker's smallest lot is already larger than the budget;
* most losses closed within 5 minutes, or most trades ending at the stop: the
  stop is too tight for the noise;
* AI retraining accuracy below a coin flip.

The code is in `core/diagnostics/` (pure functions, tested without MT5) and the
tests are `tests/unit/test_diagnostics_core.py` and
`tests/unit/test_collect_diagnostics_cli.py`.
