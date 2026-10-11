# Trading Platform

A multi-asset algorithmic trading platform for **Windows 11**, integrating with
MetaTrader 5 and PostgreSQL. **No Docker, WSL, Linux host or virtual machine is
required.**

```
Windows 11
├── Python 3.12 + .venv
├── MetaTrader 5 terminal  ──┐
├── MT5 Python integration    ├── broker
├── FastAPI  (api/)          ┘
├── PostgreSQL 16
└── PowerShell tooling  (scripts/windows/)
```

---

## Quick start

```powershell
git clone <your-repo-url> Trading-gold
cd Trading-gold

.\scripts\windows\doctor.ps1      # see what is missing
.\scripts\windows\install.ps1      # venv + deps + config + database
.\scripts\windows\start.ps1        # start the API
```

Then open <http://127.0.0.1:8000/docs>.

Run the trading bot (requires a logged-in MetaTrader 5 terminal):

```powershell
.\scripts\windows\start.ps1 -Bot
```

Stop everything:

```powershell
.\scripts\windows\stop.ps1
```

---

## Safety defaults

| Setting | Default | Meaning |
|---|---|---|
| `TRADING_MODE` | `paper` | No order reaches the broker |
| `DRY_RUN` gating | on in paper mode | Orders are simulated |
| Manual positions | `observe_only` | Not touched unless configured |
| Kill switch | available | `RiskEngine.trigger_emergency_stop()` |
| Unknown order outcome | never auto-retried | Reconciled instead |
| Missing instrument metadata | trade refused | Never substituted with guesses |

**Paper mode is the default and cannot be bypassed by accident.** An
unrecognised `TRADING_MODE` falls back to paper, not live.

See [docs/TRADING_SAFETY.md](docs/TRADING_SAFETY.md).

---

## Scripts

All under `scripts/windows/`:

| Script | Purpose |
|---|---|
| `doctor.ps1` | Diagnose the install: PASS / WARN / FAIL per check |
| `install.ps1` | One-shot installer (venv → deps → config → database) |
| `setup-venv.ps1` | Create `.venv` with Python 3.12 |
| `install-dependencies.ps1` | Install pinned dependencies |
| `setup-database.ps1` | Verify/create the database |
| `migrate-database.ps1` | Apply Alembic migrations (`-Stamp`, `-ShowCurrent`) |
| `start.ps1` | Start the API (`-Bot`, `-Bridge`, `-Port`) |
| `stop.ps1` | Stop processes (`-Target`, `-WhatIf`) |
| `status.ps1` | Show process, health and database state |
| `test.ps1` | Run the test suite |
| `backup.ps1` | Back up PostgreSQL or SQLite |
| `restore.ps1` | Restore from a backup (asks first) |
| `update.ps1` | Back up, pull, reinstall, migrate, restart |
| `uninstall.ps1` | Remove local runtime files (keeps your data) |

`common.ps1` holds the shared helpers every script dot-sources.

---

## Running the tests

```powershell
.\scripts\windows\test.ps1
.\scripts\windows\test.ps1 -Keyword risk
```

**Never run bare `pytest`.** Always use `pytest tests\` (or `test.ps1`).

---

## Documentation

| Document | Covers |
|---|---|
| [docs/WINDOWS_INSTALLATION.md](docs/WINDOWS_INSTALLATION.md) | Step-by-step installation |
| [docs/WINDOWS_ARCHITECTURE.md](docs/WINDOWS_ARCHITECTURE.md) | Components and data flow |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every environment variable |
| [docs/MT5_SETUP.md](docs/MT5_SETUP.md) | MetaTrader 5 installation and diagnosis |
| [docs/DATABASE.md](docs/DATABASE.md) | PostgreSQL and SQLite setup |
| [docs/TRADING_SAFETY.md](docs/TRADING_SAFETY.md) | Risk controls and kill switch |
| [docs/BACKUP_AND_RECOVERY.md](docs/BACKUP_AND_RECOVERY.md) | Backup and restore |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common problems |
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | What is implemented vs. not |
| [AGENTS.md](AGENTS.md) | Engineering rules for this repo |

---

## Known limitations

Be aware of these — they are real, not hypothetical:

- **MetaTrader 5 is Windows-only.** That is the reason for this architecture.
- **The API has no authentication yet.** It binds to `127.0.0.1` only, which
  is a mitigation, not a security control.
- **Position modification and closing via the bridge return 501** — not
  implemented yet.
- **Money is stored as floating point**, not decimal. Not yet suitable for
  accounting-grade precision.
- **Backtesting is minimal** and has not been validated against look-ahead
  bias.
- **No broker connection has been verified end-to-end.** The broker adapter is
  tested against a mock only.
- **Docker files remain** as an optional Linux deployment path; they are not
  used by the Windows workflow.

Nothing here guarantees profitable trading. Backtest and demo results do not
predict live performance.

---

## License

Internal project. No warranty or financial advice is provided.