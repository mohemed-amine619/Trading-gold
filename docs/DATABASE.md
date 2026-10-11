# Database

PostgreSQL is the primary database. SQLite is supported for single-user
research. Schema is managed by **Alembic** — never created implicitly.

---

## 1. Choose a backend

| | SQLite | PostgreSQL |
|---|---|---|
| Setup | none | install + create database |
| Concurrency | single writer | multi-user |
| Recommended for | research, single user | anything long-running |
| Config | `DATABASE_URL=sqlite:///./trading.db` | `DATABASE_URL=postgresql://user:pass@localhost:5432/trading` |

---

## 2. PostgreSQL on Windows

### Install

Download the installer from
<https://www.postgresql.org/download/windows/> (16.x recommended). Note the
superuser password — you need it for `setup-database.ps1`.

The installer registers a Windows service named `postgresql*`.

### Create the role and database

```sql
CREATE USER trading WITH PASSWORD 'your_password';
CREATE DATABASE trading OWNER trading;
```

Run from `psql` (found under `C:\Program Files\PostgreSQL\16\bin`).

### Configure

```dotenv
DATABASE_URL=postgresql://trading:your_password@localhost:5432/trading
```

Passwords containing `@`, `:` or `/` must be URL-encoded.

### Verify

```powershell
.\scripts\windows\setup-database.ps1
```

This **never** creates or drops a database. If the database is missing it
prints the exact SQL to run.

---

## 3. Migrations

```powershell
.\scripts\windows\migrate-database.ps1                 # apply
.\scripts\windows\migrate-database.ps1 -ShowCurrent    # inspect
.\scripts\windows\migrate-database.ps1 -Autogenerate -Message "add x"
```

Safe to re-run — applying already-applied migrations is a no-op.

### Legacy databases

A database created before Alembic (via the old `create_all` path) has the
schema but an **empty** `alembic_version` table. Applying migrations fails with
`table X already exists`. The script detects this and tells you to stamp:

```powershell
.\scripts\windows\migrate-database.ps1 -Stamp
```

Stamping records the revision; it does not modify data.

---

## 4. Schema

19 tables covering accounts, instruments, orders, positions, deals, strategy
signals, AI decisions, risk reviews and events, daily metrics, system config
and a config audit trail.

Current migration: `aedbfe3f600c` (initial schema).

---

## 5. Data types

> **Known limitation:** monetary columns use `Float`, not `Decimal`. This is
> not suitable for accounting-grade precision. Float rounding can accumulate
> over many trades. Conversion to `Numeric` is planned.

UTC is used for all timestamps.

---

## 6. Backup

```powershell
.\scripts\windows\backup.ps1
.\scripts\windows\backup.ps1 -KeepDays 30
```

PostgreSQL uses `pg_dump`; SQLite checkpoints WAL then copies. Output goes to
`backups\trading-<timestamp>.*`.

Retention removes **only** files matching `trading-*`.

Automate with Task Scheduler — daily, e.g. 02:00:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File C:\path\to\Trading-gold\scripts\windows\backup.ps1
```

---

## 7. Restore

```powershell
.\scripts\windows\restore.ps1 -File .\backups\trading-20260101-020000.sql
```

Restoring **overwrites** the database. The script:

1. Asks you to type `RESTORE` (case-sensitive).
2. Takes a safety backup of the current state.
3. Stops the API so nothing writes concurrently.

---

## 8. Connection handling

| Setting | Value | Reason |
|---|---|---|
| `pool_pre_ping` | on | Drops dead connections |
| `busy_timeout` (SQLite) | 5000 ms | Waits instead of failing immediately |
| SQLite journal | DELETE on Windows | WAL sidecars lock harder on Windows |
| `check_same_thread` | False | Required by SQLAlchemy for SQLite |

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `could not connect to server` | Is the service running? `Get-Service postgresql*` |
| `password authentication failed` | Check the URL credentials |
| `database does not exist` | `CREATE DATABASE` — the script will not do it |
| `table X already exists` | `migrate-database.ps1 -Stamp` |
| `database is locked` (SQLite) | Close other tools using the file |
| `psql not found` | Add `PostgreSQL\16\bin` to PATH |

---

## 10. Safety guarantees

The tooling **never** runs `DROP DATABASE`, `DROP SCHEMA`, `createdb` or
`alembic downgrade` automatically. Every destructive path requires an explicit
command and, for restore, typed confirmation.