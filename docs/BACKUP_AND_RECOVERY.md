# Backup and Recovery

Trading history and configuration are irreplaceable. This covers how to back
them up and — more importantly — how to prove a restore works.

---

## 1. What to protect

| Asset | Location | Loss impact |
|---|---|---|
| Database | PostgreSQL / `trading.db` | **Total** trade history |
| Configuration | `.env` | Credentials, broker settings |
| Logs | `logs/` | Audit trail |
| Source | repository | Git history |

---

## 2. Backing up the database

```powershell
.\scripts\windows\backup.ps1
.\scripts\windows\backup.ps1 -KeepDays 30
```

Output: `backups\trading-<yyyyMMdd-HHmmss>.<ext>`

| Backend | Method | Restores to |
|---|---|---|
| PostgreSQL | `pg_dump` (plain SQL) | Any PostgreSQL |
| SQLite | file copy after WAL checkpoint | SQLite |

Retention deletes **only** files matching `trading-*`, older than `-KeepDays`.

### Back up `.env` separately

It is git-ignored, so Git will not protect it. Store it in a password manager
or an encrypted backup — **never** in the repository.

---

## 3. Restoring

```powershell
.\scripts\windows\restore.ps1 -File .\backups\trading-20260101-020000.sql
```

The script will:

1. Show exactly what will be overwritten.
2. Require you to type `RESTORE` (case-sensitive).
3. **Take a safety backup of the current database first.**
4. Stop the API so nothing writes during the restore.
5. Restore, then verify.

Verify afterwards:

```powershell
.\scripts\windows\migrate-database.ps1 -ShowCurrent
.\scripts\windows\doctor.ps1
```

---

## 4. A backup is not verified until restored

> **An untested backup is a hope, not a plan.**

Test the full cycle on a scratch database at least once:

```powershell
# 1. Back up
.\scripts\windows\backup.ps1

# 2. Point at a scratch database
#    (edit DATABASE_URL in .env, or export it for this session)
$env:DATABASE_URL = "sqlite:///./restore-test.db"

# 3. Restore into it
.\scripts\windows\restore.ps1 -File .\backups\<your-backup> -Force

# 4. Confirm the schema and data survived
.\.venv\Scripts\python.exe -c "import sqlite3; print(sqlite3.connect('restore-test.db').execute('select count(*) from strategies').fetchone())"
```

Only once this succeeds is the backup process trustworthy.

---

## 5. Automating with Task Scheduler

Daily backup at 02:00:

```powershell
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File C:\path\to\Trading-gold\scripts\windows\backup.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
Register-ScheduledTask -TaskName "TradingPlatformBackup" -Action $action -Trigger $trigger `
  -Description "Daily database backup"
```

Remove it:

```powershell
Unregister-ScheduledTask -TaskName "TradingPlatformBackup" -Confirm:$false
```

---

## 6. Recovery scenarios

### The application crashed

```powershell
.\scripts\windows\doctor.ps1
.\scripts\windows\start.ps1
```

### The database is corrupted

```powershell
.\scripts\windows\stop.ps1
.\scripts\windows\restore.ps1 -File .\backups\<most recent>
.\scripts\windows\migrate-database.ps1
.\scripts\windows\start.ps1
```

### `.env` was lost

Restore from your password manager, or re-create from `.env.example` and
re-enter credentials.

### Positions differ from the broker after a crash

The platform reconciles broker state against internal state before trading.
**Verify manually before going live:**

```powershell
.\scripts\windows\start.ps1
# compare /api/positions with your MT5 terminal
```

---

## 7. PostgreSQL service recovery

```powershell
Get-Service postgresql*                       # status
Restart-Service postgresql*                  # restart
Start-Service postgresql*                     # start
```

---

## 8. Uninstall / rollback

```powershell
.\scripts\windows\stop.ps1
.\scripts\windows\uninstall.ps1
```

Removes the virtual environment and runtime directories. **Preserves** source
code, `.env`, and `backups\`. Removes the database only with an explicit
`-RemoveDatabase` and typed confirmation.

---

## 9. Checklist before relying on backups

- [ ] `backup.ps1` runs without error
- [ ] A restore has been tested into a scratch database
- [ ] Scheduled task created and verified
- [ ] `.env` stored outside the repository
- [ ] Backups copied to a second location or external drive

```powershell
Get-ScheduledTask -TaskName "TradingPlatformBackup" | Get-ScheduledTaskInfo
```