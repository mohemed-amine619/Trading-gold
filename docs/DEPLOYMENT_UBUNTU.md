# DEPLOYMENT — Ubuntu Core (Docker Compose + PostgreSQL)

The Ubuntu core runs the application, API and database. **It does not contain
MetaTrader5.** The MT5 terminal and its bridge run on a Windows 11 VM and are
reached over HTTP — see [MT5_WINDOWS_BRIDGE.md](./MT5_WINDOWS_BRIDGE.md).

---

## 1. Prerequisites

| Requirement | Check |
|---|---|
| Docker Engine | `docker --version` |
| Docker Compose v2 | `docker compose version` (must be `v2.x`+) |
| Host ports free | 8000 (app) and 5432 (database) |
| ~2 GB disk | image + PostgreSQL volume |

Verified on Ubuntu 24.04.5 LTS, Docker 29.8.2, Compose v5.5.1, PostgreSQL 16.

---

## 2. First-time setup

### 2.1 Create the environment file

```bash
cd /path/to/Trading-gold
cp .env.example .env
```

Generate a strong database password:

```bash
openssl rand -base64 32
```

Edit `.env` and set at minimum:

```dotenv
POSTGRES_PASSWORD=<paste the generated value>
```

> `.env` is git-ignored. Never commit it. Compose refuses to start without
> `POSTGRES_PASSWORD` rather than falling back to an empty password.

### 2.2 Optional: point at your Windows VM bridge

```dotenv
MT5_BRIDGE_URL=http://<windows-vm-ip>:8100
MT5_BRIDGE_TOKEN=<same token configured on the VM>
```

Leave `MT5_BRIDGE_URL` empty to run with no broker connection at all.

### 2.3 Start the stack

```bash
docker compose up -d --build
```

On first run the app container applies migrations before serving:

```
[entrypoint] applying database migrations...
INFO  [alembic.runtime.migration] Running upgrade  -> aedbfe3f600c, initial schema
[entrypoint] starting application...
```

### 2.4 Verify

```bash
docker compose ps
```

Both services must report `(healthy)`:

```
NAME                 STATUS                    PORTS
trading-gold-app-1   Up (healthy)              127.0.0.1:8000->8000/tcp
trading-gold-db-1    Up (healthy)              127.0.0.1:5432->5432/tcp
```

```bash
curl http://127.0.0.1:8000/api/health   # {"status":"ok"}          — process alive
curl http://127.0.0.1:8000/api/ready    # {"status":"ready",...}   — DB reachable
curl http://127.0.0.1:8000/api/status   # trading_mode: paper
```

---

## 3. Everyday commands

```bash
docker compose up -d --build      # start (rebuilds image)
docker compose up -d              # start (reuses image)
docker compose ps                 # status + health
docker compose logs -f app        # follow application logs
docker compose logs -f db         # follow database logs
docker compose restart app        # restart just the app
docker compose down               # STOP — keeps all data
docker compose down -v            # STOP and DELETE the database volume
```

> **`down -v` destroys every trade, position and audit record.** Only use it
> to intentionally reset a development database. Backup first — see
> [DATABASE_BACKUP_RESTORE.md](./DATABASE_BACKUP_RESTORE.md).

---

## 4. Migrations

Schema is owned by Alembic. The container runs `alembic upgrade head` on every
start, and it is idempotent — already-migrated databases are a no-op.

```bash
# Current revision
docker compose exec app alembic current

# Apply pending migrations (automatic at startup, manual here)
docker compose exec app alembic upgrade head

# Create a new migration after changing core/database/models.py
docker compose exec app alembic revision --autogenerate -m "describe the change"

# Roll back one revision
docker compose exec app alembic downgrade -1

# Full history
docker compose exec app alembic history
```

**Always review a generated migration before committing it.** Alembic's
autogenerate can miss data-preserving concerns (column renames appear as
drop + add and destroy data).

---

## 5. Health and readiness

| Endpoint | Meaning | Used by |
|---|---|---|
| `/api/health` | Process is alive. Does not touch the database. | Dockerfile `HEALTHCHECK` |
| `/api/ready` | Process is alive **and** the database is reachable. Returns **503** otherwise. | Compose healthcheck |

`/api/ready` deliberately does not crash when PostgreSQL is down — the app
stays up and reports `not_ready`, so orchestration can distinguish "waiting for
the database" from "broken process".

```bash
curl -i http://127.0.0.1:8000/api/ready | head -1   # HTTP/1.1 200 or 503
```

---

## 6. Security properties

| Property | How it is enforced |
|---|---|
| PostgreSQL not exposed | Bound to `127.0.0.1` only; app reaches it over the private `trading_net` bridge |
| API not exposed | Bound to `127.0.0.1` only |
| No secrets in image | `.dockerignore` excludes `.env`; credentials injected at runtime |
| Non-root container | Runs as `appuser` (uid 1000) |
| Credential-safe logs | `_redact_url()` strips `user:password@` before logging the DB URL |
| Missing password | Compose aborts rather than defaulting to empty |
| Reduced privileges | `security_opt: no-new-privileges:true` |

Verify the port bindings at any time:

```bash
docker inspect trading-gold-db-1 --format '{{json .NetworkSettings.Ports}}'
# {"5432/tcp":[{"HostIp":"127.0.0.1","HostPort":"5432"}]}
```

> The API currently ships with **no authentication**. It is loopback-bound as
> a stopgap only. Do not expose port 8000 beyond `127.0.0.1` until auth is
> added — see [SECURITY.md](./SECURITY.md).

---

## 7. Data persistence

Data lives in the named Docker volume `trading_pgdata`, independent of
container lifecycle. `docker compose down` and container recreation preserve
it; only `down -v` or `docker volume rm trading_pgdata` destroys it.

```bash
docker volume ls | grep trading_pgdata
```

---

## 8. Local development without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-core.txt
pip install pytest pytest-asyncio

export DATABASE_URL="sqlite:///./trading.db"
alembic upgrade head
uvicorn api.main:app --reload
```

`requirements-core.txt` deliberately excludes MetaTrader5, so it installs
cleanly on Linux. The Windows-only packages live in `requirements-mt5.txt`.

Run the tests with an explicit path — bare `pytest` collects `test_mt5.py` at
the repo root, which opens a live MT5 terminal at import:

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `POSTGRES_PASSWORD ... is missing` | `.env` unset or empty | `cp .env.example .env`, set the password |
| App restarting, log shows `ImportError: greenlet` | Stale image built before greenlet was pinned | `docker compose up -d --build` |
| App `503` on `/api/ready` | Database not reachable | `docker compose logs db` |
| `connection refused` on 5432 | Database still starting | Wait for `(healthy)`, then retry |
| Migration fails on an existing DB | Schema drift | `alembic current`, then `alembic upgrade head` |
| Port 8000 already in use | Another process | `APP_PORT=8080` in `.env` |

View the last startup lines:

```bash
docker compose logs app | tail -30
```

---

## 10. What is NOT in this stack

Honest scope statement — do not assume these exist:

- **No MT5 access.** The bridge is a separate, not-yet-built component.
- **No authentication** on the API.
- **No Redis**, no caching layer, no background worker.
- **No trading loop in the container.** `core/` has engines and models, but no
  scheduler wires them into a continuous run yet.
- **`api/main.py` endpoints are partly stubs.** `/api/status` reports hardcoded
  values (`is_connected: false`, `open_positions: 0`); `/api/emergency-stop`
  does not stop anything.