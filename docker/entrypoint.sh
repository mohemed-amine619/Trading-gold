#!/bin/sh
# Container entrypoint: apply migrations, then serve.
#
# Migrations run here (not at import time) so schema changes are an explicit,
# ordered, auditable step. `alembic upgrade head` is idempotent — re-running
# on an already-migrated database is a no-op.
set -e

echo "[entrypoint] applying database migrations..."
# SQLAlchemy's async driver name is stripped by alembic/env.py, so the sync
# URL is used for migrations.
alembic upgrade head

echo "[entrypoint] starting application..."
# exec so uvicorn becomes PID 1 and receives SIGTERM directly, giving it a
# chance to drain in-flight requests before shutdown.
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --no-access-log