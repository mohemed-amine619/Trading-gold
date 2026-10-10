# ─────────────────────────────────────────────────────────────────────────────
# Ubuntu core image — application, API and database tooling.
#
# This image does NOT contain MetaTrader5 (Windows-only). The Linux core talks
# to the Windows MT5 bridge over HTTP instead; see requirements-core.txt and
# docs/DEPLOYMENT_UBUNTU.md.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libpq5 is required by psycopg (PostgreSQL client library).
# curl is used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer: copied first so `pip install` is cached until the
# requirements file itself changes.
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Application code.
COPY alembic.ini ./
COPY alembic/ ./alembic/
COPY core/ ./core/
COPY api/ ./api/
COPY tests/ ./tests/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Drop privileges: run as an unprivileged user owned by the app.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app \
    && chmod +x /usr/local/bin/entrypoint.sh
USER appuser

EXPOSE 8000

# Liveness only. Readiness (DB connectivity) is enforced by the compose
# healthcheck, which probes the app's /api/ready endpoint.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["/usr/local/bin/entrypoint.sh"]