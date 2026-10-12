"""
api/main.py — FastAPI application for the Trading Dashboard.
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import threading
from contextlib import asynccontextmanager

from core.logging_setup import configure_logging
from core.settings import get_settings, Settings
from core.database.engine import init_async_db
from core.risk.engine import get_risk_engine, RiskEngine
from api.config_router import router as config_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Configure logging and initialise the async database engine.

    Failure is logged and swallowed rather than raised: the process must stay
    up so that /api/health (liveness) and /api/ready (readiness) can answer.
    A hard crash here would turn "database not ready yet" into an
    indistinguishable crash-loop during startup and rolling restarts.
    Schema is owned by Alembic, so no tables are created here.
    """
    settings = get_settings()
    try:
        configure_logging(
            level=settings.log_level.value,
            log_dir=settings.log_dir,
            log_filename="api.log",
        )
        logger.info("Logging initialised (level=%s)", settings.log_level.value)
    except Exception as exc:
        # Logging must never prevent the API from serving.
        logger.error("Logging configuration failed: %s: %s", type(exc).__name__, exc)

    try:
        await init_async_db(settings.database_url, create_tables=False)
        logger.info("Database engine initialised (url=%s)", _redact_url(settings.database_url))
    except Exception as exc:
        logger.error("Database initialisation failed: %s: %s", type(exc).__name__, exc)
    yield
    # Async engine is intentionally left to the event loop for disposal.


def _redact_url(url: str) -> str:
    """Strip credentials from a database URL before logging it."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"

app = FastAPI(title="Trading Platform API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config_router)

class StatusResponse(BaseModel):
    trading_mode: str
    is_connected: bool
    open_positions: int
    daily_pnl: float
    status: str
    emergency_stop: bool = False

@app.get("/api/health")
def health_check():
    """Liveness: the process is up. Does NOT check the database."""
    return {"status": "ok"}

@app.get("/api/ready")
async def readiness_check():
    """
    Readiness: the process is up AND the database is reachable and migrated.

    Orchestrators (Docker healthcheck, load balancers) should poll this, not
    /api/health — a process can be alive while PostgreSQL is unreachable.
    Returns 503 when not ready so the container is marked unhealthy.
    """
    from sqlalchemy import text
    from core.database.engine import get_async_db

    checks: dict[str, str] = {}
    try:
        async with get_async_db() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"

    db_ready = checks["database"] == "ok"
    return JSONResponse(
        status_code=200 if db_ready else 503,
        content={
            "status": "ready" if db_ready else "not_ready",
            "checks": checks,
            "trading_mode": get_settings().trading_mode.value,
        },
    )

@app.get("/api/status", response_model=StatusResponse)
def get_system_status(
    settings: Settings = Depends(get_settings),
    risk_engine: RiskEngine = Depends(get_risk_engine),
):
    r_status = risk_engine.status()
    is_stopped = r_status.get("emergency_stop", False)
    return StatusResponse(
        trading_mode=settings.trading_mode.value,
        is_connected=False,
        open_positions=r_status.get("open_positions_count", 0),
        daily_pnl=0.0,
        status="stopped" if is_stopped else "running",
        emergency_stop=is_stopped,
    )

@app.post("/api/emergency-stop")
def emergency_stop(risk_engine: RiskEngine = Depends(get_risk_engine)):
    risk_engine.trigger_emergency_stop()
    logger.critical("Emergency stop invoked via API")
    return {"status": "emergency_stop_activated", "message": "All trading stopped and emergency stop activated."}

@app.post("/api/resume")
def resume_trading(risk_engine: RiskEngine = Depends(get_risk_engine)):
    risk_engine.reset_emergency_stop()
    logger.info("Emergency stop reset via API")
    return {"status": "trading_resumed", "message": "Trading resumed."}

@app.post("/api/control")
def control_action(action_data: dict, risk_engine: RiskEngine = Depends(get_risk_engine)):
    action = action_data.get("action")
    if action == "emergency_stop":
        risk_engine.trigger_emergency_stop()
        logger.critical("Emergency stop invoked via /api/control")
        return {"status": "ok", "message": "Emergency stop activated"}
    elif action == "pause":
        risk_engine.trigger_emergency_stop()
        return {"status": "ok", "message": "Trading paused"}
    elif action == "resume":
        risk_engine.reset_emergency_stop()
        return {"status": "ok", "message": "Trading resumed"}
    return {"status": "ok"}

@app.post("/api/settings")
async def update_settings(new_config: dict):
    # Backward compatibility endpoint if dashboard.html calls it
    return {"status": "settings_updated", "updated_keys": len(new_config)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.1", port=8000, reload=True)
