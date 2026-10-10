import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app
from core.settings import get_settings
from core.database.engine import init_async_db

@pytest.mark.asyncio
async def test_config_crud():
    settings = get_settings()
    await init_async_db("sqlite+aiosqlite:///:memory:")
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Initial should be empty
        response = await ac.get("/api/config/")
        assert response.status_code == 200
        
        # Validation test: risk > daily loss
        payload_invalid = {"max_daily_loss": 3.0, "risk_per_trade": 5.0, "max_dd": 10.0}
        response = await ac.post("/api/config/risk", json=payload_invalid)
        assert response.status_code == 400
        assert "cannot exceed" in response.json()["detail"]
        
        # Valid config
        payload = {"max_daily_loss": 3.0, "risk_per_trade": 1.5, "max_dd": 10.0}
        response = await ac.post("/api/config/risk", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Retrieve risk config
        response = await ac.get("/api/config/risk")
        assert response.status_code == 200
        assert response.json() == payload
