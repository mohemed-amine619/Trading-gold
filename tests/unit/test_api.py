from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "trading_mode" in data
    assert "is_connected" in data

def test_emergency_stop():
    response = client.post("/api/emergency-stop")
    assert response.status_code == 200
    assert response.json()["status"] == "emergency_stop_activated"
