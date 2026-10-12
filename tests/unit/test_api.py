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
    from core.risk.engine import get_risk_engine
    engine = get_risk_engine()
    engine.reset_emergency_stop()
    assert engine.emergency_stopped is False

    response = client.post("/api/emergency-stop")
    assert response.status_code == 200
    assert response.json()["status"] == "emergency_stop_activated"
    assert engine.emergency_stopped is True

    # Check that status reports stopped
    status_resp = client.get("/api/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "stopped"
    assert status_resp.json()["emergency_stop"] is True

    # Resume trading
    resume_resp = client.post("/api/resume")
    assert resume_resp.status_code == 200
    assert resume_resp.json()["status"] == "trading_resumed"
    assert engine.emergency_stopped is False

    # Status back to running
    status_resp2 = client.get("/api/status")
    assert status_resp2.status_code == 200
    assert status_resp2.json()["status"] == "running"
    assert status_resp2.json()["emergency_stop"] is False


def test_control_endpoint():
    from core.risk.engine import get_risk_engine
    engine = get_risk_engine()
    engine.reset_emergency_stop()

    # Pause via control
    resp_pause = client.post("/api/control", json={"action": "pause"})
    assert resp_pause.status_code == 200
    assert engine.emergency_stopped is True

    # Resume via control
    resp_resume = client.post("/api/control", json={"action": "resume"})
    assert resp_resume.status_code == 200
    assert engine.emergency_stopped is False

    # Emergency stop via control
    resp_stop = client.post("/api/control", json={"action": "emergency_stop"})
    assert resp_stop.status_code == 200
    assert engine.emergency_stopped is True

    # Clean up
    engine.reset_emergency_stop()


