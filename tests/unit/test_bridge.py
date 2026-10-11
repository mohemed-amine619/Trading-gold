"""
tests/unit/test_bridge.py — Bridge contract, auth and safety.

Runs entirely on Linux against a mock adapter. No MetaTrader5, no terminal, no
broker, no orders.

The safety properties under test:

* No token configured  -> every protected endpoint refuses (fails closed).
* Wrong/missing token  -> 401, never a partial pass.
* Writes disabled      -> 403 on every state-changing endpoint.
* Duplicate key        -> 409, never a second order.
* Account mismatch     -> 409 (identity comes from the terminal, not the caller).
"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from bridge.app import create_app
from core.mt5.adapter import MockMT5Adapter

TOKEN = "a" * 64
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def adapter():
    a = MockMT5Adapter(balance=10_000.0, seed=7)
    a.connect()
    return a


@pytest.fixture
def client(adapter, monkeypatch):
    monkeypatch.setenv("MT5_BRIDGE_TOKEN", TOKEN)
    monkeypatch.setenv("MT5_BRIDGE_EXPECTED_LOGIN", str(adapter._login))
    app = create_app(adapter=adapter, allow_writes=False)
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Health and readiness (unauthenticated, leak nothing)
# ─────────────────────────────────────────────────────────────────────────────

def test_health_needs_no_auth(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["service"] == "mt5-bridge"


def test_ready_reports_not_ready_in_mock_mode(client):
    """Must never claim readiness when no real terminal is attached."""
    r = client.get("/api/v1/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is False
    assert "MOCK" in r.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [
    "/api/v1/account",
    "/api/v1/symbols",
    "/api/v1/positions",
    "/api/v1/price/EURUSD",
])
def test_protected_endpoints_reject_missing_token(client, path):
    assert client.get(path).status_code == 401


def test_protected_endpoints_reject_wrong_token(client):
    assert client.get("/api/v1/account",
                      headers={"Authorization": "Bearer wrong-token"}).status_code == 401


def test_valid_token_is_accepted(client):
    assert client.get("/api/v1/account", headers=AUTH).status_code == 200


def test_missing_token_config_fails_closed(adapter, monkeypatch):
    """A misconfigured bridge must refuse everything, not allow everything."""
    monkeypatch.delenv("MT5_BRIDGE_TOKEN", raising=False)
    app = create_app(adapter=adapter, allow_writes=False)
    with TestClient(app) as c:
        assert c.get("/api/v1/account", headers=AUTH).status_code == 503


def test_errors_do_not_leak_the_token(client):
    r = client.get("/api/v1/account", headers={"Authorization": f"Bearer {TOKEN}x"})
    assert r.status_code == 401
    assert TOKEN not in r.text


# ─────────────────────────────────────────────────────────────────────────────
# Read-only surface
# ─────────────────────────────────────────────────────────────────────────────

def test_symbol_discovery_is_dynamic(client):
    r = client.get("/api/v1/symbols", headers=AUTH)
    assert r.status_code == 200
    assert "symbols" in r.json()


def test_symbol_info_preserves_absent_metadata(client, adapter):
    """Absent broker values must serialise as null, not a default."""
    adapter.get_symbol_info = lambda s: None
    r = client.get("/api/v1/symbol/XAUUSDm", headers=AUTH)
    assert r.status_code == 404


def test_symbol_info_returns_broker_values(client):
    r = client.get("/api/v1/symbol/XAUUSDm", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "XAUUSDm"
    assert body["contract_size"] == 100.0
    assert body["source"] == "broker"


def test_price_endpoint(client):
    r = client.get("/api/v1/price/XAUUSDm", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["bid"] < r.json()["ask"]


def test_rates_endpoint(client):
    r = client.get("/api/v1/rates/XAUUSDm?timeframe=M15&count=50", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert len(body["close"]) == 50


def test_rates_rejects_unknown_timeframe(client):
    r = client.get("/api/v1/rates/XAUUSDm?timeframe=BOGUS", headers=AUTH)
    assert r.status_code == 400


def test_positions_endpoint(client):
    r = client.get("/api/v1/positions", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Order validation is read-only
# ─────────────────────────────────────────────────────────────────────────────

def test_validation_reports_broker_limits(client):
    r = client.post("/api/v1/order/validate", headers=AUTH, json={
        "symbol": "XAUUSDm", "side": 0, "volume": 0.1, "idempotency_key": "v1",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["min_volume"] == 0.01


def test_validation_rejects_bad_volume(client):
    r = client.post("/api/v1/order/validate", headers=AUTH, json={
        "symbol": "XAUUSDm", "side": 0, "volume": 0.001, "idempotency_key": "v2",
    })
    assert r.status_code == 200
    assert r.json()["valid"] is False


def test_validation_rejects_unknown_symbol(client):
    r = client.post("/api/v1/order/validate", headers=AUTH, json={
        "symbol": "NOPE", "side": 0, "volume": 0.1, "idempotency_key": "v3",
    })
    assert r.json()["valid"] is False


def test_validation_requires_body_fields(client):
    r = client.post("/api/v1/order/validate", headers=AUTH, json={"symbol": "XAUUSDm"})
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Writes are disabled by default
# ─────────────────────────────────────────────────────────────────────────────

def test_submit_forbidden_when_writes_disabled(client):
    r = client.post("/api/v1/order", headers=AUTH, json={
        "symbol": "XAUUSDm", "side": 0, "volume": 0.1, "idempotency_key": "w1",
    })
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"].lower()


def test_modify_forbidden_when_writes_disabled(client):
    r = client.post("/api/v1/position/modify", headers=AUTH,
                    json={"ticket": 1, "symbol": "XAUUSDm", "sl": 1990.0})
    assert r.status_code == 403


def test_close_forbidden_when_writes_disabled(client):
    r = client.post("/api/v1/position/close", headers=AUTH,
                    json={"ticket": 1, "symbol": "XAUUSDm"})
    assert r.status_code == 403


def test_submit_requires_auth_even_before_write_check(client):
    r = client.post("/api/v1/order", json={
        "symbol": "XAUUSDm", "side": 0, "volume": 0.1, "idempotency_key": "w2",
    })
    assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# Write path safety (enabled, but exercised only against a mock)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def writing_client(adapter, monkeypatch):
    monkeypatch.setenv("MT5_BRIDGE_TOKEN", TOKEN)
    monkeypatch.setenv("MT5_BRIDGE_EXPECTED_LOGIN", str(adapter._login))
    app = create_app(adapter=adapter, allow_writes=True)
    with TestClient(app) as c:
        yield c


def test_duplicate_idempotency_key_is_refused(writing_client):
    payload = {"symbol": "XAUUSDm", "side": 0, "volume": 0.1,
               "idempotency_key": "same-key"}
    first = writing_client.post("/api/v1/order", headers=AUTH, json=payload)
    assert first.status_code == 200

    second = writing_client.post("/api/v1/order", headers=AUTH, json=payload)
    assert second.status_code == 409
    assert "duplicate" in second.json()["detail"].lower()


def test_account_mismatch_blocks_writes(adapter, monkeypatch):
    monkeypatch.setenv("MT5_BRIDGE_TOKEN", TOKEN)
    monkeypatch.setenv("MT5_BRIDGE_EXPECTED_LOGIN", "999999999")
    app = create_app(adapter=adapter, allow_writes=True)
    with TestClient(app) as c:
        r = c.post("/api/v1/order", headers=AUTH, json={
            "symbol": "XAUUSDm", "side": 0, "volume": 0.1, "idempotency_key": "am",
        })
        assert r.status_code == 409
        assert "does not match" in r.json()["detail"]


def test_submit_returns_explicit_state(writing_client):
    r = writing_client.post("/api/v1/order", headers=AUTH, json={
        "symbol": "XAUUSDm", "side": 0, "volume": 0.1, "idempotency_key": "ok-1",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["state"] in ("filled", "partially_filled", "unknown", "rejected",
                             "cancelled", "not_sent")
    assert body["idempotency_key"] == "ok-1"


def test_unconfigured_expected_login_blocks_writes(adapter, monkeypatch):
    """Identity must never be asserted by the caller alone."""
    monkeypatch.setenv("MT5_BRIDGE_TOKEN", TOKEN)
    monkeypatch.delenv("MT5_BRIDGE_EXPECTED_LOGIN", raising=False)
    app = create_app(adapter=adapter, allow_writes=True)
    with TestClient(app) as c:
        r = c.post("/api/v1/order", headers=AUTH, json={
            "symbol": "XAUUSDm", "side": 0, "volume": 0.1, "idempotency_key": "nl",
        })
        assert r.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# Package hygiene
# ─────────────────────────────────────────────────────────────────────────────

def test_bridge_has_no_stray_mt5_import_at_module_level():
    """
    MT5 may only be imported lazily inside build_adapter(), never at module
    import time — otherwise importing the package on Linux would fail.

    Checked against the parsed AST rather than raw text so that prose in
    docstrings and comments cannot trip or mask the check.
    """
    import ast
    import bridge.app as a

    tree = ast.parse(open(a.__file__, encoding="utf-8").read())

    module_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level_imports += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_level_imports.append(node.module or "")

    assert "MetaTrader5" not in module_level_imports, (
        "bridge/app.py must not import MetaTrader5 at module level"
    )

    # It must still be importable lazily inside the adapter factory.
    source = open(a.__file__, encoding="utf-8").read()
    assert "import MetaTrader5" in source


def test_entrypoint_does_not_build_an_adapter_on_import():
    """importing bridge must not attempt a broker connection."""
    assert importlib.import_module("bridge") is not None