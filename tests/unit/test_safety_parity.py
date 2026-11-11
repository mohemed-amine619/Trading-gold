"""
Safety parity: rules checked through run.py itself (the live path), not core/.

Gap 1 - POST /api/config/strategy wrote ANY key with ANY value into the live
        config (run.py, section == "strategy"): go live without confirmation,
        disable the demo guard, zero every drawdown cap. The dashboard's Strategy
        form only ever sends seven display keys, so the route accepts exactly those.
Gap 2 - _modify_sltp had no never-loosen check; its broker stops-level clamp
        moved an existing stop AWAY from price. (Live order flow: test only,
        fix awaits approval.)
Gap 3 - an unreadable risk-state file loaded as a blank state and silently
        cleared a total-drawdown halt (fail open). It now blocks new entries until
        a manual reset, keeps the bad file as evidence, and survives restarts.
"""

from __future__ import annotations

import json
import types

import pytest

from tests.unit.test_dashboard_settings_api import AUTH, run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import make_bot


# ── Gap 1: /api/config/strategy bypassed every check ──────────────────────────

def _client(run_module, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(run_module, "_update_env_file", lambda *a, **k: True)
    bot = run_module.StandaloneAIBot(dict(run_module.CONFIG))
    return bot, TestClient(run_module.create_app([bot]), headers=AUTH)


@pytest.mark.parametrize("payload", [
    {"dry_run": False},                              # go live, no confirm
    {"require_demo_account": False},                 # drop the demo guard
    {"max_total_dd": 0}, {"max_weekly_dd": 0},       # disable drawdown caps
    {"max_daily_dd": 0}, {"risk_per_trade": 1.0},    # 100% of equity per trade
    {"magic_number": 1},                             # own trades become "manual"
    {"news_builtin": False},
])
def test_strategy_section_cannot_write_arbitrary_keys(run_and_stub, monkeypatch, payload):
    run_module, _ = run_and_stub
    bot, client = _client(run_module, monkeypatch)
    before = {k: bot.cfg.get(k) for k in payload}
    res = client.post("/api/config/strategy", json=payload)
    assert res.status_code == 422
    assert {k: bot.cfg.get(k) for k in payload} == before
    assert {k: run_module.CONFIG.get(k) for k in payload} == before


def test_a_mixed_strategy_payload_applies_nothing(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    bot, client = _client(run_module, monkeypatch)
    res = client.post("/api/config/strategy", json={"trend_fast_ma": 12, "dry_run": False})
    assert res.status_code == 422 and "dry_run" in res.text
    assert bot.cfg.get("trend_fast_ma") != 12 and bot.cfg["dry_run"] is True


def test_the_dashboard_strategy_form_still_saves(run_and_stub, monkeypatch):
    run_module, _ = run_and_stub
    bot, client = _client(run_module, monkeypatch)
    form = {"trend_enabled": True, "trend_fast_ma": 12, "trend_slow_ma": 48, "trend_atr": 1.8,
            "trend_min_score": 58, "mr_enabled": False, "mr_rsi": 14}
    res = client.post("/api/config/strategy", json=form)
    assert res.status_code == 200
    assert bot.cfg["trend_fast_ma"] == 12


# ── Gap 2: _modify_sltp could loosen an existing stop ─────────────────────────

@pytest.mark.xfail(strict=True, reason="GAP 2, open: _modify_sltp loosens a stop inside the "
                   "broker stops level (BUY 1999.80 -> 1999.50). Live order flow: fix awaits "
                   "approval. strict=True makes the suite fail once it is fixed, so the "
                   "marker gets removed.")
@pytest.mark.parametrize("is_buy,existing_sl", [(True, 1999.80), (False, 2000.30)])
def test_modify_sltp_never_loosens_an_existing_stop(run_and_stub, is_buy, existing_sl):
    """bid 2000.00 / ask 2000.10, broker stops level 50 points (0.50). The stop is
    0.20 from price, inside the stops level; the clamp used to move it to
    price -/+ 0.50, i.e. further from price."""
    run_module, stub = run_and_stub
    mt5 = run_module.mt5
    sent = []
    mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL = 0, 1
    mt5.symbol_info = lambda s: types.SimpleNamespace(digits=2, trade_stops_level=50, point=0.01)
    mt5.symbol_info_tick = lambda s: types.SimpleNamespace(bid=2000.00, ask=2000.10)
    mt5.positions_get = lambda *a, **k: []
    mt5.order_send = lambda req: sent.append(req) or types.SimpleNamespace(retcode=10009)
    cfg = dict(run_module.CONFIG, dry_run=False)
    bot = run_module.StandaloneAIBot(cfg)
    pos = types.SimpleNamespace(ticket=7, symbol="XAUUSDm", type=0 if is_buy else 1,
                                sl=existing_sl, tp=0.0, price_open=2000.0, magic=0)
    bot._modify_sltp(pos, existing_sl, 2010.0 if is_buy else 1990.0)
    for req in sent:
        if is_buy:
            assert req["sl"] >= existing_sl - 1e-9, req
        else:
            assert req["sl"] <= existing_sl + 1e-9, req


# ── Gap 3: an unreadable risk-state file cleared a halt ───────────────────────

def _halted_then_corrupted(run_module, tmp_path, garbage):
    path = str(tmp_path / "risk_XAUUSDm.json")
    first = make_bot(run_module, path)
    first._halt_reason = "total DD 31.0% >= 30.0% - manual reset required"
    first._persist_risk_state()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(garbage)
    second = make_bot(run_module, path)
    second._restore_risk_state(400.0)
    return second, path


@pytest.mark.parametrize("garbage", ["", "{not json", "[1, 2, 3]", '{"daily_start_eq'])
def test_an_unreadable_state_file_blocks_entries_instead_of_clearing_the_halt(
        run_and_stub, tmp_path, garbage):
    run_module, _ = run_and_stub
    bot, _ = _halted_then_corrupted(run_module, tmp_path, garbage)
    allowed, mult, reason = bot._entry_gate(400.0)
    assert allowed is False and mult == 0.0
    assert "risk state" in reason


def test_the_block_survives_a_second_restart_and_keeps_the_evidence(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot, path = _halted_then_corrupted(run_module, tmp_path, "{not json")
    bot._persist_risk_state()                          # the bot writes a fresh state
    again = make_bot(run_module, path)
    again._restore_risk_state(400.0)
    assert again._entry_gate(400.0)[0] is False        # still blocked
    assert list(tmp_path.glob("risk_XAUUSDm.json.corrupt*"))   # original bytes kept


def test_the_block_does_not_skip_position_management(run_and_stub, tmp_path):
    """Entries only: the hard-drawdown path (which closes positions and skips the
    tick) must not fire just because the state file was unreadable."""
    run_module, _ = run_and_stub
    bot, _ = _halted_then_corrupted(run_module, tmp_path, "{not json")
    assert bot._hard_drawdown_reason(400.0) is None


def test_a_manual_reset_clears_the_block(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot, path = _halted_then_corrupted(run_module, tmp_path, "{not json")
    bot.reset_risk_halt("operator checked the state file")
    assert bot._entry_gate(400.0)[0] is True
    again = make_bot(run_module, path)
    again._restore_risk_state(400.0)
    assert again._entry_gate(400.0)[0] is True


def test_a_missing_file_on_first_start_is_not_a_block(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    bot = make_bot(run_module, str(tmp_path / "never_written.json"))
    bot._restore_risk_state(400.0)
    assert bot._entry_gate(400.0)[0] is True


def test_a_valid_file_still_restores_the_halt(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    path = str(tmp_path / "risk_XAUUSDm.json")
    first = make_bot(run_module, path)
    first._halt_reason = "total DD 31.0% >= 30.0% - manual reset required"
    first._persist_risk_state()
    assert json.load(open(path, encoding="utf-8"))["halted_reason"]
    second = make_bot(run_module, path)
    second._restore_risk_state(400.0)
    assert second._halt_reason and second._entry_gate(400.0)[0] is False
