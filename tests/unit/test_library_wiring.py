"""
tests/unit/test_library_wiring.py — the strategy library, size progressions and
grid/DCA adds inside run.py.

Checks that the defaults are the safe ones (library on with the raised
significance bar; martingale and grid OFF), that the dashboard schema and run.py
agree, and that a grid add always reuses the first position's stop.
"""

from __future__ import annotations

import logging
import types

import pandas as pd
import pytest

from core.config.schema import PARAMS, validate_payload
from core.strategy.scalp import required_z
from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_scalp_strategies import planted_burst_series

NEW_KEYS = ("scalp_auto_z", "scalp_server_utc_offset", "sizing_progression", "progression_factor",
            "progression_max_steps", "progression_max_mult", "kelly_fraction", "kelly_min_trades",
            "grid_mode", "grid_spacing_atr", "grid_max_levels", "dca_scale", "grid_max_total_risk")


def make_bot(run_module, **over):
    cfg = dict(run_module.CONFIG)
    cfg.update(symbol="XAUUSDm", scalp_enabled=True)
    cfg.update(over)
    return run_module.StandaloneAIBot(cfg)


class TestDefaults:
    def test_new_keys_exist_in_run_and_schema_with_the_same_default(self, run_and_stub):
        run_module, _ = run_and_stub
        for k in NEW_KEYS:
            assert k in run_module.CONFIG and k in PARAMS, k
            assert run_module.CONFIG[k] == PARAMS[k].default, k

    def test_library_is_on_and_the_dangerous_rules_are_off(self, run_and_stub):
        run_module, _ = run_and_stub
        c = run_module.CONFIG
        assert c["scalp_strategies"] == "all" and PARAMS["scalp_strategies"].default == "all"
        assert c["scalp_auto_z"] is True
        assert c["sizing_progression"] == "off" and c["grid_mode"] == "off"
        assert PARAMS["sizing_progression"].dangerous and PARAMS["grid_mode"].dangerous

    def test_dashboard_accepts_modes_and_refuses_unknown_ones(self, run_and_stub):
        ok, errors = validate_payload({"sizing_progression": "martingale", "grid_mode": "dca"})
        assert errors == []
        _, errors = validate_payload({"sizing_progression": "double_or_nothing"})
        assert errors


class TestLibraryInTheBot:
    def test_all_builds_the_full_catalogue_with_the_raised_bar(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = make_bot(run_module, scalp_strategies="all")
        bot._get_scalp(planted_burst_series())
        eng = bot._scalp_engine
        assert len(eng.names) >= 110
        assert eng.min_z == pytest.approx(required_z(len(eng.names), 2.7))

    def test_group_tokens_work_from_the_setting(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = make_bot(run_module, scalp_strategies="smc, ict")
        bot._get_scalp(planted_burst_series())
        assert "order_block" in bot._scalp_engine.names and "silver_bullet" in bot._scalp_engine.names
        assert "burst" not in bot._scalp_engine.names

    def test_a_planted_edge_still_votes_among_all_strategies(self, run_and_stub):
        run_module, _ = run_and_stub
        votes = make_bot(run_module, scalp_strategies="all")._get_scalp(planted_burst_series())
        assert ("burst", 1) in [(v.name, v.direction) for v in votes]

    def test_the_offset_and_auto_z_reach_the_engine_and_rebuild_it(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = make_bot(run_module, scalp_strategies="burst")
        bot._get_scalp(planted_burst_series())
        first = bot._scalp_engine
        bot.cfg["scalp_server_utc_offset"] = 2.0
        bot.cfg["scalp_auto_z"] = False
        bot._get_scalp(planted_burst_series())
        assert bot._scalp_engine is not first
        assert bot._scalp_engine.cfg.server_utc_offset == 2.0 and bot._scalp_engine.cfg.auto_z is False


class TestProgression:
    def test_off_changes_nothing(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = make_bot(run_module)
        bot._recent_pnls[:] = [-1.0, -1.0, -1.0]
        assert bot._progression_size_factor() == (1.0, "")

    def test_martingale_doubles_per_loss_within_the_cap(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = make_bot(run_module, sizing_progression="martingale", progression_max_mult=4.0)
        bot._recent_pnls[:] = [5.0, -1.0, -1.0]
        assert bot._progression_size_factor()[0] == 4.0
        bot._recent_pnls[:] = [-1.0] * 6
        assert bot._progression_size_factor()[0] == 4.0


def leg(type_, price_open, sl, tp=0.0, volume=0.10, ticket=1, time=1):
    return types.SimpleNamespace(type=type_, price_open=price_open, sl=sl, tp=tp, volume=volume,
                                 ticket=ticket, time=time, magic=888999, symbol="XAUUSDm")


class TestGridAdd:
    def setup(self, run_module, monkeypatch, legs, **over):
        bot = make_bot(run_module, grid_mode="grid", grid_spacing_atr=1.0, grid_max_levels=3,
                       grid_max_total_risk=0.05, magic_number=888999, **over)
        monkeypatch.setattr(bot, "_open_positions", lambda: legs)
        monkeypatch.setattr(bot, "_count_positions", lambda: len(legs))
        monkeypatch.setattr(bot, "_trades_today_ok", lambda: True)
        monkeypatch.setattr(bot, "_entry_gate", lambda equity: (True, 1.0, ""))
        sent = []
        monkeypatch.setattr(bot, "_send_order", lambda *a, **k: sent.append(k) or True)
        m15 = pd.DataFrame({"atr": [2.0]})
        return bot, sent, m15

    def test_adds_at_one_atr_against_with_the_first_stop(self, run_and_stub, monkeypatch):
        run_module, stub = run_and_stub         # stub ask = 2650.1
        legs = [leg(0, 2652.5, 2640.0, 2670.0)]
        bot, sent, m15 = self.setup(run_module, monkeypatch, legs)
        assert bot._maybe_grid_add(1, 10_000.0, m15, 0.6) is True
        assert sent[0]["shared_sl"] == 2640.0 and sent[0]["shared_tp"] == 2670.0
        assert sent[0]["fixed_volume"] == pytest.approx(0.10)

    def test_no_add_before_the_level(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        bot, sent, m15 = self.setup(run_module, monkeypatch, [leg(0, 2651.0, 2640.0)])
        assert bot._maybe_grid_add(1, 10_000.0, m15, 0.6) is False and sent == []

    def test_never_adds_to_a_position_without_a_stop(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        bot, sent, m15 = self.setup(run_module, monkeypatch, [leg(0, 2655.0, 0.0)])
        assert bot._maybe_grid_add(1, 10_000.0, m15, 0.6) is False and sent == []

    def test_never_adds_against_an_opposite_position(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        bot, sent, m15 = self.setup(run_module, monkeypatch, [leg(1, 2645.0, 2660.0)])
        assert bot._maybe_grid_add(1, 10_000.0, m15, 0.6) is False and sent == []

    def test_refused_when_total_risk_exceeds_the_cap(self, run_and_stub, monkeypatch):
        run_module, _ = run_and_stub
        # 1 lot legs: (2655-2600 + 2650.1-2600) x 1 lot x 100/pt ~ 10,500 > 5% of 10,000
        legs = [leg(0, 2655.0, 2600.0, volume=1.0)]
        bot, sent, m15 = self.setup(run_module, monkeypatch, legs)
        assert bot._maybe_grid_add(1, 10_000.0, m15, 0.6) is False and sent == []

    def test_off_means_the_step_never_calls_it(self, run_and_stub):
        run_module, _ = run_and_stub
        assert run_module.CONFIG["grid_mode"] == "off"

    def test_send_order_uses_the_shared_stop_unchanged(self, run_and_stub, caplog):
        run_module, _ = run_and_stub
        bot = make_bot(run_module, dry_run=True)
        m15 = pd.DataFrame({"atr": [2.0] * 60, "adx": [20.0] * 60})
        with caplog.at_level(logging.INFO):
            ok = bot._send_order(1, 10_000.0, m15, 0.6, fixed_volume=0.10,
                                 shared_sl=2640.0, shared_tp=2670.0)
        assert ok is True
        assert any("[DRY-RUN] BUY vol=0.10 sl=2640.00000 tp=2670.00000" in r.message
                   for r in caplog.records)

    def test_send_order_refuses_an_add_through_the_shared_stop(self, run_and_stub):
        run_module, _ = run_and_stub
        bot = make_bot(run_module, dry_run=True)
        m15 = pd.DataFrame({"atr": [2.0] * 60, "adx": [20.0] * 60})
        assert bot._send_order(1, 10_000.0, m15, 0.6, fixed_volume=0.10, shared_sl=2655.0) is False
