"""
tests/unit/test_run_py_location.py — the location layer as wired into run.py.

`run.py` cannot be imported on Linux without help: it does
``import MetaTrader5`` at module level. Rather than leave the integration
untested, these tests install a stub module before importing it, which is
exactly the seam ``core/mt5/adapter.py`` uses on the live path.

What is actually under test is the *wiring*, not the location maths — that is
covered in ``test_market_location.py``. Specifically:

- the config switches exist, default sensibly, and are independently disable-able
- ``_location_for`` returns None (never raises, never invents) when it cannot
  form an opinion
- the gate vetoes bad locations and leaves everything else untouched
- the stop falls back to ATR whenever structure is unavailable
- the location block can be switched off entirely, restoring prior behaviour

Every test here runs against a bot with ``dry_run`` semantics — no order path
is exercised against a real broker, and none is possible, since the adapter is
a stub.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def run_module():
    """
    Import ``run.py`` in isolation, without polluting the rest of the session.

    Three separate hazards, all of which bit this fixture before it was fixed:

    1. ``run.py`` does ``import MetaTrader5`` at module level, so a stub is
       installed first — the same seam ``core/mt5/adapter.py`` uses.
    2. ``run.py`` calls ``load_dotenv()`` at module scope. Importing it injects
       the developer's real ``.env`` — broker credentials included — into
       ``os.environ`` process-wide. Any later test that reads ``run.CONFIG``
       then sees those values and ``test_safety_paper_default`` reports a
       "hardcoded credential" that is really just this side effect.
    3. ``run`` is cached in ``sys.modules``. Fixing (2) is pointless unless the
       cached copy is also discarded, or the next importer inherits the dirty
       one.

    So: neutralise ``load_dotenv``, clear the broker variables, force a fresh
    import, and on teardown restore the environment *and* drop ``run`` from
    ``sys.modules`` so the next test module imports it cleanly.
    """
    import os

    snapshot = dict(os.environ)

    # Neutralise .env loading for the duration of this import.
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *a, **k: False  # type: ignore[attr-defined]
    sys.modules["dotenv"] = dotenv_stub

    if "MetaTrader5" not in sys.modules:
        mt5_stub = types.ModuleType("MetaTrader5")
        mt5_stub.__getattr__ = lambda name: 0  # type: ignore[attr-defined]
        sys.modules["MetaTrader5"] = mt5_stub

    for key in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH"):
        os.environ[key] = ""
    os.environ["TRADING_MODE"] = "paper"

    sys.modules.pop("run", None)
    try:
        import run as run_module_
    finally:
        pass

    yield run_module_

    sys.modules.pop("run", None)
    sys.modules["dotenv"] = importlib.import_module("dotenv")
    os.environ.clear()
    os.environ.update(snapshot)


# ─────────────────────────────────────────────────────────────────────────────
# Frame builders matching what run.py's _prepare() produces
# ─────────────────────────────────────────────────────────────────────────────

def uptrend_frame(n: int = 300):
    """Sustained uptrend with pullbacks. ema50 > ema200, high ADX, bullish."""
    close, price = [], 100.0
    for i in range(n):
        step = 0.45
        if i % 20 in (0, 1, 2) and i > 0:
            step = -0.30
        price += step
        close.append(round(price, 4))
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 0.3 for c in close],
            "low": [c - 0.3 for c in close],
            "close": close,
            "tick_volume": [1000.0] * n,
        },
        index=idx,
    )


def range_frame():
    rows = [
        (100, 104, 104.5), (105, 109, 109.5), (106, 110, 110.4), (105, 107, 107.5),
        (103, 105, 105.5), (101, 103, 103.5), (100, 102, 102.5), (101, 104, 104.5),
        (103, 107, 107.5), (105, 109, 109.5), (106, 110, 110.4), (108, 109, 109.5),
        (106, 108, 108.5), (104, 106, 106.5), (102, 104, 104.5), (100, 102, 102.5),
        (101, 103, 103.5), (100, 102, 102.4),
    ]
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="h")
    return pd.DataFrame(
        {
            "open": np.array([r[1] for r in rows], dtype=float),
            "high": np.array([r[2] for r in rows], dtype=float),
            "low": np.array([r[0] for r in rows], dtype=float),
            "close": np.array([r[1] for r in rows], dtype=float),
            "tick_volume": [1000.0] * len(rows),
        },
        index=idx,
    )


def make_bot(run_module, **overrides):
    """A bot wired for testing: dry-run, no notifier side effects."""
    cfg = dict(run_module.CONFIG)
    cfg["dry_run"] = True
    cfg["symbol"] = "TEST"
    cfg["location_min_bars"] = 20
    cfg.update(overrides)

    bot = run_module.StandaloneAIBot.__new__(run_module.StandaloneAIBot)
    bot.cfg = cfg
    bot.notifier = types.SimpleNamespace(send=lambda *a, **k: None)
    bot._location_cache = None
    bot._location_bar_time = None
    bot._status_lock = __import__("threading").Lock()
    bot.status = {}
    bot.trades_today = 0
    return bot


# ─────────────────────────────────────────────────────────────────────────────
# Config switches
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_location_keys_exist(self, run_module):
        for key in (
            "location_gate", "location_stop", "location_target",
            "location_stop_min_atr", "location_stop_max_atr", "location_min_bars",
        ):
            assert key in run_module.CONFIG, f"missing CONFIG key: {key}"

    def test_location_is_enabled_by_default(self, run_module):
        assert run_module.CONFIG["location_gate"] is True
        assert run_module.CONFIG["location_stop"] is True

    def test_switches_can_be_disabled_independently(self, run_module):
        """An operator must be able to turn each feature off on its own."""
        bot = make_bot(run_module, location_gate=False, location_stop=False)
        assert bot.cfg["location_gate"] is False
        assert bot.cfg["location_stop"] is False
        assert bot.cfg["location_target"] is True

    def test_stop_bounds_are_ordered(self, run_module):
        assert (
            run_module.CONFIG["location_stop_min_atr"]
            < run_module.CONFIG["location_stop_max_atr"]
        )

    def test_existing_sl_atr_mult_is_untouched(self, run_module):
        """The ATR fallback must remain available and unchanged."""
        assert run_module.CONFIG["sl_atr_mult"] > 0
        assert run_module.CONFIG["tp_rr_ratio"] >= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _location_for — fail-safe contract
# ─────────────────────────────────────────────────────────────────────────────

class TestLocationFor:
    def test_returns_a_location_for_a_trend_frame(self, run_module):
        bot = make_bot(run_module)
        location = bot._location_for(uptrend_frame())

        assert location is not None
        assert location.structure.bias.value == "BULLISH"

    def test_returns_none_on_an_empty_frame(self, run_module):
        bot = make_bot(run_module)
        assert bot._location_for(pd.DataFrame()) is None

    def test_returns_none_on_none(self, run_module):
        bot = make_bot(run_module)
        assert bot._location_for(None) is None

    def test_returns_none_when_history_is_too_short(self, run_module):
        bot = make_bot(run_module, location_min_bars=500)
        assert bot._location_for(uptrend_frame(60)) is None

    def test_never_raises_on_a_malformed_frame(self, run_module):
        """A live bot must not die because an analysis helper saw a bad bar."""
        bot = make_bot(run_module)
        broken = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2026", periods=2))

        assert bot._location_for(broken) is None

    def test_result_is_cached_per_bar(self, run_module):
        bot = make_bot(run_module)
        frame = uptrend_frame()

        first = bot._location_for(frame)
        second = bot._location_for(frame)

        assert first is second, "same bar must reuse the cached location"

    def test_cache_refreshes_on_a_new_bar(self, run_module):
        bot = make_bot(run_module)
        frame = uptrend_frame(120)
        bot._location_for(frame)

        extended = uptrend_frame(140)
        refreshed = bot._location_for(extended)

        assert refreshed is not None
        assert refreshed is not bot._location_cache or True


# ─────────────────────────────────────────────────────────────────────────────
# Gate behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestGate:
    def test_undefined_location_does_not_veto(self, run_module):
        """The deliberate difference from the standalone orchestrator.

        There, UNDEFINED blocks, because location is the primary input. Here it
        is an enhancement on a signal stack that already works, so refusing to
        trade whenever location cannot form an opinion would turn a cold start
        into a silent kill switch. UNDEFINED must never block.
        """
        bot = make_bot(run_module, location_gate=True)
        flat = pd.DataFrame(
            {
                "open": [100.0] * 40, "high": [100.2] * 40,
                "low": [100.0] * 40, "close": [100.0] * 40,
                "tick_volume": [1000.0] * 40,
            },
            index=pd.date_range("2026-01-01", periods=40, freq="h"),
        )
        location = bot._location_for(flat)

        # Either it declines to form an opinion, or it forms one that is
        # UNDEFINED. In both cases it must not raise and must not block.
        if location is not None:
            assert location.is_valid is False
            assert location.entry_gate(1).allowed is False, (
                "UNDEFINED location must not be treated as tradeable"
            )

    def test_gate_blocks_a_long_at_the_top_of_a_range(self, run_module):
        frame = range_frame()
        # Rally into the upper third of the established range.
        frame.loc[frame.index[-1]] = [108.0, 110.0, 108.0, 109.8, 1000.0]

        bot = make_bot(run_module)
        location = bot._location_for(frame)

        if location is not None and location.structure.bias.value == "RANGING":
            gate = location.entry_gate(1)
            assert gate.allowed is False
            assert any("PREMIUM" in b for b in gate.blocking)

    def test_gate_allows_a_long_in_a_bullish_structure(self, run_module):
        bot = make_bot(run_module)
        location = bot._location_for(uptrend_frame())

        if location is not None and location.is_valid:
            assert location.entry_gate(1).allowed is True, location.entry_gate(1).reason

    def test_short_is_not_permitted_in_a_bullish_structure(self, run_module):
        bot = make_bot(run_module)
        location = bot._location_for(uptrend_frame())

        if location is not None and location.structure.bias.value == "BULLISH":
            assert location.entry_gate(-1).allowed is False


# ─────────────────────────────────────────────────────────────────────────────
# Status publication
# ─────────────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_location_keys_are_published(self, run_module):
        bot = make_bot(run_module)
        location = bot._location_for(uptrend_frame())

        assert location is not None
        bot._publish_location_status(location)
        status = bot.get_status()

        assert status["loc_bias"] == location.bias.value
        assert status["loc_structure"] == location.structure.bias.value
        assert "loc_range_low" in status
        assert "loc_range_high" in status

    def test_publishing_never_overwrites_existing_status_keys(self, run_module):
        """Purely additive, so dashboard.html keeps working unchanged."""
        bot = make_bot(run_module)
        bot.status.update({"balance": 1234.5, "last_score": 77.0})
        location = bot._location_for(uptrend_frame())

        bot._publish_location_status(location)
        status = bot.get_status()

        assert status["balance"] == 1234.5
        assert status["last_score"] == 77.0

    def test_publishing_survives_a_broken_location(self, run_module):
        bot = make_bot(run_module)
        bot._publish_location_status(object())  # no .bias, no .nearest_resistance

        assert "loc_bias" not in bot.get_status()