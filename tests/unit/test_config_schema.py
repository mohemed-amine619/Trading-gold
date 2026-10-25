"""
tests/unit/test_config_schema.py — the declarative parameter registry.

The point of these tests is not coverage for its own sake. The registry is the
only thing standing between a browser and the bot's risk settings, so the
guarantees it makes have to be enforced, not documented:

* every declared default must satisfy its own validator,
* an unknown key must be refused rather than silently ignored,
* a boolean must never sneak through as the number 1,
* cross-field invariants must reject a payload that is individually valid.
"""

from __future__ import annotations

import pytest

from core.config.schema import (
    GROUP_ORDER,
    PARAMS,
    TIMEFRAME_LABELS,
    coerce_value,
    schema_payload,
    validate_payload,
)
from core.config.audit import recent_changes, record_changes
from core.risk.state import RiskState, load_state, save_state, state_path


# ─────────────────────────────────────────────────────────────────────────────
# Registry shape
# ─────────────────────────────────────────────────────────────────────────────

def test_every_parameter_is_in_a_known_group():
    for key, spec in PARAMS.items():
        assert spec.group in GROUP_ORDER, f"{key} has unknown group {spec.group!r}"


def test_groups_are_not_empty_and_ordered():
    payload = schema_payload()
    names = [g["name"] for g in payload["groups"]]
    assert names == [n for n in GROUP_ORDER if n in names]
    for group in payload["groups"]:
        assert group["params"]


def test_every_default_passes_its_own_validator():
    """A default that its own validator rejects is a latent startup bug."""
    for key, spec in PARAMS.items():
        values, errors = validate_payload({key: spec.default})
        assert not errors, f"default for {key} is invalid: {errors}"
        assert values[key] == spec.default


def test_no_duplicate_keys_survive_import():
    # PARAMS is a dict so duplicates would silently overwrite; the module also
    # raises at import, but assert the count is what the schema reports.
    assert schema_payload()["count"] == len(PARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# Coercion
# ─────────────────────────────────────────────────────────────────────────────

def test_boolean_is_not_accepted_as_a_number():
    spec = PARAMS["risk_per_trade"]
    with pytest.raises(ValueError):
        coerce_value(spec, True)


def test_numeric_string_is_accepted():
    spec = PARAMS["risk_per_trade"]
    assert coerce_value(spec, "0.02") == 0.02


def test_out_of_range_is_rejected_not_clamped():
    """Silently clamping hides a typo; the operator must see the rejection."""
    values, errors = validate_payload({"risk_per_trade": 0.99})
    assert values == {}
    assert any("above the maximum" in e for e in errors)


def test_int_kind_rejects_fractional_values():
    values, errors = validate_payload({"max_open_pos": 3.5})
    assert values == {}
    assert any("integer" in e for e in errors)


def test_nan_is_rejected():
    values, errors = validate_payload({"risk_per_trade": float("nan")})
    assert values == {}
    assert any("finite" in e for e in errors)


def test_boolean_accepts_form_strings():
    spec = PARAMS["fib_enabled"]
    assert coerce_value(spec, "true") is True
    assert coerce_value(spec, "off") is False
    with pytest.raises(ValueError):
        coerce_value(spec, "maybe")


def test_enum_rejects_unknown_choice():
    values, errors = validate_payload({"llm_provider": "ollama"})
    assert values == {}
    assert any("not one of" in e for e in errors)


def test_enum_is_case_insensitive():
    values, errors = validate_payload({"llm_provider": "EMBEDDED"})
    assert not errors
    assert values["llm_provider"] == "embedded"


# ─────────────────────────────────────────────────────────────────────────────
# Timeframes
# ─────────────────────────────────────────────────────────────────────────────

def test_timeframe_accepts_label_and_integer():
    spec = PARAMS["timeframe"]
    assert coerce_value(spec, "M15") == TIMEFRAME_LABELS["M15"]
    assert coerce_value(spec, 15) == 15


def test_timeframe_rejects_unknown_value():
    spec = PARAMS["timeframe"]
    with pytest.raises(ValueError):
        coerce_value(spec, 7)
    with pytest.raises(ValueError):
        coerce_value(spec, "M7")


def test_schema_renders_timeframe_as_label():
    payload = schema_payload({"timeframe": 15})
    entry = next(
        p for g in payload["groups"] for p in g["params"] if p["key"] == "timeframe"
    )
    assert entry["value"] == "M15"
    assert "M15" in entry["choices"]


# ─────────────────────────────────────────────────────────────────────────────
# Payload-level rules
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_parameter_is_refused():
    values, errors = validate_payload({"definitely_not_real": 1})
    assert values == {}
    assert any("unknown parameter" in e for e in errors)


def test_control_flags_are_not_treated_as_parameters():
    values, errors = validate_payload({"confirm": True, "_source": "test"})
    assert values == {}
    assert errors == []


def test_risk_must_be_below_daily_drawdown_cap():
    values, errors = validate_payload({"risk_per_trade": 0.04, "max_daily_dd": 0.03})
    assert any("must be smaller" in e for e in errors)


def test_risk_scale_bounds_are_ordered():
    _, errors = validate_payload({"min_risk_scale": 0.9, "max_risk_scale": 0.5})
    assert any("must not exceed" in e for e in errors)


def test_fib_zone_bounds_are_ordered():
    _, errors = validate_payload({"fib_zone_low": 0.8, "fib_zone_high": 0.6})
    assert any("must be smaller" in e for e in errors)


def test_stop_bounds_are_ordered():
    _, errors = validate_payload(
        {"location_stop_min_atr": 4.0, "location_stop_max_atr": 2.0}
    )
    assert any("must not exceed" in e for e in errors)


def test_valid_payload_passes_cleanly():
    payload = {
        "risk_per_trade": 0.0075,
        "max_daily_dd": 0.04,
        "max_open_pos": 4,
        "fib_enabled": True,
        "fib_weight": 12.0,
        "timeframe": "M5",
    }
    values, errors = validate_payload(payload)
    assert errors == []
    assert values["risk_per_trade"] == 0.0075
    assert values["max_open_pos"] == 4
    assert values["timeframe"] == 5


def test_dangerous_parameters_are_declared():
    """Paper -> live must be flagged, or the API cannot demand confirmation."""
    assert PARAMS["dry_run"].dangerous is True
    assert PARAMS["dry_run"].safe_live is False


def test_percent_parameters_are_bounded_fractions():
    for key, spec in PARAMS.items():
        if spec.percent:
            assert spec.kind == "float", key
            assert spec.min is not None and spec.max is not None, key
            assert spec.max <= 1.0, key


# ─────────────────────────────────────────────────────────────────────────────
# Audit log
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_round_trip(tmp_path):
    db = str(tmp_path / "audit.db")
    record_changes(db, "test", {"risk_per_trade": (0.01, 0.02)})

    rows = recent_changes(db, limit=10)
    assert len(rows) == 1
    assert rows[0]["key"] == "risk_per_trade"
    assert rows[0]["old"] == 0.01
    assert rows[0]["new"] == 0.02
    assert rows[0]["source"] == "test"


def test_audit_is_newest_first(tmp_path):
    db = str(tmp_path / "audit.db")
    record_changes(db, "a", {"max_open_pos": (3, 4)})
    record_changes(db, "b", {"max_open_pos": (4, 5)})

    rows = recent_changes(db, limit=10)
    assert [r["source"] for r in rows] == ["b", "a"]


def test_audit_read_on_missing_db_is_empty(tmp_path):
    rows = recent_changes(str(tmp_path / "does-not-exist.db"), limit=10)
    assert rows == []


def test_audit_never_raises_on_bad_path():
    # Unwritable path: must be swallowed, not propagated into a settings write.
    record_changes("/proc/definitely/not/writable.db", "test", {"x": (1, 2)})


# ─────────────────────────────────────────────────────────────────────────────
# Persisted risk state
# ─────────────────────────────────────────────────────────────────────────────

def test_risk_state_round_trip(tmp_path):
    path = state_path(str(tmp_path), "XAUUSDm")
    state = RiskState(daily_start_equity=10_000.0, trades_today=3, peak_equity=11_000.0)
    assert save_state(path, state) is True

    loaded = load_state(path)
    assert loaded.daily_start_equity == 10_000.0
    assert loaded.trades_today == 3
    assert loaded.peak_equity == 11_000.0


def test_risk_state_missing_file_is_fresh_state(tmp_path):
    loaded = load_state(state_path(str(tmp_path), "EURUSD"))
    assert loaded == RiskState()


def test_risk_state_corrupt_file_is_fresh_state(tmp_path):
    path = state_path(str(tmp_path), "EURUSD")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    assert load_state(path) == RiskState()


def test_risk_state_ignores_unknown_fields(tmp_path):
    path = state_path(str(tmp_path), "EURUSD")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"trades_today": 2, "from_the_future": 42}')
    assert load_state(path).trades_today == 2


def test_state_path_sanitises_the_symbol(tmp_path):
    path = state_path(str(tmp_path), "XAU/USD m")
    assert "/" not in path.rsplit("/", 1)[-1]
    assert " " not in path.rsplit("/", 1)[-1]


def test_save_state_leaves_no_temp_files(tmp_path):
    path = state_path(str(tmp_path), "XAUUSDm")
    save_state(path, RiskState(trades_today=1))
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_roll_day_resets_daily_start_and_flag():
    state = RiskState(daily_date="2020-01-01", daily_start_equity=1000.0,
                      dd_positions_closed=True)
    state.roll_day(2000.0)
    assert state.daily_start_equity == 2000.0
    assert state.dd_positions_closed is False
    assert state.daily_date is not None


def test_in_cooldown_is_false_without_a_deadline():
    assert RiskState().in_cooldown() is False


def test_in_cooldown_is_true_in_the_future():
    from datetime import datetime, timedelta
    until = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
    assert RiskState(cooldown_until=until).in_cooldown() is True


def test_in_cooldown_expires():
    from datetime import datetime, timedelta
    past = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
    assert RiskState(cooldown_until=past).in_cooldown() is False


def test_in_cooldown_tolerates_garbage():
    assert RiskState(cooldown_until="not-a-date").in_cooldown() is False
