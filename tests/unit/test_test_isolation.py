"""The suite must never write the live bot's files (on Windows the repo IS the
live folder): the trades database, journal, scalp reports and risk state."""

import os
from pathlib import Path

from tests.unit.test_dashboard_settings_api import run_and_stub  # noqa: F401
from tests.unit.test_risk_ladder import make_bot

REPO_DATA = (Path(__file__).resolve().parents[2] / "data").resolve()


def _inside_repo_data(path: str) -> bool:
    p = Path(path)
    p = p if p.is_absolute() else Path.cwd() / p
    try:
        p.resolve().relative_to(REPO_DATA)
        return True
    except ValueError:
        return False


def test_the_trades_database_is_not_the_live_one(run_and_stub):
    run_module, _ = run_and_stub
    assert not _inside_repo_data(run_module._get_sqlite_db_path())


def test_the_risk_state_file_is_not_the_live_one(run_and_stub):
    run_module, _ = run_and_stub
    bot = run_module.StandaloneAIBot(dict(run_module.CONFIG))
    assert not _inside_repo_data(bot._risk_state_file())


def test_persisting_and_recording_a_trade_leave_the_live_folder_alone(run_and_stub, tmp_path):
    run_module, _ = run_and_stub
    live_state = REPO_DATA / "state" / "risk_XAUUSDm.json"
    before = live_state.stat().st_mtime_ns if live_state.exists() else None
    bot = run_module.StandaloneAIBot(dict(run_module.CONFIG))
    bot._persist_risk_state()
    bot._record_trade_result(-1.0, "sl")
    after = live_state.stat().st_mtime_ns if live_state.exists() else None
    assert before == after
    assert os.path.exists(run_module._get_sqlite_db_path())   # it went to tmp instead
