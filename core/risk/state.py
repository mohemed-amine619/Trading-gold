"""
core/risk/state.py — risk state that survives a restart.

The daily drawdown guard, the trade counter and the loss-streak cooldown all
lived in RAM. Restarting the bot at 14:00 after a bad morning reset
``daily_start`` to the current equity, which silently re-armed the full daily
loss budget. That is the exact opposite of what a daily cap is for.

This module persists that state to a small JSON file per symbol under
``data/state/``. It holds no broker dependency, so it is importable and testable
on Linux.

Design notes
------------
* Writes are atomic (temp file + ``os.replace``) so a crash mid-write cannot
  leave a truncated file that fails to parse.
* A corrupt or missing file is treated as "no history", never as a reason to
  refuse to start. A bot that will not boot is riskier than one that forgot.
* Dates are ISO strings so the file stays readable and diff-friendly.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Everything the risk guard needs to remember across a restart."""

    daily_date: Optional[str] = None
    daily_start_equity: float = 0.0
    trade_date: Optional[str] = None
    trades_today: int = 0
    dd_positions_closed: bool = False
    consecutive_losses: int = 0
    cooldown_until: Optional[str] = None
    peak_equity: float = 0.0
    week_key: Optional[str] = None
    week_start_equity: float = 0.0
    halted_reason: Optional[str] = None

    # -- helpers ---------------------------------------------------------
    def is_new_day(self, today: Optional[date] = None) -> bool:
        today = today or date.today()
        return self.daily_date != today.isoformat()

    def roll_day(self, equity: float, today: Optional[date] = None) -> None:
        today = today or date.today()
        self.daily_date = today.isoformat()
        self.daily_start_equity = float(equity)
        self.dd_positions_closed = False

    def roll_trade_day(self, today: Optional[date] = None) -> None:
        today = today or date.today()
        if self.trade_date != today.isoformat():
            self.trade_date = today.isoformat()
            self.trades_today = 0

    def in_cooldown(self, now: Optional[datetime] = None) -> bool:
        if not self.cooldown_until:
            return False
        now = now or datetime.utcnow()
        try:
            until = datetime.fromisoformat(self.cooldown_until)
        except ValueError:
            return False
        return now < until

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RiskState":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (raw or {}).items() if k in known})


def state_path(directory: str, symbol: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(symbol))
    return os.path.join(directory, f"risk_{safe}.json")


def load_state(path: str) -> RiskState:
    """Load state, or a fresh empty state if absent/corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError("state file is not an object")
        return RiskState.from_dict(raw)
    except FileNotFoundError:
        return RiskState()
    except Exception as exc:
        logger.warning("could not read risk state %s (%s); starting fresh", path, exc)
        return RiskState()


def save_state(path: str, state: RiskState) -> bool:
    """Atomically persist *state*. Returns False on failure, never raises."""
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent or ".", prefix=".risk_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state.to_dict(), fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("could not persist risk state %s: %s", path, exc)
        return False
