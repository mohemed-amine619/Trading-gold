"""
StateManager - in-memory mirror of the positions WE own.

`mt5.positions_get()` returns EVERY position on the account (manual trades,
other bots), and it has no magic-number filter parameter - so we fetch all
and filter by MAGIC_NUMBER in Python. The mirror is used to:
  * enforce one-position-per-symbol
  * detect entries/exits (for alerts and logging)
  * drive the trailing-stop loop
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticket: int
    symbol: str
    direction: int            # +1 buy, -1 sell
    volume: float
    entry: float
    sl: Optional[float]
    tp: Optional[float]
    profit: float
    magic: int

    @classmethod
    def from_mt5(cls, pos) -> "Position":
        return cls(
            ticket=pos.ticket,
            symbol=pos.symbol,
            direction=1 if pos.type == mt5.POSITION_TYPE_BUY else -1,
            volume=pos.volume,
            entry=pos.price_open,
            sl=pos.sl or None,
            tp=pos.tp or None,
            profit=pos.profit,
            magic=pos.magic,
        )


class StateManager:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self._positions: Dict[str, Position] = {}   # keyed by symbol

    async def refresh(self) -> Dict[str, Position]:
        """Re-sync the mirror from MT5. Returns the current {symbol: Position} map."""
        # positions_get() has no magic filter -> fetch all, filter in Python.
        all_positions = await self.engine.call(mt5.positions_get) or []
        ours = [p for p in all_positions if p.magic == self.config.MAGIC_NUMBER]

        new_map = {p.symbol: Position.from_mt5(p) for p in ours}

        # Log exits we were tracking (useful for alerts & debugging).
        for symbol in list(self._positions):
            if symbol not in new_map:
                logger.info("Position on %s no longer open (closed/stopped out) - "
                            "ticket %s", symbol, self._positions[symbol].ticket)
        self._positions = new_map
        return self._positions

    def get(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def symbols(self) -> List[str]:
        return list(self._positions)

    def open_positions(self) -> List[Position]:
        return list(self._positions.values())

    def count(self) -> int:
        return len(self._positions)

    async def equity(self) -> float:
        acc = await self.engine.call(mt5.account_info)
        return acc.equity if acc else 0.0
