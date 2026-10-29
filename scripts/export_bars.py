"""
scripts/export_bars.py — export closed MT5 bars to CSV for offline evaluation.

Run on Windows, where the MT5 terminal is installed and logged in. The script
only reads history (copy_rates_range). It never sends orders.

    .venv\\Scripts\\python.exe scripts\\export_bars.py --symbol XAUUSDm --timeframe M15 --days 365 --out data\\history\\XAUUSDm_M15.csv

The output matches what the live bot builds from copy_rates (time index,
open/high/low/close/tick_volume/spread, forming candle dropped), so an offline
evaluation sees the same inputs as the bot. Times are the broker's server time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

TIMEFRAMES = ("M5", "M15", "H1")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="M15", choices=TIMEFRAMES)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    # Imported here so the module can be read or tested on any platform.
    import MetaTrader5 as mt5
    import pandas as pd

    tf = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}[args.timeframe]

    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return 1
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
        rates = mt5.copy_rates_range(args.symbol, tf, start, end)
        if rates is None or len(rates) == 0:
            print(f"no data returned for {args.symbol} {args.timeframe}: {mt5.last_error()}")
            return 1
        df = pd.DataFrame(rates).iloc[:-1].copy()     # drop the forming candle
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")[["open", "high", "low", "close", "tick_volume", "spread"]]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.astype(float).to_csv(args.out)
        print(f"wrote {len(df)} closed {args.timeframe} bars to {args.out}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
