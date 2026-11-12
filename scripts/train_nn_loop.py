"""
scripts/train_nn_loop.py — keep the sequence model training in the background.

Run it in a SECOND terminal next to run.py, on the machine where MT5 is logged
in. It only reads history (copy_rates_from_pos); it never sends orders.

    .venv\\Scripts\\python.exe scripts\\train_nn_loop.py --symbol XAUUSDm --timeframe M5 --out data\\models\\nn_XAUUSDm_M5.pt

Every ``--interval-min`` minutes it fetches the latest bars, trains a challenger,
and compares it with the saved model on the newest bars that model has never
seen. The saved file is replaced only if the challenger is measurably better
(see core/ai/nn_trainer.py). Ctrl+C stops it after the current cycle.

Training more often does not make the model better by itself. If no challenger
ever clears the gate (measured edge whose lower confidence bound beats
``--floor``, better than the naive always-one-direction baseline), the file is
never written and the predictor keeps saying "no trade". That is the safe
outcome, not a fault.

``run.py`` starts this script itself (see core/runtime/helpers.py), passing
``--parent-pid`` so it stops if the bot dies. Exit code 3 means "cannot run on
this machine" (torch missing); the supervisor does not restart it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TIMEFRAMES = ("M5", "M15", "H1")
EXIT_UNAVAILABLE = 3
PARENT_POLL_S = 5.0


def mt5_fetcher(symbol: str, timeframe: str, bars: int) -> Callable[[], pd.DataFrame]:
    """Closed bars from the running MT5 terminal (the forming candle is dropped)."""
    import MetaTrader5 as mt5                         # Windows only; imported here on purpose

    tf = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}[timeframe]
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    def fetch() -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no data for {symbol} {timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(rates).iloc[:-1].copy()
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.set_index("time")[["open", "high", "low", "close", "tick_volume"]].astype(float)

    return fetch


def main(argv: Optional[List[str]] = None,
         fetch_bars: Optional[Callable[[], pd.DataFrame]] = None,
         stop: Optional[threading.Event] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="XAUUSDm")
    parser.add_argument("--timeframe", default="M5", choices=TIMEFRAMES)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bars", type=int, default=30000)
    parser.add_argument("--interval-min", type=float, default=60.0)
    parser.add_argument("--holdout-bars", type=int, default=1500)
    parser.add_argument("--min-new-bars", type=int, default=300)
    parser.add_argument("--min-train-bars", type=int, default=1000)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--barrier-mult", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.72,
                        help="MINIMUM decision threshold: the trainer picks the threshold "
                             "on validation in [max(this, 0.40), 0.90] and stores it in "
                             "the model (run.py forwards NN_MIN_CONFIDENCE)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--models", type=int, default=3, help="seeds averaged per model")
    parser.add_argument("--arch", choices=("gru", "dual"), default="gru",
                        help="network: the original GRU, or the dual-stream deep net")
    parser.add_argument("--preset", choices=("small", "medium", "large"), default="small",
                        help="dual only: capacity preset")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="a challenger must beat the champion's edge by this much")
    parser.add_argument("--floor", type=float, default=0.0,
                        help="the lower confidence bound of the holdout edge must exceed "
                             "this (never below 0; run.py forwards NN_MIN_OOS_EDGE)")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="stop when this process is gone (set by run.py)")
    args = parser.parse_args(argv)

    try:
        from core.ai.nn_model import TrainConfig
        from core.ai.nn_trainer import ContinuousTrainer
    except ImportError as exc:
        print(f"torch is required: {exc}")
        return EXIT_UNAVAILABLE

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = TrainConfig(window=args.window, horizon=args.horizon,
                      barrier_mult=args.barrier_mult, threshold=args.threshold,
                      epochs=args.epochs, n_models=args.models,
                      arch=args.arch, preset=args.preset)
    fetch = fetch_bars or mt5_fetcher(args.symbol, args.timeframe, args.bars)

    trainer = ContinuousTrainer(
        fetch, str(args.out), cfg, holdout_bars=args.holdout_bars,
        min_new_bars=args.min_new_bars, min_train_bars=args.min_train_bars,
        floor=args.floor, tolerance=args.tolerance, interval_s=args.interval_min * 60.0)

    if args.once:
        result = trainer.run_once()
        print(f"{result['action']}: {result['reason']}")
        return 0 if result["action"] != "error" else 1

    stop = stop or threading.Event()
    if args.parent_pid:
        from core.runtime.helpers import start_parent_watch
        start_parent_watch(args.parent_pid, stop, PARENT_POLL_S)
    print(f"training {args.symbol} {args.timeframe} every {args.interval_min:g} min; "
          f"saving to {args.out}. Ctrl+C to stop.")
    try:
        trainer.run_forever(stop)
    except KeyboardInterrupt:
        stop.set()
        print("stopping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
