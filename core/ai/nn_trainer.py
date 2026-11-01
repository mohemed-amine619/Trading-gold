"""
core/ai/nn_trainer.py — keep the sequence model trained, without ever letting a
worse model replace a better one.

Each cycle:

1. fetch the latest closed bars;
2. choose a HOLDOUT: the newest bars the current champion has never seen
   (everything after its ``trained_through``), capped at ``holdout_bars``;
3. train a challenger on the bars before the holdout, leaving a gap of one label
   horizon so no training label reaches into it;
4. score champion and challenger with the SAME function on the SAME holdout;
5. promote the challenger only if ``should_promote`` says it is measurably
   better; otherwise the champion stays.

A promoted model is written atomically (temp file, then ``os.replace``), the old
one is kept as ``<file>.prev``, and every cycle appends one line to
``<file>.history.jsonl``. ``ModelPredictor`` notices the new file on its next
call, so nothing restarts.

Training more often does not make predictions better by itself. What protects
the live bot is step 5: an unmeasured or no-better model is never promoted.

torch is imported lazily (only when the default ``fit``/``evaluate`` are used),
so ``should_promote`` and the module import work without it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def should_promote(champion: Optional[dict], challenger: dict, floor: float = 0.0,
                   tolerance: float = 0.02) -> tuple:
    """Decide whether the challenger replaces the champion.

    Both dicts are ``trade_metrics`` results on the same holdout. An ``edge`` of
    ``None`` is unmeasured (too few trades), which is never treated as zero.
    Returns ``(promote, reason)``.
    """
    c_edge = challenger.get("edge")
    if c_edge is None:
        return False, "challenger edge unmeasured (too few holdout trades)"
    if c_edge <= floor:
        return False, f"challenger edge {c_edge:+.3f} is not above the floor {floor:+.3f}"
    if champion is None:
        return True, "no champion; challenger is above the floor"
    h_edge = champion.get("edge")
    if h_edge is None:
        return True, "champion edge unmeasured on this holdout; challenger is measured and above the floor"
    if c_edge > h_edge + tolerance:
        return True, (f"challenger {c_edge:+.3f} beats champion {h_edge:+.3f} "
                      f"by more than {tolerance:.3f}")
    return False, (f"challenger {c_edge:+.3f} does not beat champion {h_edge:+.3f} "
                   f"by {tolerance:.3f}")


def _result(action: str, reason: str, **extra) -> dict:
    return {"action": action, "reason": reason, **extra}


class ContinuousTrainer:
    def __init__(self, fetch_bars: Callable[[], pd.DataFrame], artifact_path: str, cfg,
                 holdout_bars: int = 1500, min_new_bars: int = 300,
                 min_train_bars: int = 1000, floor: float = 0.0, tolerance: float = 0.02,
                 interval_s: float = 3600.0, fit: Optional[Callable] = None,
                 evaluate: Optional[Callable] = None):
        self.fetch_bars = fetch_bars
        self.path = artifact_path
        self.cfg = cfg
        self.holdout_bars = holdout_bars
        self.min_new_bars = min_new_bars
        self.min_train_bars = min_train_bars
        self.floor = floor
        self.tolerance = tolerance
        self.interval_s = interval_s
        self._fit = fit
        self._evaluate = evaluate
        self._cycle_data_end: Optional[str] = None
        self._last_data_end: Optional[str] = self._read_last_data_end()

    # -- helpers ---------------------------------------------------------
    def _functions(self) -> tuple:
        if self._fit is None or self._evaluate is None:
            from core.ai.nn_model import evaluate_artifact, fit_artifact
            return (self._fit or fit_artifact, self._evaluate or evaluate_artifact)
        return self._fit, self._evaluate

    def _load_champion(self) -> Optional[dict]:
        if not os.path.exists(self.path):
            return None
        try:
            from core.ai.nn_model import load_artifact
            return load_artifact(self.path)
        except Exception as exc:
            logger.warning("champion unreadable, treating as absent: %r", exc)
            return None

    def _read_last_data_end(self) -> Optional[str]:
        """The newest bar a previous cycle trained on, from the history file, so a
        restart does not retrain on data it has already used."""
        try:
            last = None
            with open(self.path + ".history.jsonl", encoding="utf-8") as fh:
                for line in fh:
                    entry = json.loads(line)
                    if entry.get("action") in ("promoted", "kept") and entry.get("data_end"):
                        last = entry["data_end"]
            return last
        except (OSError, ValueError):
            return None

    def _log(self, entry: dict) -> None:
        try:
            line = json.dumps({"at": datetime.now(timezone.utc).isoformat(), **entry},
                              default=str)
            with open(self.path + ".history.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:                      # a log failure must not stop training
            logger.warning("could not write training history: %r", exc)

    def _write_atomically(self, artifact: dict, had_champion: bool) -> None:
        from core.ai.nn_model import save_artifact
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        save_artifact(tmp, artifact)
        if had_champion and os.path.exists(self.path):
            shutil.copyfile(self.path, self.path + ".prev")
        os.replace(tmp, self.path)

    # -- one cycle -------------------------------------------------------
    def run_once(self) -> dict:
        self._cycle_data_end = None
        result = self._cycle()
        if self._cycle_data_end is not None:
            result["data_end"] = self._cycle_data_end
            if result["action"] in ("promoted", "kept"):
                self._last_data_end = self._cycle_data_end
        logger.info("nn trainer: %s - %s", result["action"], result["reason"])
        self._log(result)
        return result

    def _cycle(self) -> dict:
        try:
            df = self.fetch_bars()
        except Exception as exc:
            return _result("error", f"could not fetch bars: {exc!r}")
        if df is None or len(df) < self.min_train_bars + self.cfg.horizon:
            return _result("skipped", "not enough bars to train")

        fit, evaluate = self._functions()
        champion = self._load_champion()
        n = len(df)
        self._cycle_data_end = str(df.index[-1])

        # Nothing to learn from if no bars have arrived since the last cycle
        # (a weekend, a closed market, or a quick restart).
        if self._last_data_end is not None:
            fresh = n - int(df.index.searchsorted(pd.Timestamp(self._last_data_end), side="right"))
            if fresh < self.min_new_bars:
                return _result("skipped", f"only {fresh} new bars since the last training cycle "
                                          f"(need {self.min_new_bars})")

        holdout_start = n - self.holdout_bars
        if champion is not None and champion.get("trained_through"):
            first_unseen = int(df.index.searchsorted(pd.Timestamp(champion["trained_through"]),
                                                     side="right"))
            if n - first_unseen < self.min_new_bars:
                return _result("skipped", f"only {n - first_unseen} bars since the champion "
                                          f"was trained (need {self.min_new_bars})")
            holdout_start = max(first_unseen, holdout_start)
        holdout_start = max(holdout_start, 0)

        train_end = holdout_start - self.cfg.horizon       # gap: labels must not reach the holdout
        if train_end < self.min_train_bars:
            return _result("skipped", f"only {train_end} bars before the holdout "
                                      f"(need {self.min_train_bars})")

        try:
            challenger = fit(df.iloc[:train_end], self.cfg)
            challenger_metrics = evaluate(challenger, df, holdout_start)
        except Exception as exc:
            return _result("error", f"challenger training failed: {exc!r}")

        champion_metrics = None
        if champion is not None:
            try:
                champion_metrics = evaluate(champion, df, holdout_start)
            except Exception as exc:
                logger.warning("champion could not be evaluated: %r", exc)

        promote, reason = should_promote(champion_metrics, challenger_metrics,
                                         self.floor, self.tolerance)
        holdout = {"holdout_start": str(df.index[holdout_start]),
                   "holdout_end": str(df.index[-1]), "holdout_bars": n - holdout_start}
        common = {"champion": champion_metrics, "challenger": challenger_metrics, **holdout}
        if not promote:
            return _result("kept", reason, **common)

        challenger["oos"] = {**challenger_metrics, "source": "holdout", **holdout}
        try:
            self._write_atomically(challenger, champion is not None)
        except Exception as exc:
            return _result("error", f"could not write the model: {exc!r}", **common)
        return _result("promoted", reason, **common)

    # -- forever ---------------------------------------------------------
    def run_forever(self, stop: threading.Event, max_cycles: Optional[int] = None) -> int:
        """Run cycles until ``stop`` is set (or ``max_cycles``). Never raises."""
        cycles = 0
        while not stop.is_set():
            try:
                self.run_once()
            except Exception as exc:                  # keep the loop alive
                logger.exception("nn trainer cycle crashed: %r", exc)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            stop.wait(self.interval_s)
        return cycles
