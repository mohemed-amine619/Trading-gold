"""
core/ai/nn_trainer.py — keep the sequence model trained, without ever letting a
worse model replace a better one.

Each cycle:

1. fetch the latest closed bars;
2. choose a HOLDOUT: the newest bars the current champion has never seen
   (everything after its ``trained_through``), capped at ``holdout_bars``;
3. train a challenger on the bars before the holdout, leaving a gap of one label
   horizon so no training label reaches into it;
4. score champion and challenger with the SAME function on the SAME holdout
   (``evaluate_artifact``: each model at its own stored threshold, trades
   counted one position at a time, compared with the naive always-long /
   always-short baselines on the same bars);
5. promote the challenger only if ``should_promote`` says it is measurably
   better: measured edge, lower confidence bound above ``floor`` (never below
   0), better than the naive baseline, timing skill beyond its direction, and
   (against a measured champion) more than ``tolerance`` above the champion.
   Otherwise the champion stays.

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

from core.ai.nn_features import MIN_TRADES_FOR_EDGE

logger = logging.getLogger(__name__)


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:+.3f}"


def should_promote(champion: Optional[dict], challenger: dict, floor: float = 0.0,
                   tolerance: float = 0.02) -> tuple:
    """Decide whether the challenger replaces the champion.

    Both dicts are ``edge_report`` results on the same holdout, each model at its
    own stored threshold, trades counted one position at a time. A value of
    ``None`` (or a missing key) is unmeasured, which is never treated as zero and
    always refuses. The challenger must clear EVERY gate:

    1. ``edge`` measured (at least ``MIN_TRADES_FOR_EDGE`` non-overlapping trades);
    2. ``edge_lcb`` (one-sided lower confidence bound) above ``floor``; the floor
       is never below 0, whatever is passed: "positive with confidence" is not
       negotiable;
    3. ``edge`` above ``baseline_edge``, the better of always-long and
       always-short on the same bars: a model no better than a naive direction
       is refused;
    4. ``excess_lcb > 0``: per trade, R minus the naive edge of the direction it
       took is positive with confidence, so riding a trend is not skill.

    Then, against a measured champion, the existing rule: beat its ``edge`` by
    more than ``tolerance``. Returns ``(promote, reason)``; the reason is shown
    in the bot log.
    """
    floor = max(float(floor), 0.0)
    c_edge = challenger.get("edge")
    n = challenger.get("trades")
    thr = challenger.get("threshold")
    at = f" at threshold {thr:.2f}" if isinstance(thr, (int, float)) else ""
    if c_edge is None:
        return False, (f"challenger edge unmeasured (too few holdout trades: {n if n is not None else 0} "
                       f"non-overlapping{at}, need {MIN_TRADES_FOR_EDGE})")
    c_lcb = challenger.get("edge_lcb")
    if c_lcb is None or c_lcb <= floor:
        return False, (f"challenger edge {c_edge:+.3f} over {n} trades{at} is not confidently "
                       f"above the floor {floor:+.3f} (lower bound {_fmt(c_lcb)})")
    base = challenger.get("baseline_edge")
    if base is None:
        return False, (f"challenger edge {c_edge:+.3f}: the naive baseline was not measured "
                       f"on this holdout, so it cannot be shown to beat it")
    if c_edge <= base:
        return False, (f"challenger edge {c_edge:+.3f}{at} is no better than the naive "
                       f"always-{challenger.get('baseline_direction') or '?'} baseline {base:+.3f}")
    excess = challenger.get("excess_lcb")
    if excess is None or excess <= 0:
        return False, (f"challenger edge {c_edge:+.3f}{at} shows no timing skill beyond its "
                       f"direction (lower bound of R minus the naive edge {_fmt(excess)})")
    summary = (f"challenger edge {c_edge:+.3f} (lower bound {c_lcb:+.3f}, {n} trades{at}, "
               f"naive baseline {base:+.3f})")
    if champion is None:
        return True, f"no champion; {summary} clears the floor {floor:+.3f}"
    h_edge = champion.get("edge")
    if h_edge is None:
        return True, f"champion edge unmeasured on this holdout; {summary} clears the floor"
    if c_edge > h_edge + tolerance:
        return True, (f"challenger {c_edge:+.3f} beats champion {h_edge:+.3f} "
                      f"by more than {tolerance:.3f}; {summary}")
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
        self.settings_key = self._settings_key()
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

    def _settings_key(self) -> str:
        """What a cycle's verdict depends on besides the data: the training
        config (any new ``TrainConfig`` field changes it, so a deploy that
        changes the training recipe retrains at once), the holdout size and the
        promotion gate (floor, tolerance). Stable across restarts."""
        try:
            from dataclasses import asdict, is_dataclass
            cfg = asdict(self.cfg) if is_dataclass(self.cfg) else dict(vars(self.cfg))
        except Exception:
            cfg = {"repr": repr(self.cfg)}
        return json.dumps({"cfg": cfg, "holdout_bars": self.holdout_bars,
                           "floor": self.floor, "tolerance": self.tolerance},
                          sort_keys=True, default=str)

    def _read_last_data_end(self) -> Optional[str]:
        """The newest bar a previous cycle trained on, from the history file, so a
        restart does not retrain on data it has already used.

        Only cycles run with the SAME settings count. When the operator changes
        the confidence threshold or the holdout size (typically to get a verdict
        other than "edge unmeasured"), the old verdict says nothing about the new
        settings, so the next cycle trains immediately instead of waiting for
        ``min_new_bars`` new bars (about a trading day)."""
        try:
            last = None
            with open(self.path + ".history.jsonl", encoding="utf-8") as fh:
                for line in fh:
                    entry = json.loads(line)
                    if (entry.get("action") in ("promoted", "kept") and entry.get("data_end")
                            and entry.get("settings") == self.settings_key):
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
        result["settings"] = self.settings_key
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
            got = 0 if df is None else len(df)
            return _result("skipped", f"not enough bars to train: got {got}, need "
                                      f"{self.min_train_bars + self.cfg.horizon} "
                                      f"(min_train_bars {self.min_train_bars} + horizon "
                                      f"{self.cfg.horizon})")

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
