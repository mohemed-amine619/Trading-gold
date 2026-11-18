"""
core/ai/nn_trainer.py — keep the sequence model trained, without ever letting a
worse model replace a better one, and without letting repeated testing turn
luck into a promotion.

Each cycle (at most once per ``min_new_bars`` new bars):

1. fetch the latest closed bars;
2. RE-CHECK the champion on bars it has never been judged on (everything after
   its ``validated_through``, at most the newest ``holdout_bars``). It is
   retired (moved to ``<file>.retired``, so the predictor has no file and is
   neutral) when that strictly fresh record is measured and its point edge is
   no longer above the floor, the naive baseline or its direction's naive edge;
   when a full ``holdout_bars`` of fresh bars fails the promotion gate; or when
   a full holdout of fresh bars still leaves it unmeasured. A model therefore
   votes only while bars it was never selected on keep agreeing with it;
3. if a CANDIDATE is on probation, judge it once on bars after its screening
   holdout (see "Two stages" below) and stop there: promoted or rejected;
4. otherwise SCREEN a new challenger: train it on the bars before the holdout
   (with a one-horizon gap so no training label reaches into the holdout) and
   score it, and the champion, with the SAME function on the SAME holdout
   (``evaluate_artifact``: each model at its own stored threshold, trades
   counted one position at a time, compared with the naive always-long /
   always-short baselines). A challenger that passes ``should_promote`` is
   saved as ``<file>.candidate``. It does not vote.

Two stages, because one test per cycle is not one test. The holdout slides by
about 5 % per cycle and every cycle trains a new challenger, so a 95 % bound
re-tested daily lets a no-skill model through within months (measured: ~0.5-1 %
per cycle). The screen is only a filter. The decision is the CONFIRMATION: the
frozen candidate (its weights, calibration and threshold fixed at screening) is
scored once on bars that arrived after its screening holdout ended, as soon as
those bars hold ``MIN_TRADES_FOR_EDGE`` non-overlapping trades (the moment
depends on how often it trades, not on whether it wins). Its bounds are taken at
``confirmation_z(k)`` for the k-th confirmation since the last promotion:
alpha-spending, alpha_k = alpha * 6 / (pi^2 k^2), which sums to ``alpha`` (5 %)
over every confirmation ever made, so the chance that a model with no edge is
EVER promoted stays below ``alpha`` however long the trainer runs. A candidate
still unmeasured after ``max_confirm_bars`` fresh bars is rejected.

A promoted model is written atomically (temp file, then ``os.replace``), the old
one is kept as ``<file>.prev``, and every cycle appends one line to
``<file>.history.jsonl`` (a retirement adds its own ``demoted`` line).
``ModelPredictor`` notices the new file on its next call, so nothing restarts.

Training more often does not make predictions better by itself. What protects
the live bot is the gate: an unmeasured, unconfirmed or no-better model is never
promoted, and a champion that stops earning its record is retired.

torch is imported lazily (only when the default ``fit``/``evaluate`` are used),
so ``should_promote`` and the module import work without it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import threading
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Callable, Optional

import pandas as pd

from core.ai.nn_features import LCB_Z, MIN_TRADES_FOR_EDGE, finite_number

logger = logging.getLogger(__name__)

# Family-wise error budget for promotions: the chance that a model with no edge
# is ever confirmed, over all confirmations since the last promotion.
CONFIRM_ALPHA = 0.05
# History actions that end a cycle having used the data up to its data_end (a
# ``demoted`` line is logged on its own, before the cycle's own verdict).
VERDICTS = ("promoted", "kept", "candidate", "probation", "rejected")


def _fmt(v) -> str:
    v = finite_number(v)
    return "n/a" if v is None else f"{v:+.3f}"


def confirmation_z(look: int, alpha: float = CONFIRM_ALPHA) -> float:
    """The z of the ``look``-th confirmation since the last promotion.

    alpha_k = alpha * 6 / (pi^2 k^2) sums to ``alpha`` over k = 1, 2, ..., so
    however many candidates are tried, the chance that any no-edge candidate is
    confirmed stays below ``alpha``. k=1 -> z 1.88, 2 -> 2.43, 3 -> 2.71,
    5 -> 3.03, 10 -> 3.43.
    """
    k = max(int(look), 1)
    a = min(max(float(alpha), 1e-12), 0.5) * 6.0 / (math.pi ** 2 * k * k)
    return NormalDist().inv_cdf(1.0 - a)


def bound_at(report: dict, mean_key: str, lcb_key: str, se_key: str,
             z: Optional[float] = None) -> Optional[float]:
    """The lower bound ``report[lcb_key]``, re-taken at the stricter ``z`` when
    one is given (``mean - z * se``; never looser than the stored bound).

    ``None`` when the bound is missing or not finite, or when it cannot be
    re-taken at ``z`` (no mean / standard error): unmeasured, never zero.
    """
    lcb = finite_number(report.get(lcb_key))
    if z is None or lcb is None:
        return lcb
    z0 = finite_number(report.get("lcb_z")) or LCB_Z
    if z <= z0:
        return lcb
    mean = finite_number(report.get(mean_key))
    se = finite_number(report.get(se_key))
    if se is None and mean is not None:
        se = (mean - lcb) / z0
    if mean is None or se is None or se < 0:
        return None
    return min(lcb, mean - z * se)


def should_promote(champion: Optional[dict], challenger: dict, floor: float = 0.0,
                   tolerance: float = 0.02, z: Optional[float] = None) -> tuple:
    """Decide whether the challenger passes the gate (and beats the champion).

    Both dicts are ``edge_report`` results on the same bars, each model at its
    own stored threshold, trades counted one position at a time. A value that
    is ``None``, missing or not a finite number is unmeasured: never treated as
    zero, always refuses. The challenger must clear EVERY gate:

    1. ``edge`` measured (at least ``MIN_TRADES_FOR_EDGE`` non-overlapping trades);
    2. ``edge_lcb`` (one-sided lower confidence bound) above ``floor``; the floor
       is never below 0, whatever is passed: "positive with confidence" is not
       negotiable;
    3. ``edge`` above ``baseline_edge``, the better of always-long and
       always-short on the same bars: a model no better than a naive direction
       is refused;
    4. ``excess_lcb > 0``: per trade, R minus the naive edge of the direction it
       took is positive with confidence, so riding a trend is not skill.

    ``z`` (the confirmation stage) re-takes both bounds at that stricter z from
    the reported standard errors. Then, against a measured champion, the
    existing rule: beat its ``edge`` by more than ``tolerance``. Returns
    ``(promote, reason)``; the reason is shown in the bot log.
    """
    floor = max(finite_number(floor) or 0.0, 0.0)
    tolerance = max(finite_number(tolerance) or 0.0, 0.0)
    c_edge = finite_number(challenger.get("edge"))
    n = challenger.get("trades")
    thr = finite_number(challenger.get("threshold"))
    at = f" at threshold {thr:.2f}" if thr is not None else ""
    zs = f" (z {z:.2f})" if z is not None else ""
    if c_edge is None:
        return False, (f"challenger edge unmeasured (too few holdout trades: {n if n is not None else 0} "
                       f"non-overlapping{at}, need {MIN_TRADES_FOR_EDGE})")
    c_lcb = bound_at(challenger, "edge", "edge_lcb", "edge_se", z)
    if c_lcb is None or not c_lcb > floor:
        return False, (f"challenger edge {c_edge:+.3f} over {n} trades{at} is not confidently "
                       f"above the floor {floor:+.3f} (lower bound {_fmt(c_lcb)}{zs})")
    base = finite_number(challenger.get("baseline_edge"))
    if base is None:
        return False, (f"challenger edge {c_edge:+.3f}: the naive baseline was not measured "
                       f"on these bars, so it cannot be shown to beat it")
    if not c_edge > base:
        return False, (f"challenger edge {c_edge:+.3f}{at} is no better than the naive "
                       f"always-{challenger.get('baseline_direction') or '?'} baseline {base:+.3f}")
    excess = bound_at(challenger, "excess", "excess_lcb", "excess_se", z)
    if excess is None or not excess > 0:
        return False, (f"challenger edge {c_edge:+.3f}{at} shows no timing skill beyond its "
                       f"direction (lower bound of R minus the naive edge {_fmt(excess)}{zs})")
    summary = (f"challenger edge {c_edge:+.3f} (lower bound {c_lcb:+.3f}{zs}, {n} trades{at}, "
               f"naive baseline {base:+.3f})")
    if champion is None:
        return True, f"no champion; {summary} clears the floor {floor:+.3f}"
    h_edge = finite_number(champion.get("edge"))
    if h_edge is None:
        return True, f"champion edge unmeasured on these bars; {summary} clears the floor"
    if c_edge > h_edge + tolerance:
        return True, (f"challenger {c_edge:+.3f} beats champion {h_edge:+.3f} "
                      f"by more than {tolerance:.3f}; {summary}")
    return False, (f"challenger {c_edge:+.3f} does not beat champion {h_edge:+.3f} "
                   f"by {tolerance:.3f}")


def _result(action: str, reason: str, **extra) -> dict:
    return {"action": action, "reason": reason, **extra}


def validated_through(art: dict) -> Optional[str]:
    """The last bar the model's record was measured on (or, for older files, the
    end of its holdout, else of its training): only bars after it are fresh."""
    oos = art.get("oos") or {}
    return art.get("validated_through") or oos.get("holdout_end") or art.get("trained_through")


class ContinuousTrainer:
    def __init__(self, fetch_bars: Callable[[], pd.DataFrame], artifact_path: str, cfg,
                 holdout_bars: int = 1500, min_new_bars: int = 300,
                 min_train_bars: int = 1000, floor: float = 0.0, tolerance: float = 0.02,
                 interval_s: float = 3600.0, fit: Optional[Callable] = None,
                 evaluate: Optional[Callable] = None, confirm_alpha: float = CONFIRM_ALPHA,
                 max_confirm_bars: Optional[int] = None):
        self.fetch_bars = fetch_bars
        self.path = artifact_path
        self.candidate_path = artifact_path + ".candidate"
        self.retired_path = artifact_path + ".retired"
        self.cfg = cfg
        self.holdout_bars = holdout_bars
        self.min_new_bars = min_new_bars
        self.min_train_bars = min_train_bars
        self.floor = floor
        self.tolerance = tolerance
        self.interval_s = interval_s
        self.confirm_alpha = confirm_alpha
        self.max_confirm_bars = int(max_confirm_bars or holdout_bars)
        self._fit = fit
        self._evaluate = evaluate
        self._cycle_data_end: Optional[str] = None
        self.settings_key = self._settings_key()
        self._last_data_end: Optional[str] = self._read_last_data_end()
        self._looks: int = self._read_looks()

    # -- helpers ---------------------------------------------------------
    def _functions(self) -> tuple:
        if self._fit is None or self._evaluate is None:
            from core.ai.nn_model import evaluate_artifact, fit_artifact
            return (self._fit or fit_artifact, self._evaluate or evaluate_artifact)
        return self._fit, self._evaluate

    def _load(self, path: str, what: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        try:
            from core.ai.nn_model import load_artifact
            return load_artifact(path)
        except Exception as exc:
            logger.warning("%s unreadable, treating as absent: %r", what, exc)
            return None

    def _load_champion(self) -> Optional[dict]:
        return self._load(self.path, "champion")

    def _settings_key(self) -> str:
        """What a cycle's verdict depends on besides the data: the training
        config (any new ``TrainConfig`` field changes it, so a deploy that
        changes the training recipe retrains at once), the holdout size and the
        promotion gate (floor, tolerance, confirmation budget). Stable across
        restarts."""
        try:
            from dataclasses import asdict, is_dataclass
            cfg = asdict(self.cfg) if is_dataclass(self.cfg) else dict(vars(self.cfg))
        except Exception:
            cfg = {"repr": repr(self.cfg)}
        return json.dumps({"cfg": cfg, "holdout_bars": self.holdout_bars,
                           "floor": self.floor, "tolerance": self.tolerance,
                           "confirm_alpha": self.confirm_alpha,
                           "max_confirm_bars": self.max_confirm_bars},
                          sort_keys=True, default=str)

    def _history(self) -> list:
        try:
            with open(self.path + ".history.jsonl", encoding="utf-8") as fh:
                out = []
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
                return out
        except OSError:
            return []

    def _read_last_data_end(self) -> Optional[str]:
        """The newest bar a previous cycle trained on, from the history file, so a
        restart does not retrain on data it has already used.

        Only cycles run with the SAME settings count. When the operator changes
        the confidence threshold or the holdout size (typically to get a verdict
        other than "edge unmeasured"), the old verdict says nothing about the new
        settings, so the next cycle trains immediately instead of waiting for
        ``min_new_bars`` new bars (about a trading day)."""
        last = None
        for entry in self._history():
            if (entry.get("action") in VERDICTS and entry.get("data_end")
                    and entry.get("settings") == self.settings_key):
                last = entry["data_end"]
        return last

    def _read_looks(self) -> int:
        """Measured confirmations that failed since the last promotion, whatever
        the settings: the alpha-spending count survives restarts and changes."""
        looks = 0
        for entry in self._history():
            if entry.get("action") == "promoted":
                looks = 0
            elif entry.get("action") == "rejected" and entry.get("look") is True:
                looks += 1
        return looks

    def _log(self, entry: dict) -> None:
        try:
            line = json.dumps({"at": datetime.now(timezone.utc).isoformat(), **entry},
                              default=str)
            with open(self.path + ".history.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:                      # a log failure must not stop training
            logger.warning("could not write training history: %r", exc)

    def _save_atomically(self, path: str, artifact: dict) -> None:
        from core.ai.nn_model import save_artifact
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        save_artifact(tmp, artifact)
        os.replace(tmp, path)

    def _write_atomically(self, artifact: dict, had_champion: bool) -> None:
        from core.ai.nn_model import save_artifact
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        save_artifact(tmp, artifact)
        if had_champion and os.path.exists(self.path):
            shutil.copyfile(self.path, self.path + ".prev")
        os.replace(tmp, self.path)

    def _discard_candidate(self) -> None:
        try:
            os.remove(self.candidate_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _row_after(df: pd.DataFrame, stamp) -> int:
        return int(df.index.searchsorted(pd.Timestamp(stamp), side="right"))

    # -- one cycle -------------------------------------------------------
    def run_once(self) -> dict:
        self._cycle_data_end = None
        result = self._cycle()
        result["settings"] = self.settings_key
        if self._cycle_data_end is not None:
            result["data_end"] = self._cycle_data_end
            if result["action"] in VERDICTS:
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
            fresh = n - self._row_after(df, self._last_data_end)
            if fresh < self.min_new_bars:
                return _result("skipped", f"only {fresh} new bars since the last training cycle "
                                          f"(need {self.min_new_bars})")

        # 1. The champion keeps its vote only while fresh bars agree with it.
        if champion is not None:
            retired = self._recheck_champion(champion, df, evaluate)
            if retired is not None and retired["action"] == "error":
                return retired
            if retired is not None:
                self._log({**retired, "settings": self.settings_key,
                           "data_end": self._cycle_data_end})
                logger.info("nn trainer: %s - %s", retired["action"], retired["reason"])
                champion = None

        # 2. A candidate on probation is confirmed or rejected, never re-screened.
        candidate = self._load(self.candidate_path, "candidate")
        if candidate is None and os.path.exists(self.candidate_path):
            self._discard_candidate()                          # unreadable: drop it
        if candidate is not None:
            return self._confirm(candidate, champion, df, evaluate)

        # 3. Screen a new challenger.
        holdout_start = n - self.holdout_bars
        if champion is not None and champion.get("trained_through"):
            first_unseen = self._row_after(df, champion["trained_through"])
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

        challenger["oos"] = {}                       # it has no confirmed record yet
        challenger["probation"] = {"settings": self.settings_key,
                                   "screened_through": str(df.index[-1]),
                                   "screen": {**challenger_metrics, "source": "screen",
                                              **holdout}}
        try:
            self._save_atomically(self.candidate_path, challenger)
        except Exception as exc:
            return _result("error", f"could not write the candidate: {exc!r}", **common)
        z = confirmation_z(self._looks + 1, self.confirm_alpha)
        return _result("candidate", f"{reason}; on probation (not voting) until confirmed "
                                    f"once on later bars at z {z:.2f}", **common)

    def _recheck_champion(self, champion: dict, df: pd.DataFrame, evaluate) -> Optional[dict]:
        """Retire the champion when bars it was never judged on stop agreeing
        with its record. Returns the ``demoted`` (or ``error``) entry, or ``None``
        when it keeps its vote."""
        since = validated_through(champion)
        if not since:
            return None
        n = len(df)
        first_fresh = self._row_after(df, since)
        fresh = n - first_fresh
        if fresh < self.min_new_bars:
            return None
        start = max(first_fresh, n - self.holdout_bars)
        full = fresh >= self.holdout_bars
        try:
            m = evaluate(champion, df, start)
        except Exception as exc:
            logger.warning("champion could not be re-checked: %r", exc)
            return None
        floor = max(finite_number(self.floor) or 0.0, 0.0)
        edge = finite_number(m.get("edge"))
        base = finite_number(m.get("baseline_edge"))
        excess = finite_number(m.get("excess"))
        why = None
        if edge is None:
            if full:
                why = (f"still unmeasured on {n - start} fresh bars "
                       f"({m.get('trades', 0)} non-overlapping trades, need {MIN_TRADES_FOR_EDGE})")
        elif not edge > floor:
            why = f"fresh edge {edge:+.3f} is no longer above the floor {floor:+.3f}"
        elif base is not None and not edge > base:
            why = (f"fresh edge {edge:+.3f} is no better than the naive "
                   f"always-{m.get('baseline_direction') or '?'} baseline {base:+.3f}")
        elif excess is not None and not excess > 0:
            why = f"fresh edge {edge:+.3f} shows no timing skill beyond its direction ({excess:+.3f})"
        elif full:
            ok, reason = should_promote(None, m, self.floor)
            if not ok:
                why = reason.replace("challenger", "fresh")
        if why is None:
            return None
        try:
            os.replace(self.path, self.retired_path)
        except Exception as exc:
            return _result("error", f"champion should be retired ({why}) but could not be "
                                    f"moved aside: {exc!r}")
        return _result("demoted", f"champion retired: {why} (bars after {since}; trades "
                                  f"{m.get('trades', 0)}, edge {_fmt(edge)}, lower bound "
                                  f"{_fmt(m.get('edge_lcb'))}, naive baseline {_fmt(base)})",
                       champion=m, window_start=str(df.index[start]),
                       window_end=str(df.index[-1]), window_bars=n - start)

    def _confirm(self, candidate: dict, champion: Optional[dict], df: pd.DataFrame,
                 evaluate) -> dict:
        """The single decisive look at a candidate, on bars after its screening
        holdout, once they hold enough non-overlapping trades."""
        info = candidate.get("probation") or {}
        if info.get("settings") != self.settings_key or not info.get("screened_through"):
            self._discard_candidate()
            return _result("rejected", "candidate was screened with other settings; discarded "
                                       "without a test", look=False)
        n = len(df)
        first_fresh = self._row_after(df, info["screened_through"])
        fresh = n - first_fresh
        look = self._looks + 1
        z = confirmation_z(look, self.confirm_alpha)
        try:
            m = evaluate(candidate, df, first_fresh)
        except Exception as exc:
            return _result("error", f"candidate could not be confirmed: {exc!r}")
        window = {"confirm_start": str(df.index[min(first_fresh, n - 1)]),
                  "confirm_end": str(df.index[-1]), "confirm_bars": fresh}
        if finite_number(m.get("edge")) is None:
            trades = m.get("trades", 0)
            if fresh >= self.max_confirm_bars:
                self._discard_candidate()
                return _result("rejected", f"candidate still unmeasured after {fresh} bars past its "
                                           f"screening holdout ({trades} non-overlapping trades, "
                                           f"need {MIN_TRADES_FOR_EDGE}); discarded",
                               look=False, candidate=m, **window)
            return _result("probation", f"candidate waiting for confirmation: {trades} "
                                        f"non-overlapping trades on {fresh} bars after its "
                                        f"screening holdout (need {MIN_TRADES_FOR_EDGE}); "
                                        f"confirmation {look} will use z {z:.2f}",
                           candidate=m, **window)

        champion_metrics = None
        if champion is not None:
            try:
                champion_metrics = evaluate(champion, df, first_fresh)
            except Exception as exc:
                logger.warning("champion could not be evaluated: %r", exc)
        ok, why = should_promote(champion_metrics, m, self.floor, self.tolerance, z=z)
        common = {"champion": champion_metrics, "candidate": m, "z": z, **window}
        if not ok:
            self._discard_candidate()
            self._looks += 1
            return _result("rejected", f"confirmation {look} on {fresh} later bars: {why}",
                           look=True, **common)

        art = {k: v for k, v in candidate.items() if k != "probation"}
        art["oos"] = {**m, "edge_lcb": bound_at(m, "edge", "edge_lcb", "edge_se", z),
                      "excess_lcb": bound_at(m, "excess", "excess_lcb", "excess_se", z),
                      "lcb_z": z, "source": "confirmation", "confirmation": look,
                      "screen": info.get("screen"), "holdout_start": window["confirm_start"],
                      "holdout_end": window["confirm_end"], "holdout_bars": fresh}
        art["validated_through"] = str(df.index[-1])
        try:
            self._write_atomically(art, champion is not None)
        except Exception as exc:
            return _result("error", f"could not write the model: {exc!r}", **common)
        self._discard_candidate()
        self._looks = 0
        return _result("promoted", f"confirmed on {fresh} bars after its screening holdout "
                                   f"(confirmation {look}): {why}", **common)

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
