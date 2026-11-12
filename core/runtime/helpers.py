"""
core/runtime/helpers.py — start, watch and stop the bot's helper processes from
``run.py``, so one command runs everything.

Two helpers, both separate OS processes (not threads), so heavy model training
can never slow the trading loop or share its interpreter lock:

* the neural-network trainer (``scripts/train_nn_loop.py``), one per symbol,
  restarted with backoff if it crashes;
* the diagnostics collector (``scripts/collect_diagnostics.py``), run on a
  schedule and once more at shutdown.

Neither helper sends orders. Both only read MT5 history and local files.

Failure policy: a helper that cannot run must never stop the bot. Every method
here swallows its own errors into a log line. A helper that keeps failing is
given up on, with one warning, instead of being restarted forever.

Pure standard library, no MetaTrader5, no torch: testable on any machine. The
clock and the process launcher are injectable for tests.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

EXIT_UNAVAILABLE = 3            # the helper cannot run here (e.g. torch missing): do not restart
SUPPORTED_TIMEFRAMES = ("M5", "M15", "H1")
# About one trading day of bars: how many new bars justify another training
# cycle. M5 is absent: the trainer's own default (300) already fits it.
MIN_NEW_BARS_PER_DAY = {"M15": 96, "H1": 24}
_FAST_FAILURE_S = 60.0          # a run shorter than this counts as a failure to start
_MAX_FAST_FAILURES = 5
_BACKOFF_BASE_S = 30.0
_BACKOFF_MAX_S = 600.0

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is still running. Works on Windows and POSIX."""
    if not pid or pid <= 0:
        return False
    if os.name == "nt":                                     # pragma: no cover - Windows only
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259               # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def start_parent_watch(pid: int, stop: threading.Event, interval_s: float = 5.0,
                       alive: Callable[[int], bool] = pid_alive) -> Optional[threading.Thread]:
    """In a helper process: set ``stop`` when the launching process (``pid``) is
    gone, so helpers never outlive a bot that crashed instead of shutting down."""
    if not pid or pid <= 0:
        return None

    def watch() -> None:
        while not stop.is_set():
            if not alive(pid):
                logger.warning("parent process %s is gone; stopping", pid)
                stop.set()
                return
            stop.wait(interval_s)

    thread = threading.Thread(target=watch, name="parent-watch", daemon=True)
    thread.start()
    return thread


def _flag(value: Optional[str], default: str) -> str:
    text = (value if value is not None and value.strip() else default).strip().lower()
    return text


def _number(value: Optional[str], default: float, minimum: float) -> float:
    try:
        number = float(value)                               # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if number >= minimum else default


@dataclass(frozen=True)
class HelperSettings:
    nn_enabled: bool
    nn_reason: str                  # why it is off, for the startup banner
    nn_interval_min: float
    nn_models: int
    nn_bars: int                    # history bars fetched for each training cycle
    nn_epochs: int                  # maximum epochs per model (early stopping applies)
    timeframe: str
    diag_enabled: bool
    diag_interval_min: float
    diag_days: int
    nn_threshold: float = 0.0       # NN_MIN_CONFIDENCE; 0 = the trainer's own default
    nn_holdout_bars: int = 0        # NN_TRAINER_HOLDOUT_BARS; 0 = the trainer's default
    nn_min_new_bars: int = 0        # NN_TRAINER_MIN_NEW_BARS; 0 = about one trading day
    nn_barrier_mult: float = 0.0    # NN_BARRIER_MULT: label move size in ATR; 0 = trainer default (1.5)
    nn_horizon: int = 0             # NN_HORIZON: label window in bars; 0 = trainer default (24)
    nn_arch: str = "gru"            # "gru" (default) | "dual" (core/ai/deep_nn.py)
    nn_preset: str = "small"        # dual only: small | medium | large

    @classmethod
    def from_env(cls, env: Mapping[str, str], torch_available: bool) -> "HelperSettings":
        timeframe = _flag(env.get("TIMEFRAME"), "M5").upper()
        nn_mode = _flag(env.get("NN_TRAINER_ENABLED"), "auto")
        reason = ""
        if nn_mode in _FALSE:
            nn_on, reason = False, "NN_TRAINER_ENABLED=false"
        elif nn_mode in _TRUE:
            nn_on = True
        else:                                               # "auto": train when torch is installed
            nn_on = torch_available
            if not nn_on:
                reason = "torch is not installed (pip install -r requirements-windows.txt)"
        if nn_on and timeframe not in SUPPORTED_TIMEFRAMES:
            nn_on, reason = False, f"timeframe {timeframe} is not one of {SUPPORTED_TIMEFRAMES}"
        diag_on = _flag(env.get("DIAGNOSTICS_ENABLED"), "true") not in _FALSE
        return cls(
            nn_enabled=nn_on, nn_reason=reason,
            nn_interval_min=_number(env.get("NN_TRAINER_INTERVAL_MIN"), 60.0, 1.0),
            nn_models=int(_number(env.get("NN_TRAINER_MODELS"), 3, 1)),
            nn_bars=int(_number(env.get("NN_TRAINER_BARS"), 30000, 3000)),
            nn_epochs=int(_number(env.get("NN_TRAINER_EPOCHS"), 50, 5)),
            nn_threshold=_number(env.get("NN_MIN_CONFIDENCE"), 0.0, 0.0),
            nn_holdout_bars=int(_number(env.get("NN_TRAINER_HOLDOUT_BARS"), 0, 0)),
            nn_min_new_bars=int(_number(env.get("NN_TRAINER_MIN_NEW_BARS"), 0, 0)),
            nn_barrier_mult=_number(env.get("NN_BARRIER_MULT"), 0.0, 0.0),
            nn_horizon=int(_number(env.get("NN_HORIZON"), 0, 0)),
            nn_arch=(env.get("NN_ARCH") or "gru").strip().lower(),
            nn_preset=(env.get("NN_PRESET") or "small").strip().lower(),
            timeframe=timeframe,
            diag_enabled=diag_on,
            diag_interval_min=_number(env.get("DIAGNOSTICS_INTERVAL_MIN"), 360.0, 1.0),
            diag_days=int(_number(env.get("DIAGNOSTICS_DAYS"), 7, 1)),
        )


def nn_command(python: str, root: Path, symbol: str, s: HelperSettings, parent_pid: int) -> List[str]:
    cmd = [python, str(root / "scripts" / "train_nn_loop.py"),
           "--symbol", symbol, "--timeframe", s.timeframe,
           "--out", str(root / "data" / "models" / f"nn_{symbol}_{s.timeframe}.pt"),
           "--interval-min", f"{s.nn_interval_min:g}", "--models", str(s.nn_models),
           "--bars", str(s.nn_bars), "--epochs", str(s.nn_epochs),
           "--parent-pid", str(parent_pid)]
    # The trainer must judge a model at the confidence the bot will use it at.
    if 0.0 < s.nn_threshold < 1.0:
        cmd += ["--threshold", f"{s.nn_threshold:g}"]
    if s.nn_holdout_bars > 0:
        cmd += ["--holdout-bars", str(s.nn_holdout_bars)]
    # What the network learns to call: a move of NN_BARRIER_MULT x ATR within
    # NN_HORIZON bars. The defaults (1.5 ATR in 24 bars) label most M5 bars as
    # "no move", so the network is rarely confident about a direction and the
    # trainer can never measure an edge. A scalp-sized label fixes that.
    if s.nn_barrier_mult > 0:
        cmd += ["--barrier-mult", f"{s.nn_barrier_mult:g}"]
    if s.nn_horizon > 0:
        cmd += ["--horizon", str(s.nn_horizon)]
    # New bars needed before retraining. The trainer's own default (300) suits
    # M5 (~1 day); on M15 it meant ~3 trading days between attempts.
    min_new = s.nn_min_new_bars or MIN_NEW_BARS_PER_DAY.get(s.timeframe, 0)
    if min_new > 0:
        cmd += ["--min-new-bars", str(min_new)]
    if s.nn_arch == "dual":                    # default (gru) keeps the command unchanged
        cmd += ["--arch", "dual", "--preset", s.nn_preset]
    return cmd


def diagnostics_command(python: str, root: Path, s: HelperSettings) -> List[str]:
    return [python, str(root / "scripts" / "collect_diagnostics.py"),
            "--root", str(root), "--days", str(s.diag_days)]


Spawn = Callable[..., "subprocess.Popen"]


HELPER_LOG_MAX_BYTES = 20 * 1024 * 1024


def rotate_if_large(log_path: Path, max_bytes: int = HELPER_LOG_MAX_BYTES) -> bool:
    """Move ``log_path`` to ``<name>.1`` (replacing an older one) once it is over
    ``max_bytes``. Helper logs are append-only and were never trimmed."""
    try:
        if log_path.exists() and log_path.stat().st_size > max_bytes:
            os.replace(log_path, log_path.with_name(log_path.name + ".1"))
            return True
    except OSError as exc:                     # a busy file must not stop the helper
        logger.warning("could not rotate %s: %r", log_path, exc)
    return False


def _default_spawn(command: Sequence[str], cwd: str, log_path: Optional[Path]) -> "subprocess.Popen":
    out = subprocess.DEVNULL
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotate_if_large(log_path)
        out = open(log_path, "ab")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(list(command), cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=flags)


def _stop_process(proc, grace_s: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception as exc:                                 # never raise out of shutdown
        logger.warning("could not stop helper cleanly: %r", exc)


class ManagedProcess:
    """A long-running helper that is restarted with backoff if it dies."""

    def __init__(self, name: str, command: Sequence[str], cwd: str,
                 log_path: Optional[Path] = None, spawn: Optional[Spawn] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.name, self.command, self.cwd, self.log_path = name, list(command), cwd, log_path
        self._spawn = spawn or _default_spawn
        self._clock = clock
        self.proc = None
        self.started_at = 0.0
        self.next_start = 0.0
        self.fast_failures = 0
        self.restarts = 0
        self.gave_up = False
        self.reason = ""

    def _launch(self) -> None:
        now = self._clock()
        try:
            self.proc = self._spawn(self.command, self.cwd, self.log_path)
            self.started_at = now
            logger.info("helper %s started (pid %s)", self.name, getattr(self.proc, "pid", "?"))
        except Exception as exc:
            self.proc = None
            self._failed(now, f"could not start: {exc!r}")

    def _failed(self, now: float, why: str) -> None:
        self.fast_failures += 1
        if self.fast_failures >= _MAX_FAST_FAILURES:
            self._give_up(f"{why}; failed {self.fast_failures} times in a row")
            return
        delay = min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * 2 ** (self.fast_failures - 1))
        self.next_start = now + delay
        logger.warning("helper %s: %s; retrying in %.0fs", self.name, why, delay)

    def _give_up(self, why: str) -> None:
        self.gave_up, self.reason = True, why
        logger.warning("helper %s gave up: %s (see the helper log; the bot is unaffected)",
                       self.name, why)

    def start(self) -> None:
        if not self.gave_up:
            self._launch()

    def tick(self) -> None:
        """Call regularly. Restarts a dead helper when its backoff has elapsed."""
        if self.gave_up:
            return
        now = self._clock()
        if self.proc is None:
            if now >= self.next_start:
                self.restarts += 1
                self._launch()
            return
        rc = self.proc.poll()
        if rc is None:
            return
        ran = now - self.started_at
        self.proc = None
        if rc == EXIT_UNAVAILABLE:
            self._give_up("it reported it cannot run on this machine")
        elif ran >= _FAST_FAILURE_S:
            self.fast_failures = 0                           # it ran a while: not a start-up failure
            self.next_start = now + _BACKOFF_BASE_S
            logger.warning("helper %s exited (code %s) after %.0fs; restarting", self.name, rc, ran)
        else:
            self._failed(now, f"exited with code {rc} after {ran:.0f}s")

    def stop(self) -> None:
        _stop_process(self.proc)
        self.proc = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class PeriodicJob:
    """A short helper run on a schedule, never overlapping itself."""

    def __init__(self, name: str, command: Sequence[str], cwd: str, interval_s: float,
                 first_delay_s: float, timeout_s: float = 300.0, log_path: Optional[Path] = None,
                 spawn: Optional[Spawn] = None, clock: Callable[[], float] = time.monotonic):
        self.name, self.command, self.cwd, self.log_path = name, list(command), cwd, log_path
        self.interval_s, self.timeout_s = interval_s, timeout_s
        self._spawn = spawn or _default_spawn
        self._clock = clock
        self.proc = None
        self.started_at = 0.0
        self.next_run = clock() + first_delay_s
        self.runs = 0

    def tick(self) -> None:
        now = self._clock()
        if self.proc is not None:
            rc = self.proc.poll()
            if rc is None:
                if now - self.started_at > self.timeout_s:
                    logger.warning("helper %s timed out after %.0fs; stopping it", self.name, self.timeout_s)
                    _stop_process(self.proc, 2.0)
                    self.proc = None
                    self.next_run = now + self.interval_s
                return
            if rc != 0:
                logger.warning("helper %s exited with code %s", self.name, rc)
            self.proc = None
            self.next_run = now + self.interval_s
            return
        if now >= self.next_run:
            try:
                self.proc = self._spawn(self.command, self.cwd, self.log_path)
                self.started_at = now
                self.runs += 1
            except Exception as exc:
                logger.warning("helper %s could not start: %r", self.name, exc)
                self.next_run = now + self.interval_s

    def run_blocking(self, timeout_s: float = 120.0) -> Optional[int]:
        """Run once and wait (used at shutdown). Returns the exit code, or ``None``."""
        _stop_process(self.proc, 2.0)
        self.proc = None
        try:
            proc = self._spawn(self.command, self.cwd, self.log_path)
            self.runs += 1
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _stop_process(proc, 2.0)
            logger.warning("final %s timed out", self.name)
        except KeyboardInterrupt:              # second Ctrl+C: skip it, exit cleanly
            _stop_process(proc, 2.0)
            logger.warning("final %s skipped (interrupted)", self.name)
        except Exception as exc:
            logger.warning("final %s failed: %r", self.name, exc)
        return None


class HelperSupervisor:
    def __init__(self, settings: HelperSettings, nn: List[ManagedProcess],
                 diagnostics: Optional[PeriodicJob]):
        self.settings, self.nn, self.diagnostics = settings, nn, diagnostics

    def describe(self) -> List[str]:
        s = self.settings
        lines = []
        if self.nn:
            lines.append(f"NN trainer   : ON  ({', '.join(p.name for p in self.nn)}) every "
                         f"{s.nn_interval_min:g} min, {s.nn_models} models")
        else:
            lines.append(f"NN trainer   : off ({s.nn_reason})")
        lines.append(f"Diagnostics  : ON  every {s.diag_interval_min:g} min and at shutdown "
                     f"-> diagnostics\\latest_summary.md" if self.diagnostics
                     else "Diagnostics  : off (DIAGNOSTICS_ENABLED=false)")
        return lines

    def start(self) -> None:
        for proc in self.nn:
            try:
                proc.start()
            except Exception as exc:
                logger.warning("helper %s failed to start: %r", proc.name, exc)

    def tick(self) -> None:
        for helper in [*self.nn, self.diagnostics]:
            if helper is None:
                continue
            try:
                helper.tick()
            except Exception as exc:                         # a helper must never stop the bot
                logger.warning("helper %s tick failed: %r", getattr(helper, "name", "?"), exc)

    def stop(self) -> None:
        for proc in self.nn:
            try:
                proc.stop()
            except Exception as exc:
                logger.warning("helper %s stop failed: %r", proc.name, exc)
        if self.diagnostics is not None:
            try:
                self.diagnostics.run_blocking()
            except Exception as exc:
                logger.warning("final diagnostics failed: %r", exc)


def torch_installed() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def build_supervisor(root: Path, symbols: Sequence[str], env: Mapping[str, str],
                     python: Optional[str] = None, parent_pid: Optional[int] = None,
                     torch_available: Optional[bool] = None, spawn: Optional[Spawn] = None,
                     clock: Callable[[], float] = time.monotonic) -> HelperSupervisor:
    root = Path(root)
    python = python or sys.executable
    parent_pid = parent_pid if parent_pid is not None else os.getpid()
    settings = HelperSettings.from_env(
        env, torch_installed() if torch_available is None else torch_available)
    logs = root / "logs"

    nn: List[ManagedProcess] = []
    if settings.nn_enabled:
        for symbol in dict.fromkeys(symbols):
            nn.append(ManagedProcess(
                f"nn-trainer:{symbol}", nn_command(python, root, symbol, settings, parent_pid),
                str(root), logs / f"nn_trainer_{symbol}.log", spawn, clock))

    diag = None
    if settings.diag_enabled:
        interval = settings.diag_interval_min * 60.0
        diag = PeriodicJob("diagnostics", diagnostics_command(python, root, settings), str(root),
                           interval_s=interval, first_delay_s=min(interval, 900.0),
                           log_path=logs / "diagnostics.log", spawn=spawn, clock=clock)
    return HelperSupervisor(settings, nn, diag)
