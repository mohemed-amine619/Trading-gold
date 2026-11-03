"""
tests/unit/test_runtime_helpers.py — the helper supervisor that run.py uses to
start the NN trainer and the diagnostics collector. Fake clock, fake processes:
no real subprocess is started except in the pid tests.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from core.runtime.helpers import (
    EXIT_UNAVAILABLE, HelperSettings, ManagedProcess, PeriodicJob, build_supervisor,
    diagnostics_command, nn_command, pid_alive, start_parent_watch,
)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class FakeProc:
    def __init__(self, pid=4242):
        self.pid, self.rc = pid, None
        self.terminated = self.killed = False
        self.hang = False

    def poll(self):
        return self.rc

    def terminate(self):
        self.terminated = True
        if not self.hang:
            self.rc = -15

    def kill(self):
        self.killed = True
        self.rc = -9

    def wait(self, timeout=None):
        if self.rc is None:
            if self.hang:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.rc = 0
        return self.rc


class Spawner:
    def __init__(self):
        self.procs, self.fail = [], False

    def __call__(self, command, cwd, log_path):
        if self.fail:
            raise OSError("cannot start")
        proc = FakeProc(pid=len(self.procs) + 100)
        self.procs.append((list(command), proc))
        return proc

    @property
    def last(self):
        return self.procs[-1][1]


ENV_ON = {"NN_TRAINER_ENABLED": "true", "TIMEFRAME": "M5"}


class TestSettings:
    def test_defaults_train_only_when_torch_is_installed(self):
        assert HelperSettings.from_env({}, torch_available=True).nn_enabled is True
        off = HelperSettings.from_env({}, torch_available=False)
        assert off.nn_enabled is False and "torch is not installed" in off.nn_reason

    def test_explicit_false_wins_over_torch(self):
        s = HelperSettings.from_env({"NN_TRAINER_ENABLED": "false"}, True)
        assert s.nn_enabled is False and "NN_TRAINER_ENABLED=false" in s.nn_reason

    def test_explicit_true_is_honoured_without_torch(self):
        assert HelperSettings.from_env({"NN_TRAINER_ENABLED": "true"}, False).nn_enabled is True

    def test_an_unsupported_timeframe_disables_the_trainer_with_a_reason(self):
        s = HelperSettings.from_env({"NN_TRAINER_ENABLED": "true", "TIMEFRAME": "D1"}, True)
        assert s.nn_enabled is False and "D1" in s.nn_reason

    def test_numbers_fall_back_on_garbage_and_nonsense(self):
        s = HelperSettings.from_env({"NN_TRAINER_INTERVAL_MIN": "abc", "NN_TRAINER_MODELS": "0",
                                     "DIAGNOSTICS_INTERVAL_MIN": "-5", "DIAGNOSTICS_DAYS": "x"}, True)
        assert (s.nn_interval_min, s.nn_models) == (60.0, 3)
        assert (s.diag_interval_min, s.diag_days) == (360.0, 7)

    def test_diagnostics_is_on_by_default_and_can_be_switched_off(self):
        assert HelperSettings.from_env({}, False).diag_enabled is True
        assert HelperSettings.from_env({"DIAGNOSTICS_ENABLED": "false"}, False).diag_enabled is False

    def test_training_volume_defaults_and_limits(self):
        s = HelperSettings.from_env({}, True)
        assert (s.nn_bars, s.nn_epochs) == (30000, 50)
        small = HelperSettings.from_env({"NN_TRAINER_BARS": "100", "NN_TRAINER_EPOCHS": "1"}, True)
        assert (small.nn_bars, small.nn_epochs) == (30000, 50)          # below the minimums: defaults
        more = HelperSettings.from_env({"NN_TRAINER_BARS": "60000", "NN_TRAINER_EPOCHS": "120"}, True)
        assert (more.nn_bars, more.nn_epochs) == (60000, 120)

    def test_numbers_are_read(self):
        s = HelperSettings.from_env({"NN_TRAINER_INTERVAL_MIN": "30", "NN_TRAINER_MODELS": "5",
                                     "DIAGNOSTICS_INTERVAL_MIN": "120", "DIAGNOSTICS_DAYS": "14"}, True)
        assert (s.nn_interval_min, s.nn_models, s.diag_interval_min, s.diag_days) == (30.0, 5, 120.0, 14)


class TestCommands:
    def test_the_trainer_command_carries_symbol_paths_and_the_parent_pid(self):
        s = HelperSettings.from_env({"NN_TRAINER_ENABLED": "true", "NN_TRAINER_INTERVAL_MIN": "30"}, True)
        cmd = nn_command("py", Path("/bot"), "XAUUSDm", s, 777)
        assert cmd[:2] == ["py", str(Path("/bot/scripts/train_nn_loop.py"))]
        assert cmd[cmd.index("--symbol") + 1] == "XAUUSDm"
        assert cmd[cmd.index("--timeframe") + 1] == "M5"
        assert cmd[cmd.index("--out") + 1] == str(Path("/bot/data/models/nn_XAUUSDm_M5.pt"))
        assert cmd[cmd.index("--interval-min") + 1] == "30"
        assert cmd[cmd.index("--parent-pid") + 1] == "777"

    def test_the_trainer_command_carries_the_training_volume(self):
        s = HelperSettings.from_env({"NN_TRAINER_ENABLED": "true", "NN_TRAINER_BARS": "45000",
                                     "NN_TRAINER_EPOCHS": "80"}, True)
        cmd = nn_command("py", Path("/bot"), "XAUUSDm", s, 1)
        assert cmd[cmd.index("--bars") + 1] == "45000"
        assert cmd[cmd.index("--epochs") + 1] == "80"

    def test_the_diagnostics_command(self):
        s = HelperSettings.from_env({"DIAGNOSTICS_DAYS": "3"}, True)
        cmd = diagnostics_command("py", Path("/bot"), s)
        assert cmd[1].endswith("collect_diagnostics.py")
        assert cmd[cmd.index("--root") + 1] == str(Path("/bot"))
        assert cmd[cmd.index("--days") + 1] == "3"


def managed(spawn, clock):
    return ManagedProcess("nn", ["x"], ".", None, spawn, clock)


class TestManagedProcess:
    def test_a_healthy_helper_is_left_alone(self):
        sp, clock = Spawner(), FakeClock()
        m = managed(sp, clock)
        m.start()
        for _ in range(5):
            clock.advance(600)
            m.tick()
        assert len(sp.procs) == 1 and m.running

    def test_a_quick_crash_is_retried_after_a_backoff(self):
        sp, clock = Spawner(), FakeClock()
        m = managed(sp, clock)
        m.start()
        clock.advance(5)
        sp.last.rc = 1                                    # died after 5 s
        m.tick()
        assert len(sp.procs) == 1                         # not restarted yet
        clock.advance(29)
        m.tick()
        assert len(sp.procs) == 1
        clock.advance(1)
        m.tick()
        assert len(sp.procs) == 2 and m.restarts == 1     # after 30 s

    def test_the_backoff_doubles_with_each_quick_crash(self):
        sp, clock = Spawner(), FakeClock()
        m = managed(sp, clock)
        m.start()
        waits = []
        for _ in range(3):
            clock.advance(2)
            sp.last.rc = 1
            m.tick()
            waits.append(m.next_start - clock())
            clock.advance(m.next_start - clock())
            m.tick()
        assert waits == [30.0, 60.0, 120.0]

    def test_it_gives_up_after_repeated_quick_failures_and_stops_spawning(self):
        sp, clock = Spawner(), FakeClock()
        m = managed(sp, clock)
        m.start()
        for _ in range(10):
            clock.advance(1)
            if sp.last.poll() is None:
                sp.last.rc = 1
            m.tick()
            clock.advance(700)
            m.tick()
        assert m.gave_up is True and "failed 5 times" in m.reason
        spawned = len(sp.procs)
        clock.advance(10_000)
        m.tick()
        assert len(sp.procs) == spawned

    def test_a_crash_after_a_long_run_resets_the_failure_count(self):
        sp, clock = Spawner(), FakeClock()
        m = managed(sp, clock)
        m.start()
        clock.advance(2)
        sp.last.rc = 1
        m.tick()                                          # one quick failure
        clock.advance(30)
        m.tick()                                          # restarted
        clock.advance(500)                                # it ran a good while
        sp.last.rc = 1
        m.tick()
        assert m.fast_failures == 0 and not m.gave_up

    def test_the_unavailable_exit_code_is_never_retried(self):
        sp, clock = Spawner(), FakeClock()
        m = managed(sp, clock)
        m.start()
        sp.last.rc = EXIT_UNAVAILABLE
        m.tick()
        assert m.gave_up and "cannot run" in m.reason
        clock.advance(10_000)
        m.tick()
        assert len(sp.procs) == 1

    def test_a_spawn_failure_is_a_failure_not_a_crash(self):
        sp, clock = Spawner(), FakeClock()
        sp.fail = True
        m = managed(sp, clock)
        m.start()                                          # must not raise
        assert m.proc is None and m.fast_failures == 1
        sp.fail = False
        clock.advance(30)
        m.tick()
        assert m.running

    def test_stop_terminates_and_kills_a_hung_helper(self):
        sp, clock = Spawner(), FakeClock()
        m = managed(sp, clock)
        m.start()
        sp.last.hang = True
        proc = sp.last
        m.stop()
        assert proc.terminated and proc.killed and m.proc is None

    def test_stop_without_a_process_is_harmless(self):
        managed(Spawner(), FakeClock()).stop()


def job(spawn, clock, interval=3600, first=600, timeout=300):
    return PeriodicJob("diag", ["x"], ".", interval, first, timeout, None, spawn, clock)


class TestPeriodicJob:
    def test_first_run_waits_for_the_first_delay(self):
        sp, clock = Spawner(), FakeClock()
        j = job(sp, clock)
        clock.advance(599)
        j.tick()
        assert sp.procs == []
        clock.advance(1)
        j.tick()
        assert len(sp.procs) == 1

    def test_runs_again_one_interval_after_it_finished(self):
        sp, clock = Spawner(), FakeClock()
        j = job(sp, clock)
        clock.advance(600)
        j.tick()
        sp.last.rc = 0
        j.tick()                                           # notices it finished
        clock.advance(3599)
        j.tick()
        assert len(sp.procs) == 1
        clock.advance(1)
        j.tick()
        assert len(sp.procs) == 2

    def test_never_overlaps_a_run_still_in_progress(self):
        sp, clock = Spawner(), FakeClock()
        j = job(sp, clock, interval=10, first=0)
        j.tick()
        clock.advance(100)
        j.tick()
        assert len(sp.procs) == 1                          # still running, no second copy

    def test_a_run_that_hangs_is_killed_after_the_timeout(self):
        sp, clock = Spawner(), FakeClock()
        j = job(sp, clock, interval=3600, first=0, timeout=300)
        j.tick()
        proc = sp.last
        proc.hang = True
        clock.advance(301)
        j.tick()
        assert proc.terminated and j.proc is None

    def test_a_spawn_failure_just_waits_for_the_next_interval(self):
        sp, clock = Spawner(), FakeClock()
        sp.fail = True
        j = job(sp, clock, first=0)
        j.tick()                                           # must not raise
        sp.fail = False
        clock.advance(10)
        j.tick()
        assert sp.procs == []

    def test_run_blocking_waits_for_the_exit_code(self):
        sp, clock = Spawner(), FakeClock()
        j = job(sp, clock)
        assert j.run_blocking(5) == 0 and len(sp.procs) == 1

    def test_run_blocking_survives_a_failure(self):
        sp, clock = Spawner(), FakeClock()
        sp.fail = True
        assert job(sp, clock).run_blocking(5) is None


class TestSupervisor:
    def build(self, env=None, symbols=("XAUUSDm",), torch=True):
        sp, clock = Spawner(), FakeClock()
        sup = build_supervisor(Path("/bot"), symbols, ENV_ON if env is None else env,
                               python="py", parent_pid=1, torch_available=torch,
                               spawn=sp, clock=clock)
        return sup, sp, clock

    def test_one_trainer_per_symbol_and_a_diagnostics_job(self):
        sup, sp, _ = self.build(symbols=("XAUUSDm", "EURUSDm", "XAUUSDm"))
        assert [p.name for p in sup.nn] == ["nn-trainer:XAUUSDm", "nn-trainer:EURUSDm"]
        assert sup.diagnostics is not None

    def test_start_launches_the_trainers(self):
        sup, sp, _ = self.build()
        sup.start()
        assert len(sp.procs) == 1 and "train_nn_loop.py" in sp.procs[0][0][1]

    def test_without_torch_in_auto_mode_there_is_no_trainer_and_the_banner_says_why(self):
        sup, sp, _ = self.build(env={}, torch=False)
        assert sup.nn == []
        assert any("NN trainer   : off" in line and "torch" in line for line in sup.describe())

    def test_diagnostics_can_be_disabled(self):
        sup, _, _ = self.build(env={"DIAGNOSTICS_ENABLED": "false", "NN_TRAINER_ENABLED": "false"})
        assert sup.diagnostics is None and sup.nn == []

    def test_tick_restarts_a_dead_trainer_and_runs_the_scheduled_diagnostics(self):
        sup, sp, clock = self.build()
        sup.start()
        clock.advance(900)
        sp.procs[0][1].rc = 1
        sup.tick()
        assert any("collect_diagnostics.py" in cmd[1] for cmd, _ in sp.procs)

    def test_tick_never_raises_even_if_a_helper_does(self):
        sup, sp, _ = self.build()
        sup.start()
        sup.nn[0].tick = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        sup.diagnostics.tick = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        sup.tick()

    def test_stop_ends_the_trainers_and_takes_one_last_diagnostics_snapshot(self):
        sup, sp, _ = self.build()
        sup.start()
        trainer = sp.procs[0][1]
        sup.stop()
        assert trainer.terminated
        assert sum("collect_diagnostics.py" in cmd[1] for cmd, _ in sp.procs) == 1

    def test_stop_never_raises(self):
        sup, sp, _ = self.build()
        sup.start()
        sup.nn[0].stop = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        sup.diagnostics.run_blocking = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        sup.stop()

    def test_the_banner_describes_what_is_running(self):
        sup, _, _ = self.build()
        text = "\n".join(sup.describe())
        assert "NN trainer   : ON" in text and "Diagnostics  : ON" in text


class TestParentWatch:
    def test_own_pid_is_alive_and_nonsense_is_not(self):
        assert pid_alive(__import__("os").getpid()) is True
        assert pid_alive(0) is False and pid_alive(-5) is False

    def test_a_finished_process_is_not_alive(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        assert pid_alive(proc.pid) is False

    def test_the_watcher_sets_stop_when_the_parent_disappears(self):
        stop = threading.Event()
        calls = {"n": 0}

        def alive(pid):
            calls["n"] += 1
            return calls["n"] < 3

        thread = start_parent_watch(123, stop, interval_s=0.01, alive=alive)
        assert stop.wait(2.0) is True
        thread.join(1.0)

    def test_the_watcher_leaves_a_live_parent_alone(self):
        stop = threading.Event()
        start_parent_watch(123, stop, interval_s=0.01, alive=lambda pid: True)
        assert stop.wait(0.1) is False
        stop.set()

    def test_no_pid_means_no_watcher(self):
        assert start_parent_watch(0, threading.Event()) is None


class TestRunPyWiring:
    """run.py's __main__ block is not importable, so the wiring is checked in the source."""

    def setup_method(self):
        self.src = (Path(__file__).resolve().parents[2] / "run.py").read_text(encoding="utf-8")

    def test_the_supervisor_is_built_started_ticked_and_stopped(self):
        for needle in ("_build_helpers(", "helpers.start()", "helpers.tick()", "helpers.stop()"):
            assert needle in self.src

    def test_helpers_are_stopped_before_mt5_shuts_down(self):
        assert self.src.index("helpers.stop()") < self.src.rindex("mt5.shutdown()")

    def test_the_import_is_defensive(self):
        assert "_build_helpers = None" in self.src
