"""
tests/unit/test_nn_predictor_neutral.py — the predictor must fail neutral and
must import without torch. These run whether or not torch is installed.
"""

from __future__ import annotations

import sys

import pytest

from core.ai.nn_predictor import ModelPredictor
from tests.unit.test_nn_features import random_walk


def test_missing_model_file_is_no_trade(tmp_path):
    p = ModelPredictor(str(tmp_path / "absent.pt"))
    sig = p.predict_signal(random_walk())
    assert sig.direction == 0 and "no model file" in sig.reason
    assert p.status.startswith("no model file")


def test_unreadable_model_file_is_no_trade_not_a_crash(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a torch file")
    sig = ModelPredictor(str(bad)).predict_signal(random_walk())
    assert sig.direction == 0


def test_torch_missing_is_no_trade(tmp_path, monkeypatch):
    f = tmp_path / "m.pt"
    f.write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "torch", None)       # import torch -> ImportError
    monkeypatch.delitem(sys.modules, "core.ai.nn_model", raising=False)
    sig = ModelPredictor(str(f)).predict_signal(random_walk())
    assert sig.direction == 0 and "torch unavailable" in sig.reason


def test_bad_input_never_raises(tmp_path):
    sig = ModelPredictor(str(tmp_path / "absent.pt")).predict_signal(None)
    assert sig.direction == 0
