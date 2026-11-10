"""Dual-stream deep network: architecture, presets, training, artifact, predictor."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from core.ai.deep_layout import (  # noqa: E402
    ARCH_DUAL, N_SEQ, build_model_features, feature_names, latest_model_window, make_excursion,
)
from core.ai.deep_nn import PRESETS, DualStreamNet, build_dual, n_parameters  # noqa: E402
from core.ai.location_features import N_STRUCT  # noqa: E402
from core.ai.nn_model import (  # noqa: E402
    TrainConfig, ensemble_proba, fit_artifact, load_artifact, members_from_artifact,
    save_artifact,
)
from core.ai.nn_predictor import ModelPredictor  # noqa: E402

N_FEAT = N_SEQ + N_STRUCT


def _bars(n=900, seed=11):
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 1.0, n))
    idx = pd.date_range("2025-03-03", periods=n, freq="5min")
    return pd.DataFrame({"open": close, "high": close + rng.random(n),
                         "low": close - rng.random(n), "close": close,
                         "tick_volume": 100.0 + rng.random(n)}, index=idx)


def test_forward_shapes_and_excursion_is_non_negative():
    net = build_dual(N_FEAT, "small", 0.1).eval()
    x = torch.randn(4, 16, N_FEAT)
    logits, exc = net.forward_full(x)
    assert logits.shape == (4, 3) and exc.shape == (4,)
    assert (exc >= 0).all()
    assert torch.equal(net(x), logits)               # forward() stays logits-only


def test_structural_branch_actually_matters():
    """Changing only the location columns of the LAST step changes the output."""
    net = build_dual(N_FEAT, "small", 0.0).eval()
    x = torch.randn(2, 16, N_FEAT)
    y = x.clone()
    y[:, -1, N_SEQ:] += 3.0
    assert not torch.allclose(net(x), net(y))


def test_older_structural_steps_are_ignored_by_design():
    net = build_dual(N_FEAT, "small", 0.0).eval()
    x = torch.randn(2, 16, N_FEAT)
    y = x.clone()
    y[:, :-1, N_SEQ:] += 3.0
    assert torch.allclose(net(x), net(y))


def test_presets_grow_and_large_matches_the_specified_widths():
    small, medium = (n_parameters(build_dual(N_FEAT, p, 0.1)) for p in ("small", "medium"))
    large_net = build_dual(N_FEAT, "large", 0.1)
    assert small < medium < n_parameters(large_net)
    assert n_parameters(large_net) > 5_000_000
    p = PRESETS["large"]
    assert (p.lstm1, p.lstm2, p.heads, p.fuse) == (256, 512, 8, (512, 256, 128))
    assert large_net.attn.num_heads == 8
    out = large_net.eval()(torch.randn(1, 8, N_FEAT))
    assert out.shape == (1, 3)


def test_unknown_preset_and_missing_struct_columns_are_refused():
    with pytest.raises(ValueError):
        DualStreamNet(N_SEQ, N_STRUCT, "enormous")
    with pytest.raises(ValueError):
        build_dual(N_SEQ, "small", 0.1)


def test_feature_assembly_layout():
    df = _bars(300)
    m = build_model_features(df, ARCH_DUAL, 8)
    assert m.shape == (300, N_FEAT) and len(feature_names(ARCH_DUAL)) == N_FEAT
    w = latest_model_window(df, 32, ARCH_DUAL)
    assert w.shape == (32, N_FEAT)
    assert np.allclose(w[:, N_SEQ:], w[-1, N_SEQ:])  # live: newest state broadcast


def test_excursion_target_is_nan_without_a_full_horizon_and_never_negative():
    df = _bars(200)
    e = make_excursion(df, horizon=24)
    assert np.isnan(e[-24:]).all()
    assert np.nanmin(e) >= 0.0 and np.nanmax(e) <= 6.0


def test_end_to_end_train_save_load_predict(tmp_path):
    df = _bars(900)
    cfg = TrainConfig(arch="dual", preset="small", window=16, epochs=3, batch=64,
                      horizon=12, threshold=0.4, val_fraction=0.25)
    art = fit_artifact(df, cfg)
    assert art["config"]["arch"] == "dual"
    assert art["feature_names"] == feature_names(ARCH_DUAL)
    assert art["members"][0]["state_dict"]
    path = tmp_path / "deep.pt"
    save_artifact(str(path), art)                    # weights_only-safe types only
    loaded = load_artifact(str(path))
    members = members_from_artifact(loaded)
    assert isinstance(members[0]["model"], DualStreamNet)
    x = np.random.default_rng(0).normal(size=(3, 16, N_FEAT))
    p = ensemble_proba(members, x)
    assert p.shape == (3, 3) and np.allclose(p.sum(axis=1), 1.0)

    pred = ModelPredictor(str(path), min_confidence=0.0, min_oos_edge=0.0)
    sig = pred.predict_signal(df)                    # edge unmeasured -> must stay neutral
    assert sig.direction == 0 and "never measured" in sig.reason


def test_predictor_uses_dual_window_when_edge_is_measured(tmp_path):
    df = _bars(900)
    cfg = TrainConfig(arch="dual", preset="small", window=16, epochs=2, batch=64,
                      horizon=12, threshold=0.0, val_fraction=0.25)
    art = fit_artifact(df, cfg)
    art["oos"] = {"edge": 0.2, "trades": 100}
    path = tmp_path / "deep.pt"
    save_artifact(str(path), art)
    sig = ModelPredictor(str(path), min_confidence=0.0, min_oos_edge=0.0).predict_signal(df)
    assert sig.direction in (-1, 0, 1)
    assert abs(sum(sig.probs) - 1.0) < 1e-6
    assert "predictor error" not in sig.reason


def test_legacy_gru_config_still_loads(tmp_path):
    """An artifact saved before ``arch`` existed has no such key and must load as GRU."""
    df = _bars(700)
    cfg = TrainConfig(window=16, epochs=2, batch=64, horizon=12, val_fraction=0.25)
    art = fit_artifact(df, cfg)
    for k in ("arch", "preset", "struct_stride", "aux_weight"):
        art["config"].pop(k, None)
    members = members_from_artifact(art)
    assert type(members[0]["model"]).__name__ == "SeqClassifier"
