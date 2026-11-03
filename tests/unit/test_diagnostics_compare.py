"""tests/unit/test_diagnostics_compare.py — A/B comparison of bot variants by magic number."""

from __future__ import annotations

import pytest

from core.diagnostics.compare import compare_variants, render, split_by_magic
from core.diagnostics.trades import Trade


def trade(pnl, magic):
    return Trade(position_id=int(abs(pnl) * 1000) + magic, symbol="XAUUSDm", direction=1, volume=0.01,
                 open_time="2026-11-03T10:00:00", close_time="2026-11-03T10:05:00", open_price=100.0,
                 close_price=100.0, net_pnl=pnl, exit_reason="sl", duration_s=300, magic=magic, comment="")


def group(pnls, magic):
    return [trade(p, magic) for p in pnls]


class TestSplit:
    def test_trades_are_split_by_magic_and_others_ignored(self):
        ts = group([1, 2], 1) + group([3], 2) + group([4], 9)
        out = split_by_magic(ts, [1, 2, 5])
        assert [t.net_pnl for t in out[1]] == [1, 2] and [t.net_pnl for t in out[2]] == [3] and out[5] == []


class TestCompare:
    def test_welch_z_hand_computed(self):
        # 1..5 twice: mean 3, sum of squared deviations 20, sample var 20/9, se^2 = 2/9;
        # B is the same shape shifted by +10. denom = sqrt(4/9) = 2/3, so z = 10 / (2/3) = 15
        a, b = group([1, 2, 3, 4, 5] * 2, 1), group([11, 12, 13, 14, 15] * 2, 2)
        r = compare_variants({"A": a, "B": b})
        s = r["variants"]["B"]
        assert s["diff_vs_baseline"] == 10.0
        assert s["z"] == 15.0
        assert "_se" not in s and "_mean" not in s
        assert "B is better than A" in r["verdict"]

    def test_identical_variants_are_not_distinguishable(self):
        a, b = group([2, -4] * 6, 1), group([2, -4] * 6, 2)
        r = compare_variants({"A": a, "B": b})
        assert r["variants"]["B"]["z"] == 0.0
        assert "Not distinguishable" in r["verdict"]

    def test_few_trades_never_crown_a_winner(self):
        r = compare_variants({"A": group([1, 2, 3], 1), "B": group([50, 60, 70], 2)})
        assert "Too few trades" in r["verdict"] and "A, B" in r["verdict"]

    def test_a_worse_variant_is_reported_as_worse(self):
        a, b = group([5, 6, 7, 8] * 3, 1), group([1, 2, 3, 4] * 3, 2)
        assert "B is worse than A" in compare_variants({"A": a, "B": b})["verdict"]

    def test_empty_and_single_trade_groups_do_not_crash_or_invent_numbers(self):
        r = compare_variants({"A": [], "B": group([3.0], 2)})
        assert r["variants"]["A"]["expectancy"] is None and r["variants"]["B"]["z"] is None
        assert "Too few trades" in r["verdict"]

    def test_needs_two_variants(self):
        with pytest.raises(ValueError):
            compare_variants({"A": group([1], 1)})

    def test_render_has_a_row_per_variant_and_the_verdict(self):
        r = compare_variants({"A": group([2, -4] * 6, 1), "B": group([2, -4] * 6, 2)})
        text = render(r)
        assert "| A |" in text and "| B |" in text and r["verdict"] in text
