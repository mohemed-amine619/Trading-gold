# Prompt — Put every strategy into the analysis (Trading-gold)

Copy everything below the line into your coding agent (Claude Code, Codex, ...) opened at the repo root.

---

You are working in the **Trading-gold** repository (Python MT5 bot for XAUUSDm, M15 entry / H1 trend / H4 macro).
Read `AGENTS.md`, `docs/STRATEGY_LIBRARY.md`, `docs/ARCHITECTURE.md` and `docs/AI_ENGINE.md` before changing anything, and follow them.

## Goal

Today the 151-entry strategy catalogue (`core/strategy/library/`: 151 named entries → 118 distinct generators via aliases, 14 entries unavailable) is used in **one** place only: the scalp vote in `run.py`
(`ScalpEngine.evaluate` → `scalp_support` → up to `SCALP_WEIGHT` points in `_score_direction`), and only for the strategies listed in `SCALP_STRATEGIES`
that pass the proof gate. A year-long evaluation on XAUUSDm M15 (2025-11-25 → 2026-11-25, results in `data/history/scalp_eval_M15_2026-11-25.csv`)
found **no strategy significant after multiple-testing correction** (best train z = 2.17 vs 3.52 required; train/holdout z correlation 0.22).

Make **every implemented strategy part of the analysis** the bot (and `scripts/predict_direction.py`) produces, in three layers, **without** letting
unproven strategies add score points:

1. **Strategy panel (observability)** — every strategy's current call, on every analysed timeframe, with its measured record.
2. **Consensus features (inputs to the models)** — the strategy signals become features for the AI / NN, so the models decide what they are worth,
   validated out-of-sample by the existing honest gates.
3. **Proven-strategy vote (unchanged rules)** — the existing scalp vote keeps its proof gate; it may now run on the full catalogue in analysis mode
   while only `SCALP_STRATEGIES` may vote.

## Hard rules (do not break)

- **No look-ahead.** Every generator is causal: `sig[t]` uses bars `<= t` only. Swing pivots are published `k` bars late, session ranges only after the
  session closes, NY time computed by hand (no tzdata). Any new code must keep this. Add a test: truncating the frame at `t` must not change `sig[:t+1]`.
- **No score points without proof.** Layer 1 and Layer 2 never add points to `_score_direction` directly. Only Layer 3 (proof gate,
  `required_z` Bonferroni over the number of strategies *tested*, `SCALP_MIN_EDGE`, `SCALP_MIN_CALLS`) can, and only up to `SCALP_WEIGHT`.
  The 100-point budget stays 100.
- **Martingale / grid stay off by default** (`SIZING_PROGRESSION=off`, `GRID_MODE=off`). Do not wire progression into the analysis.
- **Never crash the trading loop.** Each generator runs inside try/except (as `ScalpEngine` already does); a failing strategy is reported as
  `error` in the panel and skipped.
- **Performance budget:** the full catalogue on 4500 M15 bars must stay under ~1.5 s per cycle on the Windows PC. Reuse the existing OHLC-hash cache in
  `ScalpEngine`; compute signals once per new bar, not per tick. Measure and print the timing in a test.
- **Never commit credentials** (`.env*` stay untracked). Paper mode only for any manual run.
- **Do not run a bare `pytest`** — run the relevant test files; the suite is large.

## Layer 1 — Strategy panel

Create `core/strategy/panel.py`:

```python
@dataclass
class StrategyReading:
    name: str
    category: str            # from library CATEGORIES
    timeframe: str           # "M15" | "H1" | "H4"
    signal: int              # -1, 0, +1 on the last CLOSED bar
    bars_since_signal: Optional[int]
    calls: int               # independent (thinned by horizon) calls in the check window
    edge: Optional[float]    # (wins - losses) / calls, barrier rule = ScalpEngine.outcomes
    z: Optional[float]
    proven: bool             # passed the same gate as the live vote
    available: bool          # False for the 14 catalogue entries without data (order flow, options...)
    error: Optional[str]

@dataclass
class StrategyPanel:
    timeframe: str
    readings: List[StrategyReading]
    long_count: int; short_count: int; flat_count: int
    net_breadth: float        # (long - short) / max(1, long + short), all strategies
    weighted_breadth: float   # same, each vote weighted by max(0, z) — 0 when nothing has z > 0
    by_category: Dict[str, Dict[str, int]]   # category -> {"long": n, "short": n}
    required_z: float
    def to_dict(self) -> dict: ...

def build_panel(df: pd.DataFrame, timeframe: str, cfg: ScalpConfig) -> StrategyPanel
```

- Use `ScalpEngine(ScalpConfig(strategies=("all",), ...))` — one engine per timeframe, cached on the bot instance.
- The last bar of a live frame may be forming: compute the reading on the **last closed bar** (follow how `run.py` already slices `m15p`).
- Wire it in:
  - `core/prediction/direction.py`: add `strategy_panel: Optional[dict]` to `DirectionPrediction` and `MultiTimeframePrediction.to_dict()`;
    `predict_direction(..., include_strategies: bool = True)`.
  - `scripts/predict_direction.py`: add `--strategies {all,classic,<group>,none}` and print a compact table (top 15 by |z| plus breadth per category);
    `--json` includes the full panel.
  - `run.py`: once per new M15 bar, build the M15/H1/H4 panels, log a one-line summary
    (`strategies M15: 23 long / 11 short / 103 flat, breadth +0.35, proven: none`) and write `data/state/strategy_panel_<symbol>.json`
    next to the existing `scalp_report_<symbol>.json`.
  - Dashboard (`dashboard.html` + whatever API already serves `scalp_report`): add a read-only "Strategies" section listing the panel. No write endpoints.

## Layer 2 — Strategy signals as model features

- Add `core/strategy/features.py` with `strategy_feature_matrix(df, cfg, names=None) -> pd.DataFrame` (index aligned to `df`):
  - one int8 column per available strategy `strat_<name>` in {-1,0,1},
  - `strat_breadth`, `strat_long_n`, `strat_short_n`, and per category `cat_<category>_net`,
  - optional decay columns `strat_<name>_age` = bars since last non-zero signal, capped at 48 (default off — dimension cost).
- Behind a config flag `AI_STRATEGY_FEATURES` (default `false`) and `NN_STRATEGY_FEATURES` (default `false`):
  - append the matrix to the AI feature builder used by the ensemble (find it from `core/ai/decision.py` / `provider.py`)
    and to `core/ai/nn_features.build_features` (and `deep_features.build_sequence_features` for the deep model).
  - Fit `FeatureScaler` on the training slice only; do not scale the categorical -1/0/1 columns or scale them consistently.
  - Feature selection, if any, must happen **inside** each walk-forward training fold (`walk_forward_splits`), never on the whole history.
- The existing honest gates (`AI_MIN_OOS_EDGE`, `NN_MIN_OOS_EDGE`, lower-95% LCB, promotion tolerance) decide whether the augmented model is used.
  Log the before/after OOS metrics on the same folds so the gain (or lack of it) is visible.
- Register the two flags in `run.py CONFIG`, `core/config/schema.py PARAMS` (bool, safe) and `docs/CONFIGURATION.md`; add both to `.env.example`
  with `false`.

## Layer 3 — Proven-strategy vote

- Keep `scalp_support` and the gate exactly as they are.
- Optimisation: when Layer 1 has already computed signals/outcomes for M15 on the same frame, reuse them for the vote instead of recomputing
  (share the engine cache, keyed on the OHLC hash + config).
- The vote's `required_z` must stay based on the number of strategies in `SCALP_STRATEGIES`; the panel's `proven` flag must use the z for
  **all** strategies (it tested all of them). Document the difference in the panel header.

## Evaluation script

Extend `scripts/evaluate_scalp.py` (or add `scripts/evaluate_strategy_features.py`) so that, on a CSV:

1. it splits off the last `--holdout-days` (default 91) as a holdout,
2. reports per strategy: thinned calls, edge, edge net of the CSV `spread` column (points × 0.001 / ATR for XAUUSDm), z — on train and holdout,
3. reports `corr(train z, holdout z)` and the z distribution (mean/sd) as a noise check,
4. trains the AI/NN with and without strategy features on the same walk-forward folds and prints OOS accuracy, AUC and edge for both.

It must run on `data/history/XAUUSDm_M15.csv` (23 625 bars) in under 10 minutes on the Linux box.

## Tests (add under `tests/unit/`)

- `test_strategy_panel.py`: panel on synthetic data has one reading per distinct generator (118 for `all`) plus one `available=False` reading per unavailable catalogue entry, counts sum correctly,
  breadth in [-1, 1], a raising generator becomes `error` without breaking the panel, `to_dict()` is JSON-serialisable.
- `test_strategy_causality.py`: for every available strategy, `sig` on `df.iloc[:t+1]` equals `sig_full[:t+1]` at 5 random cut points
  (use the existing `with_news_spikes` fixture data so event strategies fire).
- `test_strategy_features.py`: matrix aligned to index, no NaN after warm-up, dtypes int8 for signal columns, flags off ⇒ feature builders
  output identical to before (parity), flags on ⇒ extra columns present.
- `test_score_budget_unchanged.py`: with panel + features enabled, `_score_direction` gives identical scores to before for the same inputs when
  no strategy is proven (the panel must not leak points).
- Performance test: full catalogue on 4500 bars < 1.5 s (mark `slow` if needed).
- Update `test_library_wiring.py` / `test_evaluate_scalp_cli.py` for any new CLI flags.

Known pre-existing failures to leave alone (do not "fix" by deleting tests): 5 tests in `test_prediction_direction.py`
because `logger` is undefined at `core/prediction/direction.py:202` — you **may** fix that bug (define `logger = logging.getLogger(__name__)`)
since you are editing that file; say so in the summary.

## Docs

- `docs/STRATEGY_LIBRARY.md`: a section "Strategies in the analysis" explaining the three layers and that a panel signal is **information, not a
  trade signal**.
- `docs/AI_ENGINE.md`: the two feature flags and how to read the before/after OOS log.
- `AGENTS.md`: add the rule "panel and features never add score points; only the gated vote does".

## Deliverable

1. Code + tests + docs as above, in small commits (panel, features, wiring, evaluation, docs).
2. Run only the touched test files and paste the pass counts.
3. Run the evaluation script on `data/history/XAUUSDm_M15.csv` and paste the summary (top 15 strategies train vs holdout, noise check,
   AI/NN with vs without strategy features).
4. A short final note: what changed, which flags to set in `.env` to try it (paper mode), and an honest verdict on whether strategy features
   improved OOS metrics. If they did not, recommend leaving the flags off.
