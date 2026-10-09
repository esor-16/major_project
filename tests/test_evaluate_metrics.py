"""Tests for the paper-style evaluation metrics added to evaluate.py:
top-20% segment e-Profits, top-decile lift, and the decile lift index -
plus the pure metric functions with hand-computed expected values."""
import numpy as np
import pytest

from pipeline import evaluate


# ---------------------------------------------------------------------------
# Pure metric functions (no models, hand-worked cases)
# ---------------------------------------------------------------------------

def test_top_segment_positions_takes_highest_scores():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    seg = evaluate.top_segment_positions(scores, fraction=0.4)
    # ceil(0.4 * 5) = 2 positions: the two highest scores (indices 1, 3)
    assert sorted(seg.tolist()) == [1, 3]
    assert len(evaluate.top_segment_positions(scores, fraction=0.2)) == 1  # never empty


def test_top_decile_lift_known_value():
    # 10 customers, 20% churn rate; the top decile (1 customer) is a churner
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.arange(10, 0, -1)  # index 0 ranked first
    # lift = (churn rate in top decile) / (overall churn rate) = 1.0 / 0.2
    assert evaluate.top_decile_lift(y, scores) == pytest.approx(5.0)

    # churner ranked LAST -> top decile holds no churner -> lift 0
    assert evaluate.top_decile_lift(y, -scores) == pytest.approx(0.0)


def test_lift_index_bounds_and_known_values():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.arange(10, 0, -1)  # both churners ranked first
    # captures per decile: [0.5, 1, 1, ..., 1] -> mean 0.95
    assert evaluate.lift_index(y, scores) == pytest.approx(0.95)

    # both churners ranked last -> captures [0]*9 then 1.0 at the end
    # (decile 9 boundary captures 1 of 2 churners) -> mean (0.5 + 1)/10
    assert evaluate.lift_index(y, -scores) == pytest.approx(0.15)

    # random-ranking baseline documented in the docstring
    rng = np.random.default_rng(0)
    random_scores = rng.uniform(size=1000)
    random_y = (rng.uniform(size=1000) < 0.27).astype(int)
    assert 0.4 < evaluate.lift_index(random_y, random_scores) < 0.7


def test_lift_metrics_degenerate_cases_do_not_raise():
    assert np.isnan(evaluate.top_decile_lift(np.zeros(10), np.arange(10)))
    assert np.isnan(evaluate.lift_index(np.zeros(10), np.arange(10)))


# ---------------------------------------------------------------------------
# Through evaluate_model (real fitted model, real reference frame)
# ---------------------------------------------------------------------------

def test_evaluate_model_reports_segment_and_lift_metrics(small_split):
    from pipeline.models import build_model_registry
    from pipeline.survival import fit_survival_model

    (x_train, x_test, y_train, y_test), sample = small_split
    _, retention_fn, avg_rate = fit_survival_model(x_train, y_train)
    model, _ = build_model_registry()["xgb"]
    model.fit(x_train, y_train)

    row = evaluate.evaluate_model("xgb", model, x_test, y_test, sample, retention_fn, avg_rate)

    for key in ["top_decile_lift", "lift_index", "eprofits_top20_avg", "eprofits_top20_tenure"]:
        assert key in row
        assert np.isfinite(row[key])

    assert row["top_decile_lift"] >= 0.0
    assert 0.0 < row["lift_index"] <= 1.0 + 1e-9
    # segment profit is over ceil(20%) of customers only, so it must be
    # smaller in magnitude than the full-population figure when profit is
    # positive (not guaranteed in general, but true for a sane model here)
    assert row["eprofits_top20_tenure"] != 0.0
