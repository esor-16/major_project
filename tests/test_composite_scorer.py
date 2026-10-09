"""Tests for make_composite_scorer_tenure: the single scorer that blends
eProfits and F1 into one scalar, so GA tuning balances profit AND
classification quality for every model (baselines and fusion hybrids
alike) instead of optimizing eProfits alone.

Numbers below are worked by hand against the actual formula (profit_margin
0.3, cpo 0.1, cost_of_contact (0, 0.3), all defaults) for customer_value=100,
retention_rate=0.8: clv=150, cost_per_offer=15, contact_cost=4.5, so a
correctly-flagged churner nets 130.5 and a false positive costs -19.5.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.eprofits import make_composite_scorer_tenure


class _FixedProbaEstimator:
    """Stub estimator returning pre-set churn probabilities, for tests that
    need exact control over which customers get flagged."""

    def __init__(self, churn_proba):
        self._proba = np.asarray(churn_proba, dtype=float)

    def predict_proba(self, x):
        return np.column_stack([1 - self._proba, self._proba])


def _constant_retention_fn(rate: float):
    def fn(t, x=None):
        # `x` is the per-customer covariate frame conditional_retention
        # passes through for the CoxPH path; a constant retention curve
        # ignores it, exactly like the Kaplan-Meier baseline does.
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        vals = np.full_like(t_arr, rate)
        return float(vals[0]) if np.isscalar(t) else vals

    return fn


@pytest.fixture
def reference_df():
    return pd.DataFrame({"customer_value": [100.0] * 4, "tenure": [10.0] * 4})


@pytest.fixture
def y(reference_df):
    return pd.Series([1, 1, 0, 0], index=reference_df.index)


def test_rejects_out_of_range_f1_weight(reference_df):
    with pytest.raises(ValueError):
        make_composite_scorer_tenure(reference_df=reference_df, retention_fn=_constant_retention_fn(0.8), f1_weight=1.5)
    with pytest.raises(ValueError):
        make_composite_scorer_tenure(reference_df=reference_df, retention_fn=_constant_retention_fn(0.8), f1_weight=-0.1)


def test_perfect_classifier_scores_one_regardless_of_f1_weight(reference_df, y):
    """Flagging exactly the true churners is simultaneously the eProfits
    ceiling (eprofits_norm=1.0) and a perfect F1 (1.0), so the blend should
    land at 1.0 no matter how eProfits and F1 are weighted against each
    other."""
    estimator = _FixedProbaEstimator([1.0, 1.0, 0.0, 0.0])
    retention_fn = _constant_retention_fn(0.8)

    for f1_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        scorer = make_composite_scorer_tenure(reference_df=reference_df, retention_fn=retention_fn, f1_weight=f1_weight)
        assert scorer(estimator, reference_df, y) == pytest.approx(1.0, abs=1e-9)


def test_f1_weight_shifts_preference_toward_classification_quality(reference_df, y):
    """flags_everyone earns real profit (both true churners still get
    flagged) but drags in two costly false positives, hurting F1 far more
    than it hurts eProfits (each false positive is a small cost against a
    150 CLV, but a direct precision hit against a small denominator in F1).
    Raising f1_weight from 0 to 1 should widen the gap between the perfect
    predictor and this one, not narrow it.
    """
    flags_everyone = _FixedProbaEstimator([1.0, 1.0, 1.0, 1.0])
    flags_true_churners_only = _FixedProbaEstimator([1.0, 1.0, 0.0, 0.0])
    retention_fn = _constant_retention_fn(0.8)

    scorer_eprofits_only = make_composite_scorer_tenure(reference_df=reference_df, retention_fn=retention_fn, f1_weight=0.0)
    scorer_f1_only = make_composite_scorer_tenure(reference_df=reference_df, retention_fn=retention_fn, f1_weight=1.0)

    gap_eprofits_only = scorer_eprofits_only(flags_true_churners_only, reference_df, y) - scorer_eprofits_only(
        flags_everyone, reference_df, y
    )
    gap_f1_only = scorer_f1_only(flags_true_churners_only, reference_df, y) - scorer_f1_only(flags_everyone, reference_df, y)

    assert gap_eprofits_only > 0  # perfect predictor still wins even under eProfits alone
    assert gap_f1_only > 0
    assert gap_f1_only > gap_eprofits_only  # but F1 punishes the false positives harder


def test_no_true_churners_in_batch_does_not_raise(reference_df):
    """Edge case: a CV fold with zero true churners has a zero-profit
    ceiling (division by zero) - the scorer should fall back to a neutral
    eprofits_norm rather than raising."""
    y_all_non_churn = pd.Series([0, 0, 0, 0], index=reference_df.index)
    estimator = _FixedProbaEstimator([0.0, 0.0, 0.0, 0.0])
    scorer = make_composite_scorer_tenure(reference_df=reference_df, retention_fn=_constant_retention_fn(0.8), f1_weight=0.5)

    score = scorer(estimator, reference_df, y_all_non_churn)
    assert np.isfinite(score)
