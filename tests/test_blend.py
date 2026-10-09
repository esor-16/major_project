"""Tests for the prediction-level blend hybrid (blend.py): a weighted
average of two tuned models' churn probabilities, with the weight chosen
on out-of-fold TRAINING predictions only.

The key contracts:
- predict_proba is exactly the convex combination of the members',
- clone()/fit() works (the F1-threshold search refits through
  cross_val_predict, same as SubsetPipeline),
- weight selection scores candidate weights on training data via the
  caller's scorer and returns a grid point in [0, 1].
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from pipeline.blend import (
    DEFAULT_WEIGHT_GRID,
    ProbabilityBlend,
    build_blend_hybrid,
    select_blend_weight,
)


def _fitted_members(x, y):
    a = XGBClassifier(n_estimators=25, max_depth=2, random_state=0, eval_metric="logloss").fit(x, y)
    b = RandomForestClassifier(n_estimators=25, max_depth=2, random_state=0).fit(x, y)
    return a, b


def _f1_scorer(estimator, x, y):
    preds = (estimator.predict_proba(x)[:, 1] >= 0.5).astype(int)
    return f1_score(y, preds, zero_division=0)


def test_blend_predict_proba_is_exact_convex_combination(small_split):
    (x_train, x_test, y_train, _), _ = small_split
    a, b = _fitted_members(x_train, y_train)

    blend = ProbabilityBlend(estimator_a=a, estimator_b=b, weight=0.3)
    got = blend.predict_proba(x_test)
    expected = 0.3 * a.predict_proba(x_test) + 0.7 * b.predict_proba(x_test)

    assert np.allclose(got, expected)
    assert set(np.unique(blend.predict(x_test))) <= {0, 1}
    # classes_ readable on a blend built around already-fitted members
    # (fit() was never called on the wrapper itself)
    assert list(blend.classes_) == [0, 1]


def test_blend_clone_refits_both_members(small_split):
    """clone(blend) must yield an unfitted estimator that fits from scratch -
    that's what find_f1_optimal_threshold's cross_val_predict relies on."""
    (x_train, x_test, y_train, _), _ = small_split
    a, b = _fitted_members(x_train, y_train)
    blend = ProbabilityBlend(estimator_a=a, estimator_b=b, weight=0.5)

    refitted = clone(blend).fit(x_train, y_train)
    proba = refitted.predict_proba(x_test)
    assert proba.shape == (len(x_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_select_blend_weight_returns_grid_point_and_sweep(small_split):
    (x_train, _, y_train, _), _ = small_split
    a, b = _fitted_members(x_train, y_train)
    models = {"a": a, "b": b}

    weight, sweep = select_blend_weight(models, ["a", "b"], x_train, y_train, _f1_scorer, cv=2)

    assert weight in DEFAULT_WEIGHT_GRID
    assert set(sweep.columns) == {"weight", "cv_score"}
    assert len(sweep) == len(DEFAULT_WEIGHT_GRID)
    assert sweep["cv_score"].notna().all()
    # the returned weight must actually be the sweep's argmax
    assert weight == sweep.loc[sweep["cv_score"].idxmax(), "weight"]


def test_select_blend_weight_rejects_missing_members(small_split):
    (x_train, _, y_train, _), _ = small_split
    a, _ = _fitted_members(x_train, y_train)
    with pytest.raises(ValueError, match="were not trained"):
        select_blend_weight({"a": a}, ["a", "b"], x_train, y_train, _f1_scorer, cv=2)


def test_build_blend_hybrid_returns_fitted_blend(small_split):
    (x_train, x_test, y_train, _), _ = small_split
    a, b = _fitted_members(x_train, y_train)

    blend, weight, sweep = build_blend_hybrid(
        {"a": a, "b": b}, member_names=["a", "b"],
        x_train=x_train, y_train=y_train, scorer=_f1_scorer, cv=2,
    )

    assert isinstance(blend, ProbabilityBlend)
    assert 0.0 <= weight <= 1.0
    assert blend.predict_proba(x_test).shape == (len(x_test), 2)
    # members were re-fit on the full training frame by build_blend_hybrid
    assert hasattr(blend, "estimator_a_") and hasattr(blend, "estimator_b_")
