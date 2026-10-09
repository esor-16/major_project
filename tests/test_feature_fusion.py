"""Covers the feature-fusion hybrid mechanics: the three aggregators, the
SubsetPipeline full-width wrapper, donor rank agreement, and the
feature-count sweep. Uses hand-built importance tables where possible (fast,
exact expectations) and the real `small_split` fixture only where a fitted
model / DataFrame is genuinely needed.

Regression guard: donor SHAP must come from TRAINING data, never x_test -
selecting features from test-set SHAP and then scoring on that same test set
would be leakage (see feature_fusion.py module docstring).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pipeline.feature_fusion import (
    AGGREGATORS,
    SubsetPipeline,
    donor_importances,
    fuse_rankings,
    rank_agreement,
    select_features,
    sweep_feature_count,
)


# ---------------------------------------------------------------------------
# Aggregators / fuse_rankings, on a hand-built table (no model fitting)
# ---------------------------------------------------------------------------

def _hand_built_importances():
    return {
        "model_a": pd.Series({"f1": 1.0, "f2": 0.4, "f3": 0.1}),
        "model_b": pd.Series({"f1": 1.0, "f2": 0.8, "f3": 0.9}),
    }


def test_union_is_elementwise_max():
    fused = fuse_rankings(_hand_built_importances())
    row = fused.set_index("feature")
    assert row.loc["f2", "union"] == pytest.approx(0.8)
    assert row.loc["f3", "union"] == pytest.approx(0.9)


def test_intersection_is_elementwise_min():
    fused = fuse_rankings(_hand_built_importances())
    row = fused.set_index("feature")
    assert row.loc["f2", "intersection"] == pytest.approx(0.4)
    assert row.loc["f3", "intersection"] == pytest.approx(0.1)


def test_rank_fusion_is_elementwise_mean():
    fused = fuse_rankings(_hand_built_importances())
    row = fused.set_index("feature")
    assert row.loc["f2", "rank_fusion"] == pytest.approx(0.6)
    assert row.loc["f3", "rank_fusion"] == pytest.approx(0.5)


def test_normalization_maps_top_feature_to_one():
    fused = fuse_rankings(_hand_built_importances())
    row = fused.set_index("feature")
    assert row.loc["f1", "model_a_norm"] == pytest.approx(1.0)
    assert row.loc["f1", "model_b_norm"] == pytest.approx(1.0)


def test_normalization_is_scale_invariant():
    """Doubling one donor's raw SHAP magnitudes shouldn't change the fused
    ranking - only relative importance within a donor matters."""
    base = _hand_built_importances()
    scaled = {"model_a": base["model_a"] * 100, "model_b": base["model_b"]}

    fused_base = fuse_rankings(base).set_index("feature")["rank_fusion"]
    fused_scaled = fuse_rankings(scaled).set_index("feature")["rank_fusion"]

    pd.testing.assert_series_equal(fused_base.sort_index(), fused_scaled.sort_index())


def test_feature_absent_from_one_donor_scores_zero_there_not_nan():
    importances = {
        "model_a": pd.Series({"f1": 1.0, "f2": 0.5}),
        "model_b": pd.Series({"f1": 1.0, "f3": 0.7}),  # no f2, has f3 instead
    }
    fused = fuse_rankings(importances).set_index("feature")
    assert fused.loc["f2", "model_b_norm"] == 0.0
    assert fused.loc["f3", "model_a_norm"] == 0.0
    assert not fused.isna().any().any()


def test_fuse_rankings_rejects_wrong_number_of_donors():
    with pytest.raises(ValueError):
        fuse_rankings({"only_one": pd.Series({"f1": 1.0})})
    with pytest.raises(ValueError):
        fuse_rankings({"a": pd.Series({"f1": 1.0}), "b": pd.Series({"f1": 1.0}), "c": pd.Series({"f1": 1.0})})


# ---------------------------------------------------------------------------
# select_features
# ---------------------------------------------------------------------------

def test_select_features_returns_exactly_m_columns_present_in_input():
    fused = fuse_rankings(_hand_built_importances())
    top2 = select_features(fused, "rank_fusion", 2)
    assert len(top2) == 2
    assert set(top2).issubset(set(fused["feature"]))
    assert top2[0] == "f1"  # f1 dominates both donors


def test_select_features_rejects_unknown_rule():
    fused = fuse_rankings(_hand_built_importances())
    with pytest.raises(ValueError):
        select_features(fused, "not_a_real_rule", 2)


# ---------------------------------------------------------------------------
# rank_agreement
# ---------------------------------------------------------------------------

def test_rank_agreement_perfect_for_identical_rankings():
    importances = {"a": pd.Series({"f1": 3.0, "f2": 2.0, "f3": 1.0}),
                    "b": pd.Series({"f1": 30.0, "f2": 20.0, "f3": 10.0})}
    assert rank_agreement(importances) == pytest.approx(1.0)


def test_rank_agreement_negative_for_reversed_rankings():
    importances = {"a": pd.Series({"f1": 3.0, "f2": 2.0, "f3": 1.0}),
                    "b": pd.Series({"f1": 1.0, "f2": 2.0, "f3": 3.0})}
    assert rank_agreement(importances) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# donor_importances - regression guard against test-set leakage
# ---------------------------------------------------------------------------

def test_donor_importances_uses_the_frame_it_is_given_not_some_other_split(small_split):
    """`donor_importances` must compute SHAP over exactly the frame passed
    to it. This is the leakage guard: callers (run.py) are responsible for
    passing x_train, never x_test: this test locks in that donor_importances
    itself has no hidden reference to a different split.
    """
    from pipeline.models import build_model_registry

    (x_train, x_test, y_train, y_test), _ = small_split
    registry = build_model_registry()
    fitted = {}
    for name in ["random_forest", "lightgbm"]:
        model, _ = registry[name]
        model.fit(x_train, y_train)
        fitted[name] = model

    importances = donor_importances(fitted, x_train, donor_names=["random_forest", "lightgbm"], sample_size=len(x_train))

    assert set(importances.keys()) == {"random_forest", "lightgbm"}
    for series in importances.values():
        assert set(series.index).issubset(set(x_train.columns))
        assert (series >= 0).all()  # mean |SHAP| is non-negative by construction


def test_donor_importances_rejects_untrained_donor(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    with pytest.raises(ValueError):
        donor_importances({"random_forest": object()}, x_train, donor_names=["random_forest", "lightgbm"])


# ---------------------------------------------------------------------------
# SubsetPipeline
# ---------------------------------------------------------------------------

def test_subset_pipeline_fit_predict_on_full_width_frame(small_split):
    from sklearn.ensemble import RandomForestClassifier

    (x_train, x_test, y_train, y_test), _ = small_split
    cols = list(x_train.columns[:3])

    pipe = SubsetPipeline(estimator=RandomForestClassifier(n_estimators=10, random_state=0), features=cols)
    pipe.fit(x_train, y_train)

    proba = pipe.predict_proba(x_test)  # FULL-width frame in, not just `cols`
    assert proba.shape == (len(x_test), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert list(pipe.classes_) == sorted(y_train.unique())


def test_subset_pipeline_wraps_an_already_fitted_estimator_without_refitting(small_split):
    """The real usage in run.py: GASearchCV already fit best_estimator_ on
    the subsetted matrix, and SubsetPipeline is constructed around it
    without a second .fit() call."""
    from sklearn.ensemble import RandomForestClassifier

    (x_train, x_test, y_train, y_test), _ = small_split
    cols = list(x_train.columns[:3])

    inner = RandomForestClassifier(n_estimators=10, random_state=0)
    inner.fit(x_train[cols], y_train)

    pipe = SubsetPipeline(estimator=inner, features=cols, all_columns=list(x_train.columns))
    proba = pipe.predict_proba(x_test)
    assert proba.shape == (len(x_test), 2)
    assert list(pipe.classes_) == list(inner.classes_)


def test_subset_pipeline_survives_clone_and_cross_val_predict(small_split):
    """Required for evaluate.find_f1_optimal_threshold, which runs the
    estimator through cross_val_predict (clone + refit per fold)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

    (x_train, x_test, y_train, y_test), _ = small_split
    cols = list(x_train.columns[:3])

    pipe = SubsetPipeline(estimator=RandomForestClassifier(n_estimators=10, random_state=0), features=cols)

    cloned = clone(pipe)
    assert cloned.features == cols
    assert not hasattr(cloned, "estimator_")  # clone() must not carry over a fitted state

    probs = cross_val_predict(pipe, x_train, y_train, cv=2, method="predict_proba")
    assert probs.shape == (len(x_train), 2)


def test_subset_pipeline_accepts_ndarray_input_via_all_columns(small_split):
    """SHAP's model-agnostic Explainer strips column names and passes numpy
    arrays; all_columns (captured at fit time) must let predict_proba
    reconstruct a DataFrame and still subset correctly."""
    from sklearn.ensemble import RandomForestClassifier

    (x_train, x_test, y_train, y_test), _ = small_split
    cols = list(x_train.columns[:3])

    pipe = SubsetPipeline(estimator=RandomForestClassifier(n_estimators=10, random_state=0), features=cols)
    pipe.fit(x_train, y_train)

    proba_df = pipe.predict_proba(x_test)
    proba_arr = pipe.predict_proba(x_test.to_numpy())
    np.testing.assert_allclose(proba_df, proba_arr)


def test_subset_pipeline_raises_helpfully_without_all_columns(small_split):
    from sklearn.ensemble import RandomForestClassifier

    (x_train, x_test, y_train, y_test), _ = small_split
    cols = list(x_train.columns[:3])
    pipe = SubsetPipeline(estimator=RandomForestClassifier(n_estimators=10, random_state=0), features=cols)
    # Not fitted (so all_columns was never captured) and given a bare array.
    with pytest.raises(TypeError):
        pipe.predict_proba(x_test.to_numpy())


# ---------------------------------------------------------------------------
# sweep_feature_count
# ---------------------------------------------------------------------------

def test_sweep_feature_count_returns_valid_m_and_covers_requested_sizes(small_split):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer, roc_auc_score

    (x_train, x_test, y_train, y_test), _ = small_split
    importances = {
        "a": pd.Series(np.random.RandomState(0).rand(len(x_train.columns)), index=x_train.columns),
        "b": pd.Series(np.random.RandomState(1).rand(len(x_train.columns)), index=x_train.columns),
    }
    fused = fuse_rankings(importances)

    scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    recipient = LogisticRegression(max_iter=200)

    best_m, sweep = sweep_feature_count(
        recipient, fused, "rank_fusion", x_train, y_train, scorer, sizes=(2, 4), cv=2,
    )

    assert 1 <= best_m <= len(x_train.columns)
    assert set(sweep["n_features"]) >= {2, 4, len(x_train.columns)}  # full-width candidate always included
    assert (sweep["rule"] == "rank_fusion").all()
    for col in ["cv_score", "cv_std"]:
        assert col in sweep.columns
    expected_m = int(sweep.loc[sweep["cv_score"].idxmax(), "n_features"])
    assert best_m == expected_m


def test_sweep_feature_count_ranks_by_whatever_scorer_it_is_given(small_split):
    """sweep_feature_count has no opinion of its own about eProfits vs. F1 -
    it just picks the m that maximizes cross_val_score under `scorer`. Feed
    it a composite scorer (eprofits.make_composite_scorer_tenure) here to
    confirm that delegation actually works end to end, since that's how
    run.py uses it: the same scorer drives both the sweep and the
    recipient's own GA tuning afterwards.
    """
    from sklearn.linear_model import LogisticRegression

    from pipeline.eprofits import make_composite_scorer_tenure
    from pipeline.survival import fit_survival_model

    (x_train, x_test, y_train, y_test), sample = small_split
    importances = {
        "a": pd.Series(np.random.RandomState(0).rand(len(x_train.columns)), index=x_train.columns),
        "b": pd.Series(np.random.RandomState(1).rand(len(x_train.columns)), index=x_train.columns),
    }
    fused = fuse_rankings(importances)

    _, retention_fn, _ = fit_survival_model(x_train, y_train)
    scorer = make_composite_scorer_tenure(reference_df=sample, retention_fn=retention_fn, f1_weight=0.5)
    recipient = LogisticRegression(max_iter=200)

    best_m, sweep = sweep_feature_count(
        recipient, fused, "rank_fusion", x_train, y_train, scorer, sizes=(2, 4), cv=2,
    )

    assert 1 <= best_m <= len(x_train.columns)
    assert sweep["cv_score"].notna().any()
