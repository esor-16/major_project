"""Covers synopsis FR7/NFR7/TC6: SHAP explanations are generated per
prediction (global + per-customer), for both a tree-based model (fast
TreeExplainer path) and EBM (model-agnostic fallback path).
"""
import numpy as np
import pandas as pd

from pipeline import explain
from pipeline.models import build_model_registry


def _fit(name, x_train, y_train):
    registry = build_model_registry()
    model, _ = registry[name]
    model.fit(x_train, y_train)
    return model


def test_pick_best_model_uses_requested_metric():
    results = pd.DataFrame(
        {"eprofits_tenure": [10, 30, 20], "roc_auc": [0.9, 0.5, 0.6]},
        index=["a", "b", "c"],
    )
    assert explain.pick_best_model(results, metric="eprofits_tenure") == "b"
    assert explain.pick_best_model(results, metric="roc_auc") == "a"


def test_tree_explainer_path(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    model = _fit("xgb", x_train, y_train)

    explainer = explain.build_explainer("xgb", model, background=x_train)
    explanation = explain.compute_shap_values(explainer, x_test, sample_size=20)

    assert explanation.values.ndim == 2
    assert explanation.values.shape[0] == 20
    assert explanation.values.shape[1] == x_train.shape[1]


def test_model_agnostic_explainer_path(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    model = _fit("ebm", x_train, y_train)

    explainer = explain.build_explainer("ebm", model, background=x_train, background_size=20)
    explanation = explain.compute_shap_values(explainer, x_test, sample_size=15)

    assert explanation.values.shape == (15, x_train.shape[1])


def test_global_feature_importance_is_sorted_descending(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    model = _fit("xgb", x_train, y_train)
    explainer = explain.build_explainer("xgb", model, background=x_train)
    explanation = explain.compute_shap_values(explainer, x_test, sample_size=20)

    importance = explain.global_feature_importance(explanation)
    assert list(importance.columns) == ["feature", "mean_abs_shap"]
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert set(importance["feature"]) == set(x_train.columns)


def test_explain_customer_returns_signed_top_features(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    model = _fit("xgb", x_train, y_train)
    explainer = explain.build_explainer("xgb", model, background=x_train)
    explanation = explain.compute_shap_values(explainer, x_test, sample_size=20)

    row = explain.explain_customer(explanation, 0, top_n=3)
    assert len(row) == 3
    assert set(row["direction"]) <= {"toward churn", "away from churn"}
    # sorted by |shap_value| descending
    abs_vals = row["shap_value"].abs().to_numpy()
    assert np.all(abs_vals[:-1] >= abs_vals[1:])
