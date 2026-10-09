"""SHAP-based explainability for the best-performing model(s) (synopsis
FR7/NFR7): model-agnostic feature-contribution explanations, both global
(which features drive churn overall) and per-customer (why this specific
customer was flagged), built on top of the six-model benchmark in
`models.py` / `evaluate.py` without retraining anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

# Models whose internal structure SHAP's fast, exact TreeExplainer supports.
TREE_MODEL_NAMES = {"xgb", "gradient_boosting", "random_forest", "lightgbm", "catboost"}


def pick_best_model(results: pd.DataFrame, metric: str = "eprofits_tenure") -> str:
    """Name of the top-ranked model in an `evaluate.evaluate_all` results table."""
    return results.sort_values(metric, ascending=False).index[0]


def _unwrap(fitted_search):
    """GASearchCV wraps the real estimator; SHAP needs the estimator itself."""
    return getattr(fitted_search, "best_estimator_", fitted_search)


def build_explainer(name: str, fitted_search, background: pd.DataFrame, background_size: int = 100, random_state: int = 42):
    """A SHAP explainer for one fitted model.

    Tree-based models (XGBoost/GB/RF/LightGBM/CatBoost) use the fast, exact
    TreeExplainer. EBM - and anything else added later - falls back to a
    model-agnostic Explainer over predict_proba, run against a small
    background sample so it stays within the pipeline's time budget (NFR3);
    EBM already ships its own glass-box explanations too, but routing it
    through SHAP here keeps one consistent explanation format across models.

    A `feature_fusion.SubsetPipeline` (a feature-fusion hybrid's tree model,
    trained on fewer columns) is unwrapped to its inner estimator and given
    the fast TreeExplainer restricted to the subset it was trained on,
    rather than falling into the slow agnostic branch below - explaining the
    winning hybrid shouldn't be the slowest explanation in the run.
    """
    estimator = _unwrap(fitted_search)

    # Local import: feature_fusion imports explain (donor_importances calls
    # build_explainer), so importing it at module load time would deadlock;
    # deferring to call time breaks the cycle.
    from .feature_fusion import SubsetPipeline

    if isinstance(estimator, SubsetPipeline):
        inner = estimator._fitted()
        inner_name = type(inner).__name__.lower()
        if any(key in inner_name for key in ("xgb", "gradientboosting", "randomforest", "lgbm", "catboost")):
            columns = list(estimator.features)
            explainer = shap.TreeExplainer(inner)
            explainer._fusion_columns = columns  # read back by compute_shap_values
            return explainer
        estimator = inner

    if name in TREE_MODEL_NAMES:
        return shap.TreeExplainer(estimator)

    if len(background) > background_size:
        background = shap.sample(background, background_size, random_state=random_state)

    columns = background.columns

    def predict_churn_proba(x):
        return estimator.predict_proba(pd.DataFrame(x, columns=columns))[:, 1]

    return shap.Explainer(predict_churn_proba, background)


def compute_shap_values(explainer, x: pd.DataFrame, sample_size: int | None = 200, random_state: int = 42):
    """SHAP values for up to `sample_size` rows of x (None = all rows).

    Explaining every test-set row can be expensive for non-tree models;
    capping the sample keeps NFR7 ("every displayed prediction has an
    explanation") true for whatever subset is actually shown/reported,
    without forcing a full-dataset permutation run over the whole test set.
    """
    if sample_size is not None and len(x) > sample_size:
        x = x.sample(sample_size, random_state=random_state)

    # A SubsetPipeline explainer (see build_explainer) was built for a
    # feature-fusion model's inner tree estimator, which only knows the
    # fused feature subset - restrict x to those columns before calling it,
    # or SHAP would hand the full-width test frame to a model trained on
    # fewer columns.
    fusion_columns = getattr(explainer, "_fusion_columns", None)
    if fusion_columns is not None:
        x = x[fusion_columns]

    explanation = explainer(x)

    # Some explainer/model combinations return one SHAP value per class
    # (values.ndim == 3); keep the positive (churn) class so downstream
    # shape is consistent regardless of which algorithm produced it.
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]

    return explanation


def global_feature_importance(explanation) -> pd.DataFrame:
    """Mean |SHAP value| per feature, ranked descending - the model's
    overall churn drivers, for cross-checking against Phase-1's
    feature-importance ranking (TC6).
    """
    importance = np.abs(explanation.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_customer(explanation, row_index: int, top_n: int = 5) -> pd.DataFrame:
    """Top contributing features for one customer's prediction, signed so
    the direction (pushes toward churn vs. away from it) is visible.
    """
    values = np.asarray(explanation.values[row_index])
    row = pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "value": np.asarray(explanation.data[row_index]),
            "shap_value": values,
        }
    )
    row["direction"] = np.where(row["shap_value"] >= 0, "toward churn", "away from churn")
    row = row.reindex(row["shap_value"].abs().sort_values(ascending=False).index)
    return row.head(top_n).reset_index(drop=True)
