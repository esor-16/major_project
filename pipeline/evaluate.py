"""Unified evaluation: traditional metrics (Accuracy, F1, ROC-AUC),
profitability metrics (EMP, eProfits_avg, eProfits_tenure, plus the
paper's top-20%-segment e-Profits), and risk-prioritisation metrics
(top-decile lift, lift index) computed together for every tuned model,
on the held-out test split - one table per pipeline run instead of the
ad-hoc per-model printouts in the Phase-1 notebook.

Fixes a bug present in the Phase-1 notebook's evaluation cell, where F1 was
computed as f1_score(y_pred, y_pred) (comparing predictions to themselves,
always 1.0) instead of f1_score(y_test, y_pred).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn import metrics as skmetrics
from sklearn.model_selection import cross_val_predict

from .eprofits import compute_eprofits, make_emp_scorer
from .survival import conditional_retention

TOP_SEGMENT_FRACTION = 0.2  # the paper's top-20% campaign scenario


def top_segment_positions(y_scores, fraction: float = TOP_SEGMENT_FRACTION) -> np.ndarray:
    """Positions of the top `fraction` of customers ranked by predicted
    churn risk (highest first) - the paper's budget-constrained campaign
    scenario: intervene only on the highest-risk `fraction` of customers.
    """
    y_scores = np.asarray(y_scores)
    n_top = max(1, int(math.ceil(fraction * len(y_scores))))
    return np.argsort(-y_scores)[:n_top]


def top_decile_lift(y_true, y_scores) -> float:
    """Concentration of churners in the top 10% of ranked customers:
    (churn rate within the top decile) / (overall churn rate). 1.0 = no
    better than random; 3.0 = the top decile is 3x as likely to churn.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    overall = y_true.mean()
    if len(y_true) == 0 or overall <= 0:
        return float("nan")
    top = top_segment_positions(y_scores, fraction=0.10)
    return float(y_true[top].mean() / overall)


def lift_index(y_true, y_scores, n_deciles: int = 10) -> float:
    """Weighted lift across ranked deciles: the mean cumulative share of
    ALL churners captured at each decile boundary (ranks descending by
    predicted risk).

    Random ranking scores 0.55 (the mean of 0.1, 0.2, ..., 1.0) and a
    perfect ranking scores 1.0, so "close to 1" means churners are
    prioritised almost as early as possible. This matches the paper's
    description of its Lift Index: a weighted index of churners across
    ranked deciles where values close to 1 indicate effective
    prioritisation (its reported values, ~0.79-0.86 for the IBM/Maven
    models, sit exactly where this definition puts strong-but-imperfect
    models).
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    total = y_true.sum()
    if len(y_true) == 0 or total <= 0:
        return float("nan")
    ranked = y_true[np.argsort(-y_scores)]
    cumulative = 0
    captures = []
    for part in np.array_split(ranked, n_deciles):
        cumulative += part.sum()
        captures.append(cumulative / total)
    return float(np.mean(captures))


def find_f1_optimal_threshold(
    estimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3,
    balance_method: str | None = None,
    random_state: int = 42,
) -> float:
    """The classification threshold that maximizes F1, found via
    out-of-fold cross-validated predictions on the *training* data (never
    the held-out test set, to avoid picking a threshold that flatters that
    specific split). `estimator` should be the model's tuned, unfitted
    hyperparameter configuration - it is refitted fresh on each fold.

    `balance_method` (e.g. "smote") reproduces the pipeline's training
    regime INSIDE each CV fold: the fold's training portion is balanced,
    but the held-out VALIDATION rows keep their real ~27% class prior.
    That matters because selecting the threshold on fully balanced
    out-of-fold predictions (the old behaviour) calibrates it to SMOTE's
    artificial 50/50 world; the resulting thresholds didn't transfer to
    the real-prior test set - and transferred worst for the wrapped models
    (fusion hybrids, blend) whose softer probability distributions landed
    on thresholds like 0.18-0.25 vs the baselines' 0.4-0.5, showing up as
    a pure threshold artefact in the accuracy column. With real-prior
    validation rows, models with near-identical test distributions get
    near-identical thresholds. `balance_method=None` (default) means plain
    cross_val_predict on the data as given.

    The models are GA-tuned on a composite eProfits+F1 objective, not F1
    alone, and e-Profits itself always uses the paper's fixed 0.5 cutoff -
    this threshold only sets where Accuracy/F1 are REPORTED, which is
    standard practice for imbalanced classification and doesn't require
    retraining or changing what the GA search optimized.
    """
    if balance_method:
        from sklearn.base import clone as sk_clone
        from sklearn.model_selection import StratifiedKFold

        from .data import balance_classes

        probabilities = np.empty(len(x_train), dtype=float)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        for train_idx, valid_idx in skf.split(x_train, y_train):
            fold_est = sk_clone(estimator)
            x_fold, y_fold = balance_classes(
                x_train.iloc[train_idx], y_train.iloc[train_idx],
                method=balance_method, random_state=random_state,
            )
            fold_est.fit(x_fold, y_fold)
            probabilities[valid_idx] = fold_est.predict_proba(x_train.iloc[valid_idx])[:, 1]
    else:
        probabilities = cross_val_predict(estimator, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = skmetrics.precision_recall_curve(y_train, probabilities)
    denom = np.maximum(precision + recall, 1e-12)
    f1_scores = 2 * precision * recall / denom
    # Last precision/recall point has no matching threshold; drop it.
    f1_scores, thresholds = f1_scores[:-1], thresholds
    # Tie-break toward the HIGHEST threshold among (numerically) equal-F1
    # operating points. The old `np.argmax` returned the FIRST maximum,
    # i.e. the LOWEST threshold on what is usually a flat plateau - that
    # kept landing the hybrids/blend on thresholds like 0.17-0.25 (vs the
    # baselines' 0.4-0.5) with *the same* OOF F1 but far worse precision,
    # accuracy and business outcomes (more false-positive interventions).
    # OOF F1 is unchanged up to the 1e-12 epsilon, so nothing is traded
    # away except the needless false positives.
    best_f1 = float(np.max(f1_scores))
    candidates = np.flatnonzero(f1_scores >= best_f1 - 1e-12)
    best_index = int(candidates[-1])
    return float(thresholds[best_index])


def evaluate_model(
    name: str,
    fitted_search,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    reference_df: pd.DataFrame,
    retention_fn,
    avg_retention_rate: float,
    tenure_column: str = "tenure",
    delta: float = 1.0,
    threshold: float = 0.5,
    classification_threshold: float | None = None,
) -> dict:
    """classification_threshold, when given, is used only for accuracy/F1 -
    the business-facing eProfits/EMP figures stay at `threshold` (0.5 by
    default), matching the operating point the GA search actually selected
    hyperparameters against.
    """
    y_scores = fitted_search.predict_proba(x_test)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    y_pred_for_metrics = (y_scores >= classification_threshold).astype(int) if classification_threshold is not None else y_pred

    scoring_df = reference_df.loc[x_test.index, ["customer_value"]].copy()
    ref_rows = reference_df.loc[x_test.index]
    tenure = ref_rows[tenure_column].astype(float).to_numpy()
    scoring_df["true"] = y_test.to_numpy()
    scoring_df["predict"] = y_pred

    scoring_df["retention_rate"] = conditional_retention(retention_fn, tenure, delta=delta, x=ref_rows)
    eprofits_tenure = compute_eprofits(scoring_df)

    scoring_df["retention_rate"] = float(avg_retention_rate)
    eprofits_avg = compute_eprofits(scoring_df)

    # Budget-constrained campaign scenario (paper Sect. 3.7): profit if only
    # the top 20% most at-risk customers (by predicted churn probability)
    # were targeted; everyone outside the segment gets no action and
    # contributes 0.
    segment = top_segment_positions(y_scores)
    segment_df = scoring_df.iloc[segment].copy()
    segment_df["retention_rate"] = conditional_retention(
        retention_fn, tenure[segment], delta=delta, x=ref_rows.iloc[segment]
    )
    eprofits_top20_tenure = compute_eprofits(segment_df)
    segment_df["retention_rate"] = float(avg_retention_rate)
    eprofits_top20_avg = compute_eprofits(segment_df)

    emp_value = make_emp_scorer()(fitted_search, x_test, y_test)

    y_true = y_test.to_numpy()
    return {
        "model": name,
        "accuracy": skmetrics.accuracy_score(y_test, y_pred_for_metrics),
        "f1": skmetrics.f1_score(y_test, y_pred_for_metrics),
        "f1_threshold": classification_threshold if classification_threshold is not None else threshold,
        "roc_auc": skmetrics.roc_auc_score(y_test, y_scores),
        "top_decile_lift": top_decile_lift(y_true, y_scores),
        "lift_index": lift_index(y_true, y_scores),
        "emp": emp_value,
        "eprofits_avg": eprofits_avg,
        "eprofits_tenure": eprofits_tenure,
        "eprofits_top20_avg": eprofits_top20_avg,
        "eprofits_top20_tenure": eprofits_top20_tenure,
    }


def evaluate_all(
    fitted_models: dict,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    reference_df: pd.DataFrame,
    retention_fn,
    avg_retention_rate: float,
    x_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    tune_f1_threshold: bool = True,
    threshold_cv: int = 3,
    balance_method: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """When `x_train`/`y_train` are given and `tune_f1_threshold` is True,
    each model's Accuracy/F1 are reported at its own F1-optimal threshold
    (see `find_f1_optimal_threshold` - pass `balance_method` matching the
    pipeline's resampling so the threshold is calibrated on real-prior
    validation rows) instead of a flat 0.5. eProfits/EMP are unaffected
    either way - they always use the business threshold 0.5.
    """
    can_tune = tune_f1_threshold and x_train is not None and y_train is not None

    rows = []
    for name, search in fitted_models.items():
        classification_threshold = None
        if can_tune:
            estimator = getattr(search, "best_estimator_", search)
            classification_threshold = find_f1_optimal_threshold(
                estimator, x_train, y_train, cv=threshold_cv,
                balance_method=balance_method,
            )
        rows.append(
            evaluate_model(
                name, search, x_test, y_test, reference_df, retention_fn, avg_retention_rate,
                classification_threshold=classification_threshold, **kwargs,
            )
        )
    return pd.DataFrame(rows).set_index("model").sort_values("eprofits_tenure", ascending=False)
