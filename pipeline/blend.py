"""Prediction-level hybrid: a probability blend of two tuned classifiers.

Where `feature_fusion.py` fuses two donors at the *feature* level (their
SHAP rankings are merged into one curated feature subset that a third
model is trained on), this module fuses two models at the *prediction*
level: each member contributes its churn probability and a single weight
`w` averages them.

    P(churn) = w * P_a(churn) + (1 - w) * P_b(churn)

Two design choices, mirroring feature_fusion's:

1. Members are two different model families (default: xgb + lightgbm),
   both already GA-tuned against the same composite eProfits+F1 scorer as
   every baseline - so any accuracy gain over the individual members
   comes from the blend itself, not from a bigger tuning budget.

2. The blend weight is chosen on **training** data only, via out-of-fold
   predictions (cross_val_predict) scored with the pipeline's own scorer.
   The held-out test set never participates in choosing `w`.

Soft-voting blends of complementary boosted/tree models routinely beat
either member on ROC-AUC and F1 - this is the "highly accurate
predictions" half of the project's aim, complementing feature fusion's
feature-pruning half. The weight sweep is exported (`blend_summary.json`)
so the selection is auditable, exactly like the feature-count sweep.

`ProbabilityBlend` implements the scikit-learn estimator API
(fit/get_params/set_params/clone) so it drops into the existing
consumers unchanged: `evaluate.evaluate_all`, the F1-optimal-threshold
search (which clones it through cross_val_predict), SHAP, and the
dashboard export all just call `predict_proba(x)` on the full-width
frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict

DEFAULT_BLEND_MEMBERS = ["xgb", "lightgbm"]
DEFAULT_WEIGHT_GRID = tuple(np.round(np.arange(0.0, 1.01, 0.05), 2))


class ProbabilityBlend(BaseEstimator, ClassifierMixin):
    """Weighted average of two classifiers' `predict_proba` outputs.

    Construct around *fitted* estimators and predict immediately (no
    refit needed - same tolerance `SubsetPipeline` shows for the GA
    search's fitted `best_estimator_`), or around unfitted configs and
    call `fit()`. `clone()` always yields the unfitted hyperparameter
    config, which is what the threshold search's cross_val_predict
    relies on.
    """

    def __init__(self, estimator_a=None, estimator_b=None, weight: float = 0.5):
        self.estimator_a = estimator_a
        self.estimator_b = estimator_b
        self.weight = weight

    def _fitted_a(self):
        return getattr(self, "estimator_a_", self.estimator_a)

    def _fitted_b(self):
        return getattr(self, "estimator_b_", self.estimator_b)

    def fit(self, x, y):
        self.estimator_a_ = clone(self.estimator_a).fit(x, y)
        self.estimator_b_ = clone(self.estimator_b).fit(x, y)
        return self  # self.classes_ reads through the property below

    @property
    def classes_(self):
        # Exposed even when constructed around already-fitted members
        # (fit() was never called on this wrapper); some sklearn utilities
        # probe .classes_ on any fitted-looking classifier.
        return self._fitted_a().classes_

    def predict_proba(self, x):
        pa = self._fitted_a().predict_proba(x)
        pb = self._fitted_b().predict_proba(x)
        w = float(self.weight)
        return w * pa + (1.0 - w) * pb

    def predict(self, x):
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


class _PrecomputedBlend:
    """Duck-typed estimator whose `predict_proba(x)` returns already-computed
    out-of-fold blend probabilities for x's rows, so the pipeline's scorer
    closures (which call `estimator.predict_proba(x)[:, 1]`) can score a
    candidate weight without retraining anything.
    """

    def __init__(self, proba: pd.DataFrame):
        self.proba = proba

    def predict_proba(self, x) -> np.ndarray:
        return self.proba.loc[x.index].to_numpy()


def select_blend_weight(
    models: dict,
    member_names: list[str],
    x: pd.DataFrame,
    y: pd.Series,
    scorer,
    cv: int = 3,
    weight_grid=DEFAULT_WEIGHT_GRID,
) -> tuple[float, pd.DataFrame]:
    """Pick the blend weight in `weight_grid` maximizing `scorer` on
    OUT-OF-FOLD probabilities from the two members' tuned configurations.

    `models` maps name -> fitted search (or estimator); each member's
    hyperparameters are taken from `best_estimator_` when present (i.e.
    the GA-tuned config), cloned, and run through cross_val_predict on
    the training data only. The test set never participates.

    Returns (best_weight, sweep_table).
    """
    missing = [m for m in member_names if m not in models]
    if missing:
        raise ValueError(f"Blend member(s) {missing} were not trained; available: {sorted(models)}")

    (name_a, name_b) = member_names
    est_a = clone(getattr(models[name_a], "best_estimator_", models[name_a]))
    est_b = clone(getattr(models[name_b], "best_estimator_", models[name_b]))

    proba_a = cross_val_predict(est_a, x, y, cv=cv, method="predict_proba")
    proba_b = cross_val_predict(est_b, x, y, cv=cv, method="predict_proba")

    index = x.index if isinstance(x, pd.DataFrame) else pd.RangeIndex(len(x))
    rows = []
    for w in weight_grid:
        blended = float(w) * proba_a + (1.0 - float(w)) * proba_b
        score = float(scorer(_PrecomputedBlend(pd.DataFrame(blended, index=index)), x, y))
        rows.append({"weight": float(w), "cv_score": score})

    sweep = pd.DataFrame(rows)
    if sweep["cv_score"].isna().all():
        raise RuntimeError(
            "cv_score was NaN for every candidate blend weight (the scorer failed "
            f"on every weight - check the scorer/blend combination); sweep:\n{sweep.to_string(index=False)}"
        )
    best_weight = float(sweep.loc[sweep["cv_score"].idxmax(), "weight"])
    return best_weight, sweep


def build_blend_hybrid(
    models: dict,
    member_names: list[str] | None = None,
    x_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    scorer=None,
    cv: int = 3,
    weight_grid=DEFAULT_WEIGHT_GRID,
) -> tuple[ProbabilityBlend, float, pd.DataFrame]:
    """Build the fitted prediction-level hybrid.

    Weight selection runs on `x_train`/`y_train` (out-of-fold, training
    only); the returned blend's members are then fitted on the full
    training frame - the exact same data every baseline was trained on,
    so the comparison blend-vs-member isolates the blending itself.

    Returns (fitted ProbabilityBlend, weight, weight-sweep table).
    """
    member_names = list(member_names or DEFAULT_BLEND_MEMBERS)
    weight, sweep = select_blend_weight(models, member_names, x_train, y_train, scorer, cv=cv, weight_grid=weight_grid)

    est_a = clone(getattr(models[member_names[0]], "best_estimator_", models[member_names[0]]))
    est_b = clone(getattr(models[member_names[1]], "best_estimator_", models[member_names[1]]))
    blend = ProbabilityBlend(estimator_a=est_a, estimator_b=est_b, weight=weight).fit(x_train, y_train)
    return blend, weight, sweep
