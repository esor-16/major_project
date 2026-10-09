"""Feature-level fusion: build a new model from the decisive features that
two *donor* models rely on.

Where `models.py` trains each classifier independently and `evaluate.py`
compares them, this module fuses them at the **feature** level: it asks two
structurally different donors (Random Forest and LightGBM by default) which
features they actually rely on - via SHAP - merges those two signals into one
curated feature set, and trains a third model on it.

Two design choices worth stating explicitly, because both are load-bearing:

1. The recipient is deliberately *neither donor* (XGBoost by default). If the
   recipient were RF or LightGBM it would be re-fitting on features it
   selected itself, which proves nothing; using a third family shows the
   feature knowledge genuinely transfers. It also creates a controlled
   ablation for free - `xgb` already exists as a full-feature baseline, so
   hybrid-vs-`xgb` differs *only* in the feature set.

2. Donor SHAP is computed on **training** data. Selecting features from
   test-set SHAP and then scoring on that same test set is leakage; the
   reported number would be optimistically biased. The pipeline's existing
   test-set SHAP pass (run.py) is untouched - it serves reporting, not
   selection. Two passes, two purposes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_score

from . import explain

DONOR_MODELS = ["random_forest", "lightgbm"]
DEFAULT_RECIPIENT = "xgb"
DEFAULT_FUSION_SIZES = (5, 8, 10, 12, 15)

# All three fusion rules are the *same* operation under a different
# aggregator, rather than three unrelated set operations. That keeps them
# directly comparable at a matched feature-set size (each produces a full
# ranking, so the size sweep is identical across rules) instead of the
# "union gives 2k features, intersection gives 3" mismatch you get from
# doing literal set algebra on top-k lists.
#
#   union        max(a, b)   a feature ranks high if EITHER donor rates it high
#   rank_fusion  mean(a, b)  balanced consensus
#   intersection min(a, b)   ranks high only if BOTH donors rate it high
AGGREGATORS = {
    "union": np.maximum,
    "rank_fusion": lambda a, b: (a + b) / 2.0,
    "intersection": np.minimum,
}
FUSION_RULES = tuple(AGGREGATORS)


def donor_importances(
    fitted_models: dict,
    x_train: pd.DataFrame,
    donor_names: list[str] | None = None,
    sample_size: int = 500,
    random_state: int = 42,
) -> dict[str, pd.Series]:
    """Mean |SHAP| per feature for each donor, computed on TRAINING data.

    `x_train` should be the real, pre-SMOTE training split: synthetic
    minority rows are interpolations, not customers, so importances derived
    from them wouldn't describe real churn behaviour. (Same reasoning the
    survival layer already applies - see survival.fit_survival_model.)
    """
    donor_names = donor_names or list(DONOR_MODELS)
    missing = [n for n in donor_names if n not in fitted_models]
    if missing:
        raise ValueError(f"Donor model(s) not trained: {missing}; available: {sorted(fitted_models)}")

    sample = x_train if len(x_train) <= sample_size else x_train.sample(sample_size, random_state=random_state)

    importances = {}
    for name in donor_names:
        explainer = explain.build_explainer(name, fitted_models[name], background=sample)
        explanation = explain.compute_shap_values(explainer, sample, sample_size=None)
        ranked = explain.global_feature_importance(explanation)
        importances[name] = ranked.set_index("feature")["mean_abs_shap"]
    return importances


def _normalize(importance: pd.Series) -> pd.Series:
    """Scale importances to [0, 1] by dividing by the maximum.

    Deliberately not min-max: mean |SHAP| is a ratio-scale quantity with a
    meaningful zero (a feature the model ignores contributes exactly 0), so
    dividing by the max preserves those ratios - a feature worth half the
    top feature scores 0.5. Min-max would instead force the *least* important
    feature to exactly 0 regardless of whether it mattered, distorting the
    comparison between donors whose importance distributions differ in
    spread.
    """
    peak = float(importance.max())
    if peak <= 0:
        return pd.Series(0.0, index=importance.index)
    return importance / peak


def fuse_rankings(importances: dict[str, pd.Series]) -> pd.DataFrame:
    """Normalize each donor and apply all three aggregators.

    Returns one table carrying every rule's score, so a single call produces
    the full exportable comparison rather than needing one call per rule.
    """
    if len(importances) != 2:
        raise ValueError(f"Fusion expects exactly 2 donors, got {len(importances)}: {sorted(importances)}")

    (name_a, imp_a), (name_b, imp_b) = importances.items()

    # Align on the union of feature names; a feature absent from one donor's
    # table scores 0 there rather than NaN, which keeps min/max/mean well
    # defined (and is semantically right - absent means "not used").
    features = imp_a.index.union(imp_b.index)
    norm_a = _normalize(imp_a).reindex(features, fill_value=0.0)
    norm_b = _normalize(imp_b).reindex(features, fill_value=0.0)

    fused = pd.DataFrame(
        {"feature": features, f"{name_a}_norm": norm_a.to_numpy(), f"{name_b}_norm": norm_b.to_numpy()}
    )
    for rule, aggregate in AGGREGATORS.items():
        fused[rule] = aggregate(norm_a.to_numpy(), norm_b.to_numpy())

    return fused.sort_values("rank_fusion", ascending=False).reset_index(drop=True)


def select_features(fused: pd.DataFrame, rule: str, m: int) -> list[str]:
    """The top `m` features under one fusion rule."""
    if rule not in AGGREGATORS:
        raise ValueError(f"Unknown fusion rule {rule!r}; choose from {sorted(AGGREGATORS)}")
    ordered = fused.sort_values(rule, ascending=False)
    return ordered["feature"].head(m).tolist()


def rank_agreement(importances: dict[str, pd.Series]) -> float:
    """Spearman correlation between the two donors' importance rankings.

    The diagnostic that justifies the donor pair: fusion can only add
    information if the donors genuinely disagree about what matters. A
    correlation near 1.0 means both models picked the same features and the
    fused set is just one donor's list wearing a hat - in which case the
    honest read is that fusion has no headroom on this dataset, not that the
    method failed.
    """
    (imp_a, imp_b) = importances.values()
    features = imp_a.index.union(imp_b.index)
    a = imp_a.reindex(features, fill_value=0.0).to_numpy()
    b = imp_b.reindex(features, fill_value=0.0).to_numpy()
    if len(features) < 3:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


class SubsetPipeline(BaseEstimator, ClassifierMixin):
    """A classifier trained on a feature subset, wrapped so it still accepts
    the FULL feature matrix and subsets internally.

    This is what lets the fused-feature models drop into the existing
    pipeline untouched: `evaluate.evaluate_all`, the F1-threshold search,
    SHAP, and the dashboard export all call `predict_proba(x_test)` with the
    full-width frame. Without this wrapper every one of those consumers
    would need to be taught which columns each model expects.

    Implements the scikit-learn estimator API (fit/get_params/set_params) so
    `clone()` works - required because `evaluate.find_f1_optimal_threshold`
    runs it through `cross_val_predict`, which refits per fold.
    """

    def __init__(self, estimator=None, features=None, all_columns=None):
        self.estimator = estimator
        self.features = features
        self.all_columns = all_columns

    def _as_frame(self, x):
        """Accept a DataFrame, or rebuild one if handed a bare array.

        SHAP's model-agnostic explainers strip column names and pass numpy
        arrays; `all_columns` (captured at fit time) lets us restore them so
        the subset lookup still works instead of raising.
        """
        if isinstance(x, pd.DataFrame):
            return x
        if self.all_columns is None:
            raise TypeError(
                "SubsetPipeline was given a non-DataFrame input and has no `all_columns` "
                "recorded to rebuild one from; pass a DataFrame or set all_columns."
            )
        return pd.DataFrame(np.asarray(x), columns=list(self.all_columns))

    def fit(self, x, y):
        x = self._as_frame(x)
        if self.all_columns is None:
            self.all_columns = list(x.columns)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x[list(self.features)], y)
        return self  # self.classes_ reads through the property below

    def _fitted(self):
        # Tolerate being constructed around an already-fitted estimator
        # (the GA search hands back a fitted best_estimator_), so callers
        # don't have to refit just to wrap it.
        return getattr(self, "estimator_", self.estimator)

    @property
    def classes_(self):
        # Exposed even when constructed around an already-fitted estimator
        # (fit() was never called on this wrapper, so the attribute wasn't
        # set directly) - some sklearn utilities probe .classes_ on any
        # fitted-looking classifier.
        return self._fitted().classes_

    def predict_proba(self, x):
        x = self._as_frame(x)
        return self._fitted().predict_proba(x[list(self.features)])

    def predict(self, x):
        x = self._as_frame(x)
        return self._fitted().predict(x[list(self.features)])


def sweep_feature_count(
    recipient,
    fused: pd.DataFrame,
    rule: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    scorer,
    sizes=DEFAULT_FUSION_SIZES,
    cv: int = 3,
) -> tuple[int, pd.DataFrame]:
    """Choose how many features to keep, by cross-validated `scorer` on the
    TRAINING split (never the test set).

    `scorer` is whatever the recipient's own GA search will be tuned
    against (see `run.py` - it's the SAME composite eProfits+F1 scorer for
    both, `eprofits.make_composite_scorer_tenure`, not a raw eProfits-only
    one) so the feature count chosen here and the hyperparameters chosen
    afterwards are optimizing for the same thing. An earlier version of this
    function did its own separate eProfits/F1 blending at the sweep stage
    while the recipient's GA search still optimized eProfits alone
    afterwards - two different, inconsistent notions of "how much F1
    matters" in two places. Pushing the blend into the scorer itself (used
    identically by the sweep and by GA tuning) fixed that.

    Scored with an untuned recipient: the point here is to compare feature
    *sets*, and running a full GA search at every candidate size would cost
    len(sizes) x the tuning budget for a decision that a default-hyperparameter
    model ranks just as well. The chosen set is GA-tuned afterwards.

    Returns (best_m, sweep_table) - the table is exported so the selection is
    auditable, and its shape is itself diagnostic: a flat curve across sizes
    means the dropped features were never hurting on this dataset.
    """
    n_features = len(fused)
    candidate_sizes = sorted({min(int(m), n_features) for m in sizes if int(m) > 0} | {n_features})

    rows = []
    for m in candidate_sizes:
        columns = select_features(fused, rule, m)
        scores = cross_val_score(clone(recipient), x_train[columns], y_train, cv=cv, scoring=scorer)
        rows.append({"rule": rule, "n_features": m, "cv_score": float(np.mean(scores)), "cv_std": float(np.std(scores))})

    sweep = pd.DataFrame(rows)
    if sweep["cv_score"].isna().all():
        raise RuntimeError(
            f"cv_score was NaN for every candidate feature-set size under rule {rule!r} "
            f"(scorer failed on every fold - check the scorer/estimator combination); "
            f"sweep table:\n{sweep.to_string(index=False)}"
        )
    best_m = int(sweep.loc[sweep["cv_score"].idxmax(), "n_features"])
    return best_m, sweep
