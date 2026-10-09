"""Model registry and GA-based hyperparameter tuning.

Expands the Phase-1 prototype's two classifiers (XGBoost, EBM) to the six
models frozen in the Phase-2 scope (synopsis Section 2.5 / 4.5): XGBoost,
Gradient Boosting, Random Forest, LightGBM, CatBoost, and the glass-box
Explainable Boosting Machine - tree ensembles for accuracy plus one
inherently interpretable model, so accuracy and interpretability can be
compared directly.
"""
from __future__ import annotations

import datetime
import random
import time

import numpy as np
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Continuous, Integer
from xgboost import XGBClassifier


class _RobustCatBoostClassifier(CatBoostClassifier):
    """CatBoostClassifier subclass working around a CatBoost/scikit-learn
    incompatibility: CatBoost's get_params() re-derives values through its
    internal C++ param store, returning a *different* object each time even
    when the value is unchanged. scikit-learn's clone() (>=1.6) does a
    strict identity check during cross-validation and raises
    "constructor either does not set or modifies parameter X" as a result -
    this only surfaces for CatBoost because the other five models' plain
    attribute get/set_params happens to preserve identity through clone().
    This subclass caches the literal kwargs passed via __init__/set_params
    and returns those from get_params(), so clone() sees identical objects;
    actual training still goes through CatBoost's own implementation.
    """

    def __init__(self, **kwargs):
        self._raw_params = dict(kwargs)
        super().__init__(**kwargs)

    def get_params(self, deep=True):
        return dict(self._raw_params)

    def set_params(self, **params):
        self._raw_params.update(params)
        return super().set_params(**params)


def build_model_registry(random_state: int = 42) -> dict:
    """name -> (unfitted estimator, GA search space)."""
    return {
        "xgb": (
            XGBClassifier(random_state=random_state, eval_metric="logloss"),
            {
                "n_estimators": Integer(100, 300),
                "learning_rate": Continuous(0.01, 0.2),
                "max_depth": Integer(3, 5),
                "colsample_bytree": Continuous(0.3, 0.7),
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "n_estimators": Integer(100, 300),
                "learning_rate": Continuous(0.01, 0.2),
                "max_depth": Integer(2, 5),
                "subsample": Continuous(0.6, 1.0),
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state),
            {
                "n_estimators": Integer(100, 400),
                "max_depth": Integer(3, 15),
                "min_samples_leaf": Integer(1, 10),
                "max_features": Categorical(["sqrt", "log2"]),
            },
        ),
        "lightgbm": (
            LGBMClassifier(random_state=random_state, verbosity=-1),
            {
                "n_estimators": Integer(100, 300),
                "learning_rate": Continuous(0.01, 0.2),
                "num_leaves": Integer(15, 63),
                "colsample_bytree": Continuous(0.3, 0.7),
            },
        ),
        "catboost": (
            _RobustCatBoostClassifier(random_state=random_state, verbose=False),
            {
                "iterations": Integer(100, 300),
                "learning_rate": Continuous(0.01, 0.2),
                "depth": Integer(3, 8),
                "l2_leaf_reg": Continuous(1.0, 10.0),
            },
        ),
        "ebm": (
            ExplainableBoostingClassifier(random_state=random_state),
            {
                "learning_rate": Continuous(0.01, 0.1),
                "max_bins": Integer(128, 256),
                "max_interaction_bins": Integer(16, 32),
                "interactions": Integer(10, 20),
            },
        ),
    }


def genetic_search(
    model,
    param_grid,
    x,
    y,
    cv,
    scoring,
    refit,
    population_size: int = 20,
    generations: int = 10,
    # Custom eProfits/EMP scorers are closures over a reference dataframe;
    # on Windows, joblib's multiprocessing backend needs to pickle those to
    # hand off to worker processes, which is unreliable for closures. Default
    # to single-process to keep the search portable; override for speed on
    # Linux/fork-based systems where this isn't an issue.
    n_jobs: int = 1,
    verbose: bool = False,
    name: str | None = None,
    random_state: int | None = None,
):
    """`random_state`, when given, reseeds Python's global `random` module
    and numpy's global RNG immediately before this search runs.

    This is the actual point of reproducibility guarantee for the whole
    pipeline (NFR5) - sklearn-genetic-opt's GASearchCV has no random_state
    of its own and draws population initialization/mutation/crossover from
    those two global sources (confirmed by inspecting its source: plain
    `random.random()`/`random.choice()` calls throughout). Seeding once at
    the top of `run_pipeline` was tried first and found insufficient:
    `sklearn.model_selection.train_test_split(..., stratify=y)` perturbs the
    global RNG as a side effect even when given its own explicit
    random_state, so anything relying on a single seed propagating cleanly
    through the rest of the pipeline is fragile - reseeding immediately
    before the one thing that actually needs it is more robust and doesn't
    depend on what unrelated library internals do in between.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    if name:
        print(f"[{datetime.datetime.now()}] tuning {name}")
    start = time.time()

    gs = GASearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        population_size=population_size,
        generations=generations,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.8,
        mutation_probability=0.2,
        algorithm="eaMuPlusLambda",
        criteria="max",
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
        verbose=verbose,
        keep_top_k=2,
    )
    gs.fit(x, y)
    gs.fit_time_ = time.time() - start
    if name:
        print(f"  > {name} done in {gs.fit_time_:.1f}s, best {refit}={gs.best_score_:.4f}")
    return gs


def train_all_models(
    x_train,
    y_train,
    cv,
    scoring_metric: str,
    scorer,
    model_names: list[str] | None = None,
    random_state: int = 42,
    **search_kwargs,
) -> dict:
    """Tune every registered model (or a named subset) against a single
    scoring metric, then evaluate all of them together downstream
    (`evaluate.evaluate_all`) - the "unified benchmarking step" the synopsis
    calls for in place of Phase-1's per-metric-per-model search matrix.

    Each model's GA search is reseeded to `random_state + <its position in
    names>` immediately before it runs (see `genetic_search`'s docstring for
    why this - not a single seed at the top of the pipeline - is what
    actually makes runs reproducible). `names` is a list, so this offset is
    stable across repeated calls with the same `model_names`/registry order.
    """
    registry = build_model_registry(random_state)
    names = model_names or list(registry.keys())
    unknown = set(names) - set(registry)
    if unknown:
        raise ValueError(f"Unknown model name(s): {sorted(unknown)}; choose from {sorted(registry)}")

    results = {}
    for i, name in enumerate(names):
        model, param_grid = registry[name]
        results[name] = genetic_search(
            model, param_grid, x_train, y_train, cv,
            scoring={scoring_metric: scorer}, refit=scoring_metric,
            name=name, random_state=random_state + i, **search_kwargs,
        )
    return results
