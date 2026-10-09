"""Covers the F1-optimal-threshold fix: models are GA-tuned to maximize
eProfits_tenure and evaluated at a flat 0.5 cutoff by default, which can
leave F1 well below NFR1's 0.70 bar on this ~27%-churn dataset even when
ROC-AUC is comfortably high. `evaluate_all` can instead report Accuracy/F1
at each model's own best operating point, found via cross-validated
predictions on the training data (never the held-out test set).
"""
import numpy as np

from pipeline import evaluate
from pipeline.models import build_model_registry


def test_f1_optimal_threshold_is_between_zero_and_one(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    model, _ = build_model_registry()["random_forest"]

    threshold = evaluate.find_f1_optimal_threshold(model, x_train, y_train, cv=2)
    assert 0.0 <= threshold <= 1.0


def test_f1_optimal_threshold_beats_or_matches_flat_half(small_split):
    """The whole point of threshold tuning: F1 at the chosen threshold
    should never be worse than F1 at a flat 0.5 cutoff, on the same data
    the threshold was searched over.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_predict

    (x_train, x_test, y_train, y_test), _ = small_split
    model, _ = build_model_registry()["random_forest"]

    threshold = evaluate.find_f1_optimal_threshold(model, x_train, y_train, cv=2)
    probas = cross_val_predict(model, x_train, y_train, cv=2, method="predict_proba")[:, 1]

    f1_at_half = f1_score(y_train, (probas >= 0.5).astype(int))
    f1_at_tuned = f1_score(y_train, (probas >= threshold).astype(int))
    assert f1_at_tuned >= f1_at_half - 1e-9


def test_f1_optimal_threshold_with_fold_internal_balancing(small_split):
    """The pipeline's actual threshold protocol: each CV fold's training
    portion is SMOTE-balanced (matching how the models were trained) while
    validation rows keep their real class prior - the threshold must still
    land in (0, 1] and the search must not touch the test set."""
    (x_train, x_test, y_train, y_test), _ = small_split
    model, _ = build_model_registry()["random_forest"]

    threshold = evaluate.find_f1_optimal_threshold(
        model, x_train, y_train, cv=2, balance_method="smote",
    )
    assert 0.0 < threshold <= 1.0


def test_evaluate_all_uses_tuned_threshold_when_train_data_given(small_split):
    (x_train, x_test, y_train, y_test), sample = small_split
    from pipeline.eprofits import make_eprofits_scorer_avg
    from pipeline.survival import fit_survival_model
    from pipeline import models

    _, retention_fn, avg_rate = fit_survival_model(x_train, y_train)
    scorer = make_eprofits_scorer_avg(reference_df=x_train, avg_retention_rate=avg_rate)
    fitted = models.train_all_models(
        x_train, y_train, cv=2, scoring_metric="eprofits_avg", scorer=scorer,
        model_names=["xgb"], population_size=4, generations=1,
    )

    results = evaluate.evaluate_all(
        fitted, x_test, y_test, reference_df=sample,
        retention_fn=retention_fn, avg_retention_rate=avg_rate,
        x_train=x_train, y_train=y_train, threshold_cv=2,
    )

    assert results.loc["xgb", "f1_threshold"] != 0.5


def test_evaluate_all_falls_back_to_flat_threshold_without_train_data(small_split):
    (x_train, x_test, y_train, y_test), sample = small_split
    from pipeline.eprofits import make_eprofits_scorer_avg
    from pipeline.survival import fit_survival_model
    from pipeline import models

    _, retention_fn, avg_rate = fit_survival_model(x_train, y_train)
    scorer = make_eprofits_scorer_avg(reference_df=x_train, avg_retention_rate=avg_rate)
    fitted = models.train_all_models(
        x_train, y_train, cv=2, scoring_metric="eprofits_avg", scorer=scorer,
        model_names=["xgb"], population_size=4, generations=1,
    )

    results = evaluate.evaluate_all(
        fitted, x_test, y_test, reference_df=sample,
        retention_fn=retention_fn, avg_retention_rate=avg_rate,
    )

    assert results.loc["xgb", "f1_threshold"] == 0.5


def test_evaluate_all_respects_tune_f1_threshold_false(small_split):
    (x_train, x_test, y_train, y_test), sample = small_split
    from pipeline.eprofits import make_eprofits_scorer_avg
    from pipeline.survival import fit_survival_model
    from pipeline import models

    _, retention_fn, avg_rate = fit_survival_model(x_train, y_train)
    scorer = make_eprofits_scorer_avg(reference_df=x_train, avg_retention_rate=avg_rate)
    fitted = models.train_all_models(
        x_train, y_train, cv=2, scoring_metric="eprofits_avg", scorer=scorer,
        model_names=["xgb"], population_size=4, generations=1,
    )

    results = evaluate.evaluate_all(
        fitted, x_test, y_test, reference_df=sample,
        retention_fn=retention_fn, avg_retention_rate=avg_rate,
        x_train=x_train, y_train=y_train, tune_f1_threshold=False,
    )

    assert results.loc["xgb", "f1_threshold"] == 0.5


def test_eprofits_unaffected_by_classification_threshold(small_split):
    """eProfits/EMP must stay pinned to the business threshold (0.5) even
    when a different classification_threshold is used for Accuracy/F1 -
    they're answering a different question (who to target) than F1 is.
    """
    (x_train, x_test, y_train, y_test), sample = small_split
    from pipeline.eprofits import make_eprofits_scorer_avg
    from pipeline.survival import fit_survival_model

    _, retention_fn, avg_rate = fit_survival_model(x_train, y_train)
    model, _ = build_model_registry()["xgb"]
    model.fit(x_train, y_train)

    row_default = evaluate.evaluate_model(
        "xgb", model, x_test, y_test, sample, retention_fn, avg_rate,
    )
    row_custom_threshold = evaluate.evaluate_model(
        "xgb", model, x_test, y_test, sample, retention_fn, avg_rate,
        classification_threshold=0.9,
    )

    assert row_default["eprofits_tenure"] == row_custom_threshold["eprofits_tenure"]
    assert row_default["eprofits_avg"] == row_custom_threshold["eprofits_avg"]
