"""Slow end-to-end smoke tests: run the full pipeline (preprocess -> balance
-> survival -> GA-tuned training -> unified evaluation) on a small subsample
with minimal GA settings, to confirm the modules wire together correctly.
Full-scale, full-dataset runs are driven via `python -m pipeline.run` (see
NFR3/TC8 timing in the synopsis).
"""
from pathlib import Path

import pytest


@pytest.mark.slow
def test_run_pipeline_smoke(tmp_path, small_split, processed_df):
    from pipeline import evaluate, models
    from pipeline.eprofits import make_eprofits_scorer_tenure
    from pipeline.survival import fit_survival_model

    (x_train, x_test, y_train, y_test), sample = small_split

    cph, retention_fn, avg_retention_rate = fit_survival_model(x_train, y_train)
    scorer = make_eprofits_scorer_tenure(reference_df=x_train, retention_fn=retention_fn)

    fitted = models.train_all_models(
        x_train, y_train, cv=2, scoring_metric="eprofits_tenure", scorer=scorer,
        model_names=["xgb", "random_forest"], population_size=4, generations=1,
    )
    assert set(fitted.keys()) == {"xgb", "random_forest"}

    results = evaluate.evaluate_all(
        fitted, x_test, y_test, reference_df=sample,
        retention_fn=retention_fn, avg_retention_rate=avg_retention_rate,
    )

    assert set(results.index) == {"xgb", "random_forest"}
    for col in ["accuracy", "f1", "f1_threshold", "roc_auc", "emp", "eprofits_avg", "eprofits_tenure"]:
        assert col in results.columns
        assert results[col].notna().all()


@pytest.mark.slow
def test_run_pipeline_smoke_with_feature_fusion_hybrid(small_split):
    """The advisor-requested addition: donor SHAP (random_forest + lightgbm,
    computed on x_train, never x_test) -> fused feature ranking -> GA-tuned
    xgb recipient per fusion rule -> wrapped as a full-width SubsetPipeline
    -> evaluated through the exact same evaluate_all() path as the baselines.
    """
    from pipeline import evaluate, models
    from pipeline.eprofits import make_eprofits_scorer_tenure
    from pipeline.feature_fusion import (
        SubsetPipeline,
        donor_importances,
        fuse_rankings,
        rank_agreement,
        select_features,
    )
    from pipeline.models import build_model_registry
    from pipeline.survival import fit_survival_model

    (x_train, x_test, y_train, y_test), sample = small_split

    cph, retention_fn, avg_retention_rate = fit_survival_model(x_train, y_train)
    scorer = make_eprofits_scorer_tenure(reference_df=x_train, retention_fn=retention_fn)

    baseline_names = ["xgb", "random_forest", "lightgbm"]
    fitted = models.train_all_models(
        x_train, y_train, cv=2, scoring_metric="eprofits_tenure", scorer=scorer,
        model_names=baseline_names, population_size=4, generations=1,
    )

    importances = donor_importances(fitted, x_train, donor_names=["random_forest", "lightgbm"], sample_size=len(x_train))
    agreement = rank_agreement(importances)
    assert -1.0 <= agreement <= 1.0

    fused = fuse_rankings(importances)
    columns = select_features(fused, "rank_fusion", m=5)
    assert len(columns) == 5

    registry = build_model_registry()
    recipient_model, recipient_param_grid = registry["xgb"]
    gs = models.genetic_search(
        recipient_model, recipient_param_grid, x_train[columns], y_train, cv=2,
        scoring={"eprofits_tenure": scorer}, refit="eprofits_tenure",
        population_size=4, generations=1, name="hybrid_rank_fusion",
    )
    fitted["hybrid_rank_fusion"] = SubsetPipeline(
        estimator=gs.best_estimator_, features=columns, all_columns=list(x_train.columns),
    )

    results = evaluate.evaluate_all(
        fitted, x_test, y_test, reference_df=sample,
        retention_fn=retention_fn, avg_retention_rate=avg_retention_rate,
    )

    assert set(results.index) == {*baseline_names, "hybrid_rank_fusion"}
    for col in ["accuracy", "f1", "f1_threshold", "roc_auc", "emp", "eprofits_avg", "eprofits_tenure"]:
        assert col in results.columns
        assert results[col].notna().all()


@pytest.mark.slow
def test_run_pipeline_end_to_end_via_run_pipeline_function(tmp_path):
    """Exercises `run.run_pipeline` itself (the real CLI entry point) end to
    end - including the fusion-hybrid wiring inside it - on the real IBM.csv
    with a minimal GA budget, writing artifacts to a temp directory rather
    than the checked-in artifacts/ folder.
    """
    from pipeline.run import run_pipeline

    results = run_pipeline(
        data_path="IBM.csv",
        model_names=["xgb", "random_forest", "lightgbm"],
        cv=2,
        population_size=3,
        generations=1,
        output_dir=str(tmp_path),
        explain_sample_size=30,
        fusion_donors=["random_forest", "lightgbm"],
        fusion_rules=["rank_fusion"],
        fusion_sizes=(5,),
        fusion_sample_size=100,
        bootstrap_resamples=100,
    )

    # three baselines + the feature-fusion hybrid + the prediction-level
    # blend hybrid (default members xgb + lightgbm, both trained here)
    assert set(results.index) == {
        "xgb", "random_forest", "lightgbm", "hybrid_rank_fusion", "hybrid_blend",
    }
    assert (tmp_path / "model_evaluation.csv").exists()
    assert (tmp_path / "feature_fusion_rankings.csv").exists()
    assert (tmp_path / "feature_fusion_summary.json").exists()
    assert (tmp_path / "blend_summary.json").exists()
    assert (tmp_path / "bootstrap_ci.csv").exists()
    assert (tmp_path / "wilcoxon_tests.csv").exists()
    assert (tmp_path / "rank_correlation.csv").exists()
    assert (tmp_path / "dashboard.json").exists()

    # the new paper-style evaluation columns all present and finite
    for col in ["top_decile_lift", "lift_index", "eprofits_top20_avg", "eprofits_top20_tenure"]:
        assert col in results.columns
        assert results[col].notna().all()
