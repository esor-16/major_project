"""Tests for pipeline.significance: bootstrap CIs, paired Wilcoxon
signed-rank tests, and Spearman AUC-vs-profit rank correlation - the
paper's Sect. 4.3 statistical validation, plus the file-writing
`run_significance_analysis` entry point used by run.py.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import significance
from pipeline.eprofits import per_customer_eprofits


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    profits = np.random.default_rng(0).normal(loc=50.0, scale=10.0, size=500)

    ci_a = significance.bootstrap_profit_ci(profits, n_boot=200, seed=7)
    ci_b = significance.bootstrap_profit_ci(profits, n_boot=200, seed=7)

    assert ci_a == ci_b  # same seed -> same interval
    assert ci_a["ci_low"] < ci_a["mean"] < ci_a["ci_high"]
    assert ci_a["n_boot"] == 200


def test_bootstrap_ci_widens_with_fewer_resamples_is_sane():
    profits = np.random.default_rng(1).normal(0.0, 100.0, size=200)
    ci = significance.bootstrap_profit_ci(profits, n_boot=500, alpha=0.05)
    # a 95% CI for a mean over 200 draws should be well inside +-2 sigma
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert (ci["ci_high"] - ci["ci_low"]) < 4 * 100.0


def test_paired_wilcoxon_identical_vectors_is_neutral():
    a = np.arange(50, dtype=float)
    res = significance.paired_wilcoxon(a, a.copy())
    assert res["mean_diff"] == 0.0
    assert res["p_value"] == 1.0


def test_paired_wilcoxon_detects_systematic_gain():
    rng = np.random.default_rng(0)
    a = rng.normal(100, 20, 300)
    b = a - rng.normal(10, 5, 300)  # b systematically worse than a
    res = significance.paired_wilcoxon(a, b)
    assert res["mean_diff"] > 0
    assert res["p_value"] < 0.01


def test_paired_wilcoxon_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="aligned"):
        significance.paired_wilcoxon(np.arange(5), np.arange(4))


def test_rank_correlation_perfect_and_inverted():
    results = pd.DataFrame(
        {
            "roc_auc": [0.70, 0.75, 0.80, 0.85],
            "f1": [0.50, 0.55, 0.60, 0.65],
            "eprofits_tenure": [10.0, 20.0, 30.0, 40.0],
        }
    )
    aligned = significance.rank_correlation(results, "roc_auc", "eprofits_tenure")
    assert aligned["spearman"] == pytest.approx(1.0)

    inverted = results.copy()
    inverted["eprofits_tenure"] = inverted["eprofits_tenure"][::-1].values
    flipped = significance.rank_correlation(inverted, "roc_auc", "eprofits_tenure")
    assert flipped["spearman"] == pytest.approx(-1.0)


def test_rank_correlation_requires_enough_models():
    too_few = pd.DataFrame({"roc_auc": [0.7, 0.8], "eprofits_tenure": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least 3"):
        significance.rank_correlation(too_few)


def test_per_customer_profit_matrix_sums_to_total_eprofits(small_split):
    """The profit matrix must be the per-customer decomposition of the
    e-Profits figure evaluate.py reports - sum over customers == total."""
    from pipeline import evaluate
    from pipeline.models import build_model_registry
    from pipeline.survival import fit_survival_model

    (x_train, x_test, y_train, y_test), sample = small_split
    _, retention_fn, avg_rate = fit_survival_model(x_train, y_train)

    model, _ = build_model_registry()["xgb"]
    model.fit(x_train, y_train)

    matrix = significance.per_customer_profit_matrix(
        {"xgb": model}, x_test, y_test, reference_df=sample,
        retention_fn=retention_fn,
    )
    row = evaluate.evaluate_model("xgb", model, x_test, y_test, sample, retention_fn, avg_rate)

    assert matrix.shape == (len(x_test), 1)
    assert float(matrix["xgb"].sum()) == pytest.approx(row["eprofits_tenure"], rel=1e-9)


def test_run_significance_analysis_writes_the_three_artifacts(tmp_path, small_split):
    from pipeline.models import build_model_registry
    from pipeline.survival import fit_km_survival_model

    (x_train, x_test, y_train, y_test), sample = small_split
    _, retention_fn, _ = fit_km_survival_model(x_train, y_train)

    # Three cheaply-fitted (untuned) models so the rank correlation has
    # enough points; significance needs >= 3 models.
    fitted = {}
    for name in ["xgb", "random_forest", "lightgbm"]:
        model, _ = build_model_registry()[name]
        model.fit(x_train, y_train)
        fitted[name] = model

    results = pd.DataFrame(
        {
            "roc_auc": [0.8, 0.7, 0.75],
            "f1": [0.6, 0.5, 0.55],
            "eprofits_tenure": [30.0, 20.0, 25.0],
        },
        index=list(fitted),
    )

    out = significance.run_significance_analysis(
        fitted, x_test, y_test,
        reference_df=sample, retention_fn=retention_fn,
        results=results, out_dir=tmp_path, n_boot=50, seed=0,
    )

    assert (tmp_path / "bootstrap_ci.csv").exists()
    assert (tmp_path / "wilcoxon_tests.csv").exists()
    assert (tmp_path / "rank_correlation.csv").exists()

    assert set(out["bootstrap"].index) == set(fitted)
    assert out["bootstrap"]["ci_low"].lt(out["bootstrap"]["ci_high"]).all()
    # 3 models -> 3 pairs
    assert len(out["wilcoxon"]) == 3
    assert out["wilcoxon"]["p_value"].between(0.0, 1.0).all()
    assert len(out["rank_correlation"]) == 2


def test_run_significance_analysis_survives_single_model_run(tmp_path, small_split):
    """`--models xgb` alone still finishes: rank correlation degrades to an
    empty table instead of raising after evaluation succeeded."""
    from pipeline.models import build_model_registry
    from pipeline.survival import fit_km_survival_model

    (x_train, x_test, y_train, y_test), sample = small_split
    _, retention_fn, _ = fit_km_survival_model(x_train, y_train)

    model, _ = build_model_registry()["xgb"]
    model.fit(x_train, y_train)

    results = pd.DataFrame(
        {"roc_auc": [0.8], "f1": [0.6], "eprofits_tenure": [30.0]}, index=["xgb"],
    )
    out = significance.run_significance_analysis(
        {"xgb": model}, x_test, y_test,
        reference_df=sample, retention_fn=retention_fn,
        results=results, out_dir=tmp_path, n_boot=20, seed=0,
    )
    assert len(out["wilcoxon"]) == 0
    assert len(out["rank_correlation"]) == 0


def test_per_customer_eprofits_sums_match_compute():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "customer_value": rng.uniform(10, 200, n),
            "retention_rate": rng.uniform(0.5, 0.99, n),
            "true": rng.integers(0, 2, n),
            "predict": rng.integers(0, 2, n),
        }
    )
    from pipeline.eprofits import compute_eprofits

    vec = per_customer_eprofits(df)
    assert vec.shape == (n,)
    assert float(vec.sum()) == pytest.approx(compute_eprofits(df))
