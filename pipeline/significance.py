"""Statistical validation of profit-based model comparisons, mirroring the
paper's Sect. 4.3:

1. Bootstrap confidence intervals for each model's test-set e-Profits
   (500 resamples of the held-out test customers, percentile 95% CI).
2. Paired Wilcoxon signed-rank tests on per-customer profit differences
   between models (are profit gains systematic rather than noise?).
3. Spearman rank correlation between predictive (ROC-AUC) rankings and
   profit-based (e-Profits) rankings - quantifies the paper's central
   claim that profit-aware evaluation re-ranks models relative to
   traditional metrics.

All three operate on the held-out test split only, using per-customer
profit vectors from `eprofits.per_customer_eprofits` with tenure-conditioned
(TRR) retention - the headline e-Profits figure.

Artifacts written by `run_significance_analysis`:
- bootstrap_ci.csv       model, mean, ci_low, ci_high, n_boot
- wilcoxon_tests.csv     model_a, model_b, mean_diff, p_value
- rank_correlation.csv   metric_a, metric_b, spearman, p_value
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .eprofits import per_customer_eprofits
from .survival import conditional_retention

DEFAULT_N_BOOT = 500


def bootstrap_profit_ci(profits, n_boot: int = DEFAULT_N_BOOT, alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile bootstrap CI for the mean per-customer profit.

    Resamples the per-customer profit vector with replacement `n_boot`
    times and takes the (alpha/2, 1-alpha/2) percentiles of the resampled
    means. Profit outcomes are inherently more variable than predictive
    metrics, so the paper reports these intervals to show observed profit
    gaps aren't sampling noise.
    """
    profits = np.asarray(profits, dtype=float)
    if len(profits) == 0:
        raise ValueError("bootstrap_profit_ci needs at least one per-customer profit")
    rng = np.random.default_rng(seed)
    n = len(profits)
    # Chunked to bound memory: (n_boot, n) floats at once is fine for the
    # IBM test set (~2k rows) but wasteful on larger future datasets.
    means = np.empty(n_boot)
    chunk = max(1, min(n_boot, 10_000_000 // max(n, 1)))
    start = 0
    while start < n_boot:
        stop = min(start + chunk, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = profits[idx].mean(axis=1)
        start = stop
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(profits.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
    }


def paired_wilcoxon(profits_a, profits_b) -> dict:
    """Paired Wilcoxon signed-rank test on per-customer profit differences
    (model A vs model B over the SAME test customers).

    Returns mean difference (A - B) and the two-sided p-value. If every
    difference is exactly zero the test is undefined; that case means the
    two models made identical targeting decisions, so p is reported as 1.0
    with a mean diff of 0.
    """
    a = np.asarray(profits_a, dtype=float)
    b = np.asarray(profits_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired_wilcoxon needs aligned profit vectors, got {a.shape} vs {b.shape}")
    diff = a - b
    mean_diff = float(diff.mean())
    if not np.any(diff != 0):
        return {"mean_diff": mean_diff, "p_value": 1.0}
    result = stats.wilcoxon(a, b, alternative="two-sided")
    return {"mean_diff": mean_diff, "p_value": float(result.pvalue)}


def rank_correlation(results: pd.DataFrame, metric_a: str = "roc_auc", metric_b: str = "eprofits_tenure") -> dict:
    """Spearman rank correlation between two evaluation metrics across models.

    A value well below 1.0 means the two metrics rank the models
    differently - the paper's model-re-ranking claim (it reports ~0.39-0.49
    between AUC and e-Profits rankings).
    """
    if metric_a not in results.columns or metric_b not in results.columns:
        raise ValueError(f"rank_correlation needs columns {metric_a!r} and {metric_b!r}; have {list(results.columns)}")
    if len(results) < 3:
        raise ValueError(f"rank_correlation needs at least 3 models, got {len(results)}")
    res = stats.spearmanr(results[metric_a], results[metric_b])
    return {"metric_a": metric_a, "metric_b": metric_b, "spearman": float(res.statistic), "p_value": float(res.pvalue)}


def per_customer_profit_matrix(
    fitted_models: dict,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    reference_df: pd.DataFrame,
    retention_fn,
    tenure_column: str = "tenure",
    delta: float = 1.0,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-customer e-Profits (TRR-based, business threshold 0.5) for every
    model, as one row per model aligned on the test-set index - the matrix
    the bootstrap/Wilcoxon tests consume.
    """
    tenure = reference_df.loc[x_test.index, tenure_column].astype(float).to_numpy()
    # reference_df may carry columns the survival model never fit on (e.g.
    # Churn); conditional_retention's Cox path picks out the fitted ones.
    rows = {}
    for name, model in fitted_models.items():
        y_scores = model.predict_proba(x_test)[:, 1]
        pred = (y_scores >= threshold).astype(int)
        scoring_df = reference_df.loc[x_test.index, ["customer_value"]].copy()
        scoring_df["retention_rate"] = conditional_retention(
            retention_fn, tenure, delta=delta, x=reference_df.loc[x_test.index]
        )
        scoring_df["true"] = np.asarray(y_test)
        scoring_df["predict"] = pred
        rows[name] = per_customer_eprofits(scoring_df)
    return pd.DataFrame(rows, index=x_test.index)


def run_significance_analysis(
    fitted_models: dict,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    reference_df: pd.DataFrame,
    retention_fn,
    results: pd.DataFrame,
    out_dir,
    tenure_column: str = "tenure",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
) -> dict:
    """Compute all three analyses and write their CSVs to `out_dir`.

    Returns {"bootstrap": DataFrame, "wilcoxon": DataFrame, "rank_correlation": DataFrame}.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profits = per_customer_profit_matrix(
        fitted_models, x_test, y_test, reference_df, retention_fn, tenure_column=tenure_column,
    )

    bootstrap_rows = [
        {"model": name, **bootstrap_profit_ci(profits[name].to_numpy(), n_boot=n_boot, seed=seed)}
        for name in profits.columns
    ]
    bootstrap = pd.DataFrame(bootstrap_rows).set_index("model")

    wilcoxon_rows = []
    for a, b in itertools.combinations(profits.columns, 2):
        wilcoxon_rows.append({"model_a": a, "model_b": b, **paired_wilcoxon(profits[a].to_numpy(), profits[b].to_numpy())})
    wilcoxon = pd.DataFrame(wilcoxon_rows, columns=["model_a", "model_b", "mean_diff", "p_value"])

    # Rank correlation is a comparison ACROSS models - meaningless with
    # fewer than three, so degrade gracefully (a --models xgb run still
    # finishes) instead of raising after evaluation has already succeeded.
    corr_rows = []
    if len(results) >= 3:
        corr_rows = [
            rank_correlation(results, "roc_auc", "eprofits_tenure"),
            rank_correlation(results, "f1", "eprofits_tenure"),
        ]
    else:
        print("  (skipping rank correlation: needs at least 3 models)")
    rank_corr = pd.DataFrame(corr_rows, columns=["metric_a", "metric_b", "spearman", "p_value"])

    bootstrap.to_csv(out_dir / "bootstrap_ci.csv")
    wilcoxon.to_csv(out_dir / "wilcoxon_tests.csv", index=False)
    rank_corr.to_csv(out_dir / "rank_correlation.csv", index=False)

    return {"bootstrap": bootstrap, "wilcoxon": wilcoxon, "rank_correlation": rank_corr}
