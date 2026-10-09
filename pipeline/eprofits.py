"""eProfits business metric engine: turns model predictions + survival-
model retention probabilities into a profitability figure, plus GA/
GridSearch-compatible scorer factories (eProfits_avg, eProfits_tenure,
EMP).

Vectorized (no per-row Python loop) so it stays cheap enough to call
thousands of times during hyperparameter search across six models -
the Phase-1 notebook's per-row loop was fine for two models but would not
meet the Phase-2 pipeline-runtime NFR once scaled to six.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from EMP.metrics import empChurn
from sklearn.metrics import f1_score

from .survival import conditional_retention

DEFAULT_PROFIT_MARGIN = 0.3
DEFAULT_COST_OF_CONTACT = (0, 0.3)
DEFAULT_CPO = 0.1
MAX_RETENTION_RATE = 0.995  # retention rate can never be 1 (division by zero in CLV)


def estimate_clv_and_cost(customer_value, retention_rate, profit_margin: float = DEFAULT_PROFIT_MARGIN, cpo: float = DEFAULT_CPO):
    """CLV and per-offer intervention cost, vectorized over one or many customers."""
    retention_rate = np.minimum(MAX_RETENTION_RATE, np.asarray(retention_rate, dtype=float))
    clv = (np.asarray(customer_value, dtype=float) * profit_margin) / (1 - retention_rate)
    cost_per_offer = cpo * clv
    return clv, cost_per_offer


def per_customer_eprofits(
    df: pd.DataFrame,
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
    cost_of_contact: tuple[float, float] = DEFAULT_COST_OF_CONTACT,
    cpo: float = DEFAULT_CPO,
) -> np.ndarray:
    """Per-customer profit/loss array (the sum of which is e-Profits).

    `df` must have columns: customer_value, retention_rate, true, predict.
    - true churner correctly flagged (true=1, predict=1): profit = CLV - offer cost - contact cost
    - non-churner wrongly flagged (true=0, predict=1): profit = -offer cost - contact cost
    - not flagged (predict=0): no intervention, profit = 0

    Kept separate from `compute_eprofits` so statistical comparisons
    (bootstrap CIs, paired Wilcoxon tests - see significance.py) can work
    on the per-customer distribution instead of only its sum.
    """
    clv, cost_per_offer = estimate_clv_and_cost(
        df["customer_value"].to_numpy(dtype=float),
        df["retention_rate"].to_numpy(dtype=float),
        profit_margin,
        cpo,
    )
    cost_of_contact_arr = np.maximum(cost_of_contact[0], cost_of_contact[1] * cost_per_offer)

    true = df["true"].to_numpy()
    predict = df["predict"].to_numpy()

    profit = np.zeros(len(df))
    churner_hit = (true == 1) & (predict == 1)
    false_positive = (true == 0) & (predict == 1)

    profit[churner_hit] = clv[churner_hit] - cost_per_offer[churner_hit] - cost_of_contact_arr[churner_hit]
    profit[false_positive] = -cost_per_offer[false_positive] - cost_of_contact_arr[false_positive]

    return profit


def compute_eprofits(
    df: pd.DataFrame,
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
    cost_of_contact: tuple[float, float] = DEFAULT_COST_OF_CONTACT,
    cpo: float = DEFAULT_CPO,
) -> float:
    """Sum per-customer profit/loss from acting on model predictions.

    See `per_customer_eprofits` for the per-customer contribution rules.
    """
    return float(per_customer_eprofits(df, profit_margin, cost_of_contact, cpo).sum())


def _score_with_retention(estimator, x, y, reference_df, retention_rate, threshold, profit_margin, cost_of_contact, cpo):
    y_scores = estimator.predict_proba(x)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    scoring_df = reference_df.loc[y.index, ["customer_value"]].copy()
    scoring_df["retention_rate"] = retention_rate
    scoring_df["true"] = np.asarray(y)
    scoring_df["predict"] = y_pred
    return compute_eprofits(scoring_df, profit_margin, cost_of_contact, cpo)


def make_eprofits_scorer_tenure(
    reference_df: pd.DataFrame,
    retention_fn,
    tenure_column: str = "tenure",
    delta: float = 1.0,
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
    cost_of_contact: tuple[float, float] = DEFAULT_COST_OF_CONTACT,
    cpo: float = DEFAULT_CPO,
    threshold: float = 0.5,
):
    """Scorer using per-customer, tenure-dependent retention probability.

    The reference rows' covariates are handed to `retention_fn` as well, so
    the CoxPH base evaluates S(t | x_i) per customer (population curve for
    the KM path, which ignores them).
    """

    def scorer(estimator, x, y):
        ref_rows = reference_df.loc[y.index]
        tenure = ref_rows[tenure_column].astype(float).to_numpy()
        retention_rate = conditional_retention(retention_fn, tenure, delta=delta, x=ref_rows)
        return _score_with_retention(estimator, x, y, reference_df, retention_rate, threshold, profit_margin, cost_of_contact, cpo)

    return scorer


def make_composite_scorer_tenure(
    reference_df: pd.DataFrame,
    retention_fn,
    tenure_column: str = "tenure",
    delta: float = 1.0,
    f1_weight: float = 0.5,
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
    cost_of_contact: tuple[float, float] = DEFAULT_COST_OF_CONTACT,
    cpo: float = DEFAULT_CPO,
    threshold: float = 0.5,
):
    """A single scorer that blends eProfits and F1 into one scalar, so GA
    hyperparameter search actually balances profit AND classification
    quality - not just eProfits, with F1 reported afterwards as an
    afterthought at whatever threshold falls out of an eProfits-only search.

    `f1_weight` (0-1, default 0.5) sets the blend: 0.0 reduces to pure
    eProfits (`make_eprofits_scorer_tenure`'s behaviour), 1.0 to pure F1,
    0.5 weighs them equally. This is the single place that decides how much
    F1 matters during GA tuning - it's meant to be used for every model
    (baselines and feature-fusion hybrids alike) so any resulting comparison
    reflects a real difference between them, not different tuning targets.

    Scale mismatch is the reason this can't just be a weighted sum of raw
    eProfits (order 1e6-1e7) and F1 (0-1): F1's contribution would be
    invisible at that scale. Instead, eProfits is normalized against the
    *perfect-classifier ceiling* for this exact batch of customers - the
    eProfits achievable by flagging precisely the true churners and no one
    else, which is provably the profit-maximizing prediction pattern for
    this formula (every correctly-flagged churner adds a positive amount,
    every false positive only subtracts a small offer/contact cost) - so
    `eprofits / ceiling` lands in roughly the same [~0, 1] range as F1
    without needing a fixed, dataset-specific magic-number scale.
    """
    if not 0.0 <= f1_weight <= 1.0:
        raise ValueError(f"f1_weight must be in [0, 1], got {f1_weight!r}")

    def scorer(estimator, x, y):
        ref_rows = reference_df.loc[y.index]
        tenure = ref_rows[tenure_column].astype(float).to_numpy()
        retention_rate = conditional_retention(retention_fn, tenure, delta=delta, x=ref_rows)

        y_scores = estimator.predict_proba(x)[:, 1]
        y_pred = (y_scores >= threshold).astype(int)

        scoring_df = reference_df.loc[y.index, ["customer_value"]].copy()
        scoring_df["retention_rate"] = retention_rate
        scoring_df["true"] = np.asarray(y)
        scoring_df["predict"] = y_pred
        eprofits = compute_eprofits(scoring_df, profit_margin, cost_of_contact, cpo)

        ceiling_df = scoring_df.copy()
        ceiling_df["predict"] = ceiling_df["true"]  # perfect classifier: flag exactly the true churners
        eprofits_ceiling = compute_eprofits(ceiling_df, profit_margin, cost_of_contact, cpo)
        eprofits_norm = eprofits / eprofits_ceiling if eprofits_ceiling > 0 else 0.0

        f1 = f1_score(y, y_pred, zero_division=0)
        return (1 - f1_weight) * eprofits_norm + f1_weight * f1

    return scorer


def make_eprofits_scorer_avg(
    reference_df: pd.DataFrame,
    avg_retention_rate: float,
    profit_margin: float = DEFAULT_PROFIT_MARGIN,
    cost_of_contact: tuple[float, float] = DEFAULT_COST_OF_CONTACT,
    cpo: float = DEFAULT_CPO,
    threshold: float = 0.5,
):
    """Scorer using a single population-average retention rate for every customer."""

    def scorer(estimator, x, y):
        return _score_with_retention(estimator, x, y, reference_df, float(avg_retention_rate), threshold, profit_margin, cost_of_contact, cpo)

    return scorer


def make_emp_scorer(alpha: float = 6, beta: float = 14, clv: float = 200, d: float = 10, f: float = 1):
    """Expected Maximum Profit (EMP) scorer, via the EMP-PY package."""

    def scorer(estimator, x, y):
        y_proba = estimator.predict_proba(x)[:, 1]
        return empChurn(
            y_proba, y, alpha=alpha, beta=beta, clv=clv, d=d, f=f,
            print_output=False, return_output=True, rounding=None,
        ).EMP

    return scorer
