"""Validates the vectorized eProfits engine against a naive per-row
reimplementation matching the Phase-1 notebook's original loop-based logic,
to confirm the vectorization (needed to keep GA search fast across six
models) didn't change the result.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.eprofits import compute_eprofits


def _naive_eprofits(df, profit_margin=0.3, cost_of_contact=(0, 0.3), cpo=0.1):
    profits = []
    for i in range(len(df)):
        monthly_revenue = df.iloc[i]["customer_value"]
        retention_rate = min(0.995, df.iloc[i]["retention_rate"])
        clv = (monthly_revenue * profit_margin) / (1 - retention_rate)
        cost_per_offer = cpo * clv
        contact_cost = max(cost_of_contact[0], cost_of_contact[1] * cost_per_offer)

        if df.iloc[i]["true"] == 1 and df.iloc[i]["predict"] == df.iloc[i]["true"]:
            profit = clv - cost_per_offer - contact_cost
        elif df.iloc[i]["predict"] == 1:
            profit = -cost_per_offer - contact_cost
        else:
            profit = 0
        profits.append(profit)
    return sum(profits)


@pytest.fixture
def sample_scoring_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "customer_value": rng.uniform(10, 200, n),
            "retention_rate": rng.uniform(0.5, 0.999, n),
            "true": rng.integers(0, 2, n),
            "predict": rng.integers(0, 2, n),
        }
    )


def test_vectorized_matches_naive_reference(sample_scoring_df):
    vectorized = compute_eprofits(sample_scoring_df)
    naive = _naive_eprofits(sample_scoring_df)
    assert vectorized == pytest.approx(naive, rel=1e-9)


def test_no_intervention_customers_contribute_zero():
    df = pd.DataFrame(
        {"customer_value": [100.0], "retention_rate": [0.8], "true": [0], "predict": [0]}
    )
    assert compute_eprofits(df) == 0.0


def test_correctly_flagged_churner_is_profitable():
    df = pd.DataFrame(
        {"customer_value": [100.0], "retention_rate": [0.8], "true": [1], "predict": [1]}
    )
    assert compute_eprofits(df) > 0


def test_false_positive_is_a_loss():
    df = pd.DataFrame(
        {"customer_value": [100.0], "retention_rate": [0.8], "true": [0], "predict": [1]}
    )
    assert compute_eprofits(df) < 0
