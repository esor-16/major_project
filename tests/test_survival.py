"""Covers synopsis TC2: CoxPH retention probabilities are generated per
customer, bounded in [0,1], and vary with covariates (not a single constant
population-level curve, unlike the Kaplan-Meier baseline)."""
import numpy as np


def test_retention_probabilities_bounded(small_split):
    from pipeline.survival import conditional_retention, fit_survival_model

    (x_train, _, y_train, _), _ = small_split
    _, retention_fn, avg_rate = fit_survival_model(x_train, y_train)

    tenure = x_train["tenure"].to_numpy(dtype=float)
    rates = conditional_retention(retention_fn, tenure, delta=1.0)

    assert np.all(rates >= 0) and np.all(rates <= 1.0001)
    assert 0 <= avg_rate <= 1


def test_survival_function_is_non_increasing(small_split):
    from pipeline.survival import fit_survival_model

    (x_train, _, y_train, _), _ = small_split
    _, retention_fn, _ = fit_survival_model(x_train, y_train)

    ts = np.arange(0, 60, 5, dtype=float)
    s_vals = retention_fn(ts)
    assert np.all(np.diff(s_vals) <= 1e-9)


def test_retention_varies_across_customers(small_split):
    """Per-customer retention differs with tenure - the core improvement
    over Kaplan-Meier's single population-level curve."""
    from pipeline.survival import conditional_retention, fit_survival_model

    (x_train, _, y_train, _), _ = small_split
    _, retention_fn, _ = fit_survival_model(x_train, y_train)

    low_tenure = conditional_retention(retention_fn, np.array([1.0]), delta=1.0)[0]
    high_tenure = conditional_retention(retention_fn, np.array([50.0]), delta=1.0)[0]
    assert low_tenure != high_tenure


def test_coxph_retention_conditions_on_covariates(small_split):
    """THE central CoxPH-vs-KM difference: two customers at the SAME
    tenure get different retention once their covariates are passed -
    S(t | x_i) = S0(t)^exp(x_i . beta), not the covariate-zero baseline
    curve. Pick the training customers with the most extreme linear
    predictors so the assertion can't flake on near-identical rows."""
    from pipeline.survival import conditional_retention, fit_survival_model

    (x_train, _, y_train, _), _ = small_split
    cph, retention_fn, _ = fit_survival_model(x_train, y_train)

    feature_cols = list(cph.params_.index)
    eta = x_train[feature_cols].astype(float).to_numpy() @ cph.params_.to_numpy(dtype=float)
    lo, hi = int(np.argmin(eta)), int(np.argmax(eta))
    assert eta[lo] != eta[hi]  # otherwise this dataset has no covariate signal at all
    rows = x_train.iloc[[lo, hi]]

    rates = conditional_retention(retention_fn, np.array([10.0, 10.0]), delta=1.0, x=rows)
    assert np.all(rates >= 0) and np.all(rates <= 1.0001)
    assert not np.isclose(rates[0], rates[1])

    # ...and without x, the same call falls back to one population curve
    same = conditional_retention(retention_fn, np.array([10.0, 10.0]), delta=1.0)
    assert np.isclose(same[0], same[1])


def test_coxph_survival_fn_non_increasing_with_covariates(small_split):
    from pipeline.survival import fit_survival_model

    (x_train, _, y_train, _), _ = small_split
    _, retention_fn, _ = fit_survival_model(x_train, y_train)

    rows = x_train.iloc[:5]
    ts = np.arange(0, 60, 5, dtype=float)
    s_vals = retention_fn(ts, x=rows.iloc[:1])  # one covariate row broadcast across t
    assert s_vals.shape == ts.shape
    assert np.all(np.diff(s_vals) <= 1e-9)


# ---------------------------------------------------------------------------
# Kaplan-Meier baseline (NFR2: CoxPH-based eProfits vs. Kaplan-Meier-based
# eProfits comparison). Covers the interface contract that makes `--survival
# km` a drop-in swap for `--survival cox` everywhere downstream.
# ---------------------------------------------------------------------------

def test_km_returns_the_same_three_tuple_shape_as_coxph(small_split):
    from pipeline.survival import fit_km_survival_model

    (x_train, _, y_train, _), _ = small_split
    model, retention_fn, avg_rate = fit_km_survival_model(x_train, y_train)

    assert callable(retention_fn)
    assert isinstance(avg_rate, float)
    assert 0 < avg_rate <= 1


def test_km_survival_function_is_non_increasing(small_split):
    from pipeline.survival import fit_km_survival_model

    (x_train, _, y_train, _), _ = small_split
    _, retention_fn, _ = fit_km_survival_model(x_train, y_train)

    ts = np.arange(0, 60, 5, dtype=float)
    s_vals = retention_fn(ts)
    assert np.all(np.diff(s_vals) <= 1e-9)
    assert s_vals[0] <= 1.0001


def test_km_ignores_covariates_every_customer_gets_the_same_curve(small_split):
    """The defining difference from CoxPH: KM has no per-customer
    conditioning, so two customers at the same tenure get identical
    retention regardless of any other covariate."""
    from pipeline.survival import conditional_retention, fit_km_survival_model

    (x_train, _, y_train, _), _ = small_split
    _, retention_fn, _ = fit_km_survival_model(x_train, y_train)

    same_tenure = np.array([10.0, 10.0, 10.0])
    rates = conditional_retention(retention_fn, same_tenure, delta=1.0)
    assert np.allclose(rates, rates[0])


def test_km_retention_bounded(small_split):
    from pipeline.survival import conditional_retention, fit_km_survival_model

    (x_train, _, y_train, _), _ = small_split
    _, retention_fn, avg_rate = fit_km_survival_model(x_train, y_train)

    tenure = x_train["tenure"].to_numpy(dtype=float)
    rates = conditional_retention(retention_fn, tenure, delta=1.0)

    assert np.all(rates >= 0) and np.all(rates <= 1.0001)
    assert 0 <= avg_rate <= 1


def test_km_ignores_passed_covariates(small_split):
    """Interface parity with CoxPH: passing x= changes nothing for KM."""
    from pipeline.survival import conditional_retention, fit_km_survival_model

    (x_train, _, y_train, _), _ = small_split
    _, retention_fn, _ = fit_km_survival_model(x_train, y_train)

    ten = np.array([10.0, 10.0, 10.0])
    with_x = conditional_retention(retention_fn, ten, delta=1.0, x=x_train.iloc[:3])
    without_x = conditional_retention(retention_fn, ten, delta=1.0)
    assert np.allclose(with_x, without_x)
