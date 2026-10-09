"""Cox Proportional Hazards survival analysis for customer retention -
the project's BASE survival mechanism.

Fits CoxPH on every training covariate and derives per-customer retention
probabilities. Unlike the paper's population-level Kaplan-Meier curve (kept
as the `--survival km` baseline for comparison), CoxPH conditions each
customer's survival on their OWN covariates:

    S(t | x_i) = S0(t) ^ exp(eta_i),   eta_i = x_i . beta

where S0(t) is the Breslow baseline survival estimated by lifelines'
`CoxPHFitter.baseline_survival_` and eta_i is the linear predictor for
customer i. The conditional one-period retention handed to e-Profits is
therefore different for two customers at the SAME tenure but different
covariates - genuinely individualised, which is what makes the CLV and
per-customer profit decomposition business-centered (a high-risk customer
gets a lower retention probability, hence a higher churn probability and
a different CLV, than a low-risk customer of the same tenure).

Historical note: an earlier version looked up only
`cph.baseline_survival_` (the covariate-zero reference curve) indexed by
tenure, so the retention reaching e-Profits was a single population curve
despite the model being fit on covariates - the README's old "CoxPH vs KM
isn't yet individualised" caveat. That is fixed here: `survival_fn` now
takes optional per-customer covariates `x` and raises if they're missing
fitted columns.

Fit on the *pre-resampling* training split (before SMOTE/ADASYN):
synthetic minority-class rows don't correspond to real time-to-churn
observations, so including them would bias the survival estimate.

Interface contract (both fitters return the same 3-tuple):
    (model, survival_fn, avg_retention_rate)
`survival_fn(t, x=None)` -> S(t) for the population when `x` is None, or
per-customer S(t | x_i) when `x` is a DataFrame whose rows align with `t`.
`conditional_retention(retention_fn, tenure, delta, x=None)` computes
S(t+delta)/S(t) elementwise, passing `x` through when given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

# Guards for the exp()/log() in the Cox per-customer path: |eta| beyond 30
# implies a hazard ratio outside ~1e-13..1e13, and S0 below 1e-300 would
# underflow to 0 in log space. Neither occurs on sane data; the clips only
# prevent inf/nan from poisoning a whole scoring batch if it did.
_MAX_ETA = 30.0
_MIN_BASELINE = 1e-300


def conditional_retention(retention_fn, tenure, delta: float = 1.0, eps: float = 1e-12, x=None):
    """P(customer survives `delta` more periods | survived to `tenure`)
    = S(t+delta) / S(t).

    Pass `x` (a DataFrame of per-customer covariates aligned with
    `tenure`) to condition on each customer's own features - the CoxPH
    path uses it to evaluate S(t | x_i); the Kaplan-Meier path ignores it
    (one population curve). When `x` is None the plain tenure-only
    `retention_fn(t)` is called, preserving the population-level
    behaviour.
    """
    ten = np.asarray(tenure, dtype=float)
    if x is None:
        s_t = np.asarray(retention_fn(ten), dtype=float)
        s_t_delta = np.asarray(retention_fn(ten + delta), dtype=float)
    else:
        s_t = np.asarray(retention_fn(ten, x=x), dtype=float)
        s_t_delta = np.asarray(retention_fn(ten + delta, x=x), dtype=float)
    return s_t_delta / np.maximum(s_t, eps)


def fit_survival_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    tenure_column: str = "tenure",
    delta: float = 1.0,
    penalizer: float = 0.1,
):
    """Fit CoxPH on the training covariates and derive:
    - a per-customer survival function S(t | x) = S0(t)^exp(x . beta),
    - the average delta-step retention rate over the training population.

    `survival_fn(t)` (no x) still works and returns the baseline
    population curve, so callers that only have tenures keep working;
    callers with covariates pass `x=` to get individualised retention.

    Returns: (cph, survival_fn, avg_retention_rate)
    """
    tenure = x_train[tenure_column].values.astype(float)
    event_observed = (y_train == 1).astype(int).values

    cox_df = x_train.copy()
    cox_df["_duration"] = tenure
    cox_df["_event"] = event_observed

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(cox_df, duration_col="_duration", event_col="_event")

    baseline_sf = cph.baseline_survival_  # DataFrame indexed by time, single column
    feature_cols = list(cph.params_.index)
    beta = cph.params_.to_numpy(dtype=float)

    def _baseline_at(t_arr: np.ndarray) -> np.ndarray:
        idx = baseline_sf.index.searchsorted(t_arr, side="right") - 1
        idx = np.clip(idx, 0, len(baseline_sf) - 1)
        return baseline_sf.iloc[idx, 0].to_numpy(dtype=float)

    def survival_fn(t, x=None):
        """S(t) for the population when x is None, else per-customer
        S(t | x_i) at each row's own time t_i.

        Shapes: t may be scalar or array; when x is given it must have
        one row per entry of t (the conditional-retention use), or a
        single row to be broadcast across t.
        """
        scalar_t = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        base = np.clip(_baseline_at(t_arr), _MIN_BASELINE, 1.0)

        if x is None:
            vals = base
        else:
            x_df = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
            missing = [c for c in feature_cols if c not in x_df.columns]
            if missing:
                raise ValueError(
                    f"survival_fn received covariates missing fitted CoxPH columns: {missing}"
                )
            if len(x_df) not in (1, len(t_arr)):
                raise ValueError(
                    f"survival_fn got {len(t_arr)} time points but {len(x_df)} covariate rows; "
                    "they must align (or x must have exactly one row to broadcast)."
                )
            eta = x_df[feature_cols].astype(float).to_numpy() @ beta
            hazard_ratio = np.exp(np.clip(eta, -_MAX_ETA, _MAX_ETA))
            # S(t|x) = S0(t)^HR, computed in log space for stability.
            # Elementwise with broadcasting, so it covers the paired case
            # (len(x) == len(t): row i gets time i), a single x row
            # broadcast across many times, and one time across many rows.
            # An outer product here would be WRONG for the paired case.
            vals = np.exp(hazard_ratio * np.log(base))

        if scalar_t and (x is None or np.ndim(vals) == 0 or len(np.atleast_1d(vals)) == 1):
            return float(np.atleast_1d(vals)[0])
        return vals

    trr_train = conditional_retention(survival_fn, tenure, delta=delta, x=x_train)
    avg_retention_rate = float(np.mean(trr_train))

    return cph, survival_fn, avg_retention_rate


def fit_km_survival_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    tenure_column: str = "tenure",
    delta: float = 1.0,
):
    """Kaplan-Meier baseline (the base eProfits paper's original approach),
    fit on the whole training population without using covariates at all.

    Returns the SAME (model, survival_fn, avg_retention_rate) 3-tuple shape
    as `fit_survival_model`, so it's a drop-in swap everywhere downstream
    (eprofits.py, evaluate.py, significance.py, run.py) - `--survival km`
    changes nothing but which function produced `retention_fn`. That makes
    CoxPH-vs-KM a direct diff of two otherwise-identical pipeline runs.

    `survival_fn(t, x=None)` accepts and IGNORES `x`: every customer at a
    given tenure gets the same retention probability, which is exactly the
    difference the NFR2 comparison is meant to expose against CoxPH's
    per-customer conditioning.
    """
    tenure = x_train[tenure_column].values.astype(float)
    event_observed = (y_train == 1).astype(int).values

    kmf = KaplanMeierFitter()
    kmf.fit(durations=tenure, event_observed=event_observed)

    survival_table = kmf.survival_function_  # DataFrame indexed by time, single column

    def survival_fn(t, x=None):
        """Population survival probability S(t) at scalar or array-valued t.
        `x` is accepted for interface parity with CoxPH's survival_fn and
        deliberately ignored - KM is population-level by definition.
        """
        scalar_t = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        idx = survival_table.index.searchsorted(t_arr, side="right") - 1
        idx = np.clip(idx, 0, len(survival_table) - 1)
        vals = survival_table.iloc[idx, 0].to_numpy(dtype=float)
        return float(vals[0]) if scalar_t else vals

    trr_train = conditional_retention(survival_fn, tenure, delta=delta)
    avg_retention_rate = float(np.mean(trr_train))

    return kmf, survival_fn, avg_retention_rate
