"""Data ingestion, preprocessing, and class-imbalance handling.

Preprocessing logic is carried over from the Phase-1 notebook's IBM-specific
cleaning cell, generalized so it runs unmodified against any dataset that
shares the IBM Telco column schema (a binary Churn label, tenure,
MonthlyCharges, TotalCharges, plus the usual categorical service columns).
This lets Phase-3 add a second dataset (e.g. Maven) without touching this
module, per NFR4/TC9.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

REQUIRED_COLUMNS = {"tenure", "MonthlyCharges", "TotalCharges", "Churn"}

# Columns that encode "no service" as a third category instead of "No" -
# collapsing this avoids spurious extra dummy columns after one-hot encoding.
NO_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_dataset(path: str) -> pd.DataFrame:
    """Load a raw telecom churn CSV and validate it has the expected schema."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset {path!r} is missing required columns: {sorted(missing)}")
    return df


def preprocess(df: pd.DataFrame, id_column: str = "customerID", return_encoders: bool = False):
    """Clean, encode, and feature-engineer a raw telecom churn dataframe.

    Steps: drop the customer id, collapse "No internet/phone service" into
    "No", coerce TotalCharges to numeric, label-encode categoricals, one-hot
    encode any remaining multi-class categoricals, drop rows with residual
    nulls, and derive the `customer_value` feature (the monthly-revenue
    CLV input R_i of the paper's Eq. 3) used throughout the survival/
    eProfits layers.

    return_encoders: when True, also returns the fitted LabelEncoders keyed
    by column name, so callers (e.g. the dashboard export) can decode a
    numeric code like Contract=0 back to "Month-to-month" for display.
    """
    df = df.copy()

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    present_no_service_cols = [c for c in NO_SERVICE_COLUMNS if c in df.columns]
    if present_no_service_cols:
        df[present_no_service_cols] = df[present_no_service_cols].replace("No internet service", "No")
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    df["TotalCharges"] = df["TotalCharges"].replace(" ", 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    categorical_columns = df.select_dtypes(include=["object"]).columns
    encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        encoders[column] = encoder

    df = df.dropna().reset_index(drop=True)
    df = pd.get_dummies(df, drop_first=True)

    # `customer_value` is the CLV input R_i for the e-Profits formula
    # (eprofits.estimate_clv_and_cost: CLV = R_i * M / (1 - r_i)). The
    # paper's Eq. 3 uses the customer's MONTHLY revenue here: the
    # 1/(1 - r_i) denominator already capitalises the future stream, so
    # multiplying by tenure as well (the old
    # `MonthlyCharges * (tenure + 1)`) double-counted tenure and inflated
    # every CLV - and therefore every e-Profits figure - by ~(tenure+1)x.
    df["customer_value"] = df["MonthlyCharges"]

    if return_encoders:
        return df, encoders

    return df


def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.3,
    random_state: int = 42,
):
    """70:30 stratified train/test split (stratified so the already-imbalanced
    churn rate is preserved in both splits before any resampling is applied).
    """
    x = df.drop(columns=[target])
    y = df[target]
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def balance_classes(x_train: pd.DataFrame, y_train: pd.Series, method: str | None = "smote", random_state: int = 42):
    """Correct churn/non-churn class imbalance on the *training* split only.

    method: "smote" | "adasyn" | "smoteenn" | None (no-op).

    Note: features are already one-hot/label-encoded numeric columns, so
    plain SMOTE/ADASYN can interpolate across them directly; this is the
    standard (if imperfect, for one-hot columns) approach and matches what
    the synopsis specifies for Phase-2 - SMOTENC-style categorical-aware
    resampling is a possible future refinement, not required here.
    """
    if method is None:
        return x_train, y_train

    method = method.lower()
    if method == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=random_state)
    elif method == "adasyn":
        from imblearn.over_sampling import ADASYN

        sampler = ADASYN(random_state=random_state)
    elif method == "smoteenn":
        from imblearn.combine import SMOTEENN

        sampler = SMOTEENN(random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing method: {method!r}")

    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    return x_resampled, y_resampled
