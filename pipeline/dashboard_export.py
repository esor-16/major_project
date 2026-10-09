"""Exports one JSON artifact (`artifacts/dashboard.json`) summarizing a
pipeline run for the static web frontend (`frontend/`) to consume: the
six-model comparison table, the best model's SHAP explanations (global +
a ranked list of real customers), and dataset-level stats - all in one
file so the frontend doesn't need to parse three separate CSVs or run any
Python itself.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import explain


def _decode(encoders: dict, column: str, code) -> str:
    """Turn a label-encoded value (e.g. Contract=0) back into its original
    string ("Month-to-month"), using the LabelEncoder captured during
    `data.preprocess(..., return_encoders=True)`. Falls back to the raw
    code if the column wasn't categorical or the code is out of range.
    """
    encoder = encoders.get(column)
    if encoder is None:
        return str(code)
    try:
        return str(encoder.classes_[int(code)])
    except (IndexError, ValueError, TypeError):
        return str(code)


def build_dataset_summary(processed: pd.DataFrame, dataset_name: str, avg_retention_rate: float | None = None) -> dict:
    return {
        "name": dataset_name,
        "rows": int(len(processed)),
        "columns": int(processed.shape[1]),
        "churn_count": int(processed["Churn"].sum()),
        "churn_rate": float(processed["Churn"].mean()),
        "avg_retention_rate": float(avg_retention_rate) if avg_retention_rate is not None else None,
        "avg_tenure": float(processed["tenure"].mean()),
        "avg_monthly_charge": float(processed["MonthlyCharges"].mean()),
    }


def build_customer_records(
    explanation,
    explain_x: pd.DataFrame,
    y_explained: pd.Series,
    fitted_model,
    encoders: dict,
    top_n_features: int = 5,
    max_customers: int = 25,
) -> list[dict]:
    """Real, SHAP-explained customers, ranked by predicted churn risk (highest
    first) so the frontend's customer list surfaces the priority cases.

    `explain_x` must be the exact rows the explanation was computed over
    (same order), so row `i` in `explanation` lines up with `explain_x.iloc[i]`.
    """
    proba = fitted_model.predict_proba(explain_x)[:, 1]
    order = np.argsort(-proba)[:max_customers]

    records = []
    for i in order:
        i = int(i)
        row = explain_x.iloc[i]
        top_features = explain.explain_customer(explanation, i, top_n=top_n_features)
        records.append(
            {
                "id": int(explain_x.index[i]),
                "churn_probability": float(proba[i]),
                "actual_churn": int(y_explained.iloc[i]),
                "tenure": float(row.get("tenure", float("nan"))),
                "monthly_charges": float(row.get("MonthlyCharges", float("nan"))),
                "contract": _decode(encoders, "Contract", row.get("Contract")),
                "internet_service": _decode(encoders, "InternetService", row.get("InternetService")),
                "payment_method": _decode(encoders, "PaymentMethod", row.get("PaymentMethod")),
                "tech_support": _decode(encoders, "TechSupport", row.get("TechSupport")),
                "online_security": _decode(encoders, "OnlineSecurity", row.get("OnlineSecurity")),
                "top_features": [
                    {
                        "feature": r.feature,
                        # Categorical features (Contract, InternetService, ...) get
                        # decoded to their original label; continuous ones (tenure,
                        # MonthlyCharges, ...) stay numeric.
                        "value": _decode(encoders, r.feature, r.value) if r.feature in encoders else float(r.value),
                        "shap_value": float(r.shap_value),
                        "direction": r.direction,
                    }
                    for r in top_features.itertuples(index=False)
                ],
            }
        )
    return records


def export_dashboard(
    out_path: str,
    dataset_summary: dict,
    results: pd.DataFrame,
    best_model_name: str,
    global_importance: pd.DataFrame,
    customer_records: list[dict],
) -> None:
    payload = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_summary,
        "best_model": best_model_name,
        "models": [
            {"name": name, **{k: float(v) for k, v in row.items()}}
            for name, row in results.to_dict(orient="index").items()
        ],
        "shap_global": global_importance.to_dict(orient="records"),
        "shap_customers": customer_records,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
