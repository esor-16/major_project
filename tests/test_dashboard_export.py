"""Covers the frontend data export: dataset/model/SHAP summaries get turned
into the exact JSON shape `frontend/app.js` expects, and categorical SHAP
feature values are decoded back to their original labels (not left as
raw label-encoded numbers).
"""
import json

import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from pipeline import dashboard_export, explain
from pipeline.models import build_model_registry


def test_build_dataset_summary_reports_real_stats(processed_df):
    summary = dashboard_export.build_dataset_summary(processed_df, dataset_name="IBM", avg_retention_rate=0.95)
    assert summary["name"] == "IBM"
    assert summary["rows"] == len(processed_df)
    assert summary["churn_count"] == int(processed_df["Churn"].sum())
    assert summary["churn_rate"] == pytest.approx(processed_df["Churn"].mean())
    assert summary["avg_retention_rate"] == 0.95


def test_build_dataset_summary_handles_missing_retention_rate(processed_df):
    summary = dashboard_export.build_dataset_summary(processed_df, dataset_name="IBM")
    assert summary["avg_retention_rate"] is None


def test_decode_turns_codes_back_into_labels():
    encoder = LabelEncoder().fit(["DSL", "Fiber optic", "No"])
    encoders = {"InternetService": encoder}
    assert dashboard_export._decode(encoders, "InternetService", 1) == "Fiber optic"


def test_decode_falls_back_for_unknown_column():
    assert dashboard_export._decode({}, "tenure", 12) == "12"


def test_decode_falls_back_for_out_of_range_code():
    encoder = LabelEncoder().fit(["DSL", "Fiber optic"])
    encoders = {"InternetService": encoder}
    assert dashboard_export._decode(encoders, "InternetService", 99) == "99"


def test_build_customer_records_decodes_categorical_top_features(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    registry = build_model_registry()
    model, _ = registry["xgb"]
    model.fit(x_train, y_train)

    explainer = explain.build_explainer("xgb", model, background=x_train)
    explanation = explain.compute_shap_values(explainer, x_test, sample_size=None)

    contract_encoder = LabelEncoder().fit(["Month-to-month", "One year", "Two year"])
    encoders = {"Contract": contract_encoder}

    records = dashboard_export.build_customer_records(
        explanation, x_test, y_test, model, encoders, top_n_features=5, max_customers=5,
    )

    assert len(records) == 5
    for record in records:
        assert 0.0 <= record["churn_probability"] <= 1.0
        assert record["actual_churn"] in (0, 1)
        for feature in record["top_features"]:
            if feature["feature"] == "Contract":
                assert feature["value"] in {"Month-to-month", "One year", "Two year"}
            else:
                assert isinstance(feature["value"], float)


def test_build_customer_records_ranks_by_churn_probability_descending(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    registry = build_model_registry()
    model, _ = registry["xgb"]
    model.fit(x_train, y_train)

    explainer = explain.build_explainer("xgb", model, background=x_train)
    explanation = explain.compute_shap_values(explainer, x_test, sample_size=None)

    records = dashboard_export.build_customer_records(
        explanation, x_test, y_test, model, encoders={}, max_customers=10,
    )
    probs = [r["churn_probability"] for r in records]
    assert probs == sorted(probs, reverse=True)


def test_export_dashboard_writes_expected_json_shape(tmp_path):
    results = pd.DataFrame(
        {
            "accuracy": [0.75],
            "f1": [0.6],
            "roc_auc": [0.83],
            "emp": [10.1],
            "eprofits_avg": [1_600_000.0],
            "eprofits_tenure": [12_000_000.0],
        },
        index=pd.Index(["random_forest"], name="model"),
    )
    global_importance = pd.DataFrame({"feature": ["Contract"], "mean_abs_shap": [0.1]})

    out_path = tmp_path / "dashboard.json"
    dashboard_export.export_dashboard(
        str(out_path),
        dataset_summary={"name": "IBM", "rows": 100},
        results=results,
        best_model_name="random_forest",
        global_importance=global_importance,
        customer_records=[{"id": 1, "churn_probability": 0.9}],
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["best_model"] == "random_forest"
    assert payload["dataset"]["name"] == "IBM"
    assert payload["models"] == [
        {
            "name": "random_forest",
            "accuracy": 0.75,
            "f1": 0.6,
            "roc_auc": 0.83,
            "emp": 10.1,
            "eprofits_avg": 1_600_000.0,
            "eprofits_tenure": 12_000_000.0,
        }
    ]
    assert payload["shap_global"] == [{"feature": "Contract", "mean_abs_shap": 0.1}]
    assert payload["shap_customers"] == [{"id": 1, "churn_probability": 0.9}]
    assert "generated_at" in payload
