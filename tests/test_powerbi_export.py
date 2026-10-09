"""Covers the Power BI integration (powerbi_export.py): tidy table
builders, the implied-confusion solver, folder export with its
cross-validation against reported metrics, and regeneration from
existing artifacts (with and without a stored test_split block).
"""
import json
import math

import pandas as pd
import pytest

from pipeline import powerbi_export as pbi


N_TEST = 2113
POSITIVES =561
# Exact confusion matrix for (accuracy=1537/2113, f1=894/1470):
# TP=447, FN=114, FP=462, TN=1090 - the paper's implied matrix for
# hybrid_union, also the ground truth this suite pins the solver against.
KNOWN_CONFUSION = {"tp": 447, "fp": 462, "fn": 114, "tn": 1090}
KNOWN_ACC = 1537 / 2113
KNOWN_F1 = 894 / 1470


def _results_frame() -> pd.DataFrame:
    """Three models; hybrid_union carries accuracy/F1 consistent with
    KNOWN_CONFUSION so exports can be validated against it."""
    return pd.DataFrame(
        [
            {
                "model": "hybrid_union", "accuracy": KNOWN_ACC, "f1": KNOWN_F1,
                "f1_threshold": 0.364, "roc_auc": 0.8297,
                "top_decile_lift": 2.7167, "lift_index": 0.7886, "emp": 9.99,
                "eprofits_avg": 691882.0, "eprofits_tenure": 707490.0,
                "eprofits_top20_avg": 425000.0, "eprofits_top20_tenure": 409725.0,
            },
            {
                "model": "xgb", "accuracy": 0.753, "f1": 0.628,
                "f1_threshold": 0.445, "roc_auc": 0.8331,
                "top_decile_lift": 2.694, "lift_index": 0.7939, "emp": 10.25,
                "eprofits_avg": 660000.0, "eprofits_tenure": 679217.0,
                "eprofits_top20_avg": 395000.0, "eprofits_top20_tenure": 381116.0,
            },
            {
                "model": "lightgbm", "accuracy": 0.745, "f1": 0.587,
                "f1_threshold": 0.328, "roc_auc": 0.8086,
                "top_decile_lift": 2.629, "lift_index": 0.7749, "emp": 9.77,
                "eprofits_avg": 505000.0, "eprofits_tenure": 520680.0,
                "eprofits_top20_avg": 365000.0, "eprofits_top20_tenure": 375796.0,
            },
        ]
    ).set_index("model")


def _dataset_summary() -> dict:
    return {
        "name": "IBM", "rows": 7043, "columns": 21, "churn_count": 1869,
        "churn_rate": 1869 / 7043, "avg_retention_rate": 0.9942,
        "avg_tenure": 32.37, "avg_monthly_charge": 64.76,
    }


def _importance() -> pd.DataFrame:
    return pd.DataFrame({
        "feature": ["Contract", "tenure", "customer_value"],
        "mean_abs_shap": [0.923, 0.530, 0.481],
    })


def _records() -> list[dict]:
    return [
        {
            "id": 101, "churn_probability": 0.91, "actual_churn": 1,
            "tenure": 2.0, "monthly_charges": 95.4, "contract": "Month-to-month",
            "internet_service": "Fiber optic", "payment_method": "Electronic check",
            "tech_support": "No", "online_security": "No",
            "top_features": [
                {"feature": "Contract", "value": "Month-to-month", "shap_value": 0.40, "direction": "toward churn"},
                {"feature": "tenure", "value": 2.0, "shap_value": -0.10, "direction": "away from churn"},
            ],
        },
        {
            "id": 102, "churn_probability": 0.22, "actual_churn": 0,
            "tenure": 48.0, "monthly_charges": 30.1, "contract": "Two year",
            "internet_service": "DSL", "payment_method": "Bank transfer",
            "tech_support": "Yes", "online_security": "Yes",
            "top_features": [
                {"feature": "tenure", "value": 48.0, "shap_value": -0.55, "direction": "away from churn"},
                {"feature": "Contract", "value": "Two year", "shap_value": 0.12, "direction": "toward churn"},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# implied_confusion solver
# ---------------------------------------------------------------------------
def test_implied_confusion_reproduces_known_matrix():
    counts = pbi.implied_confusion(KNOWN_ACC, KNOWN_F1, N_TEST, POSITIVES)
    assert counts == KNOWN_CONFUSION


def test_implied_confusion_rejects_inconsistent_inputs():
    # No non-negative integer matrix can have accuracy 0.9 and F1 0.1 on
    # 100 customers with 50 churners - must fail loudly, not export junk.
    with pytest.raises(ValueError):
        pbi.implied_confusion(0.9, 0.1, 100, 50)


def test_implied_confusion_rejects_bad_split_sizes():
    with pytest.raises(ValueError):
        pbi.implied_confusion(0.7, 0.6, 100, 0)
    with pytest.raises(ValueError):
        pbi.implied_confusion(0.7, 0.6, 100, 100)
    with pytest.raises(ValueError):
        pbi.implied_confusion(1.5, 0.6, 100, 50)


# ---------------------------------------------------------------------------
# confusion from raw predictions
# ---------------------------------------------------------------------------
def test_confusion_from_predictions_counts_and_operating_point():
    y_true = [1, 1, 0, 0, 1]
    y_prob = [0.9, 0.4, 0.6, 0.1, 0.5]
    counts = pbi.confusion_from_predictions(y_true, y_prob, threshold=0.5)
    assert counts == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    # Reconstructed metrics must match hand-computed values at that threshold
    n = 5
    acc = (counts["tp"] + counts["tn"]) / n
    f1 = 2 * counts["tp"] / (2 * counts["tp"] + counts["fp"] + counts["fn"])
    assert acc == pytest.approx(3 / 5)
    assert f1 == pytest.approx(4 / 6)


def test_confusion_threshold_is_inclusive():
    counts = pbi.confusion_from_predictions([1], [0.5], threshold=0.5)
    assert counts["tp"] == 1


# ---------------------------------------------------------------------------
# table builders
# ---------------------------------------------------------------------------
def test_models_table_sorted_by_profit_with_flags():
    table = pbi.build_models_table(_results_frame(), best_model="hybrid_union")
    assert list(table["model"]) == ["hybrid_union", "xgb", "lightgbm"]
    assert list(table["is_best"]) == [True, False, False]
    assert list(table["is_hybrid"]) == [True, False, False]


def test_metrics_long_covers_every_metric_for_every_model():
    table = pbi.build_metrics_long(_results_frame(), best_model="hybrid_union")
    assert len(table) == 3 * len(pbi.METRIC_COLUMNS)
    assert set(table["metric"]) == {col for col, _ in pbi.METRIC_COLUMNS}
    assert all(table["metric_label"].str.len() > 0)
    # one metric_order per metric, 1..N
    assert sorted(table["metric_order"].unique()) == list(range(1, len(pbi.METRIC_COLUMNS) + 1))
    # the best model's rows are flagged wherever they appear
    assert table.loc[table["model"] == "hybrid_union", "is_best"].all()
    assert not table.loc[table["model"] != "hybrid_union", "is_best"].any()


def test_shap_table_ranks_and_share_sums_to_100():
    table = pbi.build_shap_table(_importance())
    assert list(table["rank"]) == [1, 2, 3]
    assert list(table["feature"]) == ["Contract", "tenure", "customer_value"]
    assert table["share_pct"].sum() == pytest.approx(100.0)


def test_customer_table_ranks_by_risk_and_extracts_primary_driver():
    table = pbi.build_customer_table(_records())
    assert list(table["customer_id"]) == [101, 102]  # already risk-ranked
    assert list(table["rank"]) == [1, 2]
    # primary driver = largest |SHAP|, signed direction kept
    assert table.iloc[0]["primary_driver"] == "Contract"
    assert table.iloc[0]["primary_driver_direction"] == "toward churn"
    assert table.iloc[1]["primary_driver"] == "tenure"
    assert table.iloc[1]["primary_driver_direction"] == "away from churn"
    # decoded categories survive as readable strings for Power BI grouping
    assert table.iloc[0]["contract"] == "Month-to-month"


def test_metrics_long_missing_all_metrics_raises():
    with pytest.raises(ValueError):
        pbi.build_metrics_long(pd.DataFrame({"model": ["a"], "unrelated": [1.0]}).set_index("model"), "a")


# ---------------------------------------------------------------------------
# export_powerbi folder
# ---------------------------------------------------------------------------
def test_export_writes_complete_validated_folder(tmp_path):
    results = _results_frame()
    out = tmp_path / "powerbi"
    written = pbi.export_powerbi(
        out,
        results=results,
        best_model="hybrid_union",
        dataset_summary=_dataset_summary(),
        global_importance=_importance(),
        customer_records=_records(),
        confusion=dict(KNOWN_CONFUSION),
        survival_method="cox",
        test_split={"rows": N_TEST, "positives": POSITIVES},
    )
    names = {p.name for p in written}
    assert names == {
        "models.csv", "metrics_long.csv", "shap_importance.csv", "top_customers.csv",
        "confusion_matrix.csv", "best_model.csv", "dataset.csv",
        "README.md", "power_query.m",
    }
    assert all(p.exists() for p in written)

    models = pd.read_csv(out / "models.csv")
    assert list(models["model"]) == ["hybrid_union", "xgb", "lightgbm"]
    assert set(models.columns) == {col for col, _ in pbi._M_TYPES["models"]}

    confusion = pd.read_csv(out / "confusion_matrix.csv")
    assert int(confusion["count"].sum()) == N_TEST
    assert set(confusion.columns) == {col for col, _ in pbi._M_TYPES["confusion_matrix"]}

    best = pd.read_csv(out / "best_model.csv")
    assert int(best.iloc[0]["tp"]) == 447
    assert float(best.iloc[0]["accuracy"]) == pytest.approx(KNOWN_ACC)
    assert float(best.iloc[0]["f1"]) == pytest.approx(KNOWN_F1)

    dataset = pd.read_csv(out / "dataset.csv")
    assert int(dataset.iloc[0]["test_rows"]) == N_TEST
    assert int(dataset.iloc[0]["test_positives"]) == POSITIVES
    assert dataset.iloc[0]["survival_method"] == "cox"

    metrics = pd.read_csv(out / "metrics_long.csv")
    assert len(metrics) == 3 * len(pbi.METRIC_COLUMNS)

    shap = pd.read_csv(out / "shap_importance.csv")
    assert shap["share_pct"].sum() == pytest.approx(100.0)

    customers = pd.read_csv(out / "top_customers.csv")
    assert list(customers["customer_id"]) == [101, 102]

    # every exported CSV's header must match the Power Query type map, or
    # the generated .m file would fail to load it
    for name in pbi._M_TYPES:
        header = list(pd.read_csv(out / f"{name}.csv").columns)
        assert set(header) == {col for col, _ in pbi._M_TYPES[name]}, name

    # no literal nan/None strings leaking into the Power BI tables
    for csv_path in out.glob("*.csv"):
        text = csv_path.read_text(encoding="utf-8").lower()
        assert ",nan" not in text and "none" not in text, csv_path.name

    m_text = (out / "power_query.m").read_text(encoding="utf-8")
    assert "SourceFolder" in m_text and "Table.PromoteHeaders" in m_text
    assert "models.csv" in m_text
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "Get data" in readme and "metrics_long.csv" in readme


def test_export_rejects_confusion_that_disagrees_with_reported_metrics(tmp_path):
    # Wrong TP: reproduces accuracy 0.58, not the reported 0.7274
    bad = {"tp": 300, "fp": 462, "fn": 261, "tn": 1090}
    with pytest.raises(ValueError):
        pbi.export_powerbi(tmp_path / "powerbi", results=_results_frame(),
                           best_model="hybrid_union", confusion=bad)
    assert not (tmp_path / "powerbi").exists()  # nothing half-written


def test_export_requires_a_confusion_matrix(tmp_path):
    with pytest.raises(ValueError):
        pbi.export_powerbi(tmp_path / "powerbi", results=_results_frame(),
                           best_model="hybrid_union", confusion=None)


def test_export_rejects_unknown_best_model(tmp_path):
    with pytest.raises(ValueError):
        pbi.export_powerbi(tmp_path / "powerbi", results=_results_frame(),
                           best_model="nope", confusion=dict(KNOWN_CONFUSION))


# ---------------------------------------------------------------------------
# regeneration from artifacts
# ---------------------------------------------------------------------------
def _write_artifacts(tmp_path, with_test_split: bool = True, rows: int = 7043,
                     churn_rate: float = 1869 / 7043):
    art = tmp_path / "artifacts"
    art.mkdir()
    _results_frame().to_csv(art / "model_evaluation.csv")
    payload = {
        "generated_at": "2026-01-01T00:00:00",
        "dataset": _dataset_summary(),
        "best_model": "hybrid_union",
        "models": [],
        "shap_global": _importance().to_dict(orient="records"),
        "shap_customers": _records(),
    }
    payload["dataset"]["rows"] = rows
    payload["dataset"]["churn_rate"] = churn_rate
    if with_test_split:
        payload["test_split"] = {"rows": N_TEST, "positives": POSITIVES}
    (art / "dashboard.json").write_text(json.dumps(payload), encoding="utf-8")
    return art


def test_export_from_artifacts_uses_stored_test_split(tmp_path):
    art = _write_artifacts(tmp_path, with_test_split=True)
    written = pbi.export_powerbi_from_artifacts(art, survival_method="cox")
    out = art / "powerbi"
    assert all(p.exists() for p in written)
    confusion = pd.read_csv(out / "confusion_matrix.csv")
    assert int(confusion["count"].sum()) == N_TEST
    wide = confusion.set_index(["actual_churn", "predicted_churn"])["count"]
    assert int(wide.loc[(1, 1)]) == 447  # TP
    assert int(wide.loc[(0, 1)]) == 462  # FP
    dataset = pd.read_csv(out / "dataset.csv")
    assert int(dataset.iloc[0]["test_rows"]) == N_TEST


def test_export_from_artifacts_derives_split_when_dashboard_is_old(tmp_path):
    # Old dashboard.json without a test_split block: split size falls back
    # to ceil(0.3 * rows) and positives to churn_rate * n. Use a synthetic
    # run where those derived values (60 / 30) are consistent with the
    # reported accuracy/F1 (matrix TP=18, FP=12, FN=12, TN=18).
    art = _write_artifacts(tmp_path, with_test_split=False, rows=200, churn_rate=0.5)
    results = pd.read_csv(art / "model_evaluation.csv").set_index("model")
    results.loc["hybrid_union", "accuracy"] = 36 / 60
    results.loc["hybrid_union", "f1"] = 0.6
    results.to_csv(art / "model_evaluation.csv")

    pbi.export_powerbi_from_artifacts(art)
    dataset = pd.read_csv(art / "powerbi" / "dataset.csv")
    assert int(dataset.iloc[0]["test_rows"]) == math.ceil(0.3 * 200) == 60
    assert int(dataset.iloc[0]["test_positives"]) == 30
    confusion = pd.read_csv(art / "powerbi" / "confusion_matrix.csv")
    assert int(confusion["count"].sum()) == 60


def test_export_from_artifacts_fails_loudly_when_derived_split_is_wrong(tmp_path):
    # Dashboard implies positives=12, but accuracy/F1 only fit positives=30
    # - the implied-matrix validation must refuse rather than export a
    # confusion matrix that contradicts model_evaluation.csv.
    art = _write_artifacts(tmp_path, with_test_split=False, rows=200, churn_rate=0.2)
    results = pd.read_csv(art / "model_evaluation.csv").set_index("model")
    results.loc["hybrid_union", "accuracy"] = 36 / 60
    results.loc["hybrid_union", "f1"] = 0.6
    results.to_csv(art / "model_evaluation.csv")

    with pytest.raises(ValueError):
        pbi.export_powerbi_from_artifacts(art)


def test_export_from_artifacts_respects_explicit_overrides(tmp_path):
    art = _write_artifacts(tmp_path, with_test_split=True)
    pbi.export_powerbi_from_artifacts(art, n_test=N_TEST, positives=POSITIVES,
                                      survival_method="km")
    dataset = pd.read_csv(art / "powerbi" / "dataset.csv")
    assert dataset.iloc[0]["survival_method"] == "km"
