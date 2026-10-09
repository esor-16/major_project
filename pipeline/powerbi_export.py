"""Power BI integration for the e-Profits pipeline.

Writes an import-ready `<output_dir>/powerbi/` folder of *tidy* CSVs -
one row per model / one row per metric / one row per customer, no nested
JSON - because Power BI Desktop consumes flat files natively via
Get data -> Text/CSV (or the included Folder query). Alongside the CSVs it
writes:

  README.md       step-by-step instructions to build and publish the report
  power_query.m   paste-ready Power Query (M) with a shared SourceFolder
                  query, so re-pointing the report at a new run is a
                  one-path edit

Tables written (all validated against the reported metrics before anything
is written, so the dashboard can never disagree with model_evaluation.csv):

  models.csv           one row per model: the full evaluation table plus
                       is_best / is_hybrid flags (sorted by eprofits_tenure)
  metrics_long.csv     model x metric long table (metric_order + paper label)
                       for the metric-slicer bar chart
  shap_importance.csv  best model's churn drivers (rank, mean |SHAP|, share %)
  top_customers.csv    priority retention list (ranked by churn probability)
                       with decoded categories and each customer's primary
                       SHAP driver
  confusion_matrix.csv 4-row actual/predicted counts for the best model
  best_model.csv       single-row KPI table (metrics + confusion counts)
  dataset.csv          single-row run summary (dataset, split, best model,
                       survival method, generated_at)

Usage:
    # written automatically at the end of a run (skip with --no-powerbi):
    python -m pipeline.run --data IBM.csv

    # or regenerate from an existing run's artifacts (no retraining):
    python -m pipeline.powerbi_export --artifacts artifacts
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# (model_evaluation.csv column, paper display label) for the metric slicer.
METRIC_COLUMNS: list[tuple[str, str]] = [
    ("eprofits_tenure", "e-Profits (tenure)"),
    ("eprofits_avg", "e-Profits (avg retention)"),
    ("eprofits_top20_tenure", "e-Profits top-20% segment"),
    ("emp", "Expected Maximum Profit"),
    ("roc_auc", "ROC-AUC"),
    ("f1", "F1"),
    ("accuracy", "Accuracy"),
    ("top_decile_lift", "Top-decile lift"),
    ("lift_index", "Lift index"),
]

# Power Query (M) type per exported column, per table. Kept explicit so the
# generated .m file loads numbers as numbers instead of re-detecting text.
_M_TYPES: dict[str, list[tuple[str, str]]] = {
    "models": [
        ("model", "type text"), ("accuracy", "type number"), ("f1", "type number"),
        ("f1_threshold", "type number"), ("roc_auc", "type number"),
        ("top_decile_lift", "type number"), ("lift_index", "type number"),
        ("emp", "type number"), ("eprofits_avg", "type number"),
        ("eprofits_tenure", "type number"), ("eprofits_top20_avg", "type number"),
        ("eprofits_top20_tenure", "type number"),
        ("is_best", "type logical"), ("is_hybrid", "type logical"),
    ],
    "metrics_long": [
        ("model", "type text"), ("is_best", "type logical"), ("is_hybrid", "type logical"),
        ("metric_order", "Int64.Type"), ("metric", "type text"),
        ("metric_label", "type text"), ("value", "type number"),
    ],
    "shap_importance": [
        ("rank", "Int64.Type"), ("feature", "type text"),
        ("mean_abs_shap", "type number"), ("share_pct", "type number"),
    ],
    "top_customers": [
        ("rank", "Int64.Type"), ("customer_id", "Int64.Type"),
        ("churn_probability", "type number"), ("actual_churn", "Int64.Type"),
        ("tenure", "type number"), ("monthly_charges", "type number"),
        ("contract", "type text"), ("internet_service", "type text"),
        ("payment_method", "type text"), ("tech_support", "type text"),
        ("online_security", "type text"), ("primary_driver", "type text"),
        ("primary_driver_value", "type text"), ("primary_driver_direction", "type text"),
    ],
    "confusion_matrix": [
        ("actual_churn", "Int64.Type"), ("predicted_churn", "Int64.Type"),
        ("count", "Int64.Type"),
    ],
    "best_model": [
        ("model", "type text"), ("threshold", "type number"),
        ("accuracy", "type number"), ("f1", "type number"),
        ("precision", "type number"), ("recall", "type number"),
        ("roc_auc", "type number"), ("eprofits_tenure", "type number"),
        ("eprofits_avg", "type number"), ("top_decile_lift", "type number"),
        ("lift_index", "type number"),
        ("tp", "Int64.Type"), ("fp", "Int64.Type"), ("fn", "Int64.Type"), ("tn", "Int64.Type"),
    ],
    "dataset": [
        ("name", "type text"), ("rows", "Int64.Type"), ("columns", "Int64.Type"),
        ("churn_count", "Int64.Type"), ("churn_rate", "type number"),
        ("avg_retention_rate", "type number"), ("avg_tenure", "type number"),
        ("avg_monthly_charge", "type number"),
        ("best_model", "type text"), ("survival_method", "type text"),
        ("test_rows", "Int64.Type"), ("test_positives", "Int64.Type"),
        ("generated_at", "type text"),
    ],
}

# Allowed |reconstructed - reported| when validating a confusion matrix
# against a reported accuracy/F1. Reported values in model_evaluation.csv
# carry full float precision, so a consistent matrix reproduces them to
# machine precision; 1e-3 also tolerates 3-decimal-rounded inputs without
# letting a genuinely wrong matrix through.
_CONFUSION_TOL = 1e-3


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------
def confusion_from_predictions(y_true, y_prob, threshold: float) -> dict:
    """Counts for `predicted = churn_probability >= threshold` - the exact
    operating point `evaluate.py` reports accuracy/F1 at."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pred = (y_prob >= threshold).astype(int)
    return {
        "tp": int(np.sum((pred == 1) & (y_true == 1))),
        "fp": int(np.sum((pred == 1) & (y_true == 0))),
        "fn": int(np.sum((pred == 0) & (y_true == 1))),
        "tn": int(np.sum((pred == 0) & (y_true == 0))),
    }


def implied_confusion(accuracy: float, f1: float, n_test: int, positives: int) -> dict:
    """Solve the 2x2 matrix implied by a reported accuracy/F1 (the paper's
    'implied confusion matrix' analysis), for regenerating the Power BI
    export from artifacts that didn't store raw predictions.

    With A = accuracy*n, d = A - n + positives (note d = TP - FP) and
    F1 = 2TP / (2TP + FP + positives), substituting FP = TP - d gives
        TP = F1 * (positives - d) / (2 * (1 - F1)).

    Raises ValueError when no non-negative integer matrix reproduces the
    reported accuracy and F1 (within _CONFUSION_TOL) - a wrong test size,
    wrong class count or inconsistent inputs must fail loudly rather than
    export a dashboard that disagrees with model_evaluation.csv.
    """
    n_test, positives = int(n_test), int(positives)
    if not (0 < positives < n_test):
        raise ValueError(f"need 0 < positives < n_test, got positives={positives}, n_test={n_test}")
    if not (0.0 <= accuracy <= 1.0 and 0.0 <= f1 <= 1.0):
        raise ValueError(f"accuracy/f1 must be in [0,1], got {accuracy=}, {f1=}")

    a_total = accuracy * n_test
    d = a_total - n_test + positives  # TP - FP

    if f1 <= 0.0:
        tp = 0
    elif f1 >= 1.0:
        tp = positives
    else:
        tp = int(round(f1 * (positives - d) / (2.0 * (1.0 - f1))))
    fp = int(round(tp - d))
    fn = positives - tp
    tn = (n_test - positives) - fp

    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    if min(counts.values()) < 0 or tp > positives:
        raise ValueError(
            f"no non-negative confusion matrix matches accuracy={accuracy}, f1={f1} "
            f"for n_test={n_test}, positives={positives} (solved to {counts})"
        )

    recon_acc = (tp + tn) / n_test
    denom = 2 * tp + fp + fn
    recon_f1 = (2 * tp / denom) if denom else 0.0
    if abs(recon_acc - accuracy) > _CONFUSION_TOL or abs(recon_f1 - f1) > _CONFUSION_TOL:
        raise ValueError(
            f"implied confusion matrix {counts} reproduces accuracy={recon_acc:.6f}, "
            f"f1={recon_f1:.6f} but reported accuracy={accuracy:.6f}, f1={f1:.6f} "
            f"(tolerance {_CONFUSION_TOL}) - check n_test/positives"
        )
    return counts


def validate_confusion(confusion: dict, results_row: pd.Series, n_test: int) -> None:
    """A supplied matrix must reproduce the model's reported accuracy/F1 -
    this is what keeps the Power BI dashboard from ever disagreeing with
    model_evaluation.csv."""
    tp, fp, fn, tn = (int(confusion[k]) for k in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    if n != int(n_test):
        raise ValueError(f"confusion matrix sums to {n}, expected n_test={n_test}")
    acc = (tp + tn) / n
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0
    if abs(acc - float(results_row["accuracy"])) > _CONFUSION_TOL:
        raise ValueError(f"confusion-implied accuracy {acc:.6f} != reported {float(results_row['accuracy']):.6f}")
    if abs(f1 - float(results_row["f1"])) > _CONFUSION_TOL:
        raise ValueError(f"confusion-implied F1 {f1:.6f} != reported {float(results_row['f1']):.6f}")


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def _results_frame(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    if df.index.name != "model":
        df.index.name = "model"
    return df.reset_index()


def build_models_table(results: pd.DataFrame, best_model: str) -> pd.DataFrame:
    """The evaluation table as a flat, sorted Power BI table."""
    df = _results_frame(results)
    df["is_best"] = df["model"] == best_model
    df["is_hybrid"] = df["model"].str.startswith("hybrid")
    sort_col = "eprofits_tenure" if "eprofits_tenure" in df.columns else df.columns[-1]
    df = df.sort_values(sort_col, ascending=False, kind="mergesort").reset_index(drop=True)
    return df


def build_metrics_long(results: pd.DataFrame, best_model: str) -> pd.DataFrame:
    """model x metric long table so one metric slicer can drive a
    Value-by-Model bar chart across every reported metric."""
    wide = build_models_table(results, best_model)
    frames = []
    for order, (col, label) in enumerate(METRIC_COLUMNS, start=1):
        if col not in wide.columns:
            continue
        part = wide[["model", "is_best", "is_hybrid"]].copy()
        part["metric_order"] = order
        part["metric"] = col
        part["metric_label"] = label
        part["value"] = wide[col].astype(float)
        frames.append(part)
    if not frames:
        raise ValueError("results frame contains none of the expected metric columns")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["metric_order", "value"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def build_shap_table(global_importance: pd.DataFrame) -> pd.DataFrame:
    df = global_importance.copy()
    df["mean_abs_shap"] = df["mean_abs_shap"].astype(float)
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = df["mean_abs_shap"].sum()
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["share_pct"] = 100.0 * df["mean_abs_shap"] / total if total > 0 else 0.0
    return df[["rank", "feature", "mean_abs_shap", "share_pct"]]


def build_customer_table(customer_records: list[dict]) -> pd.DataFrame:
    """Flattens the dashboard's nested top_features into one row per
    customer, ranked by churn probability (records arrive pre-ranked), with
    the strongest SHAP driver pulled out as its own column so Power BI can
    group/filter on it directly."""
    rows = []
    for rank, rec in enumerate(customer_records, start=1):
        drivers = rec.get("top_features") or []
        primary = max(drivers, key=lambda t: abs(float(t.get("shap_value", 0.0))), default=None)
        rows.append({
            "rank": rank,
            "customer_id": rec.get("id"),
            "churn_probability": float(rec.get("churn_probability", float("nan"))),
            "actual_churn": rec.get("actual_churn"),
            "tenure": rec.get("tenure"),
            "monthly_charges": rec.get("monthly_charges"),
            "contract": rec.get("contract"),
            "internet_service": rec.get("internet_service"),
            "payment_method": rec.get("payment_method"),
            "tech_support": rec.get("tech_support"),
            "online_security": rec.get("online_security"),
            "primary_driver": primary["feature"] if primary else None,
            "primary_driver_value": primary["value"] if primary else None,
            "primary_driver_direction": primary["direction"] if primary else None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("churn_probability", ascending=False, kind="mergesort").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    return df


def build_confusion_table(confusion: dict) -> pd.DataFrame:
    tp, fp, fn, tn = (int(confusion[k]) for k in ("tp", "fp", "fn", "tn"))
    return pd.DataFrame([
        {"actual_churn": 1, "predicted_churn": 1, "count": tp},
        {"actual_churn": 1, "predicted_churn": 0, "count": fn},
        {"actual_churn": 0, "predicted_churn": 1, "count": fp},
        {"actual_churn": 0, "predicted_churn": 0, "count": tn},
    ])


def build_best_model_table(results: pd.DataFrame, best_model: str, confusion: dict) -> pd.DataFrame:
    row = results.loc[best_model]
    tp, fp, fn, tn = (int(confusion[k]) for k in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return pd.DataFrame([{
        "model": best_model,
        "threshold": float(row.get("f1_threshold", 0.5)),
        "accuracy": float(row["accuracy"]),
        "f1": float(row["f1"]),
        "precision": precision,
        "recall": recall,
        "roc_auc": float(row["roc_auc"]),
        "eprofits_tenure": float(row.get("eprofits_tenure", float("nan"))),
        "eprofits_avg": float(row.get("eprofits_avg", float("nan"))),
        "top_decile_lift": float(row.get("top_decile_lift", float("nan"))),
        "lift_index": float(row.get("lift_index", float("nan"))),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }])


def build_dataset_table(
    dataset_summary: dict,
    best_model: str,
    survival_method: str,
    test_split: dict | None = None,
    generated_at: str | None = None,
) -> pd.DataFrame:
    test_split = test_split or {}
    return pd.DataFrame([{
        "name": dataset_summary.get("name"),
        "rows": dataset_summary.get("rows"),
        "columns": dataset_summary.get("columns"),
        "churn_count": dataset_summary.get("churn_count"),
        "churn_rate": dataset_summary.get("churn_rate"),
        "avg_retention_rate": dataset_summary.get("avg_retention_rate"),
        "avg_tenure": dataset_summary.get("avg_tenure"),
        "avg_monthly_charge": dataset_summary.get("avg_monthly_charge"),
        "best_model": best_model,
        "survival_method": survival_method,
        "test_rows": test_split.get("rows"),
        "test_positives": test_split.get("positives"),
        "generated_at": generated_at or datetime.datetime.now().isoformat(timespec="seconds"),
    }])


# ---------------------------------------------------------------------------
# README + Power Query companions
# ---------------------------------------------------------------------------
def _readme_text() -> str:
    return """# Power BI data — e-Profits churn pipeline

Flat, import-ready tables for one pipeline run. Re-running the pipeline
overwrites these files in place, so Power BI's **Home → Refresh** picks up
new results with no re-import.

## Load

1. Open **Power BI Desktop** → **Home → Get data → Text/CSV** → pick a file
   here → **Transform Data** (power query opens with headers already
   promoted by the CSV loader) → **Close & Apply**. Repeat for each file,
   or use **Get data → Folder** pointed at this folder and use **Combine →
   Combine Files**.
2. Alternatively open `power_query.m`, paste each `// Query: <name>` block
   into a new blank query's **Advanced Editor**, and set the `SourceFolder`
   query's text to this folder's absolute path once.
3. No relationships are required — every table is standalone.

## Tables

| File | Grain | Use it for |
|---|---|---|
| `models.csv` | one row per model (sorted by e-Profits) | main results table; bar chart of e-Profits by model |
| `metrics_long.csv` | one row per model × metric | single-select metric slicer driving a Value-by-Model bar chart |
| `shap_importance.csv` | one row per feature | churn-driver bar chart (share %) |
| `top_customers.csv` | one row per customer | ranked retention call list with decoded categories + primary driver |
| `confusion_matrix.csv` | one row per actual/predicted cell | 2×2 matrix (Actual on rows, Predicted on columns, Count as values) |
| `best_model.csv` | one row | KPI cards: threshold, precision/recall, confusion counts |
| `dataset.csv` | one row | report header: dataset, split, survival method, generated_at |

## Suggested report page

- **Bar chart**: `eprofits_tenure` by `model` from `models.csv` (sort
  descending) — the paper's e-Profits comparison figure.
- **Slicer** on `metrics_long.metric_label` (single select) + **bar chart**
  `value` by `model` — switches the same visual between AUC / F1 /
  e-Profits / lift metrics.
- **Cards** from `best_model.csv` (accuracy, F1, ROC-AUC, threshold).
- **Donut/bar** of `share_pct` by `feature` from `shap_importance.csv`.
- **Matrix** from `confusion_matrix.csv` with a `predicted_churn` slicer
  shown as 0/1 to make the 2×2 readable.
- **Table visual** of `top_customers.csv` for the campaign list.

## Refresh & publish

- Local file sources refresh on **Home → Refresh** while the folder stays
  where it is; publishing to the **Power BI Service** needs a
  on-premises data gateway for this local path (or move the folder to
  SharePoint/OneDrive and use those connectors instead).
- Parent-folder artifacts (`bootstrap_ci.csv`, `wilcoxon_tests.csv`,
  `rank_correlation.csv`, `feature_fusion_summary.json`, ...) live one
  level up and can be imported the same way for a significance page.
"""


def _power_query_m() -> str:
    """One paste-ready block per query. SourceFolder is a plain text query
    used by every table query, so re-pointing the report = editing one
    string."""
    blocks = [
        "// ============================================================\n"
        "// e-Profits pipeline — Power BI (M) queries\n"
        "// Paste each `// Query: <name>` block into a NEW blank query's\n"
        "// Advanced Editor (Home > Get data > Blank query > Advanced Editor)\n"
        "// and rename the query to <name>. Set SourceFolder's path ONCE.\n"
        "// ============================================================\n\n"
        "// Query: SourceFolder\n"
        "let\n"
        '    Source = "PASTE_THE_POWERBI_FOLDER_PATH_HERE"\n'
        "in\n"
        "    Source\n"
    ]
    for name, types in _M_TYPES.items():
        type_lines = ",\n                ".join(f'("{c}", {t})' for c, t in types)
        blocks.append(
            f"// Query: {name}\n"
            "let\n"
            f'    Source = Csv.Document(File.Contents(SourceFolder & "\\{name}.csv"), '
            "[Delimiter=\",\", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),\n"
            "    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),\n"
            "    Typed = Table.TransformColumnTypes(Promoted, [\n"
            f"                {type_lines}\n"
            "            ])\n"
            "in\n"
            "    Typed\n"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_powerbi(
    out_dir: str | Path,
    results: pd.DataFrame,
    best_model: str,
    dataset_summary: dict | None = None,
    global_importance: pd.DataFrame | None = None,
    customer_records: list[dict] | None = None,
    confusion: dict | None = None,
    survival_method: str = "cox",
    test_split: dict | None = None,
) -> list[Path]:
    """Writes the Power BI folder; returns the list of files created.

    Everything is built and cross-validated BEFORE the first file is
    written, so a failed validation leaves no half-written report behind.
    """
    if confusion is None:
        raise ValueError(
            "confusion matrix is required - pass confusion_from_predictions(...) or "
            "use export_powerbi_from_artifacts(...) which derives it"
        )
    if best_model not in results.index:
        raise ValueError(f"best_model {best_model!r} not in results index {list(results.index)}")
    n_test = int(sum(int(confusion[k]) for k in ("tp", "fp", "fn", "tn")))
    validate_confusion(confusion, results.loc[best_model], n_test)

    tables: dict[str, pd.DataFrame] = {"models": build_models_table(results, best_model)}
    tables["metrics_long"] = build_metrics_long(results, best_model)
    tables["confusion_matrix"] = build_confusion_table(confusion)
    tables["best_model"] = build_best_model_table(results, best_model, confusion)
    if dataset_summary is not None:
        tables["dataset"] = build_dataset_table(
            dataset_summary, best_model, survival_method,
            test_split=test_split,
        )
    if global_importance is not None and len(global_importance):
        tables["shap_importance"] = build_shap_table(global_importance)
    if customer_records:
        customer_table = build_customer_table(customer_records)
        if not customer_table.empty:
            tables["top_customers"] = customer_table

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, table in tables.items():
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        written.append(path)

    readme = out / "README.md"
    readme.write_text(_readme_text(), encoding="utf-8")
    written.append(readme)
    m_file = out / "power_query.m"
    m_file.write_text(_power_query_m(), encoding="utf-8")
    written.append(m_file)
    return written


def export_powerbi_from_artifacts(
    artifacts_dir: str | Path,
    out_dir: str | Path | None = None,
    survival_method: str = "cox",
    n_test: int | None = None,
    positives: int | None = None,
) -> list[Path]:
    """Regenerates the Power BI folder from an existing run's artifacts
    (model_evaluation.csv + dashboard.json) with no retraining.

    Test-split size and churner count come from dashboard.json's
    `test_split` block when present; older dashboards fall back to the
    stratified 70:30 split arithmetic (ceil(0.3 * rows), rate * n) and the
    implied confusion matrix must then reproduce the reported accuracy/F1
    or the export fails loudly - pass --n-test/--positives to override.
    """
    art = Path(artifacts_dir)
    results = pd.read_csv(art / "model_evaluation.csv").set_index("model")

    payload = {}
    dash_path = art / "dashboard.json"
    if dash_path.exists():
        payload = json.loads(dash_path.read_text(encoding="utf-8"))

    dataset_summary = payload.get("dataset") or {}
    best_model = payload.get("best_model") or str(results["eprofits_tenure"].idxmax())
    if best_model not in results.index:
        raise ValueError(f"dashboard.json best_model {best_model!r} missing from model_evaluation.csv")

    stored_split = payload.get("test_split") or {}
    if n_test is None:
        n_test = int(stored_split["rows"]) if "rows" in stored_split else math.ceil(0.3 * int(dataset_summary["rows"]))
    if positives is None:
        positives = (
            int(stored_split["positives"]) if "positives" in stored_split
            else round(float(dataset_summary["churn_rate"]) * n_test)
        )

    row = results.loc[best_model]
    confusion = implied_confusion(float(row["accuracy"]), float(row["f1"]), n_test, positives)

    global_importance = pd.DataFrame(payload["shap_global"]) if payload.get("shap_global") else None
    customer_records = payload.get("shap_customers") or None

    return export_powerbi(
        out_dir or (art / "powerbi"),
        results=results,
        best_model=best_model,
        dataset_summary=dataset_summary or None,
        global_importance=global_importance,
        customer_records=customer_records,
        confusion=confusion,
        survival_method=survival_method,
        test_split={"rows": n_test, "positives": positives},
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default="artifacts", help="run directory containing model_evaluation.csv / dashboard.json")
    parser.add_argument("--out", default=None, help="output folder (default: <artifacts>/powerbi)")
    parser.add_argument("--survival", default="cox", choices=["cox", "km"], help="which survival run the artifacts came from (recorded in dataset.csv)")
    parser.add_argument("--n-test", type=int, default=None, help="override the test-split size (default: dashboard.json, else ceil(0.3 * rows))")
    parser.add_argument("--positives", type=int, default=None, help="override the number of test churners (default: dashboard.json, else churn_rate * n_test)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    files = export_powerbi_from_artifacts(
        args.artifacts, out_dir=args.out,
        survival_method=args.survival, n_test=args.n_test, positives=args.positives,
    )
    for f in files:
        print(f"wrote {f}")
