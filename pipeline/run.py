"""End-to-end Phase-2 churn pipeline:

  raw CSV -> preprocess -> class-balance (train split only) -> CoxPH/KM
  survival -> GA-tuned baseline training (5 models) -> feature-fusion
  hybrid (donor SHAP on training data -> fused feature ranking -> GA-tuned
  recipient per fusion rule) + prediction-level blend hybrid (two tuned
  members, weight chosen on out-of-fold training predictions) -> unified
  traditional + eProfits/EMP evaluation (incl. top-20% segment profit and
  lift metrics) -> statistical validation (bootstrap CIs, paired Wilcoxon,
  Spearman AUC-vs-profit re-ranking) -> SHAP explainability on the best
  model -> results + explanation artifacts written to <output-dir>/,
  including dashboard.json for the static web frontend (frontend/) to
  fetch and render, and a Power BI-ready powerbi/ folder (tidy CSVs +
  Power Query + build guide, `--no-powerbi` to skip).

Usage:
    python -m pipeline.run --data IBM.csv
    python -m pipeline.run --data IBM.csv --models xgb random_forest --balance adasyn
    python -m pipeline.run --data IBM.csv --population-size 20 --generations 10 --cv 5
    python -m pipeline.run --data IBM.csv --no-explain
    python -m pipeline.run --data IBM.csv --survival km --output-dir artifacts_km
    python -m pipeline.run --data IBM.csv --no-fusion --no-blend

Survival: CoxPH is the BASE mechanism (per-customer S(t | x_i) retention
conditioned on each customer's own covariates - see survival.py).
`--survival km` swaps in the paper's population-level Kaplan-Meier curve
via the same (model, survival_fn, avg_retention_rate) interface, so
running once with each and diffing model_evaluation.csv's eprofits
columns is the CoxPH-vs-KM comparison.

The project runs on the IBM Telco dataset (`--data IBM.csv`, schema
`--dataset-schema ibm` - the only schema currently defined; the schema
hook exists so a foreign dataset can be added later on its own native
columns rather than remapped onto IBM's schema). Point `--output-dir` at
a separate folder per experiment so results don't overwrite each other.

Every model - baselines, feature-fusion hybrids, and the blend hybrid
alike - is GA-tuned against a single composite scorer
(`eprofits.make_composite_scorer_tenure`) that blends eProfits with F1
(`--f1-weight`, default 0.5 - equal weight), not eProfits alone: a
pure-eProfits objective was found to leave every model free to trade
F1/accuracy for profit at a flat 0.5 threshold. Using the same composite
objective for everyone means any resulting difference between models
reflects the models themselves, not different tuning targets -
`model_evaluation.csv`'s reported eprofits/f1/accuracy columns are
unaffected by this, since evaluate.py computes those independently from
each model's actual test-set predictions.

The feature-fusion hybrid (on by default) asks two donor models
(`--fusion-donors`, default random_forest + lightgbm) which features they
actually rely on via SHAP computed on TRAINING data (never the test set -
that would be feature-selection leakage), fuses those rankings under three
aggregators (`--fusion-rules`, default all three: union/rank_fusion/
intersection), sweeps how many features to keep per rule using that same
composite scorer, then GA-tunes a recipient model (`--fusion-recipient`,
default xgb - deliberately neither donor) on each rule's selected subset
with the SAME GA budget as the baselines. Results land in
model_evaluation.csv alongside the baselines as hybrid_union /
hybrid_rank_fusion / hybrid_intersection, so the ablation (hybrid vs. plain
xgb on full features) is a straight row-vs-row read of one table.

The prediction-level blend hybrid (on by default, `hybrid_blend`) averages
two tuned members' churn probabilities (`--blend-members`, default xgb +
lightgbm) with a weight selected on out-of-fold TRAINING predictions only
(see blend.py) - the "highly accurate predictions" complement to feature
fusion's feature-pruning, since soft-voting complementary boosters
routinely beats either member on ROC-AUC/F1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from . import dashboard_export, data, evaluate, explain, feature_fusion, models, powerbi_export, significance
from .blend import DEFAULT_BLEND_MEMBERS, build_blend_hybrid
from .eprofits import make_composite_scorer_tenure
from .feature_fusion import SubsetPipeline
from .survival import fit_km_survival_model, fit_survival_model

RANDOM_STATE = 42

DATASET_SCHEMAS = {
    "ibm": {
        "loader": data.load_dataset,
        "preprocess": data.preprocess,
        "id_column": "customerID",
        "tenure_column": "tenure",
    },
}

SURVIVAL_METHODS = {
    "cox": fit_survival_model,
    "km": fit_km_survival_model,
}

# The baselines the hybrids are benchmarked against. Random Forest and
# LightGBM double as the default fusion donors, and xgb + lightgbm are the
# default blend members, so the default configuration trains exactly what
# both hybrids need with no extra runs. EBM completes the paper's six-model
# line-up (the fifth of the six here; gradient_boosting is available via
# --models but left out of the defaults to keep the run budget sane).
DEFAULT_MODEL_NAMES = ["xgb", "random_forest", "lightgbm", "catboost", "ebm"]


def _build_fusion_hybrids(
    fitted_models: dict,
    x_train: pd.DataFrame,
    x_train_bal: pd.DataFrame,
    y_train_bal: pd.Series,
    scorer,
    fusion_donors: list[str],
    fusion_recipient: str,
    fusion_rules: list[str],
    fusion_sizes: tuple[int, ...],
    cv: int,
    population_size: int,
    generations: int,
    fusion_sample_size: int,
    out_dir: Path,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Build one feature-fusion hybrid per rule, add each to `fitted_models`
    (mutated in place, also returned), and write the fusion diagnostic
    artifacts (rankings, donor agreement, per-rule feature-count sweep and
    selected features) to `out_dir`.

    Donor SHAP is computed on `x_train` (real, pre-SMOTE rows) - selecting
    features from test-set SHAP would be leakage, and synthetic SMOTE rows
    aren't genuine customers to base a feature-importance judgement on.
    The feature-count sweep and the recipient's GA tuning both then use
    `x_train_bal`/`y_train_bal`, matching the exact data the baselines were
    trained on, so the only thing that differs between a hybrid and the
    plain recipient baseline is the feature set - not the training
    distribution or the search budget.

    `scorer` (built by the caller via `eprofits.make_composite_scorer_tenure`)
    is the SAME composite eProfits+F1 scorer used for every baseline model -
    both the feature-count sweep and the recipient's own GA tuning below use
    it as-is, so "how much F1 matters" is decided in exactly one place
    (the scorer's own `f1_weight`), not separately at the sweep stage and
    again (differently) at tuning time.
    """
    missing_donors = [d for d in fusion_donors if d not in fitted_models]
    if missing_donors:
        raise ValueError(
            f"Fusion donor(s) {missing_donors} were not trained (--models was "
            f"restricted to {sorted(fitted_models)}); either include the donors "
            f"in --models or pass --no-fusion."
        )

    print(f"Building feature-fusion hybrid(s) from donors {fusion_donors} -> recipient {fusion_recipient!r}")

    importances = feature_fusion.donor_importances(
        fitted_models, x_train, donor_names=fusion_donors, sample_size=fusion_sample_size, random_state=random_state,
    )
    agreement = feature_fusion.rank_agreement(importances)
    print(f"  donor rank agreement (Spearman): {agreement:.4f}"
          + ("  [>0.9: donors largely agree, fusion may have little headroom]" if agreement > 0.9 else ""))

    fused = feature_fusion.fuse_rankings(importances)
    fused.to_csv(out_dir / "feature_fusion_rankings.csv", index=False)

    registry = models.build_model_registry(random_state)
    if fusion_recipient not in registry:
        raise ValueError(f"Unknown fusion recipient {fusion_recipient!r}; choose from {sorted(registry)}")
    recipient_model, recipient_param_grid = registry[fusion_recipient]

    all_columns = list(x_train_bal.columns)
    sweep_tables = []
    selection_summary = {"donor_rank_agreement_spearman": agreement, "donors": fusion_donors, "recipient": fusion_recipient, "rules": {}}

    for i, rule in enumerate(fusion_rules):
        print(f"  [{rule}] sweeping feature count...")
        best_m, sweep = feature_fusion.sweep_feature_count(
            recipient_model, fused, rule, x_train_bal, y_train_bal, scorer,
            sizes=fusion_sizes, cv=cv,
        )
        sweep_tables.append(sweep)
        columns = feature_fusion.select_features(fused, rule, best_m)
        selection_summary["rules"][rule] = {"n_features": best_m, "features": columns}
        print(f"  [{rule}] selected {best_m} features: {columns}")

        # Offset by +100 so hybrid seeds never collide with baseline model
        # seeds (train_all_models uses random_state + <position in names>);
        # not required for correctness, just keeps the two seed ranges
        # visibly distinct.
        gs = models.genetic_search(
            clone(recipient_model), recipient_param_grid,
            x_train_bal[columns], y_train_bal, cv,
            scoring={"fitness": scorer}, refit="fitness",
            population_size=population_size, generations=generations,
            name=f"hybrid_{rule}", random_state=random_state + 100 + i,
        )

        wrapped = SubsetPipeline(estimator=gs.best_estimator_, features=columns, all_columns=all_columns)
        fitted_models[f"hybrid_{rule}"] = wrapped

    pd.concat(sweep_tables, ignore_index=True).to_csv(out_dir / "feature_fusion_sweep.csv", index=False)
    (out_dir / "feature_fusion_summary.json").write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")

    return fitted_models


def run_pipeline(
    data_path: str,
    dataset_schema: str = "ibm",
    balance_method: str | None = "smote",
    cv: int = 3,
    population_size: int = 8,
    generations: int = 5,
    model_names: list[str] | None = None,
    output_dir: str = "artifacts",
    explain_best_model: bool = True,
    explain_sample_size: int = 200,
    tune_f1_threshold: bool = True,
    threshold_cv: int = 3,
    survival_method: str = "cox",
    build_fusion: bool = True,
    fusion_donors: list[str] | None = None,
    fusion_recipient: str = "xgb",
    fusion_rules: list[str] | None = None,
    fusion_sizes: tuple[int, ...] = feature_fusion.DEFAULT_FUSION_SIZES,
    fusion_sample_size: int = 500,
    build_blend: bool = True,
    blend_members: list[str] | None = None,
    significance_analysis: bool = True,
    bootstrap_resamples: int = significance.DEFAULT_N_BOOT,
    build_powerbi: bool = True,
    f1_weight: float = 0.5,
) -> "pd.DataFrame":
    if dataset_schema not in DATASET_SCHEMAS:
        raise ValueError(f"Unknown dataset_schema {dataset_schema!r}; choose from {sorted(DATASET_SCHEMAS)}")
    if survival_method not in SURVIVAL_METHODS:
        raise ValueError(f"Unknown survival_method {survival_method!r}; choose from {sorted(SURVIVAL_METHODS)}")

    schema = DATASET_SCHEMAS[dataset_schema]
    tenure_column = schema["tenure_column"]

    model_names = model_names or list(DEFAULT_MODEL_NAMES)
    fusion_donors = fusion_donors or list(feature_fusion.DONOR_MODELS)
    fusion_rules = fusion_rules or list(feature_fusion.FUSION_RULES)

    raw = schema["loader"](data_path)
    processed, encoders = schema["preprocess"](raw, id_column=schema["id_column"], return_encoders=True)
    x_train, x_test, y_train, y_test = data.split_data(processed, random_state=RANDOM_STATE)

    # Fit on the real (pre-resampling) training data - synthetic SMOTE/ADASYN
    # rows aren't genuine time-to-churn observations.
    fit_fn = SURVIVAL_METHODS[survival_method]
    survival_model, retention_fn, avg_retention_rate = fit_fn(x_train, y_train, tenure_column=tenure_column)

    x_train_bal, y_train_bal = data.balance_classes(x_train, y_train, method=balance_method, random_state=RANDOM_STATE)

    # This scorer decides what EVERY model (baselines and fusion hybrids
    # alike) is tuned to maximize - blending eProfits with F1 (weighted by
    # `f1_weight`) rather than eProfits alone. A pure-eProfits objective was
    # found to leave every model, hybrids and baselines both, free to
    # sacrifice F1/accuracy for profit at a flat 0.5 threshold; using the
    # same composite objective everywhere means any resulting difference
    # between models reflects the models themselves, not different tuning
    # targets. `model_evaluation.csv`'s reported eprofits_avg/eprofits_tenure/
    # f1/accuracy columns are unaffected - those are computed independently
    # in evaluate.py from each model's actual test-set predictions,
    # regardless of what objective it was tuned against.
    scorer = make_composite_scorer_tenure(
        reference_df=x_train_bal, retention_fn=retention_fn, tenure_column=tenure_column, f1_weight=f1_weight,
    )

    fitted_models = models.train_all_models(
        x_train_bal,
        y_train_bal,
        cv=cv,
        scoring_metric="fitness",
        scorer=scorer,
        model_names=model_names,
        random_state=RANDOM_STATE,
        population_size=population_size,
        generations=generations,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if build_fusion:
        fitted_models = _build_fusion_hybrids(
            fitted_models, x_train, x_train_bal, y_train_bal, scorer,
            fusion_donors=fusion_donors, fusion_recipient=fusion_recipient,
            fusion_rules=fusion_rules, fusion_sizes=fusion_sizes,
            cv=cv, population_size=population_size, generations=generations,
            fusion_sample_size=fusion_sample_size, out_dir=out_dir,
        )

    if build_blend:
        members = list(blend_members or DEFAULT_BLEND_MEMBERS)
        missing = [m for m in members if m not in fitted_models]
        if missing:
            print(f"Skipping blend hybrid: member(s) {missing} were not trained; "
                  f"include them in --models or pass --no-blend.")
        else:
            print(f"Building prediction-level blend hybrid from members {members}...")
            blend_model, blend_weight, blend_sweep = build_blend_hybrid(
                fitted_models, member_names=members,
                x_train=x_train_bal, y_train=y_train_bal,
                scorer=scorer, cv=cv,
            )
            fitted_models["hybrid_blend"] = blend_model
            blend_summary = {
                "members": members,
                "weight": blend_weight,
                "selection": "out-of-fold composite eProfits+F1 score on the training split (test never touched)",
                "sweep": blend_sweep.to_dict(orient="records"),
            }
            (out_dir / "blend_summary.json").write_text(json.dumps(blend_summary, indent=2), encoding="utf-8")
            print(f"  > hybrid_blend: weight={blend_weight:.2f} "
                  f"(members: {members[0]}={blend_weight:.0%}, {members[1]}={1 - blend_weight:.0%})")

    results = evaluate.evaluate_all(
        fitted_models,
        x_test,
        y_test,
        reference_df=processed,
        retention_fn=retention_fn,
        avg_retention_rate=avg_retention_rate,
        # Threshold search gets the REAL (pre-resampling) training split:
        # folds are balanced internally to match how the models were
        # trained, but validation rows keep the deployment class prior, so
        # the chosen threshold transfers to the test set for wrapped models
        # (fusion hybrids, blend) as well as plain baselines. See
        # evaluate.find_f1_optimal_threshold.
        x_train=x_train,
        y_train=y_train,
        tune_f1_threshold=tune_f1_threshold,
        threshold_cv=threshold_cv,
        balance_method=balance_method,
        tenure_column=tenure_column,
    )

    results.to_csv(out_dir / "model_evaluation.csv")

    if significance_analysis:
        print("Running significance analysis (bootstrap CIs, Wilcoxon, rank correlation)...")
        significance.run_significance_analysis(
            fitted_models, x_test, y_test,
            reference_df=processed, retention_fn=retention_fn,
            results=results, out_dir=out_dir,
            tenure_column=tenure_column, n_boot=bootstrap_resamples,
            seed=RANDOM_STATE,
        )
        print(f"Wrote bootstrap_ci.csv / wilcoxon_tests.csv / rank_correlation.csv to {out_dir}")

    if explain_best_model:
        best_name = explain.pick_best_model(results)
        print(f"Explaining best model ({best_name}) with SHAP...")

        # Pre-sample once (rather than letting compute_shap_values sample
        # internally) so we keep the real row index and can align it back
        # to y_test for the dashboard export's "actual_churn" field.
        explain_x = x_test if len(x_test) <= explain_sample_size else x_test.sample(explain_sample_size, random_state=RANDOM_STATE)
        y_explained = y_test.loc[explain_x.index]

        explainer = explain.build_explainer(best_name, fitted_models[best_name], background=x_train_bal)
        explanation = explain.compute_shap_values(explainer, explain_x, sample_size=None)

        global_importance = explain.global_feature_importance(explanation)
        global_importance.to_csv(out_dir / "shap_global_importance.csv", index=False)

        per_customer_rows = []
        for i in range(len(explanation)):
            top_features = explain.explain_customer(explanation, i, top_n=5)
            top_features.insert(0, "customer_row", i)
            per_customer_rows.append(top_features)
        per_customer = pd.concat(per_customer_rows, ignore_index=True)
        per_customer.to_csv(out_dir / "shap_customer_explanations.csv", index=False)

        print(f"Top churn drivers ({best_name}):")
        print(global_importance.head(5).to_string(index=False))

        if dataset_schema == "ibm":
            # dashboard_export's field names (tenure, MonthlyCharges, Contract,
            # InternetService, ...) are IBM-schema-specific - a future
            # non-IBM dataset's own analysis still gets the CSVs above, just
            # not the web dashboard JSON.
            #
            # A feature-fusion explanation over a subset of columns can leave
            # `explanation` missing some of the fields dashboard_export reads
            # by name (tenure, MonthlyCharges, Contract, ...) - build_customer_
            # records/_decode already read via `.get(...)`, so a field simply
            # comes back NaN/None rather than raising when the hybrid didn't
            # select it.
            dataset_summary = dashboard_export.build_dataset_summary(
                processed, dataset_name=Path(data_path).stem, avg_retention_rate=avg_retention_rate,
            )
            customer_records = dashboard_export.build_customer_records(
                explanation, explain_x, y_explained, fitted_models[best_name], encoders,
            )
            dashboard_path = out_dir / "dashboard.json"
            dashboard_export.export_dashboard(
                str(dashboard_path),
                dataset_summary=dataset_summary,
                results=results,
                best_model_name=best_name,
                global_importance=global_importance,
                customer_records=customer_records,
                test_split={"rows": int(len(y_test)), "positives": int(y_test.sum())},
            )
            print(f"Wrote frontend dashboard data to {dashboard_path}")

            if build_powerbi:
                # The confusion matrix uses the exact operating point the
                # reported accuracy/F1 use (results' f1_threshold), and
                # export_powerbi cross-validates it against those reported
                # numbers before writing a single file - the Power BI
                # dashboard can never disagree with model_evaluation.csv.
                best_proba = fitted_models[best_name].predict_proba(x_test)[:, 1]
                confusion = powerbi_export.confusion_from_predictions(
                    y_test.to_numpy(), best_proba, float(results.loc[best_name, "f1_threshold"]),
                )
                powerbi_files = powerbi_export.export_powerbi(
                    out_dir / "powerbi",
                    results=results,
                    best_model=best_name,
                    dataset_summary=dataset_summary,
                    global_importance=global_importance,
                    customer_records=customer_records,
                    confusion=confusion,
                    survival_method=survival_method,
                    test_split={"rows": int(len(y_test)), "positives": int(y_test.sum())},
                )
                print(f"Wrote {len(powerbi_files)} Power BI files to {out_dir / 'powerbi'}")

    return results


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="IBM.csv", help="path to the raw telecom churn CSV")
    parser.add_argument(
        "--dataset-schema", default="ibm", choices=sorted(DATASET_SCHEMAS),
        help="which raw schema --data is in (default: ibm)",
    )
    parser.add_argument("--balance", default="smote", choices=["smote", "adasyn", "smoteenn", "none"])
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=f"subset of baseline models to run, e.g. --models xgb random_forest "
             f"(default: {DEFAULT_MODEL_NAMES})",
    )
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument(
        "--no-explain", dest="explain", action="store_false",
        help="skip SHAP explainability on the best model (on by default)",
    )
    parser.add_argument(
        "--explain-sample-size", type=int, default=200,
        help="max number of test rows to run SHAP over (default: 200)",
    )
    parser.add_argument(
        "--no-tune-f1-threshold", dest="tune_f1_threshold", action="store_false",
        help="report Accuracy/F1 at a flat 0.5 cutoff instead of each model's F1-optimal threshold (on by default)",
    )
    parser.add_argument(
        "--threshold-cv", type=int, default=3,
        help="CV folds used to find each model's F1-optimal threshold (default: 3)",
    )
    parser.add_argument(
        "--survival", dest="survival_method", default="cox", choices=sorted(SURVIVAL_METHODS),
        help="retention-probability estimator: CoxPH (per-customer, covariate-conditioned - the BASE) "
             "or Kaplan-Meier (population-level baseline, for the CoxPH-vs-KM comparison)",
    )
    parser.add_argument(
        "--no-fusion", dest="build_fusion", action="store_false",
        help="skip the feature-fusion hybrid model(s) (on by default)",
    )
    parser.add_argument(
        "--fusion-donors", nargs=2, default=None, metavar=("MODEL_A", "MODEL_B"),
        help=f"the two donor models whose SHAP-decisive features get fused (default: {feature_fusion.DONOR_MODELS})",
    )
    parser.add_argument(
        "--fusion-recipient", default="xgb",
        help="model family trained on the fused feature set (default: xgb - deliberately neither donor, "
             "so hybrid-vs-xgb isolates the feature-selection effect)",
    )
    parser.add_argument(
        "--fusion-rules", nargs="*", default=None, choices=list(feature_fusion.FUSION_RULES),
        help=f"which fusion aggregator(s) to build a hybrid for (default: all three, {list(feature_fusion.FUSION_RULES)})",
    )
    parser.add_argument(
        "--fusion-sizes", nargs="*", type=int, default=list(feature_fusion.DEFAULT_FUSION_SIZES),
        help="candidate feature-set sizes to sweep per fusion rule (default: 5 8 10 12 15; the full "
             "feature count is always included as an extra candidate)",
    )
    parser.add_argument(
        "--fusion-sample-size", type=int, default=500,
        help="max training rows used for donor SHAP importance (default: 500)",
    )
    parser.add_argument(
        "--no-blend", dest="build_blend", action="store_false",
        help="skip the prediction-level blend hybrid (on by default)",
    )
    parser.add_argument(
        "--blend-members", nargs=2, default=None, metavar=("MODEL_A", "MODEL_B"),
        help=f"the two tuned models whose churn probabilities get blended (default: {DEFAULT_BLEND_MEMBERS})",
    )
    parser.add_argument(
        "--no-significance", dest="significance_analysis", action="store_false",
        help="skip the bootstrap/Wilcoxon/rank-correlation significance analysis (on by default)",
    )
    parser.add_argument(
        "--no-powerbi", dest="build_powerbi", action="store_false",
        help="skip the Power BI export (powerbi/ folder of tidy CSVs + Power Query + guide; on by default)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=significance.DEFAULT_N_BOOT,
        help=f"bootstrap resamples for the profit CIs (default: {significance.DEFAULT_N_BOOT}, as in the paper)",
    )
    parser.add_argument(
        "--f1-weight", type=float, default=0.5,
        help="how much EVERY model's GA tuning (baselines and fusion hybrids alike) leans on F1 vs. "
             "eProfits (0.0 = eProfits only, 1.0 = F1 only, default: 0.5 - weighed equally). Applies to "
             "the GA search's own fitness function, not just feature-count selection - a pure-eProfits "
             "objective was found to leave every model free to trade F1/accuracy for profit at a flat "
             "0.5 threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    balance = None if args.balance == "none" else args.balance
    results = run_pipeline(
        data_path=args.data,
        dataset_schema=args.dataset_schema,
        balance_method=balance,
        cv=args.cv,
        population_size=args.population_size,
        generations=args.generations,
        model_names=args.models,
        output_dir=args.output_dir,
        explain_best_model=args.explain,
        explain_sample_size=args.explain_sample_size,
        tune_f1_threshold=args.tune_f1_threshold,
        threshold_cv=args.threshold_cv,
        survival_method=args.survival_method,
        build_fusion=args.build_fusion,
        fusion_donors=args.fusion_donors,
        fusion_recipient=args.fusion_recipient,
        fusion_rules=args.fusion_rules,
        fusion_sizes=tuple(args.fusion_sizes),
        fusion_sample_size=args.fusion_sample_size,
        build_blend=args.build_blend,
        blend_members=args.blend_members,
        significance_analysis=args.significance_analysis,
        bootstrap_resamples=args.bootstrap,
        build_powerbi=args.build_powerbi,
        f1_weight=args.f1_weight,
    )
    print(results)
