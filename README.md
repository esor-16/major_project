# EPROFITS — telecom churn analysis

## Install

```bash
pip install -r requirements.txt
```

## Layout

- `ibm_model_another_genetic.ipynb` — original Phase-1 prototype notebook (CoxPH survival analysis + eProfits metric, XGBoost/EBM only). Kept for reference.
- `pipeline/` — Phase-2 modular pipeline that supersedes the notebook:
  - `data.py` — ingestion, preprocessing, SMOTE/ADASYN class-imbalance handling
  - `survival.py` — **CoxPH is the base survival mechanism**, now genuinely per-customer: `S(t | x_i) = S0(t)^exp(x_i·β)` evaluated with each customer's own covariates, threaded through `conditional_retention` into every scorer, evaluation and significance calculation. The paper's population-level Kaplan-Meier curve is kept as `fit_km_survival_model` (`--survival km`) for the comparison — see [CoxPH vs. Kaplan-Meier](#coxph-vs-kaplan-meier-nfr2) below
  - `eprofits.py` — vectorized eProfits (avg/tenure) and EMP scoring, incl. GA/GridSearch-compatible scorer factories — see [Multi-objective GA fitness](#multi-objective-ga-fitness-eprofits--f1) for `make_composite_scorer_tenure`, the one every model is actually tuned against
  - `models.py` — registry of six classifiers (XGBoost, Gradient Boosting, Random Forest, LightGBM, CatBoost, EBM) with GA-search hyperparameter spaces; also the actual point of reproducibility guarantee for GA search (see [Reproducibility](#reproducibility-nfr5) below)
  - `feature_fusion.py` — the feature-fusion hybrid model — see [Feature-fusion hybrid](#feature-fusion-hybrid) below
  - `blend.py` — the prediction-level blend hybrid (`hybrid_blend`): a weighted average of two tuned models' churn probabilities, weight chosen on out-of-fold **training** predictions — see [Blend hybrid](#blend-hybrid) below
  - `significance.py` — the paper's statistical validation: bootstrap 95% CIs on test-set e-Profits, paired Wilcoxon signed-rank tests between models, Spearman AUC-vs-profit rank correlation
  - `evaluate.py` — unified traditional (Accuracy/F1/ROC-AUC) + profitability (EMP/eProfits incl. top-20% segment) + risk-prioritisation (top-decile lift, lift index) evaluation per model
  - `explain.py` — SHAP explainability for the best-performing model: global feature importance + per-customer, signed contributions (TreeExplainer for tree-based models including feature-fusion hybrids, a model-agnostic Explainer otherwise)
  - `dashboard_export.py` — packages the evaluation table + SHAP results into `artifacts/dashboard.json` for the web frontend
  - `run.py` — end-to-end CLI orchestrator
- `tests/` — pytest suite (`pytest -m "not slow"` for the fast checks, drop the `-m` filter to include the end-to-end GA smoke tests)
- `frontend/` — web dashboard, wired to real pipeline results via `artifacts/dashboard.json` (see [frontend/README.md](frontend/README.md))

## Run the pipeline

```bash
python -m pipeline.run --data IBM.csv
```

By default this trains **5 baseline models** (XGBoost, Random Forest, LightGBM, CatBoost, EBM) plus **3 feature-fusion hybrid variants** (one per fusion rule) plus **1 prediction-level blend hybrid** — 9 rows in the results table — and runs the significance analysis on all of them.

Useful flags:

```bash
python -m pipeline.run --data IBM.csv --models xgb random_forest --balance adasyn
python -m pipeline.run --data IBM.csv --population-size 20 --generations 10 --cv 5
python -m pipeline.run --data IBM.csv --no-fusion --no-blend     # baselines only, skip both hybrids
python -m pipeline.run --data IBM.csv --survival km               # population-level Kaplan-Meier baseline instead of CoxPH
python -m pipeline.run --data IBM.csv --blend-members xgb catboost
python -m pipeline.run --data IBM.csv --no-significance --bootstrap 500
```

Writes to `artifacts/`:
- `model_evaluation.csv` — one row per model (accuracy, f1, f1_threshold, roc_auc, top_decile_lift, lift_index, emp, eprofits_avg, eprofits_tenure, eprofits_top20_avg, eprofits_top20_tenure), sorted by eprofits_tenure
- `blend_summary.json` — the blend's members, chosen weight, and the full out-of-fold weight sweep
- `bootstrap_ci.csv` — per-model bootstrap mean and 95% CI for test-set e-Profits (500 resamples, as in the paper)
- `wilcoxon_tests.csv` — paired Wilcoxon signed-rank tests of per-customer profit between every pair of models
- `rank_correlation.csv` — Spearman rank correlation of ROC-AUC/F1 rankings vs. e-Profits rankings (the paper's re-ranking evidence)
- `feature_fusion_rankings.csv` — each donor's normalized SHAP importance per feature, plus the union/rank_fusion/intersection fused score
- `feature_fusion_summary.json` — donor rank agreement (Spearman), and per-rule selected feature count + feature list
- `feature_fusion_sweep.csv` — cross-validated composite (eProfits+F1) score at every candidate feature-set size, per rule (the evidence behind each rule's chosen feature count)
- `shap_global_importance.csv` — the best model's churn drivers, ranked by mean |SHAP value|
- `shap_customer_explanations.csv` — top 5 signed feature contributions per explained customer
- `dashboard.json` — the above, packaged for `frontend/` to fetch and render (see [frontend/README.md](frontend/README.md) to view it)

SHAP runs automatically on the top-ranked model after evaluation (`--no-explain` to skip it, `--explain-sample-size N` to change how many test-set rows get explained — default 200).

Note on runtime: the default `--population-size 8 --generations 5 --cv 3` is a deliberately reduced GA budget. The current default runs 9 GA searches (5 baselines + 3 fusion hybrids, plus the feature-count sweep) plus the blend's cheap weight sweep and the significance analysis — expect several minutes on the full IBM dataset. For reported/final numbers, raise the budget (`--population-size 20 --generations 10 --cv 5`) once you've confirmed what your hardware can take.

## The e-Profits economic inputs (paper Eq. 3–5)

- **CLV input `customer_value` = monthly revenue** (`MonthlyCharges` on the IBM dataset). The paper's Eq. 3 uses monthly revenue `R_i`: `CLV = R_i · M / (1 − min(r_i, 0.995))`. An earlier version used lifetime-to-date revenue (`MonthlyCharges × (tenure+1)`), which double-counted tenure on top of the perpetuity denominator and inflated every CLV — and therefore every e-Profits figure — by roughly `(tenure+1)×`. Fixed; `tests/test_data.py` now pins the invariant.
- **Retention `r_i`** comes from CoxPH per-customer conditional retention (TRR), or its population mean (ARR); both are reported (`eprofits_tenure` / `eprofits_avg`). KM-fitted retention is train-split-only, as the paper requires.
- **Costs** default to the paper's: margin `M=0.3`, offer cost `CPO=0.1 · CLV`, contact cost `max(0, 0.3 · C_offer)` — all overridable in `eprofits.py`.
- **Decision threshold** for e-Profits is the paper's fixed 0.5. Accuracy/F1 are additionally reported at each model's F1-optimal threshold, found by out-of-fold train-CV only (`--no-tune-f1-threshold` to disable): each fold's *training* portion is balanced the same way the models were trained, while validation rows keep the real ~27% class prior — so the threshold is calibrated to the deployment/test distribution. Selecting it on fully balanced predictions (the old behaviour) mis-calibrated thresholds, worst for the hybrids/blend, and showed up as a pure threshold artefact in the accuracy column. Among (numerically) equal-F1 operating points the highest threshold wins — same F1, fewer wasted retention offers.

## Multi-objective GA fitness (eProfits + F1)

Every model — the 5 baselines, the feature-fusion hybrids, and the blend hybrid's members and weight alike — is GA-tuned (or weight-selected) against `eprofits.make_composite_scorer_tenure`, not a pure-eProfits scorer. `--f1-weight` (0–1, default `0.5`) controls the blend: `0.0` is pure eProfits, `1.0` is pure F1, `0.5` weighs them equally.

**Why:** an earlier version tuned every model purely against `eprofits_tenure` at a flat 0.5 threshold. That left every model — baselines included — free to trade F1/accuracy for profit, and on IBM.csv the feature-fusion hybrids did so far more than the baselines: every hybrid variant landed below every baseline (including its own recipient, `xgb`) on both F1 and accuracy. The first fix attempt only changed how many features (`m`) get selected to also weigh F1 — but the recipient's actual hyperparameters were still tuned on pure eProfits afterward, so the gap barely moved. The real fix had to be in the GA fitness function itself.

**How the blend works:** raw eProfits (order 1e6–1e7) and F1 (0–1) aren't on comparable scales, so a plain weighted sum would make F1 invisible. `make_composite_scorer_tenure` instead normalizes eProfits against the *perfect-classifier ceiling* for that batch of customers — the eProfits achievable by flagging precisely the true churners and no one else, which is provably the profit-maximizing prediction pattern for this formula (every correctly-flagged churner adds a positive amount, every false positive only subtracts a small offer/contact cost). `eprofits / ceiling` lands in roughly the same [~0, 1] range as F1 without needing a fixed, dataset-specific scale constant, so `(1 - f1_weight) * eprofits_norm + f1_weight * f1` is a meaningful blend regardless of dataset.

**What this does and doesn't change:** the feature-fusion hybrid's sweep (choosing `m`) and its recipient's GA tuning (choosing hyperparameters) both now use this same scorer, so there's exactly one place deciding "how much F1 matters" — not two different, inconsistent weightings. `model_evaluation.csv`'s reported `eprofits_avg`/`eprofits_tenure`/`f1`/`accuracy`/`roc_auc` columns are unaffected by any of this: `evaluate.py` computes those independently from each model's actual test-set predictions, regardless of what it was tuned against.

## Reproducibility (NFR5)

`sklearn-genetic-opt`'s `GASearchCV` has no `random_state`/seed parameter of its own — inspecting its source shows it draws population initialization, mutation, and crossover from Python's global `random` module (plain `random.random()`/`random.choice()` calls) and numpy's global RNG. Left unseeded, every GA search in this pipeline is non-deterministic across process runs even with every other `random_state` fixed, which breaks "identical results across repeated runs with fixed seeds".

The fix is **not** a single seed at the top of `run_pipeline` — that was tried first and found insufficient: `sklearn.model_selection.train_test_split(..., stratify=y, random_state=42)` (used in `data.split_data`) perturbs the global RNG as a side effect even though it's given its own explicit seed, so anything downstream relying on one seed propagating cleanly through unrelated library internals is fragile. Instead, `models.genetic_search` takes a `random_state` parameter and reseeds Python's global `random` and numpy's global RNG *immediately before* each GA search runs — `train_all_models` derives a distinct, deterministic seed per model (`random_state + <position in the model list>`), and `run.py`'s fusion-hybrid loop does the same per rule (offset by `+100` so the two ranges don't overlap). Verified empirically: two full `run_pipeline` calls with the same inputs now produce bit-identical `best_estimator_` hyperparameters, fitness scores, and final test-set metrics.

## Feature-fusion hybrid

The hybrid model is **feature-level fusion**, not prediction-level stacking: two donor models report which features they actually rely on (via SHAP), those two rankings are merged into one, and a third model is trained on the merged feature set.

1. **Donors** (`--fusion-donors`, default `random_forest lightgbm`) — cross-family (bagging vs. leaf-wise boosting), so their importance rankings are derived by genuinely different mechanisms rather than largely agreeing by construction.
2. **Donor SHAP is computed on TRAINING data**, never the test set. Selecting features from test-set SHAP and then scoring on that same test set would be leakage; the reported eProfits/accuracy numbers would be optimistically biased. It's also computed on the pre-SMOTE `x_train` — synthetic interpolated rows aren't real customers and shouldn't drive which features are judged "decisive".
3. **Fusion rule** (`--fusion-rules`, default all three) — each donor's importances are normalized to [0, 1] (divide by that donor's own max, so ratios are preserved), then combined:
   - `union` = `max(donor_a, donor_b)` — a feature ranks high if **either** donor rates it high
   - `rank_fusion` = `mean(donor_a, donor_b)` — balanced consensus
   - `intersection` = `min(donor_a, donor_b)` — ranks high only if **both** donors rate it high
4. **Feature-count sweep** (`--fusion-sizes`, default `5 8 10 12 15` plus the full feature count) — for each rule, the top-`m` features are tried at several sizes and scored on the **training** split (the test set is never touched during selection), using an *untuned* recipient — the point is comparing feature sets cheaply, not re-running a full GA search at every candidate size. `m` is chosen by whichever size maximizes the same composite eProfits+F1 scorer described in [Multi-objective GA fitness](#multi-objective-ga-fitness-eprofits--f1) — the sweep has no separate opinion of its own about how much F1 should matter; it just delegates to whatever scorer `run.py` hands it, which is the identical one used for the recipient's own GA tuning right after.
5. **Recipient** (`--fusion-recipient`, default `xgb`) — deliberately **neither donor**. This matters for two reasons: it proves the feature knowledge genuinely transfers across model families rather than a model just re-fitting on features it picked itself, and since `xgb` already exists as a full-feature baseline, `hybrid_<rule> vs. xgb` becomes a controlled ablation — same algorithm, same GA budget, only the feature set differs.
6. The recipient is then **GA-tuned on the selected subset with the same population/generations/cv budget as the baselines** — a hybrid must not win by getting a bigger search budget than the models it's compared against.
7. The tuned model is wrapped in a `SubsetPipeline` so it drops into the rest of the pipeline (`evaluate.evaluate_all`, SHAP, the F1-threshold search, the dashboard export) unchanged: it accepts the full-width feature matrix like every other model and subsets to its own columns internally.

**Reading the results honestly:** `feature_fusion_summary.json`'s `donor_rank_agreement_spearman` is the diagnostic that tells you whether fusion had any headroom on this run. A value close to 1.0 means the two donors picked nearly the same features, in which case union/rank_fusion/intersection will converge to similar sets and the hybrid's edge (if any) is coming from feature *pruning*, not from fusing genuinely different signals. `feature_fusion_sweep.csv`'s `cv_score` column (the composite scorer's cross-validated value) tells you whether pruning even helped on this dataset: a flat curve across sizes means the dropped features were never hurting.

## Blend hybrid

The second hybrid, and the "highly accurate predictions" half of the project aim: a **prediction-level** probability blend (`hybrid_blend` in the results) that complements feature fusion's *feature-level* one.

1. **Members** (`--blend-members`, default `xgb lightgbm`) — two tuned, cross-family models whose churn probabilities are averaged: `P = w·P_a + (1−w)·P_b`.
2. **Weight selection uses training data only.** Each member's *tuned* configuration runs through `cross_val_predict` on the training split; candidate weights (`0.00 … 1.00` in 0.05 steps) are scored with the same composite eProfits+F1 scorer everyone else tunes against, and the argmax wins. The held-out test set never participates. The full sweep is exported to `blend_summary.json` so the choice is auditable — same standard as the feature-count sweep.
3. The members are then fitted on the full training frame (the exact data the baselines saw), so `hybrid_blend vs. its members` isolates the blending itself — no extra tuning budget, no different training distribution.
4. `ProbabilityBlend` implements the scikit-learn estimator API, so it drops into `evaluate_all`, the F1-threshold search, SHAP and the dashboard exactly like every other model.

Read it honestly: the blend can only beat its members where they make *uncorrelated* errors (check the members' ROC-AUCs in `model_evaluation.csv`; if they're near-identical the weight will sit near 0.5 and the gain will be marginal). The comparison that matters is the `hybrid_blend` row vs. its `xgb` and `lightgbm` rows in the same table.

## CoxPH vs. Kaplan-Meier (NFR2)

The base eProfits paper fits a single **Kaplan-Meier** population curve — everyone at the same tenure gets the same retention. This project's survival mechanism is **CoxPH**, and (as of the re-route) it now does what the old README claimed but the code did not do:

- `survival_fn(t, x)` evaluates **`S(t | x_i) = S0(t)^exp(x_i·β)`** per customer — the Breslow baseline raised to each customer's own hazard ratio — instead of the covariate-zero `baseline_survival_` lookup.
- `conditional_retention(..., x=...)` threads those covariates from every consumer: the GA scorers in `eprofits.py`, `evaluate.py`'s full-population and top-20% figures, and `significance.py`'s per-customer profit matrix. Two customers at the same tenure now genuinely get different retention (pinned by `test_coxph_retention_conditions_on_covariates`).
- `fit_km_survival_model` accepts and ignores `x`, keeping the identical 3-tuple interface, so the comparison is a clean one-flag diff of otherwise-identical runs:

```bash
python -m pipeline.run --data IBM.csv --survival cox --output-dir artifacts_cox
python -m pipeline.run --data IBM.csv --survival km  --output-dir artifacts_km
```

The two paths differ in exactly one thing — whether retention is individualised — so any e-Profits gap between the two output folders is attributable to that. Note the paper's own headline numbers come from the KM/TRR formulation; CoxPH is this project's extension of it, with the paper's ARR-style population average still reported alongside (`eprofits_avg`).

**Observed outcome (pop 10 / gen 6 / cv 3, IBM, current code):** KM reports *higher* full-population `eprofits_tenure` than CoxPH for every model (e.g. hybrid_union 909k vs 707k; totals 7.81M vs 5.84M). The direction is mechanical, not a quality verdict: CLV = R·M/(1−r), so the population KM curve's uniformly high one-period retention inflates every CLV, while CoxPH discounts retention per customer — a high-risk customer's CLV (and the profit booked from saving them) is honestly lower. ROC-AUCs also shift slightly between the two runs because retention feeds the GA's composite tuning objective, so each run tunes to its own economic landscape. `artifacts/` (CoxPH, the base) and `artifacts_km/` are produced by identical commands except `--survival`.

## Run the tests

```bash
pytest tests/ -m "not slow"   # fast unit tests (~1.5min)
pytest tests/                  # includes the end-to-end smoke tests incl. blend + significance (~2min)
```
