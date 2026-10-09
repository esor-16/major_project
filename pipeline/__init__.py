"""Phase-2 churn analysis pipeline.

Layered modules replacing the Phase-1 notebook (`ibm_model_another_genetic.ipynb`):

- data            : ingestion, preprocessing, class-imbalance handling
- survival        : CoxPH-based retention probability estimation, plus a
                     Kaplan-Meier baseline for the NFR2 comparison
- eprofits        : eProfits / EMP profitability metric engine
- models          : six-classifier registry + GA-based hyperparameter tuning
- feature_fusion  : feature-fusion hybrid model (donor SHAP -> fused feature
                     ranking -> GA-tuned recipient on the selected subset)
- explain         : SHAP explainability, global + per-customer
- evaluate        : unified traditional + profitability evaluation
- dashboard_export: packages results/SHAP into dashboard.json for frontend/
- run             : end-to-end orchestration / CLI entry point
"""
