"""Command-line demo: fit the pipeline's best model and render SHAP graphs.
Matches the exact random_state / balancing used by `pipeline.run`.

Run from inside the EPROFITS_with_new_changes folder:
    python shap_demo.py

Produces shap_bar.png, shap_beeswarm.png, shap_waterfall.png in this folder.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from pipeline import data, explain
from pipeline.models import build_model_registry

RANDOM_STATE = 42

print("Loading and preprocessing IBM.csv ...")
processed = data.preprocess(data.load_dataset("IBM.csv"))
x_train, x_test, y_train, y_test = data.split_data(processed, random_state=RANDOM_STATE)
x_train_bal, y_train_bal = data.balance_classes(x_train, y_train, method="smote", random_state=RANDOM_STATE)

print("Fitting random_forest ...")
model, _ = build_model_registry(RANDOM_STATE)["random_forest"]
model.fit(x_train_bal, y_train_bal)

print("Building SHAP explainer and computing values on 300 test customers ...")
explainer = explain.build_explainer("random_forest", model, background=x_train_bal)
exp = explain.compute_shap_values(explainer, x_test, sample_size=300, random_state=RANDOM_STATE)

print("\nGlobal churn drivers:")
print(explain.global_feature_importance(exp).head(10).to_string(index=False))

high_risk_pos = int(np.argmax(exp.values.sum(axis=1)))
print(f"\nExplaining customer at row {high_risk_pos} (highest churn push in this sample):")
print(explain.explain_customer(exp, high_risk_pos, top_n=6).to_string(index=False))

plt.figure(figsize=(8, 6))
shap.plots.bar(exp, show=False, max_display=12)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
shap.plots.beeswarm(exp, show=False, max_display=12)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
shap.plots.waterfall(exp[high_risk_pos], show=False, max_display=10)
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved shap_bar.png, shap_beeswarm.png, shap_waterfall.png in the current folder.")
