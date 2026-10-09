"""Visualize the NFR2 comparison: CoxPH-based eProfits vs Kaplan-Meier-based
eProfits, per model, from two pipeline runs (--survival cox vs --survival km
on the same dataset).

Usage:
    python visualize_nfr2.py [cox_csv] [km_csv] [output_png]

Defaults: artifacts/model_evaluation.csv, artifacts_km/model_evaluation.csv,
nfr2_comparison.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cox_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/model_evaluation.csv"
km_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts_km/model_evaluation.csv"
out_path = sys.argv[3] if len(sys.argv) > 3 else "nfr2_comparison.png"

cox = pd.read_csv(cox_path).set_index("model")
km = pd.read_csv(km_path).set_index("model")

models = sorted(set(cox.index) & set(km.index))
cox = cox.loc[models]
km = km.loc[models]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, metric, title in zip(
    axes,
    ["eprofits_tenure", "eprofits_avg"],
    ["eProfits (tenure-based retention)", "eProfits (average retention)"],
):
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, cox[metric] / 1e6, width, label="CoxPH", color="#4C72B0")
    ax.bar(x + width / 2, km[metric] / 1e6, width, label="Kaplan-Meier", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("eProfits (millions)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("NFR2: CoxPH vs Kaplan-Meier eProfits, per model (observed result on this run)", fontsize=13)
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"Saved chart to {out_path}")
