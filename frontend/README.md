# EPROFITS frontend

Dashboard UI for the EPROFITS telecom churn project.

## Contents

- `index.html` - dashboard shell and views
- `styles.css` - Sketch-inspired finance dashboard styling
- `app.js` - tab navigation, live pipeline data loading, customer risk simulator, SVG charts, model metrics, and retention panels

## Live pipeline data

On load, the dashboard fetches `../artifacts/dashboard.json` - the file written by `python -m pipeline.run` (see the top-level [README](../README.md)). When it's present, the **IBM Telco baseline** dataset option shows real numbers instead of demo data:

- **Overview** - real churn rate, churned-customer count, and eProfits from the best-performing model
- **Models** - the real evaluation table for every model actually trained (not a fixed 2-3 row demo), with the best one starred, plus real SHAP-based "XAI Drivers"
- **Customer → Real Customer Explanations** - the highest-risk real customers from the run, ranked by predicted churn probability; selecting one shows the actual SHAP feature contributions behind that prediction

If `dashboard.json` doesn't exist yet (pipeline hasn't been run) or can't be fetched, the dashboard falls back to its built-in demo data automatically - nothing breaks, it just won't show the "live pipeline run" note.

Switching the dataset selector to "High-risk pilot sample," "Loyal customer sample," or an uploaded CSV always shows illustrative/heuristic data, since the pipeline hasn't scored those - only "IBM Telco baseline" reflects a real trained-model run.

The customer risk **sliders** (Contract, Internet, Tenure, etc.) remain a separate, frontend-only what-if simulator using hand-set weights - they are not the same thing as the real SHAP explanations panel below them.

## Run

Because the dashboard fetches a local JSON file, opening `index.html` directly (`file://`) will fail silently in most browsers (the fetch is blocked) and you'll only see demo data. Serve it over HTTP instead, from the `EPROFITS_with_new_changes` folder (one level up from this one) so the relative path to `artifacts/` resolves:

```bash
cd EPROFITS_with_new_changes
python -m http.server 8000
```

Then open `http://localhost:8000/frontend/index.html`.

To see real data, run the pipeline first (from the same folder):

```bash
python -m pipeline.run --data IBM.csv
```

This writes `artifacts/dashboard.json`; refresh the dashboard page afterward.
