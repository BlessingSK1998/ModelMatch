# ModelMatch
Created by **Blessing Sekeramavi**.

A runnable Streamlit data science workspace: upload → quality review → interactive exploration → goal selection → validation comparison → held-out evaluation → predictions → exports.

## Live application
[Open ModelMatch](https://blessing-modelmatch.streamlit.app/)

Deployed on Streamlit Community Cloud. The source includes public/synthetic examples; uploaded datasets remain session-scoped.

## Run
Python 3.11 or 3.12 recommended.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501. Run checks with `python -m pytest -q`.

## Deployment: Streamlit Community Cloud
1. Create a GitHub repository and commit this project (including modelmatch/, examples/, requirements.txt and app.py).
2. Do not commit private datasets, .env, secrets.toml or exported models. Synthetic/public example data is intentionally included.
3. Sign into https://share.streamlit.io/ and select **Create app**.
4. Choose the repository and branch; main file path: **app.py**.
5. Under advanced settings choose Python 3.11 or 3.12; click Deploy.
6. Wait for dependency installation, open the assigned URL, run each example, and test one uploaded CSV and XLSX before sharing.
The live application is deployed at https://blessing-modelmatch.streamlit.app/. Community Cloud apps can sleep and have resource limits; this is not an uptime guarantee.

## BI Studio
A report canvas with a field pane, configurable visuals, KPI cards, category filters, title editing, visual ordering, and JSON project save/reopen. Dataset-backed visuals use the active data and report filters. Pinned query results are clearly labelled snapshots and are independent of later report filters.

The code workbench supports read-only SQLite SELECT queries against `dataset` and a deliberately restricted, interpreted pandas-expression subset using `df`. Query results can be downloaded and pinned as a table or chart. SQL runs in a fresh in-memory database with an authorizer, a small function allowlist, progress limits and disabled extensions. Pandas expressions are parsed and interpreted through an AST allowlist; Python `eval`/`exec`, imports, user-defined functions, files, external APIs and arbitrary server-side Python are not used. Limits: 10,000 input records, 5,000 displayed result rows, 10,000 code characters, 250 expression nodes and 12 report visuals. SQL cancellation uses a progress/time budget; a single pandas operation cannot be preempted mid-operation.

For full Python, the scratchpad accepts scripts and exports `.py` and `.ipynb` plus the active dataset separately. Full scripts must be reviewed and run in the user's own Python/Jupyter environment. Session query history is not saved to a database. Report project exports include any pinned query-result records; treat those exports as sensitive when applicable.

## Design
A responsive navy navigation rail, light report canvas, consistent typography, indigo/teal chart theme, page-specific workspace headers, a calculated overview snapshot and workflow shortcuts. All charts remain based on actual records; decorative styling adds no fabricated scores or insights.

## Workflow
- Original data is preserved in session state. Confirm semantic types, preview cleaning, then apply. Cleaning invalidates trained results.
- Raw original and cleaned active data are separate; filters only affect dashboard outputs, not training.
- CSV / XLSX (sheet selection), three built-in datasets, inferred roles, quality flags and date coverage.
- Histogram, frequency, box, scatter, heatmap, group comparison, time trend and explicit lat/lon map charts. Every chart has standalone interactive HTML export. Geographic names are not geocoded.
- Dashboard filters apply consistently to KPIs, charts and data; table search is separately labelled. Settings JSON can be reloaded. Data is supplied separately to the standalone dashboard package.
- Regression: dummy, Ridge, random forest, histogram gradient boosting. Classification: dummy, balanced logistic regression, balanced random forest, histogram gradient boosting. Explicit goals allow numeric category targets.
- Train/validation/test approximately 60/20/20; time splits keep equal timestamps together. Group splits keep entities apart. Three-fold CV only within fit partition where feasible. Selection uses validation MAE or macro F1. Refit selected model on development data; evaluate once on untouched test. Permutation importance uses validation, not test.
- Pipelines fit imputers, scalers, and bounded one-hot encoders on training folds only. Dates become year/month/weekday/day features. Missing target rows are excluded and counted. IDs/group fields default excluded. Unseen categories are supported with warnings.
- Clustering: K-means or mini-batch K-means; in-sample profiles and bounded silhouette sampling. Anomaly detection: Isolation Forest, configurable contamination, scores/flags, no accuracy claim.
- Univariate forecasting: regular aggregation, naive, seasonal naive (when sufficient cycles), recursive lag Ridge; fixed-origin validation and test horizon, selected model fitted to all history for future forecasts. Empty periods rejected. No future target values enter features.
- Prediction Studio: schema-derived forms, batch CSV/XLSX with template, type validation, extra-column preservation and unknown/out-of-range warnings; what-if comparison, forecasts and uncalibrated probabilities.
- Export cleaned/filtered/predicted data, chart HTML, HTML report, validation metrics, test metrics, fitted model with versions, deterministic script/notebook, standalone dashboard project.

## Resource and privacy policy
Default upload limit 25 MB; lower it with MODELMATCH_MAX_UPLOAD_MB and the Streamlit server limit. Hard limits: 30,000 rows, 150 columns; model training 10,000 rows and 40 features; bounded categories 25, scatter/map 5,000 points, silhouette sample 1,500, importance 500 validation records. Fixed model budgets replace unbounded searches. No strict wall-clock cancellation is available; large feature matrices can still be expensive. Avoid concurrent heavy training on small hosts.
Data is held in per-session memory, without global data caches or persistent upload directories. No uploaded code is executed, records logged, or external AI used. Downloaded artifacts may contain sensitive data or fitted parameters; handle privately. Clear-data reset resets session state, but does not erase files you already downloaded. Model deserialization is disabled in the UI. Load serialized models only from trusted sources with matching dependencies.

## Limitations
This is a general first version, not an automated substitute for domain validation. Automated leakage flags cannot discover all post-outcome variables. Correlation, importance and what-if predictions are not causal. Calendar feature extraction is deliberately simple. Category encodings may merge rare levels. No custom arbitrary code, deep learning, hyperparameter search, calibrated probabilities, confidence/prediction intervals, multiseries forecasts, external forecast regressors, image/PDF ingestion or external AI assistant is implemented. There is no saved multiuser project database; dashboard settings are saved as JSON and records remain session-scoped. Notebook paths must be changed to match your dataset; keep modules alongside exports. Reports may show filtered EDA and full active-data model metrics, clearly separated.

## Examples
- health_regression.csv: scikit-learn diabetes dataset; standardized measurements, target disease progression (source: sklearn.datasets.load_diabetes).
- iris_classification.csv: scikit-learn Iris dataset, species names (sklearn.datasets.load_iris).
- daily_sales.csv: synthetic seeded daily series; not real business results.
- examples.xlsx: regression/classification sheets.
