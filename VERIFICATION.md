# Verification

Created by Blessing Sekeramavi.

Verified in the build environment:
- CSV and multi-sheet XLSX loading; empty/unsupported/invalid inputs.
- Quality flags, explicit cleaning, original-data preservation and filters.
- Regression and classification candidate comparison, test metrics, baseline comparison.
- Disjoint train/validation/test partitions; entity separation and chronological ordering.
- Train/development-only preprocessing statistics and missing-target exclusion.
- Single and batch prediction engine, UI single prediction, preserved identifiers and row order.
- Missing/incompatible input columns, out-of-range values and unseen/boolean categories.
- Forecasting against naive and seasonal-naive baselines; changing final test outcomes does not change backtest predictions.
- Clustering/anomaly fits and supported new-record scoring.
- Generated scripts executed for regression, classification and forecasting.
- Generated notebook cell executed against the included regression example.
- Export packages and HTML chart/report content.
- Streamlit UI navigation and trained-model screens.
- Running server: /_stcore/health returned ok; / returned HTTP 200.

Run `python -m pytest -q` to repeat the tests. See tested_versions.json for the environment versions.
Public deployment verified on Streamlit Community Cloud: https://blessing-modelmatch.streamlit.app/ . The hosted regression workflow completed successfully (Ridge; held-out test MAE 42.86 vs dummy baseline MAE 64.01). The local automated suite passed 12 tests. Hosted dependency versions may differ from tested_versions.json; exports include the live runtime versions.

## Premium workspace and BI Studio update
- 35 automated tests passed after the UI and code-workbench update.
- Read-only SQL aggregates and ordinary CTE queries validated. Writes, system-table access, extensions, PRAGMA, multiple statements and recursive SQL rejected.
- Supported pandas filtering, arithmetic, grouping and aggregation validated. Imports, arbitrary attributes, file-writing, eval/query/apply/pipe and unbounded arithmetic paths rejected.
- BI Studio navigation, running queries, pinning results and overview navigation shortcuts verified using Streamlit AppTest.
- Original modelling, prediction, export and generated-code execution checks continue to pass.

## Automatic analysis update
- Upload-triggered chart planning verified for numeric, category, time, grouped and geographic fields.
- Numeric correlations exclude confirmed identifiers. Geographic plots require supplied, valid coordinate pairs.
- Time chart tests verify missing periods remain gaps and monthly values remain aligned.
- Target prioritisation tested for regression, classification and forecasting examples.
- Streamlit integration test runs the full automatic comparison/evaluation/prediction workflow and verifies prediction row count equals the held-out test size.
- Full interactive report generation and automatic dashboard creation tested.
