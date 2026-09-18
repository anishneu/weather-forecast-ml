# Weather Forecast ML

Two scikit-learn pipelines built over one historical daily weather dataset.

Weather Forecast ML has no interactive frontend — it's two standalone Python
scripts sharing a single dataset. `Main_Forecast.py` label-encodes the
historical record and trains a Decision Tree classifier per label (heat,
wet) to predict each of the next 31 days, writing the result to a dated
Excel forecast. `Main_Prediction.py` takes weather readings entered at a
prompt and compares three regressors — Linear Regression, KNN, and Random
Forest — for predicting mean temperature, saving a diagnostic plot for each.

Stack: Python 3 · pandas / NumPy · scikit-learn · matplotlib / seaborn · openpyxl

Author: Anish Kuila

## Table of contents
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Scale](#scale)
- [Install](#install)
- [Quickstart](#quickstart)
- [Project structure](#project-structure)
- [Scripts reference](#scripts-reference)
- [Limitations](#limitations)

## What it does

```
data.xlsx ──▶ [Main_Forecast.py] ──▶ data.csv, dataset.png, forecast-<date>.xlsx
                    │
                    ▼
            Decision Tree × 2
        (heat classifier, wet classifier)

data.csv ──▶ [Main_Prediction.py] ──▶ figures/*.png
                    │
        ┌───────────┼───────────────┐
        ▼           ▼               ▼
  Linear Regr.      KNN        Random Forest
        (mean_temp regression, compared side by side)
```

`Main_Forecast.py` doesn't just print predictions — it label-encodes every
column, trains a decision tree per target so heat and wet are scored
independently, reports classifier accuracy, and produces the next 31 days
as a day-by-day Excel forecast rather than a single point estimate.
`Main_Prediction.py` runs the same train/test split through three different
regressors so their R², RMSE, and MAE can be compared directly instead of
trusting one model's fit.

## Architecture

| Layer | Technology |
|---|---|
| Language / runtime | Python 3.9+ |
| Data handling | pandas, NumPy, openpyxl |
| Classification | scikit-learn (`DecisionTreeClassifier`, `LabelEncoder`) |
| Regression | scikit-learn (`LinearRegression`, `KNeighborsRegressor`, `RandomForestRegressor`) |
| Visualization | matplotlib, seaborn |
| Data source | Static Excel/CSV dataset (no external API) |

## Scale

Counted directly from the scripts and dataset:

| Metric | Count |
|---|---|
| Scripts | 2 (classification, regression comparison) |
| ML models trained | 5 (2 decision tree classifiers, linear regression, KNN, random forest) |
| Dataset records | 649 daily readings (August, 1999–2019) |
| Dataset columns | 11 (date parts + 6 weather metrics + 2 labels) |
| Generated figures | 9 (1 dataset histogram + 8 regression/EDA plots) |

## Install

Requires Python 3.9+.

```bash
git clone https://github.com/anishneu/weather-forecast-ml.git
cd weather-forecast-ml
pip install -r requirements.txt
```

## Quickstart

```bash
# 1. Train the classifiers and generate next month's forecast
python Main_Forecast.py

# 2. Compare regression models (prompts for weather readings)
python Main_Prediction.py
```

`Main_Forecast.py` must run first — it converts `data.xlsx` into the
`data.csv` that `Main_Prediction.py` reads directly.

## Project structure

```
weather-forecast-ml/
├── Main_Forecast.py            # heat/wet classification + 31-day forecast
├── Main_Prediction.py          # mean-temperature regression comparison
├── data.xlsx                   # source dataset (daily weather records)
├── dataset.png                 # sample histogram output
├── forecast-2021-07-25.xlsx    # sample forecast output
├── figures/                    # sample regression/EDA plots
├── requirements.txt
└── LICENSE
```

## Scripts reference

| Script | Inputs | Outputs |
|---|---|---|
| `Main_Forecast.py` | `data.xlsx` | `data.csv`, `dataset.png`, `forecast-<date>.xlsx`, console accuracy for heat & wet classifiers |
| `Main_Prediction.py` | `data.csv`, 5 weather readings via prompt | `figures/*.png`, console R² / RMSE / MAE / MSE per model, estimated mean temperature per model |

## Limitations

- No automated test suite.
- No CI/CD pipeline configured.
- Dataset covers a single location's August-only records (1999–2019); no year-round or multi-location coverage.
- No CLI arguments — inputs are hardcoded file paths or interactive prompts.

## License

MIT — see [LICENSE](LICENSE).
