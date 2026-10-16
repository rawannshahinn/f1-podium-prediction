# F1 Podium Prediction — Milestone 1 (Team 11)

Predict, for every driver in every Formula 1 race, whether he finishes on the **podium (top 3)**, using only information available **after qualifying and before the race starts**.

- **Kaggle notebook (runs top to bottom):** https://www.kaggle.com/code/nouransamieh/f1-podium-prediction-milestone-1-team-11
- **Report:** [`report/F1_Podium_Prediction_Report.pdf`](report/F1_Podium_Prediction_Report.pdf)
- **Notebook file:** [`notebook/f1-podium-prediction-milestone-1-team-11.ipynb`](notebook/f1-podium-prediction-milestone-1-team-11.ipynb) (Kaggle version 13, with all outputs and charts)
- **Data:** [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) — Kaggle / Ergast, 14 relational CSV tables (not stored in this repo; attach the dataset on Kaggle).

## Team

| Name | ID |
|-|-|
| Rawan Shahin | 58-0516 |
| Omar Elghazouly | 58-2262 |
| Nouran Ahmed | 58-0948 |
| Rawan Osama | 58-1750 |

## Results (test = 2022–2024 seasons, never used for any choice)

The models are trained on **all seasons, 1950–2018** (24,525 driver-races in the modeling table), validated on 2019–2021 and tested on 2022–2024.

| Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1 |
|-|-|-|-|-|
| Rule: podium if grid ≤ 3 | 0.683 | 0.647 | 0.913 | 0.649 |
| Logistic regression (baseline) | 0.729 | 0.718 | 0.936 | 0.630 |
| Random forest, tuned | 0.743 | 0.716 | 0.939 | 0.679 |
| **XGBoost, tuned (final)** | **0.745** | **0.711** | **0.940** | **0.639** |
| Shallow neural network (Keras), tuned | 0.716 | 0.717 | 0.935 | 0.681 |

PR-AUC is the main metric (only about 14% of drivers finish on the podium), ROC-AUC the secondary one; the final model is the attempt with the best validation PR-AUC, and its probability threshold (0.476) was chosen on validation and applied unchanged to test.

## What the notebook contains

| Section | Content |
|-|-|
| 1–2 | Setup (constants, seeds, shared `evaluate()`), loading all 14 tables as-is |
| 3 | Audit: types, `\N` sentinels, 41 primary / natural / foreign-key checks, duplicate driver-race pairs |
| 4 | EDA before cleaning (qualifying coverage, grid, class balance, retirements, …) |
| 5 | Cleaning: sentinels → NaN, semantic typing (dates, lap times → seconds), Indy 500 / non-starters / shared drives removed → one row per (raceId, driverId) |
| 6 | EDA after cleaning (before vs after) |
| 7 | Data-engineering questions Q1 (front-row conversion by circuit, before/after 2014), Q2 (home advantage controlling for grid), Q3 (mechanical retirements by constructor) |
| 8 | Feature engineering on all seasons: 20 leakage-safe features in 6 groups (grid, qualifying, driver form, constructor form, standings, context), leakage checks, one chart per group, correlation heatmap |
| 9 | Time-based split (train 1950–2018, validation 2019–2021, test 2022–2024), 3 options compared |
| 10 | Models: grid rule, logistic regression, random forest, XGBoost, shallow NN — 13 labelled attempts, thresholds on validation, test scored once |
| 11 | Feature-group ablation, two preprocessing-order experiments, training years (all seasons vs 1994+ / 2003+, with bootstrap), missing-value handling (6 options), withdrawn drivers (keep vs drop) |
| 12 | XAI: permutation importance, SHAP (beeswarm, bar, waterfall), LIME (two drivers) |
| 13 | Inference function `predict_podium(record)` — demo on a complete and a partially-missing record |

Every decision (22 in total) is written in the notebook as *Decision / Why / Evidence* and summarised in the report's appendix.

## How to run

Open the notebook on Kaggle with the dataset attached and click **Run All** (about 15–20 minutes on CPU). Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, tensorflow/keras, shap, lime (all in the Kaggle image). The data folder is found automatically (`find_data_dir()`), so it also runs locally with the CSVs in `data/`.

> Model explanations show what the model **relies on**, not what **causes** podiums.
