# Credit Card Fraud Detection — Leakage-Aware Benchmark

An independently reproducible research portfolio project using the [ULB credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Research question
How do logistic regression and histogram-based gradient boosting compare on rare fraud transactions under chronological holdout and validation-selected thresholds?

## Results (duplicate-aware sensitivity test)
| Metric | Logistic regression | Hist gradient boosting |
|---|---:|---:|
| ROC-AUC | 0.9818 | 0.9772 |
| PR-AUC | 0.7475 | 0.7231 |
| Recall | 83.78% | 83.78% |
| Precision | 42.18% | 14.35% |
| False positives | 85 | 370 |

Thresholds were selected using validation data to target at least 80% recall. These figures are descriptive held-out benchmark results, not evidence of statistically significant superiority or real-world cost savings. See `docs/PHASE4_TECHNICAL_REPORT.md` for uncertainty and limitations.

## Repository contents
- `dashboard/Credit_Card_Fraud_Excel_Dashboard.xlsx` — completed, multi-sheet Excel dashboard with updated Phase 4 results.
- `notebooks/` — Jupyter model-comparison and duplicate-sensitivity summaries.
- `scripts/build_phase4.py` — reproduction and sensitivity analysis script.
- `outputs/` — model metrics, confidence intervals, review capacity and feature coefficients.
- `docs/PHASE4_TECHNICAL_REPORT.md` — technical research brief.
- `legacy_original/` — preserved earlier project scripts and charts.
- `requirements.txt` — Python environment dependencies.

## Reproduce
1. Download `creditcard.csv` from the Kaggle link above and place it at the project root.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python scripts/build_phase4.py` from the project root. See the script for expected output paths.

The large raw dataset is **not redistributed** in this repository. The dataset spans roughly two days, is anonymized, and cannot establish prospective effectiveness, fraud savings, or adoption by financial institutions.

## Research relevance
This project demonstrates transferable machine-learning evaluation techniques for highly imbalanced risk detection. It is **not a healthcare claims dataset**, and should not be represented as direct evidence of Medicare/Medicaid payment-integrity outcomes.
