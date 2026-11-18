# Phase 4 — Duplicate Sensitivity, Uncertainty and Explainability

## Methodology
The original 60/20/20 chronological split is compared with an alternative that removes **exact duplicate rows before splitting**. Each experiment refits preprocessing on its own training data and selects thresholds using validation data at a target of at least 80% validation recall. Test cohorts differ between experiments; metrics must **not** be interpreted as a controlled treatment effect of deduplication.

## Results

| experiment   | model                  |   n_test |   fraud_test |   roc_auc |   pr_auc |   precision |   recall |   false_positive_rate |   true_positive |   false_positive |   false_negative |   true_negative |   validation_threshold |
|:-------------|:-----------------------|---------:|-------------:|----------:|---------:|------------:|---------:|----------------------:|----------------:|-----------------:|-----------------:|----------------:|-----------------------:|
| original     | Logistic regression    |    56962 |           75 |   0.98203 |  0.74363 |     0.41447 |  0.84000 |               0.00156 |              63 |               89 |               12 |           56798 |                0.96275 |
| original     | Hist gradient boosting |    56962 |           75 |   0.96722 |  0.73538 |     0.09249 |  0.85333 |               0.01104 |              64 |              628 |               11 |           56259 |                0.00117 |
| deduplicated | Logistic regression    |    56746 |           74 |   0.98181 |  0.74746 |     0.42177 |  0.83784 |               0.00150 |              62 |               85 |               12 |           56587 |                0.96495 |
| deduplicated | Hist gradient boosting |    56746 |           74 |   0.97722 |  0.72310 |     0.14352 |  0.83784 |               0.00653 |              62 |              370 |               12 |           56302 |                0.00132 |

## Duplicate overlap

{
  "train_validation_overlap_unique_row_hashes": 0,
  "train_test_overlap_unique_row_hashes": 0,
  "validation_test_overlap_unique_row_hashes": 0,
  "duplicates_removed": 1081
}

## Interpretability
Standardized logistic regression coefficients are exported as a **global directional diagnostic**, not causal explanations. PCA features V1–V28 are anonymized and cannot be assigned meaningful real-world labels. Large coefficients do not establish why a transaction is fraudulent.

## Uncertainty
400 paired bootstrap resamples of the original chronological test cohort estimate 95% percentile intervals for ROC-AUC and PR-AUC. Bootstrap observations are not independent in time, so intervals are exploratory and may understate temporal dependence.

## Limitations
This public dataset spans approximately two days and does not establish live detection effectiveness, real financial savings, or deployment. The Phase 3 Excel dashboard is included unchanged and shows the original experiment; new Phase 4 sensitivity findings are in CSV and this report.
