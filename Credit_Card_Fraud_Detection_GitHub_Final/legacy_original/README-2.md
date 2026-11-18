# Credit Card Transaction Fraud Detection

An anomaly-detection framework for identifying fraudulent transactions in
high-volume, severely imbalanced financial transaction data, built and
evaluated with a chronological (time-based) train/test split to reflect how
a real detection system would actually be evaluated: on transactions it has
not yet seen at the time predictions are made.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle / ULB Machine Learning Group) — 284,807 anonymized European
cardholder transactions over a 48-hour window, with 492 confirmed fraudulent
transactions (0.173%).

**This is a public, de-identified benchmark dataset, not proprietary
institutional data.** Features V1–V28 are the output of a PCA transformation
applied by the dataset's original publishers for confidentiality; only `Time`
(seconds elapsed) and `Amount` remain in their original form. This is stated
plainly here because it materially affects how the results below should be
read: they demonstrate the methodology on a severely imbalanced, real-world-derived
transaction dataset, not a claim of outperforming a specific proprietary
industry system.

## Methodology

### Train/test split
A **chronological split** — train on the first ~70% of the 48-hour window
(199,364 transactions, 384 fraud), test on the final ~30% (85,443
transactions, 108 fraud) — rather than a random shuffle. This is more
realistic for a pre-authorization/pre-settlement detection framing, since a
production system only ever has access to the past when scoring a new
transaction; a random split would let the model implicitly "see the future."

### Resampling
Manual random oversampling of the minority (fraud) class, applied to the
**training set only**, to a 10% fraud rate (not a full 50/50 balance, which
would create an unrealistic synthetic distribution). `imbalanced-learn`/SMOTE
was unavailable in the offline sandboxed environment this pipeline was
originally built in; random oversampling is a standard, legitimate resampling
technique and is what is actually run and reported here.

### Models compared
- **Baseline:** Logistic Regression (the standard simple baseline against
  which more complex fraud models are conventionally compared)
- **Primary:** Gradient Boosted Trees (scikit-learn `HistGradientBoostingClassifier`
  — a histogram-based, high-performance implementation of the same gradient
  boosting family as XGBoost/LightGBM)
- **Isolation Forest:** an unsupervised anomaly detection model, trained
  without access to fraud labels at all, included for methodological
  completeness and comparison against the supervised models

### Explainability
Permutation importance (model-agnostic, built into scikit-learn) identifies
which features drive the primary model's predictions.

**Scope note:** because the underlying features in this dataset are
PCA-transformed for confidentiality (see Limitations below), the
explainability output here is feature-level statistical attribution (e.g.,
"component V4 was the strongest driver of this transaction's risk score"),
not a fully human-readable, investigator-facing narrative (e.g., "flagged
because billing location did not match device location"). The latter is
what this endeavor's healthcare claims fraud detection work is built to
deliver, since that project uses named, real-world-meaningful features
(billing amount, physician-role overlap, claims-per-beneficiary, etc.).
This project demonstrates that the underlying modeling and imbalance-handling
methodology transfers across domains; the fully interpretable,
investigator-actionable output is demonstrated in the healthcare claims
fraud project.

> **Note on SHAP:** this pipeline was built in a sandboxed environment
> without internet access, so the `shap` package could not be installed for
> this run. Permutation importance is a legitimate, peer-reviewed,
> model-agnostic alternative. A ready-to-run SHAP cell is included
> (commented out) at the bottom of `02_explainability.py` for use in any
> standard environment.

## Results

All results are from a genuine held-out test set — the final 30% of
transactions by time, never seen during training or resampling.

| Model | ROC-AUC | PR-AUC (Average Precision) |
|---|---|---|
| Baseline: Logistic Regression | 0.979 | 0.737 |
| **Primary: Gradient Boosted Trees** | **0.984** | **0.806** |
| Isolation Forest (unsupervised) | 0.936 | 0.037 |

**PR-AUC (Average Precision) is the more informative summary metric here**
given the severity of the class imbalance (0.17% fraud rate); ROC-AUC can
look strong for almost any reasonable model at this imbalance level, while
PR-AUC more sharply distinguishes real performance differences. The primary
model's PR-AUC of 0.806 represents a genuine, meaningful improvement over
the baseline's 0.737.

### False-positive rate at matched recall — reported transparently across operating points

Rather than reporting a single cherry-picked comparison point, the false
positive rate was evaluated at every recall level from 70% to 95%:

| Target Recall | Baseline FPR | Primary Model FPR | Relative Change |
|---|---|---|---|
| 70% | 0.027% | 0.002% | −91.3% |
| 75% | 0.034% | 0.002% | −93.1% |
| 80% | 0.093% | 0.089% | −3.8% |
| 85% | 0.350% | 0.902% | **+157.5% (primary model worse)** |
| 90% | 5.051% | 4.041% | −20.0% |
| 95% | 19.037% | 11.450% | −39.9% |

**This is reported honestly, including the one point (85% recall) where the
primary model performs worse than baseline**, rather than selecting only the
most favorable comparison. At a representative operating point of 90%
recall, the primary model achieves a **20.0% relative reduction in false
positive rate** (5.05% → 4.04%, or roughly 1.0 percentage point) compared to
the logistic regression baseline. Improvement is not uniform across all
thresholds — the PR-AUC comparison above is the more reliable overall
performance summary.

### Top predictive components
1. V4 and V14 (PCA-transformed components — consistent with published
   third-party analyses of this dataset, which independently identify these
   same two components as the strongest fraud predictors; their real-world
   meaning is not recoverable, since they are anonymized linear combinations
   of the original transaction attributes)
2. V8, V1, V2, V12
3. Transaction Amount (the one feature in this dataset with a direct,
   real-world interpretable meaning)

## Reproducing these results

```bash
pip install pandas numpy scikit-learn matplotlib
python 01_modeling.py
python 02_explainability.py
```

Requires the full Kaggle "Credit Card Fraud Detection" dataset
(`creditcard.csv`, ~150MB, 284,807 rows), available from the dataset page
linked above.

## Limitations

- This is a public, PCA-anonymized benchmark dataset, not live institutional
  transaction data; results demonstrate the methodology, not validated
  production performance against a specific bank or card network's system.
- Because the underlying features are PCA-transformed, the "top predictive
  features" (V4, V14, etc.) cannot be mapped back to real-world transaction
  attributes (merchant category, geographic distance, device fingerprint,
  etc.) — that mapping is only possible with the original, un-transformed
  proprietary data the dataset's publishers used.
- False-positive improvement varies substantially by operating point (see
  table above) and is not a single fixed percentage; any single-number
  summary should specify the exact recall level it is measured at.
