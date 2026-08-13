"""
Credit Card Transaction Fraud Detection
Baseline vs Primary Model with Time-Based Split and Resampling

Dataset: Kaggle "Credit Card Fraud Detection" -- 284,807 anonymized European
cardholder transactions over 48 hours, 492 confirmed fraudulent (0.173%).
Features V1-V28 are PCA-transformed for confidentiality (this is a public,
de-identified benchmark dataset, not proprietary institutional data -- that
is stated plainly here and in the README rather than implied otherwise).

Methodology notes:
- TIME-BASED split (not random shuffle): train on the first ~70% of the
  48-hour window, test on the final ~30%. This is more realistic for a
  "detect before settlement" framing than a random split, since it never
  lets the model see the future to predict the past.
- Manual random oversampling of the minority (fraud) class on the TRAINING
  set only. imbalanced-learn/SMOTE was unavailable in this sandboxed,
  offline environment; random oversampling is a standard, legitimate
  resampling technique and is what is actually run and reported here.
- Isolation Forest is included as an unsupervised anomaly-detection method
  (named explicitly in the petition) alongside the supervised baseline and
  primary models, since unsupervised anomaly detection and supervised
  classification answer different but complementary questions.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("/mnt/user-data/uploads/creditcard.csv")
print(f"Total transactions: {len(df):,}")
print(f"Fraudulent: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

# ---------------------------------------------------------------------------
# 2. Time-based train/test split (chronological, not random)
# ---------------------------------------------------------------------------
df = df.sort_values("Time").reset_index(drop=True)
split_point = int(len(df) * 0.70)
split_time = df.iloc[split_point]["Time"]

train_df = df[df["Time"] < split_time].copy()
test_df = df[df["Time"] >= split_time].copy()

print(f"\nTrain: {len(train_df):,} transactions ({train_df['Class'].sum()} fraud, "
      f"{train_df['Class'].mean()*100:.3f}%), first {split_time/3600:.1f} hours")
print(f"Test:  {len(test_df):,} transactions ({test_df['Class'].sum()} fraud, "
      f"{test_df['Class'].mean()*100:.3f}%), remaining {(df['Time'].max()-split_time)/3600:.1f} hours")

feature_cols = [c for c in df.columns if c not in ("Class",)]

X_train, y_train = train_df[feature_cols], train_df["Class"]
X_test, y_test = test_df[feature_cols], test_df["Class"]

# Scale Amount and Time (V1-V28 are already PCA-standardized)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

# ---------------------------------------------------------------------------
# 3. Manual random oversampling of the minority class -- TRAINING SET ONLY
# ---------------------------------------------------------------------------
train_fraud = X_train_scaled[y_train == 1]
train_legit = X_train_scaled[y_train == 0]

n_target = len(train_legit) // 10  # oversample fraud to 10% of legit count (not full 50/50 --
                                    # avoids an unrealistic synthetic class balance)
fraud_oversampled = train_fraud.sample(n=n_target, replace=True, random_state=RANDOM_STATE)

X_train_resampled = pd.concat([train_legit, fraud_oversampled], ignore_index=True)
y_train_resampled = pd.concat(
    [pd.Series(0, index=train_legit.index), pd.Series(1, index=fraud_oversampled.index)],
    ignore_index=True
)
# shuffle
shuffle_idx = np.random.RandomState(RANDOM_STATE).permutation(len(X_train_resampled))
X_train_resampled = X_train_resampled.iloc[shuffle_idx].reset_index(drop=True)
y_train_resampled = y_train_resampled.iloc[shuffle_idx].reset_index(drop=True)

print(f"\nAfter resampling training set: {len(X_train_resampled):,} rows "
      f"({y_train_resampled.sum():,} fraud, {y_train_resampled.mean()*100:.2f}%)")

# ---------------------------------------------------------------------------
# 4. Baseline: Logistic Regression on resampled, scaled features
# ---------------------------------------------------------------------------
baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
baseline.fit(X_train_resampled, y_train_resampled)
baseline_proba = baseline.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# 5. Primary model: Gradient Boosted Trees on resampled features
# ---------------------------------------------------------------------------
primary = HistGradientBoostingClassifier(
    max_iter=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
)
primary.fit(X_train_resampled, y_train_resampled)
primary_proba = primary.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# 6. Isolation Forest: unsupervised anomaly detection (no labels used in fit)
# ---------------------------------------------------------------------------
iso_forest = IsolationForest(
    n_estimators=100, contamination=float(y_train.mean()), random_state=RANDOM_STATE, n_jobs=-1
)
iso_forest.fit(X_train_scaled)  # unsupervised -- fit on features only, no labels
# decision_function: higher = more normal, lower = more anomalous. Flip sign so
# higher = more anomalous, consistent with the other two models' "proba" convention.
iso_scores = -iso_forest.decision_function(X_test_scaled)

# ---------------------------------------------------------------------------
# 7. Evaluate all three
# ---------------------------------------------------------------------------
results = {}
for name, scores in [("Baseline (Logistic Regression)", baseline_proba),
                      ("Primary (Gradient Boosted Trees)", primary_proba),
                      ("Isolation Forest (unsupervised)", iso_scores)]:
    auc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)
    results[name] = {"ROC_AUC": auc, "PR_AUC": ap}
    print(f"\n{name}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC (Average Precision): {ap:.4f}")

# ---------------------------------------------------------------------------
# 8. False-positive rate comparison at matched recall
# ---------------------------------------------------------------------------
def fpr_at_recall(y_true, scores, target_recall):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = np.searchsorted(tpr, target_recall)
    idx = min(idx, len(tpr) - 1)
    return fpr[idx], tpr[idx], thresholds[idx]

target_recall = 0.80
print("\n" + "=" * 70)
print(f"FALSE POSITIVE RATE COMPARISON AT MATCHED RECALL (target = {target_recall})")
print("=" * 70)

baseline_fpr, baseline_recall, baseline_thresh = fpr_at_recall(y_test, baseline_proba, target_recall)
primary_fpr, primary_recall, primary_thresh = fpr_at_recall(y_test, primary_proba, target_recall)

print(f"\nBaseline (Logistic Regression):")
print(f"  At recall = {baseline_recall:.3f}, false positive rate = {baseline_fpr:.5f} ({baseline_fpr*100:.3f}%)")

print(f"\nPrimary (Gradient Boosted Trees):")
print(f"  At recall = {primary_recall:.3f}, false positive rate = {primary_fpr:.5f} ({primary_fpr*100:.3f}%)")

relative_reduction = (baseline_fpr - primary_fpr) / baseline_fpr * 100 if baseline_fpr > 0 else None
pct_point_reduction = (baseline_fpr - primary_fpr) * 100
print(f"\n>>> Relative FPR reduction: {relative_reduction:.1f}%")
print(f">>> Percentage-point FPR reduction: {pct_point_reduction:.3f} percentage points")

# In raw transaction counts, since percentages are tiny at this scale
n_legit_test = (y_test == 0).sum()
baseline_fp_count = int(round(baseline_fpr * n_legit_test))
primary_fp_count = int(round(primary_fpr * n_legit_test))
print(f"\nIn absolute terms (test set has {n_legit_test:,} legitimate transactions):")
print(f"  Baseline flags {baseline_fp_count:,} legitimate transactions as false positives")
print(f"  Primary model flags {primary_fp_count:,} legitimate transactions as false positives")

# ---------------------------------------------------------------------------
# 9. Confusion matrix at matched-recall threshold + full classification report
# ---------------------------------------------------------------------------
primary_preds = (primary_proba >= primary_thresh).astype(int)
cm = confusion_matrix(y_test, primary_preds)
print(f"\nPrimary model confusion matrix at recall={primary_recall:.3f} threshold:")
print(f"                    Predicted Legit   Predicted Fraud")
print(f"Actual Legit          {cm[0][0]:>7,}          {cm[0][1]:>7,}")
print(f"Actual Fraud          {cm[1][0]:>7,}          {cm[1][1]:>7,}")

print("\nFull classification report (primary model, 0.5 threshold):")
print(classification_report(y_test, (primary_proba >= 0.5).astype(int),
                             target_names=["Legitimate", "Fraud"]))

# ---------------------------------------------------------------------------
# 10. Save everything
# ---------------------------------------------------------------------------
summary = {
    "n_total_transactions": int(len(df)),
    "n_total_fraud": int(df["Class"].sum()),
    "fraud_rate_overall": float(df["Class"].mean()),
    "split_method": "time-based (chronological), 70/30",
    "n_train": int(len(X_train)),
    "n_train_fraud": int(y_train.sum()),
    "n_test": int(len(X_test)),
    "n_test_fraud": int(y_test.sum()),
    "resampling_method": "manual random oversampling of minority class, training set only, to 10% fraud rate",
    "baseline_roc_auc": float(results["Baseline (Logistic Regression)"]["ROC_AUC"]),
    "baseline_pr_auc": float(results["Baseline (Logistic Regression)"]["PR_AUC"]),
    "primary_roc_auc": float(results["Primary (Gradient Boosted Trees)"]["ROC_AUC"]),
    "primary_pr_auc": float(results["Primary (Gradient Boosted Trees)"]["PR_AUC"]),
    "isoforest_roc_auc": float(results["Isolation Forest (unsupervised)"]["ROC_AUC"]),
    "isoforest_pr_auc": float(results["Isolation Forest (unsupervised)"]["PR_AUC"]),
    "target_recall": target_recall,
    "baseline_fpr_at_target_recall": float(baseline_fpr),
    "baseline_actual_recall": float(baseline_recall),
    "primary_fpr_at_target_recall": float(primary_fpr),
    "primary_actual_recall": float(primary_recall),
    "relative_fpr_reduction_pct": float(relative_reduction) if relative_reduction is not None else None,
    "pct_point_fpr_reduction": float(pct_point_reduction),
    "n_legit_test": int(n_legit_test),
    "baseline_fp_count": baseline_fp_count,
    "primary_fp_count": primary_fp_count,
}

with open("/home/claude/credit_card_analysis/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

with open("/home/claude/credit_card_analysis/primary_model.pkl", "wb") as f:
    pickle.dump(primary, f)

X_test_scaled.assign(Label=y_test.values, PrimaryModelProba=primary_proba, BaselineProba=baseline_proba).to_csv(
    "/home/claude/credit_card_analysis/test_predictions.csv", index=False
)
X_train_resampled.to_csv("/home/claude/credit_card_analysis/X_train_resampled.csv", index=False)

print("\nSaved: results_summary.json, primary_model.pkl, test_predictions.csv, X_train_resampled.csv")
