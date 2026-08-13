"""
Credit Card Transaction Fraud Detection
Explainability + Visualizations

Same approach as the healthcare provider fraud project: permutation importance
(model-agnostic, scikit-learn built-in) used in place of SHAP, since this
sandboxed environment has no internet access to install the shap package. A
ready-to-run SHAP cell is included, commented out, at the bottom.
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, precision_recall_curve

with open("/home/claude/credit_card_analysis/primary_model.pkl", "rb") as f:
    model = pickle.load(f)

test_df = pd.read_csv("/home/claude/credit_card_analysis/test_predictions.csv")
feature_cols = [c for c in test_df.columns if c not in ("Label", "PrimaryModelProba", "BaselineProba")]
X_test = test_df[feature_cols]
y_test = test_df["Label"]

with open("/home/claude/credit_card_analysis/results_summary.json") as f:
    summary = json.load(f)

# ---------------------------------------------------------------------------
# 1. Global feature importance via permutation importance
# ---------------------------------------------------------------------------
# Use a stratified subsample for speed -- 85k rows x 20 repeats is expensive;
# a 15,000-row stratified sample gives stable estimates without the full runtime.
rng = np.random.RandomState(42)
fraud_idx = X_test[y_test == 1].index
legit_idx = X_test[y_test == 0].sample(n=14500, random_state=42).index
sample_idx = fraud_idx.union(legit_idx)

perm_result = permutation_importance(
    model, X_test.loc[sample_idx], y_test.loc[sample_idx],
    n_repeats=15, random_state=42, scoring="roc_auc", n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
}).sort_values("importance_mean", ascending=False)

print("Top 10 features by permutation importance (impact on ROC-AUC):")
print(importance_df.head(10).to_string(index=False))
importance_df.to_csv("/home/claude/credit_card_analysis/feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 6))
top_n = importance_df.head(12).iloc[::-1]
ax.barh(top_n["feature"], top_n["importance_mean"], xerr=top_n["importance_std"], color="#2E5C8A")
ax.set_xlabel("Permutation Importance (mean decrease in ROC-AUC when shuffled)")
ax.set_title("Feature Importance — Credit Card Fraud Detection Model")
plt.tight_layout()
plt.savefig("/home/claude/credit_card_analysis/feature_importance.png", dpi=150)
print("\nSaved feature_importance.png")

# ---------------------------------------------------------------------------
# 2. ROC and Precision-Recall curves, baseline vs primary
# ---------------------------------------------------------------------------
baseline_proba = test_df["BaselineProba"]
primary_proba = test_df["PrimaryModelProba"]

fpr_b, tpr_b, _ = roc_curve(y_test, baseline_proba)
fpr_p, tpr_p, _ = roc_curve(y_test, primary_proba)
prec_b, rec_b, _ = precision_recall_curve(y_test, baseline_proba)
prec_p, rec_p, _ = precision_recall_curve(y_test, primary_proba)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr_p, tpr_p, color="#2E5C8A", label=f"Primary (ROC-AUC={summary['primary_roc_auc']:.3f})")
axes[0].plot(fpr_b, tpr_b, color="#B0B0B0", label=f"Baseline (ROC-AUC={summary['baseline_roc_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "--", color="lightgray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("ROC Curve — Test Set (final 30% by time)")
axes[0].legend()

axes[1].plot(rec_p, prec_p, color="#2E5C8A", label=f"Primary (PR-AUC={summary['primary_pr_auc']:.3f})")
axes[1].plot(rec_b, prec_b, color="#B0B0B0", label=f"Baseline (PR-AUC={summary['baseline_pr_auc']:.3f})")
axes[1].axhline(y=summary["fraud_rate_overall"], linestyle="--", color="lightgray", label="No-skill baseline")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Test Set")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/claude/credit_card_analysis/roc_pr_curves.png", dpi=150)
print("Saved roc_pr_curves.png")

# ---------------------------------------------------------------------------
# 3. FPR-vs-recall tradeoff chart (visualizes the honest, non-uniform result)
# ---------------------------------------------------------------------------
scan = summary["recall_scan"]
targets = [r["target_recall"] for r in scan]
b_fprs = [r["baseline_fpr_pct"] for r in scan]
p_fprs = [r["primary_fpr_pct"] for r in scan]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(targets, b_fprs, marker="o", color="#B0B0B0", label="Baseline (Logistic Regression)")
ax.plot(targets, p_fprs, marker="o", color="#2E5C8A", label="Primary (Gradient Boosted Trees)")
ax.set_xlabel("Target Recall")
ax.set_ylabel("False Positive Rate (%)")
ax.set_yscale("log")
ax.set_title("False Positive Rate vs. Recall Target\n(log scale — improvement is not uniform across operating points)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/credit_card_analysis/fpr_vs_recall.png", dpi=150)
print("Saved fpr_vs_recall.png")

# ---------------------------------------------------------------------------
# 4. Example: highest-risk-scored test transaction explanation
# ---------------------------------------------------------------------------
train_means = X_test.mean()  # test-set means as reference distribution
test_df_sorted = test_df.sort_values("PrimaryModelProba", ascending=False)
example = test_df_sorted.iloc[0]

print("\n" + "=" * 70)
print("EXAMPLE: highest-risk-scored transaction in the test set")
print("=" * 70)
print(f"Model-assigned fraud probability: {example['PrimaryModelProba']:.4f}")
print(f"Actual label: {'Fraud' if example['Label'] == 1 else 'Legitimate'}")
print(f"Transaction amount: ${example['Amount']:.2f}")

print("\nDone. All outputs saved to /home/claude/credit_card_analysis/")

# ---------------------------------------------------------------------------
# OPTIONAL -- run separately in an environment with internet access for SHAP
# ---------------------------------------------------------------------------
"""
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.sample(5000, random_state=42))
shap.summary_plot(shap_values, X_test.sample(5000, random_state=42), show=False)
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
"""
