"""
Binary Ablation: Phishing vs Valid Only (Spam Excluded)
=========================================================
Reuses the exact same unified data-loading pipeline as
33_official_final_experiment.py (same features, same train/test split,
same preprocessing), but filters out all Spam-labelled rows from both
the training and test sets before training. This tests the supervisor's
suggestion of evaluating Phishing/Valid separately from Spam, to see
whether removing the three-way classification difficulty improves
Phishing-vs-Valid discrimination.

Trains XGBoost only, on the full semantic + structural + psychological
feature set (the official final configuration), for direct comparability
with the official 3-class result (93.58% Macro F1).

USAGE:
    python 43_binary_phishing_vs_valid.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# --- Import build_everything() and clip_fit_scale() from the official script ---
SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "official_final_experiment", SCRIPT_DIR / "33_official_final_experiment.py"
)
official = importlib.util.module_from_spec(spec)
sys.modules["official_final_experiment"] = official
# Prevent the imported module's __main__ block from running
official.__name__ = "official_final_experiment"
spec.loader.exec_module(official)

print("\n" + "=" * 70)
print("Loading the same unified dataset used by the official experiment...")
print("=" * 70)
data = official.build_everything()

# --- Build the full (semantic + structural + psychological) feature set,
#     identical to Stage 3 of the official experiment ---
train_ext = pd.concat(
    [data["train_struct"].reset_index(drop=True), data["train_psych"].reset_index(drop=True)], axis=1
)
test_ext = pd.concat(
    [data["test_struct"].reset_index(drop=True), data["test_psych"].reset_index(drop=True)], axis=1
)
train_scaled, test_scaled, _ = official.clip_fit_scale(train_ext, test_ext)
X_train_full = np.hstack([data["X_train_embeddings"], train_scaled])
X_test_full = np.hstack([data["X_test_embeddings"], test_scaled])
y_train_full = data["y_train"].reset_index(drop=True)
y_test_full = data["y_test"].reset_index(drop=True)

print(f"\nFull (3-class) training set: {len(y_train_full)} emails")
print(f"Full (3-class) test set: {len(y_test_full)} emails")
print(f"Class distribution (test): {y_test_full.value_counts().to_dict()}")

# --- Filter out Spam from both train and test ---
train_mask = y_train_full != "Spam"
test_mask = y_test_full != "Spam"

X_train_bin = X_train_full[train_mask.to_numpy()]
X_test_bin = X_test_full[test_mask.to_numpy()]
y_train_bin = y_train_full[train_mask].reset_index(drop=True)
y_test_bin = y_test_full[test_mask].reset_index(drop=True)

print(f"\nBinary (Phishing vs Valid) training set: {len(y_train_bin)} emails")
print(f"Binary (Phishing vs Valid) test set: {len(y_test_bin)} emails")
print(f"Class distribution (test): {y_test_bin.value_counts().to_dict()}")

# --- Train XGBoost (same hyperparameters as the official final model) ---
print("\nTraining XGBoost on Phishing-vs-Valid only...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_bin)
model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric="logloss", n_jobs=-1,
)
model.fit(X_train_bin, y_train_enc)
y_pred_bin = le.inverse_transform(model.predict(X_test_bin))

# --- Evaluate ---
acc = accuracy_score(y_test_bin, y_pred_bin)
macro_f1 = f1_score(y_test_bin, y_pred_bin, average="macro")
cm = confusion_matrix(y_test_bin, y_pred_bin, labels=["Phishing", "Valid"])
report = classification_report(y_test_bin, y_pred_bin, output_dict=True, zero_division=0)

fp = int(cm[1, 0])  # Valid predicted as Phishing
fn = int(cm[0, 1])  # Phishing predicted as Valid

print("\n" + "=" * 70)
print("RESULTS: Binary Phishing-vs-Valid (Spam excluded)")
print("=" * 70)
print(f"Accuracy:  {acc:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")
print(f"Phishing Precision: {report['Phishing']['precision']:.4f}")
print(f"Phishing Recall:    {report['Phishing']['recall']:.4f}")
print(f"False Positives (Valid->Phishing): {fp}")
print(f"False Negatives (Phishing->Valid): {fn}")
print(f"\nConfusion matrix (rows=true, cols=pred, order=[Phishing, Valid]):")
print(cm)

print("\n" + "=" * 70)
print("COMPARISON: Official 3-class result (from official_final_experiment_results.csv)")
print("=" * 70)
results_csv = Path("Results") / "official_final_experiment_results.csv"
if results_csv.exists():
    official_results = pd.read_csv(results_csv)
    stage3_xgb = official_results[
        (official_results["Stage"] == "3_plus_structural_psychological")
        & (official_results["Model"] == "XGBoost")
    ]
    if len(stage3_xgb) > 0:
        row = stage3_xgb.iloc[0]
        print(f"3-class Accuracy: {row['Test Accuracy']:.4f}")
        print(f"3-class Macro F1: {row['Test Macro F1']:.4f}")
        print(f"3-class Phishing FP: {row['Test Phishing FP']}")
        print(f"3-class Phishing FN: {row['Test Phishing FN']}")
        print(f"\n--- DELTA (binary - 3-class) ---")
        print(f"Accuracy delta: {acc - row['Test Accuracy']:+.4f}")
        print(f"Macro F1 delta: {macro_f1 - row['Test Macro F1']:+.4f}")

# Save results
out_path = Path("Results") / "binary_phishing_vs_valid_results.csv"
pd.DataFrame([{
    "Experiment": "Binary Phishing vs Valid (Spam excluded)",
    "Model": "XGBoost",
    "Accuracy": acc,
    "Macro F1": macro_f1,
    "Phishing Precision": report["Phishing"]["precision"],
    "Phishing Recall": report["Phishing"]["recall"],
    "FP (Valid as Phishing)": fp,
    "FN (Phishing as Valid)": fn,
    "N_train": len(y_train_bin),
    "N_test": len(y_test_bin),
}]).to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")