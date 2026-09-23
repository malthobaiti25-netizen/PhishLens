"""
Robustness Evaluation of the FINAL CANDIDATE ARCHITECTURE
=============================================================
Unlike Development_Archive/20_heldout_llm_robustness.py (which tests an
earlier Stage-2, semantic+structural-only SVM model), this script trains
the ACTUAL final model architecture -- XGBoost on the full 412-dimensional
semantic + structural + psychological feature set, identical to the
official final model -- but restricted to HUMAN-AUTHORED training data
only. It is then evaluated on both the human-authored test set and the
held-out AI-generated test set, for a methodologically consistent
comparison against the official unified-trained model
(93.58% Macro F1 on the same combined test set).

This directly and rigorously tests the claim in the Methodology chapter's
"Motivating Evaluation" subsection using the same architecture as the
final model, rather than an earlier development-stage model.

USAGE:
    python 39_human_only_final_architecture_robustness.py

REQUIRES the same prerequisite files as 39_official_final_experiment.py
(see that script's docstring for the full list).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

PSYCH_FEATURE_NAMES = [
    "authority", "scarcity_urgency", "commitment_consistency", "liking_rapport",
    "fear", "greed", "sensitive_information_request", "call_to_action",
]
CLASS_ORDER = ["Phishing", "Spam", "Valid"]
RESULTS_DIR = Path("Results")


def load_psych_scores(path: Path, id_key: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records).sort_values(id_key).reset_index(drop=True)
    scores = pd.DataFrame({feat: df[feat].apply(lambda d: d["score"]) for feat in PSYCH_FEATURE_NAMES})
    scores[id_key] = df[id_key].values
    return scores


def struct_block(mask, llm_structural_full):
    raw = llm_structural_full.loc[mask].reset_index(drop=True)
    return raw.drop(columns=["No.", "Original_ID", "Type", "has_risky_attachment"], errors="ignore")


def clip_fit_scale(train_df, test_df):
    combined = pd.concat([train_df, test_df], ignore_index=True)
    lower = combined.quantile(0.01)
    upper = combined.quantile(0.99)
    train_clipped = train_df.clip(lower=lower, upper=upper, axis=1)
    test_clipped = test_df.clip(lower=lower, upper=upper, axis=1)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_clipped)
    test_scaled = scaler.transform(test_clipped)
    return train_scaled, test_scaled, scaler


print("=" * 70)
print("Loading HUMAN-AUTHORED training data only...")
print("=" * 70)

orig_cleaned = pd.read_csv("Data/Processed/phishfuzzer_cleaned_english.csv")
orig_embeddings = np.load("Data/Processed/phishfuzzer_embeddings_english.npy")
orig_structural = pd.read_csv("Data/Processed/phishfuzzer_structural_features_english.csv")
orig_labels = pd.read_csv("Data/Processed/phishfuzzer_labels_english.csv")
orig_psych = load_psych_scores(Path("Results/psychological_features.jsonl"), "row_id")

train_indices = np.load("Data/train_indices.npy")
test_indices = np.load("Data/test_indices.npy")
y_all = orig_labels["Type"].astype(str)

orig_X_structural_full = orig_structural.drop(columns=["Original_ID", "Type", "has_risky_attachment"]).copy()
orig_X_psych_full = orig_psych[PSYCH_FEATURE_NAMES]

# --- HUMAN-ONLY training set (no AI-generated augmentation) ---
X_train_embeddings = orig_embeddings[train_indices]
y_train = y_all.iloc[train_indices].reset_index(drop=True)
train_struct = orig_X_structural_full.iloc[train_indices].reset_index(drop=True)
train_psych = orig_X_psych_full.iloc[train_indices].reset_index(drop=True)

# --- Human-authored TEST set ---
X_test_human_embeddings = orig_embeddings[test_indices]
y_test_human = y_all.iloc[test_indices].reset_index(drop=True)
test_struct_human = orig_X_structural_full.iloc[test_indices].reset_index(drop=True)
test_psych_human = orig_X_psych_full.iloc[test_indices].reset_index(drop=True)

print(f"Human-only training set: {len(y_train)} emails")
print(f"Human-authored test set: {len(y_test_human)} emails")

# --- AI-generated held-out TEST set (variants of test-set originals only) ---
print("\nLoading AI-generated held-out test data...")
llm_cleaned = pd.read_csv("Data/Processed/phishfuzzer_llm_rephrased_cleaned_english.csv")
llm_embeddings_full = np.load("Data/Processed/phishfuzzer_llm_rephrased_embeddings_english.npy")
llm_structural_full = pd.read_csv("Data/Processed/phishfuzzer_llm_rephrased_structural_features_english.csv")

heldout_ids = set(orig_cleaned.iloc[test_indices]["Original_ID"].tolist())
heldout_mask = llm_cleaned["Original_ID"].isin(heldout_ids)
heldout_subset = llm_cleaned.loc[heldout_mask].reset_index(drop=True)
heldout_subset["row_no"] = heldout_subset.index
heldout_psych = load_psych_scores(Path("Results/psychological_features_llm_rephrased_heldout.jsonl"), "row_no")
heldout_psych_indexed = heldout_psych.set_index("row_no")
avail = set(heldout_psych["row_no"].tolist())
keep = heldout_subset["row_no"].isin(avail).to_numpy()
heldout_subset = heldout_subset.loc[keep].reset_index(drop=True)
X_test_ai_embeddings = llm_embeddings_full[heldout_mask.to_numpy()][keep]
test_struct_ai = struct_block(heldout_mask, llm_structural_full).loc[keep].reset_index(drop=True)
test_psych_ai = pd.DataFrame({
    feat: heldout_psych_indexed.loc[heldout_subset["row_no"], feat].to_numpy() for feat in PSYCH_FEATURE_NAMES
})
y_test_ai = heldout_subset["Type"].astype(str).reset_index(drop=True)

print(f"AI-generated held-out test set: {len(y_test_ai)} emails")

# --- Build full 412-dim feature vectors (semantic + structural + psychological) ---
print("\nBuilding full feature vectors (semantic + structural + psychological)...")
train_ext = pd.concat([train_struct.reset_index(drop=True), train_psych.reset_index(drop=True)], axis=1)
test_ext_human = pd.concat([test_struct_human.reset_index(drop=True), test_psych_human.reset_index(drop=True)], axis=1)
test_ext_ai = pd.concat([test_struct_ai.reset_index(drop=True), test_psych_ai.reset_index(drop=True)], axis=1)

# Fit scaler on human-only training data; apply to both test sets
combined_test_ext = pd.concat([test_ext_human, test_ext_ai], ignore_index=True)
lower = train_ext.quantile(0.01)
upper = train_ext.quantile(0.99)
train_ext_clipped = train_ext.clip(lower=lower, upper=upper, axis=1)
test_ext_human_clipped = test_ext_human.clip(lower=lower, upper=upper, axis=1)
test_ext_ai_clipped = test_ext_ai.clip(lower=lower, upper=upper, axis=1)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_ext_clipped)
test_human_scaled = scaler.transform(test_ext_human_clipped)
test_ai_scaled = scaler.transform(test_ext_ai_clipped)

X_train_full = np.hstack([X_train_embeddings, train_scaled])
X_test_human_full = np.hstack([X_test_human_embeddings, test_human_scaled])
X_test_ai_full = np.hstack([X_test_ai_embeddings, test_ai_scaled])

print(f"Feature dimensionality: {X_train_full.shape[1]} (matches official final model)")

# --- Train XGBoost (identical hyperparameters to the official final model) ---
print("\nTraining XGBoost (human-authored data only)...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric="mlogloss", n_jobs=-1,
)
model.fit(X_train_full, y_train_enc)


def evaluate(X, y_true, label):
    y_pred = le.inverse_transform(model.predict(X))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'=' * 70}\n{label} (n={len(y_true)})\n{'=' * 70}")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, labels=CLASS_ORDER, digits=4))
    return {"segment": label, "n": len(y_true), "accuracy": acc, "macro_f1": f1}


print("\n" + "=" * 70)
print("RESULTS: Human-only-trained FINAL ARCHITECTURE (XGBoost, full 412 features)")
print("=" * 70)

results = []
results.append(evaluate(X_test_human_full, y_test_human, "Human-authored test set"))
results.append(evaluate(X_test_ai_full, y_test_ai, "AI-generated held-out test set"))

pd.DataFrame(results).to_csv(RESULTS_DIR / "human_only_final_architecture_robustness.csv", index=False)
print(f"\nSaved to {RESULTS_DIR / 'human_only_final_architecture_robustness.csv'}")

print("\n" + "=" * 70)
print("COMPARISON: Official unified-trained model (from official_final_experiment_results.csv)")
print("=" * 70)
official_csv = RESULTS_DIR / "official_final_experiment_results.csv"
if official_csv.exists():
    official = pd.read_csv(official_csv)
    stage3 = official[
        (official["Stage"] == "3_plus_structural_psychological") & (official["Model"] == "XGBoost")
    ]
    if len(stage3) > 0:
        row = stage3.iloc[0]
        print(f"Unified-trained model, combined test set: Macro F1 = {row['Test Macro F1']:.4f}")
        print(f"Human-only-trained model, AI-generated test: Macro F1 = {results[1]['macro_f1']:.4f}")
        print(f"Delta: {row['Test Macro F1'] - results[1]['macro_f1']:+.4f}")