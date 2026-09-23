"""
Confusion Matrix Confirmation for the Official Final Model
==============================================================
Computes and prints the full 3x3 confusion matrix (Phishing/Spam/Valid)
for the official final model on the combined unified test set
(4,375 emails), and saves it as a labelled PNG image.

USAGE:
    python 40_confusion_matrix_confirmation.py

Run from the project root (same folder you run your other scripts from).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PSYCH_FEATURE_NAMES = [
    "authority", "scarcity_urgency", "commitment_consistency", "liking_rapport",
    "fear", "greed", "sensitive_information_request", "call_to_action",
]
CLASS_ORDER = ["Phishing", "Spam", "Valid"]

RESULTS_DIR = Path("Results")
MODELS_DIR = Path("Models")


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


print("Loading data (unified human + AI-generated TEST set)...")
orig_cleaned = pd.read_csv("Data/Processed/phishfuzzer_cleaned_english.csv")
orig_embeddings = np.load("Data/Processed/phishfuzzer_embeddings_english.npy")
orig_structural = pd.read_csv("Data/Processed/phishfuzzer_structural_features_english.csv")
orig_labels = pd.read_csv("Data/Processed/phishfuzzer_labels_english.csv")
orig_psych = load_psych_scores(Path("Results/psychological_features.jsonl"), "row_id")

test_indices = np.load("Data/test_indices.npy")
y_all = orig_labels["Type"].astype(str)

orig_X_structural_full = orig_structural.drop(columns=["Original_ID", "Type", "has_risky_attachment"]).copy()
orig_X_psych_full = orig_psych[PSYCH_FEATURE_NAMES]

X_test_orig_embeddings = orig_embeddings[test_indices]
y_test_orig = y_all.iloc[test_indices].reset_index(drop=True)
test_struct_orig = orig_X_structural_full.iloc[test_indices].reset_index(drop=True)
test_psych_orig = orig_X_psych_full.iloc[test_indices].reset_index(drop=True)

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
heldout_embeddings = llm_embeddings_full[heldout_mask.to_numpy()][keep]
heldout_struct = struct_block(heldout_mask, llm_structural_full).loc[keep].reset_index(drop=True)
heldout_psych_df = pd.DataFrame({
    feat: heldout_psych_indexed.loc[heldout_subset["row_no"], feat].to_numpy() for feat in PSYCH_FEATURE_NAMES
})
y_test_heldout = heldout_subset["Type"].astype(str).reset_index(drop=True)

scaler = joblib.load(MODELS_DIR / "official_final_scaler.joblib")
model = joblib.load(MODELS_DIR / "official_final_model.joblib")
le = joblib.load(MODELS_DIR / "label_encoder.joblib")

# --- Combine human + AI-generated test sets (the official reported test set) ---
X_combined_embeddings = np.vstack([X_test_orig_embeddings, heldout_embeddings])
struct_combined = pd.concat([test_struct_orig, heldout_struct], ignore_index=True)
psych_combined = pd.concat([test_psych_orig, heldout_psych_df], ignore_index=True)
y_combined = pd.concat([y_test_orig, y_test_heldout], ignore_index=True)

ext_combined = pd.concat([struct_combined.reset_index(drop=True), psych_combined.reset_index(drop=True)], axis=1)
struct_scaled = scaler.transform(ext_combined)
X_combined = np.hstack([X_combined_embeddings, struct_scaled])

y_pred = model.predict(X_combined)
y_pred_labels = le.inverse_transform(y_pred) if not np.issubdtype(np.array(y_pred).dtype, np.str_) else y_pred

cm = confusion_matrix(y_combined, y_pred_labels, labels=CLASS_ORDER)

print("\n" + "=" * 60)
print("CONFUSION MATRIX (rows = true label, columns = predicted label)")
print("=" * 60)
print(f"Order: {CLASS_ORDER}")
print(f"Total test emails: {len(y_combined)}")
print()
header = "              " + "  ".join(f"{c:>10}" for c in CLASS_ORDER)
print(header)
for i, row_label in enumerate(CLASS_ORDER):
    row_str = "  ".join(f"{cm[i, j]:>10}" for j in range(len(CLASS_ORDER)))
    print(f"{row_label:>12}  {row_str}")

print("\n--- Off-diagonal breakdown ---")
for i, true_label in enumerate(CLASS_ORDER):
    for j, pred_label in enumerate(CLASS_ORDER):
        if i != j and cm[i, j] > 0:
            print(f"{true_label} misclassified as {pred_label}: {cm[i, j]}")

spam_idx, valid_idx = CLASS_ORDER.index("Spam"), CLASS_ORDER.index("Valid")
spam_valid_confusion = cm[spam_idx, valid_idx] + cm[valid_idx, spam_idx]
print(f"\nTotal Spam<->Valid confusion: {spam_valid_confusion}")

# Save as labelled image
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=300)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=11)
    plt.tight_layout()
    out_path = RESULTS_DIR / "figures_final" / "fig_confusion_matrix_confirmed.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path)
    plt.savefig(str(out_path).replace(".png", ".pdf"))
    print(f"\nSaved confirmed confusion matrix image to: {out_path}")
except ImportError:
    print("\n(matplotlib not available -- image not saved, but numbers above are confirmed)")

# Save raw numbers to CSV too
cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
cm_df.to_csv(RESULTS_DIR / "confusion_matrix_confirmed.csv")
print(f"Saved raw numbers to: {RESULTS_DIR / 'confusion_matrix_confirmed.csv'}")