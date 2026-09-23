"""
Official Final Experiment — Model Selection on the Unified Dataset
=====================================================================
This script does NOT clean data, remove duplicates, or compute embeddings
itself. It assumes all upstream preprocessing has already been run, and
simply LOADS the resulting files to build the unified (human-authored +
AI-generated) training and test sets, then runs the 3-stage model
selection process (semantic-only -> +structural -> +psychological).

REQUIRED PREREQUISITE SCRIPTS (must be run first, in this order, to
produce the files this script depends on):

  Human-authored data pipeline:
    04_preprocessing.py                    -> phishfuzzer_cleaned_english.csv
                                               (cleaning + duplicate removal)
    05_feature_extraction.py               -> phishfuzzer_embeddings_english.npy
    09_structural_features.py              -> phishfuzzer_structural_features_english.csv
    21_extract_psychological_features.py   -> psychological_features.jsonl

  AI-generated (LLM-rephrased) data pipeline:
    17_prepare_llm_dataset.py              -> phishfuzzer_llm_rephrased_cleaned_english.csv
                                               phishfuzzer_llm_rephrased_embeddings_english.npy
                                               phishfuzzer_llm_rephrased_structural_features_english.csv
    32_extract_psychological_features_rephrased_heldout.py
                                            -> psychological_features_llm_rephrased_heldout.jsonl
    34_extract_psychological_features_augmentation_sample.py
                                            -> psychological_features_augmentation_sample.jsonl
    36_extract_psychological_features_full_train_remaining.py
                                            -> psychological_features_full_train_augmentation.jsonl

  Train/test split (shared by both pipelines, defines the grouping used
  throughout the whole project to prevent data leakage):
    Data/train_indices.npy, Data/test_indices.npy

USAGE (run once per stage/model combination — see project documentation
for the full 7-command sequence):
    python 39_official_final_experiment.py <stage: 1|2|3> <model: svm|logreg|rf|xgb>

OUTPUT:
    Results/official_final_experiment_results.csv  (appends one row per run)
    Models/official_final_model.joblib              (saved only on stage 3)
    Models/official_final_scaler.joblib              (saved only on stage 3)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

PSYCH_FEATURE_NAMES = [
    "authority", "scarcity_urgency", "commitment_consistency", "liking_rapport",
    "fear", "greed", "sensitive_information_request", "call_to_action",
]
CONTINUOUS_FEATURES = [
    "url_count", "url_length_max", "url_length_avg", "max_subdomain_count",
    "attachment_count", "sender_domain_length", "sender_domain_digit_count",
    "subject_length", "body_length", "body_word_count",
    "exclamation_count", "question_mark_count",
]
CLASS_ORDER = ["Phishing", "Spam", "Valid"]
PHISHING_IDX = CLASS_ORDER.index("Phishing")

RESULTS_DIR = Path("Results")
MODELS_DIR = Path("Models")
RESULTS_CSV = RESULTS_DIR / "official_final_experiment_results.csv"


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


def build_everything():
    """Build the unified combined (human + AI-generated) train and test sets,
    for all three feature stages. Returns a dict with everything needed."""
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

    X_train_orig_embeddings = orig_embeddings[train_indices]
    X_test_orig_embeddings = orig_embeddings[test_indices]
    y_train_orig = y_all.iloc[train_indices].reset_index(drop=True)
    y_test_orig = y_all.iloc[test_indices].reset_index(drop=True)

    train_struct_orig = orig_X_structural_full.iloc[train_indices].reset_index(drop=True)
    test_struct_orig = orig_X_structural_full.iloc[test_indices].reset_index(drop=True)
    train_psych_orig = orig_X_psych_full.iloc[train_indices].reset_index(drop=True)
    test_psych_orig = orig_X_psych_full.iloc[test_indices].reset_index(drop=True)

    aug_psych_small = load_psych_scores(Path("Results/psychological_features_augmentation_sample.jsonl"), "row_no")
    aug_psych_full = load_psych_scores(Path("Results/psychological_features_full_train_augmentation.jsonl"), "row_no")

    llm_cleaned = pd.read_csv("Data/Processed/phishfuzzer_llm_rephrased_cleaned_english.csv")
    llm_embeddings_full = np.load("Data/Processed/phishfuzzer_llm_rephrased_embeddings_english.npy")
    llm_structural_full = pd.read_csv("Data/Processed/phishfuzzer_llm_rephrased_structural_features_english.csv")

    train_ids_all = orig_cleaned.iloc[train_indices][["Original_ID", "Type"]]
    train_ids_set = set(train_ids_all["Original_ID"].tolist())
    SAMPLE_PER_CLASS = 70
    parts = []
    for cls in train_ids_all["Type"].unique():
        group = train_ids_all[train_ids_all["Type"] == cls]
        parts.append(group.sample(min(SAMPLE_PER_CLASS, len(group)), random_state=42))
    small_sample_ids = pd.concat(parts)["Original_ID"]

    small_mask = llm_cleaned["Original_ID"].isin(small_sample_ids)
    full_train_rephrased_mask = llm_cleaned["Original_ID"].isin(train_ids_set)
    remaining_mask = full_train_rephrased_mask & ~small_mask

    small_subset = llm_cleaned.loc[small_mask].reset_index(drop=True)
    remaining_subset = llm_cleaned.loc[remaining_mask].reset_index(drop=True)

    aug_embeddings_all = np.vstack([
        llm_embeddings_full[small_mask.to_numpy()],
        llm_embeddings_full[remaining_mask.to_numpy()],
    ])
    aug_struct_all = pd.concat(
        [struct_block(small_mask, llm_structural_full), struct_block(remaining_mask, llm_structural_full)],
        ignore_index=True,
    )
    aug_psych_all = pd.concat(
        [aug_psych_small[PSYCH_FEATURE_NAMES], aug_psych_full[PSYCH_FEATURE_NAMES]], ignore_index=True
    )
    y_aug_all = pd.concat([
        small_subset["Type"].astype(str).reset_index(drop=True),
        remaining_subset["Type"].astype(str).reset_index(drop=True),
    ], ignore_index=True)

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

    # === UNIFIED combined train/test (human-authored + AI-generated treated as ONE dataset) ===
    X_train_embeddings_combined = np.vstack([X_train_orig_embeddings, aug_embeddings_all])
    X_test_embeddings_combined = np.vstack([X_test_orig_embeddings, heldout_embeddings])
    y_train_combined = pd.concat([y_train_orig, y_aug_all], ignore_index=True)
    y_test_combined = pd.concat([y_test_orig, y_test_heldout], ignore_index=True)

    train_struct_combined = pd.concat([train_struct_orig, aug_struct_all], ignore_index=True)
    test_struct_combined = pd.concat([test_struct_orig, heldout_struct], ignore_index=True)

    train_psych_combined = pd.concat([train_psych_orig, aug_psych_all], ignore_index=True)
    test_psych_combined = pd.concat([test_psych_orig, heldout_psych_df], ignore_index=True)

    print(f"UNIFIED combined training set: {len(y_train_combined)} emails "
          f"({len(y_train_orig)} human-authored + {len(y_aug_all)} AI-generated)")
    print(f"UNIFIED combined test set: {len(y_test_combined)} emails "
          f"({len(y_test_orig)} human-authored + {len(y_test_heldout)} AI-generated)")

    return {
        "X_train_embeddings": X_train_embeddings_combined,
        "X_test_embeddings": X_test_embeddings_combined,
        "y_train": y_train_combined,
        "y_test": y_test_combined,
        "train_struct": train_struct_combined,
        "test_struct": test_struct_combined,
        "train_psych": train_psych_combined,
        "test_psych": test_psych_combined,
    }


def clip_fit_scale(train_df, test_df):
    present = [c for c in CONTINUOUS_FEATURES if c in train_df.columns]
    lower = train_df[present].quantile(0.01)
    upper = train_df[present].quantile(0.99)
    train_c = train_df.copy()
    train_c[present] = train_c[present].clip(lower=lower, upper=upper, axis=1)
    test_c = test_df.copy()
    test_c[present] = test_c[present].clip(lower=lower, upper=upper, axis=1)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_c)
    test_scaled = scaler.transform(test_c)
    return train_scaled, test_scaled, scaler


def evaluate_and_save(stage_name, model_name, y_true, y_pred, extra_info=""):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    fp = int(cm[:, PHISHING_IDX].sum() - cm[PHISHING_IDX, PHISHING_IDX])
    fn = int(cm[PHISHING_IDX, :].sum() - cm[PHISHING_IDX, PHISHING_IDX])
    row = {
        "Stage": stage_name,
        "Model": model_name,
        "Test Accuracy": accuracy_score(y_true, y_pred),
        "Test Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Test Phishing Precision": report["Phishing"]["precision"],
        "Test Phishing Recall": report["Phishing"]["recall"],
        "Test Phishing FP": fp,
        "Test Phishing FN": fn,
        "Notes": extra_info,
    }
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        existing = existing[~((existing["Stage"] == stage_name) & (existing["Model"] == model_name))]
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])
    combined.to_csv(RESULTS_CSV, index=False)
    print(f"\n{stage_name} | {model_name}: Test Macro F1={row['Test Macro F1']:.4f} | "
          f"Acc={row['Test Accuracy']:.4f} | FP={fp} | FN={fn}")
    return row


if __name__ == "__main__":
    stage = sys.argv[1]   # "1", "2", or "3"
    model_arg = sys.argv[2]  # "svm", "logreg", "rf", "xgb"

    data = build_everything()

    if stage == "1":
        X_train, X_test = data["X_train_embeddings"], data["X_test_embeddings"]
        stage_name = "1_semantic_only"

    elif stage == "2":
        train_scaled, test_scaled, scaler = clip_fit_scale(data["train_struct"], data["test_struct"])
        X_train = np.hstack([data["X_train_embeddings"], train_scaled])
        X_test = np.hstack([data["X_test_embeddings"], test_scaled])
        stage_name = "2_plus_structural"
        joblib.dump(scaler, MODELS_DIR / f"official_stage2_scaler_{model_arg}.joblib")

    elif stage == "3":
        train_ext = pd.concat([data["train_struct"].reset_index(drop=True), data["train_psych"].reset_index(drop=True)], axis=1)
        test_ext = pd.concat([data["test_struct"].reset_index(drop=True), data["test_psych"].reset_index(drop=True)], axis=1)
        train_scaled, test_scaled, scaler = clip_fit_scale(train_ext, test_ext)
        X_train = np.hstack([data["X_train_embeddings"], train_scaled])
        X_test = np.hstack([data["X_test_embeddings"], test_scaled])
        stage_name = "3_plus_structural_psychological"
        joblib.dump(scaler, MODELS_DIR / "official_final_scaler.joblib")
    else:
        raise ValueError("stage must be 1, 2, or 3")

    y_train, y_test = data["y_train"], data["y_test"]

    if model_arg == "svm":
        model_name = "Linear SVM"
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_arg == "logreg":
        model_name = "Logistic Regression"
        model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_arg == "rf":
        model_name = "Random Forest"
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_arg == "xgb":
        from xgboost import XGBClassifier
        model_name = "XGBoost"
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="mlogloss", n_jobs=-1)
        model.fit(X_train, y_train_enc)
        y_pred = le.inverse_transform(model.predict(X_test))
    else:
        raise ValueError("model must be svm, logreg, rf, xgb")

    row = evaluate_and_save(stage_name, model_name, y_test, y_pred, extra_info=f"features_dim={X_train.shape[1]}")

    if stage == "3":
        joblib.dump(model, MODELS_DIR / "official_final_model.joblib")
        print("Saved as Models/official_final_model.joblib")