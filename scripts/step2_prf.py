"""step2_prf.py — Precision/Recall/F1 for all models x all test sets."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
)

DATA    = REPO / "data"
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────
def load_split(split_name):
    df = pd.read_csv(DATA / "mitulshah_split.csv")
    return df[df["split"] == split_name].reset_index(drop=True)

def metrics_row(model, dataset, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    mp   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mr   = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    mf1  = f1_score       (y_true, y_pred, average="macro", zero_division=0)
    wf1  = f1_score       (y_true, y_pred, average="weighted", zero_division=0)
    return {
        "model": model, "dataset": dataset,
        "accuracy": round(acc, 4), "macro_P": round(mp, 4),
        "macro_R": round(mr, 4),  "macro_F1": round(mf1, 4),
        "weighted_F1": round(wf1, 4),
        "n": len(y_true),
    }

def per_class_rows(model, dataset, y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    rows = []
    for cls in labels:
        d = report.get(cls, {})
        rows.append({
            "model": model, "dataset": dataset, "category": cls,
            "precision": round(d.get("precision", 0), 4),
            "recall":    round(d.get("recall",    0), 4),
            "f1":        round(d.get("f1-score",  0), 4),
            "support":   int(d.get("support",     0)),
        })
    return rows

def save_cm(y_true, y_pred, model_name, dataset_name, outpath):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=[l[:14] for l in labels],
           yticklabels=[l[:14] for l in labels],
           xlabel="Predicted", ylabel="True",
           title=f"{model_name} — {dataset_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontsize=6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=90, bbox_inches="tight")
    plt.close()
    return cm, labels

# ── load test sets ──────────────────────────────────────────────────────────────
print("Loading test sets...")
clean_df  = load_split("test")
noisy_df  = pd.read_csv(DATA / "noisy_test_v2.csv")
typo_df   = noisy_df[noisy_df["noise_type"] == "char_swap"].reset_index(drop=True)  # char_swap = closest to typo
indian_df = pd.read_csv(DATA / "own_test_indian.csv")

# N=96 mask from saved per-row file
per_row   = pd.read_csv(RESULTS / "n96_per_row_predictions.csv")
mask_96   = ~per_row["in_overlapping_9"].astype(bool)
indian_96 = indian_df[mask_96.values].reset_index(drop=True)

print(f"  clean: {len(clean_df)}, noisy_v2: {len(noisy_df)}, typo: {len(typo_df)}, indian_105: {len(indian_df)}, indian_96: {len(indian_96)}")

# ── load saved DistilBERT predictions ──────────────────────────────────────────
db_clean  = pd.read_csv(RESULTS / "distilbert_preds_clean_test.csv")
db_noisy  = pd.read_csv(RESULTS / "distilbert_preds_noisy_v2.csv")
db_indian = pd.read_csv(RESULTS / "distilbert_preds_indian_105.csv")
# DistilBERT typo = filter noisy by noise_type
db_typo   = db_noisy[db_noisy["noise_type"] == "char_swap"].reset_index(drop=True)

# ── live inference for other models ───────────────────────────────────────────
from src.predict import TransactionPredictor

MODEL_FILES = {
    "Candidate_SVM":  "variant_b_candidate.joblib",
    "Legacy_SVM":     "final_model.joblib",
    "LightGBM":       "lightgbm_transaction_classifier.joblib",
    "XGBoost":        "xgboost_transaction_classifier.joblib",
    "MiniLM_LogReg":  "minilm_lr_transaction_classifier.joblib",
}

TEST_SETS = {
    "clean":       (clean_df,  "transaction_description", "category"),
    "noisy_v2":    (noisy_df,  "transaction_description", "category"),
    "typo":        (typo_df,   "transaction_description", "category"),
    "indian_105":  (indian_df, "transaction_description", "category"),
    "indian_96":   (indian_96, "transaction_description", "category"),
}

all_preds = {}   # (model, dataset) -> (y_true, y_pred)

# DistilBERT from saved preds
def align_db_preds(db_df, test_df, desc_col="transaction_description"):
    """Use saved preds directly — aligned by row order."""
    return db_df["true_label"].tolist(), db_df["pred_label"].tolist()

def align_db_preds_96():
    # N=96 subset: use n96_per_row_predictions col pred_DistilBERT
    pr = per_row[mask_96.values]
    return pr["category"].tolist(), pr["pred_DistilBERT"].tolist()

all_preds[("DistilBERT", "clean")]      = align_db_preds(db_clean,  clean_df)
all_preds[("DistilBERT", "noisy_v2")]   = align_db_preds(db_noisy,  noisy_df)
all_preds[("DistilBERT", "typo")]       = align_db_preds(db_typo,   typo_df)
all_preds[("DistilBERT", "indian_105")] = align_db_preds(db_indian, indian_df)
all_preds[("DistilBERT", "indian_96")]  = align_db_preds_96()

# SVM/LGB/XGB/MiniLM from n96_per_row for Indian, live for clean/noisy/typo
def run_inference(model_name, model_file, test_df, desc_col, cat_col):
    p = REPO / "models" / model_file
    if not p.exists():
        print(f"  [MISSING] {model_file}")
        return None, None
    pred_obj = TransactionPredictor(model_path=p)
    result   = pred_obj.predict_batch(test_df[[desc_col]].rename(columns={desc_col: "transaction_description"}))
    return test_df[cat_col].tolist(), result["predicted_category"].tolist()

# Pre-populate Indian preds from n96_per_row for models that have it
col_map = {
    "Candidate_SVM": "pred_Candidate_Linear_SVM_20k",
    "LightGBM":      "pred_LightGBM",
    "MiniLM_LogReg": "pred_MiniLM_LogReg",
}
for mname, col in col_map.items():
    all_preds[(mname, "indian_105")] = (per_row["category"].tolist(), per_row[col].tolist())
    all_preds[(mname, "indian_96")]  = (per_row[mask_96.values]["category"].tolist(),
                                        per_row[mask_96.values][col].tolist())

for mname, mfile in MODEL_FILES.items():
    for dsname, (df, desc_col, cat_col) in TEST_SETS.items():
        if (mname, dsname) in all_preds:
            continue  # already have it
        if len(df) == 0:
            print(f"  Skipping {mname}/{dsname}: empty dataset")
            continue
        print(f"  Inference: {mname} on {dsname} (N={len(df)})...")
        y_true, y_pred = run_inference(mname, mfile, df, desc_col, cat_col)
        if y_true is not None:
            all_preds[(mname, dsname)] = (y_true, y_pred)

print(f"\nTotal (model, dataset) pairs with predictions: {len(all_preds)}")

# ── summary metrics ─────────────────────────────────────────────────────────────
summary_rows = []
per_class_all = []
for (mname, dsname), (y_true, y_pred) in sorted(all_preds.items()):
    summary_rows.append(metrics_row(mname, dsname, y_true, y_pred))
    per_class_all.extend(per_class_rows(mname, dsname, y_true, y_pred))

summary_df = pd.DataFrame(summary_rows)
pc_df      = pd.DataFrame(per_class_all)

summary_df.to_csv(RESULTS / "prf_summary.csv", index=False)
pc_df.to_csv(RESULTS / "prf_per_class.csv", index=False)
print(f"Saved prf_summary.csv ({len(summary_df)} rows)")
print(f"Saved prf_per_class.csv ({len(pc_df)} rows)")

# ── confusion matrices ──────────────────────────────────────────────────────────
cm_pairs = [
    ("Candidate_SVM", "indian_105"),
    ("Candidate_SVM", "noisy_v2"),
    ("DistilBERT",    "indian_105"),
    ("DistilBERT",    "noisy_v2"),
    ("LightGBM",      "indian_105"),
]
for mname, dsname in cm_pairs:
    key = (mname, dsname)
    if key not in all_preds:
        continue
    y_true, y_pred = all_preds[key]
    fname = f"confusion_matrix_{mname.lower().replace(' ','_')}_{dsname}.png"
    cm, labels = save_cm(y_true, y_pred, mname, dsname, RESULTS / fname)
    print(f"Saved {fname}")

# ── print summary table ─────────────────────────────────────────────────────────
print("\n=== PRF SUMMARY TABLE ===")
print(summary_df.to_string(index=False))

# ── weakest classes by recall per model ────────────────────────────────────────
print("\n=== 5 WEAKEST CLASSES BY RECALL (Indian N=105, model-only) ===")
for mname in list(MODEL_FILES.keys()) + ["DistilBERT"]:
    sub = pc_df[(pc_df["model"]==mname) & (pc_df["dataset"]=="indian_105") & (pc_df["support"]>0)]
    if sub.empty:
        continue
    worst = sub.nsmallest(5, "recall")[["category","recall","f1","support"]]
    print(f"\n  {mname}:")
    print(worst.to_string(index=False))

# ── top confusions for SVM and DistilBERT ──────────────────────────────────────
print("\n=== TOP 5 CONFUSIONS (indian_105) ===")
for mname in ["Candidate_SVM", "DistilBERT"]:
    key = (mname, "indian_105")
    if key not in all_preds:
        continue
    y_true, y_pred = all_preds[key]
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # off-diagonal
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i,j] > 0:
                confusions.append((labels[i], labels[j], int(cm[i,j])))
    confusions.sort(key=lambda x: -x[2])
    print(f"\n  {mname}:")
    for true_cls, pred_cls, cnt in confusions[:5]:
        print(f"    True={true_cls!r:32} → Pred={pred_cls!r:32}  count={cnt}")

# ── Rent row analysis ───────────────────────────────────────────────────────────
print("\n=== RENT ROWS ANALYSIS (indian_105) ===")
for mname in ["Candidate_SVM", "DistilBERT", "LightGBM"]:
    key = (mname, "indian_105")
    if key not in all_preds:
        continue
    y_true, y_pred = all_preds[key]
    rent_idx = [i for i,y in enumerate(y_true) if y == "Rent"]
    if not rent_idx:
        print(f"  {mname}: no Rent rows in true labels")
        continue
    rent_preds = [y_pred[i] for i in rent_idx]
    print(f"  {mname}: {len(rent_idx)} Rent rows → preds: {dict(pd.Series(rent_preds).value_counts())}")

# ── Corrected Macro-F1 check ────────────────────────────────────────────────────
print("\n=== CORRECTED MACRO-F1 vs REPORTED (Indian N=105) ===")
print("Reported in doc: SVM 0.9213 (suspicious: > accuracy 0.8762)")
for mname in ["Candidate_SVM", "DistilBERT"]:
    key = (mname, "indian_105")
    if key not in all_preds:
        continue
    y_true, y_pred = all_preds[key]
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"  {mname}: acc={acc:.4f}, macro_F1={mf1:.4f} (model-only, no rent rule)")

print("\nDONE step2.")
