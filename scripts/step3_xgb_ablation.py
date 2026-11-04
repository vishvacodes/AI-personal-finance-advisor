"""step3_xgb_ablation.py — Fair XGBoost rerun: 5 variants on 3k char n-gram features.
All artifacts saved to results/xgb_ablation/ — never to models/.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

DATA    = REPO / "data"
RESULTS = REPO / "results"
ABL_DIR = RESULTS / "xgb_ablation"
ABL_DIR.mkdir(exist_ok=True)

SEED = 42

# ── load training data ──────────────────────────────────────────────────────────
print("Loading data...")
split_df = pd.read_csv(DATA / "mitulshah_split.csv")
train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
val_df   = split_df[split_df["split"] == "val"].reset_index(drop=True) \
           if "val" in split_df["split"].values else \
           split_df[split_df["split"] == "test"].reset_index(drop=True).sample(
               frac=0.1, random_state=SEED)
test_df  = split_df[split_df["split"] == "test"].reset_index(drop=True)

noisy_df = pd.read_csv(DATA / "noisy_test_v2.csv")
typo_df  = noisy_df[noisy_df["noise_type"] == "char_swap"].reset_index(drop=True)  # char_swap = closest to character-level typo
indian_df = pd.read_csv(DATA / "own_test_indian.csv")

from src.canonicalize import canonicalize
def preprocess(texts):
    return [canonicalize(t) for t in texts]

print("Canonicalizing train/test...")
X_train_raw = preprocess(train_df["transaction_description"])
X_val_raw   = preprocess(val_df["transaction_description"])
X_test_raw  = preprocess(test_df["transaction_description"])
X_noisy_raw = preprocess(noisy_df["transaction_description"])
X_typo_raw  = preprocess(typo_df["transaction_description"])
X_ind_raw   = preprocess(indian_df["transaction_description"])

# 3k char n-gram vectorizer — same as matched LightGBM config
print("Fitting 3k char n-gram vectorizer on train...")
vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(2, 4),
    max_features=3000, sublinear_tf=True,
)
X_train = vec.fit_transform(X_train_raw)
X_val   = vec.transform(X_val_raw)
X_test  = vec.transform(X_test_raw)
X_noisy = vec.transform(X_noisy_raw)
if len(X_typo_raw) > 0:
    X_typo  = vec.transform(X_typo_raw)
else:
    X_typo = None
X_ind   = vec.transform(X_ind_raw)

le = LabelEncoder()
y_train = le.fit_transform(train_df["category"])
y_val   = le.transform(val_df["category"])
y_test  = le.transform(test_df["category"])
y_noisy = le.transform(noisy_df["category"])
y_typo  = le.transform(typo_df["category"]) if X_typo is not None else None
# Indian test has "Rent" which is not in training classes — keep as raw strings for eval
y_ind_str = indian_df["category"].tolist()   # used directly for string-level eval
CLASSES = le.classes_

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Classes ({len(CLASSES)}): {list(CLASSES)}")

# Load existing matched LightGBM for comparison
lgb_path = REPO / "models" / "matched_lightgbm_3k.joblib"
lgb_model = None
if lgb_path.exists():
    lgb_obj = joblib.load(lgb_path)
    # LightGBM artifact structure may have vectorizer; try predict directly
    try:
        from src.predict import TransactionPredictor
        lgb_pred_obj = TransactionPredictor(model_path=lgb_path)
        lgb_preds_test  = lgb_pred_obj.predict_batch(test_df[["transaction_description"]])["predicted_category"].tolist()
        lgb_preds_noisy = lgb_pred_obj.predict_batch(noisy_df[["transaction_description"]])["predicted_category"].tolist()
        lgb_preds_ind   = lgb_pred_obj.predict_batch(indian_df[["transaction_description"]])["predicted_category"].tolist()
        lgb_acc_clean   = accuracy_score(test_df["category"], lgb_preds_test)
        lgb_f1_clean    = f1_score(test_df["category"], lgb_preds_test, average="macro", zero_division=0)
        lgb_acc_noisy   = accuracy_score(noisy_df["category"], lgb_preds_noisy)
        lgb_f1_noisy    = f1_score(noisy_df["category"], lgb_preds_noisy, average="macro", zero_division=0)
        lgb_acc_ind     = accuracy_score(indian_df["category"], lgb_preds_ind)
        lgb_f1_ind      = f1_score(indian_df["category"], lgb_preds_ind, average="macro", zero_division=0)
        print(f"\nLightGBM 3k reference: clean_acc={lgb_acc_clean:.4f} clean_F1={lgb_f1_clean:.4f} | noisy_acc={lgb_acc_noisy:.4f} noisy_F1={lgb_f1_noisy:.4f} | indian_acc={lgb_acc_ind:.4f} indian_F1={lgb_f1_ind:.4f}")
        lgb_model = "loaded"
    except Exception as e:
        print(f"  LightGBM load error: {e}")

def eval_xgb(model, name, params_str):
    """Evaluate XGBoost model on all test sets and return result dict."""
    results = {"variant": name, "params": params_str}
    eval_sets = [
        ("clean",      X_test,  test_df["category"].tolist()),
        ("noisy_v2",   X_noisy, noisy_df["category"].tolist()),
        ("indian_105", X_ind,   y_ind_str),
    ]
    if X_typo is not None:
        eval_sets.insert(2, ("char_swap", X_typo, typo_df["category"].tolist()))
    for label, X_sp, y_arr in eval_sets:
        y_pred_enc = model.predict(X_sp)
        y_pred_cat = le.inverse_transform(y_pred_enc)
        acc = accuracy_score(y_arr, y_pred_cat)
        mf1 = f1_score(y_arr, y_pred_cat, average="macro", zero_division=0)
        results[f"{label}_acc"]      = round(acc, 4)
        results[f"{label}_macro_F1"] = round(mf1, 4)
    return results

abl_results = []
all_per_class = []

# ── (a) baseline default ───────────────────────────────────────────────────────
print("\n--- Variant (a): default config (sparse input) ---")
params_a = dict(n_estimators=100, max_depth=6, learning_rate=0.3,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=SEED, n_jobs=-1, tree_method="hist")
xgb_a = XGBClassifier(**params_a)
xgb_a.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
joblib.dump(xgb_a, ABL_DIR / "xgb_a_baseline.joblib")
res_a = eval_xgb(xgb_a, "a_baseline", str(params_a))
abl_results.append(res_a)
print(f"  clean_acc={res_a['clean_acc']} macro_F1={res_a['clean_macro_F1']}")

# ── (b) dense input (missing=0) ───────────────────────────────────────────────
print("\n--- Variant (b): dense input (toarray) ---")
params_b = dict(n_estimators=100, max_depth=6, learning_rate=0.3,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=SEED, n_jobs=-1, tree_method="hist")
xgb_b = XGBClassifier(**params_b)
xgb_b.fit(X_train.toarray(), y_train, eval_set=[(X_val.toarray(), y_val)], verbose=False)
joblib.dump(xgb_b, ABL_DIR / "xgb_b_dense.joblib")

def eval_xgb_dense(model, name, params_str):
    results = {"variant": name, "params": params_str}
    eval_sets = [
        ("clean",      X_test,  test_df["category"].tolist()),
        ("noisy_v2",   X_noisy, noisy_df["category"].tolist()),
        ("indian_105", X_ind,   y_ind_str),
    ]
    if X_typo is not None:
        eval_sets.insert(2, ("char_swap", X_typo, typo_df["category"].tolist()))
    for label, X_sp, y_arr in eval_sets:
        y_pred_enc = model.predict(X_sp.toarray())
        y_pred_cat = le.inverse_transform(y_pred_enc)
        acc = accuracy_score(y_arr, y_pred_cat)
        mf1 = f1_score(y_arr, y_pred_cat, average="macro", zero_division=0)
        results[f"{label}_acc"]      = round(acc, 4)
        results[f"{label}_macro_F1"] = round(mf1, 4)
    return results

res_b = eval_xgb_dense(xgb_b, "b_dense", str(params_b))
abl_results.append(res_b)
print(f"  clean_acc={res_b['clean_acc']} macro_F1={res_b['clean_macro_F1']}")

# ── (c) grow_policy=lossguide ─────────────────────────────────────────────────
print("\n--- Variant (c): grow_policy=lossguide ---")
params_c = dict(n_estimators=100, max_depth=6, learning_rate=0.3,
                grow_policy="lossguide", max_leaves=64,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=SEED, n_jobs=-1, tree_method="hist")
xgb_c = XGBClassifier(**params_c)
xgb_c.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
joblib.dump(xgb_c, ABL_DIR / "xgb_c_lossguide.joblib")
res_c = eval_xgb(xgb_c, "c_lossguide", str(params_c))
abl_results.append(res_c)
print(f"  clean_acc={res_c['clean_acc']} macro_F1={res_c['clean_macro_F1']}")

# ── (d) dense + lossguide ─────────────────────────────────────────────────────
print("\n--- Variant (d): dense + lossguide ---")
params_d = dict(n_estimators=100, max_depth=6, learning_rate=0.3,
                grow_policy="lossguide", max_leaves=64,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=SEED, n_jobs=-1, tree_method="hist")
xgb_d = XGBClassifier(**params_d)
xgb_d.fit(X_train.toarray(), y_train, eval_set=[(X_val.toarray(), y_val)], verbose=False)
joblib.dump(xgb_d, ABL_DIR / "xgb_d_dense_lossguide.joblib")
res_d = eval_xgb_dense(xgb_d, "d_dense_lossguide", str(params_d))
abl_results.append(res_d)
print(f"  clean_acc={res_d['clean_acc']} macro_F1={res_d['clean_macro_F1']}")

# ── (e) dense + lossguide + RandomizedSearchCV ────────────────────────────────
print("\n--- Variant (e): RandomizedSearchCV (20 iters, val only) ---")
from scipy.stats import randint, uniform

param_dist = {
    "n_estimators":     randint(100, 400),
    "max_depth":        randint(4, 10),
    "learning_rate":    uniform(0.05, 0.3),
    "min_child_weight": randint(1, 6),
    "subsample":        uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.5, 0.5),
    "max_leaves":       randint(32, 128),
}
base_e = XGBClassifier(
    grow_policy="lossguide",
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=SEED, n_jobs=-1, tree_method="hist",
)
# Use only train+val data; test is never touched
X_tv = np.vstack([X_train.toarray(), X_val.toarray()])
y_tv = np.concatenate([y_train, y_val])
# Cross-val on train only (not test)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
rscv = RandomizedSearchCV(
    base_e, param_dist, n_iter=20, cv=cv,
    scoring="f1_macro", random_state=SEED,
    n_jobs=1, verbose=1, refit=True,
)
rscv.fit(X_train.toarray(), y_train)
best_params_e = rscv.best_params_
print(f"  Best params: {best_params_e}")
xgb_e = rscv.best_estimator_
joblib.dump(xgb_e, ABL_DIR / "xgb_e_tuned.joblib")
res_e = eval_xgb_dense(xgb_e, "e_tuned", str(best_params_e))
abl_results.append(res_e)
print(f"  clean_acc={res_e['clean_acc']} macro_F1={res_e['clean_macro_F1']}")

# ── save ablation results ──────────────────────────────────────────────────────
abl_df = pd.DataFrame(abl_results)

# Add LightGBM reference row
if lgb_model:
    lgb_row = {
        "variant": "LightGBM_3k_ref",
        "params": "matched_lightgbm_3k.joblib (loaded)",
        "clean_acc": round(lgb_acc_clean, 4),
        "clean_macro_F1": round(lgb_f1_clean, 4),
        "noisy_v2_acc": round(lgb_acc_noisy, 4),
        "noisy_v2_macro_F1": round(lgb_f1_noisy, 4),
        "indian_105_acc": round(lgb_acc_ind, 4),
        "indian_105_macro_F1": round(lgb_f1_ind, 4),
    }
    # fill missing
    for col in abl_df.columns:
        if col not in lgb_row:
            lgb_row[col] = None
    abl_df = pd.concat([abl_df, pd.DataFrame([lgb_row])], ignore_index=True)

abl_df.to_csv(RESULTS / "xgb_ablation_results.csv", index=False)
print(f"\nSaved results/xgb_ablation_results.csv")

print("\n=== XGB ABLATION RESULTS ===")
cols_show = ["variant","clean_acc","clean_macro_F1","noisy_v2_acc","noisy_v2_macro_F1","indian_105_acc","indian_105_macro_F1"]
print(abl_df[[c for c in cols_show if c in abl_df.columns]].to_string(index=False))

# ── gap analysis ───────────────────────────────────────────────────────────────
if lgb_model:
    best_xgb_f1 = max(r["clean_macro_F1"] for r in abl_results)
    print(f"\nBest XGBoost variant clean macro_F1: {best_xgb_f1}")
    print(f"LightGBM 3k reference clean macro_F1: {round(lgb_f1_clean,4)}")
    gap = lgb_f1_clean - best_xgb_f1
    print(f"Gap remaining: {gap:.4f}")
    if gap > 0.01:
        print("CONCLUSION: Gap to LightGBM does NOT close with these ablations. EFB was not tested.")
    else:
        print("CONCLUSION: Gap to LightGBM substantially closes with tuning.")

print("\nDONE step3.")
