"""step1_class_balance.py — class distribution across all datasets."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA   = REPO / "data"
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

CATEGORIES = [
    "Charity & Donations","Entertainment & Recreation","Financial Services",
    "Food & Dining","Government & Legal","Healthcare & Medical","Income",
    "Rent","Shopping & Retail","Transportation","Utilities & Services",
]

def class_stats(df, col="category"):
    counts = df[col].value_counts()
    total = len(df)
    out = {}
    for c in sorted(counts.index):
        n = counts.get(c, 0)
        out[c] = {"count": int(n), "pct": round(100*n/total, 2)}
    return out, total

def imbalance_ratio(stats):
    counts = [v["count"] for v in stats.values()]
    if not counts or min(counts) == 0:
        return None
    return round(max(counts) / min(counts), 2)

def load_split(split_name):
    df = pd.read_csv(DATA / "mitulshah_split.csv")
    return df[df["split"] == split_name].reset_index(drop=True)

datasets = {}
datasets["train"]        = load_split("train")
datasets["clean_test"]   = load_split("test")
datasets["noisy_v2"]     = pd.read_csv(DATA / "noisy_test_v2.csv")
# typo subset: rows where transaction_description != original  — not separate file
# use noisy_v2 as proxy; mark typo from noise_ablation_matrix if available
typo_path = RESULTS / "per_noise_type_evaluation.csv"
# We'll label "noisy_v2" as the comprehensive noise set
datasets["indian_105"]   = pd.read_csv(DATA / "own_test_indian.csv")
datasets["indian_seed"]  = pd.read_csv(DATA / "indian_merchants_train_seed.csv")

print("=== STEP 1: CLASS BALANCE ===\n")
rows = []
for ds_name, df in datasets.items():
    col = "category"
    stats, total = class_stats(df, col)
    ratio = imbalance_ratio(stats)
    small_classes = [c for c,v in stats.items() if v["count"] < 10]
    print(f"Dataset: {ds_name}  (N={total})  imbalance_ratio={ratio}")
    for c,v in stats.items():
        print(f"  {c:<30} {v['count']:>6}  {v['pct']:>6.2f}%")
        rows.append({
            "dataset": ds_name, "category": c,
            "count": v["count"], "pct": v["pct"],
            "total_n": total,
        })
    if small_classes:
        print(f"  ** Classes with <10 rows: {small_classes}")
    print()

dist_df = pd.DataFrame(rows)
out_csv = RESULTS / "class_distribution.csv"
dist_df.to_csv(out_csv, index=False)
print(f"Saved {out_csv}")

# --- Bar chart ---
ds_names = list(datasets.keys())
cats = sorted(dist_df["category"].unique())
n_ds = len(ds_names)
fig, axes = plt.subplots(1, n_ds, figsize=(5*n_ds, 6), sharey=False)
colors = plt.cm.tab10.colors

for ax, ds_name in zip(axes, ds_names):
    sub = dist_df[dist_df["dataset"]==ds_name].set_index("category")
    counts = [sub.loc[c,"count"] if c in sub.index else 0 for c in cats]
    short_cats = [c.split(" & ")[0][:12] for c in cats]
    bars = ax.barh(short_cats, counts, color=colors[:len(cats)])
    ax.set_title(ds_name, fontsize=9, fontweight="bold")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    for bar, val in zip(bars, counts):
        if val > 0:
            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                    str(val), va="center", fontsize=7)

fig.suptitle("Class Distribution by Dataset", fontsize=12, fontweight="bold")
plt.tight_layout()
out_png = RESULTS / "class_distribution.png"
plt.savefig(out_png, dpi=100, bbox_inches="tight")
print(f"Saved {out_png}")

# Summary table
print("\n=== IMBALANCE SUMMARY ===")
print(f"{'Dataset':<20} {'N':>8} {'Imbalance Ratio':>18} {'<10 classes'}")
for ds_name, df in datasets.items():
    stats, total = class_stats(df)
    ratio = imbalance_ratio(stats)
    small = [c for c,v in stats.items() if v["count"] < 10]
    print(f"{ds_name:<20} {total:>8} {str(ratio):>18} {small}")

print("\nDONE step1.")
