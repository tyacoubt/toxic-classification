"""
error_analysis.py
Systematic error analysis comparing Majority Baseline, NBOW+LR, and DistilBERT.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CATEGORY_DEFINITIONS = {
    "sarcasm":            "Toxic in intent but no overtly toxic surface words. Irony or mock-politeness.",
    "non_toxic_profanity":"Profanity used casually or affectionately, not directed at anyone.",
    "ambiguous":          "Could be toxic or non-toxic without additional context.",
    "discusses_hate":     "Describes or quotes hateful content without being hateful itself.",
    "implicit_toxicity":  "Toxic through implication, condescension, or dog-whistles. No explicit slurs.",
    "other":              "Does not fit the above. Use sparingly.",
}


# ---------------------------------------------------------------------------
# Merge all three models
# ---------------------------------------------------------------------------

def merge_predictions(
    test_path:  str = "data/test_split.csv",
    nbow_path:  str = "results/test_nbow_lr_preds.csv",
    bert_path:  str = "results/test_distilbert_preds.csv",
) -> pd.DataFrame:
    """
    Merge ground-truth labels with predictions from both models.
    Majority baseline (always predicts 0) is added as a third column.
    Adds error type columns for each model: 'correct', 'false_positive', 'false_negative'.
    """
    test = pd.read_csv(test_path)
    nbow = pd.read_csv(nbow_path)[["id", "nbow_lr_pred", "nbow_lr_prob"]]
    bert = pd.read_csv(bert_path)[["id", "distilbert_pred", "distilbert_prob"]]

    df = test.merge(nbow, on="id", how="left").merge(bert, on="id", how="left")

    # Majority baseline always predicts non-toxic
    df["majority_pred"] = 0

    def _etype(label, pred):
        if label == pred: return "correct"
        if label == 1:    return "false_negative"
        return "false_positive"

    df["majority_error"] = df.apply(lambda r: _etype(r["label"], r["majority_pred"]),    axis=1)
    df["nbow_error"]     = df.apply(lambda r: _etype(r["label"], r["nbow_lr_pred"]),      axis=1)
    df["bert_error"]     = df.apply(lambda r: _etype(r["label"], r["distilbert_pred"]),   axis=1)
    df["all_wrong"]      = (df["majority_error"] != "correct") & \
                           (df["nbow_error"] != "correct") & \
                           (df["bert_error"] != "correct")

    for model, col in [
        ("Majority Baseline", "majority_error"),
        ("NBOW + LR",         "nbow_error"),
        ("DistilBERT",        "bert_error"),
    ]:
        counts = df[col].value_counts()
        print(f"\n{model}:")
        for k, v in counts.items():
            print(f"  {k}: {v:,}")

    print(f"\nAll three models wrong on same example: {df['all_wrong'].sum():,}")
    return df


def error_overlap_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show how many examples each model gets right/wrong and where they overlap.
    Useful for understanding whether DistilBERT's gains are on top of NBOW's
    or on a completely different set of examples.
    """
    rows = []
    total = len(df)

    # Per-model correct counts
    for model, col in [("Majority", "majority_error"), ("NBOW+LR", "nbow_error"), ("DistilBERT", "bert_error")]:
        correct = (df[col] == "correct").sum()
        rows.append({"comparison": f"{model} correct", "n": correct, "pct": f"{correct/total*100:.1f}%"})

    # Pairwise: one right, other wrong
    combos = [
        ("Only NBOW correct",      (df["nbow_error"]=="correct") & (df["bert_error"]!="correct")),
        ("Only BERT correct",      (df["bert_error"]=="correct") & (df["nbow_error"]!="correct")),
        ("Both NBOW+BERT correct", (df["nbow_error"]=="correct") & (df["bert_error"]=="correct")),
        ("Both NBOW+BERT wrong",   (df["nbow_error"]!="correct") & (df["bert_error"]!="correct")),
        ("All three wrong",        df["all_wrong"]),
    ]
    for label, mask in combos:
        n = mask.sum()
        rows.append({"comparison": label, "n": n, "pct": f"{n/total*100:.1f}%"})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Annotation file preparation
# ---------------------------------------------------------------------------

def prepare_annotation_files(
    df: pd.DataFrame,
    n_samples: int = 50,
    random_state: int = 42,
) -> None:
    """
    Save sampled FP/FN CSVs for NBOW+LR and DistilBERT with a blank 'category' column.
    Majority baseline is excluded — it has no false positives and its false negatives
    are every toxic comment, which is not interesting to annotate individually.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    specs = [
        ("nbow", "nbow_error", "nbow_lr_pred",    "nbow_lr_prob"),
        ("bert", "bert_error", "distilbert_pred", "distilbert_prob"),
    ]

    for tag, error_col, pred_col, prob_col in specs:
        for etype in ["false_positive", "false_negative"]:
            subset = df[df[error_col] == etype]
            sample = subset.sample(n=min(n_samples, len(subset)), random_state=random_state)
            out = sample[["id", "comment_text", "label", pred_col, prob_col]].copy()
            out["category"] = ""
            out["notes"]    = ""
            path = RESULTS_DIR / f"errors_{tag}_{etype[:2]}.csv"
            out.to_csv(path, index=False)
            print(f"Saved {len(out)} rows → {path}")

    print("\nCategory keys:")
    for key, defn in CATEGORY_DEFINITIONS.items():
        print(f"  {key:<22} {defn[:80]}")


# ---------------------------------------------------------------------------
# Annotated file loading and validation
# ---------------------------------------------------------------------------

def load_annotated(path: str) -> pd.DataFrame:
    """Load an annotated CSV and validate category values."""
    df = pd.read_csv(path)
    valid = set(CATEGORY_DEFINITIONS.keys())
    unannotated = df["category"].eq("").sum()
    if unannotated:
        print(f"WARNING: {unannotated} unannotated rows in {path}")
    invalid = df[~df["category"].isin(valid) & df["category"].ne("")]
    if len(invalid):
        raise ValueError(
            f"Invalid categories in {path}: {invalid['category'].unique()}\n"
            f"Valid: {sorted(valid)}"
        )
    return df


def compute_agreement(path_a: str, path_b: str, category_col: str = "category") -> float:
    """Cohen's kappa between two annotators. >0.6 = substantial."""
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    merged = df_a[["id", category_col]].merge(
        df_b[["id", category_col]], on="id", suffixes=("_a", "_b")
    )
    both = merged[merged[f"{category_col}_a"].ne("") & merged[f"{category_col}_b"].ne("")]
    if len(both) == 0:
        print("No rows annotated by both.")
        return float("nan")
    kappa = cohen_kappa_score(both[f"{category_col}_a"], both[f"{category_col}_b"])
    agree_pct = (both[f"{category_col}_a"] == both[f"{category_col}_b"]).mean() * 100
    label = "substantial" if kappa >= 0.6 else "moderate" if kappa >= 0.4 else "poor"
    print(f"Rows compared: {len(both)} | Raw agreement: {agree_pct:.1f}% | Kappa: {kappa:.4f} ({label})")
    return kappa


# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------

def category_breakdown(annotated_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute % of each error category per model/error-type.
    annotated_dfs keys should be descriptive, e.g. 'NBOW FN', 'BERT FN'.
    """
    rows = []
    for label, df in annotated_dfs.items():
        labelled = df[df["category"].ne("")]
        for cat, pct in labelled["category"].value_counts(normalize=True).mul(100).items():
            rows.append({"label": label, "category": cat, "pct": round(pct, 1), "n": len(labelled)})
    return pd.DataFrame(rows)


def plot_category_breakdown(
    breakdown: pd.DataFrame,
    title: str = "Error category breakdown",
    save: bool = True,
) -> None:
    """Side-by-side horizontal bar charts for each entry in breakdown."""
    labels     = breakdown["label"].unique()
    categories = list(CATEGORY_DEFINITIONS.keys())
    colors     = ["#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#EF9F27", "#888780"]

    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), sharey=False)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        subset = breakdown[breakdown["label"] == label].set_index("category")["pct"]
        subset = subset.reindex(categories, fill_value=0)
        n = breakdown.loc[breakdown["label"] == label, "n"].iloc[0] if len(breakdown[breakdown["label"] == label]) else 0

        bars = ax.barh(subset.index, subset.values,
                       color=[colors[i % len(colors)] for i in range(len(subset))])
        ax.set_xlim(0, 110)
        ax.set_xlabel("% of errors")
        ax.set_title(f"{label} (n={n})", fontsize=12)
        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.text(w + 1, bar.get_y() + bar.get_height() / 2,
                        f"{w:.1f}%", va="center", fontsize=9)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig.savefig(RESULTS_DIR / f"{title.lower().replace(' ','_')}.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison_bars(df: pd.DataFrame, save: bool = True) -> None:
    """
    Bar chart comparing error type counts across all three models side by side.
    Gives an at-a-glance view of how much each model improves over the baseline.
    """
    models = [
        ("Majority Baseline", "majority_error"),
        ("NBOW + LR",         "nbow_error"),
        ("DistilBERT",        "bert_error"),
    ]
    etypes = ["false_negative", "false_positive", "correct"]
    colors = {"correct": "#1D9E75", "false_negative": "#D85A30", "false_positive": "#EF9F27"}

    data = {}
    for mname, mcol in models:
        counts = df[mcol].value_counts()
        data[mname] = {e: counts.get(e, 0) for e in etypes}

    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, etype in enumerate(etypes):
        vals = [data[m][etype] for m, _ in models]
        bars = ax.bar(x + i * width, vals, width, label=etype.replace("_", " ").title(),
                      color=colors[etype], alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 50,
                        f"{h:,}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m for m, _ in models])
    ax.set_ylabel("Number of examples")
    ax.set_title("Prediction outcomes by model — test set")
    ax.legend()
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig.savefig(RESULTS_DIR / "model_outcome_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_results_table(results_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all JSON result files and return a sorted comparison table."""
    results_dir = results_dir or RESULTS_DIR
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {results_dir}")
    rows = [json.loads(f.read_text()) for f in json_files]
    df = pd.DataFrame(rows)
    cols = ["model", "f1_toxic", "precision_toxic", "recall_toxic", "f1_macro", "f1_weighted"]
    return df[[c for c in cols if c in df.columns]].sort_values("f1_toxic", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("=== Category definitions ===\n")
    for key, defn in CATEGORY_DEFINITIONS.items():
        print(f"{key}:\n  {defn}\n")
    try:
        df = merge_predictions()
        prepare_annotation_files(df)
    except FileNotFoundError as e:
        print(f"\nCould not load predictions: {e}")
        print("Run notebooks 04 and 05 first.")
