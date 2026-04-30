"""
evaluation.py
Shared evaluation utilities used by all three model notebooks.
All models call evaluate() so metrics are computed identically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    y_prob: Optional[np.ndarray] = None,
    save: bool = True,
) -> dict:
    """Print classification report + confusion matrix. Returns results dict for JSON logging."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
    print(classification_report(y_true, y_pred, target_names=["non-toxic", "toxic"], digits=4))

    _plot_confusion_matrix(y_true, y_pred, model_name, save)

    results = {
        "model":            model_name,
        "f1_toxic":         round(f1_score(y_true, y_pred, pos_label=1), 4),
        "precision_toxic":  round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_toxic":     round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_nontoxic":      round(f1_score(y_true, y_pred, pos_label=0), 4),
        "f1_macro":         round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted":      round(f1_score(y_true, y_pred, average="weighted"), 4),
    }

    if y_prob is not None:
        results["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        _plot_roc(y_true, y_prob, model_name, save)

    return results


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save: bool,
) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
        xticklabels=["non-toxic", "toxic"],
        yticklabels=["non-toxic", "toxic"],
        linewidths=0.5,
    )
    ax.set_title(f"{model_name} — confusion matrix")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    if save:
        slug = model_name.lower().replace(" ", "_").replace("+", "plus")
        fig.savefig(RESULTS_DIR / f"{slug}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


def _plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save: bool,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_name} — ROC curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        slug = model_name.lower().replace(" ", "_").replace("+", "plus")
        fig.savefig(RESULTS_DIR / f"{slug}_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_all_models(results_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all JSON result files and plot a side-by-side metric comparison."""
    results_dir = results_dir or RESULTS_DIR
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON result files in {results_dir}")

    df = pd.DataFrame([json.loads(f.read_text()) for f in json_files])
    df = df.sort_values("f1_toxic", ascending=True)

    metrics = ["f1_toxic", "precision_toxic", "recall_toxic", "f1_macro"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4), sharey=True)
    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.barh(df["model"], df[metric], color=color, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Score")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", fontsize=9)

    fig.suptitle("Model comparison — test set", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    cols = ["model", "f1_toxic", "precision_toxic", "recall_toxic", "f1_macro", "f1_weighted"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def extract_errors(
    df: pd.DataFrame,
    true_col: str = "label",
    pred_col: str = "pred",
    n_samples: int = 50,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Sample false positives and false negatives with a blank 'category' column for annotation."""
    fp = df[(df[true_col] == 0) & (df[pred_col] == 1)]
    fn = df[(df[true_col] == 1) & (df[pred_col] == 0)]

    fp_sample = fp.sample(n=min(n_samples, len(fp)), random_state=random_state).copy()
    fn_sample = fn.sample(n=min(n_samples, len(fn)), random_state=random_state).copy()

    fp_sample["category"] = ""
    fn_sample["category"] = ""

    print(f"False positives: {len(fp):,} total  |  {len(fp_sample)} sampled")
    print(f"False negatives: {len(fn):,} total  |  {len(fn_sample)} sampled")

    return {"false_positives": fp_sample, "false_negatives": fn_sample}


def interannotator_agreement(
    ann_a: pd.DataFrame,
    ann_b: pd.DataFrame,
    category_col: str = "category",
) -> float:
    """Cohen's kappa between two annotators. >0.6 = substantial agreement."""
    if len(ann_a) != len(ann_b):
        raise ValueError("Both DataFrames must have the same number of rows.")
    kappa = cohen_kappa_score(ann_a[category_col], ann_b[category_col])
    label = "substantial" if kappa >= 0.6 else "moderate" if kappa >= 0.4 else "poor"
    print(f"Cohen's kappa: {kappa:.4f}  ({label})")
    return kappa


def plot_error_categories(
    labeled_errors: dict[str, pd.DataFrame],
    model_name: str,
    save: bool = True,
) -> None:
    """Horizontal bar chart of error category breakdown for one model."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (error_type, df_err) in zip(axes, labeled_errors.items()):
        if df_err["category"].eq("").all():
            ax.text(0.5, 0.5, "Not yet annotated", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(f"{model_name} — {error_type.replace('_', ' ')}")
            continue

        counts = df_err["category"].value_counts(normalize=True).mul(100).sort_values()
        bars = ax.barh(counts.index, counts.values, color="#378ADD", alpha=0.85)
        ax.set_xlim(0, 110)
        ax.set_xlabel("% of errors")
        ax.set_title(f"{model_name} — {error_type.replace('_', ' ')} (n={len(df_err)})")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", fontsize=9)

    fig.suptitle(f"Error categories: {model_name}", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        slug = model_name.lower().replace(" ", "_").replace("+", "plus")
        fig.savefig(RESULTS_DIR / f"{slug}_error_categories.png", dpi=150, bbox_inches="tight")
    plt.show()