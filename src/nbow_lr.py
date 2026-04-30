"""
nbow_lr.py
----------
NBOW (TF-IDF) + Logistic Regression classifier for toxic comment detection.

"NBOW" stands for Neural Bag of Words — here it means we represent each
comment as a weighted bag of its words (TF-IDF), with no word order or
context. Logistic Regression then learns which words and bigrams are most
predictive of toxicity.

"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

from src.data_utils import RANDOM_SEED
from src.evaluation import evaluate, extract_errors
from src.preprocessing import clean_text_ablation

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Vectorizer
# ---------------------------------------------------------------------------

def build_vectorizer(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Build an unfitted TF-IDF vectorizer configured for toxicity detection.

    Parameter choices and rationale
    --------------------------------
    max_features=50_000
        Caps vocabulary size. The full Jigsaw vocabulary is ~300k tokens
        but the long tail (very rare words) adds noise without signal.
        50k covers the vast majority of meaningful terms.

    ngram_range=(1, 2)
        Include both single words (unigrams) and two-word phrases (bigrams).
        Bigrams capture phrases like "go die", "shut up", "you idiot" that
        are more predictive than any single word alone.
        (1, 3) adds trigrams but increases feature space with diminishing returns.

    min_df=3
        Ignore terms appearing in fewer than 3 documents. Removes typos,
        user-specific tokens, and rare profanity variants that won't
        generalise to the test set.

    max_df=0.95
        Ignore terms appearing in more than 95% of documents. Removes
        near-universal stopwords that TF-IDF would upweight incorrectly.
        Complements min_df at the other end of the frequency spectrum.

    sublinear_tf=True
        Replace raw term frequency tf with log(1 + tf). This prevents
        documents that repeat a word 100 times from dominating the
        representation — a comment that says "idiot" once is not 100x
        less toxic than one that says it 100 times.

    strip_accents="unicode"
        Normalise accented characters (e.g. "naïve" → "naive"). Reduces
        vocabulary size and handles copy-paste from other languages.

    token_pattern=r"\\w{1,}"
        Match any word character sequence of length >= 1. The sklearn
        default excludes single characters, which would drop tokens like
        "i", "a", "u" that can be meaningful in context (e.g. "f u").

    Args:
        max_features: Maximum vocabulary size.
        ngram_range:  (min_n, max_n) for n-gram extraction.
        min_df:       Minimum document frequency for a term to be included.
        max_df:       Maximum document frequency fraction for inclusion.
        sublinear_tf: Whether to apply log(1 + tf) scaling.

    Returns:
        Unfitted TfidfVectorizer ready to call .fit_transform() on.

    Example:
        >>> vec = build_vectorizer()
        >>> X_train = vec.fit_transform(train['clean_text'])
        >>> X_test  = vec.transform(test['clean_text'])   # NOT fit_transform
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
    )


def vectorize_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    text_col: str = "clean_text",
    vectorizer: Optional[TfidfVectorizer] = None,
    **vectorizer_kwargs,
) -> tuple[spmatrix, spmatrix, spmatrix, TfidfVectorizer]:
    """
    Fit a TF-IDF vectorizer on train and transform all three splits.

    CRITICAL: The vectorizer is ONLY fitted on the training split.
    Fitting on val or test would leak test distribution information
    into the vocabulary and IDF weights, inflating results.

    Args:
        train:             Training split DataFrame.
        val:               Validation split DataFrame.
        test:              Test split DataFrame.
        text_col:          Column containing cleaned text (default 'clean_text').
                           Must exist in all three DataFrames.
        vectorizer:        Pre-built TfidfVectorizer to use. If None, calls
                           build_vectorizer(**vectorizer_kwargs).
        **vectorizer_kwargs: Passed to build_vectorizer() if vectorizer is None.

    Returns:
        (X_train, X_val, X_test, fitted_vectorizer)
        Sparse matrices suitable for LogisticRegression.

    Raises:
        ValueError: If text_col is missing from any split.

    Example:
        >>> X_train, X_val, X_test, vec = vectorize_splits(train, val, test)
        >>> print(X_train.shape)   # (127656, 50000)
    """
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        if text_col not in split.columns:
            raise ValueError(
                f"Column '{text_col}' not found in {split_name} split. "
                f"Available columns: {list(split.columns)}\n"
                "Did you run 02_preprocessing.ipynb to create the clean_text column?"
            )

    if vectorizer is None:
        vectorizer = build_vectorizer(**vectorizer_kwargs)

    print("Fitting TF-IDF vectorizer on training split only ...")
    X_train = vectorizer.fit_transform(train[text_col].fillna(""))
    X_val   = vectorizer.transform(val[text_col].fillna(""))
    X_test  = vectorizer.transform(test[text_col].fillna(""))

    vocab_size = len(vectorizer.vocabulary_)
    print(f"  Vocabulary size : {vocab_size:,}")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_val shape     : {X_val.shape}")
    print(f"  X_test shape    : {X_test.shape}")
    print(f"  Matrix density  : {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.5f}")

    return X_train, X_val, X_test, vectorizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lr(
    X_train: spmatrix,
    y_train: pd.Series | np.ndarray,
    C_values: Optional[list[float]] = None,
    cv_folds: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> tuple[LogisticRegression, GridSearchCV]:
    """
    Train Logistic Regression with cross-validated regularisation strength (C).

    Design decisions
    ----------------
    class_weight='balanced'
        Automatically upweights the minority (toxic) class by the inverse of
        its frequency. Equivalent to passing class_weight_dict(train) from
        data_utils.py. Without this, the model learns to predict non-toxic
        for almost everything because that minimises plain cross-entropy on
        the 9:1 imbalanced dataset.

    scoring='f1'
        Cross-validation selects the C value that maximises F1 on the toxic
        class, NOT accuracy. This is critical — with class imbalance, a model
        optimised for accuracy would just predict the majority class.

    solver='lbfgs'
        Works well for small-to-medium feature sets. For very large sparse
        matrices (>100k features), 'saga' is faster.

    max_iter=1000
        The default (100) is often not enough for convergence on a 50k feature
        space. 1000 is safe; the solver stops early if it converges.

    random_state=RANDOM_SEED
        Makes the optimisation deterministic (lbfgs is iterative and can
        have small numerical differences run-to-run without a seed).

    Args:
        X_train:   Sparse TF-IDF matrix, shape (n_train, n_features).
        y_train:   Binary labels, shape (n_train,).
        C_values:  Regularisation strengths to search over.
                   Smaller C = stronger regularisation = simpler model.
                   Default: [0.01, 0.1, 1.0, 10.0, 100.0]
        cv_folds:  Number of stratified cross-validation folds (default 5).
        n_jobs:    Parallel workers (-1 = all cores). CV is embarrassingly
                   parallel so this speeds up training significantly.
        verbose:   Verbosity level passed to GridSearchCV (default 1).

    Returns:
        (best_estimator, grid_search_object)
        best_estimator is already refitted on the full X_train.
        grid_search_object contains cv_results_ for plotting.

    Example:
        >>> best_lr, grid = train_lr(X_train, train['label'])
        >>> val_preds = best_lr.predict(X_val)
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    grid = GridSearchCV(
        estimator=lr,
        param_grid={"C": C_values},
        cv=cv,
        scoring="f1",
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,      # refit on full X_train with best C before returning
        return_train_score=True,
    )

    print(f"\nRunning {cv_folds}-fold cross-validation over C={C_values} ...")
    grid.fit(X_train, y_train)

    best_C  = grid.best_params_["C"]
    best_f1 = grid.best_score_

    print(f"\nGrid search complete.")
    print(f"  Best C     : {best_C}")
    print(f"  Best val F1: {best_f1:.4f}")

    cv_df = (
        pd.DataFrame(grid.cv_results_)
        [["param_C", "mean_test_score", "std_test_score", "mean_train_score"]]
        .rename(columns={
            "param_C":          "C",
            "mean_test_score":  "mean_val_f1",
            "std_test_score":   "std_val_f1",
            "mean_train_score": "mean_train_f1",
        })
    )
    print("\nFull CV results:")
    print(cv_df.to_string(index=False))

    return grid.best_estimator_, grid


def plot_cv_results(grid: GridSearchCV, save: bool = True) -> None:
    """
    Plot cross-validation F1 scores across C values.

    Shows mean ± 1 std across folds for both train and validation.
    A large train-val gap at high C indicates overfitting.

    Args:
        grid: Fitted GridSearchCV object from train_lr().
        save: Save figure to results/nbow_lr_cv_results.png.

    Example:
        >>> _, grid = train_lr(X_train, y_train)
        >>> plot_cv_results(grid)
    """
    results = pd.DataFrame(grid.cv_results_)
    C_values = results["param_C"].astype(float).values

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.semilogx(C_values, results["mean_train_score"], "o-",
                label="Train F1", color="#1D9E75", linewidth=2, markersize=6)
    ax.fill_between(
        C_values,
        results["mean_train_score"] - results["std_train_score"],
        results["mean_train_score"] + results["std_train_score"],
        alpha=0.15, color="#1D9E75"
    )

    ax.semilogx(C_values, results["mean_test_score"], "o-",
                label="Val F1 (toxic class)", color="#378ADD", linewidth=2, markersize=6)
    ax.fill_between(
        C_values,
        results["mean_test_score"] - results["std_test_score"],
        results["mean_test_score"] + results["std_test_score"],
        alpha=0.15, color="#378ADD"
    )

    best_C = grid.best_params_["C"]
    ax.axvline(best_C, linestyle="--", color="#D85A30", alpha=0.8,
               label=f"Best C = {best_C}")

    ax.set_xlabel("C  (regularisation strength, log scale)", fontsize=11)
    ax.set_ylabel("F1 score (toxic class)", fontsize=11)
    ax.set_title("NBOW + LR — cross-validation results", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = RESULTS_DIR / "nbow_lr_cv_results.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"CV results plot saved to {path}")

    plt.show()


# ---------------------------------------------------------------------------
# Feature inspection
# ---------------------------------------------------------------------------

def get_top_features(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    n: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the top N features most predictive of toxic and non-toxic.

    The logistic regression coefficient for a feature is its log-odds
    contribution. A large positive coefficient means the feature strongly
    predicts toxic (class 1). A large negative coefficient means it
    strongly predicts non-toxic (class 0).

    Args:
        vectorizer:  Fitted TfidfVectorizer.
        classifier:  Fitted LogisticRegression.
        n:           Number of top features per class.

    Returns:
        (toxic_df, nontoxic_df) — DataFrames with columns ['feature', 'coefficient']
        sorted by absolute coefficient value descending.

    Example:
        >>> toxic_df, nontoxic_df = get_top_features(vec, lr, n=20)
        >>> print(toxic_df.head(10))
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]

    sorted_idx    = np.argsort(coefs)
    toxic_idx     = sorted_idx[-n:][::-1]
    nontoxic_idx  = sorted_idx[:n]

    toxic_df = pd.DataFrame({
        "feature":     feature_names[toxic_idx],
        "coefficient": coefs[toxic_idx].round(4),
    })
    nontoxic_df = pd.DataFrame({
        "feature":     feature_names[nontoxic_idx],
        "coefficient": coefs[nontoxic_idx].round(4),
    })

    return toxic_df, nontoxic_df


def print_top_features(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    n: int = 25,
) -> None:
    """
    Print the top N features for each class in a readable format.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        classifier: Fitted LogisticRegression.
        n:          Number of features per class (default 25).
    """
    toxic_df, nontoxic_df = get_top_features(vectorizer, classifier, n=n)

    print(f"\n{'='*55}")
    print(f"  Top {n} features → TOXIC  (positive coefficients)")
    print(f"{'='*55}")
    for _, row in toxic_df.iterrows():
        print(f"  {row['feature']:<30}  {row['coefficient']:+.4f}")

    print(f"\n{'='*55}")
    print(f"  Top {n} features → NON-TOXIC  (negative coefficients)")
    print(f"{'='*55}")
    for _, row in nontoxic_df.iterrows():
        print(f"  {row['feature']:<30}  {row['coefficient']:+.4f}")


def plot_top_features(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    n: int = 20,
    save: bool = True,
) -> None:
    """
    Plot horizontal bar charts of the top N features per class.

    Side-by-side chart: left panel = most toxic predictors,
    right panel = most non-toxic predictors. Include this figure in
    the write-up to show what lexical signals the model relies on.
    This also directly motivates error analysis — if "you" is a top
    toxic predictor, you'd expect false positives on "I love you".

    Args:
        vectorizer: Fitted TfidfVectorizer.
        classifier: Fitted LogisticRegression.
        n:          Number of features per panel (default 20).
        save:       Save to results/nbow_lr_top_features.png.
    """
    toxic_df, nontoxic_df = get_top_features(vectorizer, classifier, n=n)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n * 0.35)))

    for ax, df, title, color in [
        (axes[0], toxic_df,    f"Top {n} → TOXIC",     "#D85A30"),
        (axes[1], nontoxic_df, f"Top {n} → NON-TOXIC", "#378ADD"),
    ]:
        bars = ax.barh(
            df["feature"][::-1],
            df["coefficient"].abs()[::-1],
            color=color, alpha=0.85, edgecolor="white", linewidth=0.3
        )
        ax.set_title(title, fontsize=12, fontweight="medium")
        ax.set_xlabel("|Coefficient|", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", fontsize=7.5, color="#444")

    plt.suptitle("NBOW + LR — top predictive features by class", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = RESULTS_DIR / "nbow_lr_top_features.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Feature plot saved to {path}")

    plt.show()


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(
    train: pd.DataFrame,
    val: pd.DataFrame,
    label_col: str = "label",
) -> pd.DataFrame:
    
    """
    Test how each vectorizer/preprocessing choice affects validation F1.
    Uses the best C from a baseline grid search for all configs.
    """

    print("Running baseline grid search for best C ...")
    baseline_vec = build_vectorizer()
    X_tr = baseline_vec.fit_transform(train["clean_text"].fillna(""))
    X_va = baseline_vec.transform(val["clean_text"].fillna(""))
    baseline_lr, grid = train_lr(X_tr, train[label_col], verbose=0)
    best_C = grid.best_params_["C"]
    print(f"Best C: {best_C}\n")

    y_tr = train[label_col].values
    y_va = val[label_col].values
    raw_tr = train["comment_text"].fillna("")
    raw_va = val["comment_text"].fillna("")
    clean_tr = train["clean_text"].fillna("")
    clean_va = val["clean_text"].fillna("")

    def _eval(X_tr_, X_va_, class_weight="balanced"):
        lr = LogisticRegression(
            C=best_C, class_weight=class_weight,
            max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED,
        )
        lr.fit(X_tr_, y_tr)
        return round(f1_score(y_va, lr.predict(X_va_), pos_label=1, zero_division=0), 4)

    def _vec(texts_tr, texts_va, token_pattern=r"\w{1,}", **kwargs):
        v = build_vectorizer(**kwargs)
        v.token_pattern = token_pattern
        return v.fit_transform(texts_tr), v.transform(texts_va)

    def _vec_raw(texts_tr, texts_va, token_pattern=r"\w{1,}", **kwargs):
        """Vectorize without the preprocessing.py cleaning pipeline."""
        v = TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2), min_df=3, max_df=0.95,
            sublinear_tf=True, strip_accents="unicode",
            token_pattern=token_pattern, **kwargs,
        )
        return v.fit_transform(texts_tr), v.transform(texts_va)

    configs = []

    def add(label, X_tr_, X_va_, **kwargs):
        configs.append({"config": label, "val_f1": _eval(X_tr_, X_va_, **kwargs)})

    # Baseline
    add("Baseline (bigrams, sublinear, balanced, 50k, clean text)", X_tr, X_va)

    # N-gram range
    add("Unigrams only (1,1)",          *_vec(clean_tr, clean_va, ngram_range=(1, 1)))
    add("Trigrams added (1,3)",          *_vec(clean_tr, clean_va, ngram_range=(1, 3)))

    # TF weighting
    add("No sublinear TF",               *_vec(clean_tr, clean_va, sublinear_tf=False))

    # Class weighting
    add("No class weighting",            X_tr, X_va, class_weight=None)

    # Vocabulary size
    add("Smaller vocab (20k)",           *_vec(clean_tr, clean_va, max_features=20_000))
    add("Larger vocab (100k)",           *_vec(clean_tr, clean_va, max_features=100_000))

    # min_df
    add("Stricter min_df (10)",          *_vec(clean_tr, clean_va, min_df=10))
    add("Looser min_df (2)",             *_vec(clean_tr, clean_va, min_df=2))

    clean_keep_punc_tr = train["comment_text"].fillna("").apply(
    lambda t: clean_text_ablation(t, remove_punctuation=False)
    )
    
    clean_keep_punc_va = val["comment_text"].fillna("").apply(
        lambda t: clean_text_ablation(t, remove_punctuation=False)
    )

    add("Clean text, keep punctuation",
        *_vec(clean_keep_punc_tr, clean_keep_punc_va, token_pattern=r"\S+"))

    df = pd.DataFrame(configs).sort_values("val_f1", ascending=False).reset_index(drop=True)
    baseline_f1 = df.loc[df["config"].str.startswith("Baseline"), "val_f1"].values[0]
    df["delta"] = (df["val_f1"] - baseline_f1).round(4)

    return df


def plot_ablation(ablation_df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot the ablation study results as a horizontal bar chart.

    Args:
        ablation_df: DataFrame from run_ablation() with 'config' and 'val_f1' columns.
        save:        Save to results/nbow_lr_ablation.png.
    """
    df = ablation_df.sort_values("val_f1")
    baseline_f1 = df.loc[df["config"].str.startswith("Baseline"), "val_f1"].values[0]

    colors = [
        "#1D9E75" if v >= baseline_f1 else "#D85A30"
        for v in df["val_f1"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.5)))
    bars = ax.barh(df["config"], df["val_f1"], color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.3)
    ax.axvline(baseline_f1, linestyle="--", color="#444", alpha=0.6, label=f"Baseline = {baseline_f1:.4f}")
    ax.set_xlabel("Validation F1 (toxic class)", fontsize=11)
    ax.set_title("NBOW + LR — ablation study", fontsize=12)
    ax.legend(fontsize=9)

    for bar, val in zip(bars, df["val_f1"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlim(0, df["val_f1"].max() * 1.12)
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = RESULTS_DIR / "nbow_lr_ablation.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Ablation plot saved to {path}")

    plt.show()


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> tuple[float, pd.DataFrame]:
    """
    Args:
        y_true:     Ground-truth binary labels.
        y_prob:     Predicted probabilities for the positive (toxic) class.
        metric:     Metric to maximise: 'f1', 'precision', or 'recall'.
        thresholds: Array of thresholds to evaluate. Default: 0.05 to 0.95
                    in steps of 0.01.

    Returns:
        (best_threshold, results_df)
        results_df has columns ['threshold', 'f1', 'precision', 'recall'].

    """
    from sklearn.metrics import precision_score, recall_score

    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    rows = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 3),
            "f1":        round(f1_score(y_true, preds, pos_label=1, zero_division=0), 4),
            "precision": round(precision_score(y_true, preds, pos_label=1, zero_division=0), 4),
            "recall":    round(recall_score(y_true, preds, pos_label=1, zero_division=0), 4),
        })

    results_df  = pd.DataFrame(rows)
    best_idx    = results_df[metric].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]

    print(f"\nThreshold tuning (optimising {metric} on validation set):")
    print(f"  Default threshold (0.5): "
          f"F1={results_df.loc[results_df['threshold']==0.5, 'f1'].values[0]:.4f}")
    print(f"  Best threshold ({best_threshold}):   "
          f"F1={results_df.loc[best_idx, 'f1']:.4f}  "
          f"P={results_df.loc[best_idx, 'precision']:.4f}  "
          f"R={results_df.loc[best_idx, 'recall']:.4f}")

    return best_threshold, results_df


def plot_threshold_curve(
    thresh_df: pd.DataFrame,
    best_threshold: float,
    save: bool = True,
) -> None:
    """
    Plot F1, precision, and recall as a function of threshold.

    Args:
        thresh_df:       DataFrame from tune_threshold().
        best_threshold:  Best threshold value to mark on the plot.
        save:            Save to results/nbow_lr_threshold_curve.png.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for metric, color in [("f1", "#378ADD"), ("precision", "#1D9E75"), ("recall", "#D85A30")]:
        ax.plot(thresh_df["threshold"], thresh_df[metric],
                label=metric.capitalize(), color=color, linewidth=2)

    ax.axvline(best_threshold, linestyle="--", color="#444", alpha=0.7,
               label=f"Best threshold = {best_threshold}")
    ax.axvline(0.5, linestyle=":", color="#888", alpha=0.6, label="Default (0.5)")

    ax.set_xlabel("Decision threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("NBOW + LR — threshold vs F1 / Precision / Recall", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = RESULTS_DIR / "nbow_lr_threshold_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Threshold curve saved to {path}")

    plt.show()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_model(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    models_dir: Path = MODELS_DIR,
) -> None:
    """
    Pickle the fitted vectorizer and classifier to models/.

    Both objects are needed to make predictions on new text:
        1. vectorizer.transform(new_texts)  → sparse matrix
        2. classifier.predict(matrix)       → binary labels

    Args:
        vectorizer:  Fitted TfidfVectorizer.
        classifier:  Fitted LogisticRegression.
        models_dir:  Directory to save into (default models/).

    Example:
        >>> save_model(vec, lr)
        # Later:
        >>> vec, lr = load_model()
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    vec_path = models_dir / "tfidf_vectorizer.pkl"
    lr_path  = models_dir / "logistic_regression.pkl"

    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(lr_path, "wb") as f:
        pickle.dump(classifier, f)

    print(f"Saved vectorizer → {vec_path}")
    print(f"Saved classifier → {lr_path}")


def load_model(
    models_dir: Path = MODELS_DIR,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """
    Load a previously saved vectorizer and classifier.

    Args:
        models_dir: Directory containing tfidf_vectorizer.pkl and
                    logistic_regression.pkl (default models/).

    Returns:
        (vectorizer, classifier) — both fitted and ready for .transform()
        and .predict() respectively.

    Raises:
        FileNotFoundError: If the pkl files don't exist.
    """
    models_dir = Path(models_dir)
    vec_path = models_dir / "tfidf_vectorizer.pkl"
    lr_path  = models_dir / "logistic_regression.pkl"

    for path in [vec_path, lr_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run run_nbow_lr() or train_lr() first to save the model."
            )

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(lr_path, "rb") as f:
        classifier = pickle.load(f)

    return vectorizer, classifier


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_nbow_lr(
    train_path: str = "data/train_split.csv",
    val_path:   str = "data/val_split.csv",
    test_path:  str = "data/test_split.csv",
    run_ablation_study: bool = True,
    tune_decision_threshold: bool = True,
) -> dict:
    """
    Full NBOW + LR pipeline: load → vectorize → train → evaluate → save.

    Args:
        train_path:               Path to train_split.csv.
        val_path:                 Path to val_split.csv.
        test_path:                Path to test_split.csv.
        run_ablation_study:       Whether to run and save the ablation study.
        tune_decision_threshold:  Whether to tune the decision threshold on val.

    Returns:
        Test set results dict (same format as evaluate() returns).

    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load data
    print("Loading splits ...")
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)
    print(f"  Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")

    # 2. Vectorize
    print("\nVectorizing ...")
    X_train, X_val, X_test, vectorizer = vectorize_splits(train, val, test)

    # 3. Train with grid search
    best_lr, grid = train_lr(X_train, train["label"])

    # 4. Plot CV results
    plot_cv_results(grid)

    # 5. Validation predictions
    val_preds = best_lr.predict(X_val)
    val_probs = best_lr.predict_proba(X_val)[:, 1]
    print("\n--- Validation set ---")
    evaluate(val["label"], val_preds, model_name="NBOW + LR (val)", y_prob=val_probs)

    # 6. Optional threshold tuning on val
    decision_threshold = 0.5
    if tune_decision_threshold:
        print("\n--- Threshold tuning on val ---")
        decision_threshold, thresh_df = tune_threshold(val["label"], val_probs)
        plot_threshold_curve(thresh_df, decision_threshold)

    # 7. Test predictions
    test_probs = best_lr.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= decision_threshold).astype(int)
    print(f"\n--- Test set (threshold = {decision_threshold}) ---")
    test_results = evaluate(
        test["label"], test_preds,
        model_name="NBOW + LR",
        y_prob=test_probs,
    )

    # 8. Feature inspection
    print_top_features(vectorizer, best_lr)
    plot_top_features(vectorizer, best_lr)

    # 9. Optional ablation
    if run_ablation_study:
        print("\n--- Ablation study ---")
        ablation_df = run_ablation(train, val)
        print(ablation_df.to_string(index=False))
        plot_ablation(ablation_df)
        ablation_df.to_csv(RESULTS_DIR / "nbow_lr_ablation.csv", index=False)

    # 10. Save predictions with full test DataFrame
    test_out = test.copy()
    test_out["nbow_lr_pred"] = test_preds
    test_out["nbow_lr_prob"] = test_probs.round(6)
    test_out.to_csv(RESULTS_DIR / "test_nbow_lr_preds.csv", index=False)
    print(f"\nPredictions saved → {RESULTS_DIR / 'test_nbow_lr_preds.csv'}")

    # 11. Save results JSON
    test_results["decision_threshold"] = decision_threshold
    with open(RESULTS_DIR / "nbow_lr_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved     → {RESULTS_DIR / 'nbow_lr_results.json'}")

    # 12. Save model
    save_model(vectorizer, best_lr)

    # 13. Save error samples for notebook 06
    errors = extract_errors(test_out, pred_col="nbow_lr_pred")
    errors["false_positives"].to_csv(RESULTS_DIR / "errors_nbow_fp.csv", index=False)
    errors["false_negatives"].to_csv(RESULTS_DIR / "errors_nbow_fn.csv", index=False)
    print(f"Error samples saved → {RESULTS_DIR}/errors_nbow_*.csv")

    return test_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_split.csv"
    val_path   = sys.argv[2] if len(sys.argv) > 2 else "data/val_split.csv"
    test_path  = sys.argv[3] if len(sys.argv) > 3 else "data/test_split.csv"

    results = run_nbow_lr(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
    )
    print(f"\nFinal test F1 (toxic class): {results['f1_toxic']:.4f}")
