"""
data_utils.py
Shared data loading, splitting, and class weight utilities.
All notebooks import from here — never duplicate this logic.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SUBTYPES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RANDOM_SEED = 42


def load_and_label(path: str | Path) -> pd.DataFrame:
    """Load raw Jigsaw train.csv and collapse subtypes into a single binary label."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            "Download from: https://www.kaggle.com/competitions/"
            "jigsaw-toxic-comment-classification-challenge/data"
        )

    df = pd.read_csv(path)
    _validate(df)
    df["label"] = df[SUBTYPES].max(axis=1).astype(int)
    return df[["id", "comment_text", "label"]].copy()


def _validate(df: pd.DataFrame) -> None:
    """Raise if the raw CSV is missing expected columns or has bad values."""
    errors = []

    missing = [c for c in ["id", "comment_text"] + SUBTYPES if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    for col in [c for c in SUBTYPES if c in df.columns]:
        bad = set(df[col].unique()) - {0, 1}
        if bad:
            errors.append(f"Column '{col}' has unexpected values: {bad}")

    if "id" in df.columns and df["id"].duplicated().any():
        errors.append(f"{df['id'].duplicated().sum()} duplicate IDs found")

    if "comment_text" in df.columns and df["comment_text"].isnull().any():
        warnings.warn(
            f"{df['comment_text'].isnull().sum()} null comment_text rows — "
            "will become empty strings after cleaning.",
            UserWarning, stacklevel=3,
        )

    if errors:
        raise ValueError("Raw data validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.10,
    test_size: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split.
    Stratification keeps the toxic % consistent across all three splits.
    """
    if "label" not in df.columns:
        raise ValueError("df must have a 'label' column. Run load_and_label() first.")

    total_held = val_size + test_size
    if total_held >= 1.0:
        raise ValueError(f"val_size + test_size = {total_held:.2f}, must be < 1.0")

    train, temp = train_test_split(
        df, test_size=total_held, stratify=df["label"], random_state=RANDOM_SEED
    )
    val, test = train_test_split(
        temp, test_size=test_size / total_held, stratify=temp["label"], random_state=RANDOM_SEED
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def verify_no_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str = "id",
) -> None:
    """Raise if any comment ID appears in more than one split."""
    train_ids, val_ids, test_ids = set(train[id_col]), set(val[id_col]), set(test[id_col])
    overlaps = []
    if tv := train_ids & val_ids:
        overlaps.append(f"Train ∩ Val: {len(tv)} IDs")
    if tt := train_ids & test_ids:
        overlaps.append(f"Train ∩ Test: {len(tt)} IDs")
    if vt := val_ids & test_ids:
        overlaps.append(f"Val ∩ Test: {len(vt)} IDs")
    if overlaps:
        raise ValueError("Data leakage:\n" + "\n".join(f"  - {o}" for o in overlaps))


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    data_dir: str | Path = "data",
) -> None:
    """Save train/val/test splits as CSVs. These are gitignored — each teammate generates them locally."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = data_dir / f"{name}_split.csv"
        split.to_csv(path, index=False)
        print(f"  {name:6s} -> {path}  ({len(split):,} rows)")


def load_splits(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-built splits. Raises clearly if 02_preprocessing.ipynb hasn't been run."""
    data_dir = Path(data_dir)
    paths = {n: data_dir / f"{n}_split.csv" for n in ["train", "val", "test"]}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Split CSVs not found:\n" + "\n".join(f"  - {m}" for m in missing)
            + "\n\nRun notebooks/02_preprocessing.ipynb first."
        )
    return pd.read_csv(paths["train"]), pd.read_csv(paths["val"]), pd.read_csv(paths["test"])


def class_weight_dict(train_df: pd.DataFrame) -> dict[int, float]:
    """
    Inverse-frequency class weights from the training split only.
    Equivalent to sklearn's class_weight='balanced' but returned as a dict
    so it can be passed to DistilBERT's loss function or logged.
    """
    if "label" not in train_df.columns:
        raise ValueError("train_df must have a 'label' column.")
    counts = train_df["label"].value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError(f"Only {len(counts)} class(es) in training data — need both 0 and 1.")
    n = len(train_df)
    return {int(cls): float(n / (len(counts) * count)) for cls, count in counts.items()}


def compute_class_weights_tensor(train_df: pd.DataFrame):
    """Class weights as a torch.FloatTensor for DistilBERT's CrossEntropyLoss."""
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch required. Install with: pip install torch") from e
    w = class_weight_dict(train_df)
    return torch.tensor([w[0], w[1]], dtype=torch.float)


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print a quick table showing row counts and toxic % for each split."""
    header = f"{'Split':<10} {'Total':>8} {'Toxic':>8} {'Non-toxic':>10} {'Toxic %':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    totals = [0, 0]
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n, nt = len(split), int(split["label"].sum())
        totals[0] += n
        totals[1] += nt
        print(f"{name:<10} {n:>8,} {nt:>8,} {n-nt:>10,} {nt/n*100:>9.2f}%")
    print(sep)
    n, nt = totals
    print(f"{'Total':<10} {n:>8,} {nt:>8,} {n-nt:>10,} {nt/n*100:>9.2f}%")
    print(f"\nImbalance: {(n-nt)/nt:.1f}:1  (non-toxic : toxic)")


def dataset_stats(df: pd.DataFrame) -> dict:
    """Summary statistics dict — useful for logging alongside model results."""
    wc = df["comment_text"].fillna("").str.split().str.len()
    n, nt = len(df), int(df["label"].sum())
    return {
        "n_total":                 n,
        "n_toxic":                 nt,
        "n_nontoxic":              n - nt,
        "toxic_rate":              round(nt / n, 4),
        "imbalance_ratio":         round((n - nt) / nt, 2),
        "avg_word_count":          round(float(wc.mean()), 1),
        "median_word_count":       round(float(wc.median()), 1),
        "p95_word_count":          int(np.percentile(wc, 95)),
        "max_word_count":          int(wc.max()),
        "avg_word_count_toxic":    round(float(wc[df["label"] == 1].mean()), 1),
        "avg_word_count_nontoxic": round(float(wc[df["label"] == 0].mean()), 1),
    }


def length_stats(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    """Per-class word count statistics. Use p95 to pick DistilBERT max_len."""
    df = df.copy()
    df["_wc"] = df[text_col].fillna("").str.split().str.len()
    return df.groupby("label")["_wc"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).round(1)


def sample_by_label(
    df: pd.DataFrame,
    n: int = 5,
    label: Optional[int] = None,
    text_col: str = "comment_text",
    max_chars: int = 250,
) -> None:
    """Print a readable sample of comments grouped by class."""
    groups = (
        [(label, df[df["label"] == label])]
        if label is not None
        else [(l, df[df["label"] == l]) for l in [0, 1]]
    )
    for lbl, subset in groups:
        name = {0: "NON-TOXIC", 1: "TOXIC"}.get(lbl, str(lbl))
        print(f"\n{'='*60}\n  {name}  (n={len(subset):,})\n{'='*60}")
        for i, (_, row) in enumerate(subset.sample(min(n, len(subset)), random_state=RANDOM_SEED).iterrows(), 1):
            text = str(row[text_col])
            print(f"\n  [{i}] {text[:max_chars]}{'...' if len(text) > max_chars else ''}")


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python src/data_utils.py path/to/train.csv [--save]")
        sys.exit(1)

    df = load_and_label(sys.argv[1])
    train, val, test = make_splits(df)
    split_summary(train, val, test)
    verify_no_leakage(train, val, test)
    print("\nDataset stats (train):")
    print(json.dumps(dataset_stats(train), indent=2))
    print("\nClass weights:", class_weight_dict(train))

    if "--save" in sys.argv:
        save_splits(train, val, test)