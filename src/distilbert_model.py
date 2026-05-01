"""
distilbert_model.py
-------------------
DistilBERT fine-tuned classifier for toxic comment detection.

We use distilbert-base-uncased because it's ~40% faster than full BERT
while keeping ~97% of its performance. The model is fine-tuned end-to-end
on the Jigsaw binary labels.

DistilBERT doesn't need the TF-IDF preprocessing pipeline — the tokenizer
handles lowercasing and sub-word splitting, so we feed it raw comment_text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from src.data_utils import RANDOM_SEED, compute_class_weights_tensor
from src.evaluation import evaluate, extract_errors

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128  # covers ~95th percentile of Jigsaw comment lengths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ToxicDataset(Dataset):
    """Tokenizes comments on-the-fly and returns tensors for the DataLoader."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: DistilBertTokenizerFast,
        max_len: int = MAX_LEN,
    ):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,  # long comments get cut off at MAX_LEN tokens
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    df: pd.DataFrame,
    tokenizer: DistilBertTokenizerFast,
    batch_size: int = 32,
    shuffle: bool = False,
    max_len: int = MAX_LEN,
    text_col: str = "comment_text",
) -> DataLoader:
    """
    Wrap a split DataFrame into a PyTorch DataLoader.

    Args:
        df:         Split DataFrame with text_col and 'label' columns.
        tokenizer:  Loaded DistilBertTokenizerFast.
        batch_size: Examples per batch (default 32).
        shuffle:    Shuffle each epoch — True for train, False for val/test.
        max_len:    Maximum token sequence length.
        text_col:   Column containing raw comment text.

    Returns:
        DataLoader ready for the training loop.
    """
    texts  = df[text_col].fillna("").tolist()
    labels = df["label"].tolist()
    dataset = ToxicDataset(texts, labels, tokenizer, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,     # 0 avoids multiprocessing issues on macOS
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: MPS (Apple Silicon), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_distilbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_len: int = MAX_LEN,
    use_class_weights: bool = True,
    text_col: str = "comment_text",
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast, dict]:
    """
    Fine-tune DistilBERT and return the best checkpoint by validation F1.

    Design decisions
    ----------------
    lr=2e-5
        Standard learning rate for fine-tuning BERT-family models.
        Higher values (e.g. 1e-4) cause catastrophic forgetting of pretrained
        weights; lower values (e.g. 1e-6) don't converge in 3 epochs.

    warmup_ratio=0.1
        Linear LR warmup over 10% of training steps, then linear decay.
        Warmup prevents destabilising the pretrained weights with large
        gradient updates in the first few batches.

    use_class_weights=True
        Weighted cross-entropy loss using inverse-frequency weights from
        data_utils.compute_class_weights_tensor(). Without this, the model
        quickly learns to predict non-toxic for everything (easy 90% accuracy
        on the 9:1 imbalanced dataset, terrible recall on toxic comments).

    gradient clipping (max_norm=1.0)
        Prevents exploding gradients, which can happen when fine-tuning on
        a domain-shifted dataset.

    Args:
        train_df:          Training split DataFrame (needs text_col and 'label').
        val_df:            Validation split DataFrame.
        batch_size:        Batch size for both loaders (default 32).
        epochs:            Number of fine-tuning epochs (default 3).
        lr:                AdamW learning rate (default 2e-5).
        warmup_ratio:      Fraction of total steps used for LR warmup.
        max_len:           Maximum token length (default 128).
        use_class_weights: Weight cross-entropy loss by inverse class frequency.
        text_col:          Column containing raw comment text.

    Returns:
        (best_model, tokenizer, history)
        best_model: checkpoint with highest validation F1
        tokenizer:  needed for inference later
        history:    dict with 'train_loss', 'val_loss', 'val_f1' per epoch
    """
    from sklearn.metrics import f1_score as _f1

    torch.manual_seed(RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_loader = make_dataloader(train_df, tokenizer, batch_size, shuffle=True,  max_len=max_len, text_col=text_col)
    val_loader   = make_dataloader(val_df,   tokenizer, batch_size, shuffle=False, max_len=max_len, text_col=text_col)

    if use_class_weights:
        weights = compute_class_weights_tensor(train_df).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        print(f"Class weights: {weights.cpu().tolist()}")
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps   = len(train_loader) * epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1 = -1.0
    best_state  = None

    for epoch in range(1, epochs + 1):
        # ---- Training loop ----
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # ---- Validation loop ----
        model.eval()
        running_val_loss = 0.0
        val_preds_list, val_labels_list = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]  ", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = loss_fn(outputs.logits, labels)
                running_val_loss += loss.item()

                preds = outputs.logits.argmax(dim=1).cpu().numpy()
                val_preds_list.extend(preds)
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)
        val_f1 = _f1(val_labels_list, val_preds_list, pos_label=1, zero_division=0)

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_f1"].append(round(val_f1, 4))

        print(
            f"Epoch {epoch}/{epochs}  |  "
            f"train loss: {avg_train_loss:.4f}  |  "
            f"val loss: {avg_val_loss:.4f}  |  "
            f"val F1 (toxic): {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best — saving checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)
    print(f"\nTraining done. Best val F1: {best_val_f1:.4f}")

    return model, tokenizer, history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    df: pd.DataFrame,
    batch_size: int = 64,
    max_len: int = MAX_LEN,
    text_col: str = "comment_text",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a DataFrame and return binary predictions and probabilities.

    Args:
        model:      Fine-tuned DistilBertForSequenceClassification.
        tokenizer:  Loaded tokenizer (saved alongside the model).
        df:         DataFrame with text_col and 'label' columns.
        batch_size: Batch size for inference (can be larger than training).
        max_len:    Maximum token length.
        text_col:   Column containing raw comment text.

    Returns:
        (preds, probs) both shape (n_examples,).
        preds: binary array {0, 1}
        probs: probability of toxic class in [0, 1]
    """
    device = get_device()
    model.eval()
    model.to(device)

    loader = make_dataloader(df, tokenizer, batch_size, shuffle=False, max_len=max_len, text_col=text_col)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds          = outputs.logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)

    return np.array(all_preds), np.array(all_probs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save: bool = True) -> None:
    """
    Plot train/val loss and val F1 across epochs.

    Args:
        history: Dict returned by train_distilbert() with keys
                 'train_loss', 'val_loss', 'val_f1'.
        save:    Save to results/distilbert_training_curves.png.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train loss", color="#D85A30", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   "o-", label="Val loss",   color="#378ADD", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("DistilBERT — training & validation loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(list(epochs))

    ax2.plot(epochs, history["val_f1"], "o-", color="#1D9E75", linewidth=2, markersize=7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 score (toxic class)")
    ax2.set_title("DistilBERT — validation F1 per epoch")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(list(epochs))

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = RESULTS_DIR / "distilbert_training_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_model(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    models_dir: Path = MODELS_DIR,
) -> None:
    """
    Save fine-tuned model and tokenizer using HuggingFace's save_pretrained.

    This saves all weights, config, and vocab files needed to reload the
    model later with load_pretrained — no manual pickle needed.

    Args:
        model:      Fine-tuned DistilBertForSequenceClassification.
        tokenizer:  Tokenizer to save alongside the model.
        models_dir: Parent directory; model goes into models/distilbert_finetuned/.
    """
    save_path = Path(models_dir) / "distilbert_finetuned"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved → {save_path}")


def load_model(
    models_dir: Path = MODELS_DIR,
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
    """
    Load a previously fine-tuned model from disk.

    Args:
        models_dir: Parent directory containing distilbert_finetuned/.

    Returns:
        (model, tokenizer) both ready for inference.

    Raises:
        FileNotFoundError: If the saved model directory doesn't exist.
    """
    load_path = Path(models_dir) / "distilbert_finetuned"
    if not load_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {load_path}\n"
            "Run train_distilbert() and save_model() first."
        )
    model     = DistilBertForSequenceClassification.from_pretrained(load_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(load_path)
    print(f"Model loaded from {load_path}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_distilbert(
    train_path: str = "data/train_split.csv",
    val_path:   str = "data/val_split.csv",
    test_path:  str = "data/test_split.csv",
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 2e-5,
) -> dict:
    """
    Full DistilBERT pipeline: load → train → evaluate → save.

    Args:
        train_path: Path to train_split.csv.
        val_path:   Path to val_split.csv.
        test_path:  Path to test_split.csv.
        batch_size: Training batch size.
        epochs:     Number of fine-tuning epochs.
        lr:         AdamW learning rate.

    Returns:
        Test set results dict (same format as evaluate() returns).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load splits
    print("Loading splits ...")
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)
    print(f"  Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")

    # Train
    model, tokenizer, history = train_distilbert(
        train, val,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
    )

    plot_training_curves(history)

    # Evaluate on val
    print("\n--- Validation set ---")
    val_preds, val_probs = predict(model, tokenizer, val)
    evaluate(val["label"].values, val_preds, model_name="DistilBERT (val)", y_prob=val_probs)

    # Evaluate on test
    print("\n--- Test set ---")
    test_preds, test_probs = predict(model, tokenizer, test)
    test_results = evaluate(
        test["label"].values, test_preds,
        model_name="DistilBERT",
        y_prob=test_probs,
    )

    # Save predictions
    test_out = test.copy()
    test_out["distilbert_pred"] = test_preds
    test_out["distilbert_prob"] = test_probs.round(6)
    test_out.to_csv(RESULTS_DIR / "test_distilbert_preds.csv", index=False)
    print(f"Predictions saved → {RESULTS_DIR / 'test_distilbert_preds.csv'}")

    # Save results JSON
    test_results["epochs"]     = epochs
    test_results["lr"]         = lr
    test_results["batch_size"] = batch_size
    test_results["history"]    = history
    with open(RESULTS_DIR / "distilbert_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved     → {RESULTS_DIR / 'distilbert_results.json'}")

    # Save model
    save_model(model, tokenizer)

    # Save error samples for error_analysis.ipynb
    errors = extract_errors(test_out, pred_col="distilbert_pred")
    errors["false_positives"].to_csv(RESULTS_DIR / "errors_distilbert_fp.csv", index=False)
    errors["false_negatives"].to_csv(RESULTS_DIR / "errors_distilbert_fn.csv", index=False)
    print(f"Error samples saved → {RESULTS_DIR}/errors_distilbert_*.csv")

    return test_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_split.csv"
    val_path   = sys.argv[2] if len(sys.argv) > 2 else "data/val_split.csv"
    test_path  = sys.argv[3] if len(sys.argv) > 3 else "data/test_split.csv"

    results = run_distilbert(train_path=train_path, val_path=val_path, test_path=test_path)
    print(f"\nFinal test F1 (toxic class): {results['f1_toxic']:.4f}")
