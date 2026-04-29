import pandas as pd
from sklearn.model_selection import train_test_split

SUBTYPES = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
RANDOM_SEED = 42   # everyone uses this same seed

def load_and_label(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['label'] = df[SUBTYPES].max(axis=1).astype(int)
    return df[['id','comment_text','label']]

def make_splits(df: pd.DataFrame):
    train, temp = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=RANDOM_SEED)
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp['label'], random_state=RANDOM_SEED)
    return train.reset_index(drop=True), \
           val.reset_index(drop=True), \
           test.reset_index(drop=True)

def class_weight_dict(train_df: pd.DataFrame) -> dict[int, float]:
    """
    Compute inverse-frequency class weights from the training split.
 
    Mathematically equivalent to scikit-learn's class_weight='balanced',
    but returned as an explicit dict so it can be logged, inspected, and
    passed to DistilBERT's loss function.
 
    Formula:
        weight[c] = n_total / (n_classes * n_samples_in_class_c)
 
    With a 9:1 imbalance (143,346 non-toxic, 16,225 toxic):
        weight[0] (non-toxic) ≈  0.56
        weight[1] (toxic)     ≈  4.92
        ratio: toxic examples contribute ~8.8x more to the loss
    """
    if "label" not in train_df.columns:
        raise ValueError("train_df must have a 'label' column.")
 
    counts = train_df["label"].value_counts().sort_index()
 
    if len(counts) < 2:
        raise ValueError(
            f"Only {len(counts)} class(es) found: {counts.index.tolist()}. "
            "Both classes (0 and 1) must be present in the training split."
        )
 
    n_total   = len(train_df)
    n_classes = len(counts)
    return {
        int(cls): float(n_total / (n_classes * count))
        for cls, count in counts.items()
    }