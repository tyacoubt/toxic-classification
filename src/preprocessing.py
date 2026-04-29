import re
import pandas as pd


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    contractions = {
        r"won't": "will not",
        r"can't": "cannot",
        r"n't": " not",
        r"'re": " are",
        r"'s": " is",
        r"'d": " would",
        r"'ll": " will",
        r"'ve": " have",
        r"'m": " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_cleaning(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    output_col: str = "clean_text",
    keep_original: bool = True,
) -> pd.DataFrame:
    
    df = df.copy()
    df[output_col] = df[text_col].apply(clean_text).fillna("")

    if not keep_original:
        df = df.drop(columns=[text_col])

    return df


def length_stats(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    df = df.copy()
    df["_word_count"] = df[text_col].fillna("").str.split().str.len()
    return df.groupby("label")["_word_count"].describe(percentiles=[0.25, 0.5, 0.75, 0.95])


def clean_text_ablation(
    text: str,
    remove_html: bool = True,
    remove_urls: bool = True,
    expand_contractions: bool = True,
    remove_punctuation: bool = True,
) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    if remove_html:
        text = re.sub(r"<[^>]+>", " ", text)

    if remove_urls:
        text = re.sub(r"http\S+|www\.\S+", " ", text)

    if expand_contractions:
        contractions = {
            r"won't": "will not", r"can't": "cannot", r"n't": " not",
            r"'re": " are", r"'s": " is", r"'d": " would",
            r"'ll": " will", r"'ve": " have", r"'m": " am",
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text)

    if remove_punctuation:
        if remove_html:
            text = re.sub(r"[^\w\s]", " ", text)
        else:
            text = re.sub(r"[^\w\s<>]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    samples = [
        "You're a complete idiot!!!",
        "<b>HELLO</b> world visit http://spam.com",
        "I can't believe you won't listen to me...",
        "This is a perfectly normal comment.",
    ]
    for s in samples:
        print(f"IN:  {s}")
        print(f"OUT: {clean_text(s)}")
        print()