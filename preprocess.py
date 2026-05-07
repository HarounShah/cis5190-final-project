from typing import List, Tuple
from urllib.parse import urlparse
import re
import unicodedata

import pandas as pd


def _get_domain(url: str) -> str:
    """Extract normalized domain from URL."""
    netloc = urlparse(str(url)).netloc.lower().strip()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def clean_text(text: str) -> str:
    """
    Normalize headline text to reduce noise:
    - unicode normalize + ascii fallback
    - lowercase
    - replace most punctuation with spaces
    - collapse whitespace
    """
    s = unicodedata.normalize("NFKD", str(text))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = s.replace("\n", " ").replace("\t", " ")
    # Keep letters/numbers and a couple common currency/percent symbols.
    s = re.sub(r"[^a-z0-9%$]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Template preprocessing for leaderboard.

    Requirements:
    - Must read the provided data path at `path`.
    - Must return a tuple (X, y):
        X: a list of model-ready inputs (these must match what your model expects in predict(...))
        y: a list of ground-truth labels aligned with X (same length)

    Notes:
    - The evaluation backend will call this function with the shared validation data
    - Ensure the output format (types, shapes) of X matches your model's predict(...) inputs.
    """
    df = pd.read_csv(path)
    required_columns = {"url", "headline"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    # Binary label mapping for this project dataset.
    label_map = {
        "foxnews.com": 1,
        "nbcnews.com": 0,
    }

    X: List[str] = []
    y: List[int] = []

    for _, row in df.iterrows():
        url = str(row["url"]).strip()
        headline = clean_text(row["headline"])
        domain = _get_domain(url)

        if domain not in label_map:
            # Skip rows with unsupported domains.
            continue

        X.append(headline)
        y.append(label_map[domain])

    if not X:
        raise ValueError("No valid rows found after preprocessing.")

    return X, y


