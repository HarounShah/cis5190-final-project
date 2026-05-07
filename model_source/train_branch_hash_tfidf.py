import argparse
import hashlib
from typing import List, Tuple

import numpy as np
import torch

from preprocess import prepare_data

from .model_branch_hash_tfidf import Model
from .paths import CHECKPOINTS, REPO_ROOT


def _tokenize(text: str) -> List[str]:
    return str(text).lower().split()


def _token_index(token: str, num_features: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % num_features


def _build_counts_and_df(texts: List[str], num_features: int) -> Tuple[List[dict], np.ndarray]:
    per_doc_counts: List[dict] = []
    df = np.zeros(num_features, dtype=np.int64)
    for text in texts:
        counts = {}
        for token in _tokenize(text):
            idx = _token_index(token, num_features)
            counts[idx] = counts.get(idx, 0) + 1
        per_doc_counts.append(counts)
        for idx in counts.keys():
            df[idx] += 1
    return per_doc_counts, df


def _counts_to_tfidf_matrix(
    per_doc_counts: List[dict],
    idf: np.ndarray,
    num_features: int,
) -> np.ndarray:
    X = np.zeros((len(per_doc_counts), num_features), dtype=np.float32)
    for i, counts in enumerate(per_doc_counts):
        for idx, tf in counts.items():
            X[i, idx] = (1.0 + np.log(float(tf))) * idf[idx]
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="Train .pt-compatible hashed TF-IDF logistic model.")
    parser.add_argument("--csv", default="url_with_headlines.csv")
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default=str(CHECKPOINTS / "model_tfidf.pt"))
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise RuntimeError(
            "Missing scikit-learn dependency. Install with:\n"
            "python3 -m pip install scikit-learn"
        ) from exc

    X_text, y = prepare_data(args.csv)
    num_features = 1 << 15

    X_train, X_val, y_train, y_val = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )

    train_counts, train_df = _build_counts_and_df(X_train, num_features)
    n_train = max(len(X_train), 1)
    idf = (np.log((1.0 + n_train) / (1.0 + train_df)) + 1.0).astype(np.float32)

    X_train_tfidf = _counts_to_tfidf_matrix(train_counts, idf, num_features)

    val_counts, _ = _build_counts_and_df(X_val, num_features)
    X_val_tfidf = _counts_to_tfidf_matrix(val_counts, idf, num_features)

    clf = LogisticRegression(
        C=args.c,
        solver="liblinear",
        max_iter=2000,
        random_state=args.seed,
    )
    clf.fit(X_train_tfidf, y_train)
    val_preds = clf.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"val_acc: {val_acc:.6f}")

    # Refit IDF on all data and retrain logistic model for final exported weights.
    all_counts, all_df = _build_counts_and_df(X_text, num_features)
    n_all = max(len(X_text), 1)
    all_idf = (np.log((1.0 + n_all) / (1.0 + all_df)) + 1.0).astype(np.float32)
    X_all_tfidf = _counts_to_tfidf_matrix(all_counts, all_idf, num_features)

    clf.fit(X_all_tfidf, y)

    model = Model()
    with torch.no_grad():
        model.idf.copy_(torch.from_numpy(all_idf))
        model.linear.weight.copy_(torch.from_numpy(clf.coef_.astype(np.float32)))
        model.linear.bias.copy_(torch.from_numpy(clf.intercept_.astype(np.float32)))

    torch.save(model.state_dict(), args.save_path)
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
