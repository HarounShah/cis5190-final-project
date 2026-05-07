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


def _counts_to_tfidf_matrix(per_doc_counts: List[dict], idf: np.ndarray, num_features: int) -> np.ndarray:
    X = np.zeros((len(per_doc_counts), num_features), dtype=np.float32)
    for i, counts in enumerate(per_doc_counts):
        for idx, tf in counts.items():
            X[i, idx] = (1.0 + np.log(float(tf))) * idf[idx]
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm
    return X


def _parse_c_values(raw: str) -> List[float]:
    return [float(v.strip()) for v in raw.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV for hashed TF-IDF .pt logistic model.")
    parser.add_argument("--csv", default=str(REPO_ROOT / "url_with_headlines.csv"))
    parser.add_argument("--c-values", default="1,2,4,8,12,16")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default=str(CHECKPOINTS / "model_tfidf_cv_best.pt"))
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold
    except Exception as exc:
        raise RuntimeError(
            "Missing scikit-learn dependency. Install with:\n"
            "python3 -m pip install scikit-learn"
        ) from exc

    X_text, y = prepare_data(args.csv)
    y_np = np.asarray(y, dtype=np.int64)
    num_features = 1 << 15
    c_values = _parse_c_values(args.c_values)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    cv_results: List[Tuple[float, float, float]] = []
    for c in c_values:
        fold_scores: List[float] = []
        for train_idx, val_idx in skf.split(X_text, y_np):
            X_train = [X_text[i] for i in train_idx.tolist()]
            y_train = y_np[train_idx]
            X_val = [X_text[i] for i in val_idx.tolist()]
            y_val = y_np[val_idx]

            train_counts, train_df = _build_counts_and_df(X_train, num_features)
            n_train = max(len(X_train), 1)
            idf = (np.log((1.0 + n_train) / (1.0 + train_df)) + 1.0).astype(np.float32)
            X_train_tfidf = _counts_to_tfidf_matrix(train_counts, idf, num_features)

            val_counts, _ = _build_counts_and_df(X_val, num_features)
            X_val_tfidf = _counts_to_tfidf_matrix(val_counts, idf, num_features)

            clf = LogisticRegression(
                C=c,
                solver="liblinear",
                max_iter=2000,
                random_state=args.seed,
            )
            clf.fit(X_train_tfidf, y_train)
            preds = clf.predict(X_val_tfidf)
            fold_scores.append(float(accuracy_score(y_val, preds)))

        mean_acc = float(np.mean(fold_scores))
        std_acc = float(np.std(fold_scores))
        cv_results.append((c, mean_acc, std_acc))
        print(f"C={c:.4g} | cv_mean={mean_acc:.6f} | cv_std={std_acc:.6f}")

    best_c, best_mean, best_std = max(cv_results, key=lambda t: t[1])
    print(f"best_c={best_c:.4g} | best_cv_mean={best_mean:.6f} | best_cv_std={best_std:.6f}")

    # Refit on all data and export as .pt checkpoint.
    all_counts, all_df = _build_counts_and_df(X_text, num_features)
    n_all = max(len(X_text), 1)
    all_idf = (np.log((1.0 + n_all) / (1.0 + all_df)) + 1.0).astype(np.float32)
    X_all_tfidf = _counts_to_tfidf_matrix(all_counts, all_idf, num_features)

    final_clf = LogisticRegression(
        C=best_c,
        solver="liblinear",
        max_iter=2000,
        random_state=args.seed,
    )
    final_clf.fit(X_all_tfidf, y_np)

    model = Model()
    with torch.no_grad():
        model.idf.copy_(torch.from_numpy(all_idf))
        model.linear.weight.copy_(torch.from_numpy(final_clf.coef_.astype(np.float32)))
        model.linear.bias.copy_(torch.from_numpy(final_clf.intercept_.astype(np.float32)))
    torch.save(model.state_dict(), args.save_path)
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
