import argparse
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import torch

from preprocess import prepare_data

from .exp_model_hash_ensemble import Model
from .paths import CHECKPOINTS, REPO_ROOT


def _tokenize(text: str) -> List[str]:
    return str(text).lower().split()


def _token_index(token: str, num_features: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % num_features


def _build_counts(texts: List[str], num_features: int) -> List[Dict[int, int]]:
    out: List[Dict[int, int]] = []
    for text in texts:
        counts: Dict[int, int] = {}
        for tok in _tokenize(text):
            idx = _token_index(tok, num_features)
            counts[idx] = counts.get(idx, 0) + 1
        out.append(counts)
    return out


def _df_from_counts(counts_list: List[Dict[int, int]], num_features: int) -> np.ndarray:
    df = np.zeros(num_features, dtype=np.int64)
    for counts in counts_list:
        for idx in counts.keys():
            df[idx] += 1
    return df


def _counts_to_tfidf_matrix(counts_list: List[Dict[int, int]], idf: np.ndarray, num_features: int) -> np.ndarray:
    X = np.zeros((len(counts_list), num_features), dtype=np.float32)
    for i, counts in enumerate(counts_list):
        for idx, tf in counts.items():
            X[i, idx] = (1.0 + np.log(float(tf))) * idf[idx]
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm
    return X


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train .pt-compatible hashed TF-IDF logistic ensemble.")
    parser.add_argument("--csv", default=str(REPO_ROOT / "url_with_headlines.csv"))
    parser.add_argument("--c-values", default="2.8,3.0,3.2,3.5,4.0")
    parser.add_argument("--num-models", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default=str(CHECKPOINTS / "model_tfidf_ensemble.pt"))
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise RuntimeError(
            "Missing scikit-learn dependency. Install with:\npython3 -m pip install scikit-learn"
        ) from exc

    c_values = _parse_float_list(args.c_values)
    if not c_values:
        raise ValueError("Provide at least one C value.")

    X, y = prepare_data(args.csv)
    num_features = 1 << 15
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    train_counts_all = _build_counts(X_train, num_features)
    val_counts = _build_counts(X_val, num_features)
    n_train = len(X_train)

    rng = np.random.default_rng(args.seed)
    ensemble_idf: List[np.ndarray] = []
    ensemble_w: List[np.ndarray] = []
    ensemble_b: List[float] = []

    for m in range(args.num_models):
        c = c_values[m % len(c_values)]
        sample_idx = rng.integers(0, n_train, size=n_train, endpoint=False)
        sample_counts = [train_counts_all[i] for i in sample_idx.tolist()]
        sample_y = [y_train[i] for i in sample_idx.tolist()]

        df = _df_from_counts(sample_counts, num_features)
        idf = (np.log((1.0 + len(sample_counts)) / (1.0 + df)) + 1.0).astype(np.float32)

        X_boot = _counts_to_tfidf_matrix(sample_counts, idf, num_features)
        clf = LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=args.seed + m)
        clf.fit(X_boot, sample_y)

        ensemble_idf.append(idf)
        ensemble_w.append(clf.coef_.astype(np.float32).reshape(-1))
        ensemble_b.append(float(clf.intercept_[0]))

    # Validation accuracy for ensemble snapshot.
    val_logits = np.zeros(len(X_val), dtype=np.float32)
    for idf, w, b in zip(ensemble_idf, ensemble_w, ensemble_b):
        Xv = _counts_to_tfidf_matrix(val_counts, idf, num_features)
        val_logits += (Xv @ w + b).astype(np.float32)
    val_logits /= max(len(ensemble_w), 1)
    val_preds = (1.0 / (1.0 + np.exp(-val_logits)) >= 0.5).astype(np.int64)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"ensemble_val_acc: {val_acc:.6f}")

    model = Model()
    if args.num_models > model.max_models:
        raise ValueError(f"num-models ({args.num_models}) exceeds max supported ({model.max_models}).")
    with torch.no_grad():
        model.num_models.copy_(torch.tensor(args.num_models, dtype=torch.int64))
        for i in range(args.num_models):
            model.model_idf[i].copy_(torch.from_numpy(ensemble_idf[i]))
            model.model_weights[i].copy_(torch.from_numpy(ensemble_w[i]))
            model.model_bias[i].copy_(torch.tensor(ensemble_b[i], dtype=torch.float32))
    torch.save(model.state_dict(), args.save_path)
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
