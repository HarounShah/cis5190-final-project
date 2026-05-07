import argparse
import hashlib
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from preprocess import prepare_data

from .exp_model_vocab_only_tfidf import Model
from .paths import CHECKPOINTS, REPO_ROOT


def _stable_hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _tokenize(text: str) -> List[str]:
    return str(text).split()


def _build_vocab(texts: List[str], max_features: int, min_df: int) -> Dict[str, int]:
    df_counter: Counter = Counter()
    for text in texts:
        uniq = set(_tokenize(text))
        for tok in uniq:
            df_counter[tok] += 1
    eligible = [(tok, df) for tok, df in df_counter.items() if df >= min_df]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    selected = eligible[:max_features]
    return {tok: i for i, (tok, _) in enumerate(selected)}


def _vectorize(
    texts: List[str], vocab: Dict[str, int], max_features: int
) -> Tuple[sp.csr_matrix, np.ndarray]:
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    df = np.zeros(max_features, dtype=np.int64)

    for i, text in enumerate(texts):
        counts: Dict[int, int] = {}
        for tok in _tokenize(text):
            idx = vocab.get(tok)
            if idx is None:
                continue
            counts[idx] = counts.get(idx, 0) + 1
        for idx, tf in counts.items():
            rows.append(i)
            cols.append(idx)
            vals.append(float(tf))
            df[idx] += 1

    X_tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), max_features), dtype=np.float32)
    return X_tf, df


def _tfidf_transform(X_tf: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    X = X_tf.copy().astype(np.float32)
    X.data = 1.0 + np.log(X.data)
    X = X.multiply(idf)
    row_norm = np.sqrt(X.multiply(X).sum(axis=1)).A1
    row_norm[row_norm == 0] = 1.0
    X = sp.diags(1.0 / row_norm).dot(X)
    return X.tocsr()


def _build_lookup_tensors(vocab: Dict[str, int], max_features: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = [(_stable_hash(tok), idx) for tok, idx in vocab.items()]
    pairs.sort(key=lambda x: x[0])

    hashes = torch.full((max_features,), 2**63 - 1, dtype=torch.int64)
    idxs = torch.full((max_features,), -1, dtype=torch.int64)
    n_vocab = len(pairs)
    for i, (h, idx) in enumerate(pairs):
        hashes[i] = int(h)
        idxs[i] = int(idx)
    vocab_size = torch.tensor(n_vocab, dtype=torch.int64)
    return hashes, idxs, vocab_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Train .pt-only vocab TF-IDF logistic model.")
    parser.add_argument("--csv", default=str(REPO_ROOT / "url_with_headlines.csv"))
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--save-path", default=str(CHECKPOINTS / "model_vocab.pt"))
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise RuntimeError(
            "Missing scikit-learn dependency. Install with:\npython3 -m pip install scikit-learn"
        ) from exc

    X_text, y = prepare_data(args.csv)
    X_train, X_val, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    vocab = _build_vocab(X_train, max_features=args.max_features, min_df=args.min_df)
    X_train_tf, df_train = _vectorize(X_train, vocab=vocab, max_features=args.max_features)
    n_train = max(len(X_train), 1)
    idf_train = (np.log((1.0 + n_train) / (1.0 + df_train)) + 1.0).astype(np.float32)
    X_train_tfidf = _tfidf_transform(X_train_tf, idf_train)

    X_val_tf, _ = _vectorize(X_val, vocab=vocab, max_features=args.max_features)
    X_val_tfidf = _tfidf_transform(X_val_tf, idf_train)

    clf = LogisticRegression(
        C=args.c, solver="liblinear", max_iter=3000, random_state=args.seed
    )
    clf.fit(X_train_tfidf, y_train)
    val_acc = accuracy_score(y_val, clf.predict(X_val_tfidf))
    print(f"val_acc: {val_acc:.6f}")

    # Refit vocabulary + idf on all data before final training.
    vocab_all = _build_vocab(X_text, max_features=args.max_features, min_df=args.min_df)
    X_all_tf, df_all = _vectorize(X_text, vocab=vocab_all, max_features=args.max_features)
    n_all = max(len(X_text), 1)
    idf_all = (np.log((1.0 + n_all) / (1.0 + df_all)) + 1.0).astype(np.float32)
    X_all_tfidf = _tfidf_transform(X_all_tf, idf_all)

    clf.fit(X_all_tfidf, y)

    model = Model()
    with torch.no_grad():
        coef = np.zeros((1, args.max_features), dtype=np.float32)
        coef[:, : clf.coef_.shape[1]] = clf.coef_.astype(np.float32)
        model.linear.weight.copy_(torch.from_numpy(coef))
        model.linear.bias.copy_(torch.from_numpy(clf.intercept_.astype(np.float32)))

        idf_padded = np.ones(args.max_features, dtype=np.float32)
        idf_padded[: len(idf_all)] = idf_all[: args.max_features]
        model.idf.copy_(torch.from_numpy(idf_padded))

        hashes, idxs, vocab_size = _build_lookup_tensors(vocab_all, max_features=args.max_features)
        model.token_hashes.copy_(hashes)
        model.token_indices.copy_(idxs)
        model.vocab_size.copy_(vocab_size)

    torch.save(model.state_dict(), args.save_path)
    print(f"saved checkpoint to {args.save_path}")
    print(f"vocab_size: {int(model.vocab_size.item())}")


if __name__ == "__main__":
    main()
