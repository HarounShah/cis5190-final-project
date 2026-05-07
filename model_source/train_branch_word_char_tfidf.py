import argparse
import hashlib
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from preprocess import prepare_data

from .model_branch_word_char_tfidf import Model
from .paths import CHECKPOINTS, REPO_ROOT


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def _word_tokens(text: str) -> List[str]:
    toks = str(text).split()
    bigrams = [f"{toks[i]}__{toks[i+1]}" for i in range(len(toks) - 1)]
    return toks + bigrams


def _char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> List[str]:
    s = str(text)
    out: List[str] = []
    for n in range(n_min, n_max + 1):
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            out.append(s[i : i + n])
    return out


def _build_word_vocab(texts: List[str], max_features: int, min_df: int) -> Dict[str, int]:
    df_counter: Counter = Counter()
    for text in texts:
        for tok in set(_word_tokens(text)):
            df_counter[tok] += 1
    eligible = [(tok, df) for tok, df in df_counter.items() if df >= min_df]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    selected = eligible[:max_features]
    return {tok: i for i, (tok, _) in enumerate(selected)}


def _build_char_vocab(texts: List[str], max_features: int, min_df: int) -> Dict[str, int]:
    df_counter: Counter = Counter()
    for text in texts:
        for gram in set(_char_ngrams(text)):
            df_counter[gram] += 1
    eligible = [(g, df) for g, df in df_counter.items() if df >= min_df]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    selected = eligible[:max_features]
    return {gram: i for i, (gram, _) in enumerate(selected)}


def _vectorize(
    texts: List[str],
    word_vocab: Dict[str, int],
    char_vocab: Dict[str, int],
    max_word_features: int,
    max_char_features: int,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    total_features = max_word_features + max_char_features
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    df = np.zeros(total_features, dtype=np.int64)

    for i, text in enumerate(texts):
        counts: Dict[int, int] = {}
        for tok in _word_tokens(text):
            idx = word_vocab.get(tok)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        for gram in _char_ngrams(text):
            idx = char_vocab.get(gram)
            if idx is not None:
                feat_idx = max_word_features + idx
                counts[feat_idx] = counts.get(feat_idx, 0) + 1
        for feat_idx, tf in counts.items():
            rows.append(i)
            cols.append(feat_idx)
            vals.append(float(tf))
            df[feat_idx] += 1

    X_tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), total_features), dtype=np.float32)
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
    parser = argparse.ArgumentParser(description="Train .pt-only vocab word+char TF-IDF logistic model.")
    parser.add_argument("--csv", default="url_with_headlines.csv")
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-word-features", type=int, default=40000)
    parser.add_argument("--max-char-features", type=int, default=20000)
    parser.add_argument("--word-min-df", type=int, default=2)
    parser.add_argument("--char-min-df", type=int, default=3)
    parser.add_argument("--save-path", default=str(CHECKPOINTS / "model_vocab_char.pt"))
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

    word_vocab = _build_word_vocab(X_train, args.max_word_features, args.word_min_df)
    char_vocab = _build_char_vocab(X_train, args.max_char_features, args.char_min_df)
    X_train_tf, df_train = _vectorize(
        X_train, word_vocab, char_vocab, args.max_word_features, args.max_char_features
    )
    n_train = max(len(X_train), 1)
    idf_train = (np.log((1.0 + n_train) / (1.0 + df_train)) + 1.0).astype(np.float32)
    X_train_tfidf = _tfidf_transform(X_train_tf, idf_train)

    X_val_tf, _ = _vectorize(
        X_val, word_vocab, char_vocab, args.max_word_features, args.max_char_features
    )
    X_val_tfidf = _tfidf_transform(X_val_tf, idf_train)

    clf = LogisticRegression(C=args.c, solver="liblinear", max_iter=4000, random_state=args.seed)
    clf.fit(X_train_tfidf, y_train)
    val_acc = accuracy_score(y_val, clf.predict(X_val_tfidf))
    print(f"val_acc: {val_acc:.6f}")

    # Refit on all data for final export.
    word_vocab_all = _build_word_vocab(X_text, args.max_word_features, args.word_min_df)
    char_vocab_all = _build_char_vocab(X_text, args.max_char_features, args.char_min_df)
    X_all_tf, df_all = _vectorize(
        X_text, word_vocab_all, char_vocab_all, args.max_word_features, args.max_char_features
    )
    n_all = max(len(X_text), 1)
    idf_all = (np.log((1.0 + n_all) / (1.0 + df_all)) + 1.0).astype(np.float32)
    X_all_tfidf = _tfidf_transform(X_all_tf, idf_all)

    clf.fit(X_all_tfidf, y)

    model = Model()
    with torch.no_grad():
        model.linear.weight.copy_(torch.from_numpy(clf.coef_.astype(np.float32)))
        model.linear.bias.copy_(torch.from_numpy(clf.intercept_.astype(np.float32)))
        model.idf.copy_(torch.from_numpy(idf_all.astype(np.float32)))

        word_hashes, word_idxs, word_size = _build_lookup_tensors(word_vocab_all, args.max_word_features)
        char_hashes, char_idxs, char_size = _build_lookup_tensors(char_vocab_all, args.max_char_features)
        model.word_hashes.copy_(word_hashes)
        model.word_indices.copy_(word_idxs)
        model.word_vocab_size.copy_(word_size)
        model.char_hashes.copy_(char_hashes)
        model.char_indices.copy_(char_idxs)
        model.char_vocab_size.copy_(char_size)

    torch.save(model.state_dict(), args.save_path)
    print(f"saved checkpoint to {args.save_path}")
    print(f"word_vocab_size: {int(model.word_vocab_size.item())}")
    print(f"char_vocab_size: {int(model.char_vocab_size.item())}")


if __name__ == "__main__":
    main()
