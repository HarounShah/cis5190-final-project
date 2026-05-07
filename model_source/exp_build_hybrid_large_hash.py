import argparse
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocess import prepare_data

from .exp_model_hybrid_large_hash import Model
from .model_branch_word_char_tfidf import Model as VocabCharModel
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _best_threshold(y_true: List[int], probs: np.ndarray) -> Tuple[float, float]:
    best_t = 0.5
    best_acc = -1.0
    y_arr = np.asarray(y_true, dtype=np.int64)
    for t in np.linspace(0.35, 0.65, 61):
        preds = (probs >= t).astype(np.int64)
        acc = float((preds == y_arr).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final model (.pt) with larger hash space + weighted hybrid.")
    parser.add_argument("--csv", default=str(REPO_ROOT / "url_with_headlines.csv"))
    parser.add_argument("--vc-ckpt", default=str(CHECKPOINTS / "model_vocab_char_c4.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default=str(CHECKPOINTS / "model_final.pt"))
    args = parser.parse_args()

    X, y = prepare_data(args.csv)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # Train hash branch with larger hash space and C sweep near best region.
    hash_num_features = 1 << 16
    train_counts = _build_counts(X_train, hash_num_features)
    val_counts = _build_counts(X_val, hash_num_features)
    train_df = _df_from_counts(train_counts, hash_num_features)
    idf = (np.log((1.0 + len(X_train)) / (1.0 + train_df)) + 1.0).astype(np.float32)
    Xtr = _counts_to_tfidf_matrix(train_counts, idf, hash_num_features)
    Xva = _counts_to_tfidf_matrix(val_counts, idf, hash_num_features)

    best = None
    for c in [2.8, 3.0, 3.2, 3.5]:
        clf = LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=args.seed)
        clf.fit(Xtr, y_train)
        logits = (Xva @ clf.coef_.reshape(-1) + clf.intercept_[0]).astype(np.float32)
        probs = _sigmoid(logits)
        t, acc = _best_threshold(y_val, probs)
        if best is None or acc > best["acc"]:
            best = {
                "c": c,
                "clf": clf,
                "thr": t,
                "acc": acc,
            }
    assert best is not None
    print(f"best_hash_c: {best['c']}")
    print(f"best_hash_val_acc: {best['acc']:.6f}")

    # Load vocab+char branch and get validation probs.
    vc = VocabCharModel()
    vc.load_state_dict(torch.load(args.vc_ckpt, map_location="cpu"), strict=False)
    vc.eval()
    xv = vc._featurize_batch(X_val)
    with torch.no_grad():
        pv = torch.sigmoid(vc.forward(xv)).cpu().numpy()

    # Hash probs from best model
    hash_logits = (Xva @ best["clf"].coef_.reshape(-1) + best["clf"].intercept_[0]).astype(np.float32)
    ph = _sigmoid(hash_logits)

    # Weight sweep for hybrid blend
    best_combo = None
    for a in np.linspace(0.55, 0.85, 13):  # hash weight
        b = 1.0 - a
        lh = np.log(ph / np.clip(1.0 - ph, 1e-8, 1.0))
        lv = np.log(pv / np.clip(1.0 - pv, 1e-8, 1.0))
        probs = _sigmoid(a * lh + b * lv)
        thr, acc = _best_threshold(y_val, probs)
        if best_combo is None or acc > best_combo["acc"]:
            best_combo = {"a": float(a), "b": float(b), "thr": float(thr), "acc": float(acc)}
    assert best_combo is not None
    print(f"best_hybrid_weights: hash={best_combo['a']:.3f}, vocab_char={best_combo['b']:.3f}")
    print(f"best_hybrid_threshold: {best_combo['thr']:.3f}")
    print(f"best_hybrid_val_acc: {best_combo['acc']:.6f}")

    # Pack final checkpoint.
    out = Model()
    with torch.no_grad():
        out.hash_weight.copy_(torch.tensor(best_combo["a"], dtype=torch.float32))
        out.vocab_char_weight.copy_(torch.tensor(best_combo["b"], dtype=torch.float32))
        out.decision_threshold.copy_(torch.tensor(best_combo["thr"], dtype=torch.float32))

        out.hash_idf.copy_(torch.from_numpy(idf))
        out.hash_w.copy_(torch.from_numpy(best["clf"].coef_.astype(np.float32).reshape(-1)))
        out.hash_b.copy_(torch.tensor(float(best["clf"].intercept_[0]), dtype=torch.float32))

        out.vc_idf.copy_(vc.idf)
        out.vc_w.copy_(vc.linear.weight.detach().reshape(-1))
        out.vc_b.copy_(vc.linear.bias.detach().reshape(()))
        out.word_hashes.copy_(vc.word_hashes)
        out.word_indices.copy_(vc.word_indices)
        out.word_vocab_size.copy_(vc.word_vocab_size)
        out.char_hashes.copy_(vc.char_hashes)
        out.char_indices.copy_(vc.char_indices)
        out.char_vocab_size.copy_(vc.char_vocab_size)

    torch.save(out.state_dict(), args.save_path)
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
