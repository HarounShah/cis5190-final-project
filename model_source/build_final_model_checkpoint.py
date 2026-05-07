import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from final_model import Model
from preprocess import prepare_data

from .model_branch_hash_tfidf import Model as HashModel
from .model_branch_word_char_tfidf import Model as VocabCharModel
from .paths import CHECKPOINTS, REPO_ROOT


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _best_threshold(y_true, probs) -> float:
    best_t = 0.5
    best_acc = -1.0
    for t in np.linspace(0.3, 0.7, 81):
        preds = (probs >= t).astype(int)
        acc = float((preds == np.asarray(y_true)).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weighted hybrid ensemble .pt checkpoint.")
    parser.add_argument("--hash-ckpt", default=str(CHECKPOINTS / "model_tfidf_c3.pt"))
    parser.add_argument("--vc-ckpt", default=str(CHECKPOINTS / "model_vocab_char_c4.pt"))
    parser.add_argument("--hash-weight", type=float, default=0.7)
    parser.add_argument("--vc-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="final_model.pt")
    args = parser.parse_args()

    # Load branch models
    hash_model = HashModel()
    hash_model.load_state_dict(torch.load(args.hash_ckpt, map_location="cpu"), strict=False)
    hash_model.eval()

    vc_model = VocabCharModel()
    vc_model.load_state_dict(torch.load(args.vc_ckpt, map_location="cpu"), strict=False)
    vc_model.eval()

    # Tune threshold on validation split using weighted probabilities
    X, y = prepare_data(str(REPO_ROOT / "url_with_headlines.csv"))
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    # Hash probs
    xh = hash_model._featurize_batch(X_val)
    with torch.no_grad():
        ph = torch.sigmoid(hash_model.forward(xh)).cpu().numpy()

    # Vocab+char probs
    xv = vc_model._featurize_batch(X_val)
    with torch.no_grad():
        pv = torch.sigmoid(vc_model.forward(xv)).cpu().numpy()

    a = float(args.hash_weight)
    b = float(args.vc_weight)
    denom = max(a + b, 1e-12)
    a, b = a / denom, b / denom

    # Blend in logit space
    logit_h = np.log(ph / np.clip(1.0 - ph, 1e-8, 1.0))
    logit_v = np.log(pv / np.clip(1.0 - pv, 1e-8, 1.0))
    blended_probs = _sigmoid(a * logit_h + b * logit_v)
    thr = _best_threshold(y_val, blended_probs)
    acc = float(((blended_probs >= thr).astype(int) == np.asarray(y_val)).mean())
    print(f"val_threshold: {thr:.3f}")
    print(f"val_acc_weighted: {acc:.6f}")

    # Pack into hybrid checkpoint
    out = Model()
    with torch.no_grad():
        out.hash_weight.copy_(torch.tensor(a, dtype=torch.float32))
        out.vocab_char_weight.copy_(torch.tensor(b, dtype=torch.float32))
        out.decision_threshold.copy_(torch.tensor(thr, dtype=torch.float32))

        out.hash_idf.copy_(hash_model.idf)
        out.hash_w.copy_(hash_model.linear.weight.detach().reshape(-1))
        out.hash_b.copy_(hash_model.linear.bias.detach().reshape(()))

        out.vc_idf.copy_(vc_model.idf)
        out.vc_w.copy_(vc_model.linear.weight.detach().reshape(-1))
        out.vc_b.copy_(vc_model.linear.bias.detach().reshape(()))
        out.word_hashes.copy_(vc_model.word_hashes)
        out.word_indices.copy_(vc_model.word_indices)
        out.word_vocab_size.copy_(vc_model.word_vocab_size)
        out.char_hashes.copy_(vc_model.char_hashes)
        out.char_indices.copy_(vc_model.char_indices)
        out.char_vocab_size.copy_(vc_model.char_vocab_size)

    torch.save(out.state_dict(), args.save_path)
    print(f"saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
