import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from collections import Counter

from final_model import Model as HybridModel
from model_source.model_branch_hash_tfidf import Model
from model_source.model_branch_word_char_tfidf import Model as WordCharModel
from preprocess import clean_text, prepare_data

_REPO = Path(__file__).resolve().parent
_CKPT_DIR = _REPO / "checkpoints"
_DATA_CSV = _REPO / "url_with_headlines.csv"


def _get_domain(url: str) -> str:
    netloc = urlparse(str(url)).netloc.lower().strip()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _ensure_figures_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_class_balance(df: pd.DataFrame, out_dir: str) -> None:
    domains = df["url"].astype(str).map(_get_domain)
    label_names = domains.map({"foxnews.com": "Fox", "nbcnews.com": "NBC"})
    counts = label_names.value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.tolist(), counts.values.tolist(), color=["#d62728", "#1f77b4"])
    plt.title("Class Distribution in Dataset")
    plt.xlabel("News Source")
    plt.ylabel("Number of Headlines")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=200)
    plt.close()


def plot_headline_lengths(df: pd.DataFrame, out_dir: str) -> None:
    domains = df["url"].astype(str).map(_get_domain)
    cleaned = df["headline"].astype(str).map(clean_text)
    lengths = cleaned.map(lambda s: len(s.split()))

    fox = lengths[domains == "foxnews.com"]
    nbc = lengths[domains == "nbcnews.com"]

    bins = np.arange(0, max(lengths.max(), 20) + 2) - 0.5
    plt.figure(figsize=(8, 4.5))
    plt.hist(fox, bins=bins, alpha=0.6, label="Fox", color="#d62728", density=True)
    plt.hist(nbc, bins=bins, alpha=0.6, label="NBC", color="#1f77b4", density=True)
    plt.title("Headline Length Distribution by Source")
    plt.xlabel("Headline Length (tokens)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "headline_length_distribution.png"), dpi=200)
    plt.close()


def plot_c_sweep(out_dir: str) -> None:
    c_values = [1, 2, 4, 8, 12, 16]
    val_acc = [0.783180, 0.800263, 0.805519, 0.804205, 0.804468, 0.804205]
    leaderboard_acc = [np.nan, np.nan, 0.8125, 0.8042, 0.79, np.nan]

    plt.figure(figsize=(7, 4.5))
    plt.plot(c_values, val_acc, marker="o", label="Validation Accuracy", color="#2ca02c")
    plt.plot(c_values, leaderboard_acc, marker="s", linestyle="--", label="Leaderboard Accuracy (tested C values)", color="#9467bd")
    plt.title("TF-IDF Regularization Sweep")
    plt.xlabel("C (Inverse Regularization Strength)")
    plt.ylabel("Accuracy")
    plt.ylim(0.75, 0.83)
    plt.xticks(c_values)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tfidf_c_sweep.png"), dpi=200)
    plt.close()


def plot_leaderboard_progression(out_dir: str) -> None:
    labels = [
        "MLP initial",
        "MLP + clean",
        "TF-IDF pre-clean",
        "TF-IDF + clean",
        "TF-IDF C=8",
        "TF-IDF C=4",
        "TF-IDF CV-best",
        "TF-IDF C=3",
        "Hash ensemble",
        "Word+char TF-IDF",
        "Hybrid (final)",
    ]
    scores = [0.7808, 0.799, 0.8008, 0.8125, 0.8042, 0.8125, 0.79, 0.8158, 0.813, 0.814, 0.8342]

    final_idx = len(scores) - 1
    colors = ["#4c78a8"] * len(scores)
    colors[final_idx] = "#2ca02c"

    plt.figure(figsize=(11, 4.8))
    x = np.arange(len(labels))
    bars = plt.bar(x, scores, color=colors)
    plt.title("Leaderboard Accuracy Across Submissions (final hybrid in green)")
    plt.xlabel("Submission Variant")
    plt.ylabel("Leaderboard Accuracy")
    plt.ylim(0.77, 0.85)
    plt.xticks(x, labels, rotation=25, ha="right")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.0006, f"{score:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "leaderboard_progression.png"), dpi=200)
    plt.close()


def plot_confusion_matrix_hybrid(out_dir: str, checkpoint_path: str) -> None:
    X, y = prepare_data(str(_DATA_CSV))
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = HybridModel()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    preds: List[int] = model.predict(X_val)
    cm = confusion_matrix(y_val, preds, labels=[0, 1])
    acc = float((np.asarray(preds) == np.asarray(y_val)).mean())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NBC (0)", "Fox (1)"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp.plot(cmap="Greens", ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Validation Confusion Matrix - Final Hybrid (acc={acc:.3f})")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "validation_confusion_matrix_final_model.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(out_dir: str, checkpoint_path: str) -> None:
    X, y = prepare_data(str(_DATA_CSV))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = Model()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    preds: List[int] = model.predict(X_val)
    cm = confusion_matrix(y_val, preds, labels=[0, 1])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NBC (0)", "Fox (1)"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.set_title("Validation Confusion Matrix (TF-IDF C=4)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "validation_confusion_matrix_tfidf_c4.png"), dpi=200)
    plt.close()


def _predict_probs(model: Model, texts: List[str]) -> np.ndarray:
    x = model._featurize_batch(texts)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def _build_vocab_from_full_data(texts: List[str], max_features: int = 50000, min_df: int = 2) -> dict:
    df_counter: Counter = Counter()
    for text in texts:
        uniq = set(str(text).split())
        for tok in uniq:
            df_counter[tok] += 1
    eligible = [(tok, df) for tok, df in df_counter.items() if df >= min_df]
    eligible.sort(key=lambda x: (-x[1], x[0]))
    selected = eligible[:max_features]
    return {tok: i for i, (tok, _) in enumerate(selected)}


def plot_top_weighted_tokens(out_dir: str, checkpoint_path: str, top_k: int = 15) -> None:
    """Word-only weights from the hybrid's word+char branch.

    The branch packs word features at indices 0..max_word_features-1 and exposes
    a word_hashes/word_indices lookup; we recover the original tokens by querying
    every distinct word from the corpus through the model.
    """
    df = pd.read_csv(str(_DATA_CSV))
    texts = df["headline"].astype(str).map(clean_text).tolist()

    model = WordCharModel()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    weights = model.linear.weight.detach().cpu().numpy().reshape(-1)

    token_set: set = set()
    for txt in texts:
        toks = str(txt).split()
        token_set.update(toks)

    token_scores = []
    for tok in token_set:
        idx = model._lookup_word_index(tok)
        if idx is not None and idx >= 0:
            token_scores.append((tok, float(weights[idx])))

    token_scores_sorted = sorted(token_scores, key=lambda t: t[1])
    top_nbc = token_scores_sorted[:top_k]            # most negative => NBC
    top_fox = token_scores_sorted[-top_k:][::-1]     # most positive => Fox

    fox_tokens, fox_vals = zip(*top_fox)
    nbc_tokens, nbc_vals = zip(*top_nbc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
    axes[0].barh(range(top_k), fox_vals[::-1], color="#d62728")
    axes[0].set_yticks(range(top_k), fox_tokens[::-1])
    axes[0].set_title("Top Tokens Indicative of Fox (Label 1)")
    axes[0].set_xlabel("Logistic Weight")
    axes[0].set_ylabel("Token")

    axes[1].barh(range(top_k), nbc_vals[::-1], color="#1f77b4")
    axes[1].set_yticks(range(top_k), nbc_tokens[::-1])
    axes[1].set_title("Top Tokens Indicative of NBC (Label 0)")
    axes[1].set_xlabel("Logistic Weight")
    axes[1].set_ylabel("Token")

    # Use a shared x-axis scale for direct magnitude comparison.
    all_abs = [abs(v) for v in fox_vals] + [abs(v) for v in nbc_vals]
    max_abs = max(all_abs) if all_abs else 1.0
    axes[0].set_xlim(0, max_abs * 1.05)
    axes[1].set_xlim(-max_abs * 1.05, 0)

    fig.suptitle("Top Weighted Words from Word+Char TF-IDF Branch (used in final hybrid)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_weighted_tokens_word_char.png"), dpi=200)
    plt.close()


def plot_top_weighted_tokens_hash(out_dir: str, checkpoint_path: str, top_k: int = 15) -> None:
    df = pd.read_csv(str(_DATA_CSV))
    texts = df["headline"].astype(str).map(clean_text).tolist()

    model = Model()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    weights = model.linear.weight.detach().cpu().numpy().reshape(-1)

    token_set = set()
    for txt in texts:
        token_set.update(txt.split())

    token_scores = []
    for tok in token_set:
        idx = model._token_index(tok)
        token_scores.append((tok, float(weights[idx])))

    token_scores_sorted = sorted(token_scores, key=lambda t: t[1])
    top_nbc = token_scores_sorted[:top_k]
    top_fox = token_scores_sorted[-top_k:][::-1]

    fox_tokens, fox_vals = zip(*top_fox)
    nbc_tokens, nbc_vals = zip(*top_nbc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
    axes[0].barh(range(top_k), fox_vals[::-1], color="#d62728")
    axes[0].set_yticks(range(top_k), fox_tokens[::-1])
    axes[0].set_title("Top Tokens Indicative of Fox (Label 1)")
    axes[0].set_xlabel("Logistic Weight")
    axes[0].set_ylabel("Token")

    axes[1].barh(range(top_k), nbc_vals[::-1], color="#1f77b4")
    axes[1].set_yticks(range(top_k), nbc_tokens[::-1])
    axes[1].set_title("Top Tokens Indicative of NBC (Label 0)")
    axes[1].set_xlabel("Logistic Weight")
    axes[1].set_ylabel("Token")

    all_abs = [abs(v) for v in fox_vals] + [abs(v) for v in nbc_vals]
    max_abs = max(all_abs) if all_abs else 1.0
    axes[0].set_xlim(0, max_abs * 1.05)
    axes[1].set_xlim(-max_abs * 1.05, 0)

    fig.suptitle("Top Weighted Tokens from Hash TF-IDF Branch (used in final hybrid)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_weighted_tokens_hash.png"), dpi=200)
    plt.close()


def plot_all_misclassified_headlines(out_dir: str, checkpoint_path: str) -> None:
    df = pd.read_csv(str(_DATA_CSV))
    df = df.copy()
    df["clean_headline"] = df["headline"].astype(str).map(clean_text)
    df["domain"] = df["url"].astype(str).map(_get_domain)
    label_map = {"nbcnews.com": 0, "foxnews.com": 1}
    df = df[df["domain"].isin(label_map)].reset_index(drop=True)
    y = df["domain"].map(label_map).astype(int).tolist()

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        df["clean_headline"].tolist(),
        y,
        df.index.tolist(),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    _ = (X_train, y_train, idx_train)  # for clarity: only validation subset used below.

    model = Model()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    probs = _predict_probs(model, X_val)
    preds = (probs >= 0.5).astype(int)

    rows = []
    for val_i, (p, pred, true) in enumerate(zip(probs, preds, y_val)):
        if int(pred) != int(true):
            original_idx = idx_val[val_i]
            rows.append(
                {
                    "headline": df.loc[original_idx, "headline"],
                    "true_label": int(true),
                    "pred_label": int(pred),
                    "pred_prob_fox": float(p),
                }
            )

    if not rows:
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "No misclassified headlines on validation split.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "misclassified_headlines_colored.png"), dpi=200)
        plt.close()
        return

    rows = sorted(rows, key=lambda r: r["pred_prob_fox"], reverse=True)
    n = len(rows)
    fig_height = max(8, min(0.35 * n + 2, 60))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    ax.set_title(
        "All Misclassified Validation Headlines\n"
        "Color indicates true source: Red=Fox, Blue=NBC",
        pad=20,
    )

    y_pos = 1.0
    step = 1.0 / (n + 1)
    for i, row in enumerate(rows, start=1):
        color = "#d62728" if row["true_label"] == 1 else "#1f77b4"
        source = "Fox" if row["true_label"] == 1 else "NBC"
        txt = f"{i}. [{source}] {row['headline']}"
        ax.text(0.01, y_pos, txt, color=color, fontsize=8.5, va="top", wrap=True, transform=ax.transAxes)
        y_pos -= step

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "misclassified_headlines_colored.png"), dpi=220)
    plt.close()


def main() -> None:
    out_dir = "Figures"
    _ensure_figures_dir(out_dir)

    df = pd.read_csv(_DATA_CSV)
    plot_class_balance(df, out_dir)
    plot_headline_lengths(df, out_dir)
    plot_c_sweep(out_dir)
    plot_leaderboard_progression(out_dir)
    plot_confusion_matrix_hybrid(out_dir, checkpoint_path=str(_REPO / "final_model.pt"))
    plot_top_weighted_tokens(out_dir, checkpoint_path=str(_CKPT_DIR / "model_vocab_char_c4.pt"), top_k=15)
    plot_top_weighted_tokens_hash(out_dir, checkpoint_path=str(_CKPT_DIR / "model_tfidf_c3.pt"), top_k=15)
    plot_all_misclassified_headlines(out_dir, checkpoint_path=str(_CKPT_DIR / "model_tfidf_c3.pt"))
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
