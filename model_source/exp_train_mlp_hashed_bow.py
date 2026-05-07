import argparse
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocess import prepare_data

from .exp_model_mlp_hashed_bow import Model
from .paths import CHECKPOINTS, REPO_ROOT


def stratified_split_indices(y: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class = {}
    for idx, label in enumerate(y):
        by_class.setdefault(int(label), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for _, indices in by_class.items():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def make_loader(model: Model, X: List[str], y: List[int], indices: List[int], batch_size: int, shuffle: bool) -> DataLoader:
    texts = [X[i] for i in indices]
    labels = torch.tensor([y[i] for i in indices], dtype=torch.float32)
    feats = model._featurize_batch(texts)
    ds = TensorDataset(feats, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: Model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train headline classifier model.")
    parser.add_argument("--csv", default=str(REPO_ROOT / "url_with_headlines.csv"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default=str(CHECKPOINTS / "model.pt"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    X, y = prepare_data(args.csv)
    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_idx, val_idx = stratified_split_indices(y, val_ratio=args.val_ratio, seed=args.seed)
    train_loader = make_loader(model, X, y, train_idx, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(model, X, y, val_idx, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_examples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = yb.size(0)
            running_loss += loss.item() * batch_size
            n_examples += batch_size

        train_loss = running_loss / max(n_examples, 1)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")

    torch.save(best_state, args.save_path)
    print(f"saved best checkpoint to {args.save_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
