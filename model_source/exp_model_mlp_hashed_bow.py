import hashlib
import torch
from torch import nn
from typing import Any, Iterable, List


class Model(nn.Module):
    """
    Template model for the leaderboard.

    Requirements:
    - Must be instantiable with no arguments (called by the evaluator).
    - Must implement `predict(batch)` which receives an iterable of inputs and
      returns a list of predictions (labels).
    - Must implement `eval()` to place the model in evaluation mode.
    - If you use PyTorch, submit a state_dict to be loaded via `load_state_dict`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Fixed-dimensional hashed BoW features from headline text.
        self.num_features = 1 << 14  # 16,384
        self.hidden_dim = 128
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_dim, 1),
        )
        self._is_eval = False

    def eval(self) -> None:
        self._is_eval = True
        super().eval()
        return self

    def _tokenize(self, text: str) -> List[str]:
        return str(text).lower().split()

    def _featurize_batch(self, batch: List[str]) -> torch.Tensor:
        x = torch.zeros((len(batch), self.num_features), dtype=torch.float32)
        for i, text in enumerate(batch):
            for tok in self._tokenize(text):
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest, byteorder="little") % self.num_features
                x[i, idx] += 1.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Implement your inference here.
        Inputs:
            batch: Iterable of preprocessed inputs (as produced by your preprocess.py)
        Returns:
            A list of predictions with the same length as `batch`.
        """
        batch_list = [str(x) for x in batch]
        if len(batch_list) == 0:
            return []

        x = self._featurize_batch(batch_list)
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(torch.int64)
        return preds.tolist()


def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()


