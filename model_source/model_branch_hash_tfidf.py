import hashlib
from typing import Any, Iterable, List

import torch
from torch import nn


class Model(nn.Module):
    """
    .pt-compatible TF-IDF + Logistic Regression model.

    - Uses stable hashed TF-IDF features from headline text.
    - Applies a single linear layer (logistic regression).
    - Works with evaluator checkpoint loading via state_dict.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_features = 1 << 15  # 32,768
        self.linear = nn.Linear(self.num_features, 1)
        self.register_buffer("idf", torch.ones(self.num_features, dtype=torch.float32))

    def eval(self) -> None:
        super().eval()
        return self

    def _tokenize(self, text: str) -> List[str]:
        return str(text).lower().split()

    def _token_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") % self.num_features

    def _featurize_batch(self, batch: List[str]) -> torch.Tensor:
        # Sublinear TF + IDF weighting.
        x = torch.zeros((len(batch), self.num_features), dtype=torch.float32)
        for i, text in enumerate(batch):
            counts = {}
            for token in self._tokenize(text):
                idx = self._token_index(token)
                counts[idx] = counts.get(idx, 0) + 1
            for idx, tf in counts.items():
                x[i, idx] = 1.0 + torch.log(torch.tensor(float(tf))).item()

        x = x * self.idf.unsqueeze(0)
        norms = torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(1e-12)
        x = x / norms
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        texts = [str(x) for x in batch]
        if len(texts) == 0:
            return []

        x = self._featurize_batch(texts)
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(torch.int64)
        return preds.tolist()


def get_model() -> Model:
    return Model()
