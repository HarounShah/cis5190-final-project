import hashlib
from typing import Any, Iterable, List

import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    """
    .pt-only vocabulary TF-IDF + Logistic Regression model.
    Vocabulary lookup tables are stored inside the state_dict as tensors.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_features = 50000
        self.linear = nn.Linear(self.max_features, 1)

        # Buffers populated by checkpoint from training script.
        self.register_buffer("idf", torch.ones(self.max_features, dtype=torch.float32))
        self.register_buffer("token_hashes", torch.full((self.max_features,), 2**63 - 1, dtype=torch.int64))
        self.register_buffer("token_indices", torch.full((self.max_features,), -1, dtype=torch.int64))
        self.register_buffer("vocab_size", torch.tensor(0, dtype=torch.int64))

    def eval(self) -> None:
        super().eval()
        return self

    def _tokenize(self, text: str) -> List[str]:
        return str(text).split()

    def _stable_hash(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)

    def _lookup_index(self, token: str) -> int:
        n_vocab = int(self.vocab_size.item())
        if n_vocab <= 0:
            return -1
        hashes = self.token_hashes[:n_vocab].cpu().numpy()
        idxs = self.token_indices[:n_vocab].cpu().numpy()
        h = self._stable_hash(token)
        pos = int(np.searchsorted(hashes, h, side="left"))
        if 0 <= pos < n_vocab and int(hashes[pos]) == h:
            return int(idxs[pos])
        return -1

    def _featurize_batch(self, batch: List[str]) -> torch.Tensor:
        x = torch.zeros((len(batch), self.max_features), dtype=torch.float32)
        idf_np = self.idf.cpu().numpy()
        for i, text in enumerate(batch):
            counts = {}
            for tok in self._tokenize(text):
                idx = self._lookup_index(tok)
                if idx >= 0:
                    counts[idx] = counts.get(idx, 0) + 1
            if not counts:
                continue
            for idx, tf in counts.items():
                x[i, idx] = (1.0 + np.log(float(tf))) * float(idf_np[idx])
            norm = torch.linalg.norm(x[i]).item()
            if norm > 0:
                x[i] /= norm
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
