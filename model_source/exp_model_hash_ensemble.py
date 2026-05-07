import hashlib
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    """
    .pt-compatible ensemble of hashed TF-IDF logistic models.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_features = 1 << 15
        self.max_models = 16
        self.register_buffer("num_models", torch.tensor(1, dtype=torch.int64))
        self.register_buffer("model_weights", torch.zeros((self.max_models, self.num_features), dtype=torch.float32))
        self.register_buffer("model_bias", torch.zeros((self.max_models,), dtype=torch.float32))
        self.register_buffer("model_idf", torch.ones((self.max_models, self.num_features), dtype=torch.float32))

    def eval(self) -> None:
        super().eval()
        return self

    def _tokenize(self, text: str) -> List[str]:
        return str(text).lower().split()

    def _token_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") % self.num_features

    def _counts(self, text: str) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for tok in self._tokenize(text):
            idx = self._token_index(tok)
            counts[idx] = counts.get(idx, 0) + 1
        return counts

    def _logit_for_model(self, counts: Dict[int, int], m: int) -> float:
        idf = self.model_idf[m].cpu().numpy()
        w = self.model_weights[m].cpu().numpy()
        b = float(self.model_bias[m].item())

        vals = []
        idxs = []
        for idx, tf in counts.items():
            v = (1.0 + np.log(float(tf))) * float(idf[idx])
            vals.append(v)
            idxs.append(idx)
        if not vals:
            return b
        vals_np = np.asarray(vals, dtype=np.float32)
        norm = float(np.linalg.norm(vals_np))
        if norm > 0:
            vals_np = vals_np / norm
        dot = float(np.dot(w[np.asarray(idxs, dtype=np.int64)], vals_np))
        return dot + b

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        texts = [str(x) for x in batch]
        if not texts:
            return []

        n_models = int(self.num_models.item())
        preds: List[int] = []
        for text in texts:
            counts = self._counts(text)
            logits = [self._logit_for_model(counts, m) for m in range(n_models)]
            avg_logit = float(np.mean(logits))
            prob = 1.0 / (1.0 + np.exp(-avg_logit))
            preds.append(1 if prob >= 0.5 else 0)
        return preds


def get_model() -> Model:
    return Model()
