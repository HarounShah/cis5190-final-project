import hashlib
from typing import Any, Iterable, List

import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    """
    .pt-only vocab word+char TF-IDF + Logistic Regression model.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_word_features = 40000
        self.max_char_features = 20000
        self.total_features = self.max_word_features + self.max_char_features
        self.char_ngram_min = 3
        self.char_ngram_max = 5

        self.linear = nn.Linear(self.total_features, 1)
        self.register_buffer("idf", torch.ones(self.total_features, dtype=torch.float32))

        self.register_buffer("word_hashes", torch.full((self.max_word_features,), 2**63 - 1, dtype=torch.int64))
        self.register_buffer("word_indices", torch.full((self.max_word_features,), -1, dtype=torch.int64))
        self.register_buffer("word_vocab_size", torch.tensor(0, dtype=torch.int64))

        self.register_buffer("char_hashes", torch.full((self.max_char_features,), 2**63 - 1, dtype=torch.int64))
        self.register_buffer("char_indices", torch.full((self.max_char_features,), -1, dtype=torch.int64))
        self.register_buffer("char_vocab_size", torch.tensor(0, dtype=torch.int64))

    def eval(self) -> None:
        super().eval()
        return self

    def _stable_hash(self, text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)

    def _tokenize_words(self, text: str) -> List[str]:
        toks = str(text).split()
        bigrams = [f"{toks[i]}__{toks[i+1]}" for i in range(len(toks) - 1)]
        return toks + bigrams

    def _char_ngrams(self, text: str) -> List[str]:
        s = str(text)
        grams: List[str] = []
        for n in range(self.char_ngram_min, self.char_ngram_max + 1):
            if len(s) < n:
                continue
            for i in range(len(s) - n + 1):
                grams.append(s[i : i + n])
        return grams

    def _lookup_word_index(self, token: str) -> int:
        n_vocab = int(self.word_vocab_size.item())
        if n_vocab <= 0:
            return -1
        h = self._stable_hash(token)
        hashes = self.word_hashes[:n_vocab].cpu().numpy()
        idxs = self.word_indices[:n_vocab].cpu().numpy()
        pos = int(np.searchsorted(hashes, h, side="left"))
        if 0 <= pos < n_vocab and int(hashes[pos]) == h:
            return int(idxs[pos])
        return -1

    def _lookup_char_index(self, gram: str) -> int:
        n_vocab = int(self.char_vocab_size.item())
        if n_vocab <= 0:
            return -1
        h = self._stable_hash(gram)
        hashes = self.char_hashes[:n_vocab].cpu().numpy()
        idxs = self.char_indices[:n_vocab].cpu().numpy()
        pos = int(np.searchsorted(hashes, h, side="left"))
        if 0 <= pos < n_vocab and int(hashes[pos]) == h:
            return int(idxs[pos])
        return -1

    def _featurize_batch(self, batch: List[str]) -> torch.Tensor:
        x = torch.zeros((len(batch), self.total_features), dtype=torch.float32)
        idf_np = self.idf.cpu().numpy()
        word_offset = 0
        char_offset = self.max_word_features

        for i, text in enumerate(batch):
            counts = {}
            for tok in self._tokenize_words(text):
                idx = self._lookup_word_index(tok)
                if idx >= 0:
                    feat_idx = word_offset + idx
                    counts[feat_idx] = counts.get(feat_idx, 0) + 1
            for gram in self._char_ngrams(text):
                idx = self._lookup_char_index(gram)
                if idx >= 0:
                    feat_idx = char_offset + idx
                    counts[feat_idx] = counts.get(feat_idx, 0) + 1

            if not counts:
                continue
            for feat_idx, tf in counts.items():
                x[i, feat_idx] = (1.0 + np.log(float(tf))) * float(idf_np[feat_idx])
            norm = torch.linalg.norm(x[i]).item()
            if norm > 0:
                x[i] /= norm
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        texts = [str(x) for x in batch]
        if not texts:
            return []
        x = self._featurize_batch(texts)
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(torch.int64)
        return preds.tolist()


def get_model() -> Model:
    return Model()
