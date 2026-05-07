import hashlib
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    """
    Final hybrid model:
    - Hash TF-IDF branch (2^16 bins)
    - Vocab+char TF-IDF branch
    - Weighted logit blending + tuned threshold
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer("hash_weight", torch.tensor(0.7, dtype=torch.float32))
        self.register_buffer("vocab_char_weight", torch.tensor(0.3, dtype=torch.float32))
        self.register_buffer("decision_threshold", torch.tensor(0.5, dtype=torch.float32))

        # Hash branch (larger space for fewer collisions)
        self.hash_num_features = 1 << 16
        self.register_buffer("hash_idf", torch.ones(self.hash_num_features, dtype=torch.float32))
        self.register_buffer("hash_w", torch.zeros(self.hash_num_features, dtype=torch.float32))
        self.register_buffer("hash_b", torch.tensor(0.0, dtype=torch.float32))

        # Vocab+char branch
        self.max_word_features = 40000
        self.max_char_features = 20000
        self.total_features = self.max_word_features + self.max_char_features
        self.char_ngram_min = 3
        self.char_ngram_max = 5
        self.register_buffer("vc_idf", torch.ones(self.total_features, dtype=torch.float32))
        self.register_buffer("vc_w", torch.zeros(self.total_features, dtype=torch.float32))
        self.register_buffer("vc_b", torch.tensor(0.0, dtype=torch.float32))

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

    def _hash_token_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") % self.hash_num_features

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

    def _hash_logit(self, text: str) -> float:
        counts: Dict[int, int] = {}
        for tok in str(text).lower().split():
            idx = self._hash_token_index(tok)
            counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return float(self.hash_b.item())

        idf = self.hash_idf.cpu().numpy()
        w = self.hash_w.cpu().numpy()
        b = float(self.hash_b.item())

        idxs, vals = [], []
        for idx, tf in counts.items():
            idxs.append(idx)
            vals.append((1.0 + np.log(float(tf))) * float(idf[idx]))
        vals_np = np.asarray(vals, dtype=np.float32)
        norm = float(np.linalg.norm(vals_np))
        if norm > 0:
            vals_np = vals_np / norm
        return float(np.dot(w[np.asarray(idxs, dtype=np.int64)], vals_np) + b)

    def _vocab_char_logit(self, text: str) -> float:
        counts: Dict[int, int] = {}
        for tok in self._tokenize_words(text):
            idx = self._lookup_word_index(tok)
            if idx >= 0:
                counts[idx] = counts.get(idx, 0) + 1
        for gram in self._char_ngrams(text):
            idx = self._lookup_char_index(gram)
            if idx >= 0:
                feat_idx = self.max_word_features + idx
                counts[feat_idx] = counts.get(feat_idx, 0) + 1
        if not counts:
            return float(self.vc_b.item())

        idf = self.vc_idf.cpu().numpy()
        w = self.vc_w.cpu().numpy()
        b = float(self.vc_b.item())

        idxs, vals = [], []
        for idx, tf in counts.items():
            idxs.append(idx)
            vals.append((1.0 + np.log(float(tf))) * float(idf[idx]))
        vals_np = np.asarray(vals, dtype=np.float32)
        norm = float(np.linalg.norm(vals_np))
        if norm > 0:
            vals_np = vals_np / norm
        return float(np.dot(w[np.asarray(idxs, dtype=np.int64)], vals_np) + b)

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        texts = [str(x) for x in batch]
        if not texts:
            return []
        a = float(self.hash_weight.item())
        b = float(self.vocab_char_weight.item())
        thr = float(self.decision_threshold.item())
        preds: List[int] = []
        for text in texts:
            l_hash = self._hash_logit(text)
            l_vc = self._vocab_char_logit(text)
            logit = a * l_hash + b * l_vc
            prob = 1.0 / (1.0 + np.exp(-logit))
            preds.append(1 if prob >= thr else 0)
        return preds


def get_model() -> Model:
    return Model()
