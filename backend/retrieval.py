import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

DATA = Path(__file__).resolve().parent.parent / "data"      # works from any working directory
MODES = ("dense", "bm25", "dense_rerank", "hybrid", "hybrid_rerank")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # BGE query instruction
_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:,\d{3})*(?:\.\d+)?")


def tokenize(text):
    """'2015,' and '2015' are now the same token; '1,234.5' becomes '1234.5'."""
    return [t.replace(",", "") for t in _TOKEN_RE.findall(text.lower())]


class Retriever:
    def __init__(self, path=DATA / "chunks.json", dense_model="BAAI/bge-small-en-v1.5",
                 rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2", query_prefix=QUERY_PREFIX):
        with open(path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.query_prefix = query_prefix
        texts = [c["text"][:2000] for c in self.chunks]
        self.emb = SentenceTransformer(dense_model)
        self.rerank = CrossEncoder(rerank_model)
        vecs = self._embeddings(texts, dense_model)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(np.ascontiguousarray(vecs, dtype="float32"))
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def _embeddings(self, texts, model_name):
        # Cache is keyed on a hash of model + texts. The old check (same row count) silently
        # reused stale vectors whenever chunks.json changed but kept the same length.
        digest = hashlib.sha256((model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
        cache, stamp = DATA / "embeddings.npy", DATA / "embeddings.sha"
        if cache.exists() and stamp.exists() and stamp.read_text().strip() == digest:
            return np.load(cache)
        vecs = self.emb.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        np.save(cache, vecs)
        stamp.write_text(digest)
        return vecs

    def _allowed(self, filters):
        """filters={'company': 'Intel', 'year': 2015} -> set of chunk indices (None = no filter).
        Needs those keys on your chunks (add them in chunks.py / ingestion)."""
        if not filters:
            return None
        def ok(c):
            return all((c.get(k) in v) if isinstance(v, (list, set, tuple)) else c.get(k) == v
                       for k, v in filters.items())
        return {i for i, c in enumerate(self.chunks) if ok(c)}

    def _rerank(self, query, cand):
        if not cand:
            return []
        ce = self.rerank.predict([(query, self.chunks[i]["text"][:1500]) for i in cand])
        return [cand[j] for j in np.argsort(-ce)]

    def search(self, query, k=5, pool=30, mode="hybrid_rerank", filters=None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        allowed = self._allowed(filters)
        pool = min(pool, len(self.chunks))

        n_dense = len(self.chunks) if allowed is not None else pool
        qv = self.emb.encode([self.query_prefix + query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        dense = [int(i) for i in ids[0] if i >= 0 and (allowed is None or int(i) in allowed)][:pool]

        scores = self.bm25.get_scores(tokenize(query))
        sparse = [int(i) for i in np.argsort(-scores) if allowed is None or int(i) in allowed][:pool]

        if mode == "dense":
            ranked = dense
        elif mode == "bm25":
            ranked = sparse
        elif mode == "dense_rerank":
            # Bug fix: this branch was nested inside `if mode == "bm25"` and could never run,
            # so "dense_rerank" silently fell through to hybrid_rerank in your ablation.
            ranked = self._rerank(query, dense)
        else:
            fused = {}
            for ranking in (dense, sparse):                 # reciprocal rank fusion
                for r, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0) + 1 / (60 + r)
            cand = sorted(fused, key=fused.get, reverse=True)[:pool]
            ranked = cand if mode == "hybrid" else self._rerank(query, cand)
        return [self.chunks[i] for i in ranked[:k]]