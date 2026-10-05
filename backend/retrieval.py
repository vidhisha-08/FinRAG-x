import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(self, path="../data/chunks.json"):
        self.chunks = json.load(open(path))
        self.emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        texts = [c["text"][:2000] for c in self.chunks]
        cache = "../data/embeddings.npy"
        if os.path.exists(cache) and len(np.load(cache)) == len(texts):
            vecs = np.load(cache)
        else:
            vecs = self.emb.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            np.save(cache, vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(np.array(vecs, dtype="float32"))
        self.bm25 = BM25Okapi([t.lower().split() for t in texts])

    def search(self, query, k=5, pool=30, mode="hybrid_rerank"):
        pool = min(pool, len(self.chunks))
        qv = self.emb.encode([query], normalize_embeddings=True).astype("float32")
        _, ids = self.index.search(qv, pool)
        dense = [int(i) for i in ids[0] if i >= 0]
        scores = self.bm25.get_scores(query.lower().split())
        sparse = [int(i) for i in np.argsort(-scores)[:pool]]

        if mode == "dense":
            return [self.chunks[i] for i in dense[:k]]
        if mode == "bm25":
            if mode == "dense_rerank":
                ce = self.rerank.predict([(query, self.chunks[i]["text"][:1500]) for i in dense])
                return [self.chunks[dense[j]] for j in np.argsort(-ce)[:k]]
            return [self.chunks[i] for i in sparse[:k]]

        fused = {}
        for ranking in (dense, sparse):          # reciprocal rank fusion
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0) + 1 / (60 + r)
        cand = sorted(fused, key=fused.get, reverse=True)[:pool]
        if mode == "hybrid":
            return [self.chunks[i] for i in cand[:k]]

        ce = self.rerank.predict([(query, self.chunks[i]["text"][:1500]) for i in cand])
        return [self.chunks[cand[j]] for j in np.argsort(-ce)[:k]]