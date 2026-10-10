import hashlib
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


DATA = Path(__file__).resolve().parent.parent / "data"

MODES = (
    "dense",
    "bm25",
    "dense_rerank",
    "hybrid",
    "hybrid_rerank",
)

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:,\d{3})*(?:\.\d+)?")


def tokenize(text):
    """Normalize words and numbers for BM25 retrieval."""
    return [
        token.replace(",", "")
        for token in _TOKEN_RE.findall(text.lower())
    ]


class Retriever:
    def __init__(
        self,
        path=DATA / "chunks.json",
        dense_model="BAAI/bge-small-en-v1.5",
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        query_prefix=QUERY_PREFIX,
    ):
        with open(path, encoding="utf-8") as file:
            self.chunks = json.load(file)

        self.query_prefix = query_prefix

        if not self.chunks:
            raise ValueError(
                f"No chunks found in {path}. "
                "Run chunks.py to generate the corpus first."
            )

        texts = [
            chunk["text"][:2000]
            for chunk in self.chunks
        ]

        print("Loading embedding model...")
        self.emb = SentenceTransformer(dense_model)

        print("Loading reranking model...")
        self.rerank = CrossEncoder(rerank_model)

        print("Loading or creating embeddings...")
        vectors = self._embeddings(texts, dense_model)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(
            np.ascontiguousarray(vectors, dtype="float32")
        )

        print("Building BM25 index...")
        self.bm25 = BM25Okapi(
            [tokenize(text) for text in texts]
        )

        print("Retriever ready.")
        print("Total indexed chunks:", len(self.chunks))

    def _embeddings(self, texts, model_name):
        """Cache embeddings using a hash of the model and chunk texts."""
        digest = hashlib.sha256(
            (
                model_name
                + "\x00"
                + "\x00".join(texts)
            ).encode("utf-8")
        ).hexdigest()[:16]

        cache = DATA / "embeddings.npy"
        stamp = DATA / "embeddings.sha"

        if (
            cache.exists()
            and stamp.exists()
            and stamp.read_text(encoding="utf-8").strip() == digest
        ):
            vectors = np.load(cache)

            if (
                vectors.ndim == 2
                and vectors.shape[0] == len(texts)
            ):
                print("Using cached embeddings.")
                return vectors.astype("float32")

        vectors = self.emb.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        vectors = np.asarray(vectors, dtype="float32")

        np.save(cache, vectors)
        stamp.write_text(digest, encoding="utf-8")

        return vectors

    def _allowed(self, filters):
        """
        Return matching chunk indices.

        Example:
        filters={"company": "Intel", "year": "2015"}
        """
        if not filters:
            return None

        def matches(chunk):
            for key, value in filters.items():
                actual = chunk.get(key)

                if isinstance(value, (list, set, tuple)):
                    if actual not in value:
                        return False
                elif actual != value:
                    return False

            return True

        return {
            index
            for index, chunk in enumerate(self.chunks)
            if matches(chunk)
        }

    def _rerank(self, query, candidates):
        """Reorder candidates using the cross-encoder."""
        if not candidates:
            return []

        pairs = [
            (query, self.chunks[index]["text"][:1500])
            for index in candidates
        ]

        scores = self.rerank.predict(pairs)

        order = np.argsort(-np.asarray(scores))

        return [
            candidates[position]
            for position in order
        ]

    def search(
        self,
        query,
        k=5,
        pool=30,
        mode="hybrid_rerank",
        filters=None,
    ):
        """Retrieve the top-k chunks using the selected search mode."""
        if mode not in MODES:
            raise ValueError(
                f"Unknown mode {mode!r}; choose from {MODES}"
            )

        if not self.chunks:
            return []

        allowed = self._allowed(filters)

        pool = max(1, min(pool, len(self.chunks)))
        k = max(0, k)

        if k == 0:
            return []

        # Search the entire index when filtering, so relevant
        # allowed chunks are not missed by filtering only top results.
        search_k = (
            len(self.chunks)
            if allowed is not None
            else pool
        )

        # 1. Dense retrieval using FAISS.
        query_vector = self.emb.encode(
            [self.query_prefix + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        _scores, ids = self.index.search(
            query_vector,
            search_k,
        )

        dense = [
            int(index)
            for index in ids[0]
            if index >= 0
            and (
                allowed is None
                or int(index) in allowed
            )
        ][:pool]

        # 2. Sparse retrieval using BM25.
        bm25_scores = self.bm25.get_scores(
            tokenize(query)
        )

        sparse = [
            int(index)
            for index in np.argsort(-bm25_scores)
            if (
                allowed is None
                or int(index) in allowed
            )
        ][:pool]

        # 3. Select retrieval strategy.
        if mode == "dense":
            ranked = dense

        elif mode == "bm25":
            ranked = sparse

        elif mode == "dense_rerank":
            ranked = self._rerank(query, dense)

        else:
            # Reciprocal Rank Fusion combines dense and BM25 rankings.
            fused_scores = {}

            for ranking in (dense, sparse):
                for rank, index in enumerate(ranking):
                    fused_scores[index] = (
                        fused_scores.get(index, 0)
                        + 1 / (60 + rank)
                    )

            candidates = sorted(
                fused_scores,
                key=fused_scores.get,
                reverse=True,
            )[:pool]

            if mode == "hybrid":
                ranked = candidates
            else:
                ranked = self._rerank(
                    query,
                    candidates,
                )

        return [
            self.chunks[index]
            for index in ranked[:k]
        ]

