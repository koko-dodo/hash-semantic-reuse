"""Exact reuse first, followed by LSH candidate search on an exact miss."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.exact_hash import ExactHashIndex
from src.lsh import RandomHyperplaneLSH


@dataclass(frozen=True)
class RetrievalResult:
    status: Literal["exact", "semantic", "no_match"]
    prompt_ids: tuple[int, ...]
    # Only semantic results have cosine scores, aligned with prompt_ids.
    similarities: tuple[float, ...] | None
    candidate_count: int
    cosine_computations: int
    embedding_generated: bool
    lsh_used: bool


class SemanticReuseRetriever:
    """Retrieve historical row IDs using exact lookup, then semantic search.

    prompt_embeddings must follow the order of prompt_texts and use the same
    embedding model as queries. Vectors are normalized here for cosine scoring.
    An exact hit returns one reuse result, not all threshold neighbors.

    embed_query is an optional callable accepting one string and returning a
    one-dimensional vector. It is called only on an exact miss without a supplied
    query_embedding. The callable can lazily load/cache an embedding model, so
    this module never loads a model itself.
    """

    def __init__(
        self,
        prompt_texts: Sequence[str],
        prompt_embeddings: np.ndarray,
        *,
        embed_query: Callable[[str], np.ndarray] | None = None,
        threshold: float = 0.8,
        k: int = 16,
        L: int = 80,
        seed: int = 42,
    ):
        texts = list(prompt_texts)
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("prompt_texts must contain strings")
        if len(set(texts)) != len(texts):
            raise ValueError("prompt_texts must be unique")
        vectors = np.asarray(prompt_embeddings, dtype=np.float32)
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(texts)
            or not texts
            or vectors.shape[1] == 0
        ):
            raise ValueError("prompt_embeddings must have shape (N, dimension), N > 0")
        if not np.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("threshold must be finite and between -1 and 1")
        if not isinstance(k, int) or not isinstance(L, int) or k <= 0 or L <= 0:
            raise ValueError("k and L must be positive integers")

        self.prompt_embeddings = self._normalize(vectors)
        self.dimension = vectors.shape[1]
        self.threshold = threshold
        self.embed_query = embed_query
        self.exact = ExactHashIndex()
        self.exact.build_index(texts)
        self.lsh = RandomHyperplaneLSH(
            dimension=self.dimension, k=k, L=L, seed=seed
        )
        self.lsh.build_index(self.prompt_embeddings)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if not np.isfinite(vectors).all():
            raise ValueError("embeddings must contain only finite values")
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("embeddings must have finite, nonzero norms")
        return vectors / norms

    def search(
        self, query_text: str, query_embedding: np.ndarray | None = None
    ) -> RetrievalResult:
        """Return an exact reuse hit or all threshold matches within LSH candidates.

        IDs and semantic scores are ordered by historical row ID. Candidate and
        cosine counts describe semantic verification only; they are zero for an
        exact hit. Invalid/missing embeddings on an exact miss raise ValueError,
        rather than being reported as no_match.
        """
        if not isinstance(query_text, str):
            raise ValueError("query_text must be a string")
        exact_id = self.exact.search(query_text)
        if exact_id is not None:
            return RetrievalResult("exact", (exact_id,), None, 0, 0, False, False)

        embedding_generated = query_embedding is None
        if query_embedding is None:
            if self.embed_query is None:
                raise ValueError("an exact miss requires query_embedding or embed_query")
            query_embedding = self.embed_query(query_text)
        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise ValueError(f"query_embedding must have shape ({self.dimension},)")
        vector = self._normalize(vector)

        candidate_ids = sorted(self.lsh.query(vector))
        if candidate_ids:
            scores = self.prompt_embeddings[candidate_ids] @ vector
            matches = [
                (prompt_id, float(score))
                for prompt_id, score in zip(candidate_ids, scores)
                if score >= self.threshold
            ]
        else:
            matches = []

        return RetrievalResult(
            status="semantic" if matches else "no_match",
            prompt_ids=tuple(prompt_id for prompt_id, _ in matches),
            similarities=tuple(score for _, score in matches),
            candidate_count=len(candidate_ids),
            cosine_computations=len(candidate_ids),
            embedding_generated=embedding_generated,
            lsh_used=True,
        )
