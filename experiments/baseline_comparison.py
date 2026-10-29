"""Compare retrieval with preloaded texts and precomputed query embeddings.

Timing includes each method's lookup, verification, and result construction.
Model loading, embedding generation, index construction, warm-up, correctness
checks, metric aggregation, and CSV output are outside the measured intervals.
The retriever's own input validation/normalization and metadata are included.
Exact reuse accuracy is reported separately from threshold-neighbor recall.
"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.exact_hash import ExactHashIndex
from src.retrieval import RetrievalResult, SemanticReuseRetriever


THRESHOLD = 0.8
K = 16
L = 80
SEED = 42
REPEATS = 5
EXACT_HIT_QUERIES = 100


@dataclass(frozen=True)
class QueryInput:
    text: str
    embedding: np.ndarray | None


def make_methods(prompts, exact, retriever):
    """All adapters take one already prepared input; no DataFrame access."""
    def brute_force(query):
        scores = prompts @ query.embedding
        return tuple(np.flatnonzero(scores >= THRESHOLD).tolist())

    def exact_only(query):
        prompt_id = exact.search(query.text)
        return () if prompt_id is None else (prompt_id,)

    def hash_lsh(query):
        return retriever.search(query.text, query_embedding=query.embedding)

    return {
        "Brute-force": brute_force,
        "Exact Hash Only": exact_only,
        "Hash + LSH": hash_lsh,
    }


def timing_statistics(times):
    """Pooled query latency statistics, in milliseconds (sample std, ddof=1)."""
    return {
        "mean_time_ms": float(np.mean(times)),
        "median_time_ms": float(np.median(times)),
        "std_time_ms": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        "p95_time_ms": float(np.percentile(times, 95)),
    }


def benchmark(methods, queries, repeats=REPEATS):
    """Warm every method on all queries, then rotate method order each round.

    Return pooled statistics and one retained row per method per round.
    std_time_ms describes pooled query variation; round_mean_std_ms describes
    variation between round means. Neither is a confidence interval.
    """
    if repeats < 5 or not queries or not methods:
        raise ValueError("benchmark requires methods, queries, and at least 5 rounds")
    names = list(methods)
    for method in methods.values():
        for query in queries:
            method(query)

    pooled = {name: [] for name in names}
    round_means = {name: [] for name in names}
    rounds = []
    for round_index in range(repeats):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        for position, name in enumerate(order, start=1):
            gc.collect()
            method = methods[name]
            times = []
            for query in queries:
                start = time.perf_counter_ns()
                method(query)
                elapsed = time.perf_counter_ns() - start
                times.append(elapsed / 1e6)
            stats = timing_statistics(times)
            rounds.append({
                "round": round_index + 1,
                "execution_order": position,
                "method": name,
                "query_count": len(queries),
                **stats,
            })
            pooled[name].extend(times)
            round_means[name].append(stats["mean_time_ms"])

    summary = {
        name: {
            **timing_statistics(pooled[name]),
            "round_mean_std_ms": float(np.std(round_means[name], ddof=1)),
            "repeats": repeats,
            "timed_query_count": len(pooled[name]),
        }
        for name in names
    }
    return summary, rounds


def evaluate_semantic(methods, queries, ground_truth, database_size):
    """Macro recall excludes empty GT; candidate counts include all queries."""
    if len(queries) != len(ground_truth) or not queries:
        raise ValueError("queries and ground_truth must have equal nonzero lengths")
    rows = []
    for name, method in methods.items():
        recalls = []
        total_true = total_hits = total_candidates = total_cosines = 0
        for query, neighbors in zip(queries, ground_truth):
            result = method(query)
            truth = set(neighbors)
            if name == "Hash + LSH":
                if (
                    not isinstance(result, RetrievalResult)
                    or result.status == "exact"
                    or not result.lsh_used
                    or result.embedding_generated
                ):
                    raise AssertionError("semantic workload must use precomputed exact-miss search")
                retrieved = set(result.prompt_ids)
                total_candidates += result.candidate_count
                total_cosines += result.cosine_computations
            else:
                retrieved = set(result)
                if name == "Brute-force":
                    total_candidates += database_size
                    total_cosines += database_size
                    if retrieved != truth:
                        raise AssertionError("brute-force differs from saved ground truth")
                elif retrieved:
                    raise AssertionError("semantic workload must have no exact hits")
            if not retrieved <= truth:
                raise AssertionError("retrieval returned a point below the ground-truth threshold")
            hits = len(retrieved & truth)
            if truth:
                recalls.append(hits / len(truth))
            total_hits += hits
            total_true += len(truth)
        rows.append({
            "method": name,
            "database_size": database_size,
            "query_count": len(queries),
            "valid_recall_queries": len(recalls),
            "mean_recall": float(np.mean(recalls)) if recalls else np.nan,
            "micro_recall": total_hits / total_true if total_true else np.nan,
            "avg_candidates": total_candidates / len(queries),
            "candidate_ratio": total_candidates / (len(queries) * database_size),
            "avg_cosine_computations": total_cosines / len(queries),
            "threshold": THRESHOLD,
            "k": K,
            "L": L,
            "seed": SEED,
        })
    return rows


def validate_exact_hits(exact, retriever, known_texts):
    """Check row IDs and prove no encoder or LSH call occurs, outside timing."""
    with (
        patch.object(retriever, "embed_query", side_effect=AssertionError("unexpected encoding")) as encoder,
        patch.object(retriever.lsh, "query", side_effect=AssertionError("unexpected LSH")) as lookup,
    ):
        for expected_id, text in enumerate(known_texts):
            if exact.search(text) != expected_id:
                raise AssertionError("exact hash returned an incorrect row ID")
            result = retriever.search(text)
            if (
                result.status != "exact"
                or result.prompt_ids != (expected_id,)
                or result.similarities is not None
                or result.lsh_used
                or result.embedding_generated
                or result.candidate_count != 0
                or result.cosine_computations != 0
            ):
                raise AssertionError("exact reuse did not short-circuit correctly")
        encoder.assert_not_called()
        lookup.assert_not_called()
    return len(known_texts)


def main():
    prompt_texts = pd.read_csv("data/prompts.csv")["text"].tolist()
    query_texts = pd.read_csv("data/queries.csv")["text"].tolist()
    prompts = np.load("data/prompt_embeddings.npy")
    query_vectors = np.load("data/query_embeddings.npy")
    ground_truth = np.load("data/ground_truth.npy", allow_pickle=True)
    if (
        len(prompt_texts) != 10000
        or len(query_texts) != 200
        or prompts.shape != (10000, 384)
        or query_vectors.shape != (200, 384)
        or len(ground_truth) != 200
    ):
        raise ValueError("this benchmark requires N=10000, Q=200, dimension=384")
    if not set(prompt_texts).isdisjoint(query_texts):
        raise ValueError("semantic queries must have no exact text overlap")
    for vectors in (prompts, query_vectors):
        if not np.isfinite(vectors).all() or not np.allclose(
            np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6, rtol=0
        ):
            raise ValueError("benchmark embeddings must be finite and normalized")

    # All text extraction, array row access, and index construction are untimed.
    queries = [QueryInput(text, vector) for text, vector in zip(query_texts, query_vectors)]
    exact = ExactHashIndex()
    exact.build_index(prompt_texts)
    retriever = SemanticReuseRetriever(
        prompt_texts, prompts, threshold=THRESHOLD, k=K, L=L, seed=SEED
    )
    methods = make_methods(prompts, exact, retriever)
    semantic_rows = evaluate_semantic(methods, queries, ground_truth, len(prompts))
    if not np.isclose(semantic_rows[2]["mean_recall"], 0.9794247178640243, atol=1e-10, rtol=0):
        raise AssertionError("hybrid macro recall differs from the reviewed baseline")

    known_texts = prompt_texts[:EXACT_HIT_QUERIES]
    known_hits = validate_exact_hits(exact, retriever, known_texts)
    semantic_timing, semantic_rounds = benchmark(methods, queries)

    # Exact reuse is a separate workload, with no embedding supplied or generated.
    exact_methods = {name: methods[name] for name in ("Exact Hash Only", "Hash + LSH")}
    exact_queries = [QueryInput(text, None) for text in known_texts]
    exact_timing, exact_rounds = benchmark(exact_methods, exact_queries)
    semantic_df = pd.DataFrame([
        {**row, **semantic_timing[row["method"]]} for row in semantic_rows
    ])
    exact_df = pd.DataFrame([
        {
            "method": name,
            "query_count": known_hits,
            "correct_exact_hits": known_hits,
            "exact_hit_accuracy": 1.0,
            "avg_candidates": 0.0,
            "candidate_ratio": 0.0,
            "avg_cosine_computations": 0.0,
            "embedding_calls": 0,
            "lsh_calls": 0,
            **exact_timing[name],
        }
        for name in exact_methods
    ])
    rounds_df = pd.DataFrame(
        [{"workload": "semantic", **row} for row in semantic_rounds]
        + [{"workload": "exact_hit", **row} for row in exact_rounds]
    )

    print("Timing: preloaded input -> lookup/verification/result construction.")
    print("Excludes model loading, embedding generation, index build and warm-up.")
    print("All methods: one full warm-up pass; 5 rounds; rotating method order.")
    print("std_time_ms: pooled query sample std; round_mean_std_ms: std of round means.")
    print("Semantic exact hits: 0 / 200; verified exact reuse IDs:", known_hits, "/ 100")
    print("\n=== Semantic queries (mean_recall = Macro Recall on nonempty GT) ===")
    print(semantic_df.to_string(index=False))
    print("\n=== Exact reuse queries (accuracy, not semantic-neighbor recall) ===")
    print(exact_df.to_string(index=False))
    print("\n=== Per-round measurements ===")
    print(rounds_df.to_string(index=False))

    results = Path("results")
    results.mkdir(exist_ok=True)
    semantic_df.to_csv(results / "baseline_comparison.csv", index=False)
    exact_df.to_csv(results / "baseline_exact_hits.csv", index=False)
    rounds_df.to_csv(results / "baseline_rounds.csv", index=False)
    print("\nSaved baseline_comparison.csv, baseline_exact_hits.csv, baseline_rounds.csv; PNGs unchanged.")


if __name__ == "__main__":
    main()
