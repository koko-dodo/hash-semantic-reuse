import gc
import time

import numpy as np
import pandas as pd

from src.exact_hash import ExactHashIndex
from src.lsh import RandomHyperplaneLSH


THRESHOLD = 0.8
REPEATS = 5


def benchmark(method, queries):
    times = []

    for _ in range(REPEATS):
        for query in queries:
            start = time.perf_counter_ns()
            method(query)
            times.append((time.perf_counter_ns() - start) / 1e6)

    return np.mean(times), np.median(times)


def main():
    prompt_df = pd.read_csv("data/prompts.csv")
    query_df = pd.read_csv("data/queries.csv")

    prompts = np.load("data/prompt_embeddings.npy")
    queries = np.load("data/query_embeddings.npy")

    ground_truth = np.load(
        "data/ground_truth.npy",
        allow_pickle=True
    )

    exact = ExactHashIndex()
    exact.build_index(prompt_df["text"].tolist())

    lsh = RandomHyperplaneLSH(
        dimension=prompts.shape[1],
        k=16,
        L=80,
        seed=42
    )
    lsh.build_index(prompts)

    def brute_force(query):
        scores = prompts @ query
        return set(np.flatnonzero(scores >= THRESHOLD).tolist())

    def exact_only(text):
        result = exact.search(text)
        return set() if result is None else {result}

    def hash_lsh(query):
        candidates = sorted(lsh.query(query))

        if not candidates:
            return set()

        scores = prompts[candidates] @ query

        return {
            prompt_id
            for prompt_id, score in zip(candidates, scores)
            if score >= THRESHOLD
        }

    # Check retrieval correctness
    for i, query in enumerate(queries):
        truth = set(ground_truth[i])
        assert brute_force(query) == truth

    exact_hits = sum(
        exact.search(text) is not None
        for text in query_df["text"]
    )

    known_exact_hits = sum(
        exact.search(text) is not None
        for text in prompt_df["text"].head(100)
    )

    print("Semantic queries with exact hits:", exact_hits)
    print("Known exact matches:", known_exact_hits, "/ 100")

    gc.collect()

    methods = {
        "Brute-force": lambda i: brute_force(queries[i]),
        "Exact Hash Only": lambda i: exact_only(
            query_df.iloc[i]["text"]
        ),
        "Hash + LSH": lambda i: hash_lsh(queries[i]),
    }

    results = []

    for name, method in methods.items():
        mean_ms, median_ms = benchmark(
            method,
            range(len(queries))
        )

        recalls = []
        total_candidates = 0

        for i in range(len(queries)):
            retrieved = method(i)
            truth = set(ground_truth[i])

            if truth:
                recalls.append(
                    len(retrieved & truth) / len(truth)
                )

            if name == "Brute-force":
                total_candidates += len(prompts)
            elif name == "Hash + LSH":
                total_candidates += len(lsh.query(queries[i]))

        if name == "Brute-force":
            avg_computations = len(prompts)
        elif name == "Exact Hash Only":
            avg_computations = 0
        else:
            avg_computations = total_candidates / len(queries)

        results.append({
            "method": name,
            "mean_recall": np.mean(recalls),
            "avg_cosine_computations": avg_computations,
            "mean_time_ms": mean_ms,
            "median_time_ms": median_ms,
        })

    df = pd.DataFrame(results)
    print("\n=== Baseline Comparison ===")
    print(df.to_string(index=False))

    df.to_csv("results/baseline_comparison.csv", index=False)


if __name__ == "__main__":
    main()