import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.lsh import RandomHyperplaneLSH


DATABASE_SIZES = [2000, 5000, 10000]

K = 16
L = 80
THRESHOLD = 0.8


def evaluate_size(prompts, queries):
    n = len(prompts)

    # Recompute ground truth for this database size
    similarities = queries @ prompts.T

    ground_truth = [
        set(np.where(row >= THRESHOLD)[0].tolist())
        for row in similarities
    ]

    # Build LSH index
    lsh = RandomHyperplaneLSH(
        dimension=prompts.shape[1],
        k=K,
        L=L,
        seed=42
    )

    start = time.perf_counter()
    lsh.build_index(prompts)
    build_time = time.perf_counter() - start

    candidate_counts = []
    recalls = []
    query_times = []
    false_candidate_counts = []

    for i, query in enumerate(queries):
        start = time.perf_counter()

        candidates = lsh.query(query)
        candidate_ids = sorted(candidates)

        if candidate_ids:
            scores = prompts[candidate_ids] @ query

            retrieved = {
                candidate_ids[j]
                for j, score in enumerate(scores)
                if score >= THRESHOLD
            }
        else:
            retrieved = set()

        elapsed = time.perf_counter() - start

        truth = ground_truth[i]

        if truth:
            recalls.append(
                len(retrieved & truth) / len(truth)
            )

        candidate_counts.append(len(candidate_ids))
        false_candidate_counts.append(
            len(candidates - truth)
        )
        query_times.append(elapsed)

    return {
        "database_size": n,
        "k": K,
        "L": L,
        "valid_recall_queries": len(recalls),
        "mean_recall": np.mean(recalls) if recalls else np.nan,
        "avg_candidates": np.mean(candidate_counts),
        "candidate_ratio": np.mean(candidate_counts) / n,
        "avg_false_candidates": np.mean(false_candidate_counts),
        "query_time_ms": np.mean(query_times) * 1000,
        "build_time_s": build_time
    }


def main():
    all_prompts = np.load("data/prompt_embeddings.npy")
    queries = np.load("data/query_embeddings.npy")

    results = []

    for n in DATABASE_SIZES:
        print(f"\nEvaluating N={n}...", flush=True)

        prompts = all_prompts[:n]

        result = evaluate_size(prompts, queries)
        results.append(result)

        print(f"Valid recall queries: {result['valid_recall_queries']}")
        print(f"Mean Recall: {result['mean_recall']:.2%}")
        print(f"Average Candidates: {result['avg_candidates']:.2f}")
        print(f"Candidate Ratio: {result['candidate_ratio']:.4%}")
        print(f"Average False Candidates: {result['avg_false_candidates']:.2f}")
        print(f"Query Time: {result['query_time_ms']:.4f} ms")
        print(f"Build Time: {result['build_time_s']:.4f} s")

    df = pd.DataFrame(results)

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/scalability.csv", index=False)

    print("\n=== Scalability Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()