import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.lsh import RandomHyperplaneLSH


K_VALUES = [12, 16, 20]
L_VALUES = [40, 80, 120]
THRESHOLD = 0.8


def evaluate_config(prompts, queries, ground_truth, k, L):
    lsh = RandomHyperplaneLSH(
        dimension=prompts.shape[1],
        k=k,
        L=L,
        seed=42
    )

    start = time.perf_counter()
    lsh.build_index(prompts)
    build_time = time.perf_counter() - start

    recalls = []
    candidate_counts = []
    query_times = []

    for query_idx, query_vector in enumerate(queries):
        start = time.perf_counter()

        candidates = lsh.query(query_vector)
        candidate_ids = sorted(candidates)

        if candidate_ids:
            scores = prompts[candidate_ids] @ query_vector
            retrieved = {
                candidate_ids[i]
                for i, score in enumerate(scores)
                if score >= THRESHOLD
            }
        else:
            retrieved = set()

        elapsed = time.perf_counter() - start

        truth = set(ground_truth[query_idx])

        if truth:
            recalls.append(
                len(retrieved & truth) / len(truth)
            )

        candidate_counts.append(len(candidate_ids))
        query_times.append(elapsed)

    return {
        "k": k,
        "L": L,
        "mean_recall": np.mean(recalls),
        "candidate_ratio": np.mean(candidate_counts) / len(prompts),
        "avg_candidates": np.mean(candidate_counts),
        "query_time_ms": np.mean(query_times) * 1000,
        "build_time_s": build_time
    }


def main():
    prompts = np.load("data/prompt_embeddings.npy")
    queries = np.load("data/query_embeddings.npy")
    ground_truth = np.load(
        "data/ground_truth.npy",
        allow_pickle=True
    )

    results = []

    for k in K_VALUES:
        for L in L_VALUES:
            print(f"Evaluating k={k}, L={L}...", flush=True)

            result = evaluate_config(
                prompts, queries, ground_truth, k, L
            )

            results.append(result)

            print(
                f"Recall={result['mean_recall']:.2%}, "
                f"Candidates={result['candidate_ratio']:.2%}, "
                f"Time={result['query_time_ms']:.4f} ms"
            )

    df = pd.DataFrame(results)

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/parameter_study.csv", index=False)

    print("\n=== Parameter Study Summary ===")
    print(df.to_string(index=False))

    print("\nResults saved to results/parameter_study.csv")


if __name__ == "__main__":
    main()