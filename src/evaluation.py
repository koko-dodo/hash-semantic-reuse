import time

import numpy as np

from src.lsh import RandomHyperplaneLSH


THRESHOLD = 0.8


def main():
    prompts = np.load("data/prompt_embeddings.npy")
    queries = np.load("data/query_embeddings.npy")
    ground_truth = np.load(
        "data/ground_truth.npy",
        allow_pickle=True
    )

    assert len(prompts) == 10000
    assert len(queries) == len(ground_truth) == 200

    # Build LSH index
    lsh = RandomHyperplaneLSH(k=16, L=120, seed=42)

    start = time.perf_counter()
    lsh.build_index(prompts)
    build_time = time.perf_counter() - start

    recalls = []
    candidate_ratios = []
    candidate_counts = []
    query_times = []
    verification_times = []

    total_true_neighbors = 0
    total_retrieved_neighbors = 0

    for query_idx, query_vector in enumerate(queries):
        # Step 1: Generate candidates
        start = time.perf_counter()
        candidates = lsh.query(query_vector)
        query_time = time.perf_counter() - start

        # Step 2: Verify with true cosine similarity
        candidate_ids = sorted(candidates)

        start = time.perf_counter()

        if candidate_ids:
            scores = prompts[candidate_ids] @ query_vector

            retrieved = {
                candidate_ids[i]
                for i, score in enumerate(scores)
                if score >= THRESHOLD
            }
        else:
            retrieved = set()

        verification_time = time.perf_counter() - start

        # Step 3: Compare with ground truth
        truth = set(ground_truth[query_idx])

        if truth:
            recall = len(retrieved & truth) / len(truth)
            recalls.append(recall)

        total_true_neighbors += len(truth)
        total_retrieved_neighbors += len(retrieved & truth)

        candidate_counts.append(len(candidates))
        candidate_ratios.append(len(candidates) / len(prompts))
        query_times.append(query_time)
        verification_times.append(verification_time)

    # Summary
    print("LSH Parameters: k =", lsh.k, ", L =", lsh.L)
    print("Database size:", len(prompts))
    print("Total queries:", len(queries))
    print("Valid recall queries:", len(recalls))

    print(f"\nIndex build time: {build_time:.4f} s")

    print(f"Mean Recall: {np.mean(recalls):.4%}")
    print(
        f"Micro Recall: "
        f"{total_retrieved_neighbors / total_true_neighbors:.4%}"
    )

    print(f"Average candidate count: {np.mean(candidate_counts):.2f}")
    print(f"Average candidate ratio: {np.mean(candidate_ratios):.4%}")

    print(
        f"Average LSH lookup time: "
        f"{np.mean(query_times) * 1000:.4f} ms"
    )
    print(
        f"Average verification time: "
        f"{np.mean(verification_times) * 1000:.4f} ms"
    )
    print(
        f"Average total query time: "
        f"{(np.mean(query_times) + np.mean(verification_times)) * 1000:.4f} ms"
    )

    print(
        "Total cosine similarity computations:",
        sum(candidate_counts)
    )
    print(
        "Brute-force similarity computations:",
        len(prompts) * len(queries)
    )


if __name__ == "__main__":
    main()