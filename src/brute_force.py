import time
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLD = 0.8


def main():
    # Load embeddings
    prompt_embeddings = np.load("data/prompt_embeddings.npy")
    query_embeddings = np.load("data/query_embeddings.npy")

    prompts = pd.read_csv("data/prompts.csv")
    queries = pd.read_csv("data/queries.csv")

    # Validate vector dimensions
    assert prompt_embeddings.shape == (len(prompts), 384)
    assert query_embeddings.shape == (len(queries), 384)

    # Brute-force search
    start = time.perf_counter()

    similarities = query_embeddings @ prompt_embeddings.T

    elapsed = time.perf_counter() - start

    # Construct ground truth
    ground_truth = [
        np.where(row >= THRESHOLD)[0].tolist()
        for row in similarities
    ]

    # Evaluation statistics
    neighbor_counts = [len(neighbors) for neighbors in ground_truth]

    print("Threshold:", THRESHOLD)
    print("Total queries:", len(queries))
    print("Database size:", len(prompts))
    print("Total similarity computations:", similarities.size)
    print("Average ground truth size:", np.mean(neighbor_counts))
    print("Queries with no neighbors:", sum(count == 0 for count in neighbor_counts))
    print(f"Total brute-force time: {elapsed:.6f} seconds")
    print(f"Average query time: {elapsed / len(queries) * 1000:.4f} ms")

    # Save ground truth
    output_dir = Path("data")
    np.save(
        output_dir / "ground_truth.npy",
        np.array(ground_truth, dtype=object),
        allow_pickle=True,
    )

    # Display example
    query_idx = 0
    print("\nExample query:", queries.iloc[query_idx]["text"])

    for prompt_idx in ground_truth[query_idx][:5]:
        print(
            f"Similarity: {similarities[query_idx, prompt_idx]:.4f}",
            "|",
            prompts.iloc[prompt_idx]["text"],
        )


if __name__ == "__main__":
    main()