import gc
import tracemalloc

import numpy as np

from src.lsh import RandomHyperplaneLSH


CONFIGS = [
    (12, 40),
    (16, 40),
    (16, 80),
    (16, 120),
    (20, 120),
]


def main():
    embeddings = np.load("data/prompt_embeddings.npy")

    print("Database size:", len(embeddings))
    print(f"Embedding memory: {embeddings.nbytes / 1024**2:.2f} MiB")

    print("\n=== LSH Memory Study ===")

    for k, L in CONFIGS:
        gc.collect()

        tracemalloc.start()

        before, _ = tracemalloc.get_traced_memory()

        lsh = RandomHyperplaneLSH(
            dimension=embeddings.shape[1],
            k=k,
            L=L,
            seed=42,
        )

        lsh.build_index(embeddings)

        current, peak = tracemalloc.get_traced_memory()

        hyperplane_memory = lsh.hyperplanes.nbytes
        traced_memory = current - before

        print(f"\nk={k}, L={L}")
        print(f"Hyperplanes: {hyperplane_memory / 1024**2:.2f} MiB")
        print(f"Tracked index allocation: {traced_memory / 1024**2:.2f} MiB")
        print(f"Tracked peak allocation: {(peak - before) / 1024**2:.2f} MiB")
        print(f"Total bucket entries: {sum(len(ids) for table in lsh.tables for ids in table.values())}")

        del lsh
        tracemalloc.stop()


if __name__ == "__main__":
    main()