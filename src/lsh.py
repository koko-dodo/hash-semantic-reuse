from collections import defaultdict

import numpy as np


class RandomHyperplaneLSH:
    def __init__(self, dimension=384, k=16, L=120, seed=42):
        self.dimension = dimension
        self.k = k
        self.L = L

        rng = np.random.default_rng(seed)

        # L independent tables, k random hyperplanes each
        self.hyperplanes = rng.standard_normal(
            (L, k, dimension)
        ).astype(np.float32)

        self.tables = [defaultdict(list) for _ in range(L)]

    def hash_vector(self, vector, table_id):
        # Random projection
        projections = self.hyperplanes[table_id] @ vector

        # Convert signs into hash bits
        bits = projections >= 0

        # Pack k bits into an integer hash key
        hash_key = 0
        for bit in bits:
            hash_key = (hash_key << 1) | int(bit)

        return hash_key

    def build_index(self, embeddings):
        """Replace all bucket entries while preserving the existing hyperplanes."""
        for table in self.tables:
            table.clear()

        for table_id in range(self.L):
            table = self.tables[table_id]

            for prompt_id, vector in enumerate(embeddings):
                key = self.hash_vector(vector, table_id)
                table[key].append(prompt_id)

    def query(self, vector):
        candidates = set()

        for table_id in range(self.L):
            key = self.hash_vector(vector, table_id)
            candidates.update(
                self.tables[table_id].get(key, [])
            )

        return candidates

def main():
    import time

    prompts = np.load("data/prompt_embeddings.npy")
    queries = np.load("data/query_embeddings.npy")

    lsh = RandomHyperplaneLSH()

    start = time.perf_counter()
    lsh.build_index(prompts)
    build_time = time.perf_counter() - start

    query_vector = queries[0]

    start = time.perf_counter()
    candidates = lsh.query(query_vector)
    query_time = time.perf_counter() - start

    print("Database size:", len(prompts))
    print("LSH parameters: k =", lsh.k, ", L =", lsh.L)
    print("Total hyperplanes:", lsh.k * lsh.L)
    print(f"Index build time: {build_time:.4f} seconds")
    print("Candidate count:", len(candidates))
    print(f"Candidate ratio: {len(candidates) / len(prompts):.4%}")
    print(f"Candidate generation time: {query_time * 1000:.4f} ms")


if __name__ == "__main__":
    main()
