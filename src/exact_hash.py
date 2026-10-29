import time

import pandas as pd


class ExactHashIndex:
    def __init__(self):
        self.table = {}

    def build_index(self, prompts):
        for prompt_id, text in enumerate(prompts):
            self.table[text] = prompt_id

    def search(self, query):
        return self.table.get(query, None)


def main():
    prompts = pd.read_csv("data/prompts.csv")["text"].tolist()
    queries = pd.read_csv("data/queries.csv")["text"].tolist()

    index = ExactHashIndex()

    start = time.perf_counter()
    index.build_index(prompts)
    build_time = time.perf_counter() - start

    # Test 1: Known exact match
    exact_query = prompts[0]
    exact_result = index.search(exact_query)

    # Test 2: Query from semantic test set
    semantic_query = queries[0]
    semantic_result = index.search(semantic_query)

    print("Hash table entries:", len(index.table))
    print(f"Index build time: {build_time:.6f} seconds")

    print("\nExact match test:")
    print("Query:", exact_query)
    print("Result:", exact_result)

    print("\nSemantic query test:")
    print("Query:", semantic_query)
    print("Result:", semantic_result)


if __name__ == "__main__":
    main()