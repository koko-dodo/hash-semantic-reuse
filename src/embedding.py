import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    prompts = pd.read_csv("data/prompts.csv")["text"].tolist()
    queries = pd.read_csv("data/queries.csv")["text"].tolist()

    print(f"Loaded {len(prompts)} prompts and {len(queries)} queries")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    start = time.perf_counter()

    prompt_embeddings = model.encode(
        prompts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embeddings = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = time.perf_counter() - start

    output_dir = Path("data")
    np.save(output_dir / "prompt_embeddings.npy", prompt_embeddings)
    np.save(output_dir / "query_embeddings.npy", query_embeddings)

    print("Prompt embeddings shape:", prompt_embeddings.shape)
    print("Query embeddings shape:", query_embeddings.shape)
    print("Embedding dtype:", prompt_embeddings.dtype)
    print(f"Embedding generation time: {elapsed:.2f} seconds")
    print("Embeddings saved successfully")


if __name__ == "__main__":
    main()