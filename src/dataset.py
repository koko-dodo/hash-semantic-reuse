from datasets import load_dataset
import random
import pandas as pd

dataset = load_dataset(
    "sentence-transformers/quora-duplicates",
    "pair-class",
    split="train"
)

# Keep valid duplicate pairs
duplicate_pairs = [
    (row["sentence1"].strip(), row["sentence2"].strip())
    for row in dataset
    if row["label"] == 1
    and row["sentence1"]
    and row["sentence2"]
    and row["sentence1"].strip()
    and row["sentence2"].strip()
]

# Reproducible random sampling
rng = random.Random(42)
rng.shuffle(duplicate_pairs)

prompts = []
queries = []
used_questions = set()

for q1, q2 in duplicate_pairs:
    if q1 in used_questions or q2 in used_questions:
        continue

    prompts.append(q1)
    queries.append(q2)

    used_questions.add(q1)
    used_questions.add(q2)

    if len(queries) == 200:
        break

# Collect all valid questions
all_questions = []

for row in dataset:
    for field in ("sentence1", "sentence2"):
        question = row[field]

        if question and question.strip():
            all_questions.append(question.strip())

# Shuffle to avoid selection bias
rng.shuffle(all_questions)

# Fill historical prompt database
for question in all_questions:
    if question in used_questions:
        continue

    prompts.append(question)
    used_questions.add(question)

    if len(prompts) == 10000:
        break

# Validate dataset
assert len(prompts) == 10000
assert len(queries) == 200
assert len(set(prompts)) == 10000
assert len(set(queries)) == 200
assert set(prompts).isdisjoint(set(queries))

print("Final prompts:", len(prompts))
print("Final queries:", len(queries))
print("Total unique questions:", len(used_questions))

# Save processed datasets
prompts_df = pd.DataFrame({
    "prompt_id": range(len(prompts)),
    "text": prompts
})

queries_df = pd.DataFrame({
    "query_id": range(len(queries)),
    "text": queries
})

prompts_df.to_csv("data/prompts.csv", index=False)
queries_df.to_csv("data/queries.csv", index=False)

print("Saved data/prompts.csv")
print("Saved data/queries.csv")