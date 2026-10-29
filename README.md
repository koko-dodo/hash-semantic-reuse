# Hash-based Semantic Reuse System

A hash-based semantic retrieval system for efficiently reusing historical LLM prompts.

## Overview

This project explores how to retrieve semantically similar historical prompts without performing an exhaustive search over the entire database.

The system combines:

- **Exact Hash:** Fast lookup for identical prompts.
- **Locality-Sensitive Hashing (LSH):** Candidate generation for semantically similar prompts.
- **Cosine Similarity:** Verification of retrieved candidates.

## Dataset

Quora Question Pairs (QQP).

## Methods

1. Brute-force Search
2. Exact Hash Only
3. Exact Hash + Random Hyperplane LSH

## Evaluation

The system will be evaluated using:

- Recall
- Candidate Ratio
- Query Cost
- Memory Cost

## Tech Stack

Python 3.12, NumPy, Pandas, Sentence Transformers, Matplotlib.

## Status

Work in progress — initial project setup.