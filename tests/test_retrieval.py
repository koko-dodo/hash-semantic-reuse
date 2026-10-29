import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.retrieval import SemanticReuseRetriever


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.embed_query = Mock(return_value=np.array([2.0, 0.0]))
        self.retriever = SemanticReuseRetriever(
            ["first prompt", "second prompt", "opposite prompt"],
            np.array([[2.0, 0.0], [0.0, 3.0], [-4.0, 0.0]]),
            embed_query=self.embed_query,
            k=4,
            L=3,
        )

    def test_exact_hits_return_correct_ids_and_skip_embedding_and_lsh(self):
        with patch.object(self.retriever.lsh, "query") as lookup:
            for text, expected_id in [("first prompt", 0), ("second prompt", 1)]:
                with self.subTest(prompt_id=expected_id):
                    result = self.retriever.search(text)
                    self.assertEqual(result.status, "exact")
                    self.assertEqual(result.prompt_ids, (expected_id,))
                    self.assertIsNone(result.similarities)
                    self.assertEqual(result.candidate_count, 0)
                    self.assertEqual(result.cosine_computations, 0)
                    self.assertFalse(result.embedding_generated)
                    self.assertFalse(result.lsh_used)
            lookup.assert_not_called()
        self.embed_query.assert_not_called()

    def test_exact_hit_ignores_supplied_invalid_embedding(self):
        result = self.retriever.search("first prompt", np.array([np.nan]))
        self.assertEqual(result.prompt_ids, (0,))
        self.assertEqual(result.status, "exact")

    def test_exact_miss_calls_encoder_then_real_lsh(self):
        with patch.object(
            self.retriever.lsh, "query", wraps=self.retriever.lsh.query
        ) as lookup:
            result = self.retriever.search("a new wording")
            lookup.assert_called_once()
            np.testing.assert_allclose(lookup.call_args.args[0], [1.0, 0.0])
        self.embed_query.assert_called_once_with("a new wording")
        self.assertEqual(result.status, "semantic")
        self.assertEqual(result.prompt_ids, (0,))
        self.assertEqual(result.similarities, (1.0,))
        self.assertTrue(result.embedding_generated)
        self.assertTrue(result.lsh_used)

    def test_precomputed_embedding_skips_encoder_but_uses_lsh(self):
        with patch.object(self.retriever.lsh, "query", return_value={0}) as lookup:
            result = self.retriever.search("new text", np.array([10.0, 0.0]))
            lookup.assert_called_once()
        self.embed_query.assert_not_called()
        self.assertEqual(result.status, "semantic")
        self.assertFalse(result.embedding_generated)

    def test_verifies_only_candidates_and_includes_threshold_boundary(self):
        retriever = SemanticReuseRetriever(
            ["identical direction", "boundary", "below threshold", "not a candidate"],
            np.array([[1, 0], [0.8, 0.6], [0.6, 0.8], [1, 0]]),
            k=1,
            L=1,
        )
        with patch.object(retriever.lsh, "query", return_value={2, 1, 0}):
            result = retriever.search("new text", np.array([1.0, 0.0]))
        self.assertEqual(result.status, "semantic")
        self.assertEqual(result.prompt_ids, (0, 1))
        np.testing.assert_allclose(result.similarities, [1.0, 0.8])
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(result.cosine_computations, 3)

    def test_real_lsh_empty_candidates_return_no_match(self):
        retriever = SemanticReuseRetriever(["historical"], np.array([[1.0, 0.0]]), k=4, L=3)
        result = retriever.search("different", np.array([-1.0, 0.0]))
        self.assertEqual(result.status, "no_match")
        self.assertEqual(result.prompt_ids, ())
        self.assertEqual(result.similarities, ())
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.cosine_computations, 0)
        self.assertTrue(result.lsh_used)

    def test_candidates_below_threshold_return_no_match(self):
        with patch.object(self.retriever.lsh, "query", return_value={1, 2}):
            result = self.retriever.search("new text", np.array([1.0, 0.0]))
        self.assertEqual(result.status, "no_match")
        self.assertEqual(result.prompt_ids, ())
        self.assertEqual(result.similarities, ())
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.cosine_computations, 2)

    def test_exact_hit_is_one_reuse_result_not_all_semantic_neighbors(self):
        retriever = SemanticReuseRetriever(
            ["original", "paraphrase"], np.array([[1.0, 0.0], [1.0, 0.0]]), k=1, L=1
        )
        self.assertEqual(retriever.search("original").prompt_ids, (0,))
        self.assertEqual(
            retriever.search("new wording", np.array([1.0, 0.0])).prompt_ids,
            (0, 1),
        )

    def test_missing_embedding_provider_raises_only_after_exact_miss(self):
        self.retriever.embed_query = None
        self.assertEqual(self.retriever.search("first prompt").status, "exact")
        with self.assertRaisesRegex(ValueError, "requires query_embedding"):
            self.retriever.search("new text")

    def test_invalid_query_embeddings_raise_before_lsh(self):
        with patch.object(self.retriever.lsh, "query") as lookup:
            for vector in ([0, 0], [np.nan, 1], [np.inf, 1], [1], [[1, 0]]):
                with self.subTest(vector=vector), self.assertRaises(ValueError):
                    self.retriever.search("new text", np.array(vector))
            lookup.assert_not_called()

    def test_encoder_errors_propagate_without_lsh(self):
        self.embed_query.side_effect = RuntimeError("encoder failed")
        with patch.object(self.retriever.lsh, "query") as lookup:
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                self.retriever.search("new text")
            lookup.assert_not_called()

    def test_constructor_rejects_invalid_historical_data(self):
        cases = [
            (["a"], [[0, 0]]),
            (["a"], [[np.nan, 0]]),
            (["a", "b"], [[1, 0]]),
            (["a", "a"], [[1, 0], [0, 1]]),
            ([], np.empty((0, 2))),
        ]
        for texts, vectors in cases:
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                SemanticReuseRetriever(texts, np.array(vectors), k=1, L=1)

    def test_normalization_does_not_mutate_input_embeddings(self):
        vectors = np.array([[2.0, 0.0]], dtype=np.float32)
        retriever = SemanticReuseRetriever(["original"], vectors, k=1, L=1)
        query = np.array([3.0, 0.0], dtype=np.float32)
        retriever.search("new text", query)
        np.testing.assert_array_equal(vectors, [[2.0, 0.0]])
        np.testing.assert_array_equal(query, [3.0, 0.0])


if __name__ == "__main__":
    unittest.main()
