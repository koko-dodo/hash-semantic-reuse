import itertools
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.baseline_comparison import (
    QueryInput,
    benchmark,
    evaluate_semantic,
    make_methods,
    timing_statistics,
    validate_exact_hits,
)
from src.exact_hash import ExactHashIndex
from src.retrieval import RetrievalResult, SemanticReuseRetriever


class BenchmarkTests(unittest.TestCase):
    def test_equal_warmup_five_rounds_and_rotating_order(self):
        calls = []
        queries = [object(), object()]
        methods = {
            name: Mock(side_effect=lambda query, name=name: calls.append((name, query)))
            for name in ("a", "b", "c")
        }
        clock = itertools.count(0, 1_000_000)
        with (
            patch("experiments.baseline_comparison.time.perf_counter_ns", side_effect=lambda: next(clock)) as timer,
            patch("experiments.baseline_comparison.gc.collect"),
        ):
            summary, rounds = benchmark(methods, queries)
        self.assertEqual(timer.call_count, 3 * 2 * 5 * 2)
        self.assertEqual(calls[:6], [(name, q) for name in methods for q in queries])
        self.assertEqual(len(rounds), 15)
        for index in range(5):
            names = list(methods)
            offset = index % 3
            expected = names[offset:] + names[:offset]
            self.assertEqual(
                [row["method"] for row in rounds if row["round"] == index + 1], expected
            )
        for name, method in methods.items():
            self.assertEqual(method.call_count, 2 * (1 + 5))
            self.assertEqual(summary[name]["timed_query_count"], 10)
            self.assertEqual(summary[name]["mean_time_ms"], 1.0)
            self.assertEqual(summary[name]["median_time_ms"], 1.0)
            self.assertEqual(summary[name]["round_mean_std_ms"], 0.0)

    def test_statistics_include_query_variation(self):
        stats = timing_statistics([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["mean_time_ms"], 2.5)
        self.assertEqual(stats["median_time_ms"], 2.5)
        self.assertAlmostEqual(stats["std_time_ms"], np.sqrt(5 / 3))
        self.assertAlmostEqual(stats["p95_time_ms"], 3.85)

    def test_benchmark_rejects_insufficient_rounds_and_empty_input(self):
        for methods, queries, repeats in [({"a": Mock()}, [1], 4), ({}, [1], 5), ({"a": Mock()}, [], 5)]:
            with self.subTest(repeats=repeats), self.assertRaises(ValueError):
                benchmark(methods, queries, repeats)


class BaselineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.texts = ["first", "second", "third"]
        self.vectors = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        self.exact = ExactHashIndex()
        self.exact.build_index(self.texts)
        self.retriever = SemanticReuseRetriever(self.texts, self.vectors, k=4, L=3)
        self.methods = make_methods(self.vectors, self.exact, self.retriever)

    def test_adapters_use_preloaded_inputs_and_real_hybrid_pipeline(self):
        query = QueryInput("different wording", np.array([1, 0], dtype=np.float32))
        self.assertEqual(self.methods["Brute-force"](query), (0,))
        self.assertEqual(self.methods["Exact Hash Only"](query), ())
        with (
            patch.object(self.retriever, "search", wraps=self.retriever.search) as search,
            patch.object(self.retriever.exact, "search", wraps=self.retriever.exact.search) as exact,
            patch.object(self.retriever.lsh, "query", wraps=self.retriever.lsh.query) as lsh,
        ):
            result = self.methods["Hash + LSH"](query)
            search.assert_called_once_with(query.text, query_embedding=query.embedding)
            exact.assert_called_once_with(query.text)
            lsh.assert_called_once()
        self.assertEqual(result.status, "semantic")
        self.assertEqual(result.prompt_ids, (0,))
        self.assertFalse(result.embedding_generated)

    def test_known_exact_validation_includes_id_zero_and_restores_methods(self):
        original_encoder = self.retriever.embed_query
        original_lookup = self.retriever.lsh.query
        self.assertEqual(validate_exact_hits(self.exact, self.retriever, self.texts), 3)
        self.assertIs(self.retriever.embed_query, original_encoder)
        self.assertEqual(self.retriever.lsh.query, original_lookup)
        result = self.methods["Hash + LSH"](QueryInput("first", None))
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.prompt_ids, (0,))

    def test_known_exact_validation_rejects_wrong_id(self):
        with patch.object(self.exact, "search", return_value=1):
            with self.assertRaisesRegex(AssertionError, "incorrect row ID"):
                validate_exact_hits(self.exact, self.retriever, self.texts)


class SemanticMetricsTests(unittest.TestCase):
    def test_macro_micro_empty_ground_truth_and_candidate_denominators(self):
        queries = [QueryInput(str(i), None) for i in range(3)]
        truth = [{0, 1}, set(), {2}]
        results = [
            RetrievalResult("semantic", (0,), (1.0,), 2, 2, False, True),
            RetrievalResult("no_match", (), (), 1, 1, False, True),
            RetrievalResult("semantic", (2,), (1.0,), 1, 1, False, True),
        ]
        methods = {
            "Brute-force": lambda q: tuple(truth[int(q.text)]),
            "Exact Hash Only": lambda q: (),
            "Hash + LSH": lambda q: results[int(q.text)],
        }
        rows = evaluate_semantic(methods, queries, truth, 4)
        brute, exact, hybrid = rows
        self.assertEqual(brute["mean_recall"], 1.0)
        self.assertEqual(brute["candidate_ratio"], 1.0)
        self.assertEqual(brute["avg_cosine_computations"], 4)
        self.assertEqual(exact["mean_recall"], 0.0)
        self.assertEqual(exact["candidate_ratio"], 0.0)
        self.assertEqual(hybrid["valid_recall_queries"], 2)
        self.assertEqual(hybrid["mean_recall"], 0.75)
        self.assertAlmostEqual(hybrid["micro_recall"], 2 / 3)
        self.assertAlmostEqual(hybrid["avg_candidates"], 4 / 3)
        self.assertAlmostEqual(hybrid["candidate_ratio"], 1 / 3)
        self.assertAlmostEqual(hybrid["avg_cosine_computations"], 4 / 3)

    def test_all_empty_ground_truth_returns_nan_recall(self):
        rows = evaluate_semantic(
            {"Brute-force": lambda q: ()}, [QueryInput("new", None)], [set()], 4
        )
        self.assertTrue(np.isnan(rows[0]["mean_recall"]))
        self.assertTrue(np.isnan(rows[0]["micro_recall"]))
        self.assertEqual(rows[0]["valid_recall_queries"], 0)

    def test_incorrect_brute_force_and_unexpected_exact_hits_are_rejected(self):
        for methods in [
            {"Brute-force": lambda q: ()},
            {"Exact Hash Only": lambda q: (0,)},
            {"Hash + LSH": lambda q: RetrievalResult("exact", (0,), None, 0, 0, False, False)},
        ]:
            with self.subTest(method=list(methods)), self.assertRaises(AssertionError):
                evaluate_semantic(methods, [QueryInput("new", None)], [{0}], 4)


if __name__ == "__main__":
    unittest.main()
