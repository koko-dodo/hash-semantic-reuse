import unittest

import numpy as np

from src.exact_hash import ExactHashIndex
from src.lsh import RandomHyperplaneLSH


class ExactHashRebuildTests(unittest.TestCase):
    def setUp(self):
        self.index = ExactHashIndex()
        self.index.build_index(["first", "second", "third"])

    def test_repeated_build_preserves_ids_and_entry_count(self):
        self.index.build_index(["first", "second", "third"])
        self.assertEqual(self.index.table, {"first": 0, "second": 1, "third": 2})

    def test_different_database_removes_old_texts_and_reassigns_ids(self):
        self.index.build_index(["second", "new"])
        self.assertEqual(self.index.table, {"second": 0, "new": 1})
        self.assertEqual(self.index.search("second"), 0)
        self.assertEqual(self.index.search("new"), 1)
        self.assertIsNone(self.index.search("first"))
        self.assertIsNone(self.index.search("third"))

    def test_reordered_database_uses_new_row_ids(self):
        self.index.build_index(["third", "first", "second"])
        self.assertEqual(self.index.search("third"), 0)
        self.assertEqual(self.index.search("first"), 1)
        self.assertEqual(self.index.search("second"), 2)

    def test_empty_database_clears_index_and_allows_later_rebuild(self):
        self.index.build_index([])
        self.assertEqual(self.index.table, {})
        self.assertIsNone(self.index.search("first"))
        self.index.build_index(["new"])
        self.assertEqual(self.index.search("new"), 0)
        self.assertEqual(self.index.table, {"new": 0})


class LSHRebuildTests(unittest.TestCase):
    def setUp(self):
        self.index = RandomHyperplaneLSH(dimension=2, k=4, L=3, seed=42)
        # Opposite directions have complementary keys in every table.
        self.positive = np.array([1.0, 0.0], dtype=np.float32)
        self.negative = -self.positive
        self.vectors = np.array([self.positive, self.negative, self.positive])
        self.index.build_index(self.vectors)

    def bucket_entries(self):
        return sum(len(ids) for table in self.index.tables for ids in table.values())

    def test_repeated_build_does_not_duplicate_bucket_entries(self):
        before = [{key: list(ids) for key, ids in table.items()} for table in self.index.tables]
        self.index.build_index(self.vectors)
        self.index.build_index(self.vectors)
        self.assertEqual(self.bucket_entries(), self.index.L * len(self.vectors))
        self.assertEqual(self.index.tables, before)
        self.assertEqual(self.index.query(self.positive), {0, 2})
        self.assertEqual(self.index.query(self.negative), {1})

    def test_smaller_database_removes_old_ids_and_buckets(self):
        smaller = np.array([self.negative])
        self.index.build_index(smaller)
        self.assertEqual(self.bucket_entries(), self.index.L)
        self.assertEqual(self.index.query(self.positive), set())
        ids = sorted(self.index.query(self.negative))
        self.assertEqual(ids, [0])
        # IDs can safely index the smaller database during cosine verification.
        np.testing.assert_allclose(smaller[ids] @ self.negative, [1.0])
        for table_id, table in enumerate(self.index.tables):
            self.assertEqual(list(table), [self.index.hash_vector(self.negative, table_id)])
            self.assertTrue(all(ids == [0] for ids in table.values()))

    def test_different_database_uses_new_row_ids(self):
        self.index.build_index(np.array([self.negative, self.positive]))
        self.assertEqual(self.index.query(self.negative), {0})
        self.assertEqual(self.index.query(self.positive), {1})
        self.assertEqual(self.bucket_entries(), 2 * self.index.L)

    def test_empty_database_clears_all_tables_and_allows_later_rebuild(self):
        for empty in (np.empty((0, 2), dtype=np.float32), []):
            with self.subTest(empty_type=type(empty).__name__):
                self.index.build_index(self.vectors)
                self.index.build_index(empty)
                self.assertEqual(self.bucket_entries(), 0)
                self.assertTrue(all(not table for table in self.index.tables))
                self.assertEqual(self.index.query(self.positive), set())
                self.assertEqual(self.index.query(self.negative), set())
        self.index.build_index(np.array([self.positive]))
        self.assertEqual(self.index.query(self.positive), {0})
        self.assertEqual(self.bucket_entries(), self.index.L)

    def test_rebuild_preserves_hyperplanes_keys_and_seed_behavior(self):
        original = self.index.hyperplanes
        planes = original.copy()
        keys = [self.index.hash_vector(self.positive, i) for i in range(self.index.L)]
        self.index.build_index(np.array([self.negative]))
        self.index.build_index([])
        self.index.build_index(self.vectors)
        self.assertIs(self.index.hyperplanes, original)
        np.testing.assert_array_equal(self.index.hyperplanes, planes)
        self.assertEqual(
            [self.index.hash_vector(self.positive, i) for i in range(self.index.L)], keys
        )
        fresh = RandomHyperplaneLSH(dimension=2, k=4, L=3, seed=42)
        fresh.build_index(self.vectors)
        np.testing.assert_array_equal(self.index.hyperplanes, fresh.hyperplanes)
        self.assertEqual(self.index.tables, fresh.tables)


if __name__ == "__main__":
    unittest.main()
