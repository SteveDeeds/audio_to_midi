import unittest

from core.inference_cache import InferenceCache
from core.learning_runtime import LearningDomain, RuntimeToken


class TestInferenceCache(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = InferenceCache()
        self.token = RuntimeToken(session=1, model=2, edits=3, settings=4)

    def test_merge_and_missing_units(self) -> None:
        merged = self.cache.merge(
            self.token,
            self.token,
            LearningDomain.PITCH,
            {10: [60], 11: [64]},
        )

        self.assertTrue(merged)
        self.assertEqual(
            self.cache.values(
                self.token, LearningDomain.PITCH, [10, 11, 12]
            ),
            {10: [60], 11: [64]},
        )
        self.assertEqual(
            self.cache.missing(
                self.token, LearningDomain.PITCH, [10, 11, 12]
            ),
            [12],
        )
        self.assertEqual(
            self.cache.completion(
                self.token, LearningDomain.PITCH, [10, 11, 12]
            ),
            (2, 3),
        )

    def test_stale_worker_result_is_rejected(self) -> None:
        stale = RuntimeToken(session=1, model=1, edits=3, settings=4)

        merged = self.cache.merge(
            stale,
            self.token,
            LearningDomain.PITCH,
            {10: [60]},
        )

        self.assertFalse(merged)
        self.assertEqual(
            self.cache.missing(self.token, LearningDomain.PITCH, [10]), [10]
        )

    def test_generation_change_is_a_lazy_cache_miss(self) -> None:
        self.cache.merge(
            self.token,
            self.token,
            LearningDomain.ONSET,
            {20: True},
        )
        newer = RuntimeToken(session=1, model=3, edits=3, settings=4)

        self.assertIsNone(self.cache.get(newer, LearningDomain.ONSET, 20))
        self.assertTrue(self.cache.get(self.token, LearningDomain.ONSET, 20))

    def test_domains_do_not_collide(self) -> None:
        self.cache.merge(
            self.token,
            self.token,
            LearningDomain.PITCH,
            {10: [60]},
        )

        self.assertIsNone(self.cache.get(self.token, LearningDomain.DRUM, 10))

    def test_old_sessions_can_be_pruned(self) -> None:
        old = RuntimeToken(session=1, model=1, edits=1, settings=1)
        current = RuntimeToken(session=2, model=1, edits=1, settings=1)
        self.cache.merge(old, old, LearningDomain.CHORD, {1: "C"})
        self.cache.merge(current, current, LearningDomain.CHORD, {1: "G"})

        self.cache.discard_other_sessions(2)

        self.assertIsNone(self.cache.get(old, LearningDomain.CHORD, 1))
        self.assertEqual(self.cache.get(current, LearningDomain.CHORD, 1), "G")


if __name__ == "__main__":
    unittest.main()