import unittest

import numpy as np

from core.logistic_utils import train_binary_logistic_minmax


class LogisticTrainingTests(unittest.TestCase):
    def test_invalid_training_data_raises(self):
        for features, labels in (
            ([[1.0]], []),
            ([[], []], [0, 1]),
            ([[1.0], [np.nan]], [0, 1]),
            ([[1.0], [2.0]], [0, 2]),
            ([[1.0], [2.0]], [[0], [1]]),
        ):
            with self.subTest(features=features, labels=labels), self.assertRaises(ValueError):
                train_binary_logistic_minmax(features, labels)

    def test_insufficient_examples_remain_an_explicit_untrained_result(self):
        for features, labels in (([], []), ([[1.0], [2.0]], [1, 1])):
            with self.subTest(features=features):
                trained, _, _ = train_binary_logistic_minmax(features, labels)
                self.assertFalse(trained)

    def test_valid_examples_train(self):
        trained, weights, (low, span) = train_binary_logistic_minmax([[0.0], [1.0]], [0, 1])
        self.assertTrue(trained)
        self.assertEqual(weights.shape, (2,))
        np.testing.assert_array_equal(low, [0.0])
        np.testing.assert_array_equal(span, [1.0])


if __name__ == "__main__":
    unittest.main()
