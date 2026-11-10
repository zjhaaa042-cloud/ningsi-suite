import unittest
import numpy as np
from ningsi.models.logistic import LinearModel, evaluate, fit_logistic, roc_auc


class ModelValidationTests(unittest.TestCase):
    def test_column_labels_match_vector_labels(self):
        x = [[-1.], [1.]]
        model = LinearModel(np.array([1.]), 0., np.zeros(1), np.ones(1))
        result = evaluate(model, x, [[0], [1]])
        self.assertEqual(result, evaluate(model, x, [0, 1]))
        self.assertEqual(sum(result['confusion'].values()), 2)
        a = fit_logistic(x, [[0], [1]], epochs=2)
        b = fit_logistic(x, [0, 1], epochs=2)
        np.testing.assert_allclose(a.weights, b.weights)

    def test_invalid_training_and_evaluation_inputs(self):
        model = LinearModel(np.ones(1), 0., np.zeros(1), np.ones(1))
        for x, y in [([], []), ([[]], [0]), ([[1]], [0, 1]),
                     ([[1]], [0.5]), ([[float('nan')]], [0]),
                     ([[1]], [float('inf')]), ([[1], [2]], [[0, 1]])]:
            with self.subTest(x=x, y=y):
                with self.assertRaises(ValueError):
                    fit_logistic(x, y)
                with self.assertRaises(ValueError):
                    evaluate(model, x, y)

    def test_auc_rejects_mismatched_or_invalid_probabilities(self):
        for scores in [[0.1], [[0.1], [0.9]], [0.1, float('nan')], [-1, 2]]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                roc_auc([0, 1], scores)
        self.assertIsNone(roc_auc([1, 1], [0.2, 0.8]))
