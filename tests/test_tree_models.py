import unittest

import numpy as np

from pm25_forecast.models.tree_models import flatten_window_features, train_random_forest_model


class TreeModelTests(unittest.TestCase):
    def test_flatten_window_features_preserves_sample_count(self):
        X = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        flattened = flatten_window_features(X)
        self.assertEqual(flattened.shape, (2, 12))
        self.assertEqual(flattened[0, 0], 0.0)
        self.assertEqual(flattened[1, -1], 23.0)

    def test_random_forest_predicts_direct_multi_output_shape(self):
        X_train = np.arange(8 * 3 * 2, dtype=float).reshape(8, 3, 2)
        y_train = np.arange(8 * 4, dtype=float).reshape(8, 4)
        model = train_random_forest_model(X_train, y_train, n_estimators=3, random_state=7, n_jobs=1)
        prediction = model.predict(flatten_window_features(X_train[:1]))
        self.assertEqual(prediction.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
