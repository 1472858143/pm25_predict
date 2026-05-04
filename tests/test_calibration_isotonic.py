import json
import unittest

import numpy as np

from pm25_forecast.utils.calibration import (
    apply_calibration,
    fit_horizon_isotonic_calibration,
)


class TestHorizonIsotonicCalibration(unittest.TestCase):
    def test_fit_returns_per_horizon_thresholds(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, size=(50, 3))
        y_pred = y_true * 0.5 + 5 + rng.normal(0, 1, size=(50, 3))
        cal = fit_horizon_isotonic_calibration(y_true, y_pred)
        self.assertEqual(cal["method"], "horizon_isotonic")
        self.assertEqual(cal["output_window"], 3)
        self.assertEqual(len(cal["x_thresholds"]), 3)
        self.assertEqual(len(cal["y_thresholds"]), 3)

    def test_apply_clips_to_min(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]])
        cal = fit_horizon_isotonic_calibration(y_true, y_pred, clip_min=0.0)
        out = apply_calibration(np.array([[-1.0, -1.0]]), cal)
        self.assertTrue((out >= 0.0).all())

    def test_apply_dispatches_method(self):
        cal_none = {"method": "none"}
        out = apply_calibration(np.array([[1.0, 2.0]]), cal_none)
        np.testing.assert_array_equal(out, np.array([[1.0, 2.0]]))

    def test_isotonic_serialization_roundtrip(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 100, size=(50, 2))
        y_pred = y_true + rng.normal(0, 5, size=(50, 2))
        cal = fit_horizon_isotonic_calibration(y_true, y_pred)
        text = json.dumps(cal)
        restored = json.loads(text)
        self.assertEqual(restored["method"], "horizon_isotonic")
        sample = np.array([[50.0, 50.0]])
        out_a = apply_calibration(sample, cal)
        out_b = apply_calibration(sample, restored)
        np.testing.assert_allclose(out_a, out_b)


if __name__ == "__main__":
    unittest.main()
