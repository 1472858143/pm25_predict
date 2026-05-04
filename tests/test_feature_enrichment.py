import unittest

import numpy as np
import pandas as pd

from pm25_forecast.utils.data_utils import build_enriched_features


class TestBuildEnrichedFeatures(unittest.TestCase):
    def _frame(self, n: int = 200) -> pd.DataFrame:
        ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="Asia/Shanghai")
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "temperature": rng.uniform(-10, 30, n),
                "humidity": rng.uniform(20, 90, n),
                "wind_speed": rng.uniform(0, 10, n),
                "precipitation": rng.uniform(0, 5, n),
                "pressure": rng.uniform(990, 1030, n),
                "pm25": rng.uniform(10, 200, n),
            }
        )

    def test_returns_expected_columns(self):
        frame = self._frame()
        out = build_enriched_features(frame)
        expected_extra = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "pm25_lag_1h",
            "pm25_lag_24h",
            "pm25_lag_168h",
            "pm25_roll24_mean",
            "pm25_roll24_max",
            "pm25_roll24_std",
        ]
        for col in expected_extra:
            self.assertIn(col, out.columns)

    def test_drops_initial_rows_with_lag_nan(self):
        frame = self._frame(n=200)
        out = build_enriched_features(frame)
        self.assertEqual(len(out), len(frame) - 168)
        self.assertFalse(out[["pm25_lag_1h", "pm25_lag_24h", "pm25_lag_168h"]].isna().any().any())
        self.assertFalse(out[["pm25_roll24_mean", "pm25_roll24_std"]].isna().any().any())

    def test_lag_values_match_pm25(self):
        frame = self._frame(n=200)
        out = build_enriched_features(frame)
        self.assertAlmostEqual(
            float(out["pm25_lag_24h"].iloc[0]),
            float(frame["pm25"].iloc[168 - 24]),
            places=6,
        )

    def test_hour_sin_periodicity(self):
        frame = self._frame(n=200)
        out = build_enriched_features(frame)
        first_hour = float(out["hour_sin"].iloc[0])
        same_hour_24h_later = float(out["hour_sin"].iloc[24])
        self.assertAlmostEqual(first_hour, same_hour_24h_later, places=6)


if __name__ == "__main__":
    unittest.main()
