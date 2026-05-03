from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from pm25_forecast.models.statistical_models import forecast_statistical_model, load_train_pm25_series


class _FakeForecastModel:
    def forecast(self, steps: int):
        return np.arange(steps, dtype=float) - 1.0


class StatisticalModelTests(unittest.TestCase):
    def test_load_train_pm25_series_uses_train_period_only(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "beijing.csv"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-01 00:00:00+08:00", periods=6, freq="h"),
                    "pm25": [10, 11, 12, 13, 99, 100],
                    "temp": [1, 1, 1, 1, 1, 1],
                    "humidity": [50, 50, 50, 50, 50, 50],
                    "wind_speed": [2, 2, 2, 2, 2, 2],
                    "precipitation": [0, 0, 0, 0, 0, 0],
                    "pressure": [1000, 1000, 1000, 1000, 1000, 1000],
                }
            )
            frame.to_csv(csv_path, index=False)
            data_config = {
                "data_path": str(csv_path),
                "train_period": {"end": "2026-01-01 03:00:00+08:00"},
            }
            series = load_train_pm25_series(data_config)
            self.assertEqual(series.tolist(), [10.0, 11.0, 12.0, 13.0])
            self.assertNotIn(99.0, series.tolist())
            self.assertNotIn(100.0, series.tolist())

    def test_forecast_statistical_model_returns_single_clipped_window(self):
        forecast = forecast_statistical_model(_FakeForecastModel(), output_window=4)
        self.assertEqual(forecast.shape, (1, 4))
        self.assertEqual(forecast.tolist(), [[0.0, 0.0, 1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
