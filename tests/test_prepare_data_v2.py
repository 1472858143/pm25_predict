import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pm25_forecast.utils.data_utils import prepare_data_bundle


class TestPrepareDataBundleV2(unittest.TestCase):
    def _make_csv(self, path: Path, n_hours: int = 1500) -> None:
        ts = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="Asia/Shanghai")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "temperature": rng.uniform(-10, 30, n_hours),
                "humidity": rng.uniform(20, 90, n_hours),
                "wind_speed": rng.uniform(0, 10, n_hours),
                "precipitation": rng.uniform(0, 5, n_hours),
                "pressure": rng.uniform(990, 1030, n_hours),
                "pm25": rng.uniform(10, 200, n_hours),
            }
        )
        df.to_csv(path, index=False)

    def test_bundle_v2_contains_full_and_future_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "data.csv"
            self._make_csv(csv_path, n_hours=5000)
            cfg = prepare_data_bundle(
                data_path=csv_path,
                output_root=tmp_path / "outputs",
                input_window=240,
                output_window=24,
                predict_start="2024-05-25 00:00:00+08:00",
            )
            with np.load(cfg["bundle_path"], allow_pickle=True) as bundle:
                self.assertEqual(int(bundle["bundle_version"]), 2)
                for key in [
                    "X_train_full",
                    "X_train_future",
                    "X_validation_full",
                    "X_validation_future",
                    "X_predict_full",
                    "X_predict_future",
                ]:
                    self.assertIn(key, bundle.files)
                self.assertEqual(bundle["X_train_full"].shape[-1], 16)
                self.assertEqual(bundle["X_train_future"].shape[-1], 9)
                self.assertEqual(bundle["X_predict_full"].shape, (1, 240, 16))
                self.assertEqual(bundle["X_predict_future"].shape, (1, 24, 9))

    def test_legacy_x_train_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "data.csv"
            self._make_csv(csv_path, n_hours=5000)
            cfg = prepare_data_bundle(
                data_path=csv_path,
                output_root=tmp_path / "outputs",
                input_window=240,
                output_window=24,
                predict_start="2024-05-25 00:00:00+08:00",
            )
            with np.load(cfg["bundle_path"], allow_pickle=True) as bundle:
                self.assertEqual(bundle["X_train"].shape[-1], 6)

    def test_v2_features_do_not_contain_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "data.csv"
            self._make_csv(csv_path, n_hours=5000)
            cfg = prepare_data_bundle(
                data_path=csv_path,
                output_root=tmp_path / "outputs",
                input_window=240,
                output_window=24,
                predict_start="2024-05-25 00:00:00+08:00",
            )
            with np.load(cfg["bundle_path"], allow_pickle=True) as bundle:
                self.assertFalse(np.isnan(bundle["X_train_full"]).any())
                self.assertFalse(np.isnan(bundle["X_train_future"]).any())


if __name__ == "__main__":
    unittest.main()
