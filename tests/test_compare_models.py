from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from Reproduce.scripts.compare_models import compare_existing_predictions


class CompareModelsTests(unittest.TestCase):
    def test_compare_existing_predictions_writes_metrics_and_all_predictions(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "window_720h_to_72h"
            start = base / "predictions" / "start_2026_03_01_0000"
            for model_name, pred in [("lstm", 11.0), ("random_forest", 12.0)]:
                model_output = start / model_name
                model_output.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "model_name": [model_name, model_name],
                        "sample_id": [0, 0],
                        "origin_timestamp": ["2026-03-01 00:00:00+08:00", "2026-03-01 00:00:00+08:00"],
                        "target_end_timestamp": ["2026-03-01 01:00:00+08:00", "2026-03-01 01:00:00+08:00"],
                        "timestamp": ["2026-03-01 00:00:00+08:00", "2026-03-01 01:00:00+08:00"],
                        "horizon": [1, 2],
                        "y_true": [10.0, 10.0],
                        "y_pred_model": [pred, pred],
                        "y_pred": [pred, pred],
                        "error": [pred - 10.0, pred - 10.0],
                        "abs_error": [abs(pred - 10.0), abs(pred - 10.0)],
                        "relative_error": [abs(pred - 10.0) / 10.0, abs(pred - 10.0) / 10.0],
                    }
                ).to_csv(model_output / "predictions.csv", index=False)
            out = compare_existing_predictions(base, "2026-03-01 00:00:00+08:00", ["lstm", "random_forest"])
            self.assertTrue((out / "model_metrics.csv").exists())
            self.assertTrue((out / "model_metrics.json").exists())
            self.assertTrue((out / "all_predictions.csv").exists())
            metrics = pd.read_csv(out / "model_metrics.csv")
            self.assertEqual(metrics["model_name"].tolist(), ["lstm", "random_forest"])
            all_predictions = pd.read_csv(out / "all_predictions.csv")
            self.assertEqual(len(all_predictions), 4)


if __name__ == "__main__":
    unittest.main()
