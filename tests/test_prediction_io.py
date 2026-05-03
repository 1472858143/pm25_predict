from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd

from Reproduce.utils.prediction_io import PREDICTION_COLUMNS, build_predictions_frame, write_prediction_outputs


class PredictionIoTests(unittest.TestCase):
    def test_build_predictions_frame_has_fixed_columns_and_errors(self):
        y_true = np.array([[10.0, 20.0, 40.0]], dtype=float)
        y_pred_model = np.array([[12.0, 18.0, 36.0]], dtype=float)
        y_pred = np.array([[11.0, 19.0, 39.0]], dtype=float)
        frame = build_predictions_frame(
            model_name="random_forest",
            y_true=y_true,
            y_pred_model=y_pred_model,
            y_pred=y_pred,
            timestamps_start=np.array(["2026-03-01 00:00:00+08:00"]),
            timestamps_end=np.array(["2026-03-01 02:00:00+08:00"]),
            timestamps_target=np.array(
                [
                    [
                        "2026-03-01 00:00:00+08:00",
                        "2026-03-01 01:00:00+08:00",
                        "2026-03-01 02:00:00+08:00",
                    ]
                ]
            ),
        )
        self.assertEqual(list(frame.columns), PREDICTION_COLUMNS)
        self.assertEqual(frame["model_name"].tolist(), ["random_forest", "random_forest", "random_forest"])
        self.assertEqual(frame["horizon"].tolist(), [1, 2, 3])
        self.assertEqual(frame["error"].round(6).tolist(), [1.0, -1.0, -1.0])
        self.assertEqual(frame["abs_error"].round(6).tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(frame["relative_error"].round(6).tolist(), [0.1, 0.05, 0.025])

    def test_write_prediction_outputs_creates_standard_files(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "predictions" / "lstm"
            frame = build_predictions_frame(
                model_name="lstm",
                y_true=np.array([[10.0, 20.0, 30.0]], dtype=float),
                y_pred_model=np.array([[9.0, 18.0, 33.0]], dtype=float),
                y_pred=np.array([[10.0, 19.0, 31.0]], dtype=float),
                timestamps_start=np.array(["2026-03-01 00:00:00+08:00"]),
                timestamps_end=np.array(["2026-03-01 02:00:00+08:00"]),
                timestamps_target=np.array(
                    [
                        [
                            "2026-03-01 00:00:00+08:00",
                            "2026-03-01 01:00:00+08:00",
                            "2026-03-01 02:00:00+08:00",
                        ]
                    ]
                ),
            )
            summary = write_prediction_outputs(
                predictions=frame,
                output_dir=output_dir,
                model_name="lstm",
                model_path=Path("E:/models/lstm/model.pt"),
                calibration_path=Path("E:/models/lstm/calibration.json"),
                calibration_applied=True,
                calibration_method="horizon_linear",
                device="cpu",
                predict_start="2026-03-01 00:00:00+08:00",
            )

            self.assertTrue((output_dir / "predictions.csv").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "metrics_model_raw.json").exists())
            self.assertTrue((output_dir / "horizon_metrics.csv").exists())
            self.assertTrue((output_dir / "horizon_metrics.json").exists())
            self.assertTrue((output_dir / "stage_metrics.csv").exists())
            self.assertTrue((output_dir / "stage_metrics.json").exists())
            self.assertTrue((output_dir / "plot_status.md").exists())
            self.assertTrue((output_dir / "prediction_summary.json").exists())
            plot_status = (output_dir / "plot_status.md").read_text(encoding="utf-8")
            plot_files = {
                "prediction_curve": output_dir / "plots" / "prediction_curve.png",
                "error_curve": output_dir / "plots" / "error_curve.png",
                "scatter": output_dir / "plots" / "scatter.png",
            }
            for plot_name, plot_path in plot_files.items():
                self.assertIn(f"- {plot_name}: `", plot_status)
                if f"- {plot_name}: `True`" in plot_status:
                    self.assertTrue(plot_path.exists(), f"{plot_name} was reported as plotted but PNG is missing")
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                pass
            else:
                for plot_path in plot_files.values():
                    self.assertTrue(plot_path.exists())
            written = pd.read_csv(output_dir / "predictions.csv")
            self.assertEqual(list(written.columns), PREDICTION_COLUMNS)
            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            stage_metrics = json.loads((output_dir / "stage_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("RMSE", metrics)
            self.assertEqual(list(stage_metrics), ["h001_003"])
            self.assertEqual(summary["model_name"], "lstm")
            self.assertEqual(summary["output_window"], 3)
