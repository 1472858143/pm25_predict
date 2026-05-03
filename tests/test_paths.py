from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch

from pm25_forecast.utils.data_utils import experiment_name
from pm25_forecast.utils.paths import (
    SUPPORTED_MODEL_NAMES,
    comparison_dir,
    data_dir,
    model_dir,
    prediction_dir,
    validate_model_name,
    window_experiment_dir,
    window_experiment_name,
)


class PathUtilityTests(unittest.TestCase):
    def test_project_package_is_pm25_forecast(self):
        self.assertIsNotNone(importlib.util.find_spec("pm25_forecast"))
        self.assertIsNone(importlib.util.find_spec("Reproduce"))

    def test_window_experiment_name_replaces_lstm_name(self):
        self.assertEqual(window_experiment_name(720, 72), "window_720h_to_72h")
        self.assertEqual(experiment_name(720, 72), "window_720h_to_72h")

    def test_model_prediction_and_comparison_dirs_are_nested_under_window(self):
        root = Path("E:/tmp/pm25_outputs")
        base = window_experiment_dir(root, 720, 72)
        self.assertEqual(base, root / "window_720h_to_72h")
        self.assertEqual(data_dir(base), base / "data")
        self.assertEqual(model_dir(base, "lstm"), base / "models" / "lstm")
        self.assertEqual(
            prediction_dir(base, "2026-03-01 00:00:00+08:00", "random_forest"),
            base / "predictions" / "start_2026_03_01_0000" / "random_forest",
        )
        self.assertEqual(
            comparison_dir(base, "2026-03-01 00:00:00+08:00"),
            base / "comparisons" / "start_2026_03_01_0000",
        )

    def test_supported_model_names_are_fixed(self):
        self.assertEqual(
            SUPPORTED_MODEL_NAMES,
            ("lstm", "attention_lstm", "xgboost", "random_forest", "arima", "sarima"),
        )

    def test_validate_model_name_requires_exact_supported_name(self):
        self.assertEqual(validate_model_name("random_forest"), "random_forest")
        for model_name in ("LSTM", " lstm ", "unknown"):
            with self.assertRaises(ValueError):
                validate_model_name(model_name)

    def test_cli_parse_args_default_to_window_constants(self):
        from pm25_forecast.scripts import predict_month, prepare_data, train_lstm

        parsers = (
            ("prepare_data", prepare_data.parse_args),
            ("train_lstm", train_lstm.parse_args),
            ("predict_month", predict_month.parse_args),
        )
        for script_name, parse_args in parsers:
            with self.subTest(script=script_name):
                with patch.object(sys, "argv", [script_name]):
                    args = parse_args()
                self.assertEqual(args.input_window, 720)
                self.assertEqual(args.output_window, 72)

    def test_lstm_prediction_labels_include_dynamic_horizon(self):
        from pm25_forecast.scripts.predict_month import prediction_plot_title, stage_metric_ranges

        self.assertEqual(
            stage_metric_ranges(72),
            {
                "h01_08": (1, 8),
                "h09_16": (9, 16),
                "h17_72": (17, 72),
            },
        )
        self.assertIn("LSTM 72h", prediction_plot_title("prediction", 72, "2026-03-01 00:00:00+08:00"))


if __name__ == "__main__":
    unittest.main()
