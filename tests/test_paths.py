from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from Reproduce.utils.data_utils import experiment_name
from Reproduce.utils.paths import (
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
            ("lstm", "xgboost", "random_forest", "arima", "sarima"),
        )

    def test_validate_model_name_requires_exact_supported_name(self):
        self.assertEqual(validate_model_name("random_forest"), "random_forest")
        for model_name in ("LSTM", " lstm ", "unknown"):
            with self.assertRaises(ValueError):
                validate_model_name(model_name)

    def test_cli_parse_args_default_to_window_constants(self):
        from Reproduce.scripts import predict_month, prepare_data, train_lstm

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


if __name__ == "__main__":
    unittest.main()
