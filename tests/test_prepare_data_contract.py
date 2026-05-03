import sys
import unittest
from unittest.mock import patch

from pm25_forecast.scripts.prepare_data import parse_args
from pm25_forecast.utils.data_utils import DEFAULT_INPUT_WINDOW, DEFAULT_OUTPUT_WINDOW, experiment_name


class PrepareDataContractTests(unittest.TestCase):
    def test_default_window_is_720_to_72(self):
        self.assertEqual(DEFAULT_INPUT_WINDOW, 720)
        self.assertEqual(DEFAULT_OUTPUT_WINDOW, 72)
        self.assertEqual(experiment_name(DEFAULT_INPUT_WINDOW, DEFAULT_OUTPUT_WINDOW), "window_720h_to_72h")
        self.assertEqual(experiment_name(DEFAULT_INPUT_WINDOW), "window_720h_to_72h")

    def test_prepare_data_parser_defaults_to_72_hour_output(self):
        with patch.object(sys, "argv", ["prepare_data.py"]):
            args = parse_args()
        self.assertEqual(args.input_window, 720)
        self.assertEqual(args.output_window, 72)


if __name__ == "__main__":
    unittest.main()
