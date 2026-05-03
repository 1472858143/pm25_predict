import unittest

from pm25_forecast.scripts.train_model import build_arg_parser


class TrainModelCliTests(unittest.TestCase):
    def test_parser_accepts_all_model_names_and_defaults_to_72h(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--model", "random_forest"])
        self.assertEqual(args.model, "random_forest")
        self.assertEqual(args.input_window, 720)
        self.assertEqual(args.output_window, 72)

    def test_parser_accepts_lstm_device_option(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--model", "lstm", "--device", "cpu", "--epochs", "1"])
        self.assertEqual(args.model, "lstm")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.epochs, 1)


if __name__ == "__main__":
    unittest.main()
