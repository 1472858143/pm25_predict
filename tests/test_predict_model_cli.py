import unittest

from pm25_forecast.scripts.predict_model import build_arg_parser


class PredictModelCliTests(unittest.TestCase):
    def test_parser_defaults_to_720_to_72(self):
        args = build_arg_parser().parse_args(["--model", "arima"])
        self.assertEqual(args.model, "arima")
        self.assertEqual(args.input_window, 720)
        self.assertEqual(args.output_window, 72)

    def test_parser_attention_lstm_seq2seq_defaults(self):
        args = build_arg_parser().parse_args(["--model", "attention_lstm_seq2seq"])
        self.assertEqual(args.encoder_num_layers, 2)
        self.assertEqual(args.decoder_num_layers, 1)
        self.assertEqual(args.num_heads, 4)


if __name__ == "__main__":
    unittest.main()
