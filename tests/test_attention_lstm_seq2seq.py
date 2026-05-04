import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestAttentionLSTMSeq2Seq(unittest.TestCase):
    def test_forward_shape_teacher_forcing(self):
        from pm25_forecast.models.attention_lstm_seq2seq import Seq2SeqConfig, build_seq2seq_model

        cfg = Seq2SeqConfig(
            input_size_history=16,
            input_size_future=9,
            hidden_size=32,
            encoder_num_layers=1,
            decoder_num_layers=1,
            num_heads=2,
            dropout=0.1,
            output_window=8,
        )
        model = build_seq2seq_model(cfg)
        model.eval()
        batch = 4
        history = torch.randn(batch, 24, 16)
        future = torch.randn(batch, 8, 9)
        first_pm25 = torch.randn(batch, 1)
        teacher = torch.randn(batch, 8)
        out = model(history, future, first_pm25, teacher_forcing_targets=teacher, teacher_forcing_prob=1.0)
        self.assertEqual(out.shape, (batch, 8))

    def test_forward_shape_autoregressive(self):
        from pm25_forecast.models.attention_lstm_seq2seq import Seq2SeqConfig, build_seq2seq_model

        cfg = Seq2SeqConfig(
            input_size_history=16,
            input_size_future=9,
            hidden_size=32,
            encoder_num_layers=1,
            decoder_num_layers=1,
            num_heads=2,
            dropout=0.0,
            output_window=8,
        )
        model = build_seq2seq_model(cfg)
        model.eval()
        history = torch.randn(2, 24, 16)
        future = torch.randn(2, 8, 9)
        first_pm25 = torch.randn(2, 1)
        out = model(history, future, first_pm25, teacher_forcing_targets=None, teacher_forcing_prob=0.0)
        self.assertEqual(out.shape, (2, 8))

    def test_hidden_not_divisible_by_heads_raises(self):
        from pm25_forecast.models.attention_lstm_seq2seq import Seq2SeqConfig, build_seq2seq_model

        cfg = Seq2SeqConfig(
            input_size_history=16,
            input_size_future=9,
            hidden_size=33,
            encoder_num_layers=1,
            decoder_num_layers=1,
            num_heads=2,
            dropout=0.0,
            output_window=4,
        )
        with self.assertRaises(ValueError):
            build_seq2seq_model(cfg)


if __name__ == "__main__":
    unittest.main()
