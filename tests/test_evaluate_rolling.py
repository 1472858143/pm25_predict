import unittest

import pandas as pd

from pm25_forecast.scripts.evaluate_rolling import generate_origin_timestamps


class TestGenerateOriginTimestamps(unittest.TestCase):
    def test_stride_24_three_months(self):
        eval_start = pd.Timestamp("2025-12-01 00:00", tz="Asia/Shanghai")
        eval_end = pd.Timestamp("2026-03-01 00:00", tz="Asia/Shanghai")
        origins = generate_origin_timestamps(eval_start, eval_end, output_window=72, stride=24)
        self.assertGreater(len(origins), 80)
        self.assertEqual(origins[0], eval_start)
        last = origins[-1]
        self.assertLessEqual(last + pd.Timedelta(hours=72), eval_end)

    def test_origin_after_eval_end_minus_window_excluded(self):
        eval_start = pd.Timestamp("2025-12-01", tz="Asia/Shanghai")
        eval_end = pd.Timestamp("2025-12-05", tz="Asia/Shanghai")
        origins = generate_origin_timestamps(eval_start, eval_end, output_window=72, stride=24)
        self.assertEqual(len(origins), 1)
        self.assertEqual(origins[0], eval_start)


if __name__ == "__main__":
    unittest.main()
