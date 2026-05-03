from pathlib import Path
import unittest

from Reproduce.scripts.predict_month import checkpoint_path
from Reproduce.scripts.train_lstm import resolve_lstm_training_paths
from Reproduce.utils.paths import model_dir, prediction_dir, window_experiment_dir


class LstmPathTests(unittest.TestCase):
    def test_training_paths_use_lstm_model_subdirectory(self):
        paths = resolve_lstm_training_paths(Path("E:/tmp/outputs"), 720, 72)
        self.assertEqual(paths["window_dir"], Path("E:/tmp/outputs/window_720h_to_72h"))
        self.assertEqual(paths["lstm_dir"], Path("E:/tmp/outputs/window_720h_to_72h/models/lstm"))
        self.assertEqual(paths["data_config_path"], Path("E:/tmp/outputs/window_720h_to_72h/data/data_config.json"))
        self.assertEqual(paths["bundle_path"], Path("E:/tmp/outputs/window_720h_to_72h/data/windows.npz"))

    def test_checkpoint_path_resolves_inside_lstm_model_dir(self):
        exp_dir = window_experiment_dir(Path("E:/tmp/outputs"), 720, 72)
        lstm_dir = model_dir(exp_dir, "lstm")
        self.assertEqual(checkpoint_path(lstm_dir, None), lstm_dir / "model.pt")

    def test_prediction_dir_for_lstm_uses_model_subdirectory(self):
        exp_dir = window_experiment_dir(Path("E:/tmp/outputs"), 720, 72)
        self.assertEqual(
            prediction_dir(exp_dir, "2026-03-01 00:00:00+08:00", "lstm"),
            Path("E:/tmp/outputs/window_720h_to_72h/predictions/start_2026_03_01_0000/lstm"),
        )


if __name__ == "__main__":
    unittest.main()
