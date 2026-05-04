from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import optuna


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pm25_forecast.scripts import evaluate_rolling, train_attention_lstm_seq2seq
from pm25_forecast.utils.data_utils import (
    DEFAULT_INPUT_WINDOW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_OUTPUT_WINDOW,
    DEFAULT_PREDICT_START,
)
from pm25_forecast.utils.paths import window_experiment_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna search for attention_lstm_seq2seq.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--input-window", type=int, default=DEFAULT_INPUT_WINDOW)
    p.add_argument("--output-window", type=int, default=DEFAULT_OUTPUT_WINDOW)
    p.add_argument("--predict-start", default=DEFAULT_PREDICT_START)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--eval-start", required=True)
    p.add_argument("--eval-end", required=True)
    p.add_argument("--eval-stride", type=int, default=72)
    p.add_argument("--max-origins", type=int, default=30)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", default="attention_lstm_seq2seq_v1")
    return p.parse_args()


def sample_config(trial: optuna.trial.Trial, base: argparse.Namespace) -> SimpleNamespace:
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 192, 256])
    candidate_heads = [h for h in [2, 4, 8] if hidden_size % h == 0]
    num_heads = trial.suggest_categorical("num_heads", candidate_heads)
    return SimpleNamespace(
        data_path=str(Path(base.output_root).parent / "data" / "processed_beijing.csv"),
        output_root=base.output_root,
        input_window=base.input_window,
        output_window=base.output_window,
        predict_start=base.predict_start,
        hidden_size=int(hidden_size),
        encoder_num_layers=trial.suggest_int("encoder_num_layers", 1, 3),
        decoder_num_layers=trial.suggest_int("decoder_num_layers", 1, 2),
        num_heads=int(num_heads),
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        epochs=int(base.max_epochs),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        scheduled_sampling_decay_end=trial.suggest_int("scheduled_sampling_decay_end", 10, 30),
        scheduled_sampling_min_prob=trial.suggest_float("scheduled_sampling_min_prob", 0.3, 0.8),
        lr_patience=5,
        lr_factor=0.5,
        early_stopping_patience=10,
        max_grad_norm=1.0,
        calibration=trial.suggest_categorical("calibration", ["none", "horizon_isotonic"]),
        calibration_fit="train",
        seed=int(base.seed),
        device=base.device,
        prepare_data=False,
    )


def main() -> None:
    args = parse_args()
    window_dir = window_experiment_dir(args.output_root, args.input_window, args.output_window)
    tuning_dir = window_dir / "tuning" / "attention_lstm_seq2seq"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    storage_path = tuning_dir / "study.db"

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{storage_path}",
        study_name=args.study_name,
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        cfg = sample_config(trial, args)
        train_attention_lstm_seq2seq.run_training(cfg)
        eval_args = SimpleNamespace(
            model="attention_lstm_seq2seq",
            data_path=cfg.data_path,
            output_root=cfg.output_root,
            input_window=cfg.input_window,
            output_window=cfg.output_window,
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            stride=int(args.eval_stride),
            hidden_size=cfg.hidden_size,
            encoder_num_layers=cfg.encoder_num_layers,
            decoder_num_layers=cfg.decoder_num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            device=cfg.device,
            max_origins=int(args.max_origins),
        )
        agg = evaluate_rolling.run_evaluation(eval_args)
        return float(agg["R2_mean"])

    study.optimize(objective, n_trials=int(args.n_trials), gc_after_trial=True)

    best = study.best_trial
    (tuning_dir / "best_params.json").write_text(
        json.dumps({"value": best.value, "params": best.params}, indent=2),
        encoding="utf-8",
    )
    df = study.trials_dataframe()
    df.to_csv(tuning_dir / "trials.csv", index=False, encoding="utf-8")
    print(f"Best R2_mean: {best.value:.4f}")
    print(f"Best params: {best.params}")


if __name__ == "__main__":
    main()
