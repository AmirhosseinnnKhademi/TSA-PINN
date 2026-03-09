from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import h5py
import pandas as pd
import tensorflow as tf

from src.model import StandardPINN, TSAPINN
from src.utils import (
    CylinderWakeDataModule,
    TrainingError,
    ensure_directories,
    load_config,
    log_exceptions,
    save_history_csv,
    save_json,
    save_loss_plot,
    set_global_seed,
    setup_logger,
    timed,
)


class Trainer:
    """
    Training pipeline for Standard PINN and TSA-PINN.
    """

    def __init__(self, config_path: str) -> None:
        self.cfg = load_config(config_path)

        self.dtype = tf.float64 if self.cfg.model.dtype == "float64" else tf.float32

        ensure_directories(
            self.cfg.paths.outputs_dir,
            self.cfg.paths.checkpoints_dir,
        )

        self.logger = setup_logger(f"{self.cfg.paths.outputs_dir}/train.log")
        set_global_seed(self.cfg.seed)

        self.logger.info("Loaded configuration from %s", config_path)
        self.logger.info("Configuration: %s", asdict(self.cfg))

        self.data_module = CylinderWakeDataModule(
            data_file=self.cfg.paths.data_file,
            dtype=self.dtype,
            logger=self.logger,
        )

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.cfg.training.learning_rate
        )

    def _build_model(self):
        model_type = self.cfg.model.model_type.lower()

        if model_type == "standard":
            model = StandardPINN(
                layers=self.cfg.model.layers,
                reynolds=self.cfg.model.reynolds,
                dtype=self.dtype,
            )
            self.logger.info("Initialized StandardPINN.")
            return model

        if model_type == "tsa":
            if self.cfg.model.initial_frequency is None:
                raise TrainingError("TSA model requires 'initial_frequency' in config.")

            model = TSAPINN(
                layers=self.cfg.model.layers,
                reynolds=self.cfg.model.reynolds,
                initial_frequency=self.cfg.model.initial_frequency,
                dtype=self.dtype,
            )
            self.logger.info(
                "Initialized TSAPINN with initial_frequency=%.4f.",
                self.cfg.model.initial_frequency,
            )
            return model

        raise TrainingError(f"Unsupported model_type: {self.cfg.model.model_type}")

@tf.function
def train_step(
    self,
    x_train: tf.Tensor,
    y_train: tf.Tensor,
    t_train: tf.Tensor,
    u_train: tf.Tensor,
    v_train: tf.Tensor,
    x_res: tf.Tensor,
    y_res: tf.Tensor,
    t_res: tf.Tensor,
    x_test: tf.Tensor,
    y_test: tf.Tensor,
    t_test: tf.Tensor,
    u_test: tf.Tensor,
    v_test: tf.Tensor,
):
    with tf.GradientTape() as tape:
        result = self.model.compute_losses(
            x_train=x_train,
            y_train=y_train,
            t_train=t_train,
            u_train=u_train,
            v_train=v_train,
            x_res=x_res,
            y_res=y_res,
            t_res=t_res,
            x_test=x_test,
            y_test=y_test,
            t_test=t_test,
            u_test=u_test,
            v_test=v_test,
        )

    gradients = tape.gradient(result.loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    # Return only tensors / nested tensors
    return (
        result.loss,
        result.loss_data,
        result.loss_res,
        result.loss_test,
        result.loss_reg,
    )

    @log_exceptions
    @timed
    def run(self) -> None:
        dataset = self.data_module.load()

        sampled = self.data_module.sample_training_data(
            dataset=dataset,
            n_train_residuals=self.cfg.training.n_train_residuals,
            n_train_bcs=self.cfg.training.n_train_bcs,
            n_train_bcs_used=self.cfg.training.n_train_bcs_used,
        )

        history_rows: list[dict] = []

        for epoch in range(self.cfg.training.epochs):
            loss, loss_data, loss_res, loss_test, loss_reg = self.train_step(
                x_train=sampled["x_train"],
                y_train=sampled["y_train"],
                t_train=sampled["t_train"],
                u_train=sampled["u_train"],
                v_train=sampled["v_train"],
                x_res=sampled["x_res"],
                y_res=sampled["y_res"],
                t_res=sampled["t_res"],
                x_test=sampled["x_test"],
                y_test=sampled["y_test"],
                t_test=sampled["t_test"],
                u_test=sampled["u_test"],
                v_test=sampled["v_test"],
            )

            if epoch % self.cfg.training.eval_every == 0 or epoch == self.cfg.training.epochs - 1:
                row = {
                    "epoch": epoch,
                    "loss": float(loss.numpy()),
                    "loss_data": float(loss_data.numpy()),
                    "loss_res": float(loss_res.numpy()),
                    "loss_test": float(loss_test.numpy()),
                    "loss_reg": float(loss_reg.numpy()),
                }
                history_rows.append(row)

                self.logger.info(
                    (
                        "Epoch %d | total=%.6e | data=%.6e | residual=%.6e | "
                        "test=%.6e | reg=%.6e"
                    ),
                    epoch,
                    row["loss"],
                    row["loss_data"],
                    row["loss_res"],
                    row["loss_test"],
                    row["loss_reg"],
                )

        if not history_rows:
            raise TrainingError("No history was recorded during training.")

        history_df = pd.DataFrame(history_rows)

        history_path = Path(self.cfg.paths.outputs_dir) / "history.csv"
        metrics_path = Path(self.cfg.paths.outputs_dir) / "metrics.json"
        plot_path = Path(self.cfg.paths.outputs_dir) / "loss_curve.png"

        save_history_csv(history_df, str(history_path))
        save_loss_plot(history_df, str(plot_path))

        checkpoint_path = Path(self.cfg.paths.checkpoints_dir) / f"{self.model.name}.weights.h5"
        self._save_checkpoint(str(checkpoint_path))

        metrics = {
            "model_name": self.model.name,
            "model_type": self.cfg.model.model_type,
            "layers": self.cfg.model.layers,
            "reynolds": self.cfg.model.reynolds,
            "dtype": self.cfg.model.dtype,
            "epochs": self.cfg.training.epochs,
            "learning_rate": self.cfg.training.learning_rate,
            "n_train_residuals": self.cfg.training.n_train_residuals,
            "n_train_bcs": self.cfg.training.n_train_bcs,
            "n_train_bcs_used": self.cfg.training.n_train_bcs_used,
            "final_loss": float(history_df["loss"].iloc[-1]),
            "final_loss_data": float(history_df["loss_data"].iloc[-1]),
            "final_loss_res": float(history_df["loss_res"].iloc[-1]),
            "final_loss_test": float(history_df["loss_test"].iloc[-1]),
            "final_loss_reg": float(history_df["loss_reg"].iloc[-1]),
            "checkpoint_path": str(checkpoint_path),
        }

        if self.cfg.model.model_type.lower() == "tsa":
            metrics["initial_frequency"] = self.cfg.model.initial_frequency
            metrics["final_frequencies"] = [
                freq.numpy().tolist() for freq in self.model.freq
            ]

        save_json(metrics, str(metrics_path))

        self.logger.info("Training completed successfully.")
        self.logger.info("Saved history to %s", history_path)
        self.logger.info("Saved metrics to %s", metrics_path)
        self.logger.info("Saved loss plot to %s", plot_path)
        self.logger.info("Saved checkpoint to %s", checkpoint_path)

    def _save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save model weights/biases/frequencies into an HDF5 file.
        This keeps the implementation simple and portable.
        """
        out_path = Path(checkpoint_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as h5f:
            h5f.attrs["model_name"] = self.model.name
            h5f.attrs["dtype"] = str(self.dtype.name)

            weights_group = h5f.create_group("weights")
            biases_group = h5f.create_group("biases")

            for i, weight in enumerate(self.model.weights):
                weights_group.create_dataset(f"W_{i}", data=weight.numpy())

            for i, bias in enumerate(self.model.biases):
                biases_group.create_dataset(f"b_{i}", data=bias.numpy())

            if hasattr(self.model, "freq"):
                freq_group = h5f.create_group("frequencies")
                for i, freq in enumerate(self.model.freq):
                    freq_group.create_dataset(f"freq_{i}", data=freq.numpy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Standard PINN or TSA-PINN for cylinder wake flow."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(config_path=args.config)
    trainer.run()