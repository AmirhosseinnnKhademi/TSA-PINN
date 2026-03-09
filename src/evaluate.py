from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.interpolate import griddata

from src.model import StandardPINN, TSAPINN
from src.utils import (
    CylinderWakeDataModule,
    EvaluationError,
    ensure_directories,
    load_config,
    log_exceptions,
    save_json,
    setup_logger,
    timed,
)


class Evaluator:
    """
    Evaluation pipeline for Standard PINN and TSA-PINN.
    """

    def __init__(self, config_path: str) -> None:
        self.cfg = load_config(config_path)
        self.dtype = tf.float64 if self.cfg.model.dtype == "float64" else tf.float32

        ensure_directories(self.cfg.paths.outputs_dir)

        self.logger = setup_logger(f"{self.cfg.paths.outputs_dir}/evaluate.log")

        self.data_module = CylinderWakeDataModule(
            data_file=self.cfg.paths.data_file,
            dtype=self.dtype,
            logger=self.logger,
        )

        self.model = self._build_model()

        checkpoint_path = Path(self.cfg.paths.checkpoints_dir) / f"{self.model.name}.weights.h5"
        self._load_checkpoint(str(checkpoint_path))

    def _build_model(self):
        model_type = self.cfg.model.model_type.lower()

        if model_type == "standard":
            model = StandardPINN(
                layers=self.cfg.model.layers,
                reynolds=self.cfg.model.reynolds,
                dtype=self.dtype,
            )
            self.logger.info("Rebuilt StandardPINN for evaluation.")
            return model

        if model_type == "tsa":
            model = TSAPINN(
                layers=self.cfg.model.layers,
                reynolds=self.cfg.model.reynolds,
                initial_frequency=self.cfg.model.initial_frequency,
                dtype=self.dtype,
            )
            self.logger.info(
                "Rebuilt TSAPINN for evaluation with initial_frequency=%.4f.",
                self.cfg.model.initial_frequency,
            )
            return model

        raise EvaluationError(f"Unsupported model_type: {self.cfg.model.model_type}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = Path(checkpoint_path)

        if not checkpoint.exists():
            raise EvaluationError(f"Checkpoint file not found: {checkpoint_path}")

        with h5py.File(checkpoint, "r") as h5f:
            weights_group = h5f["weights"]
            biases_group = h5f["biases"]

            for i, weight in enumerate(self.model.weights):
                weight.assign(weights_group[f"W_{i}"][:])

            for i, bias in enumerate(self.model.biases):
                bias.assign(biases_group[f"b_{i}"][:])

            if hasattr(self.model, "freq") and "frequencies" in h5f:
                freq_group = h5f["frequencies"]
                for i, freq in enumerate(self.model.freq):
                    freq.assign(freq_group[f"freq_{i}"][:])

        self.logger.info("Loaded checkpoint from %s", checkpoint_path)

    @log_exceptions
    @timed
    def evaluate_snapshot(self) -> None:
        """
        Evaluate one selected time snapshot and save relative L2 errors + contour plots.
        """
        dataset = self.data_module.load()
        snap = self.cfg.training.snapshot_index

        snapshot = self.data_module.get_snapshot(dataset, snap=snap)

        x_star = tf.convert_to_tensor(snapshot["x_star"], dtype=self.dtype)
        y_star = tf.convert_to_tensor(snapshot["y_star"], dtype=self.dtype)
        t_star = tf.convert_to_tensor(snapshot["t_star"], dtype=self.dtype)

        u_true = snapshot["u_star"]
        v_true = snapshot["v_star"]

        u_pred, v_pred, _, _, _ = self.model.compute_flow_quantities(x_star, y_star, t_star)

        u_pred_np = u_pred.numpy()
        v_pred_np = v_pred.numpy()

        error_u = np.linalg.norm(u_true - u_pred_np, 2) / np.linalg.norm(u_true, 2)
        error_v = np.linalg.norm(v_true - v_pred_np, 2) / np.linalg.norm(v_true, 2)

        metrics = {
            "snapshot_index": int(snap),
            "relative_l2_error_u": float(error_u),
            "relative_l2_error_v": float(error_v),
        }

        metrics_path = Path(self.cfg.paths.outputs_dir) / "eval_snapshot_metrics.json"
        save_json(metrics, str(metrics_path))

        self.logger.info(
            "Snapshot %d evaluation | rel L2 error u = %.6e | rel L2 error v = %.6e",
            snap,
            error_u,
            error_v,
        )

        self._plot_snapshot_comparison(
            x_star=snapshot["x_star"],
            y_star=snapshot["y_star"],
            u_true=u_true,
            v_true=v_true,
            u_pred=u_pred_np,
            v_pred=v_pred_np,
        )

    def _plot_snapshot_comparison(
        self,
        x_star: np.ndarray,
        y_star: np.ndarray,
        u_true: np.ndarray,
        v_true: np.ndarray,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
    ) -> None:
        """
        Save 2x2 contour comparison plot for u and v.
        """
        nn = 200
        x_min, x_max = np.min(x_star), np.max(x_star)
        y_min, y_max = np.min(y_star), np.max(y_star)

        grid_x = np.linspace(x_min, x_max, nn)
        grid_y = np.linspace(y_min, y_max, nn)
        X, Y = np.meshgrid(grid_x, grid_y)

        U_true_grid = griddata(
            (x_star.flatten(), y_star.flatten()),
            u_true.flatten(),
            (X, Y),
            method="cubic",
        )
        U_pred_grid = griddata(
            (x_star.flatten(), y_star.flatten()),
            u_pred.flatten(),
            (X, Y),
            method="cubic",
        )
        V_true_grid = griddata(
            (x_star.flatten(), y_star.flatten()),
            v_true.flatten(),
            (X, Y),
            method="cubic",
        )
        V_pred_grid = griddata(
            (x_star.flatten(), y_star.flatten()),
            v_pred.flatten(),
            (X, Y),
            method="cubic",
        )

        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        plots = [
            (axs[0, 0], U_true_grid, "Reference u"),
            (axs[0, 1], U_pred_grid, "Predicted u"),
            (axs[1, 0], V_true_grid, "Reference v"),
            (axs[1, 1], V_pred_grid, "Predicted v"),
        ]

        for ax, field, title in plots:
            contour = ax.contourf(X, Y, field, levels=100, cmap="rainbow")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(contour, ax=ax)

        plt.tight_layout()
        out_path = Path(self.cfg.paths.outputs_dir) / "snapshot_comparison.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info("Saved snapshot comparison plot to %s", out_path)

    @log_exceptions
    @timed
    def evaluate_probe(self) -> None:
        """
        Evaluate the velocity time history at the configured probe location.
        """
        dataset = self.data_module.load()

        probe = self.data_module.get_probe_series(
            dataset=dataset,
            x_sample=self.cfg.training.x_probe,
            y_sample=self.cfg.training.y_probe,
        )

        u_pred, v_pred, _, _, _ = self.model.compute_flow_quantities(
            probe["x_selected"],
            probe["y_selected"],
            probe["t_selected"],
        )

        u_pred_np = u_pred.numpy()
        v_pred_np = v_pred.numpy()
        u_true_np = probe["u_selected"].numpy()
        v_true_np = probe["v_selected"].numpy()
        t_true_np = probe["t_selected"].numpy()

        error_u = np.linalg.norm(u_true_np - u_pred_np, 2) / np.linalg.norm(u_true_np, 2)
        error_v = np.linalg.norm(v_true_np - v_pred_np, 2) / np.linalg.norm(v_true_np, 2)

        metrics = {
            "x_probe": float(self.cfg.training.x_probe),
            "y_probe": float(self.cfg.training.y_probe),
            "relative_l2_error_u_probe": float(error_u),
            "relative_l2_error_v_probe": float(error_v),
        }

        metrics_path = Path(self.cfg.paths.outputs_dir) / "eval_probe_metrics.json"
        save_json(metrics, str(metrics_path))

        self.logger.info(
            "Probe evaluation at (%.3f, %.3f) | rel L2 error u = %.6e | rel L2 error v = %.6e",
            self.cfg.training.x_probe,
            self.cfg.training.y_probe,
            error_u,
            error_v,
        )

        self._plot_probe_series(
            t_true=t_true_np,
            u_true=u_true_np,
            v_true=v_true_np,
            u_pred=u_pred_np,
            v_pred=v_pred_np,
        )

    def _plot_probe_series(
        self,
        t_true: np.ndarray,
        u_true: np.ndarray,
        v_true: np.ndarray,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
    ) -> None:
        """
        Save probe comparison plots for u(t) and v(t).
        """
        out_u = Path(self.cfg.paths.outputs_dir) / "probe_u_vs_time.png"
        out_v = Path(self.cfg.paths.outputs_dir) / "probe_v_vs_time.png"

        plt.figure(figsize=(10, 5))
        plt.plot(t_true, u_pred, label="Predicted u")
        plt.plot(t_true, u_true, "--", label="Reference u")
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Probe Velocity Comparison: u(t)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_u, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(t_true, v_pred, label="Predicted v")
        plt.plot(t_true, v_true, "--", label="Reference v")
        plt.xlabel("t")
        plt.ylabel("v")
        plt.title("Probe Velocity Comparison: v(t)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_v, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info("Saved probe plots to %s and %s", out_u, out_v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Standard PINN or TSA-PINN for cylinder wake flow."
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
    evaluator = Evaluator(config_path=args.config)
    evaluator.evaluate_snapshot()
    evaluator.evaluate_probe()