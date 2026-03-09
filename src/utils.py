from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
import yaml


# ============================================================
# Custom Exceptions
# ============================================================

class ProjectError(Exception):
    """Base exception for the project."""


class ConfigurationError(ProjectError):
    """Raised when the YAML configuration is invalid or incomplete."""


class DataLoadingError(ProjectError):
    """Raised when loading the dataset fails."""


class TrainingError(ProjectError):
    """Raised when training fails."""


class EvaluationError(ProjectError):
    """Raised when evaluation fails."""


# ============================================================
# Logging
# ============================================================

def setup_logger(log_file: str = "outputs/run.log", level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that writes both to console and to a file.
    The file is overwritten each run.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("tsa_pinn")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


# ============================================================
# Decorators
# ============================================================

def timed(func: Callable) -> Callable:
    """Decorator to log execution time."""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        logger = kwargs.get("logger", None)
        if logger is None and len(args) > 0 and hasattr(args[0], "logger"):
            logger = args[0].logger

        if logger is not None:
            logger.info("Function '%s' finished in %.3f s", func.__name__, elapsed)
        else:
            print(f"[TIMER] {func.__name__}: {elapsed:.3f} s")

        return result

    return wrapper


def log_exceptions(func: Callable) -> Callable:
    """Decorator to log uncaught exceptions."""

    def wrapper(*args, **kwargs):
        logger = kwargs.get("logger", None)
        if logger is None and len(args) > 0 and hasattr(args[0], "logger"):
            logger = args[0].logger

        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if logger is not None:
                logger.exception("Unhandled exception in '%s': %s", func.__name__, exc)
            raise

    return wrapper


# ============================================================
# Reproducibility
# ============================================================

def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# Config Dataclasses
# ============================================================

@dataclass
class PathsConfig:
    data_file: str
    outputs_dir: str
    checkpoints_dir: str


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    n_train_residuals: int
    n_train_bcs: int
    n_train_bcs_used: int
    eval_every: int
    snapshot_index: int
    x_probe: float
    y_probe: float


@dataclass
class ModelConfig:
    model_type: str
    layers: list[int]
    dtype: str
    reynolds: float
    initial_frequency: float | None = None


@dataclass
class Config:
    seed: int
    paths: PathsConfig
    training: TrainingConfig
    model: ModelConfig


@log_exceptions
def load_config(config_path: str) -> Config:
    """Load a YAML config file into structured dataclasses."""
    path = Path(config_path)

    if not path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        raise ConfigurationError(f"Failed to read config file: {config_path}") from exc

    try:
        cfg = Config(
            seed=raw["seed"],
            paths=PathsConfig(**raw["paths"]),
            training=TrainingConfig(**raw["training"]),
            model=ModelConfig(**raw["model"]),
        )
    except KeyError as exc:
        raise ConfigurationError(f"Missing configuration key: {exc}") from exc
    except TypeError as exc:
        raise ConfigurationError(f"Invalid configuration structure in: {config_path}") from exc

    if cfg.training.n_train_bcs_used > cfg.training.n_train_bcs:
        raise ConfigurationError(
            "n_train_bcs_used cannot be greater than n_train_bcs."
        )

    return cfg


# ============================================================
# Dataset Dataclass
# ============================================================

@dataclass
class CylinderWakeDataset:
    X_star: np.ndarray
    U_star: np.ndarray
    P_star: np.ndarray
    t_star: np.ndarray

    XX: np.ndarray
    YY: np.ndarray
    TT: np.ndarray
    UU: np.ndarray
    VV: np.ndarray
    PP: np.ndarray

    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray


# ============================================================
# Data Module
# ============================================================

class CylinderWakeDataModule:
    """
    Handles loading and preprocessing of the cylinder wake dataset.
    """

    def __init__(self, data_file: str, dtype: tf.dtypes.DType, logger: logging.Logger):
        self.data_file = data_file
        self.dtype = dtype
        self.logger = logger

    @log_exceptions
    @timed
    def load(self) -> CylinderWakeDataset:
        """
        Load the MATLAB dataset and flatten it into training-ready arrays.
        """
        data_path = Path(self.data_file)

        if not data_path.exists():
            raise DataLoadingError(f"Dataset file not found: {self.data_file}")

        try:
            data = scipy.io.loadmat(data_path)
        except Exception as exc:
            raise DataLoadingError(f"Failed to load MAT file: {self.data_file}") from exc

        try:
            X_star = data["X_star"]      # (N, 2)
            U_star = data["U_star"]      # (N, 2, T)
            P_star = data["p_star"]      # (N, T)
            t_star = data["t"]           # (T, 1)
        except KeyError as exc:
            raise DataLoadingError(f"Missing expected dataset key: {exc}") from exc

        N = X_star.shape[0]
        T = t_star.shape[0]

        XX = np.tile(X_star[:, 0:1], (1, T))
        YY = np.tile(X_star[:, 1:2], (1, T))
        TT = np.tile(t_star, (1, N)).T

        UU = U_star[:, 0, :]
        VV = U_star[:, 1, :]
        PP = P_star

        x = XX.flatten()[:, None]
        y = YY.flatten()[:, None]
        t = TT.flatten()[:, None]
        u = UU.flatten()[:, None]
        v = VV.flatten()[:, None]
        p = PP.flatten()[:, None]

        self.logger.info("Loaded dataset from %s", self.data_file)
        self.logger.info("Spatial points: %d | Time steps: %d | Total samples: %d", N, T, N * T)

        return CylinderWakeDataset(
            X_star=X_star,
            U_star=U_star,
            P_star=P_star,
            t_star=t_star,
            XX=XX,
            YY=YY,
            TT=TT,
            UU=UU,
            VV=VV,
            PP=PP,
            x=x,
            y=y,
            t=t,
            u=u,
            v=v,
            p=p,
        )

    @log_exceptions
    def sample_training_data(
        self,
        dataset: CylinderWakeDataset,
        n_train_residuals: int,
        n_train_bcs: int,
        n_train_bcs_used: int,
    ) -> dict[str, tf.Tensor]:
        """
        Reproduce the notebook logic:
        - sample residual points
        - sample BC/data points
        - use only the first n_train_bcs_used for training
        - keep the rest as held-out data
        """
        total_samples = dataset.x.shape[0]

        if n_train_residuals > total_samples:
            raise DataLoadingError("n_train_residuals exceeds total number of available samples.")

        if n_train_bcs > total_samples:
            raise DataLoadingError("n_train_bcs exceeds total number of available samples.")

        if n_train_bcs_used > n_train_bcs:
            raise DataLoadingError("n_train_bcs_used cannot exceed n_train_bcs.")

        idx_residuals = np.random.choice(total_samples, n_train_residuals, replace=False)
        idx_bcs = np.random.choice(total_samples, n_train_bcs, replace=False)

        train_idx = idx_bcs[:n_train_bcs_used]
        test_idx = idx_bcs[n_train_bcs_used:]

        sampled = {
            # residual points
            "x_res": tf.convert_to_tensor(dataset.x[idx_residuals], dtype=self.dtype),
            "y_res": tf.convert_to_tensor(dataset.y[idx_residuals], dtype=self.dtype),
            "t_res": tf.convert_to_tensor(dataset.t[idx_residuals], dtype=self.dtype),

            # train data points
            "x_train": tf.convert_to_tensor(dataset.x[train_idx], dtype=self.dtype),
            "y_train": tf.convert_to_tensor(dataset.y[train_idx], dtype=self.dtype),
            "t_train": tf.convert_to_tensor(dataset.t[train_idx], dtype=self.dtype),
            "u_train": tf.convert_to_tensor(dataset.u[train_idx], dtype=self.dtype),
            "v_train": tf.convert_to_tensor(dataset.v[train_idx], dtype=self.dtype),

            # held-out data points
            "x_test": tf.convert_to_tensor(dataset.x[test_idx], dtype=self.dtype),
            "y_test": tf.convert_to_tensor(dataset.y[test_idx], dtype=self.dtype),
            "t_test": tf.convert_to_tensor(dataset.t[test_idx], dtype=self.dtype),
            "u_test": tf.convert_to_tensor(dataset.u[test_idx], dtype=self.dtype),
            "v_test": tf.convert_to_tensor(dataset.v[test_idx], dtype=self.dtype),
        }

        self.logger.info(
            "Sampled %d residual points, %d BC/data points (%d used for training, %d held out for test).",
            n_train_residuals,
            n_train_bcs,
            n_train_bcs_used,
            n_train_bcs - n_train_bcs_used,
        )

        return sampled

    def get_snapshot(self, dataset: CylinderWakeDataset, snap: int) -> dict[str, np.ndarray]:
        """
        Get one snapshot at time index 'snap' for evaluation/plotting.
        """
        return {
            "x_star": dataset.X_star[:, 0:1],
            "y_star": dataset.X_star[:, 1:2],
            "t_star": dataset.TT[:, snap:snap + 1],
            "u_star": dataset.U_star[:, 0, snap:snap + 1],
            "v_star": dataset.U_star[:, 1, snap:snap + 1],
            "p_star": dataset.P_star[:, snap:snap + 1],
        }

    def get_probe_series(
        self,
        dataset: CylinderWakeDataset,
        x_sample: float,
        y_sample: float,
    ) -> dict[str, tf.Tensor]:
        """
        Extract the time series for the closest spatial point to (x_sample, y_sample).
        """
        distances = np.sqrt(
            (dataset.XX[:, 0] - x_sample) ** 2 +
            (dataset.YY[:, 0] - y_sample) ** 2
        )
        idx = np.argmin(distances)

        x_selected = dataset.XX[idx, 0]
        y_selected = dataset.YY[idx, 0]
        t_selected = dataset.TT[0, :]
        u_selected = dataset.UU[idx, :]
        v_selected = dataset.VV[idx, :]

        n_time = len(t_selected)

        return {
            "x_selected": tf.convert_to_tensor(np.full((n_time, 1), x_selected), dtype=self.dtype),
            "y_selected": tf.convert_to_tensor(np.full((n_time, 1), y_selected), dtype=self.dtype),
            "t_selected": tf.convert_to_tensor(t_selected[:, None], dtype=self.dtype),
            "u_selected": tf.convert_to_tensor(u_selected[:, None], dtype=self.dtype),
            "v_selected": tf.convert_to_tensor(v_selected[:, None], dtype=self.dtype),
        }


# ============================================================
# Output Helpers
# ============================================================

def ensure_directories(*dirs: str) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: str) -> None:
    """Save a dictionary to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_history_csv(history: pd.DataFrame, path: str) -> None:
    """Save training history as CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_path, index=False)


def save_loss_plot(history: pd.DataFrame, path: str) -> None:
    """Save the training loss curve."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["loss"], label="Total Loss")
    plt.plot(history["epoch"], history["loss_data"], label="Data Loss")
    plt.plot(history["epoch"], history["loss_res"], label="Residual Loss")
    if "loss_test" in history.columns:
        plt.plot(history["epoch"], history["loss_test"], label="Test Loss")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()