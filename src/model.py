from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class ForwardResult:
    """
    Container for model outputs and losses.
    """
    u: tf.Tensor
    v: tf.Tensor
    p: tf.Tensor
    ru: tf.Tensor
    rv: tf.Tensor
    loss: tf.Tensor
    loss_data: tf.Tensor
    loss_res: tf.Tensor
    loss_test: tf.Tensor
    loss_reg: tf.Tensor


class BasePINN(ABC):
    """
    Abstract base class for PINN models based on the streamfunction-pressure formulation.

    Shared responsibilities:
    - network initialization
    - streamfunction/pressure interpretation
    - automatic differentiation for u, v
    - Navier-Stokes residual computation
    - common loss computation
    """

    def __init__(
        self,
        layers: list[int],
        reynolds: float,
        dtype: tf.dtypes.DType = tf.float64,
        name: str = "BasePINN",
    ) -> None:
        self.layers = layers
        self.reynolds = tf.constant(reynolds, dtype=dtype)
        self.dtype = dtype
        self.name = name

        self.weights, self.biases = self._initialize_network(self.layers)

    # ============================================================
    # Initialization
    # ============================================================

    @staticmethod
    def xavier_init(in_dim: int, out_dim: int, dtype: tf.dtypes.DType) -> tf.Variable:
        stddev = np.sqrt(2.0 / (in_dim + out_dim))
        values = tf.random.normal([in_dim, out_dim], stddev=stddev, dtype=dtype)
        return tf.Variable(values, dtype=dtype, trainable=True)

    def _initialize_network(self, layers: list[int]) -> tuple[list[tf.Variable], list[tf.Variable]]:
        weights: list[tf.Variable] = []
        biases: list[tf.Variable] = []

        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            W = self.xavier_init(in_dim, out_dim, self.dtype)
            b = tf.Variable(tf.zeros([1, out_dim], dtype=self.dtype), trainable=True)
            weights.append(W)
            biases.append(b)

        return weights, biases

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return self.weights + self.biases + self.extra_trainables()

    def extra_trainables(self) -> list[tf.Variable]:
        """
        Subclasses can override this to add extra trainable variables
        such as TSA frequency parameters.
        """
        return []

    # ============================================================
    # Abstract architecture hook
    # ============================================================

    @abstractmethod
    def forward_network(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Subclasses define the network architecture here.
        Input shape: (N, 3) -> [x, y, t]
        Output shape: (N, 2) -> [psi, p]
        """
        raise NotImplementedError

    # ============================================================
    # Physics helpers
    # ============================================================

    def predict_stream_pressure(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        outputs = self.forward_network(inputs)
        psi = outputs[:, 0:1]
        p = outputs[:, 1:2]
        return psi, p

    def compute_flow_quantities(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        t: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute:
        - u, v, p
        - residuals ru, rv
        using automatic differentiation.
        """

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])

            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, t])

                with tf.GradientTape(persistent=True) as tape0:
                    tape0.watch([x, y, t])

                    inputs = tf.concat([x, y, t], axis=1)
                    psi, p = self.predict_stream_pressure(inputs)

                u = tape0.gradient(psi, y)
                v = -tape0.gradient(psi, x)
                p_x = tape0.gradient(p, x)
                p_y = tape0.gradient(p, y)

            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            u_t = tape1.gradient(u, t)

            v_x = tape1.gradient(v, x)
            v_y = tape1.gradient(v, y)
            v_t = tape1.gradient(v, t)

        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)

        del tape0
        del tape1
        del tape2

        ru = u_t + u * u_x + v * u_y + p_x - (u_xx + u_yy) / self.reynolds
        rv = v_t + u * v_x + v * v_y + p_y - (v_xx + v_yy) / self.reynolds

        return u, v, p, ru, rv

    # ============================================================
    # Losses
    # ============================================================

    def regularization_loss(self) -> tf.Tensor:
        """
        Default: no extra regularization.
        TSA-PINN overrides this.
        """
        return tf.constant(0.0, dtype=self.dtype)

    def compute_losses(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        t_train: tf.Tensor,
        u_train: tf.Tensor,
        v_train: tf.Tensor,
        x_res: tf.Tensor,
        y_res: tf.Tensor,
        t_res: tf.Tensor,
        x_test: tf.Tensor | None = None,
        y_test: tf.Tensor | None = None,
        t_test: tf.Tensor | None = None,
        u_test: tf.Tensor | None = None,
        v_test: tf.Tensor | None = None,
    ) -> ForwardResult:
        """
        Compute notebook-style losses:
        - data loss on training data
        - residual loss on sampled residual points
        - optional held-out data loss
        - TSA regularization term if applicable
        """

        # Data loss on train points
        u_pred_train, v_pred_train, p_pred_train, _, _ = self.compute_flow_quantities(
            x_train, y_train, t_train
        )
        loss_data = tf.reduce_mean(
            tf.square(u_pred_train - u_train) + tf.square(v_pred_train - v_train)
        )

        # Residual loss
        _, _, _, ru, rv = self.compute_flow_quantities(x_res, y_res, t_res)
        loss_res = tf.reduce_mean(tf.square(ru) + tf.square(rv))

        # Test loss
        if all(item is not None for item in [x_test, y_test, t_test, u_test, v_test]):
            u_pred_test, v_pred_test, _, _, _ = self.compute_flow_quantities(
                x_test, y_test, t_test
            )
            loss_test = tf.reduce_mean(
                tf.square(u_pred_test - u_test) + tf.square(v_pred_test - v_test)
            )
        else:
            loss_test = tf.constant(0.0, dtype=self.dtype)

        # Regularization
        loss_reg = self.regularization_loss()

        # Total
        loss = loss_data + loss_res + loss_reg

        return ForwardResult(
            u=u_pred_train,
            v=v_pred_train,
            p=p_pred_train,
            ru=ru,
            rv=rv,
            loss=loss,
            loss_data=loss_data,
            loss_res=loss_res,
            loss_test=loss_test,
            loss_reg=loss_reg,
        )


class StandardPINN(BasePINN):
    """
    Baseline PINN aligned with the notebook.

    Important notebook detail:
    the baseline applies tanh to every layer, including the final layer.
    That behavior is preserved here intentionally.
    """

    def __init__(
        self,
        layers: list[int],
        reynolds: float,
        dtype: tf.dtypes.DType = tf.float64,
    ) -> None:
        super().__init__(
            layers=layers,
            reynolds=reynolds,
            dtype=dtype,
            name="StandardPINN",
        )

    def forward_network(self, inputs: tf.Tensor) -> tf.Tensor:
        h = inputs

        # Notebook-faithful baseline:
        # tanh is applied on all layers, including the last one.
        for W, b in zip(self.weights, self.biases):
            h = tf.tanh(tf.matmul(h, W) + b)

        return h


class TSAPINN(BasePINN):
    """
    TSA-PINN aligned with the notebook.

    Hidden layers:
        0.5 * (sin(freq * z + b) + cos(freq * z + b))

    Output layer:
        linear

    Frequency tensors are trainable and sized as (1, layer_width)
    for each hidden layer.
    """

    def __init__(
        self,
        layers: list[int],
        reynolds: float,
        initial_frequency: float,
        dtype: tf.dtypes.DType = tf.float64,
    ) -> None:
        super().__init__(
            layers=layers,
            reynolds=reynolds,
            dtype=dtype,
            name="TSAPINN",
        )

        # One trainable frequency tensor per hidden layer
        self.freq = [
            tf.Variable(
                tf.constant(initial_frequency, shape=(1, layers[i + 1]), dtype=dtype),
                trainable=True,
                name=f"freq_{i}",
            )
            for i in range(len(layers) - 2)
        ]

    def extra_trainables(self) -> list[tf.Variable]:
        return self.freq

    def forward_network(self, inputs: tf.Tensor) -> tf.Tensor:
        h = inputs

        # Hidden layers: adaptive sin/cos
        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = tf.matmul(h, W)
            h = 0.5 * (
                tf.sin(self.freq[i] * z + b) +
                tf.cos(self.freq[i] * z + b)
            )

        # Final layer: linear
        outputs = tf.matmul(h, self.weights[-1]) + self.biases[-1]
        return outputs

    def regularization_loss(self) -> tf.Tensor:
        """
        Notebook-style TSA regularization:
            1 / sum(exp(mean(freq_i)))
        """
        reg_sum = tf.add_n([tf.exp(tf.reduce_mean(f_i)) for f_i in self.freq])
        return 1.0 / reg_sum