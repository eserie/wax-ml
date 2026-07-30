# Copyright 2021 The WAX-ML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flax-based SNARIMAX module for adaptive ARMA forecasting."""

from collections.abc import Callable
from typing import cast

import flax.linen as nn
import jax.numpy as jnp

from .buffer import Buffer
from .fill_nan_inf import FillNanInf


class SNARIMAX(nn.Module):
    """Flax-based SNARIMAX (Adaptive ARMA) forecasting module.

    Implements an adaptive ARMA model that can incorporate exogenous regressors
    for time-series forecasting. SNARIMAX stands for Seasonal Non-stationary
    Autoregressive Integrated Moving Average with eXogenous regressors.
    """

    lags_x: int = 1  # Number of autoregressive lags for x
    lags_y: int = 1  # Number of moving average lags for y (prediction errors)
    # The regressor maps a feature matrix to predictions, so it yields an array.
    regressor: Callable[..., jnp.ndarray] | None = None

    def setup(self) -> None:
        """Setup the SNARIMAX module."""
        # Create buffers for lagged values
        if self.lags_x > 0:
            self.buffer_x = Buffer(maxlen=self.lags_x, fill_value=0.0)

        if self.lags_y > 0:
            self.buffer_y = Buffer(maxlen=self.lags_y, fill_value=0.0)

        # Create regressor if not provided
        self.regressor_fn: Callable[..., jnp.ndarray]
        if self.regressor is None:
            # Default to a simple linear layer
            self.regressor_fn = nn.Dense(features=1, use_bias=True)
        else:
            self.regressor_fn = self.regressor

        # Module for handling NaN/Inf values
        self.fill_nan_inf = FillNanInf()

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray | None = None) -> jnp.ndarray:
        """Apply SNARIMAX forecasting model.

        Args:
            x: Input features/exogenous variables
            y: Target values (for computing prediction errors), optional

        Returns:
            Forecasted/predicted values
        """
        # Ensure input is an array
        x = jnp.asarray(x)

        # Initialize features list
        features = []

        # Add lagged x values (autoregressive component)
        if self.lags_x > 0:
            # buffer_x is built with return_state left at its default (False).
            x_lagged = cast(jnp.ndarray, self.buffer_x(x))
            # Flatten the lagged values and add to features
            features.append(x_lagged.flatten())

        # Add lagged prediction errors (moving average component)
        if self.lags_y > 0 and y is not None:
            # Compute prediction error if y is provided
            # For now, use a simple prediction (can be improved with actual model output)
            prediction_error = y - x  # Simple error approximation
            y_lagged = cast(jnp.ndarray, self.buffer_y(prediction_error))
            # Flatten the lagged values and add to features
            features.append(y_lagged.flatten())

        # Add current x value
        if x.ndim == 0:
            # Scalar input
            features.append(jnp.array([x]))
        else:
            # Vector input
            features.append(x.flatten())

        # Concatenate all features
        if features:
            feature_vector = jnp.concatenate(features)
        else:
            feature_vector = jnp.array([x]) if x.ndim == 0 else x.flatten()

        # Ensure feature vector is 2D for the regressor
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Apply regressor to get prediction
        prediction = self.regressor_fn(feature_vector)

        # Handle output shape
        if prediction.ndim > 1:
            prediction = prediction.squeeze()

        # Handle NaN/Inf values
        prediction = self.fill_nan_inf(prediction)

        return prediction


def create_snarimax(
    lags_x: int = 1,
    lags_y: int = 1,
    regressor: Callable[..., jnp.ndarray] | None = None,
) -> SNARIMAX:
    """Factory function to create SNARIMAX module.

    Args:
        lags_x: Number of autoregressive lags for x
        lags_y: Number of moving average lags for y (prediction errors)
        regressor: Optional regressor function

    Returns:
        SNARIMAX module instance
    """
    return SNARIMAX(
        lags_x=lags_x,
        lags_y=lags_y,
        regressor=regressor,
    )
