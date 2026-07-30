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
"""Flax-based ARMA module for Autoregressive Moving Average filtering."""

from typing import cast

import flax.linen as nn
import jax.numpy as jnp

from .buffer import Buffer
from .fill_nan_inf import FillNanInf


class ARMA(nn.Module):
    """Flax-based ARMA (Autoregressive Moving Average) linear filter.

    Implements an ARMA filter that combines autoregressive (AR) and
    moving average (MA) components for time-series processing.

    The ARMA model is defined as:
    y[t] = sum(alpha[i] * y[t-i]) + sum(beta[j] * eps[t-j])

    where:
    - alpha: AR coefficients
    - beta: MA coefficients
    - eps: Input noise/error terms
    """

    alpha: jnp.ndarray  # AR coefficients
    beta: jnp.ndarray  # MA coefficients

    def setup(self) -> None:
        """Setup the ARMA module."""
        # Validate coefficients
        if self.alpha.ndim != 1:
            raise ValueError("alpha must be a 1-D array")
        if self.beta.ndim != 1:
            raise ValueError("beta must be a 1-D array")

        # Create buffers for AR and MA components
        # AR buffer stores previous outputs
        if len(self.alpha) > 0:
            self.ar_buffer = Buffer(maxlen=len(self.alpha), fill_value=0.0)

        # MA buffer stores previous inputs (noise terms)
        if len(self.beta) > 0:
            self.ma_buffer = Buffer(maxlen=len(self.beta), fill_value=0.0)

        # Module for handling NaN/Inf values
        self.fill_nan_inf = FillNanInf()

    def __call__(self, eps: jnp.ndarray) -> jnp.ndarray:
        """Apply ARMA filter to input noise term.

        Args:
            eps: Input noise/error term

        Returns:
            Filtered output from ARMA model
        """
        # Ensure input is an array
        eps = jnp.asarray(eps)

        # Initialize output with current input (eps term)
        output = eps

        # Autoregressive component: sum(alpha[i] * y[t-i])
        if len(self.alpha) > 0:
            # Get previous outputs from AR buffer (placeholder initialization)
            # Get current state; the buffer is built with return_state left at
            # its default (False), so the call yields the array alone.
            ar_history = cast(jnp.ndarray, self.ar_buffer(jnp.zeros_like(output)))

            # Compute AR contribution
            ar_contribution = jnp.sum(self.alpha * ar_history)
            output = output + ar_contribution

        # Moving average component: sum(beta[j] * eps[t-j])
        if len(self.beta) > 0:
            # Get previous inputs from MA buffer
            ma_history = cast(jnp.ndarray, self.ma_buffer(eps))

            # Compute MA contribution
            ma_contribution = jnp.sum(self.beta * ma_history)
            output = output + ma_contribution

        # Update AR buffer with current output
        if len(self.alpha) > 0:
            self.ar_buffer(output)  # Store current output for next iteration

        # Handle NaN/Inf values
        output = self.fill_nan_inf(output)

        return output


def create_arma(alpha: jnp.ndarray, beta: jnp.ndarray) -> ARMA:
    """Factory function to create ARMA module.

    Args:
        alpha: AR coefficients (1-D array)
        beta: MA coefficients (1-D array)

    Returns:
        ARMA module instance
    """
    return ARMA(alpha=alpha, beta=beta)
