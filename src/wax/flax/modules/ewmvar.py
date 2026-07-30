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
"""Flax-based Exponentially Weighted Moving Variance module."""

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers


class EWMVar(nn.Module):
    """Flax-based Exponentially Weighted Moving Variance module.

    Computes exponentially weighted variance using the incremental formula:
    Var(X) = Mean(x^2) - Mean(x)^2

    References:
    Finch, T., 2009. Incremental calculation of weighted mean and variance.
    """

    com: float | None = None
    alpha: float | None = None
    adjust: bool | str = True

    def setup(self):
        """Setup the EWMVar module parameters."""
        # Validate parameters (same logic as Haiku version)
        assert self.com is not None or self.alpha is not None, (
            "com or alpha parameters must be specified."
        )

        if self.com is not None:
            assert self.alpha is None
            com = self.com
        elif self.alpha is not None:
            assert self.com is None
            com = 1.0 / self.alpha - 1.0

        # Store computed com value
        self._com = com

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute EWMVar using Flax state management.

        Args:
            x: Input data array

        Returns:
            Exponentially weighted variance
        """
        # Trainable parameter for com (log-space for numerical stability)
        logcom = self.param(
            "logcom",
            initializers.constant(jnp.log(self._com)),
            (),  # scalar shape
        )
        com = jnp.exp(logcom)
        alpha = 1.0 / (1.0 + com)

        # State variables using Flax's Variable collections
        mean = self.variable("state", "mean", lambda: jnp.full(x.shape, jnp.nan, x.dtype))

        variance = self.variable("state", "variance", lambda: jnp.full(x.shape, jnp.nan, x.dtype))

        nobs = self.variable("state", "nobs", lambda: jnp.full(x.shape, 0.0, x.dtype))

        # Get current state values
        current_mean = mean.value
        current_variance = variance.value
        current_nobs = nobs.value

        # Initialize on first non-nan value
        updated_mean = jnp.where(jnp.isnan(current_mean), x, current_mean)
        updated_variance = jnp.where(jnp.isnan(current_variance), 0.0, current_variance)

        mask = jnp.logical_not(jnp.isnan(x))
        updated_nobs = jnp.where(mask, current_nobs + 1, current_nobs)

        # Alpha adjustment scheme
        if self.adjust == "linear":
            tscale = 1.0 / alpha
            tscale = jnp.where(updated_nobs < tscale, updated_nobs, tscale)
            alpha = jnp.where(tscale > 0, 1.0 / tscale, jnp.nan)
        elif self.adjust:
            # Exponential scheme (as in pandas)
            alpha = alpha / (1.0 - (1.0 - alpha) ** updated_nobs)

        # Incremental variance update using Finch's formula
        diff = x - updated_mean
        incr = alpha * diff

        # Update state
        final_mean = jnp.where(mask, updated_mean + incr, updated_mean)
        final_variance = jnp.where(
            mask, (1 - alpha) * (updated_variance + diff * incr), updated_variance
        )

        # Update state variables
        mean.value = final_mean
        variance.value = final_variance
        nobs.value = updated_nobs

        return final_variance


def create_ewmvar(
    com: float | None = None, alpha: float | None = None, adjust: bool | str = True
) -> EWMVar:
    """Factory function to create EWMVar module with given parameters.

    Args:
        com: Center of mass parameter
        alpha: Smoothing factor
        adjust: Adjustment scheme

    Returns:
        EWMVar module instance
    """
    return EWMVar(com=com, alpha=alpha, adjust=adjust)
