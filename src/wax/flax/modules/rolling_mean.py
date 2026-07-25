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
"""Flax-based RollingMean module for rolling window mean calculations."""

import flax.linen as nn
import jax.numpy as jnp

from .buffer import Buffer


class RollingMean(nn.Module):
    """Flax-based module for computing rolling mean over a specified window.

    This module computes the mean over a rolling window of the last `horizon`
    observations, handling NaN values gracefully and supporting a minimum
    number of required observations.
    """

    horizon: int
    min_periods: int = 1

    def setup(self):
        """Setup the RollingMean module."""
        # Create internal buffer to store the rolling window
        self.buffer = Buffer(maxlen=self.horizon, return_state=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute rolling mean.

        Args:
            x: Input data value

        Returns:
            Rolling mean over the window, or NaN if insufficient data
        """
        # Get buffer with current window of values
        buffer, attrs = self.buffer(x)

        # Sum valid (non-NaN) values in the buffer
        # Replace NaN with 0.0 for proper summation
        sum_values = jnp.where(jnp.logical_not(jnp.isnan(buffer)), buffer, 0.0).sum(axis=0)

        # Count valid (non-NaN) values in the buffer
        count = jnp.where(jnp.logical_not(jnp.isnan(buffer)), 1, 0).sum(axis=0)

        # Compute mean only if we have enough valid observations
        mean = jnp.where(count >= self.min_periods, sum_values / count, jnp.nan)

        return mean


def create_rolling_mean(horizon: int, min_periods: int = 1) -> RollingMean:
    """Factory function to create RollingMean module.

    Args:
        horizon: Window size for rolling calculation
        min_periods: Minimum valid observations required (default: 1)

    Returns:
        RollingMean module instance
    """
    return RollingMean(horizon=horizon, min_periods=min_periods)
