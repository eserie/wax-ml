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
"""Flax-based EWMCov module for exponentially weighted moving covariance."""

import warnings

import flax.linen as nn
import jax.numpy as jnp

from .ewma import EWMA


class EWMCov(nn.Module):
    """Flax-based exponentially weighted moving covariance module.

    Computes the exponentially weighted covariance between two variables using:
    Cov(X,Y) = E[XY] - E[X]E[Y] (general case)
    Cov(X,Y) = E[XY] (assume_centered case)
    """

    com: float | None = None
    alpha: float | None = None
    min_periods: int = 1
    adjust: bool | str = True
    ignore_na: bool = False
    initial_value: float = jnp.nan
    assume_centered: bool = False

    def setup(self):
        """Setup the EWMCov module."""
        # Validate parameters
        if self.com is None and self.alpha is None:
            raise ValueError("Must specify either com or alpha")
        if self.com is not None and self.alpha is not None:
            raise ValueError("Cannot specify both com and alpha")

        # Convert between com and alpha
        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = 1.0 / (1.0 + self.com)

        # Create EWMA instances for covariance calculation
        ewma_params = {
            "alpha": alpha,
            "min_periods": self.min_periods,
            "adjust": self.adjust,
            "ignore_na": self.ignore_na,
            "initial_value": self.initial_value,
        }

        # EWMA for E[XY] (outer product)
        self.mean_xy = EWMA(**ewma_params)

        if not self.assume_centered:
            # EWMA for E[X] and E[Y] when not assuming centered data
            self.mean_x = EWMA(**ewma_params)
            self.mean_y = EWMA(**ewma_params)

    def __call__(self, *args) -> jnp.ndarray:
        """Compute exponentially weighted covariance.

        Args:
            *args: Either (x, y) as separate arguments, or ((x, y),) as tuple

        Returns:
            Exponentially weighted covariance matrix
        """
        # Handle different calling patterns for backward compatibility
        if len(args) == 1 and isinstance(args[0], tuple | list):
            # Legacy tuple input: EWMCov()((x, y))
            warnings.warn(
                "Passing (x, y) as a tuple is deprecated. Use EWMCov()(x, y) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            x, y = args[0]
        elif len(args) == 2:
            # Modern separate arguments: EWMCov()(x, y)
            x, y = args
        else:
            raise ValueError("Expected either (x, y) or ((x, y),)")

        # Ensure inputs are arrays (handle scalar inputs)
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        # Compute outer product for XY term
        xy = jnp.outer(x, y)

        # Compute exponentially weighted mean of XY
        mean_xy = self.mean_xy(xy)

        if self.assume_centered:
            # If assuming centered data, covariance is just E[XY]
            return jnp.array(mean_xy)
        else:
            # Compute individual means
            mean_x = self.mean_x(x)
            mean_y = self.mean_y(y)

            # Compute covariance: E[XY] - E[X]E[Y]
            mean_x_mean_y = jnp.outer(mean_x, mean_y)
            covariance = mean_xy - mean_x_mean_y

            return jnp.array(covariance)


def create_ewmcov(
    com: float | None = None,
    alpha: float | None = None,
    min_periods: int = 1,
    adjust: bool | str = True,
    ignore_na: bool = False,
    initial_value: float = jnp.nan,
    assume_centered: bool = False,
) -> EWMCov:
    """Factory function to create EWMCov module.

    Args:
        com: Center of mass decay parameter where α = 1 / (1 + com)
        alpha: Direct smoothing factor (0 < α ≤ 1)
        min_periods: Minimum observations required before returning non-NaN values
        adjust: Controls adjustment factor handling
        ignore_na: How to handle missing values in weight calculations
        initial_value: Initial state value
        assume_centered: Whether to assume zero mean

    Returns:
        EWMCov module instance
    """
    return EWMCov(
        com=com,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na,
        initial_value=initial_value,
        assume_centered=assume_centered,
    )
