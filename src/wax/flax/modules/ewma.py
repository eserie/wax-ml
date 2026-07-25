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
"""Flax-based Exponential Moving Average module."""

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers


class EWMA(nn.Module):
    """Flax-based Exponential Moving Average module.

    This is a Flax implementation that mirrors the functionality of the
    Haiku-based EWMA module, demonstrating the architectural differences
    between implicit state management (Haiku) and explicit state
    management (Flax).
    """

    com: float | None = None
    alpha: float | None = None
    min_periods: int = 0
    adjust: bool | str = True
    ignore_na: bool = False
    initial_value: float = jnp.nan
    return_info: bool = False

    def setup(self):
        """Setup the EWMA module parameters."""
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
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute EWMA using Flax state management.

        Args:
            x: Input data array

        Returns:
            result: EWMA result
            info: Optional dictionary with additional variables (if return_info=True)
        """
        info = {}

        # Trainable parameter for com (log-space for numerical stability)
        logcom = self.param(
            "logcom",
            initializers.constant(jnp.log(self._com)),
            (),  # scalar shape
        )
        com = jnp.exp(logcom)

        # State variables using Flax's Variable collections
        mean = self.variable(
            "state", "mean", lambda: jnp.full(x.shape, self.initial_value, x.dtype)
        )

        old_wt = self.variable("state", "old_wt", lambda: jnp.full(x.shape, 1.0, x.dtype))

        nobs = self.variable("state", "nobs", lambda: jnp.full(x.shape, 0, dtype=jnp.int32))

        # Get current state values
        current_mean = mean.value
        current_old_wt = old_wt.value
        current_nobs = nobs.value

        # EWMA computation logic (identical to Haiku version)
        is_observation = ~jnp.isnan(x)
        isnan_mean = jnp.isnan(current_mean)

        # Fill NaN with zero to avoid NaNs in gradient computations
        x_filled = jnp.nan_to_num(x)
        mean_filled = jnp.nan_to_num(current_mean)

        alpha = 1.0 / (1.0 + com)

        if self.adjust:
            new_wt = jnp.array(1.0)
        else:
            new_wt = alpha

        if self.adjust == "linear":
            # Linear adjustment for effective com
            old_wt_factor = jnp.where(
                is_observation, 1.0, jnp.maximum(0.0, (current_old_wt - 1.0) / current_old_wt)
            )
            updated_old_wt = jnp.minimum(current_old_wt, com)
        else:
            old_wt_factor = 1.0 - alpha
            updated_old_wt = current_old_wt

        if self.ignore_na:
            updated_old_wt = jnp.where(
                is_observation, updated_old_wt * old_wt_factor, updated_old_wt
            )
        else:
            updated_old_wt = updated_old_wt * old_wt_factor

        updated_old_wt = jnp.where(isnan_mean, 0.0, updated_old_wt)

        # Update mean
        updated_mean = jnp.where(
            is_observation,
            (updated_old_wt * mean_filled + new_wt * x_filled) / (updated_old_wt + new_wt),
            current_mean,
        )

        if self.return_info:
            info["com_eff"] = updated_old_wt / new_wt

        # Update weights for next iteration
        if self.adjust:
            final_old_wt = jnp.where(is_observation, updated_old_wt + new_wt, updated_old_wt)
        else:
            final_old_wt = jnp.where(is_observation, 1.0, updated_old_wt)

        # Restore NaN values
        final_mean = jnp.where(jnp.logical_and(~is_observation, isnan_mean), jnp.nan, updated_mean)

        # Update observation count
        updated_nobs = jnp.where(is_observation, current_nobs + 1, current_nobs)

        # Update state variables
        mean.value = final_mean
        old_wt.value = final_old_wt
        nobs.value = updated_nobs

        if self.return_info:
            info["nobs"] = updated_nobs

        # Apply minimum periods constraint
        if self.min_periods:
            result = jnp.where(updated_nobs >= self.min_periods, final_mean, jnp.nan)
        else:
            result = final_mean

        if self.return_info:
            return result, info
        else:
            return result


def create_ewma(**kwargs) -> EWMA:
    """Factory function to create EWMA module with given parameters.

    This function provides a convenient way to create EWMA modules
    that can be used with the transform functions.

    Args:
        **kwargs: Arguments to pass to EWMA constructor

    Returns:
        EWMA module instance
    """
    return EWMA(**kwargs)
