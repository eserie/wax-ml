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
"""Flax-based Forward Fill module."""

import flax.linen as nn
import jax.numpy as jnp


class Ffill(nn.Module):
    """Flax-based Forward Fill module.

    This module forward fills missing (NaN) values with the last valid value.
    """

    periods: int = 1  # For compatibility, though not used in current implementation

    @nn.compact
    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        """Forward fill missing values.

        Args:
            input: Input data that may contain NaN values

        Returns:
            Data with NaN values replaced by last valid values
        """
        # State variable to store the last valid value
        valid_value = self.variable(
            "state", "valid_value", lambda: jnp.full(input.shape, jnp.nan, input.dtype)
        )

        # Get current valid value
        current_valid = valid_value.value

        # Update with new valid values, keep old for NaN inputs
        updated_valid = jnp.where(jnp.isnan(input), current_valid, input)

        # Update state
        valid_value.value = updated_valid

        return updated_valid


def create_ffill(periods: int = 1) -> Ffill:
    """Factory function to create Ffill module.

    Args:
        periods: Compatibility parameter (not used)

    Returns:
        Ffill module instance
    """
    return Ffill(periods=periods)
