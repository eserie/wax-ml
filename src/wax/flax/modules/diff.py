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
"""Flax-based Diff module for computing differences on sequential data."""

from typing import cast

import flax.linen as nn
import jax.numpy as jnp

from .buffer import Buffer


class Diff(nn.Module):
    """Flax-based module for computing differences on sequential data.

    This module computes the difference between current value and the value
    from `periods` steps ago using a Buffer for storage.
    """

    periods: int = 1

    def setup(self) -> None:
        """Setup the Diff module."""
        assert self.periods == 1, "periods > 1 not implemented."

        # Create internal buffer for storing values
        self.buffer = Buffer(maxlen=self.periods + 1, fill_value=jnp.nan)

    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        """Compute difference.

        Args:
            input: Input data

        Returns:
            Difference between current and lagged values
        """
        # Get buffer with current and previous values. The buffer is built with
        # return_state left at its default (False), so it hands back the array
        # alone rather than the (array, state) pair.
        buffer = cast(jnp.ndarray, self.buffer(input))

        # Compute difference: current (last) - lagged (first)
        diff = buffer[-1] - buffer[0]

        return diff


def create_diff(periods: int = 1) -> Diff:
    """Factory function to create Diff module with given parameters.

    Args:
        periods: Number of periods to lag for difference calculation

    Returns:
        Diff module instance
    """
    return Diff(periods=periods)
