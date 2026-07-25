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
"""Flax-based Lag (delay) operator module."""

import flax.linen as nn
import jax.numpy as jnp
from jax.tree_util import tree_map

from .buffer import Buffer, BufferState


class Lag(nn.Module):
    """Flax-based Lag (delay) operator using composition instead of inheritance.

    This module provides a delay operator that returns the value from
    `lag` steps ago, using a Buffer module as internal storage.
    """

    lag: int
    fill_value: float = jnp.nan
    return_state: bool = False

    def setup(self):
        """Setup the Lag module by creating internal buffer."""
        # Create internal buffer with size lag+1 to store current + lag previous values
        self.buffer = Buffer(
            maxlen=self.lag + 1, fill_value=self.fill_value, return_state=self.return_state
        )

    def __call__(self, input: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray, BufferState]:
        """Apply lag operation to input.

        Args:
            input: Current input value

        Returns:
            Value from `lag` steps ago (or fill_value if not enough history)
        """
        # Use internal Buffer to maintain the buffer
        if self.return_state:
            buffer, buffer_state = self.buffer(input)
            # Return the oldest value (first element) and state
            return buffer[0], buffer_state
        else:
            buffer = self.buffer(input)
            # Return the oldest value (first element)
            return buffer[0]


def create_lag(lag: int, fill_value: float = jnp.nan, return_state: bool = False) -> Lag:
    """Factory function to create Lag module with given parameters.

    Args:
        lag: Number of steps to delay
        fill_value: Value to return when insufficient history
        return_state: Whether to return state information

    Returns:
        Lag module instance
    """
    return Lag(lag=lag, fill_value=fill_value, return_state=return_state)


def tree_lag(shift: int = 1):
    """Create a function that applies Lag module to a PyTree.

    Args:
        shift: Number of steps to lag

    Returns:
        Function that applies lag to each element of a PyTree
    """

    def apply_fn(*pytree):
        return tree_map(lambda x: Lag(lag=shift)(x) if x is not None else None, pytree)

    return apply_fn
