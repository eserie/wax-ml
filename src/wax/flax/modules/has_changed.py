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
"""Flax-based HasChanged module."""

import flax.linen as nn
import jax.numpy as jnp

# Import from original wax for compatibility
from wax.stream import DTYPE_INIT_VALUES


class HasChanged(nn.Module):
    """Detect if input has changed from previous call."""

    @nn.compact
    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        """Check if input has changed from previous value.

        Args:
            input: Current input to check for changes

        Returns:
            Boolean indicating if input has changed
        """
        # Initialize previous value with appropriate default
        prev_value = self.variable(
            "state",
            "prev_value",
            lambda: jnp.full_like(input, DTYPE_INIT_VALUES[input.dtype.type], input.dtype),
        )

        # Check if any element has changed
        current_prev = prev_value.value
        has_changed = jnp.not_equal(input, current_prev).any()

        # Update previous value with current input
        prev_value.value = input

        return has_changed


def create_has_changed() -> HasChanged:
    """Factory function to create HasChanged module.

    Returns:
        HasChanged module instance
    """
    return HasChanged()
