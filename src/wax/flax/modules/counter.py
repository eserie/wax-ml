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
"""Flax-based Counter module."""

import flax.linen as nn
import jax.numpy as jnp


class Counter(nn.Module):
    """Simple counter that increments on each call."""

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """Increment and return current count.

        Returns:
            Current count value
        """
        count = self.variable("state", "count", lambda: jnp.array(0, dtype=jnp.uint32))

        # Increment counter (same logic as Haiku)
        new_count = count.value + 1
        count.value = new_count

        return new_count


def create_counter() -> Counter:
    """Factory function to create Counter module.

    Returns:
        Counter module instance
    """
    return Counter()
