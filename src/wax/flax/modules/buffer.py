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
"""Flax-based buffering mechanism."""

from typing import Any, NamedTuple

import flax.linen as nn
import jax.numpy as jnp


class BufferState(NamedTuple):
    """State structure for buffer operations."""

    buffer: Any  # Ordered buffer view (oldest-first)
    # Traced scalars, not Python ints: these are carried through jit and scan, so
    # they arrive as rank-0 arrays rather than as concrete values.
    len_buffer: jnp.ndarray
    write_idx: jnp.ndarray


class Buffer(nn.Module):
    """Flax-based Buffer module for streaming data buffering.

    This module implements a circular buffer that maintains a fixed-size
    window of recent observations, crucial for streaming time-series operations.

    Internally, a write-pointer indexes into a fixed array so each step
    performs a single O(1) write instead of shifting the entire buffer.
    The returned output is always in logical order (oldest first, newest last).
    """

    maxlen: int
    fill_value: float = jnp.nan
    return_state: bool = False

    @nn.compact
    def __call__(self, input: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray, BufferState]:
        """Record input data in the buffer.

        Args:
            input: Data to record in the buffer

        Returns:
            buffer: Current buffer contents in logical order (oldest first),
                    or tuple with state if return_state=True
        """
        # Internal circular buffer (physical order, not logical)
        buffer: nn.Variable[jnp.ndarray] = self.variable(
            "state",
            "buffer",
            lambda: jnp.full((self.maxlen,) + input.shape, self.fill_value, dtype=input.dtype),
        )

        len_buffer: nn.Variable[jnp.ndarray] = self.variable(
            "state", "len_buffer", lambda: jnp.array(0, dtype=jnp.int32)
        )

        write_idx: nn.Variable[jnp.ndarray] = self.variable(
            "state", "write_idx", lambda: jnp.array(0, dtype=jnp.int32)
        )

        current_buffer = buffer.value
        current_len = len_buffer.value
        current_write_idx = write_idx.value

        # O(1) write at the current position
        write_pos = current_write_idx % self.maxlen
        updated_buffer = current_buffer.at[write_pos].set(input)

        updated_len = jnp.minimum(current_len + 1, self.maxlen)
        new_write_idx = current_write_idx + 1

        # Persist internal state
        buffer.value = updated_buffer
        len_buffer.value = updated_len
        write_idx.value = new_write_idx

        # Return buffer in logical order (oldest first, newest last).
        # After writing at write_pos, the oldest element is at (write_pos+1) % maxlen.
        start = new_write_idx % self.maxlen
        indices = (jnp.arange(self.maxlen) + start) % self.maxlen
        ordered = updated_buffer[indices]

        if self.return_state:
            state = BufferState(ordered, updated_len, new_write_idx)
            return ordered, state
        else:
            return ordered


def create_buffer(maxlen: int, fill_value: float = jnp.nan, return_state: bool = False) -> Buffer:
    """Factory function to create Buffer module with given parameters.

    Args:
        maxlen: Maximum length of the buffer
        fill_value: Value to use for unfilled buffer positions
        return_state: Whether to return state information

    Returns:
        Buffer module instance
    """
    return Buffer(maxlen=maxlen, fill_value=fill_value, return_state=return_state)
