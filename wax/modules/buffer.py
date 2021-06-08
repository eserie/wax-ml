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
"""Implement buffering mechanism."""

from typing import Any, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map, tree_unflatten


class BufferState(NamedTuple):
    buffer: Any
    len_buffer: int
    i_start: int


class Buffer(hk.Module):
    """Implement buffering mechanism."""

    def __init__(
        self,
        maxlen: int,
        fill_value=jnp.nan,
        return_state: bool = False,
        name: str = None,
    ):
        """Initialize the module/

        Args:
            maxlen : length of the buffer
            fill_value : value to use to fill buffer while no data has been append.
            return_state : if true, the module returns a tuple (buffer, state) where state is
                           the full buffer state (buffer, len_buffer, i_start). If false, the buffer is
                           returned.
            name : name of the module.
        """
        super().__init__(name=name)
        self.maxlen = maxlen
        self.fill_value = fill_value
        self.return_state = return_state

    def __call__(self, input: jnp.ndarray):
        """Record input data in the buffer.

        Args:
            input: data to record.
        """

        def _initial_state(shape, dtype):
            nonlocal input
            if type(self.fill_value) == type(input):
                input_flat, treedef = tree_flatten(input)
                fill_value_flat, _ = tree_flatten(self.fill_value)
                buffer_flat = list(
                    map(
                        lambda x, f: jnp.full(
                            (self.maxlen,) + x.shape, f, dtype=x.dtype
                        ),
                        input_flat,
                        fill_value_flat,
                    )
                )
                buffer = tree_unflatten(treedef, buffer_flat)
            else:
                buffer = tree_map(
                    lambda x: jnp.full(
                        (self.maxlen,) + x.shape, self.fill_value, dtype=x.dtype
                    ),
                    input,
                )
            len_buffer = 0
            i_start = self.maxlen
            return BufferState(buffer, len_buffer, i_start)

        buffer_state = hk.get_state(
            "buffer_state",
            [],
            init=_initial_state,
        )

        buffer, len_buffer, i_start = buffer_state

        input, treedef = tree_flatten(input)
        buffer, _ = tree_flatten(buffer)

        for i, (x_i, buffer_i) in enumerate(zip(input, buffer)):
            buffer_i = jnp.roll(buffer_i, -1, axis=0)
            buffer_i = jax.ops.index_update(buffer_i, -1, x_i)
            buffer[i] = buffer_i

        buffer = tree_unflatten(treedef, buffer)

        len_buffer = jnp.minimum(len_buffer + 1, self.maxlen)
        i_start = self.maxlen - len_buffer

        next_state = BufferState(buffer, len_buffer, i_start)

        hk.set_state("buffer_state", next_state)
        if self.return_state:
            return buffer, next_state
        else:
            return buffer
