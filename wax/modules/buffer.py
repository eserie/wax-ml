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

from typing import Any, Callable, NamedTuple, Optional

import haiku as hk
import jax.numpy as jnp


class BufferState(NamedTuple):
    buffer: Any
    len_buffer: int
    i_start: int


class BufferFun(NamedTuple):
    init: Callable
    apply: Callable


def buffer_fn(maxlen: int, fill_value=jnp.nan):
    def init(shape, dtype):
        buffer = jnp.full((maxlen,) + shape, fill_value, dtype=dtype)
        len_buffer = 0
        i_start = maxlen
        return BufferState(buffer, len_buffer, i_start)

    def apply(x, state):
        buffer, len_buffer, i_start = state

        buffer = jnp.roll(buffer, -1, axis=0)
        buffer = buffer.at[-1].set(x)
        len_buffer = jnp.minimum(len_buffer + 1, maxlen)
        i_start = maxlen - len_buffer

        return buffer, BufferState(buffer, len_buffer, i_start)

    return BufferFun(init, apply)


class Buffer(hk.Module):
    """Implement buffering mechanism."""

    def __init__(
        self,
        maxlen: int,
        fill_value=jnp.nan,
        return_state: bool = False,
        name: Optional[str] = None,
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

        fun = buffer_fn(self.maxlen, self.fill_value)

        buffer_state = hk.get_state(
            "buffer_state",
            input.shape,
            input.dtype,
            init=fun.init,
        )

        buffer, buffer_state = fun.apply(input, buffer_state)

        hk.set_state("buffer_state", buffer_state)
        if self.return_state:
            return buffer, buffer_state
        else:
            return buffer
