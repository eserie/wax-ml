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
"""Detect if something has changed."""
import haiku as hk
import jax.numpy as jnp

from wax.stream import DTYPE_INIT_VALUES


class HasChanged(hk.Module):
    """Detect if something has changed."""

    def __call__(self, input):
        prev_input = hk.get_state(
            "prev_value",
            [],
            init=lambda *_: jnp.full_like(
                input, DTYPE_INIT_VALUES[input.dtype.type], input.dtype
            ),
        )
        has_changed = jnp.not_equal(input, prev_input).any()
        hk.set_state("prev_value", input)
        return has_changed
