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
"""Forward fill the values."""
import haiku as hk
from jax import numpy as jnp


class Ffill(hk.Module):
    """Ffill current element."""

    def __init__(self, periods=1, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        valid_value = hk.get_state(
            "valid_value",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, jnp.nan, dtype),
        )

        valid_value = jnp.where(jnp.isnan(x), valid_value, x)

        hk.set_state("valid_value", valid_value)
        return valid_value
