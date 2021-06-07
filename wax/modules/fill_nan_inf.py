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
"""Fill nan, posinf and neginf values."""
from typing import Any

import haiku as hk
import jax.numpy as jnp
from jax import tree_map


class FillNanInf(hk.Module):
    """Fill nan, posinf and neginf values."""

    def __init__(self, fill_value: Any = 0.0, name: str = None):
        """Initialize module.

        Args:
            fill_value : value used to replace nan, posinf or neginf encountered values.
            name : name of the module
        """
        super().__init__(name=name)
        self.fill_value = fill_value

    def __call__(self, input):
        def fill_nan(x):
            # mask values
            return jnp.nan_to_num(
                x, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value
            )

        return tree_map(fill_nan, input)
