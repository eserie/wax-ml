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
"""Flax-based FillNanInf module."""

from typing import Any, TypeVar, cast

import flax.linen as nn
import jax.numpy as jnp
from jax.tree_util import tree_map

T = TypeVar("T")


class FillNanInf(nn.Module):
    """Fill NaN, positive infinity, and negative infinity values with specified value."""

    fill_value: Any = 0.0

    # tree_map rebuilds the pytree it was given, so the result has the same type as
    # the input. Saying so lets callers that pass an array keep an array.
    def __call__(self, input: T) -> T:
        """Fill NaN and infinity values in input.

        Args:
            input: Input data (can be nested structure)

        Returns:
            Input with NaN/inf values replaced by fill_value
        """

        def fill_nan(x):
            """Fill NaN and infinity values in a single array."""
            return jnp.nan_to_num(
                x, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value
            )

        return cast(T, tree_map(fill_nan, input))


def create_fill_nan_inf(fill_value: Any = 0.0) -> FillNanInf:
    """Factory function to create FillNanInf module.

    Args:
        fill_value: Value to replace NaN/inf with

    Returns:
        FillNanInf module instance
    """
    return FillNanInf(fill_value=fill_value)
