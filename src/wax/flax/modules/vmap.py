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
"""Flax-based VMap module for vectorized function application (DEPRECATED)."""

import warnings
from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map


class VMap(nn.Module):
    """Flax-based vectorized function mapping module.

    DEPRECATED: This module is deprecated. Use vmap_lift_with_state instead.

    This module provides vectorized application of functions over batch dimensions,
    with optional averaging of results.
    """

    fun: Callable
    take_mean: bool = True

    def __call__(self, *args, **kwargs) -> Any:
        """Apply function in vectorized manner.

        Args:
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of vectorized function application
        """
        # Issue deprecation warning
        warnings.warn(
            "VMap is deprecated. Use vmap_lift_with_state instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # For Flax implementation, we'll use JAX's vmap directly
        # Note: This is a simplified implementation compared to the Haiku version
        # which used vmap_lift_with_state for proper state handling

        try:
            # Apply vmap to the function
            vmapped_fun = jax.vmap(self.fun)
            result = vmapped_fun(*args, **kwargs)

            # Apply mean reduction if requested
            if self.take_mean:
                result = add_batch(result)

            return result

        except Exception as e:
            raise ValueError(f"VMap failed to apply function: {e}") from e


def add_batch(pytree: Any) -> Any:
    """Add batch dimension by taking mean across batch axis.

    Args:
        pytree: Input PyTree structure

    Returns:
        PyTree with mean applied across first axis
    """

    def mean_first_axis(x):
        if isinstance(x, jnp.ndarray) and x.ndim > 0:
            return jnp.mean(x, axis=0)
        return x

    return tree_map(mean_first_axis, pytree)


def create_vmap(fun: Callable, take_mean: bool = True) -> VMap:
    """Factory function to create VMap module.

    DEPRECATED: Use vmap_lift_with_state instead.

    Args:
        fun: Function to vectorize
        take_mean: Whether to average results across batch dimension

    Returns:
        VMap module instance
    """
    warnings.warn(
        "create_vmap is deprecated. Use vmap_lift_with_state instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return VMap(fun=fun, take_mean=take_mean)
