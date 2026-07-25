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
"""Flax transform utilities for WAX-ML."""

from collections.abc import Callable
from typing import Any, NamedTuple

import flax.linen as nn
import jax.numpy as jnp
from flax.core import FrozenDict


class FlaxTransformed(NamedTuple):
    """Flax equivalent of Haiku's TransformedWithState.

    This provides a unified interface for stateful Flax modules
    that mirrors Haiku's transform_with_state functionality.
    """

    init: Callable
    apply: Callable


def flax_transform_with_state(
    module_fn: Callable[..., nn.Module] | nn.Module,
) -> FlaxTransformed:
    """Transform a Flax module into init/apply functions.

    This function provides a Haiku-like interface for Flax modules,
    enabling seamless integration with WAX-ML's sequential processing.

    Args:
        module_fn: A function that returns a Flax module, or a Flax module instance.

    Returns:
        FlaxTransformed: A pair of (init, apply) functions.
    """
    if isinstance(module_fn, nn.Module):
        module = module_fn
    else:
        # Assume it's a function that creates a module
        module = module_fn()

    def init_fn(rng: jnp.ndarray, *args, **kwargs) -> tuple[FrozenDict, FrozenDict]:
        """Initialize parameters and state.

        Returns:
            params: Trainable parameters
            state: Non-trainable state variables
        """
        variables = module.init(rng, *args, **kwargs)
        params = variables.get("params", FrozenDict())
        state = {k: v for k, v in variables.items() if k != "params"}
        return params, FrozenDict(state)

    def apply_fn(
        params: FrozenDict, state: FrozenDict, rng: jnp.ndarray | None, *args, **kwargs
    ) -> tuple[Any, FrozenDict]:
        """Apply the module with given parameters and state.

        Args:
            params: Trainable parameters
            state: Non-trainable state variables
            rng: Random number generator key
            *args, **kwargs: Inputs to the module

        Returns:
            output: Module output
            new_state: Updated state variables
        """
        variables = {"params": params, **state}

        if rng is not None:
            output, new_variables = module.apply(
                variables, *args, **kwargs, rngs={"default": rng}, mutable=True
            )
        else:
            output, new_variables = module.apply(variables, *args, **kwargs, mutable=True)

        # Extract new state (everything except params)
        new_state = {k: v for k, v in new_variables.items() if k != "params"}
        return output, FrozenDict(new_state)

    return FlaxTransformed(init_fn, apply_fn)


def combine_variables(params: FrozenDict, state: FrozenDict) -> FrozenDict:
    """Combine parameters and state into a single variables dict."""
    return FrozenDict({"params": params, **state})


def split_variables(variables: FrozenDict) -> tuple[FrozenDict, FrozenDict]:
    """Split variables into parameters and state."""
    params = variables.get("params", FrozenDict())
    state = {k: v for k, v in variables.items() if k != "params"}
    return params, FrozenDict(state)
