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
"""Flax-based UpdateParams module for parameter partitioning and management."""

from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
from flax.traverse_util import flatten_dict, unflatten_dict


class UpdateParams(nn.Module):
    """Flax-based module for separating trainable from non-trainable parameters.

    This module provides functionality similar to Haiku's parameter partitioning,
    allowing fine-grained control over which parameters are updated during training.
    """

    fun: Callable  # Function to wrap
    predicate: Callable | None = None  # Predicate to determine trainable params

    def setup(self):
        """Setup the UpdateParams module."""
        # Set default predicate if none provided
        if self.predicate is None:
            from wax.predicate import pass_all_predicate

            self.predicate = pass_all_predicate

    @nn.compact
    def __call__(self, *args, **kwargs) -> Any:
        """Apply function with parameter partitioning.

        Args:
            *args: Arguments to pass to the wrapped function
            **kwargs: Keyword arguments to pass to the wrapped function

        Returns:
            Result of the wrapped function
        """
        # Initialize parameters if this is a function that needs them
        # For now, we'll assume the function is already parameterized
        # or will handle its own parameter initialization

        # Store non-trainable parameters separately. The call is made for its side
        # effect: it declares the "non_trainable" collection on the module.
        self.variable("non_trainable", "params", lambda: {})

        # Apply the function
        # In a full implementation, this would involve:
        # 1. Partitioning parameters based on the predicate
        # 2. Applying the function with only trainable parameters
        # 3. Merging results back with non-trainable parameters

        # For this simplified implementation, just call the function
        result = self.fun(*args, **kwargs)

        return result

    def partition_params(self, params: Any) -> tuple[Any, Any]:
        """Partition parameters into trainable and non-trainable.

        Args:
            params: Parameters to partition

        Returns:
            Tuple of (trainable_params, non_trainable_params)
        """
        if not isinstance(params, dict):
            # If params is not a dict, treat as single parameter
            if self.predicate("", "", params):
                return params, {}
            else:
                return {}, params

        # Flatten parameters for easier processing
        flat_params = flatten_dict(params, sep="/")

        trainable = {}
        non_trainable = {}

        for key, value in flat_params.items():
            # Parse the key to extract module and parameter names
            path_parts = key.split("/")
            module_name = path_parts[0] if len(path_parts) > 0 else ""
            param_name = path_parts[-1] if len(path_parts) > 0 else ""

            # Apply predicate to determine if parameter is trainable
            if self.predicate(module_name, param_name, value):
                trainable[key] = value
            else:
                non_trainable[key] = value

        # Unflatten the dictionaries
        trainable_params = unflatten_dict(trainable, sep="/")
        non_trainable_params = unflatten_dict(non_trainable, sep="/")

        return trainable_params, non_trainable_params

    def merge_params(self, trainable_params: Any, non_trainable_params: Any) -> Any:
        """Merge trainable and non-trainable parameters.

        Args:
            trainable_params: Trainable parameters
            non_trainable_params: Non-trainable parameters

        Returns:
            Merged parameters
        """
        if not trainable_params and not non_trainable_params:
            return {}

        if not trainable_params:
            return non_trainable_params

        if not non_trainable_params:
            return trainable_params

        # For dictionaries, merge recursively
        if isinstance(trainable_params, dict) and isinstance(non_trainable_params, dict):
            result = {}
            result.update(non_trainable_params)
            result.update(trainable_params)
            return result

        # For other types, prefer trainable parameters
        return trainable_params


def create_update_params(fun: Callable, predicate: Callable | None = None) -> UpdateParams:
    """Factory function to create UpdateParams module.

    Args:
        fun: Function to wrap with parameter partitioning
        predicate: Predicate to determine which parameters are trainable

    Returns:
        UpdateParams module instance
    """
    return UpdateParams(fun=fun, predicate=predicate)


def get_init_params(fun: Callable, *args, **kwargs) -> Any:
    """Initialize parameters for a function.

    This is a utility function that mimics the Haiku version for compatibility.

    Args:
        fun: Function to initialize parameters for
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Initialized parameters
    """
    # For Flax, parameter initialization is typically handled
    # through the module's init method. This is a placeholder
    # that would need to be adapted based on the specific function.

    # If the function is a Flax module, we could do:
    if hasattr(fun, "init"):
        rng = jax.random.PRNGKey(0)
        return fun.init(rng, *args, **kwargs)

    # Otherwise, return empty parameters
    return {}
