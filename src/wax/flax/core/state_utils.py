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
"""Utilities for handling Flax state in streaming modules."""

from collections.abc import Callable
from typing import Any

from flax import linen as nn
from flax.core import FrozenDict

from .transform import flax_transform_with_state


def apply_with_state(
    module: nn.Module, variables: FrozenDict, *args, **kwargs
) -> tuple[Any, FrozenDict]:
    """Apply a Flax module with proper state handling.

    This function automatically handles mutable state collections for streaming modules.

    Args:
        module: Flax module to apply
        variables: Module variables (params and state)
        *args: Arguments to pass to module
        **kwargs: Keyword arguments to pass to module

    Returns:
        Tuple of (output, new_variables)
    """
    # Determine which collections are mutable
    mutable_collections = []
    if "state" in variables:
        mutable_collections.append("state")
    if "cache" in variables:
        mutable_collections.append("cache")

    if mutable_collections:
        output, new_variables = module.apply(
            variables, *args, **kwargs, mutable=mutable_collections
        )
        return output, new_variables
    else:
        # No mutable collections, just apply normally
        output = module.apply(variables, *args, **kwargs)
        return output, variables


def create_streaming_module_wrapper(module_class: type[nn.Module]) -> Callable:
    """Create a streaming wrapper for a Flax module that handles state properly.

    Args:
        module_class: Flax module class to wrap

    Returns:
        Factory function that creates streaming transform
    """

    def create_module(*args, **kwargs):
        """Create and wrap a streaming module."""
        module = module_class(*args, **kwargs)
        return flax_transform_with_state(module)

    return create_module


class StreamingModuleWrapper(nn.Module):
    """Generic wrapper for streaming modules that handles state properly."""

    wrapped_module: nn.Module

    @nn.compact
    def __call__(self, *args, **kwargs):
        """Apply the wrapped module with proper state handling."""
        return self.wrapped_module(*args, **kwargs)


def make_streaming_compatible(module: nn.Module) -> flax_transform_with_state:
    """Make any Flax module streaming-compatible.

    Args:
        module: Flax module to make streaming compatible

    Returns:
        Streaming transform with proper state handling
    """
    wrapper = StreamingModuleWrapper(wrapped_module=module)
    return flax_transform_with_state(wrapper)
