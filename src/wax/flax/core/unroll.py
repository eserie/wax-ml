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
"""Flax-based unroll functionality for sequential processing."""

import logging
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.tree_util import tree_map

from .transform import FlaxTransformed, flax_transform_with_state

logger = logging.getLogger(__name__)


class FlaxUnrollTransformed(NamedTuple):
    """Flax equivalent of Haiku's UnrollTransformedWithState."""

    init: Callable
    apply: Callable


class FlaxScanState(NamedTuple):
    """State for Flax-based scanning operations."""

    fun_state: FrozenDict
    rng: jnp.ndarray | None


def flax_unroll_transform(
    module_or_fn: nn.Module | Callable[..., nn.Module] | FlaxTransformed,
    skip_first: bool = False,
    dynamic: bool = True,
    pbar: bool = False,
) -> FlaxUnrollTransformed:
    """Flax equivalent of unroll_transform_with_state.

    Transforms a Flax module into unroll-compatible init/apply functions.

    Args:
        module_or_fn: Flax module, module factory function, or FlaxTransformed pair
        skip_first: If true, first value of the sequence is not used in apply
        dynamic: If true, unroll using jax.lax.scan
        pbar: If true, activate progress bar (only works when dynamic=False)

    Returns:
        FlaxUnrollTransformed: Pair of (init, apply) functions for unrolling
    """
    # Handle different input types
    if isinstance(module_or_fn, FlaxTransformed):
        tfunc = module_or_fn
    elif isinstance(module_or_fn, nn.Module) or callable(module_or_fn):
        tfunc = flax_transform_with_state(module_or_fn)
    else:
        raise TypeError(
            f"Expected Flax module, callable, or FlaxTransformed, got {type(module_or_fn)}"
        )

    def init_fn(rng: jnp.ndarray, *args, **kwargs) -> tuple[FrozenDict, FrozenDict]:
        """Initialize parameters and state using first element of sequences."""
        xs = (args, kwargs)
        args_0, kwargs_0 = tree_map(lambda x: x[0], xs)
        params, state = tfunc.init(rng, *args_0, **kwargs_0)
        return params, state

    def apply_fn(
        params: FrozenDict, state: FrozenDict, rng: jnp.ndarray | None, *args, **kwargs
    ) -> tuple[Any, FrozenDict]:
        """Apply function unrolled over sequences."""

        def scan_f(scan_state: FlaxScanState, inputs: tuple) -> tuple[FlaxScanState, Any]:
            """Scan function for sequential processing."""
            state, rng = scan_state.fun_state, scan_state.rng
            args_step, kwargs_step = inputs

            if rng is not None:
                rng, sub_rng = jax.random.split(rng)
            else:
                sub_rng = None

            outputs, new_state = tfunc.apply(params, state, sub_rng, *args_step, **kwargs_step)
            return FlaxScanState(new_state, rng), outputs

        xs = (args, kwargs)

        if skip_first:
            xs = tree_map(lambda x: x[1:], xs)

        # Choose scan implementation
        if dynamic:
            scan = jax.lax.scan
        else:
            # Import static_scan from original wax module
            from ...unroll import static_scan

            scan = partial(static_scan, pbar=pbar)

        scan_state, output_sequence = scan(scan_f, init=FlaxScanState(state, rng), xs=xs)

        final_state = scan_state.fun_state
        return output_sequence, final_state

    return FlaxUnrollTransformed(init_fn, apply_fn)


def flax_unroll(
    module_or_fn: nn.Module | Callable[..., nn.Module] | FlaxTransformed | FlaxUnrollTransformed,
    skip_first: bool = False,
    dynamic: bool = True,
    pbar: bool = False,
    return_final_state: bool = False,
    rng: jnp.ndarray | None = None,
    params: FrozenDict | None = None,
    state: FrozenDict | None = None,
) -> Callable:
    """Flax equivalent of the unroll function.

    Creates a function that applies a Flax module unrolled over sequences.

    Args:
        module_or_fn: Flax module, module factory, or transformed functions
        skip_first: If true, first value of sequence is not used in apply
        dynamic: If true, unroll using jax.lax.scan
        pbar: If true, activate progress bar (dynamic=False only)
        return_final_state: If true, return both outputs and final state
        rng: Random number generator key
        params: Pre-initialized parameters
        state: Pre-initialized state

    Returns:
        apply_fn: Function that applies the module to sequential data
    """
    if not isinstance(module_or_fn, FlaxUnrollTransformed):
        fun = flax_unroll_transform(module_or_fn, skip_first=skip_first, dynamic=dynamic, pbar=pbar)
    else:
        fun = module_or_fn

    def apply_fn(*args, **kwargs):
        """Apply the unrolled function to input sequences."""
        # Initialize if needed
        if params is None or state is None:
            init_params, init_state = fun.init(rng, *args, **kwargs)
            use_params = init_params if params is None else params
            use_state = init_state if state is None else state
        else:
            use_params, use_state = params, state

        # Apply unrolled function
        output, final_state = fun.apply(use_params, use_state, rng, *args, **kwargs)

        if return_final_state:
            return output, final_state
        else:
            return output

    return apply_fn


def create_flax_module_factory(
    module_class: type[nn.Module], **module_kwargs
) -> Callable[[], nn.Module]:
    """Create a factory function for Flax modules.

    This is useful for creating modules with specific parameters that can be
    passed to the transform functions.

    Args:
        module_class: The Flax module class
        **module_kwargs: Keyword arguments to pass to the module constructor

    Returns:
        A factory function that creates the module
    """

    def factory() -> nn.Module:
        return module_class(**module_kwargs)

    return factory
