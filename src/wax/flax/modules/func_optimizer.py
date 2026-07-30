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
"""Flax-based FuncOptimizer module for function optimization."""

from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
from jax.tree_util import tree_map

from .fill_nan_inf import FillNanInf
from .optax_optimizer import OptaxOptimizer
from .update_params import UpdateParams


class FuncOptimizer(nn.Module):
    """Flax-based function optimizer using iterative solvers.

    This module optimizes any function using gradient-based optimization,
    with support for parameter partitioning and gradient clipping.
    """

    fun: Callable  # Function to optimize
    opt: Any  # Optimizer (OptaxOptimizer or optax.GradientTransformation)
    has_aux: bool = False  # Whether function returns auxiliary data
    params_predicate: Callable | None = None  # Predicate for trainable parameters

    def setup(self):
        """Setup the FuncOptimizer module."""
        # Create UpdateParams module for parameter partitioning
        self.update_params = UpdateParams(fun=self.fun, predicate=self.params_predicate)

        # Wrap optimizer if needed
        if not isinstance(self.opt, OptaxOptimizer):
            if hasattr(self.opt, "init") and hasattr(self.opt, "update"):
                # It's an optax.GradientTransformation
                self.optimizer = OptaxOptimizer(opt=self.opt)
            else:
                raise ValueError("opt must be OptaxOptimizer or optax.GradientTransformation")
        else:
            self.optimizer = self.opt

        # Module for handling NaN/Inf gradients
        self.fill_nan_inf = FillNanInf()

    @nn.compact
    def __call__(self, *args, **kwargs) -> Any:
        """Perform one optimization step.

        Args:
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Function output and optimization info
        """
        # Initialize trainable parameters
        trainable_params: nn.Variable[dict[str, Any]] = self.variable(
            "trainable_params", "params", lambda: {}
        )

        # Initialize function if needed
        if not trainable_params.value:
            # Try to initialize parameters from the function
            if hasattr(self.fun, "init"):
                # Function is a Flax module
                rng = jax.random.PRNGKey(0)
                init_params = self.fun.init(rng, *args, **kwargs)

                # Partition parameters
                train_params, non_train_params = self.update_params.partition_params(init_params)
                trainable_params.value = train_params

                # Store non-trainable parameters separately. The call is made for
                # its side effect: it registers the collection on the module, which
                # is later read back through ``self.variables``.
                self.variable("non_trainable_params", "params", lambda: non_train_params)
            else:
                # Function doesn't have parameters, create dummy parameters
                trainable_params.value = {}

        # Get current parameters
        current_trainable = trainable_params.value

        # Define loss function for gradient computation
        def loss_fn(params):
            # Merge trainable and non-trainable parameters if needed
            if hasattr(self, "variable") and "non_trainable_params" in self.variables:
                non_trainable = self.variables["non_trainable_params"]["params"]
                merged_params = self.update_params.merge_params(params, non_trainable)
            else:
                merged_params = params

            # Apply function
            if hasattr(self.fun, "apply"):
                # Function is a Flax module
                result = self.fun.apply(merged_params, *args, **kwargs)
            else:
                # Function is a regular callable
                result = self.fun(*args, **kwargs)

            # Handle auxiliary data
            if self.has_aux:
                if isinstance(result, tuple):
                    loss = result[0]
                    aux = result[1:] if len(result) > 2 else result[1]
                else:
                    loss = result
                    aux = None
            else:
                loss = result
                aux = None

            return loss, aux

        # Compute gradients
        if current_trainable:
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_trainable)

            # Handle NaN/Inf gradients (fill with zeros)
            grads = tree_map(lambda x: self.fill_nan_inf(x), grads)

            # Apply optimizer update
            updated_params, opt_info = self.optimizer(current_trainable, grads)

            # Update parameters
            trainable_params.value = updated_params
        else:
            # No trainable parameters, just evaluate function
            loss, aux = loss_fn({})
            grads = {}
            opt_info = None

        # Return function output and optimization info
        if self.has_aux:
            return (loss, aux), grads, opt_info
        else:
            return loss, grads, opt_info


def create_func_optimizer(
    fun: Callable,
    opt: Any,
    has_aux: bool = False,
    params_predicate: Callable | None = None,
) -> FuncOptimizer:
    """Factory function to create FuncOptimizer module.

    Args:
        fun: Function to optimize
        opt: Optimizer (OptaxOptimizer or GradientTransformation)
        has_aux: Whether function returns auxiliary data
        params_predicate: Predicate for determining trainable parameters

    Returns:
        FuncOptimizer module instance
    """
    return FuncOptimizer(
        fun=fun,
        opt=opt,
        has_aux=has_aux,
        params_predicate=params_predicate,
    )
