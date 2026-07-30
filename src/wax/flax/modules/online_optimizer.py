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
"""Flax-based OnlineOptimizer module for online learning."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from .optax_optimizer import OptaxOptimizer


@dataclass
class OptInfo:
    """Information returned by online optimizer."""

    loss: jnp.ndarray
    aux: Any
    grads: Any
    params: Any


class OnlineOptimizer(nn.Module):
    """Flax-based online optimizer for streaming optimization.

    This module provides a comprehensive framework for online learning,
    combining models, loss functions, and optimizers with support for
    parameter projection and loss regularization.
    """

    model: Callable
    opt: Any  # OptaxOptimizer or optax.GradientTransformation
    project_params: Callable | None = None
    regularize_loss: Callable | None = None
    params_predicate: Callable | None = None

    def setup(self):
        """Setup the OnlineOptimizer module."""
        # Set default predicate if none provided
        if self.params_predicate is None:
            from wax.predicate import pass_all_predicate

            self.params_predicate = pass_all_predicate

        # Wrap optimizer if it's not already an OptaxOptimizer
        if not isinstance(self.opt, OptaxOptimizer):
            if hasattr(self.opt, "init") and hasattr(self.opt, "update"):
                # It's an optax.GradientTransformation
                self.optimizer = OptaxOptimizer(opt=self.opt)
            else:
                raise ValueError("opt must be OptaxOptimizer or optax.GradientTransformation")
        else:
            self.optimizer = self.opt

    @nn.compact
    def __call__(self, *args, **kwargs) -> OptInfo:
        """Perform one step of online optimization.

        Args:
            *args: Arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model

        Returns:
            OptInfo containing loss, auxiliary data, gradients, and parameters
        """
        # Initialize model parameters and state if needed
        # Holds the wrapped model's own variable collections, keyed by collection
        # name ("params", "state", ...); empty until the model has been initialized.
        model_vars: nn.Variable[dict[str, Any]] = self.variable("model_vars", "vars", lambda: {})

        # Check if model is a TransformedWithState or regular function
        if hasattr(self.model, "init") and hasattr(self.model, "apply"):
            # It's a TransformedWithState
            if not model_vars.value:
                # Initialize model
                dummy_rng = jax.random.PRNGKey(0)
                init_vars = self.model.init(dummy_rng, *args, **kwargs)
                model_vars.value = init_vars

            def model_fn(params, state, *args, **kwargs):
                vars_dict = {"params": params, **state}
                result = self.model.apply(vars_dict, *args, **kwargs)
                return result

        else:
            # It's a regular function - need to transform it
            import haiku as hk

            @hk.transform_with_state
            def wrapped_model(*args, **kwargs):
                return self.model(*args, **kwargs)

            if not model_vars.value:
                dummy_rng = jax.random.PRNGKey(0)
                params, state = wrapped_model.init(dummy_rng, *args, **kwargs)
                model_vars.value = {"params": params, "state": state}

            def model_fn(params, state, *args, **kwargs):
                return wrapped_model.apply(params, state, jax.random.PRNGKey(0), *args, **kwargs)

        # Extract current parameters and state
        current_vars = model_vars.value
        if "params" in current_vars:
            params = current_vars["params"]
            state = {k: v for k, v in current_vars.items() if k != "params"}
        else:
            # Assume everything is parameters for now
            params = current_vars
            state = {}

        # Partition parameters into trainable and non-trainable
        def partition_params(params):
            trainable = {}
            non_trainable = {}

            def partition_fn(path, value):
                # Convert path to string representation
                path_str = "/".join(str(p) for p in path)
                module_name = path[0] if path else ""
                param_name = path[-1] if len(path) > 1 else path[0] if path else ""

                if self.params_predicate(module_name, param_name, value):
                    return "trainable"
                else:
                    return "non_trainable"

            from flax.traverse_util import flatten_dict, unflatten_dict

            flat_params = flatten_dict(params, sep="/")

            for key, value in flat_params.items():
                path = key.split("/")
                if self.params_predicate(path[0] if path else "", path[-1] if path else "", value):
                    trainable[key] = value
                else:
                    non_trainable[key] = value

            return unflatten_dict(trainable, sep="/"), unflatten_dict(non_trainable, sep="/")

        trainable_params, non_trainable_params = partition_params(params)

        # Define loss function for gradient computation
        def loss_fn(trainable_params):
            # Merge trainable and non-trainable params
            merged_params = {}
            merged_params.update(non_trainable_params)
            merged_params.update(trainable_params)

            # Compute model output
            result = model_fn(merged_params, state, *args, **kwargs)

            # Handle different return types
            if isinstance(result, tuple):
                if len(result) == 2:
                    loss, aux = result
                else:
                    loss = result[0]
                    aux = result[1:] if len(result) > 2 else result[1]
            else:
                loss = result
                aux = None

            # Apply loss regularization if provided
            if self.regularize_loss is not None:
                loss = self.regularize_loss(loss, merged_params, aux)

            return loss, aux

        # Compute gradients
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(trainable_params)

        # Handle NaN gradients
        grads = tree_map(lambda x: jnp.nan_to_num(x), grads)

        # Apply optimizer update
        updated_trainable_params, opt_info = self.optimizer(trainable_params, grads)

        # Apply parameter projection if provided
        if self.project_params is not None:
            updated_trainable_params = self.project_params(updated_trainable_params)

        # Merge back updated parameters
        updated_params = {}
        updated_params.update(non_trainable_params)
        updated_params.update(updated_trainable_params)

        # Update model variables
        if "params" in current_vars:
            model_vars.value = {"params": updated_params, **state}
        else:
            model_vars.value = updated_params

        return OptInfo(loss=loss, aux=aux, grads=grads, params=updated_params)


def create_online_optimizer(
    model: Callable,
    opt: Any,
    project_params: Callable | None = None,
    regularize_loss: Callable | None = None,
    params_predicate: Callable | None = None,
) -> OnlineOptimizer:
    """Factory function to create OnlineOptimizer module.

    Args:
        model: Model function or TransformedWithState
        opt: Optimizer (OptaxOptimizer or GradientTransformation)
        project_params: Optional parameter projection function
        regularize_loss: Optional loss regularization function
        params_predicate: Function to determine trainable parameters

    Returns:
        OnlineOptimizer module instance
    """
    return OnlineOptimizer(
        model=model,
        opt=opt,
        project_params=project_params,
        regularize_loss=regularize_loss,
        params_predicate=params_predicate,
    )
