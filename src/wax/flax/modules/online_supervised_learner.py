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
"""Flax-based OnlineSupervisedLearner module for online supervised learning."""

from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from .func_optimizer import FuncOptimizer


class OnlineSupervisedLearner(nn.Module):
    """Flax-based online supervised learning module.

    This module wraps a model with a loss function and optimizer for
    online supervised learning, providing both predictions and training updates.
    """

    model: Callable  # Model function/module
    loss_fn: Callable  # Loss function
    opt: Any  # Optimizer
    params_predicate: Callable | None = None  # Predicate for trainable parameters

    def setup(self):
        """Setup the OnlineSupervisedLearner module."""

        # Create a combined function that applies model and computes loss
        def model_loss_fn(x, y):
            """Combined model and loss function."""
            # Apply model to get predictions
            predictions = self.model(x)

            # Compute loss
            loss = self.loss_fn(predictions, y)

            # Return loss and predictions as auxiliary data
            return loss, predictions

        # Create function optimizer
        self.func_optimizer = FuncOptimizer(
            fun=model_loss_fn,
            opt=self.opt,
            has_aux=True,  # Returns predictions as auxiliary data
            params_predicate=self.params_predicate,
        )

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, Any]:
        """Perform one step of online supervised learning.

        Args:
            x: Input features
            y: Target values

        Returns:
            Tuple of (predictions, training_info) where training_info contains
            loss, gradients, and optimizer information
        """
        # Apply function optimizer to get training step results
        (loss, predictions), grads, opt_info = self.func_optimizer(x, y)

        # Package training information
        training_info = {
            "loss": loss,
            "grads": grads,
            "opt_info": opt_info,
        }

        return predictions, training_info


class SimpleOnlineLearner(nn.Module):
    """Simplified online learner that just returns predictions and loss."""

    model: Callable  # Model function/module
    loss_fn: Callable  # Loss function
    opt: Any  # Optimizer

    def setup(self):
        """Setup the SimpleOnlineLearner module."""
        # Create online supervised learner
        self.learner = OnlineSupervisedLearner(
            model=self.model,
            loss_fn=self.loss_fn,
            opt=self.opt,
        )

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Perform learning step and return predictions and loss.

        Args:
            x: Input features
            y: Target values

        Returns:
            Tuple of (predictions, loss)
        """
        predictions, training_info = self.learner(x, y)
        loss = training_info["loss"]

        return predictions, loss


def create_online_supervised_learner(
    model: Callable,
    loss_fn: Callable,
    opt: Any,
    params_predicate: Callable | None = None,
) -> OnlineSupervisedLearner:
    """Factory function to create OnlineSupervisedLearner module.

    Args:
        model: Model function/module
        loss_fn: Loss function
        opt: Optimizer
        params_predicate: Predicate for determining trainable parameters

    Returns:
        OnlineSupervisedLearner module instance
    """
    return OnlineSupervisedLearner(
        model=model,
        loss_fn=loss_fn,
        opt=opt,
        params_predicate=params_predicate,
    )


def create_simple_online_learner(
    model: Callable,
    loss_fn: Callable,
    opt: Any,
) -> SimpleOnlineLearner:
    """Factory function to create SimpleOnlineLearner module.

    Args:
        model: Model function/module
        loss_fn: Loss function
        opt: Optimizer

    Returns:
        SimpleOnlineLearner module instance
    """
    return SimpleOnlineLearner(
        model=model,
        loss_fn=loss_fn,
        opt=opt,
    )
