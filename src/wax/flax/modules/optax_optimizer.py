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
"""Flax-based OptaxOptimizer module for integrating Optax optimizers."""

from typing import Any

import flax.linen as nn
import optax


class OptaxOptimizer(nn.Module):
    """Flax-based wrapper for Optax optimizers.

    This module provides a way to use Optax optimizers within the Flax/WAX-ML
    framework, maintaining optimizer state across training steps.
    """

    opt: optax.GradientTransformation

    @nn.compact
    def __call__(self, params: Any, grads: Any) -> tuple[Any, Any]:
        """Apply optimizer update to parameters.

        Args:
            params: Current parameters
            grads: Gradients for parameter update

        Returns:
            Tuple of (updated_params, optimizer_info)
        """
        # Initialize optimizer state if needed
        opt_state = self.variable("state", "opt_state", lambda: self.opt.init(params))

        # Compute updates and new optimizer state
        updates, new_opt_state = self.opt.update(grads, opt_state.value, params)

        # Apply updates to parameters
        new_params = optax.apply_updates(params, updates)

        # Update state
        opt_state.value = new_opt_state

        # Return updated parameters and optimizer info
        return new_params, new_opt_state


def create_optax_optimizer(opt: optax.GradientTransformation) -> OptaxOptimizer:
    """Factory function to create OptaxOptimizer module.

    Args:
        opt: Optax GradientTransformation instance

    Returns:
        OptaxOptimizer module instance
    """
    return OptaxOptimizer(opt=opt)
