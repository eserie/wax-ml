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
"""Tests for Flax OptaxOptimizer module."""

import jax
import jax.numpy as jnp
import optax

from wax.flax.core import flax_transform_with_state
from wax.flax.modules.optax_optimizer import create_optax_optimizer


class TestOptaxOptimizer:
    """Test cases for OptaxOptimizer module."""

    def test_basic_optimization_step(self):
        """Test basic optimization step with SGD."""
        # Create optimizer
        opt = create_optax_optimizer(optax.sgd(0.1))
        tf = flax_transform_with_state(opt)

        # Initialize parameters
        key = jax.random.PRNGKey(42)
        params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.5)}
        grads = {"w": jnp.array([0.1, 0.2]), "b": jnp.array(0.05)}

        # Initialize optimizer
        opt_params, opt_state = tf.init(key, params, grads)

        # Apply optimizer step
        (updated_params, _), new_state = tf.apply(opt_params, opt_state, None, params, grads)

        # Check that parameters were updated correctly (SGD: p = p - lr * grad)
        expected_w = jnp.array([1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2])
        expected_b = 0.5 - 0.1 * 0.05

        assert jnp.allclose(updated_params["w"], expected_w)
        assert jnp.allclose(updated_params["b"], expected_b)

    def test_adam_optimizer(self):
        """Test with Adam optimizer."""
        # Create Adam optimizer
        opt = create_optax_optimizer(optax.adam(0.01))
        tf = flax_transform_with_state(opt)

        # Initialize parameters
        key = jax.random.PRNGKey(42)
        params = {"w": jnp.array([1.0])}
        grads = {"w": jnp.array([0.1])}

        # Initialize optimizer
        opt_params, opt_state = tf.init(key, params, grads)

        # First step
        (params1, _), state1 = tf.apply(opt_params, opt_state, None, params, grads)

        # Second step
        (params2, _), state2 = tf.apply(opt_params, state1, None, params1, grads)

        # Parameters should change between steps
        assert not jnp.allclose(params["w"], params1["w"])
        assert not jnp.allclose(params1["w"], params2["w"])

    def test_zero_gradients(self):
        """Test that zero gradients don't change parameters."""
        # Create optimizer
        opt = create_optax_optimizer(optax.sgd(0.1))
        tf = flax_transform_with_state(opt)

        # Initialize parameters
        key = jax.random.PRNGKey(42)
        params = {"w": jnp.array([1.0, 2.0])}
        zero_grads = {"w": jnp.zeros_like(params["w"])}

        # Initialize optimizer
        opt_params, opt_state = tf.init(key, params, zero_grads)

        # Apply zero gradients
        (updated_params, _), new_state = tf.apply(opt_params, opt_state, None, params, zero_grads)

        # Parameters should remain unchanged
        assert jnp.allclose(updated_params["w"], params["w"])

    def test_optimizer_state_persistence(self):
        """Test that optimizer state is maintained between calls."""
        # Create optimizer that maintains state (Adam)
        opt = create_optax_optimizer(optax.adam(0.01))
        tf = flax_transform_with_state(opt)

        # Initialize parameters
        key = jax.random.PRNGKey(42)
        params = {"w": jnp.array([1.0])}
        grads = {"w": jnp.array([0.1])}

        # Initialize optimizer
        opt_params, opt_state = tf.init(key, params, grads)

        # Multiple steps should show different behavior due to state
        (params1, _), state1 = tf.apply(opt_params, opt_state, None, params, grads)
        (params2, _), state2 = tf.apply(opt_params, state1, None, params1, grads)
        (params3, _), state3 = tf.apply(opt_params, state2, None, params2, grads)

        # All parameter values should be different due to Adam's momentum
        assert not jnp.allclose(params["w"], params1["w"])
        assert not jnp.allclose(params1["w"], params2["w"])
        assert not jnp.allclose(params2["w"], params3["w"])
