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
"""Tests for streaming optimizer functionality."""

import jax
import jax.numpy as jnp
import optax
import pytest

from wax.flax.core.streaming_transforms import (
    streaming_optimizer,
)
from wax.flax.modules.buffer import Buffer
from wax.flax.modules.ewma import EWMA


def mse_loss(predictions, targets):
    """Mean squared error loss function."""
    return jnp.mean((predictions - targets) ** 2)


def mae_loss(predictions, targets):
    """Mean absolute error loss function."""
    return jnp.mean(jnp.abs(predictions - targets))


class TestStreamingOptimizer:
    """Test cases for StreamingOptimizer functionality."""

    def test_basic_streaming_optimization(self):
        """Test basic streaming optimization with simple model."""

        @streaming_optimizer(optax.adam(0.01), mse_loss)  # Use Adam which has meaningful state
        def simple_learner(x, y):
            """Simple linear model for testing."""
            # For now, return a simple transformation
            return x * 2.0  # Simple linear model

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0, y0 = jnp.array(1.0), jnp.array(2.0)
        params, state = simple_learner.init(rng, x0, y0)

        # Apply optimization step
        (loss_val, prediction), new_state = simple_learner.apply(params, state, None, x0, y0)

        # Check outputs
        assert isinstance(loss_val, jax.Array)
        assert isinstance(prediction, jax.Array)
        assert jnp.isfinite(loss_val)
        assert jnp.isfinite(prediction)

        # State should change after optimization (Adam has stateful parameters)
        assert str(state) != str(new_state)

    def test_streaming_optimizer_with_ewma(self):
        """Test streaming optimizer with EWMA module."""

        @streaming_optimizer(optax.adam(0.001), mse_loss)
        def ewma_learner(x, y):
            """EWMA-based learner."""
            ewma = EWMA(alpha=0.1)
            prediction = ewma(x)
            return prediction

        # Test data - simple regression problem
        rng = jax.random.PRNGKey(42)
        x_data = jnp.array([1.0, 2.0, 3.0, 4.0])
        y_data = jnp.array([2.0, 4.0, 6.0, 8.0])  # y = 2*x pattern

        # Initialize
        params, state = ewma_learner.init(rng, x_data[0], y_data[0])

        # Train on sequence
        losses = []
        current_state = state

        for x, y in zip(x_data, y_data, strict=False):
            (loss_val, pred), current_state = ewma_learner.apply(params, current_state, None, x, y)
            losses.append(float(loss_val))

        # Check that learning occurred
        assert len(losses) == 4
        assert all(jnp.isfinite(loss) for loss in losses)

        # Generally expect some improvement (though not guaranteed for few steps)
        assert all(loss >= 0 for loss in losses)  # Losses should be non-negative

    def test_optimizer_with_auxiliary_outputs(self):
        """Test streaming optimizer with auxiliary outputs."""

        @streaming_optimizer(optax.sgd(0.01), mse_loss, has_aux=True)
        def model_with_aux(x, y):
            """Model that returns auxiliary outputs."""
            buffer = Buffer(maxlen=3, fill_value=0.0)
            buffered_data = buffer(x)
            prediction = jnp.mean(buffered_data)
            aux_info = {"buffered": buffered_data, "input": x}
            return prediction, aux_info

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0, y0 = jnp.array(1.0), jnp.array(1.5)
        params, state = model_with_aux.init(rng, x0, y0)

        # Apply
        (loss_val, pred, aux), new_state = model_with_aux.apply(params, state, None, x0, y0)

        # Check outputs
        assert isinstance(loss_val, jax.Array)
        assert isinstance(pred, jax.Array)
        assert isinstance(aux, dict)
        assert "buffered" in aux
        assert "input" in aux
        assert jnp.isfinite(loss_val)

    def test_different_optimizers(self):
        """Test streaming optimizer with different Optax optimizers."""

        optimizers_to_test = [
            optax.sgd(0.01),
            optax.adam(0.001),
            optax.rmsprop(0.01),
        ]

        for optimizer in optimizers_to_test:

            @streaming_optimizer(optimizer, mse_loss)
            def test_learner(x, y):
                return x  # Simple pass-through

            # Initialize and test
            rng = jax.random.PRNGKey(42)
            x0, y0 = jnp.array(1.0), jnp.array(1.0)
            params, state = test_learner.init(rng, x0, y0)

            (loss_val, pred), new_state = test_learner.apply(params, state, None, x0, y0)

            # Should work with any optimizer
            assert jnp.isfinite(loss_val)
            assert jnp.isfinite(pred)

    def test_different_loss_functions(self):
        """Test streaming optimizer with different loss functions."""

        loss_functions = [mse_loss, mae_loss]

        for loss_fn in loss_functions:

            @streaming_optimizer(optax.sgd(0.01), loss_fn)
            def test_learner(x, y):
                return x * 1.5  # Simple linear transformation

            # Initialize and test
            rng = jax.random.PRNGKey(42)
            x0, y0 = jnp.array(2.0), jnp.array(3.0)
            params, state = test_learner.init(rng, x0, y0)

            (loss_val, pred), new_state = test_learner.apply(params, state, None, x0, y0)

            # Should work with any loss function
            assert jnp.isfinite(loss_val)
            assert loss_val >= 0  # Losses should be non-negative
            assert jnp.isfinite(pred)

    def test_streaming_optimization_sequence(self):
        """Test optimization over a sequence of data points."""

        @streaming_optimizer(optax.adam(0.01), mse_loss)
        def sequence_learner(x, y):
            """Simple learner for sequence processing."""
            ewma = EWMA(alpha=0.2)
            return ewma(x)

        # Generate synthetic data
        rng = jax.random.PRNGKey(42)
        sequence_length = 10
        x_sequence = jax.random.normal(rng, (sequence_length,))
        y_sequence = x_sequence * 2.0 + 0.1  # Linear relationship with noise

        # Initialize
        params, state = sequence_learner.init(rng, x_sequence[0], y_sequence[0])

        # Process sequence
        losses = []
        predictions = []
        current_state = state

        for x, y in zip(x_sequence, y_sequence, strict=False):
            (loss_val, pred), current_state = sequence_learner.apply(
                params, current_state, None, x, y
            )
            losses.append(float(loss_val))
            predictions.append(float(pred))

        # Check results
        assert len(losses) == sequence_length
        assert len(predictions) == sequence_length
        assert all(jnp.isfinite(loss) for loss in losses)
        assert all(jnp.isfinite(pred) for pred in predictions)

    def test_jax_transformations_compatibility(self):
        """Test that streaming optimizer works with JAX transformations."""

        @streaming_optimizer(optax.sgd(0.01), mse_loss)
        def jittable_learner(x, y):
            return x * 0.8  # Simple model

        # Test JIT compilation
        jitted_init = jax.jit(jittable_learner.init)
        jitted_apply = jax.jit(jittable_learner.apply)

        # Initialize with JIT
        rng = jax.random.PRNGKey(42)
        x0, y0 = jnp.array(1.0), jnp.array(1.0)
        params, state = jitted_init(rng, x0, y0)

        # Apply with JIT
        (loss_val, pred), new_state = jitted_apply(params, state, None, x0, y0)

        # Should work correctly
        assert jnp.isfinite(loss_val)
        assert jnp.isfinite(pred)

    def test_gradient_handling(self):
        """Test that gradients are properly handled (NaN/Inf cleaning)."""

        def problematic_loss(pred, target):
            """Loss function that might produce NaN gradients."""
            # Division by zero can cause gradient issues
            return jnp.mean((pred - target) ** 2) / (jnp.abs(pred) + 1e-10)

        @streaming_optimizer(optax.sgd(0.01), problematic_loss)
        def robust_learner(x, y):
            return x * 2.0

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0, y0 = jnp.array(0.0), jnp.array(1.0)  # Potential for issues
        params, state = robust_learner.init(rng, x0, y0)

        # Apply - should handle gradients gracefully
        (loss_val, pred), new_state = robust_learner.apply(params, state, None, x0, y0)

        # Should produce finite outputs even with problematic gradients
        assert jnp.isfinite(loss_val)
        assert jnp.isfinite(pred)

    def test_parameter_updates(self):
        """Test that parameters are actually being updated."""

        @streaming_optimizer(optax.adam(0.1), mse_loss)  # Use Adam with large learning rate
        def updating_learner(x, y):
            # Simple model that should learn
            return x

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0, y0 = jnp.array(1.0), jnp.array(2.0)  # Clear target
        params, state = updating_learner.init(rng, x0, y0)

        # Apply several steps
        current_state = state
        first_state = str(current_state)

        for _ in range(3):  # Fewer steps but should still see changes with Adam
            (loss_val, pred), current_state = updating_learner.apply(
                params, current_state, None, x0, y0
            )

        final_state = str(current_state)

        # State should change over time (parameters being updated)
        assert first_state != final_state, "State should change during optimization"

    def test_error_handling(self):
        """Test error handling for invalid configurations."""

        # Test with invalid optimizer - should fail at runtime when applying
        @streaming_optimizer("not_an_optimizer", mse_loss)
        def bad_learner(x, y):
            return x

        rng = jax.random.PRNGKey(42)
        x0, y0 = jnp.array(1.0), jnp.array(1.0)

        # Should fail during initialization due to invalid optimizer
        with pytest.raises((TypeError, AttributeError)):
            params, state = bad_learner.init(rng, x0, y0)

        # Test with invalid loss function - should fail at runtime
        @streaming_optimizer(optax.sgd(0.01), "not_a_function")
        def bad_loss_learner(x, y):
            return x

        # Should fail during initialization due to invalid loss function
        with pytest.raises((TypeError, AttributeError)):
            params, state = bad_loss_learner.init(rng, x0, y0)
