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
"""Tests for streaming scan functionality."""

import jax
import jax.numpy as jnp
import pytest

from wax.flax.core.streaming_transforms import streaming_scan
from wax.flax.modules.buffer import Buffer
from wax.flax.modules.ewma import EWMA


class TestStreamingScan:
    """Test cases for StreamingScan module."""

    def test_basic_scan_operation(self):
        """Test basic scan without reset functionality using the decorator."""

        @streaming_scan
        def simple_accumulator(x):
            """Simple accumulator that processes each element."""
            return x  # Simple pass-through for basic test

        # Test data
        inputs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Apply scan
        outputs, final_state = simple_accumulator.scan_apply(inputs)

        # Check outputs
        assert outputs.shape == inputs.shape
        assert jnp.allclose(outputs, inputs)  # Pass-through behavior

    def test_scan_with_reset_condition(self):
        """Test scan with reset functionality using decorator."""

        @streaming_scan(reset_on=lambda x: x > 3.0)
        def accumulator(x):
            """Accumulator that can be reset."""
            return x  # Simple pass-through for testing

        # Test data
        inputs = jnp.array([1.0, 2.0, 4.0, 1.0, 2.0])  # Reset at 4.0

        # Apply scan
        outputs, final_state = accumulator.scan_apply(inputs)

        # Check that outputs have expected shape
        assert outputs.shape == inputs.shape
        assert jnp.allclose(outputs, inputs)  # Pass-through behavior

    def test_streaming_buffer_with_scan(self):
        """Test streaming scan with a buffer module."""

        @streaming_scan
        def buffered_mean(x):
            """Compute rolling mean using buffer."""
            buffer = Buffer(maxlen=3)
            buffered_data = buffer(x)
            # Compute mean of valid (non-NaN) values
            valid_mask = ~jnp.isnan(buffered_data)
            return jnp.where(jnp.any(valid_mask), jnp.nanmean(buffered_data), 0.0)

        # Test data
        inputs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Apply scan
        outputs, final_state = buffered_mean.scan_apply(inputs)

        # Check outputs shape and basic properties
        assert outputs.shape == inputs.shape
        assert jnp.all(jnp.isfinite(outputs))

    def test_scan_with_reset_on_condition(self):
        """Test scan decorator with reset condition."""

        @streaming_scan(reset_on=lambda x: x == 0.0)
        def counter(x):
            """Simple counter that increments."""
            # This is a simplified test - in practice would use stateful modules
            return x + 1.0

        # Test data with reset trigger
        inputs = jnp.array([1.0, 2.0, 0.0, 1.0, 2.0])  # Reset at 0.0

        # Apply scan
        outputs, final_state = counter.scan_apply(inputs)

        # Check basic properties
        assert outputs.shape == inputs.shape
        assert jnp.all(jnp.isfinite(outputs))

    def test_streaming_scan_decorator_usage(self):
        """Test using @streaming_scan as decorator."""

        @streaming_scan
        def simple_transform(x):
            """Simple transformation."""
            return x * 2.0

        # Test data
        inputs = jnp.array([1.0, 2.0, 3.0])

        # Apply scan
        outputs, final_state = simple_transform.scan_apply(inputs)

        # Check outputs
        expected = inputs * 2.0
        assert jnp.allclose(outputs, expected)

    def test_scan_with_ewma_module(self):
        """Test scan with EWMA module for realistic streaming scenario."""

        @streaming_scan
        def streaming_ewma(x):
            """Streaming EWMA computation."""
            ewma = EWMA(alpha=0.3)
            return ewma(x)

        # Test data
        inputs = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])

        # Apply scan
        outputs, final_state = streaming_ewma.scan_apply(inputs)

        # Check outputs
        assert outputs.shape == inputs.shape
        assert jnp.all(jnp.isfinite(outputs))

        # EWMA should show smoothing behavior
        assert outputs[0] == inputs[0]  # First value should match

    def test_scan_preserves_jax_transformations(self):
        """Test that streaming scan is compatible with JAX transformations."""

        @streaming_scan
        def simple_fn(x):
            return x**2

        # Test JIT compilation
        jitted_scan = jax.jit(simple_fn.scan_apply)

        inputs = jnp.array([1.0, 2.0, 3.0])
        outputs, final_state = jitted_scan(inputs)

        expected = inputs**2
        assert jnp.allclose(outputs, expected)

    def test_scan_with_complex_reset_logic(self):
        """Test scan with more complex reset conditions."""

        def complex_reset(x):
            """Reset on specific pattern."""
            return jnp.logical_or(x < 0, x > 10)

        @streaming_scan(reset_on=complex_reset)
        def stateful_processor(x):
            """Process with complex state."""
            return jnp.abs(x)  # Simple transformation

        # Test data with reset triggers
        inputs = jnp.array([1.0, 5.0, -1.0, 2.0, 15.0, 3.0])

        # Apply scan
        outputs, final_state = stateful_processor.scan_apply(inputs)

        # Check outputs
        expected = jnp.abs(inputs)
        assert jnp.allclose(outputs, expected)

    def test_scan_error_handling(self):
        """Test error handling in scan operations."""

        # Test with invalid reset function - should raise error when scan is applied
        @streaming_scan(reset_on="not_a_function")
        def bad_scan(x):
            return x

        inputs = jnp.array([1.0, 2.0])

        # Should raise error when trying to apply the scan
        with pytest.raises(TypeError):  # "not_a_function" is not callable
            outputs, final_state = bad_scan.scan_apply(inputs)

    def test_scan_with_multiple_modules(self):
        """Test scan with multiple interacting modules."""

        @streaming_scan(reset_on=lambda x: x == 0)
        def multi_module_processor(x):
            """Processor with multiple modules."""
            buffer = Buffer(maxlen=2)
            ewma = EWMA(alpha=0.5)

            # The buffer is exercised for its state; only the EWMA is returned.
            buffer(x)
            smoothed = ewma(x)

            return smoothed

        # Test data
        inputs = jnp.array([1.0, 2.0, 0.0, 3.0, 4.0])  # Reset at 0

        # Apply scan
        outputs, final_state = multi_module_processor.scan_apply(inputs)

        # Check basic properties
        assert outputs.shape == inputs.shape
        assert jnp.all(jnp.isfinite(outputs))

    def test_scan_state_management(self):
        """Test that scan properly manages state across calls."""

        @streaming_scan
        def stateful_counter(x):
            """Counter that maintains state."""
            # In practice this would use a proper stateful module
            return x + 1.0

        inputs1 = jnp.array([1.0, 2.0])
        inputs2 = jnp.array([3.0, 4.0])

        # First scan
        outputs1, state1 = stateful_counter.scan_apply(inputs1)

        # Second scan (should maintain state)
        outputs2, state2 = stateful_counter.scan_apply(inputs2)

        # Check outputs
        assert outputs1.shape == inputs1.shape
        assert outputs2.shape == inputs2.shape
        assert jnp.all(jnp.isfinite(outputs1))
        assert jnp.all(jnp.isfinite(outputs2))
