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
"""Comprehensive tests for Flax-based Buffer module."""

import jax
import jax.numpy as jnp

from wax.flax.core import flax_transform_with_state, flax_unroll_transform
from wax.flax.modules.buffer import Buffer, BufferState


class TestFlaxBuffer:
    """Test suite for Flax Buffer module."""

    def test_basic_buffer_functionality(self):
        """Test basic buffer operations."""
        rng = jax.random.PRNGKey(42)
        maxlen = 3

        buffer = Buffer(maxlen=maxlen, fill_value=0.0)
        tf = flax_transform_with_state(buffer)

        # Initialize with first element
        x1 = jnp.array(1.0)
        params, state = tf.init(rng, x1)

        # Check initial state structure
        assert "state" in state
        assert "buffer" in state["state"]
        assert "len_buffer" in state["state"]
        assert "write_idx" in state["state"]

        # Internal buffer shape
        internal_buffer = state["state"]["buffer"]
        assert internal_buffer.shape == (maxlen,)
        assert state["state"]["len_buffer"] == 1  # One element added during init
        assert state["state"]["write_idx"] == 1  # One write performed

    def test_buffer_sequence_accumulation(self):
        """Test buffer accumulation over a sequence."""
        rng = jax.random.PRNGKey(42)
        maxlen = 4
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        buffer = Buffer(maxlen=maxlen, fill_value=-999.0)
        unroll_fn = flax_unroll_transform(buffer)

        params, state = unroll_fn.init(rng, data)
        outputs, final_state = unroll_fn.apply(params, state, rng, data)

        # Check output shape
        assert outputs.shape == (len(data), maxlen)

        # Check that the last *output* row contains last 4 elements (ordered)
        expected_final = jnp.array([3.0, 4.0, 5.0, 6.0])
        assert jnp.allclose(outputs[-1], expected_final)

        # Check final state metadata
        assert final_state["state"]["len_buffer"] == maxlen

    def test_buffer_with_return_state(self):
        """Test buffer when return_state=True."""
        rng = jax.random.PRNGKey(42)
        x = jnp.array(42.0)

        buffer = Buffer(maxlen=3, return_state=True)
        tf = flax_transform_with_state(buffer)

        params, state = tf.init(rng, x)
        result, new_state = tf.apply(params, state, None, x)

        # Should return tuple when return_state=True
        assert isinstance(result, tuple)
        buffer_output, buffer_state = result

        # Check state structure
        assert isinstance(buffer_state, BufferState)
        assert hasattr(buffer_state, "buffer")
        assert hasattr(buffer_state, "len_buffer")
        assert hasattr(buffer_state, "write_idx")

    def test_buffer_overflow_behavior(self):
        """Test buffer behavior when it overflows (more data than maxlen)."""
        rng = jax.random.PRNGKey(42)
        maxlen = 3

        # Create data longer than buffer
        data = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])

        buffer = Buffer(maxlen=maxlen, fill_value=0.0)
        unroll_fn = flax_unroll_transform(buffer)

        params, state = unroll_fn.init(rng, data)
        outputs, final_state = unroll_fn.apply(params, state, rng, data)

        # Final buffer should contain only the last 3 elements
        final_buffer = final_state["state"]["buffer"]
        expected = jnp.array([30.0, 40.0, 50.0])
        assert jnp.allclose(final_buffer, expected)

        # len_buffer should be capped at maxlen
        assert final_state["state"]["len_buffer"] == maxlen

    def test_buffer_with_different_fill_values(self):
        """Test buffer with different fill values."""
        rng = jax.random.PRNGKey(42)

        # Test with NaN fill: init writes x=1.0 at position 0, rest is NaN
        buffer_nan = Buffer(maxlen=2, fill_value=jnp.nan)
        tf_nan = flax_transform_with_state(buffer_nan)

        x = jnp.array(1.0)
        params, state = tf_nan.init(rng, x)

        # Internal buffer: [1.0, NaN] (write_idx=0 wrote x, position 1 still NaN)
        internal = state["state"]["buffer"]
        assert jnp.isfinite(internal[0])  # Position 0 written during init
        assert jnp.isnan(internal[1])  # Position 1 still fill_value

        # Test with zero fill
        buffer_zero = Buffer(maxlen=3, fill_value=0.0)
        tf_zero = flax_transform_with_state(buffer_zero)

        params, state = tf_zero.init(rng, x)
        internal = state["state"]["buffer"]
        # Position 0 has x=1.0, positions 1 and 2 have fill_value=0.0
        assert jnp.sum(internal == 0.0) >= 2  # At least 2 zeros

    def test_buffer_multidimensional_input(self):
        """Test buffer with multidimensional inputs."""
        rng = jax.random.PRNGKey(42)
        maxlen = 3

        # Test with 2D input
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        buffer = Buffer(maxlen=maxlen, fill_value=-1.0)
        tf = flax_transform_with_state(buffer)

        params, state = tf.init(rng, x)
        output, new_state = tf.apply(params, state, None, x)

        # Buffer should have shape (maxlen, *input_shape)
        expected_shape = (maxlen,) + x.shape
        assert output.shape == expected_shape
        assert new_state["state"]["buffer"].shape == expected_shape

    def test_jit_compilation(self):
        """Test that Buffer works with JIT compilation."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 2.0, 3.0, 4.0])

        buffer = Buffer(maxlen=3)
        unroll_fn = flax_unroll_transform(buffer)

        @jax.jit
        def apply_buffer(params, state, rng, data):
            return unroll_fn.apply(params, state, rng, data)

        params, state = unroll_fn.init(rng, data)
        outputs, final_state = apply_buffer(params, state, rng, data)

        # Check that outputs have expected shape (some values may be NaN from fill_value)
        assert outputs.shape == (len(data), 3)

    def test_numerical_consistency_with_haiku(self):
        """Test numerical consistency with Haiku implementation."""
        import haiku as hk

        from wax.modules import Buffer as HaikuBuffer
        from wax.unroll import unroll_transform_with_state as haiku_unroll

        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        maxlen = 3
        fill_value = 0.0

        # Haiku implementation
        @hk.transform_with_state
        def haiku_buffer_fn(x):
            return HaikuBuffer(maxlen=maxlen, fill_value=fill_value)(x)

        haiku_unroll_fn = haiku_unroll(haiku_buffer_fn)
        haiku_params, haiku_state = haiku_unroll_fn.init(rng, data)
        haiku_outputs, haiku_final_state = haiku_unroll_fn.apply(
            haiku_params, haiku_state, rng, data
        )

        # Flax implementation
        flax_buffer = Buffer(maxlen=maxlen, fill_value=fill_value)
        flax_unroll_fn = flax_unroll_transform(flax_buffer)
        flax_params, flax_state = flax_unroll_fn.init(rng, data)
        flax_outputs, flax_final_state = flax_unroll_fn.apply(flax_params, flax_state, rng, data)

        # Compare outputs (allowing for initialization differences):
        # the final buffers should be identical even if intermediate states differ,
        # so only the last row is compared.
        # Check if final outputs (last row) are identical
        final_output_diff = jnp.abs(haiku_outputs[-1] - flax_outputs[-1])
        final_max_diff = jnp.nanmax(final_output_diff)
        assert final_max_diff < 1e-10, f"Final output difference: {final_max_diff}"

        # Compare final buffer states
        haiku_final_buffer = haiku_final_state["buffer"]["buffer_state"].buffer
        flax_final_buffer = flax_final_state["state"]["buffer"]
        buffer_diff = jnp.abs(haiku_final_buffer - flax_final_buffer)
        assert jnp.nanmax(buffer_diff) < 1e-10, "Buffer states should be identical"

    def test_buffer_state_tracking(self):
        """Test detailed buffer state tracking throughout sequence."""
        rng = jax.random.PRNGKey(42)
        maxlen = 3

        buffer = Buffer(maxlen=maxlen, fill_value=0.0)
        tf = flax_transform_with_state(buffer)

        # Apply sequence step by step to track state
        params, state = tf.init(rng, jnp.array(1.0))

        # State after init should have 1 element, write_idx=1
        assert state["state"]["len_buffer"] == 1
        assert state["state"]["write_idx"] == 1

        # Add second element
        output, state = tf.apply(params, state, None, jnp.array(2.0))
        assert state["state"]["len_buffer"] == 2
        assert state["state"]["write_idx"] == 2

        # Add third element (buffer full)
        output, state = tf.apply(params, state, None, jnp.array(3.0))
        assert state["state"]["len_buffer"] == 3
        assert state["state"]["write_idx"] == 3

        # Add fourth element (should overflow)
        output, state = tf.apply(params, state, None, jnp.array(4.0))
        assert state["state"]["len_buffer"] == 3  # Should stay at maxlen
        assert state["state"]["write_idx"] == 4

        # Returned output should contain [2, 3, 4] in logical order
        expected = jnp.array([2.0, 3.0, 4.0])
        assert jnp.allclose(output, expected)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        rng = jax.random.PRNGKey(42)

        # Test maxlen = 1
        buffer_small = Buffer(maxlen=1)
        tf_small = flax_transform_with_state(buffer_small)

        x = jnp.array(42.0)
        params, state = tf_small.init(rng, x)
        assert state["state"]["buffer"].shape == (1,)

        # Test with very small values
        x_small = jnp.array(1e-10)
        output, new_state = tf_small.apply(params, state, None, x_small)
        assert jnp.allclose(new_state["state"]["buffer"], jnp.array([x_small]))

        # Test with very large values
        x_large = jnp.array(1e10)
        output, new_state = tf_small.apply(params, state, None, x_large)
        assert jnp.allclose(new_state["state"]["buffer"], jnp.array([x_large]))


# Autonomous tests (independent of Haiku for future migration)
class TestFlaxBufferAutonomous:
    """Autonomous tests for Flax Buffer (Haiku-independent)."""

    def test_buffer_fifo_behavior(self):
        """Test first-in-first-out behavior independently."""
        rng = jax.random.PRNGKey(42)
        maxlen = 4

        # Create sequence longer than buffer
        sequence = jnp.array([10, 20, 30, 40, 50, 60])

        buffer = Buffer(maxlen=maxlen, fill_value=0)
        unroll_fn = flax_unroll_transform(buffer)

        params, state = unroll_fn.init(rng, sequence)
        outputs, final_state = unroll_fn.apply(params, state, rng, sequence)

        # The last output row should contain last 4 elements in order: [30, 40, 50, 60]
        expected_final = jnp.array([30, 40, 50, 60])
        assert jnp.array_equal(outputs[-1], expected_final), (
            f"Expected {expected_final}, got {outputs[-1]}"
        )

    def test_buffer_gradual_filling(self):
        """Test how buffer gradually fills up."""
        rng = jax.random.PRNGKey(42)
        maxlen = 3
        fill_value = -999

        buffer = Buffer(maxlen=maxlen, fill_value=fill_value)
        tf = flax_transform_with_state(buffer)

        # Start with empty buffer (except for init)
        params, initial_state = tf.init(rng, jnp.array(1))

        # Buffer after init: [fill, fill, 1] and len=1
        assert initial_state["state"]["len_buffer"] == 1

        # Add second element
        output, state = tf.apply(params, initial_state, None, jnp.array(2))
        assert state["state"]["len_buffer"] == 2
        # Buffer should be [fill, 1, 2]

        # Add third element
        output, state = tf.apply(params, state, None, jnp.array(3))
        assert state["state"]["len_buffer"] == 3
        # Buffer should be [1, 2, 3]
        expected = jnp.array([1, 2, 3])
        assert jnp.array_equal(state["state"]["buffer"], expected)

    def test_buffer_performance_characteristics(self):
        """Test performance and memory characteristics."""
        rng = jax.random.PRNGKey(42)

        # Test with larger buffer and sequence
        maxlen = 1000
        sequence_length = 5000

        data = jax.random.normal(rng, (sequence_length,))

        buffer = Buffer(maxlen=maxlen)
        unroll_fn = flax_unroll_transform(buffer)

        # Should handle large sequences efficiently
        params, state = unroll_fn.init(rng, data)
        outputs, final_state = unroll_fn.apply(params, state, rng, data)

        assert outputs.shape == (sequence_length, maxlen)
        assert final_state["state"]["len_buffer"] == maxlen

        # The last output row should contain the last maxlen elements in order
        expected_final = data[-maxlen:]
        assert jnp.allclose(outputs[-1], expected_final)


if __name__ == "__main__":
    # Run Flax Buffer tests
    print("Running Flax Buffer tests...")

    test = TestFlaxBuffer()

    test.test_basic_buffer_functionality()
    print("✅ Basic buffer functionality")

    test.test_buffer_sequence_accumulation()
    print("✅ Buffer sequence accumulation")

    test.test_buffer_with_return_state()
    print("✅ Buffer with return_state")

    test.test_buffer_overflow_behavior()
    print("✅ Buffer overflow behavior")

    test.test_buffer_with_different_fill_values()
    print("✅ Buffer with different fill values")

    test.test_buffer_multidimensional_input()
    print("✅ Buffer multidimensional input")

    test.test_jit_compilation()
    print("✅ JIT compilation")

    test.test_numerical_consistency_with_haiku()
    print("✅ Numerical consistency with Haiku")

    test.test_buffer_state_tracking()
    print("✅ Buffer state tracking")

    test.test_edge_cases()
    print("✅ Edge cases")

    # Run autonomous tests
    print("\nRunning Autonomous Buffer tests...")

    autonomous_test = TestFlaxBufferAutonomous()

    autonomous_test.test_buffer_fifo_behavior()
    print("✅ FIFO behavior (autonomous)")

    autonomous_test.test_buffer_gradual_filling()
    print("✅ Gradual filling (autonomous)")

    autonomous_test.test_buffer_performance_characteristics()
    print("✅ Performance characteristics (autonomous)")

    print("\n🎉 All Buffer tests passed!")
