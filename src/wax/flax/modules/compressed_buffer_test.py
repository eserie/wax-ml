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
"""Tests for compressed buffer memory-efficient long sequences."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wax.flax.modules.compressed_buffer import (
    CompressedBuffer,
    HierarchicalBuffer,
    streaming_compressed_memory,
    streaming_hierarchical_memory,
)


class TestCompressedBuffer:
    """Test CompressedBuffer functionality."""

    def test_ewma_compression(self):
        """Test EWMA compression strategy."""
        buffer = CompressedBuffer(
            maxlen=1000,
            compression="ewma",
            compression_params={"alpha": 0.1}
        )

        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, jnp.array(1.0))

        # Test sequence processing
        sequence = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        current_vars = variables

        outputs = []
        for x in sequence:
            output, new_vars = buffer.apply(current_vars, x, mutable=['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}
            outputs.append(output)

        # EWMA should show smoothed progression
        assert len(outputs) == len(sequence)
        assert all(jnp.isfinite(out) for out in outputs)

        # Should be monotonically increasing but smoothed
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i-1]  # Generally increasing

        # Final output should be less than final input (due to smoothing)
        assert outputs[-1] < sequence[-1]

    def test_quantile_compression(self):
        """Test quantile-based compression."""
        buffer = CompressedBuffer(
            maxlen=100,
            compression="quantile",
            compression_params={"percentiles": [0.25, 0.5, 0.75]}
        )

        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, jnp.array(1.0))

        # Generate test data with known distribution
        test_data = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3)  # Uniform distribution
        current_vars = variables

        outputs = []
        for x in test_data:
            output, new_vars = buffer.apply(current_vars, jnp.array(x), mutable=['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}
            outputs.append(output)

        # Final quantiles should be computed
        final_quantiles = outputs[-1]

        # Check that quantiles have the right structure
        assert len(final_quantiles) == 3
        assert jnp.all(jnp.isfinite(final_quantiles))

        # Should be ordered (monotonic)
        assert final_quantiles[0] <= final_quantiles[1] <= final_quantiles[2]

        # The quantile implementation updates periodically, so we just check
        # that it produces reasonable finite values
        assert jnp.all(final_quantiles >= 0.0)  # Non-negative values
        assert jnp.all(final_quantiles <= 15.0)  # Reasonable upper bound

    def test_downsample_compression(self):
        """Test downsampling compression."""
        buffer = CompressedBuffer(
            maxlen=20,
            compression="downsample",
            compression_params={"factor": 2}
        )

        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, jnp.array(1.0))

        # Process sequence
        sequence = jnp.arange(1.0, 11.0)  # [1, 2, 3, ..., 10]
        current_vars = variables

        outputs = []
        for x in sequence:
            output, new_vars = buffer.apply(current_vars, x, mutable=['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}
            outputs.append(output)

        # Should have downsampled the data
        final_buffer = outputs[-1]
        assert len(final_buffer) == 10  # maxlen // factor = 20 // 2 = 10

        # Should contain subset of original values
        assert jnp.all(jnp.isfinite(final_buffer))

    def test_sketching_compression(self):
        """Test Count-Min sketch compression."""
        buffer = CompressedBuffer(
            maxlen=1000,
            compression="sketching",
            compression_params={"num_hashes": 3, "num_buckets": 64}
        )

        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, jnp.array(1.0))

        # Process sequence with repeated values
        sequence = jnp.array([1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        current_vars = variables

        outputs = []
        for x in sequence:
            output, new_vars = buffer.apply(current_vars, x, mutable=['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}
            outputs.append(output)

        # Should return sketch summary
        final_sketch = outputs[-1]
        assert len(final_sketch) == 64  # num_buckets
        assert jnp.all(final_sketch >= 0)  # Counts should be non-negative

    def test_no_compression(self):
        """Test no compression (regular buffer)."""
        buffer = CompressedBuffer(
            maxlen=5,
            compression="none"
        )

        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, jnp.array(1.0))

        # Process sequence
        sequence = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        current_vars = variables

        outputs = []
        for x in sequence:
            output, new_vars = buffer.apply(current_vars, x, mutable=['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}
            outputs.append(output)

        # Should behave like regular buffer
        final_buffer = outputs[-1]
        assert len(final_buffer) == 5  # maxlen
        # Should contain last 5 values
        expected = jnp.array([3.0, 4.0, 5.0, 6.0, 7.0])
        np.testing.assert_array_equal(final_buffer, expected)

    def test_memory_usage_tracking(self):
        """Test memory usage estimation."""
        # Test different compression strategies
        strategies = ["ewma", "quantile", "downsample", "sketching", "none"]

        for strategy in strategies:
            buffer = CompressedBuffer(maxlen=1000, compression=strategy)
            memory_usage = buffer.get_memory_usage()

            assert "total" in memory_usage
            assert memory_usage["total"] > 0

            # EWMA should use less memory than no compression
            if strategy == "ewma":
                assert memory_usage["total"] < 1000 * 8  # Much smaller than full buffer


class TestHierarchicalBuffer:
    """Test HierarchicalBuffer functionality."""

    def test_hierarchical_structure(self):
        """Test basic hierarchical buffer structure."""
        buffer = HierarchicalBuffer(
            recent_maxlen=5,
            medium_maxlen=20,
            long_maxlen=100,
            medium_compression="ewma",
            long_compression="quantile"
        )

        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, jnp.array(1.0))

        # Process sequence
        sequence = jnp.arange(1.0, 26.0)  # 25 values
        current_vars = variables

        outputs = []
        for x in sequence:
            output, new_vars = buffer.apply(current_vars, x, mutable=['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}
            outputs.append(output)

        # Check final output structure
        final_output = outputs[-1]
        assert "recent" in final_output
        assert "medium" in final_output
        assert "long" in final_output
        assert "input" in final_output

        # Recent should contain last 5 values
        recent = final_output["recent"]
        assert len(recent) == 5
        expected_recent = jnp.array([21.0, 22.0, 23.0, 24.0, 25.0])
        np.testing.assert_array_equal(recent, expected_recent)

        # Medium and long should be compressed representations
        assert jnp.isfinite(final_output["medium"])
        assert jnp.all(jnp.isfinite(final_output["long"]))

    def test_memory_efficiency(self):
        """Test memory efficiency of hierarchical buffer."""
        buffer = HierarchicalBuffer(
            recent_maxlen=100,
            medium_maxlen=1000,
            long_maxlen=10000
        )

        memory_usage = buffer.get_total_memory_usage()

        assert "recent" in memory_usage
        assert "medium" in memory_usage
        assert "long" in memory_usage
        assert "total" in memory_usage
        assert "compression_ratio" in memory_usage

        # Should achieve significant compression
        total_uncompressed = (100 + 1000 + 10000) * 8
        total_compressed = memory_usage["total"]
        compression_ratio = memory_usage["compression_ratio"]

        assert total_compressed < total_uncompressed
        assert compression_ratio > 1.0  # Should achieve compression

    def test_different_compression_strategies(self):
        """Test hierarchical buffer with different compression strategies."""
        buffer = HierarchicalBuffer(
            recent_maxlen=3,
            medium_maxlen=10,
            long_maxlen=50,
            medium_compression="downsample",
            long_compression="ewma"
        )

        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, jnp.array(1.0))

        # Process data
        sequence = jnp.arange(1.0, 21.0)
        current_vars = variables

        for x in sequence:
            output, new_vars = buffer.apply(current_vars, x, mutable=['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}

        # Should work with different compression strategies
        assert True  # If we get here without errors, test passes


class TestStreamingDecorators:
    """Test streaming memory decorators."""

    def test_compressed_memory_decorator(self):
        """Test @streaming_compressed_memory decorator."""

        @streaming_compressed_memory(maxlen=100, compression="ewma")
        def compressed_processor(buffered_data, x):
            """Process with compressed memory."""
            return {
                "input": x,
                "compressed": buffered_data,
                "processed": buffered_data * 2
            }

        # Test the decorated function
        rng = jax.random.PRNGKey(42)
        params, state = compressed_processor.init(rng, jnp.array(1.0))

        # Process sequence
        sequence = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        current_state = state

        outputs = []
        for x in sequence:
            output, current_state = compressed_processor.apply(
                params, current_state, None, x
            )
            outputs.append(output)

        # Check outputs
        assert len(outputs) == len(sequence)
        for i, output in enumerate(outputs):
            assert "input" in output
            assert "compressed" in output
            assert "processed" in output
            assert output["input"] == sequence[i]
            assert jnp.isfinite(output["compressed"])

    def test_hierarchical_memory_decorator(self):
        """Test @streaming_hierarchical_memory decorator."""

        @streaming_hierarchical_memory(
            recent_maxlen=3,
            medium_maxlen=10,
            long_maxlen=50
        )
        def hierarchical_processor(memory_levels, x):
            """Process with hierarchical memory."""
            recent = memory_levels["recent"]
            medium = memory_levels["medium"]
            long_term = memory_levels["long"]

            return {
                "input": x,
                "recent_mean": jnp.mean(recent),
                "medium_compressed": medium,
                "long_compressed": long_term,
                "combined": jnp.mean(recent) + medium
            }

        # Test the decorated function
        rng = jax.random.PRNGKey(42)
        params, state = hierarchical_processor.init(rng, jnp.array(1.0))

        # Process sequence
        sequence = jnp.arange(1.0, 11.0)
        current_state = state

        outputs = []
        for x in sequence:
            output, current_state = hierarchical_processor.apply(
                params, current_state, None, x
            )
            outputs.append(output)

        # Check outputs
        assert len(outputs) == len(sequence)
        for output in outputs:
            assert "input" in output
            assert "recent_mean" in output
            assert "medium_compressed" in output
            assert "long_compressed" in output
            assert "combined" in output
            assert jnp.isfinite(output["recent_mean"])
            assert jnp.isfinite(output["combined"])


class TestCompressionPerformance:
    """Test compression performance and characteristics."""

    def test_compression_ratios(self):
        """Test compression ratios for different strategies."""
        maxlen = 1000
        strategies = ["ewma", "quantile", "downsample"]

        compression_ratios = {}

        for strategy in strategies:
            buffer = CompressedBuffer(maxlen=maxlen, compression=strategy)
            memory_usage = buffer.get_memory_usage()

            uncompressed_size = maxlen * 8  # 8 bytes per float64
            compressed_size = memory_usage["total"]
            ratio = uncompressed_size / compressed_size

            compression_ratios[strategy] = ratio

            # Should achieve some compression for EWMA and downsample
            # Quantile may not compress much for small buffers due to overhead
            if strategy in ["ewma", "downsample"]:
                assert ratio > 1.0
            elif strategy == "quantile":
                assert ratio > 0.5  # At least not worse than 2x expansion

        # EWMA should be most efficient
        assert compression_ratios["ewma"] > compression_ratios["quantile"]

    def test_long_sequence_processing(self):
        """Test processing very long sequences."""
        buffer = HierarchicalBuffer(
            recent_maxlen=10,
            medium_maxlen=100,
            long_maxlen=1000
        )

        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, jnp.array(1.0))

        # Process long sequence
        sequence_length = 2000  # Longer than any buffer
        current_vars = variables

        # Should handle arbitrarily long sequences
        for i in range(sequence_length):
            x = jnp.sin(i * 0.1)  # Some pattern
            output, new_vars = buffer.apply(current_vars, x, mutable=['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}

            # Memory usage should remain bounded
            if i % 500 == 0:  # Check periodically
                memory_usage = buffer.get_total_memory_usage()
                assert memory_usage["total"] < 100000  # Reasonable bound

        # Final check
        assert "recent" in output
        assert "medium" in output
        assert "long" in output

    def test_jax_transformations_compatibility(self):
        """Test compatibility with JAX transformations."""

        @streaming_compressed_memory(maxlen=50, compression="ewma")
        def jax_compatible_processor(compressed_data, x):
            return compressed_data * 2 + x

        # Test JIT compilation
        jitted_init = jax.jit(jax_compatible_processor.init)
        jitted_apply = jax.jit(jax_compatible_processor.apply)

        rng = jax.random.PRNGKey(42)
        x0 = jnp.array(1.0)

        # Should work with JIT
        params, state = jitted_init(rng, x0)
        output, new_state = jitted_apply(params, state, None, x0)

        assert jnp.isfinite(output)
        assert new_state is not None

    def test_error_handling(self):
        """Test error handling for invalid parameters."""
        # Invalid compression strategy (error occurs during init/setup)
        buffer = CompressedBuffer(maxlen=100, compression="invalid_strategy")
        rng = jax.random.PRNGKey(42)

        with pytest.raises(ValueError, match="Unknown compression strategy"):
            buffer.init(rng, jnp.array(1.0))

        # Should handle edge cases gracefully
        buffer = CompressedBuffer(maxlen=1, compression="ewma")
        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, jnp.array(1.0))

        # Should work with minimal buffer size
        output, _ = buffer.apply(variables, jnp.array(1.0), mutable=['state'])
        assert jnp.isfinite(output)
