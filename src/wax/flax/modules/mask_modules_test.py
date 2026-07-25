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
"""Comprehensive tests for mask-related Flax modules."""

import jax
import jax.numpy as jnp

from wax.flax.core import flax_transform_with_state
from wax.flax.modules import ApplyMask, MaskMean, MaskNormalize, MaskStd


class TestFlaxApplyMask:
    """Test suite for Flax ApplyMask module."""

    def test_basic_apply_mask_functionality(self):
        """Test basic ApplyMask operations."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = jnp.array([True, False, True, False, True])

        apply_mask = ApplyMask(mask_value=-999.0)
        tf = flax_transform_with_state(apply_mask)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Check that masked values are replaced with mask_value
        expected = jnp.array([1.0, -999.0, 3.0, -999.0, 5.0])
        assert jnp.allclose(output, expected)

    def test_apply_mask_with_axis(self):
        """Test ApplyMask with axis parameter."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = jnp.array([True, False, True])

        apply_mask = ApplyMask(axis=0, mask_value=0.0)
        tf = flax_transform_with_state(apply_mask)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Check that axis=0 masking works correctly
        expected = jnp.array([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]])
        assert jnp.allclose(output, expected)

    def test_apply_mask_numerical_consistency_with_haiku(self):
        """Test numerical consistency with Haiku implementation."""
        import haiku as hk

        from wax.modules.apply_mask import ApplyMask as HaikuApplyMask

        rng = jax.random.PRNGKey(42)
        data = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mask = jnp.array([True, True, False, True, False])
        mask_value = -1.0

        # Haiku implementation
        @hk.transform
        def haiku_apply_mask_fn(mask, x):
            return HaikuApplyMask(mask_value=mask_value)(mask, x)

        haiku_params = haiku_apply_mask_fn.init(rng, mask, data)
        haiku_output = haiku_apply_mask_fn.apply(haiku_params, rng, mask, data)

        # Flax implementation
        flax_apply_mask = ApplyMask(mask_value=mask_value)
        flax_tf = flax_transform_with_state(flax_apply_mask)
        flax_params, flax_state = flax_tf.init(rng, mask, data)
        flax_output, flax_new_state = flax_tf.apply(flax_params, flax_state, None, mask, data)

        # Compare outputs (should be identical for this stateless operation)
        assert jnp.allclose(haiku_output, flax_output)


class TestFlaxMaskMean:
    """Test suite for Flax MaskMean module."""

    def test_basic_mask_mean_functionality(self):
        """Test basic MaskMean operations."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = jnp.array([True, False, True, False, True])

        mask_mean = MaskMean()
        tf = flax_transform_with_state(mask_mean)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Expected mean: (1.0 + 3.0 + 5.0) / 3 = 3.0
        expected = 3.0
        assert jnp.allclose(output, expected)

    def test_mask_mean_with_axis(self):
        """Test MaskMean with axis parameter."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = jnp.array([True, False, True])

        mask_mean = MaskMean(axis=0)
        tf = flax_transform_with_state(mask_mean)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Expected mean along axis 0: [(1+5)/2, (2+6)/2] = [3.0, 4.0]
        expected = jnp.array([3.0, 4.0])
        assert jnp.allclose(output, expected)

    def test_mask_mean_numerical_consistency_with_haiku(self):
        """Test numerical consistency with Haiku implementation."""
        import haiku as hk

        from wax.modules.mask_mean import MaskMean as HaikuMaskMean

        rng = jax.random.PRNGKey(42)
        data = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
        mask = jnp.array([True, True, False, True, True])

        # Haiku implementation
        @hk.transform
        def haiku_mask_mean_fn(mask, x):
            return HaikuMaskMean()(mask, x)

        haiku_params = haiku_mask_mean_fn.init(rng, mask, data)
        haiku_output = haiku_mask_mean_fn.apply(haiku_params, rng, mask, data)

        # Flax implementation
        flax_mask_mean = MaskMean()
        flax_tf = flax_transform_with_state(flax_mask_mean)
        flax_params, flax_state = flax_tf.init(rng, mask, data)
        flax_output, flax_new_state = flax_tf.apply(flax_params, flax_state, None, mask, data)

        # Compare outputs (should be very close)
        assert jnp.allclose(haiku_output, flax_output, rtol=1e-6)


class TestFlaxMaskStd:
    """Test suite for Flax MaskStd module."""

    def test_basic_mask_std_functionality(self):
        """Test basic MaskStd operations."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = jnp.array([True, False, True, False, True])

        mask_std = MaskStd()
        tf = flax_transform_with_state(mask_std)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Check that output is finite and non-negative
        assert jnp.isfinite(output)
        assert output >= 0

    def test_mask_std_assume_centered(self):
        """Test MaskStd with assume_centered=True."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = jnp.array([True, False, True, False, True])

        mask_std = MaskStd(assume_centered=True)
        tf = flax_transform_with_state(mask_std)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Check that output is finite and non-negative
        assert jnp.isfinite(output)
        assert output >= 0

    def test_mask_std_numerical_consistency_with_haiku(self):
        """Test numerical consistency with Haiku implementation."""
        import haiku as hk

        from wax.modules.mask_std import MaskStd as HaikuMaskStd

        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0])
        mask = jnp.array([True, True, False, True, True])

        # Haiku implementation
        @hk.transform
        def haiku_mask_std_fn(mask, x):
            return HaikuMaskStd()(mask, x)

        haiku_params = haiku_mask_std_fn.init(rng, mask, data)
        haiku_output = haiku_mask_std_fn.apply(haiku_params, rng, mask, data)

        # Flax implementation
        flax_mask_std = MaskStd()
        flax_tf = flax_transform_with_state(flax_mask_std)
        flax_params, flax_state = flax_tf.init(rng, mask, data)
        flax_output, flax_new_state = flax_tf.apply(flax_params, flax_state, None, mask, data)

        # Compare outputs (allowing for small numerical differences)
        assert jnp.allclose(haiku_output, flax_output, rtol=1e-5)


class TestFlaxMaskNormalize:
    """Test suite for Flax MaskNormalize module."""

    def test_basic_mask_normalize_functionality(self):
        """Test basic MaskNormalize operations."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = jnp.array([True, False, True, False, True])

        mask_normalize = MaskNormalize()
        tf = flax_transform_with_state(mask_normalize)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Check that output is finite
        assert jnp.all(jnp.isfinite(output))

    def test_mask_normalize_assume_centered(self):
        """Test MaskNormalize with assume_centered=True."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = jnp.array([True, False, True, False, True])

        mask_normalize = MaskNormalize(assume_centered=True)
        tf = flax_transform_with_state(mask_normalize)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Check that output is finite
        assert jnp.all(jnp.isfinite(output))

    def test_mask_normalize_numerical_consistency_with_haiku(self):
        """Test numerical consistency with Haiku implementation."""
        import haiku as hk

        from wax.modules.mask_normalize import MaskNormalize as HaikuMaskNormalize

        rng = jax.random.PRNGKey(42)
        data = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
        mask = jnp.array([True, True, False, True, True])

        # Haiku implementation
        @hk.transform
        def haiku_mask_normalize_fn(mask, x):
            return HaikuMaskNormalize()(mask, x)

        haiku_params = haiku_mask_normalize_fn.init(rng, mask, data)
        haiku_output = haiku_mask_normalize_fn.apply(haiku_params, rng, mask, data)

        # Flax implementation
        flax_mask_normalize = MaskNormalize()
        flax_tf = flax_transform_with_state(flax_mask_normalize)
        flax_params, flax_state = flax_tf.init(rng, mask, data)
        flax_output, flax_new_state = flax_tf.apply(flax_params, flax_state, None, mask, data)

        # Compare outputs (allowing for small numerical differences)
        assert jnp.allclose(haiku_output, flax_output, rtol=1e-5)


# Autonomous tests (independent of Haiku for future migration)
class TestMaskModulesAutonomous:
    """Autonomous tests for mask modules (Haiku-independent)."""

    def test_apply_mask_autonomous(self):
        """Test ApplyMask behavior independently."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([10, 20, 30, 40, 50])
        mask = jnp.array([True, False, True, False, True])

        apply_mask = ApplyMask(mask_value=-1)
        tf = flax_transform_with_state(apply_mask)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Expected: [10, -1, 30, -1, 50]
        expected = jnp.array([10, -1, 30, -1, 50])
        assert jnp.array_equal(output, expected)

    def test_mask_mean_autonomous(self):
        """Test MaskMean behavior independently."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([2, 4, 6, 8, 10])
        mask = jnp.array([True, True, False, False, True])

        mask_mean = MaskMean()
        tf = flax_transform_with_state(mask_mean)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Expected mean: (2 + 4 + 10) / 3 = 16 / 3 ≈ 5.333
        expected = 16.0 / 3.0
        assert jnp.allclose(output, expected)

    def test_mask_std_autonomous(self):
        """Test MaskStd behavior independently."""
        rng = jax.random.PRNGKey(42)
        # Use simple data for predictable std calculation
        data = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0])
        mask = jnp.array([True, False, True, False, True])  # Use [0, 4, 8]

        mask_std = MaskStd()
        tf = flax_transform_with_state(mask_std)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Manual calculation: mean = (0+4+8)/3 = 4
        # variance = ((0-4)^2 + (4-4)^2 + (8-4)^2) / 3 = (16 + 0 + 16) / 3 = 32/3
        # std = sqrt(32/3) ≈ 3.266
        expected_var = 32.0 / 3.0
        expected_std = jnp.sqrt(expected_var)
        assert jnp.allclose(output, expected_std, rtol=1e-5)

    def test_mask_normalize_autonomous(self):
        """Test MaskNormalize behavior independently."""
        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = jnp.array([True, False, True, False, True])

        mask_normalize = MaskNormalize()
        tf = flax_transform_with_state(mask_normalize)

        params, state = tf.init(rng, mask, data)
        output, new_state = tf.apply(params, state, None, mask, data)

        # Check that output is finite and has reasonable values
        assert jnp.all(jnp.isfinite(output))
        # Normalized values should generally be in a reasonable range
        assert jnp.all(jnp.abs(output) < 100)  # Sanity check


if __name__ == "__main__":
    # Run ApplyMask tests
    print("Running Flax ApplyMask tests...")
    apply_mask_test = TestFlaxApplyMask()

    apply_mask_test.test_basic_apply_mask_functionality()
    print("✅ Basic ApplyMask functionality")

    apply_mask_test.test_apply_mask_with_axis()
    print("✅ ApplyMask with axis")

    apply_mask_test.test_apply_mask_numerical_consistency_with_haiku()
    print("✅ ApplyMask numerical consistency with Haiku")

    # Run MaskMean tests
    print("\nRunning Flax MaskMean tests...")
    mask_mean_test = TestFlaxMaskMean()

    mask_mean_test.test_basic_mask_mean_functionality()
    print("✅ Basic MaskMean functionality")

    mask_mean_test.test_mask_mean_with_axis()
    print("✅ MaskMean with axis")

    mask_mean_test.test_mask_mean_numerical_consistency_with_haiku()
    print("✅ MaskMean numerical consistency with Haiku")

    # Run MaskStd tests
    print("\nRunning Flax MaskStd tests...")
    mask_std_test = TestFlaxMaskStd()

    mask_std_test.test_basic_mask_std_functionality()
    print("✅ Basic MaskStd functionality")

    mask_std_test.test_mask_std_assume_centered()
    print("✅ MaskStd assume_centered")

    mask_std_test.test_mask_std_numerical_consistency_with_haiku()
    print("✅ MaskStd numerical consistency with Haiku")

    # Run MaskNormalize tests
    print("\nRunning Flax MaskNormalize tests...")
    mask_normalize_test = TestFlaxMaskNormalize()

    mask_normalize_test.test_basic_mask_normalize_functionality()
    print("✅ Basic MaskNormalize functionality")

    mask_normalize_test.test_mask_normalize_assume_centered()
    print("✅ MaskNormalize assume_centered")

    mask_normalize_test.test_mask_normalize_numerical_consistency_with_haiku()
    print("✅ MaskNormalize numerical consistency with Haiku")

    # Run autonomous tests
    print("\nRunning Autonomous Mask tests...")
    autonomous_test = TestMaskModulesAutonomous()

    autonomous_test.test_apply_mask_autonomous()
    print("✅ ApplyMask autonomous")

    autonomous_test.test_mask_mean_autonomous()
    print("✅ MaskMean autonomous")

    autonomous_test.test_mask_std_autonomous()
    print("✅ MaskStd autonomous")

    autonomous_test.test_mask_normalize_autonomous()
    print("✅ MaskNormalize autonomous")

    print("\n🎉 All mask module tests passed!")
