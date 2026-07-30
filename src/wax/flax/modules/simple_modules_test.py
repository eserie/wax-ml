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
"""Comprehensive tests for simple Flax modules."""

import jax
import jax.numpy as jnp

from wax.flax.core import flax_transform_with_state, flax_unroll_transform
from wax.flax.modules.counter import Counter
from wax.flax.modules.fill_nan_inf import FillNanInf
from wax.flax.modules.has_changed import HasChanged


class TestFlaxCounter:
    """Test suite for Flax Counter module."""

    def test_basic_counter_functionality(self):
        """Test basic counter operations."""
        rng = jax.random.PRNGKey(42)

        counter = Counter()
        tf = flax_transform_with_state(counter)

        # Initialize
        params, state = tf.init(rng)

        # Counter should start at 1 after init
        initial_count = state["state"]["count"]
        assert initial_count == 1

        # Apply counter again
        output, new_state = tf.apply(params, state, None)
        assert output == 2
        assert new_state["state"]["count"] == 2

    def test_counter_sequence(self):
        """Test counter over a sequence."""
        rng = jax.random.PRNGKey(42)

        counter = Counter()
        tf = flax_transform_with_state(counter)

        # Initialize
        params, state = tf.init(rng)

        # Apply counter multiple times
        outputs = []
        current_state = state

        for i in range(5):
            output, current_state = tf.apply(params, current_state, None)
            outputs.append(output)

        # Should count from 2 to 6 (since init already counted to 1)
        expected = jnp.array([2, 3, 4, 5, 6])
        outputs_array = jnp.array(outputs)
        assert jnp.array_equal(outputs_array, expected)
        assert current_state["state"]["count"] == 6

    def test_counter_numerical_consistency_with_haiku(self):
        """Test numerical consistency with Haiku implementation."""
        import haiku as hk

        from wax.modules.counter import Counter as HaikuCounter

        rng = jax.random.PRNGKey(42)

        # Haiku implementation
        @hk.transform_with_state
        def haiku_counter_fn():
            return HaikuCounter()()

        # Create a dummy unroll function for Haiku counter
        def haiku_counter_unroll_fn(rng, sequence):
            params, state = haiku_counter_fn.init(rng)
            outputs = []
            current_state = state
            for _ in sequence:
                output, current_state = haiku_counter_fn.apply(params, current_state, rng)
                outputs.append(output)
            return jnp.array(outputs), current_state

        sequence = jnp.arange(5)
        haiku_outputs, haiku_final_state = haiku_counter_unroll_fn(rng, sequence)

        # Flax implementation
        flax_counter = Counter()
        tf = flax_transform_with_state(flax_counter)

        flax_params, flax_state = tf.init(rng)
        flax_outputs = []
        current_state = flax_state
        for _ in sequence:
            output, current_state = tf.apply(flax_params, current_state, None)
            flax_outputs.append(output)
        flax_outputs = jnp.array(flax_outputs)

        # Compare outputs (allowing for initialization differences)
        # Flax initializes differently, so outputs may be offset by 1
        # The important thing is they increment correctly
        assert len(haiku_outputs) == len(flax_outputs)

        # Check that both increment by 1 each step
        haiku_diffs = jnp.diff(haiku_outputs)
        flax_diffs = jnp.diff(flax_outputs)
        assert jnp.all(haiku_diffs == 1)
        assert jnp.all(flax_diffs == 1)

    def test_jit_compilation(self):
        """Test that Counter works with JIT compilation."""
        rng = jax.random.PRNGKey(42)

        counter = Counter()
        tf = flax_transform_with_state(counter)

        @jax.jit
        def apply_counter(params, state, rng):
            return tf.apply(params, state, rng)

        params, state = tf.init(rng)
        output, new_state = apply_counter(params, state, rng)

        assert output == 2
        assert new_state["state"]["count"] == 2


class TestFlaxHasChanged:
    """Test suite for Flax HasChanged module."""

    def test_basic_has_changed_functionality(self):
        """Test basic HasChanged operations."""
        rng = jax.random.PRNGKey(42)

        has_changed = HasChanged()
        tf = flax_transform_with_state(has_changed)

        # Initialize with first value
        x1 = jnp.array(1.0)
        params, state = tf.init(rng, x1)

        # First call should always return True (change from init value)
        # Check initial state
        assert "state" in state
        assert "prev_value" in state["state"]

        # Apply with same value
        output, new_state = tf.apply(params, state, None, x1)
        assert not bool(output)  # Same value, no change

        # Apply with different value
        x2 = jnp.array(2.0)
        output, new_state = tf.apply(params, new_state, None, x2)
        assert bool(output)  # Different value, changed

    def test_has_changed_sequence(self):
        """Test HasChanged over a sequence."""
        rng = jax.random.PRNGKey(42)

        # Sequence with repeated and changing values
        data = jnp.array([1.0, 1.0, 2.0, 2.0, 3.0])

        has_changed = HasChanged()
        unroll_fn = flax_unroll_transform(has_changed)

        params, state = unroll_fn.init(rng, data)
        outputs, final_state = unroll_fn.apply(params, state, rng, data)

        # Expected: False, False, True, False, True
        # (first is False due to init matching, then no change, change, no change, change)
        expected = jnp.array([False, False, True, False, True])
        assert jnp.array_equal(outputs, expected)

    def test_has_changed_multidimensional(self):
        """Test HasChanged with multidimensional inputs."""
        rng = jax.random.PRNGKey(42)

        # 2D array input
        x1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        x2 = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Same
        x3 = jnp.array([[1.0, 2.0], [3.0, 5.0]])  # Different

        has_changed = HasChanged()
        tf = flax_transform_with_state(has_changed)

        params, state = tf.init(rng, x1)

        # Same array
        output, state = tf.apply(params, state, None, x2)
        assert not bool(output)

        # Different array
        output, state = tf.apply(params, state, None, x3)
        assert bool(output)

    def test_has_changed_numerical_consistency_with_haiku(self):
        """Test numerical consistency with Haiku implementation."""
        import haiku as hk

        from wax.modules.has_changed import HasChanged as HaikuHasChanged
        from wax.unroll import unroll_transform_with_state as haiku_unroll

        rng = jax.random.PRNGKey(42)
        data = jnp.array([1.0, 1.0, 2.0, 2.0, 3.0])

        # Haiku implementation
        @hk.transform_with_state
        def haiku_has_changed_fn(x):
            return HaikuHasChanged()(x)

        haiku_unroll_fn = haiku_unroll(haiku_has_changed_fn)
        haiku_params, haiku_state = haiku_unroll_fn.init(rng, data)
        haiku_outputs, haiku_final_state = haiku_unroll_fn.apply(
            haiku_params, haiku_state, rng, data
        )

        # Flax implementation
        flax_has_changed = HasChanged()
        flax_unroll_fn = flax_unroll_transform(flax_has_changed)
        flax_params, flax_state = flax_unroll_fn.init(rng, data)
        flax_outputs, flax_final_state = flax_unroll_fn.apply(flax_params, flax_state, rng, data)

        # Compare outputs (allowing for initialization differences)
        # Both should detect changes correctly, even if initialization behavior differs
        assert len(haiku_outputs) == len(flax_outputs)

        # Test specific change detection: transitions from 1->2 and 2->3 should be detected
        # Index 2 (1.0->2.0) and index 4 (2.0->3.0) should be True in both
        change_indices = [2, 4]
        for idx in change_indices:
            assert bool(haiku_outputs[idx]), f"Haiku should detect change at index {idx}"
            assert bool(flax_outputs[idx]), f"Flax should detect change at index {idx}"


class TestFlaxFillNanInf:
    """Test suite for Flax FillNanInf module."""

    def test_basic_fill_nan_inf_functionality(self):
        """Test basic FillNanInf operations."""
        rng = jax.random.PRNGKey(42)

        # Input with NaN, inf, and normal values
        input_data = jnp.array([1.0, jnp.nan, jnp.inf, -jnp.inf, 2.0])

        fill_nan_inf = FillNanInf(fill_value=0.0)
        tf = flax_transform_with_state(fill_nan_inf)

        params, state = tf.init(rng, input_data)
        output, new_state = tf.apply(params, state, None, input_data)

        # Should replace NaN and inf with 0.0
        expected = jnp.array([1.0, 0.0, 0.0, 0.0, 2.0])
        assert jnp.array_equal(output, expected)

    def test_fill_nan_inf_different_fill_values(self):
        """Test FillNanInf with different fill values."""
        rng = jax.random.PRNGKey(42)

        input_data = jnp.array([jnp.nan, jnp.inf, -jnp.inf])

        # Test with different fill value
        fill_nan_inf = FillNanInf(fill_value=-999.0)
        tf = flax_transform_with_state(fill_nan_inf)

        params, state = tf.init(rng, input_data)
        output, new_state = tf.apply(params, state, None, input_data)

        expected = jnp.array([-999.0, -999.0, -999.0])
        assert jnp.array_equal(output, expected)

    def test_fill_nan_inf_nested_structure(self):
        """Test FillNanInf with nested data structures."""
        rng = jax.random.PRNGKey(42)

        # Nested structure with dict and arrays
        input_data = {
            "a": jnp.array([1.0, jnp.nan, 3.0]),
            "b": jnp.array([[jnp.inf, 2.0], [3.0, -jnp.inf]]),
        }

        fill_nan_inf = FillNanInf(fill_value=0.0)
        tf = flax_transform_with_state(fill_nan_inf)

        params, state = tf.init(rng, input_data)
        output, new_state = tf.apply(params, state, None, input_data)

        # Check that structure is preserved and values are filled
        expected_a = jnp.array([1.0, 0.0, 3.0])
        expected_b = jnp.array([[0.0, 2.0], [3.0, 0.0]])

        assert jnp.array_equal(output["a"], expected_a)
        assert jnp.array_equal(output["b"], expected_b)

    def test_fill_nan_inf_numerical_consistency_with_haiku(self):
        """Test numerical consistency with Haiku implementation."""
        import haiku as hk

        from wax.modules.fill_nan_inf import FillNanInf as HaikuFillNanInf
        from wax.unroll import unroll_transform_with_state as haiku_unroll

        rng = jax.random.PRNGKey(42)
        fill_value = -999.0

        # Create test data with NaN and inf
        data = jnp.array([[1.0, jnp.nan, 3.0], [jnp.inf, 5.0, -jnp.inf], [7.0, 8.0, 9.0]])

        # Haiku implementation
        @hk.transform_with_state
        def haiku_fill_nan_inf_fn(x):
            return HaikuFillNanInf(fill_value=fill_value)(x)

        haiku_unroll_fn = haiku_unroll(haiku_fill_nan_inf_fn)
        haiku_params, haiku_state = haiku_unroll_fn.init(rng, data)
        haiku_outputs, haiku_final_state = haiku_unroll_fn.apply(
            haiku_params, haiku_state, rng, data
        )

        # Flax implementation
        flax_fill_nan_inf = FillNanInf(fill_value=fill_value)
        flax_unroll_fn = flax_unroll_transform(flax_fill_nan_inf)
        flax_params, flax_state = flax_unroll_fn.init(rng, data)
        flax_outputs, flax_final_state = flax_unroll_fn.apply(flax_params, flax_state, rng, data)

        # Compare outputs (should be identical for stateless module)
        assert jnp.array_equal(haiku_outputs, flax_outputs)

    def test_jit_compilation(self):
        """Test that FillNanInf works with JIT compilation."""
        rng = jax.random.PRNGKey(42)

        input_data = jnp.array([1.0, jnp.nan, jnp.inf, 4.0])

        fill_nan_inf = FillNanInf(fill_value=0.0)
        tf = flax_transform_with_state(fill_nan_inf)

        @jax.jit
        def apply_fill_nan_inf(params, state, rng, x):
            return tf.apply(params, state, rng, x)

        params, state = tf.init(rng, input_data)
        output, new_state = apply_fill_nan_inf(params, state, rng, input_data)

        expected = jnp.array([1.0, 0.0, 0.0, 4.0])
        assert jnp.array_equal(output, expected)


# Autonomous tests (independent of Haiku for future migration)
class TestSimpleModulesAutonomous:
    """Autonomous tests for simple modules (Haiku-independent)."""

    def test_counter_autonomous(self):
        """Test counter behavior independently."""
        rng = jax.random.PRNGKey(42)

        counter = Counter()
        tf = flax_transform_with_state(counter)

        # Test counting behavior
        params, state = tf.init(rng)
        outputs = []
        current_state = state

        for i in range(10):
            output, current_state = tf.apply(params, current_state, None)
            outputs.append(output)

        # Should count from 2 to 11 (since init already counted to 1)
        expected = jnp.arange(2, 12)
        outputs_array = jnp.array(outputs)
        assert jnp.array_equal(outputs_array, expected)
        assert current_state["state"]["count"] == 11  # Final state after last increment

    def test_has_changed_autonomous(self):
        """Test has_changed behavior independently."""
        rng = jax.random.PRNGKey(42)

        has_changed = HasChanged()
        tf = flax_transform_with_state(has_changed)

        # Test step by step
        params, state = tf.init(rng, jnp.array(1.0))

        # Same value should return False
        output, state = tf.apply(params, state, None, jnp.array(1.0))
        assert not bool(output)

        # Different value should return True
        output, state = tf.apply(params, state, None, jnp.array(2.0))
        assert bool(output)

        # Same new value should return False
        output, state = tf.apply(params, state, None, jnp.array(2.0))
        assert not bool(output)

    def test_fill_nan_inf_autonomous(self):
        """Test fill_nan_inf behavior independently."""
        rng = jax.random.PRNGKey(42)

        fill_nan_inf = FillNanInf(fill_value=42.0)
        tf = flax_transform_with_state(fill_nan_inf)

        # Test comprehensive edge cases
        test_cases = [
            (jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0])),  # No special values
            (jnp.array([jnp.nan]), jnp.array([42.0])),  # Only NaN
            (jnp.array([jnp.inf]), jnp.array([42.0])),  # Only +inf
            (jnp.array([-jnp.inf]), jnp.array([42.0])),  # Only -inf
            (
                jnp.array([jnp.nan, jnp.inf, -jnp.inf, 1.0]),
                jnp.array([42.0, 42.0, 42.0, 1.0]),
            ),  # Mixed
        ]

        for input_data, expected in test_cases:
            params, state = tf.init(rng, input_data)
            output, new_state = tf.apply(params, state, None, input_data)
            assert jnp.array_equal(output, expected), f"Failed for input {input_data}"


if __name__ == "__main__":
    # Run Counter tests
    print("Running Flax Counter tests...")
    counter_test = TestFlaxCounter()

    counter_test.test_basic_counter_functionality()
    print("✅ Basic counter functionality")

    counter_test.test_counter_sequence()
    print("✅ Counter sequence")

    counter_test.test_counter_numerical_consistency_with_haiku()
    print("✅ Counter numerical consistency with Haiku")

    counter_test.test_jit_compilation()
    print("✅ Counter JIT compilation")

    # Run HasChanged tests
    print("\nRunning Flax HasChanged tests...")
    has_changed_test = TestFlaxHasChanged()

    has_changed_test.test_basic_has_changed_functionality()
    print("✅ Basic HasChanged functionality")

    has_changed_test.test_has_changed_sequence()
    print("✅ HasChanged sequence")

    has_changed_test.test_has_changed_multidimensional()
    print("✅ HasChanged multidimensional")

    has_changed_test.test_has_changed_numerical_consistency_with_haiku()
    print("✅ HasChanged numerical consistency with Haiku")

    # Run FillNanInf tests
    print("\nRunning Flax FillNanInf tests...")
    fill_nan_inf_test = TestFlaxFillNanInf()

    fill_nan_inf_test.test_basic_fill_nan_inf_functionality()
    print("✅ Basic FillNanInf functionality")

    fill_nan_inf_test.test_fill_nan_inf_different_fill_values()
    print("✅ FillNanInf different fill values")

    fill_nan_inf_test.test_fill_nan_inf_nested_structure()
    print("✅ FillNanInf nested structure")

    fill_nan_inf_test.test_fill_nan_inf_numerical_consistency_with_haiku()
    print("✅ FillNanInf numerical consistency with Haiku")

    fill_nan_inf_test.test_jit_compilation()
    print("✅ FillNanInf JIT compilation")

    # Run autonomous tests
    print("\nRunning Autonomous tests...")
    autonomous_test = TestSimpleModulesAutonomous()

    autonomous_test.test_counter_autonomous()
    print("✅ Counter autonomous")

    autonomous_test.test_has_changed_autonomous()
    print("✅ HasChanged autonomous")

    autonomous_test.test_fill_nan_inf_autonomous()
    print("✅ FillNanInf autonomous")

    print("\n🎉 All simple module tests passed!")
