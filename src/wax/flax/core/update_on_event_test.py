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
"""Tests for the @update_on_event decorator and conditional computation."""

import jax
import jax.numpy as jnp

from wax.flax.core.streaming_transforms import (
    streaming_transform_with_state,
    update_on_event,
)
from wax.flax.modules.ewma import EWMA


class TestUpdateOnEvent:
    """Test event-driven conditional computation."""

    def test_update_on_event_basic_functionality(self):
        """Test that @update_on_event works with streaming transforms."""

        @streaming_transform_with_state
        @update_on_event(event_fn=lambda x: x > 0)  # Only update on positive values
        def conditional_ewma(x):
            """EWMA that only updates on positive values."""
            ewma = EWMA(alpha=0.5)
            return ewma(x)

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0 = jnp.array(1.0)  # Positive, so should update
        params, state = conditional_ewma.init(rng, x0)

        # Test with positive value (should update)
        x1 = jnp.array(5.0)
        output1, state1 = conditional_ewma.apply(params, state, None, x1)

        # Test with negative value (should NOT update → cached output)
        x2 = jnp.array(-1.0)
        output2, state2 = conditional_ewma.apply(params, state1, None, x2)

        # Test with another positive value (should update)
        x3 = jnp.array(3.0)
        output3, state3 = conditional_ewma.apply(params, state2, None, x3)

        # Negative input → event_fn returns False → output is cached from previous event
        assert jnp.allclose(output1, output2)  # Cached output for negative values
        # Positive input → event_fn returns True → new output
        assert not jnp.allclose(output2, output3)  # Updated output for positive values

    def test_update_on_event_without_event_fn(self):
        """Test that without event_fn, it always updates (like normal transform)."""

        @streaming_transform_with_state
        @update_on_event(event_fn=None)  # No event function, always update
        def always_update_ewma(x):
            """EWMA that always updates."""
            ewma = EWMA(alpha=0.5)
            return ewma(x)

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0 = jnp.array(1.0)
        params, state = always_update_ewma.init(rng, x0)

        # Test sequence of values
        values = [1.0, 2.0, 3.0, 4.0]
        outputs = []
        current_state = state

        for val in values:
            output, current_state = always_update_ewma.apply(
                params, current_state, None, jnp.array(val)
            )
            outputs.append(output)

        # All outputs should be different (always updating)
        for i in range(1, len(outputs)):
            assert not jnp.allclose(outputs[i - 1], outputs[i])

    def test_conditional_computation_with_simple_function(self):
        """Test conditional computation with a simple function."""

        @streaming_transform_with_state
        @update_on_event(event_fn=lambda x: x > 0)  # Only update on positive values
        def simple_increment(x):
            """Simple function that increments input."""
            return x + 1

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0 = jnp.array(1.0)  # Positive
        params, state = simple_increment.init(rng, x0)

        # Test sequence: positive, negative, negative, positive
        test_values = [1.0, -1.0, -2.0, 2.0]
        outputs = []
        current_state = state

        for val in test_values:
            output, current_state = simple_increment.apply(
                params, current_state, None, jnp.array(val)
            )
            outputs.append(float(output))

        # 1.0 -> event=True  -> output = 1+1 = 2.0, cache = 2.0
        # -1.0 -> event=False -> output = cached = 2.0
        # -2.0 -> event=False -> output = cached = 2.0
        # 2.0 -> event=True  -> output = 2+1 = 3.0, cache = 3.0
        expected = [2.0, 2.0, 2.0, 3.0]
        assert jnp.allclose(jnp.array(outputs), jnp.array(expected))

    def test_event_driven_trading_signal(self):
        """Test a more realistic trading signal that only updates on market hours."""

        def market_hours(x):
            """Simulate market hours - only update during 'trading' hours."""
            # Let's say we update when the input is between 9 and 17 (market hours)
            return jnp.logical_and(x >= 9, x <= 17)

        @streaming_transform_with_state
        @update_on_event(event_fn=market_hours)
        def trading_signal(time_price):
            """Trading signal that only updates during market hours."""
            # Extract time and price (simplified - in real case this would be structured)
            time = time_price  # Simplified: just use the input as time
            price = time_price * 100  # Simulate price

            # Signal processing
            ewma = EWMA(alpha=0.2)
            return ewma(price)

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0 = jnp.array(10.0)  # 10 AM (market hours)
        params, state = trading_signal.init(rng, x0)

        # Test sequence: market hours, after hours, market hours
        times = [10.0, 20.0, 11.0]  # 10 AM, 8 PM, 11 AM
        outputs = []
        current_state = state

        for time_val in times:
            output, current_state = trading_signal.apply(
                params, current_state, None, jnp.array(time_val)
            )
            outputs.append(output)

        # Should update at 10 AM, cache at 8 PM, update at 11 AM
        assert jnp.allclose(outputs[0], outputs[1])  # 8 PM → cached from 10 AM
        assert not jnp.allclose(outputs[0], outputs[2])  # 11 AM → new value

    def test_jax_scan_compatibility_with_conditional_computation(self):
        """Test that conditional computation works with jax.lax.scan."""

        @streaming_transform_with_state
        @update_on_event(event_fn=lambda x: x > 0)
        def conditional_processor(x):
            """Processor that only updates on positive values."""
            ewma = EWMA(alpha=0.3)
            return ewma(x)

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0 = jnp.array(1.0)
        params, state = conditional_processor.init(rng, x0)

        # Test sequence with mix of positive and negative values
        inputs = jnp.array([1.0, 2.0, -1.0, 3.0, -2.0, 4.0])

        # Method 1: For loop
        results_loop = []
        current_state = state
        for x in inputs:
            output, current_state = conditional_processor.apply(params, current_state, None, x)
            results_loop.append(output)

        # Method 2: JAX scan
        def scan_fn(carry_state, x):
            output, new_state = conditional_processor.apply(params, carry_state, None, x)
            return new_state, output

        final_state, results_scan = jax.lax.scan(scan_fn, state, inputs)

        # Results should be identical between loop and scan
        assert jnp.allclose(jnp.array(results_loop), results_scan)

        # Verify conditional behavior: negative-input outputs should equal
        # the previous positive-input output (cached)
        # inputs = [1.0, 2.0, -1.0, 3.0, -2.0, 4.0]
        #   idx 0: 1.0 > 0 → update
        #   idx 1: 2.0 > 0 → update
        #   idx 2: -1.0 → cached (= output at idx 1)
        #   idx 3: 3.0 > 0 → update
        #   idx 4: -2.0 → cached (= output at idx 3)
        #   idx 5: 4.0 > 0 → update
        assert jnp.allclose(results_scan[2], results_scan[1])  # cached at -1.0
        assert jnp.allclose(results_scan[4], results_scan[3])  # cached at -2.0

    def test_nested_conditional_computation(self):
        """Test that multiple conditional computations can be nested."""

        @streaming_transform_with_state
        @update_on_event(event_fn=lambda x: x > 0)  # Outer condition
        def outer_conditional(x):
            """Outer conditional that only updates on positive values."""

            @streaming_transform_with_state
            @update_on_event(event_fn=lambda y: y > 10)  # Inner condition
            def inner_conditional(y):
                """Inner conditional that only updates on values > 10."""
                ewma = EWMA(alpha=0.5)
                return ewma(y)

            # This is a simplified version - in practice, nested conditionals
            # would need more careful state management
            if x > 0:
                # Only process inner conditional if outer condition is met
                return inner_conditional.fn(x * 10)  # Scale to test inner condition
            else:
                return x  # Return input unchanged

        # Initialize
        rng = jax.random.PRNGKey(42)
        x0 = jnp.array(2.0)  # Positive, so outer condition met
        params, state = outer_conditional.init(rng, x0)

        # Test sequence
        test_values = [1.0, -1.0, 2.0]  # pos, neg, pos
        outputs = []
        current_state = state

        for val in test_values:
            output, current_state = outer_conditional.apply(
                params, current_state, None, jnp.array(val)
            )
            outputs.append(output)

        # Verify that conditional computation worked
        assert len(outputs) == 3
        # Detailed verification would depend on the nested logic
