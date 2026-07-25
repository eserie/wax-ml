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
"""Tests for Flax OHLC module."""

import jax
import jax.numpy as jnp

from wax.flax.modules.ohlc import OHLCData, create_ohlc


def apply_stateful(module, variables, *args, **kwargs):
    """Helper to apply a module with proper state handling."""
    output, new_variables = module.apply(variables, *args, **kwargs, mutable=["state"])
    return output, new_variables


class TestOHLC:
    """Test cases for OHLC module."""

    def test_single_price_update(self):
        """Test OHLC with a single price."""
        # Create OHLC module
        ohlc = create_ohlc()

        # Initialize with first price
        key = jax.random.PRNGKey(42)
        price = jnp.array(100.0)
        event = jnp.array(True)

        variables = ohlc.init(key, price, event)
        result, new_variables = apply_stateful(ohlc, variables, price, event)

        # All OHLC values should be the same for first price
        assert jnp.allclose(result.OPEN, 100.0)
        assert jnp.allclose(result.HIGH, 100.0)
        assert jnp.allclose(result.LOW, 100.0)
        assert jnp.allclose(result.CLOSE, 100.0)

    def test_price_sequence_with_events(self):
        """Test OHLC with a sequence of prices and events."""
        # Create OHLC module
        ohlc = create_ohlc()

        # Initialize
        key = jax.random.PRNGKey(42)
        variables = ohlc.init(key, jnp.array(100.0), jnp.array(True))

        # Apply sequence of prices with events
        prices = [100.0, 105.0, 95.0, 102.0]
        events = [True, False, False, True]  # Reset at start and end

        results = []
        current_variables = variables
        for price, event in zip(prices, events, strict=False):
            result, current_variables = apply_stateful(
                ohlc, current_variables, jnp.array(price), jnp.array(event)
            )
            results.append(result)

        # First update (reset event)
        assert jnp.allclose(results[0].OPEN, 100.0)
        assert jnp.allclose(results[0].HIGH, 100.0)
        assert jnp.allclose(results[0].LOW, 100.0)
        assert jnp.allclose(results[0].CLOSE, 100.0)

        # Second update (no reset, price goes up)
        assert jnp.allclose(results[1].OPEN, 100.0)  # OPEN unchanged
        assert jnp.allclose(results[1].HIGH, 105.0)  # HIGH updated
        assert jnp.allclose(results[1].LOW, 100.0)  # LOW unchanged
        assert jnp.allclose(results[1].CLOSE, 105.0)  # CLOSE updated

        # Third update (no reset, price goes down)
        assert jnp.allclose(results[2].OPEN, 100.0)  # OPEN unchanged
        assert jnp.allclose(results[2].HIGH, 105.0)  # HIGH unchanged
        assert jnp.allclose(results[2].LOW, 95.0)  # LOW updated
        assert jnp.allclose(results[2].CLOSE, 95.0)  # CLOSE updated

        # Fourth update (reset event)
        assert jnp.allclose(results[3].OPEN, 102.0)  # OPEN reset
        assert jnp.allclose(results[3].HIGH, 102.0)  # HIGH reset
        assert jnp.allclose(results[3].LOW, 102.0)  # LOW reset
        assert jnp.allclose(results[3].CLOSE, 102.0)  # CLOSE reset

    def test_no_events_continuous_update(self):
        """Test OHLC with continuous updates (no reset events)."""
        # Create OHLC module
        ohlc = create_ohlc()

        # Initialize
        key = jax.random.PRNGKey(42)
        variables = ohlc.init(key, jnp.array(100.0), jnp.array(True))

        # Apply sequence without reset events
        prices = [100.0, 110.0, 90.0, 95.0, 105.0]

        current_variables = variables
        for i, price in enumerate(prices):
            event = jnp.array(i == 0)  # Only first is a reset event
            result, current_variables = apply_stateful(
                ohlc, current_variables, jnp.array(price), event
            )

        # Final result should capture the full range
        assert jnp.allclose(result.OPEN, 100.0)  # First price
        assert jnp.allclose(result.HIGH, 110.0)  # Maximum price
        assert jnp.allclose(result.LOW, 90.0)  # Minimum price
        assert jnp.allclose(result.CLOSE, 105.0)  # Last price

    def test_vector_prices(self):
        """Test OHLC with vector prices."""
        # Create OHLC module
        ohlc = create_ohlc()

        # Initialize with vector price
        key = jax.random.PRNGKey(42)
        price = jnp.array([100.0, 200.0])
        event = jnp.array(True)

        variables = ohlc.init(key, price, event)
        result, new_variables = apply_stateful(ohlc, variables, price, event)

        # All OHLC values should match input vector
        assert jnp.allclose(result.OPEN, jnp.array([100.0, 200.0]))
        assert jnp.allclose(result.HIGH, jnp.array([100.0, 200.0]))
        assert jnp.allclose(result.LOW, jnp.array([100.0, 200.0]))
        assert jnp.allclose(result.CLOSE, jnp.array([100.0, 200.0]))

        # Update with new prices
        new_price = jnp.array([105.0, 195.0])
        result2, final_variables = apply_stateful(ohlc, new_variables, new_price, jnp.array(False))

        # Check element-wise OHLC updates
        assert jnp.allclose(result2.OPEN, jnp.array([100.0, 200.0]))  # Unchanged
        assert jnp.allclose(
            result2.HIGH, jnp.array([105.0, 200.0])
        )  # First higher, second unchanged
        assert jnp.allclose(result2.LOW, jnp.array([100.0, 195.0]))  # First unchanged, second lower
        assert jnp.allclose(result2.CLOSE, jnp.array([105.0, 195.0]))  # Both updated

    def test_ohlc_data_structure(self):
        """Test OHLCData structure properties."""
        # Create sample OHLC data
        ohlc_data = OHLCData(
            OPEN=jnp.array(100.0),
            HIGH=jnp.array(110.0),
            LOW=jnp.array(95.0),
            CLOSE=jnp.array(105.0),
        )

        # Check that all fields are accessible
        assert jnp.allclose(ohlc_data.OPEN, 100.0)
        assert jnp.allclose(ohlc_data.HIGH, 110.0)
        assert jnp.allclose(ohlc_data.LOW, 95.0)
        assert jnp.allclose(ohlc_data.CLOSE, 105.0)
