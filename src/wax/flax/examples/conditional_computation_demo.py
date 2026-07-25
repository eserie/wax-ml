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
"""Demonstration of event-driven conditional computation with @update_on_event.

This shows how to create streaming functions that only update their state
when specific events occur, preserving computational efficiency.
"""

import jax
import jax.numpy as jnp

from wax.flax.core import streaming_transform_with_state, update_on_event
from wax.flax.modules import EWMA, Buffer


@streaming_transform_with_state
@update_on_event(event_fn=lambda price: price > 100)  # Only update when price > 100
def high_price_signal(price):
    """Trading signal that only updates when price exceeds threshold.

    This demonstrates conditional computation - the EWMA state is only
    updated when the price exceeds 100, otherwise cached output is returned.
    """
    # These modules only execute when price > 100
    ewma = EWMA(alpha=0.2)
    buffer = Buffer(maxlen=5, fill_value=100.0)

    # Signal processing
    smoothed = ewma(price)
    recent = buffer(price)
    volatility = jnp.std(recent)

    # Generate signal
    signal_strength = (price - smoothed) / (volatility + 1e-6)
    signal = jnp.tanh(signal_strength)

    return {
        "price": price,
        "smoothed": smoothed,
        "signal": signal,
        "volatility": volatility,
        "updated": True,  # Indicates this was computed, not cached
    }


@streaming_transform_with_state
@update_on_event(event_fn=lambda data: data["volume"] > 1000)  # Update on high volume
def volume_based_signal(data):
    """Signal that only updates during high volume periods.

    Demonstrates using structured data with event conditions.
    """
    price = data["price"]
    volume = data["volume"]

    # Only process during high volume
    price_ewma = EWMA(alpha=0.1)
    volume_ewma = EWMA(alpha=0.3)

    smoothed_price = price_ewma(price)
    smoothed_volume = volume_ewma(volume)

    # Volume-weighted signal
    volume_ratio = volume / (smoothed_volume + 1)
    price_momentum = (price - smoothed_price) / (smoothed_price + 1e-6)

    signal = price_momentum * jnp.log1p(volume_ratio)

    return {
        "price": price,
        "volume": volume,
        "signal": signal,
        "volume_ratio": volume_ratio,
        "high_volume_event": True,
    }


@streaming_transform_with_state
def always_updating_baseline(price):
    """Baseline signal that always updates for comparison."""
    ewma = EWMA(alpha=0.2)
    buffer = Buffer(maxlen=5, fill_value=100.0)

    smoothed = ewma(price)
    recent = buffer(price)
    volatility = jnp.std(recent)

    signal_strength = (price - smoothed) / (volatility + 1e-6)
    signal = jnp.tanh(signal_strength)

    return {
        "price": price,
        "smoothed": smoothed,
        "signal": signal,
        "volatility": volatility,
        "always_updated": True,
    }


def demo_conditional_vs_always_updating():
    """Compare conditional computation vs always updating."""
    print("🎯 Conditional Computation Demo")
    print("=" * 50)

    # Setup
    rng = jax.random.PRNGKey(42)

    # Initialize both processors
    initial_price = jnp.array(105.0)  # Above threshold

    params_cond, state_cond = high_price_signal.init(rng, initial_price)
    params_always, state_always = always_updating_baseline.init(rng, initial_price)

    print(f"✅ Initialized processors with price ${initial_price:.2f}")

    # Test price sequence: high, low, low, high, low, high
    price_sequence = [105.0, 95.0, 90.0, 110.0, 85.0, 120.0]

    print("\n📈 Processing price sequence:")
    print(f"   Prices: {[f'${p:.0f}' for p in price_sequence]}")
    print("   Threshold: $100 (conditional updates only above this)")

    results_conditional = []
    results_always = []

    state_c = state_cond
    state_a = state_always

    for i, price in enumerate(price_sequence):
        price_val = jnp.array(price)

        # Conditional processor
        output_c, state_c = high_price_signal.apply(params_cond, state_c, None, price_val)
        results_conditional.append(output_c)

        # Always updating processor
        output_a, state_a = always_updating_baseline.apply(params_always, state_a, None, price_val)
        results_always.append(output_a)

        # Show what happened
        will_update = price > 100
        print(f"   Step {i + 1}: ${price:.0f} -> {'UPDATE' if will_update else 'CACHE'}")

    print("\n🔍 Results Comparison:")
    print(f"{'Step':<4} {'Price':<6} {'Conditional':<12} {'Always':<12} {'Difference':<12}")
    print("-" * 60)

    for i, (price, cond, always) in enumerate(
        zip(price_sequence, results_conditional, results_always, strict=False)
    ):
        cond_signal = float(cond["signal"])
        always_signal = float(always["signal"])
        diff = abs(cond_signal - always_signal)

        print(
            f"{i + 1:<4} ${price:<5.0f} {cond_signal:<12.6f} {always_signal:<12.6f} {diff:<12.6f}"
        )

    print("\n💡 Key Observations:")
    print("   🎯 Conditional computation saves work when price < $100")
    print("   📊 Signals differ because of selective state updates")
    print("   ⚡ Conditional version has lower computational cost")
    print("   🎛️  Event-driven updates preserve efficiency")


def demo_volume_based_conditional():
    """Demonstrate volume-based conditional computation."""
    print("\n🔊 Volume-Based Conditional Demo")
    print("=" * 40)

    # Setup
    rng = jax.random.PRNGKey(42)

    # Initialize volume-based processor
    initial_data = {"price": jnp.array(100.0), "volume": jnp.array(1500.0)}  # High volume
    params, state = volume_based_signal.init(rng, initial_data)

    print("✅ Initialized volume-based processor")
    print("   Volume threshold: 1000 (only updates above this)")

    # Test sequence with varying volume
    market_data = [
        {"price": 100.0, "volume": 1500.0},  # High volume - UPDATE
        {"price": 101.0, "volume": 800.0},  # Low volume - CACHE
        {"price": 102.0, "volume": 500.0},  # Low volume - CACHE
        {"price": 103.0, "volume": 2000.0},  # High volume - UPDATE
        {"price": 104.0, "volume": 300.0},  # Low volume - CACHE
        {"price": 105.0, "volume": 1200.0},  # High volume - UPDATE
    ]

    print("\n📊 Processing market data:")

    results = []
    current_state = state

    for i, data in enumerate(market_data):
        data_jax = {"price": jnp.array(data["price"]), "volume": jnp.array(data["volume"])}

        output, current_state = volume_based_signal.apply(params, current_state, None, data_jax)
        results.append(output)

        will_update = data["volume"] > 1000
        status = "UPDATE" if will_update else "CACHE"

        print(
            f"   Step {i + 1}: Price=${data['price']:.0f}, Volume={data['volume']:.0f} -> {status}"
        )

    print("\n📈 Signal Evolution:")
    for i, (data, result) in enumerate(zip(market_data, results, strict=False)):
        signal = float(result["signal"])
        vol_ratio = float(result["volume_ratio"])

        print(f"   Step {i + 1}: Signal={signal:.4f}, Vol Ratio={vol_ratio:.2f}")

    print("\n💡 Key Insights:")
    print("   🔊 Signal only updates during high volume periods")
    print("   📊 Low volume periods preserve previous signal")
    print("   ⚡ Computational savings during quiet market periods")
    print("   🎯 Focus computation on significant market events")


def demo_jax_scan_with_conditional():
    """Demonstrate that conditional computation works with JAX scan."""
    print("\n⚡ JAX Scan + Conditional Computation")
    print("=" * 45)

    # Setup
    rng = jax.random.PRNGKey(42)
    initial_price = jnp.array(105.0)
    params, state = high_price_signal.init(rng, initial_price)

    # Test sequence
    prices = jnp.array([105.0, 95.0, 90.0, 110.0, 85.0, 120.0, 95.0, 115.0])
    print(f"✅ Testing with {len(prices)} prices")
    print(f"   Prices: {[f'${p:.0f}' for p in prices]}")

    # Method 1: For loop
    print("\n🔁 Method 1: For loop processing")
    results_loop = []
    loop_state = state

    for price in prices:
        output, loop_state = high_price_signal.apply(params, loop_state, None, price)
        results_loop.append(float(output["signal"]))

    print(f"   Signals: {[f'{s:.4f}' for s in results_loop]}")

    # Method 2: JAX scan
    print("\n⚡ Method 2: JAX scan processing")

    def scan_fn(carry_state, price):
        output, new_state = high_price_signal.apply(params, carry_state, None, price)
        return new_state, output["signal"]

    final_state, results_scan = jax.lax.scan(scan_fn, state, prices)
    results_scan_list = [float(s) for s in results_scan]

    print(f"   Signals: {[f'{s:.4f}' for s in results_scan_list]}")

    # Verify identical results
    match = jnp.allclose(jnp.array(results_loop), results_scan)
    print(f"\n✅ Results match: {match}")

    if match:
        print("🎉 SUCCESS: Conditional computation is fully JAX-scan compatible!")
        print("   📊 Same results from both execution methods")
        print("   ⚡ Can choose performance vs debuggability as needed")
        print("   🎯 Event-driven computation scales to large datasets")
    else:
        print("❌ ERROR: Results differ between methods")


def main():
    """Run all conditional computation demonstrations."""
    print("🎭 WAX-ML Conditional Computation Demonstrations")
    print("=" * 60)

    demo_conditional_vs_always_updating()
    demo_volume_based_conditional()
    demo_jax_scan_with_conditional()

    print("\n🏆 Summary of Conditional Computation Benefits:")
    print("   ⚡ Computational Efficiency: Skip processing when events don't occur")
    print("   🎯 Event-Driven Logic: Natural way to express conditional updates")
    print("   📊 State Preservation: Maintain state during inactive periods")
    print("   🔄 JAX Compatible: Works with both loops and scan operations")
    print("   🧩 Composable: Can nest and combine conditional computations")
    print("   📈 Real-World Applicable: Perfect for trading, sensors, time series")


if __name__ == "__main__":
    main()
