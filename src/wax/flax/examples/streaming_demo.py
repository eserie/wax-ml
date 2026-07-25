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
"""Demo of the new streaming transform architecture.

This demonstrates how WAX-ML's streaming transforms provide a Haiku-like
experience while being fully compatible with JAX/Flax patterns.
"""

import jax
import jax.numpy as jnp

from wax.flax.core import streaming_transform_with_state
from wax.flax.modules import EWMA, Buffer


@streaming_transform_with_state
def streaming_signal_processor(price):
    """A streaming signal processing pipeline.

    This looks like stateful object-oriented code but compiles to
    pure JAX functions, just like Haiku.
    """
    # Natural syntax for stateful components
    price_buffer = Buffer(maxlen=10, fill_value=0.0)
    fast_ma = EWMA(alpha=0.3)
    slow_ma = EWMA(alpha=0.1)
    volatility = EWMA(alpha=0.2)

    # Store recent prices
    recent_prices = price_buffer(price)

    # Compute moving averages
    fast_signal = fast_ma(price)
    slow_signal = slow_ma(price)

    # Compute volatility as EMA of absolute deviations
    deviation = jnp.abs(price - slow_signal)
    vol = volatility(deviation)

    # Generate trading signal
    momentum = fast_signal - slow_signal
    normalized_momentum = momentum / (vol + 1e-6)
    signal = jnp.tanh(normalized_momentum)

    return {
        "price": price,
        "recent_prices": recent_prices,
        "fast_ma": fast_signal,
        "slow_ma": slow_signal,
        "volatility": vol,
        "signal": signal,
        "momentum": momentum,
    }


@streaming_transform_with_state
def portfolio_manager(prices):
    """Multi-asset portfolio management using streaming transforms."""
    # Each asset gets its own signal processor
    asset_signals = {}
    for asset_name, price in prices.items():
        # Create signal processor for this asset
        signal_proc = streaming_signal_processor
        # Note: In real implementation, we'd want separate state for each asset
        # This is simplified for demo purposes
        signals = signal_proc.fn(price)  # Access the wrapped function
        asset_signals[asset_name] = signals

    return asset_signals


def demo_streaming_architecture():
    """Demonstrate the streaming architecture in action."""
    print("🚀 WAX-ML Streaming Architecture Demo")
    print("=" * 50)

    # Initialize the streaming processor
    rng = jax.random.PRNGKey(42)
    initial_price = jnp.array(100.0)

    # This feels exactly like Haiku
    params, state = streaming_signal_processor.init(rng, initial_price)

    print("✅ Initialized streaming processor")
    print(f"   Parameters: {len(params) if params else 0} modules")
    print(f"   State collections: {list(state.keys()) if state else []}")

    # Simulate a price stream
    price_stream = [100.0, 101.5, 99.8, 102.3, 104.1, 103.2, 105.8, 104.5, 106.2, 107.1]

    print(f"\n📈 Processing price stream: {len(price_stream)} prices")

    results = []
    current_state = state

    for i, price in enumerate(price_stream):
        output, current_state = streaming_signal_processor.apply(
            params, current_state, None, jnp.array(price)
        )
        results.append(output)

        if i < 3:  # Show first few results
            print(
                f"   Step {i + 1}: Price={price:.1f}, Signal={output['signal']:.3f}, "
                f"Fast MA={output['fast_ma']:.2f}, Slow MA={output['slow_ma']:.2f}"
            )

    print(f"   ... processed {len(results)} time steps")

    # Show final state
    final_result = results[-1]
    print("\n📊 Final State:")
    print(f"   Price: {final_result['price']:.2f}")
    print(f"   Fast MA: {final_result['fast_ma']:.2f}")
    print(f"   Slow MA: {final_result['slow_ma']:.2f}")
    print(f"   Volatility: {final_result['volatility']:.3f}")
    print(f"   Signal: {final_result['signal']:.3f}")
    print(f"   Buffer shape: {final_result['recent_prices'].shape}")

    print("\n✨ Key Architectural Points Demonstrated:")
    print("   🔄 Stateful streaming computation with transparent state management")
    print("   🧩 Natural composition of multiple streaming modules")
    print("   ⚡ Pure functional JAX compatibility under the hood")
    print("   📦 Haiku-like API that feels object-oriented but is functional")


if __name__ == "__main__":
    demo_streaming_architecture()
