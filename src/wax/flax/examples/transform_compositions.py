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
"""Examples of building complex modules through transform composition.

This demonstrates how to build sophisticated streaming modules by composing
the core streaming transforms: @streaming_transform_with_state, @update_on_event,
@streaming_scan, and @streaming_optimizer.
"""

import jax
import jax.numpy as jnp
import optax

from wax.flax.core.streaming_transforms import (
    streaming_optimizer,
    streaming_scan,
    streaming_transform_with_state,
    update_on_event,
)
from wax.flax.modules.buffer import Buffer
from wax.flax.modules.ewma import EWMA

# =============================================================================
# Example 1: Technical Analysis Indicators via Transform Composition
# =============================================================================


@streaming_transform_with_state
def bollinger_bands(price, window=20, num_std=2.0):
    """Bollinger Bands indicator using streaming transforms.

    Composition:
    - Buffer for rolling window
    - EWMA for center line
    - Statistical computations for bands
    """
    # Rolling price buffer
    price_buffer = Buffer(maxlen=window, fill_value=jnp.nan)
    buffered_prices = price_buffer(price)

    # Center line (moving average)
    center_line = EWMA(alpha=2.0 / (window + 1))(price)

    # Rolling statistics
    valid_prices = buffered_prices[~jnp.isnan(buffered_prices)]
    rolling_std = jnp.std(valid_prices) if len(valid_prices) > 1 else 0.0

    # Bollinger bands
    upper_band = center_line + (num_std * rolling_std)
    lower_band = center_line - (num_std * rolling_std)

    # Band position (where price sits relative to bands)
    band_width = upper_band - lower_band
    band_position = jnp.where(
        band_width > 1e-8,
        (price - lower_band) / band_width,
        0.5,  # Middle when bands are too narrow
    )

    return {
        "price": price,
        "upper_band": upper_band,
        "lower_band": lower_band,
        "center_line": center_line,
        "band_position": band_position,
        "band_width": band_width,
    }


@streaming_transform_with_state
def macd_indicator(price, fast_period=12, slow_period=26, signal_period=9):
    """MACD indicator using multiple EWMA compositions.

    Composition:
    - Multiple EWMA transforms
    - Derived signal calculations
    """
    # Fast and slow moving averages
    fast_ema = EWMA(alpha=2.0 / (fast_period + 1))(price)
    slow_ema = EWMA(alpha=2.0 / (slow_period + 1))(price)

    # MACD line
    macd_line = fast_ema - slow_ema

    # Signal line (EMA of MACD)
    signal_line = EWMA(alpha=2.0 / (signal_period + 1))(macd_line)

    # Histogram
    histogram = macd_line - signal_line

    return {
        "price": price,
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
        "fast_ema": fast_ema,
        "slow_ema": slow_ema,
    }


# =============================================================================
# Example 2: Event-Driven Trading System via @update_on_event
# =============================================================================


@streaming_transform_with_state
@update_on_event(event_fn=lambda price, volume: volume > 1000)  # High volume filter
def volume_breakout_detector(price, volume):
    """Breakout detector that only updates on high volume.

    Composition:
    - @update_on_event for conditional computation
    - Buffer for price history
    - Volatility and momentum calculations
    """
    # Price momentum buffer
    price_buffer = Buffer(maxlen=20, fill_value=price)
    price_history = price_buffer(price)

    # Volatility calculation
    price_returns = jnp.diff(price_history, prepend=price_history[0])
    volatility = jnp.std(price_returns)

    # Momentum calculation
    momentum = price - jnp.mean(price_history)

    # Breakout signal strength
    breakout_strength = jnp.abs(momentum) / (volatility + 1e-8)

    return {
        "price": price,
        "volume": volume,
        "momentum": momentum,
        "volatility": volatility,
        "breakout_strength": breakout_strength,
        "is_breakout": breakout_strength > 2.0,
    }


@streaming_transform_with_state
def regime_aware_processor(price, volatility_threshold=1.0):
    """Processor that adapts behavior based on market regime.

    Composition:
    - Multiple processing modes
    - Conditional logic for regime detection
    """
    # Market regime detection
    price_buffer = Buffer(maxlen=50, fill_value=price)
    recent_prices = price_buffer(price)
    current_volatility = jnp.std(recent_prices)

    # Regime-specific processing
    if current_volatility > volatility_threshold:
        # High volatility regime: responsive signals
        signal = EWMA(alpha=0.3)(price)
        mode = "high_vol"
    else:
        # Low volatility regime: smooth signals
        signal = EWMA(alpha=0.05)(price)
        mode = "low_vol"

    return {"price": price, "signal": signal, "volatility": current_volatility, "regime": mode}


# =============================================================================
# Example 3: Scan-Based Sequence Processing via @streaming_scan
# =============================================================================


@streaming_scan(reset_on=lambda price: price < 0)  # Reset on negative prices
def cumulative_returns_with_reset(price):
    """Cumulative returns that reset on negative prices.

    Composition:
    - @streaming_scan for sequence processing
    - Reset logic for regime changes
    """
    # Simple return calculation (simplified)
    return_rate = jnp.log(price + 1e-8)  # Avoid log(0)
    return return_rate


@streaming_scan
def adaptive_position_sizing(signal, risk_per_trade=0.02):
    """Adaptive position sizing based on signal strength.

    Composition:
    - @streaming_scan for sequential position updates
    - Risk management logic
    """
    # Position size based on signal strength
    signal_strength = jnp.abs(signal)
    base_position = jnp.sign(signal) * jnp.minimum(signal_strength, 1.0)

    # Risk-adjusted position
    position = base_position * risk_per_trade

    return {"signal": signal, "position": position, "risk_adjusted": True}


# =============================================================================
# Example 4: Online Learning via @streaming_optimizer
# =============================================================================


def prediction_loss(prediction, target):
    """Loss function for online learning."""
    return jnp.mean((prediction - target) ** 2)


@streaming_optimizer(optax.adam(0.001), prediction_loss)
def adaptive_price_predictor(features, target):
    """Online learning price predictor.

    Composition:
    - @streaming_optimizer for online learning
    - Feature processing pipeline
    - Prediction model
    """
    # Feature processing
    processed_features = EWMA(alpha=0.1)(features)

    # Simple linear prediction model
    prediction = processed_features * 1.1  # Simplified model

    return prediction


@streaming_optimizer(optax.sgd(0.01), prediction_loss, has_aux=True)
def ensemble_predictor(price, volume, target):
    """Ensemble predictor with auxiliary outputs.

    Composition:
    - Multiple feature streams
    - Model ensemble
    - Auxiliary diagnostics
    """
    # Feature engineering
    price_signal = EWMA(alpha=0.2)(price)
    volume_signal = EWMA(alpha=0.15)(volume)

    # Ensemble prediction
    price_pred = price_signal * 1.05
    volume_pred = volume_signal * 0.001
    ensemble_pred = (price_pred + volume_pred) / 2

    # Auxiliary information
    aux_info = {
        "price_component": price_pred,
        "volume_component": volume_pred,
        "price_signal": price_signal,
        "volume_signal": volume_signal,
    }

    return ensemble_pred, aux_info


# =============================================================================
# Example 5: Multi-Transform Complex System
# =============================================================================


@streaming_transform_with_state
def complete_trading_system(price, volume):
    """Complete trading system using all transform types.

    Composition:
    - Technical indicators (inline implementation)
    - Event-driven logic (@update_on_event simulation)
    - Sequential processing (@streaming_scan simulation)
    - Online adaptation (@streaming_optimizer simulation)
    """
    # Stage 1: Technical Analysis (inline implementation)
    # Simplified Bollinger-like calculation
    price_buffer = Buffer(maxlen=20, fill_value=jnp.nan)
    recent_prices = price_buffer(price)
    center_line = EWMA(alpha=2.0 / 21)(price)

    # Simple volatility measure (JAX-compatible)
    # Use jnp.nanstd instead of filtering with boolean indexing
    volatility = jnp.nanstd(recent_prices)
    volatility = jnp.where(jnp.isnan(volatility), 0.01, volatility)

    # Simplified MACD
    fast_ema = EWMA(alpha=2.0 / 13)(price)
    slow_ema = EWMA(alpha=2.0 / 27)(price)
    macd_line = fast_ema - slow_ema

    # Store indicator results
    bollinger = {
        "center_line": center_line,
        "band_position": (price - center_line) / (volatility + 1e-8),
        "band_width": volatility,
    }

    macd = {
        "macd": macd_line,
        "histogram": macd_line,  # Simplified
    }

    # Stage 2: Event-Driven Signal Processing
    # Simulate high-volume events (simplified threshold)
    volume_threshold = 1200.0  # Fixed threshold for demo
    is_high_volume = volume > volume_threshold

    # Use JAX-compatible conditional logic
    responsive_signal = EWMA(alpha=0.3)(price)
    smooth_signal = EWMA(alpha=0.1)(price)
    momentum_signal = jax.lax.cond(is_high_volume, lambda: responsive_signal, lambda: smooth_signal)

    # Stage 3: Signal Combination and Risk Management
    # Combine multiple signals
    technical_signal = jnp.tanh(bollinger["band_position"] - 0.5)  # -1 to 1
    momentum_component = jnp.tanh(momentum_signal / price - 1)  # Momentum relative to price
    macd_component = jnp.tanh(macd["histogram"] * 10)  # MACD histogram signal

    # Ensemble signal
    raw_signal = (technical_signal + momentum_component + macd_component) / 3

    # Risk adjustment based on volatility
    volatility = bollinger["band_width"] / bollinger["center_line"]
    risk_adjusted_signal = raw_signal / (1 + volatility)

    # Position sizing
    max_position = 0.1  # Maximum 10% position
    final_position = jnp.clip(risk_adjusted_signal * max_position, -max_position, max_position)

    return {
        "price": price,
        "volume": volume,
        "bollinger": bollinger,
        "macd": macd,
        "technical_signal": technical_signal,
        "momentum_signal": momentum_component,
        "macd_signal": macd_component,
        "raw_signal": raw_signal,
        "risk_adjusted_signal": risk_adjusted_signal,
        "final_position": final_position,
        "volatility": volatility,
        "is_high_volume": is_high_volume,
    }


# =============================================================================
# Demo and Testing Functions
# =============================================================================


def demo_transform_compositions():
    """Demonstrate the composed transforms in action."""
    print("🔧 Transform Composition Demo")
    print("=" * 50)

    # Setup
    rng = jax.random.PRNGKey(42)

    # Generate synthetic market data
    n_steps = 50
    price_key, volume_key = jax.random.split(rng)

    base_price = 100.0
    price_moves = jax.random.normal(price_key, (n_steps,)) * 0.02
    prices = base_price * jnp.exp(jnp.cumsum(price_moves))

    base_volume = 1000.0
    volume_noise = jax.random.normal(volume_key, (n_steps,)) * 200
    volumes = jnp.maximum(base_volume + volume_noise, 100)

    print(f"📊 Generated {n_steps} price/volume data points")
    print(f"   Price range: {prices.min():.2f} - {prices.max():.2f}")
    print(f"   Volume range: {volumes.min():.0f} - {volumes.max():.0f}")

    # Test 1: Technical Indicators
    print("\n📈 Testing Technical Indicators")

    # Bollinger Bands
    bollinger_params, bollinger_state = bollinger_bands.init(rng, prices[0])

    bollinger_outputs = []
    current_state = bollinger_state
    for price in prices[:10]:  # Test first 10 prices
        output, current_state = bollinger_bands.apply(bollinger_params, current_state, None, price)
        bollinger_outputs.append(output)

    final_bollinger = bollinger_outputs[-1]
    print(f"   Final Bollinger Band Position: {final_bollinger['band_position']:.3f}")
    print(f"   Band Width: {final_bollinger['band_width']:.3f}")

    # MACD
    macd_params, macd_state = macd_indicator.init(rng, prices[0])

    macd_outputs = []
    current_state = macd_state
    for price in prices[:10]:
        output, current_state = macd_indicator.apply(macd_params, current_state, None, price)
        macd_outputs.append(output)

    final_macd = macd_outputs[-1]
    print(f"   Final MACD: {final_macd['macd']:.3f}")
    print(f"   Signal: {final_macd['signal']:.3f}")
    print(f"   Histogram: {final_macd['histogram']:.3f}")

    # Test 2: Complete Trading System
    print("\n🎯 Testing Complete Trading System")

    system_params, system_state = complete_trading_system.init(rng, prices[0], volumes[0])

    system_outputs = []
    current_state = system_state
    for price, volume in zip(prices[:20], volumes[:20], strict=False):
        output, current_state = complete_trading_system.apply(
            system_params, current_state, None, price, volume
        )
        system_outputs.append(output)

    final_system = system_outputs[-1]
    print(f"   Final Position: {final_system['final_position']:.4f}")
    print(f"   Risk Adjusted Signal: {final_system['risk_adjusted_signal']:.4f}")
    print(f"   Volatility: {final_system['volatility']:.4f}")
    print(f"   High Volume Event: {final_system['is_high_volume']}")

    # Test 3: Scan-Based Processing
    print("\n🔄 Testing Scan-Based Processing")

    # Test cumulative returns with reset
    test_prices = jnp.array([100.0, 105.0, 110.0, -1.0, 102.0, 107.0])  # Negative triggers reset

    returns_outputs, _ = cumulative_returns_with_reset.scan_apply(test_prices)
    print(f"   Cumulative returns with resets: {returns_outputs}")

    # Test 4: Online Learning Demo (conceptual)
    print("\n🧠 Testing Online Learning (Conceptual)")

    # Simple features and targets
    features = prices[:10] / 100.0  # Normalized features
    targets = prices[1:11] / 100.0  # Next-step targets

    predictor_params, predictor_state = adaptive_price_predictor.init(rng, features[0], targets[0])

    predictor_outputs = []
    current_state = predictor_state
    for feat, targ in zip(features, targets, strict=False):
        (loss, pred), current_state = adaptive_price_predictor.apply(
            predictor_params, current_state, None, feat, targ
        )
        predictor_outputs.append({"loss": loss, "prediction": pred})

    final_pred = predictor_outputs[-1]
    print(f"   Final Prediction Loss: {final_pred['loss']:.6f}")
    print(f"   Final Prediction: {final_pred['prediction']:.4f}")

    print("\n✨ Transform Composition Demo Complete")
    print("   🏗️  Demonstrated hierarchical composition")
    print("   🎛️  Showed event-driven processing")
    print("   🔄 Illustrated scan-based sequences")
    print("   🧠 Tested online learning integration")


if __name__ == "__main__":
    demo_transform_compositions()
