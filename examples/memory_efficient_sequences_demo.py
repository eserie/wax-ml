#!/usr/bin/env python3
"""Demonstration of Memory-Efficient Long Sequences in WAX-ML.

This demo showcases the hierarchical buffering and compression strategies
for handling arbitrarily long sequences with bounded memory usage.

Key features demonstrated:
1. CompressedBuffer with different compression strategies
2. HierarchicalBuffer for multi-resolution memory management  
3. Memory usage tracking and compression ratios
4. Performance comparison vs uncompressed approaches
5. Real-world application scenarios
"""

import jax
import jax.numpy as jnp
import time

from wax.flax.modules.compressed_buffer import (
    CompressedBuffer,
    HierarchicalBuffer,
    streaming_compressed_memory,
    streaming_hierarchical_memory,
)
from wax.flax.core.streaming_transforms import streaming_transform_with_state


def demo_compression_strategies():
    """Demonstrate different compression strategies."""
    print("🗜️  COMPRESSION STRATEGIES DEMO")
    print("=" * 50)
    
    # Test sequence: long-term trend with noise
    sequence_length = 1000
    t = jnp.linspace(0, 10, sequence_length)
    trend = jnp.sin(0.5 * t) + 0.1 * jnp.sin(5 * t)  # Multi-scale signal
    noise = jax.random.normal(jax.random.PRNGKey(42), (sequence_length,)) * 0.1
    signal = trend + noise
    
    print(f"📊 Processing signal with {sequence_length} time steps")
    
    # Test different compression strategies
    strategies = {
        "none": {"compression": "none"},
        "ewma": {"compression": "ewma", "compression_params": {"alpha": 0.02}},
        "quantile": {"compression": "quantile", "compression_params": {"percentiles": [0.1, 0.5, 0.9]}},
        "downsample": {"compression": "downsample", "compression_params": {"factor": 4}},
    }
    
    results = {}
    
    for name, config in strategies.items():
        print(f"\n🔧 Testing {name.upper()} compression...")
        
        # Create buffer
        buffer = CompressedBuffer(maxlen=500, **config)
        
        # Initialize
        rng = jax.random.PRNGKey(42)
        variables = buffer.init(rng, signal[0])
        
        # Measure processing time
        start_time = time.time()
        
        # Process entire sequence
        current_vars = variables
        final_output = None
        
        for i, x in enumerate(signal):
            output, new_vars = buffer.apply(current_vars, x, mutable=['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}
            final_output = output
            
            # Print progress every 200 steps
            if (i + 1) % 200 == 0:
                print(f"   Processed {i + 1}/{sequence_length} steps")
        
        processing_time = time.time() - start_time
        
        # Get memory usage
        memory_usage = buffer.get_memory_usage()
        
        # Store results
        results[name] = {
            "final_output": final_output,
            "memory_usage": memory_usage,
            "processing_time": processing_time
        }
        
        print(f"   ✅ Completed in {processing_time:.3f}s")
        print(f"   📏 Memory usage: {memory_usage['total']:,} bytes")
        if name != "none":
            uncompressed_size = 500 * 8  # 500 floats * 8 bytes
            compression_ratio = uncompressed_size / memory_usage['total']
            print(f"   📈 Compression ratio: {compression_ratio:.1f}x")
    
    # Compare results
    print(f"\n📊 COMPRESSION COMPARISON")
    print("-" * 30)
    
    baseline_memory = results["none"]["memory_usage"]["total"]
    
    for name, result in results.items():
        memory = result["memory_usage"]["total"]
        time_taken = result["processing_time"]
        memory_reduction = (baseline_memory - memory) / baseline_memory * 100
        
        print(f"{name:>12}: {memory:>8,} bytes ({memory_reduction:>5.1f}% reduction), {time_taken:.3f}s")


def demo_hierarchical_buffer():
    """Demonstrate hierarchical buffer for ultra-long sequences."""
    print("\n\n🏗️  HIERARCHICAL BUFFER DEMO")
    print("=" * 50)
    
    # Create hierarchical buffer with multiple levels
    hierarchical_buffer = HierarchicalBuffer(
        recent_maxlen=50,      # High-resolution recent history
        medium_maxlen=500,     # Medium-term with light compression
        long_maxlen=5000,      # Long-term with heavy compression
        medium_compression="ewma",
        long_compression="quantile"
    )
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    variables = hierarchical_buffer.init(rng, jnp.array(1.0))
    
    print("📊 Processing ultra-long sequence (10,000 steps)")
    print("   🔍 Recent: 50 steps (full resolution)")
    print("   🔧 Medium: 500 steps (EWMA compression)")
    print("   📦 Long: 5,000 steps (quantile compression)")
    
    # Generate very long sequence
    sequence_length = 10000
    
    # Multi-scale signal: daily + weekly + monthly patterns
    t = jnp.arange(sequence_length)
    daily = jnp.sin(2 * jnp.pi * t / 24)  # Daily cycle
    weekly = 0.5 * jnp.sin(2 * jnp.pi * t / (24 * 7))  # Weekly cycle
    monthly = 0.3 * jnp.sin(2 * jnp.pi * t / (24 * 30))  # Monthly cycle
    trend = 0.01 * t  # Long-term trend
    noise = jax.random.normal(jax.random.PRNGKey(123), (sequence_length,)) * 0.1
    
    signal = daily + weekly + monthly + trend + noise + 100.0  # Offset to 100
    
    # Process sequence
    start_time = time.time()
    current_vars = variables
    final_output = None
    
    for i, x in enumerate(signal):
        output, new_vars = hierarchical_buffer.apply(current_vars, x, mutable=['state'])
        current_vars = {**current_vars, 'state': new_vars['state']}
        final_output = output
        
        # Print progress
        if (i + 1) % 2000 == 0:
            print(f"   Processed {i + 1:,}/{sequence_length:,} steps")
    
    processing_time = time.time() - start_time
    
    # Analyze final state
    print(f"\n✅ Processing completed in {processing_time:.3f}s")
    
    # Memory analysis
    memory_usage = hierarchical_buffer.get_total_memory_usage()
    print(f"\n📊 MEMORY ANALYSIS")
    print("-" * 20)
    print(f"Recent buffer:  {memory_usage['recent']:>8,} bytes")
    print(f"Medium buffer:  {memory_usage['medium']:>8,} bytes") 
    print(f"Long buffer:    {memory_usage['long']:>8,} bytes")
    print(f"Total:          {memory_usage['total']:>8,} bytes")
    print(f"Compression:    {memory_usage['compression_ratio']:>8.1f}x")
    
    # Show final hierarchical state
    print(f"\n🎯 FINAL HIERARCHICAL STATE")
    print("-" * 28)
    print(f"Recent (last 5):   {final_output['recent'][-5:]}")
    print(f"Medium compress:   {final_output['medium']:.3f}")
    print(f"Long compress:     {final_output['long']}")
    print(f"Input value:       {final_output['input']:.3f}")
    
    # Compare with naive approach
    naive_memory = sequence_length * 8  # Store everything
    actual_memory = memory_usage['total']
    memory_savings = (naive_memory - actual_memory) / naive_memory * 100
    
    print(f"\n💾 MEMORY EFFICIENCY")
    print("-" * 18)
    print(f"Naive approach:    {naive_memory:>8,} bytes (store all)")
    print(f"Hierarchical:      {actual_memory:>8,} bytes")
    print(f"Memory savings:    {memory_savings:>8.1f}%")


def demo_streaming_decorators():
    """Demonstrate streaming decorators for memory-efficient processing."""
    print("\n\n🎨 STREAMING DECORATORS DEMO")
    print("=" * 50)
    
    # Example 1: Compressed memory for long-term pattern recognition
    @streaming_compressed_memory(maxlen=1000, compression="ewma")
    def pattern_recognizer(compressed_history, current_value):
        """Recognize patterns using compressed long-term memory."""
        # Simple pattern: detect if current value is anomalous
        # compared to long-term average
        long_term_avg = compressed_history
        deviation = jnp.abs(current_value - long_term_avg)
        
        # Adaptive threshold based on recent history scale
        threshold = jnp.maximum(0.5, 0.1 * jnp.abs(long_term_avg))
        is_anomaly = deviation > threshold
        
        return {
            "value": current_value,
            "long_term_avg": long_term_avg,
            "deviation": deviation,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
            "anomaly_score": deviation / threshold
        }
    
    print("🔍 Testing pattern recognition with compressed memory...")
    
    # Generate test signal with anomalies
    normal_signal = jax.random.normal(jax.random.PRNGKey(42), (200,)) * 0.5 + 10.0
    anomaly_indices = jnp.array([50, 120, 180])
    anomaly_signal = normal_signal.at[anomaly_indices].add(5.0)  # Add large spikes
    
    # Initialize pattern recognizer
    rng = jax.random.PRNGKey(42)
    params, state = pattern_recognizer.init(rng, anomaly_signal[0])
    
    # Process signal and detect anomalies
    current_state = state
    anomaly_detections = []
    
    for i, value in enumerate(anomaly_signal):
        output, current_state = pattern_recognizer.apply(
            params, current_state, None, value
        )
        anomaly_detections.append(output)
    
    # Analyze results
    detected_anomalies = [i for i, det in enumerate(anomaly_detections) 
                         if det["is_anomaly"]]
    
    print(f"   📊 Signal length: {len(anomaly_signal)}")
    print(f"   🎯 True anomalies at indices: {anomaly_indices}")
    print(f"   🔍 Detected anomalies at indices: {detected_anomalies}")
    
    # Check detection accuracy
    true_positives = len(set(anomaly_indices) & set(detected_anomalies))
    precision = true_positives / len(detected_anomalies) if detected_anomalies else 0
    recall = true_positives / len(anomaly_indices)
    
    print(f"   ✅ Detection precision: {precision:.2f}")
    print(f"   ✅ Detection recall: {recall:.2f}")
    
    # Example 2: Hierarchical memory for multi-scale analysis
    @streaming_hierarchical_memory(
        recent_maxlen=10,
        medium_maxlen=100, 
        long_maxlen=1000
    )
    def multi_scale_analyzer(memory_levels, current_price):
        """Analyze market data at multiple time scales."""
        recent_prices = memory_levels["recent"]
        medium_trend = memory_levels["medium"]  # Compressed medium-term
        long_baseline = memory_levels["long"]   # Compressed long-term
        
        # Multi-scale analysis
        short_term_momentum = jnp.mean(recent_prices[-5:]) - jnp.mean(recent_prices[:5])
        medium_term_signal = current_price - medium_trend
        long_term_position = current_price - jnp.mean(long_baseline) if hasattr(long_baseline, '__len__') else current_price - long_baseline
        
        # Combine signals
        composite_signal = (
            0.5 * jnp.tanh(short_term_momentum) +
            0.3 * jnp.tanh(medium_term_signal / 10.0) +
            0.2 * jnp.tanh(long_term_position / 100.0)
        )
        
        return {
            "price": current_price,
            "short_momentum": short_term_momentum,
            "medium_signal": medium_term_signal,
            "long_position": long_term_position,
            "composite_signal": composite_signal,
            "recent_volatility": jnp.std(recent_prices)
        }
    
    print(f"\n📈 Testing multi-scale market analysis...")
    
    # Generate realistic price data
    price_length = 500
    t = jnp.arange(price_length)
    base_price = 100.0
    trend = 0.02 * t
    cycles = 5 * jnp.sin(2 * jnp.pi * t / 50) + 2 * jnp.sin(2 * jnp.pi * t / 20)
    noise = jax.random.normal(jax.random.PRNGKey(456), (price_length,)) * 1.0
    prices = base_price + trend + cycles + noise
    
    # Initialize analyzer
    rng = jax.random.PRNGKey(42)
    params, state = multi_scale_analyzer.init(rng, prices[0])
    
    # Process price data
    current_state = state
    analysis_results = []
    
    for price in prices:
        output, current_state = multi_scale_analyzer.apply(
            params, current_state, None, price
        )
        analysis_results.append(output)
    
    # Analyze final results
    final_result = analysis_results[-1]
    
    print(f"   📊 Processed {len(prices)} price points")
    print(f"   📈 Final price: {final_result['price']:.2f}")
    print(f"   ⚡ Short momentum: {final_result['short_momentum']:.3f}")
    print(f"   📊 Medium signal: {final_result['medium_signal']:.3f}")
    print(f"   📍 Long position: {final_result['long_position']:.3f}")
    print(f"   🎯 Composite signal: {final_result['composite_signal']:.3f}")
    print(f"   📊 Recent volatility: {final_result['recent_volatility']:.3f}")


def demo_performance_comparison():
    """Compare performance of compressed vs uncompressed approaches."""
    print("\n\n⚡ PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Test parameters
    sequence_length = 5000
    buffer_size = 1000
    
    print(f"📊 Comparing performance on {sequence_length:,} element sequence")
    print(f"   Buffer size: {buffer_size:,} elements")
    
    # Generate test data
    rng = jax.random.PRNGKey(42)
    test_data = jax.random.normal(rng, (sequence_length,))
    
    configs = {
        "Uncompressed": {"compression": "none"},
        "EWMA (α=0.01)": {"compression": "ewma", "compression_params": {"alpha": 0.01}},
        "Quantile": {"compression": "quantile"},
        "Downsample (4x)": {"compression": "downsample", "compression_params": {"factor": 4}},
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n🔧 Testing {name}...")
        
        # Create buffer
        buffer = CompressedBuffer(maxlen=buffer_size, **config)
        
        # Initialize
        variables = buffer.init(rng, test_data[0])
        
        # Measure performance
        start_time = time.time()
        
        # JIT compile first iteration
        jitted_apply = jax.jit(buffer.apply, static_argnums=3)
        output, new_vars = jitted_apply(variables, test_data[0], ['state'])
        
        compilation_time = time.time() - start_time
        
        # Measure actual processing time
        start_time = time.time()
        current_vars = variables
        
        for x in test_data:
            output, new_vars = jitted_apply(current_vars, x, ['state'])
            current_vars = {**current_vars, 'state': new_vars['state']}
        
        processing_time = time.time() - start_time
        
        # Memory usage
        memory_usage = buffer.get_memory_usage()
        
        results[name] = {
            "compilation_time": compilation_time,
            "processing_time": processing_time,
            "memory_usage": memory_usage["total"],
            "throughput": sequence_length / processing_time
        }
        
        print(f"   ⏱️  Compilation: {compilation_time:.3f}s")
        print(f"   🚀 Processing: {processing_time:.3f}s")
        print(f"   📏 Memory: {memory_usage['total']:,} bytes")
        print(f"   📈 Throughput: {results[name]['throughput']:,.0f} elements/s")
    
    # Summary comparison
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("-" * 30)
    
    baseline = results["Uncompressed"]
    
    for name, result in results.items():
        if name == "Uncompressed":
            continue
            
        speedup = baseline["processing_time"] / result["processing_time"]
        memory_ratio = baseline["memory_usage"] / result["memory_usage"]
        
        print(f"{name:>15}: {speedup:>5.2f}x speed, {memory_ratio:>5.1f}x memory efficiency")


def main():
    """Run all demonstrations."""
    print("🚀 WAX-ML Memory-Efficient Long Sequences Demo")
    print("=" * 60)
    print("Demonstrating hierarchical buffering and compression for")
    print("handling arbitrarily long sequences with bounded memory.")
    print()
    
    # Run all demos
    demo_compression_strategies()
    demo_hierarchical_buffer()
    demo_streaming_decorators()
    demo_performance_comparison()
    
    print("\n\n🎯 DEMO SUMMARY")
    print("=" * 20)
    print("✅ Compression strategies: EWMA, quantile, downsampling")
    print("✅ Hierarchical memory: Multi-resolution buffering")
    print("✅ Streaming decorators: Easy integration patterns")
    print("✅ Performance: JAX-optimized implementations")
    print("✅ Memory efficiency: Significant reduction in memory usage")
    print()
    print("🔮 Use cases:")
    print("   • Financial time series with long history")
    print("   • Sensor data streams with bounded memory")
    print("   • Real-time anomaly detection")
    print("   • Multi-scale pattern recognition")
    print("   • Online learning with long context")
    print()
    print("📚 Built on research in streaming algorithms,")
    print("   data structures, and time series compression.")


if __name__ == "__main__":
    main()