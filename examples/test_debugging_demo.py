#!/usr/bin/env python3
"""Test script to validate debugging and profiling demo functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import jax
import jax.numpy as jnp
import time

from wax.flax.debug import (
    StreamingDebugger,
    StreamingProfiler,
    MemoryTracker,
    debug_streaming,
    profile_streaming,
    track_memory_usage,
    value_threshold,
    step_interval,
)
from wax.flax.core.streaming_transforms import streaming_transform_with_state
from wax.flax.modules.ewma import EWMA
from wax.flax.modules.buffer import Buffer


def test_debugging_functionality():
    """Test basic debugging functionality."""
    print("🔧 Testing Debugging Functionality")
    
    debugger = StreamingDebugger()
    debugger.add_breakpoint('high_value', value_threshold(8.0))
    
    @debug_streaming(debugger, 'test_ewma')
    @streaming_transform_with_state
    def test_ewma_fn(x):
        return EWMA(alpha=0.2)(x)
    
    rng = jax.random.PRNGKey(42)
    params, state = test_ewma_fn.init(rng, jnp.array(1.0))
    
    # Process some data including trigger value
    current_state = state
    for x in [1.0, 5.0, 10.0]:  # 10.0 should trigger breakpoint
        output, current_state = test_ewma_fn.apply(
            params, current_state, None, jnp.array(x)
        )
    
    summary = debugger.get_summary()
    triggered = sum(stats['hit_count'] for stats in summary['hooks'].values())
    
    print(f"   ✅ Processed {summary['total_steps']} steps")
    print(f"   ✅ Breakpoints triggered: {triggered}")
    
    return triggered > 0


def test_profiling_functionality():
    """Test basic profiling functionality."""
    print("🔧 Testing Profiling Functionality")
    
    profiler = StreamingProfiler()
    
    @profile_streaming(profiler, 'test_buffer')
    @streaming_transform_with_state
    def test_buffer_fn(x):
        time.sleep(0.001)  # Small delay to measure
        return Buffer(maxlen=10)(x)
    
    rng = jax.random.PRNGKey(42)
    params, state = test_buffer_fn.init(rng, jnp.array(1.0))
    
    # Process some data
    current_state = state
    for i in range(5):
        output, current_state = test_buffer_fn.apply(
            params, current_state, None, jnp.array(float(i))
        )
    
    result = profiler.finalize()
    summary = result.get_summary()
    
    print(f"   ✅ Processed {result.total_steps} steps")
    print(f"   ✅ Avg time: {summary['execution']['avg_time_ms']:.2f} ms")
    
    return result.total_steps > 0


def test_memory_tracking():
    """Test basic memory tracking functionality."""
    print("🔧 Testing Memory Tracking Functionality")
    
    tracker = MemoryTracker(snapshot_interval=1)  # Snapshot every step
    
    @track_memory_usage(tracker, 'test_memory')
    @streaming_transform_with_state
    def test_memory_fn(x):
        return Buffer(maxlen=20)(x)
    
    rng = jax.random.PRNGKey(42)
    params, state = test_memory_fn.init(rng, jnp.array(1.0))
    
    # Process some data
    current_state = state
    for i in range(3):
        output, current_state = test_memory_fn.apply(
            params, current_state, None, jnp.array(float(i))
        )
    
    summary = tracker.get_memory_summary()
    
    if "error" not in summary:
        print(f"   ✅ Processed {summary['session']['total_steps']} steps")
        print(f"   ✅ Snapshots taken: {summary['session']['total_snapshots']}")
        return True
    else:
        print(f"   ⚠️  Memory tracking had issues: {summary['error']}")
        return False


def test_combined_monitoring():
    """Test combined monitoring functionality."""
    print("🔧 Testing Combined Monitoring")
    
    debugger = StreamingDebugger()
    profiler = StreamingProfiler()
    tracker = MemoryTracker(snapshot_interval=2)
    
    @debug_streaming(debugger, 'combined')
    @profile_streaming(profiler, 'combined')
    @track_memory_usage(tracker, 'combined')
    @streaming_transform_with_state
    def combined_fn(x):
        return EWMA(alpha=0.3)(x)
    
    rng = jax.random.PRNGKey(42)
    params, state = combined_fn.init(rng, jnp.array(1.0))
    
    # Process data
    current_state = state
    for i in range(5):
        output, current_state = combined_fn.apply(
            params, current_state, None, jnp.array(float(i))
        )
    
    debug_summary = debugger.get_summary()
    profile_result = profiler.finalize()
    memory_summary = tracker.get_memory_summary()
    
    debug_steps = debug_summary['total_steps']
    profile_steps = profile_result.total_steps
    memory_steps = memory_summary.get('session', {}).get('total_steps', 0)
    
    print(f"   ✅ Debug steps: {debug_steps}")
    print(f"   ✅ Profile steps: {profile_steps}")
    print(f"   ✅ Memory steps: {memory_steps}")
    
    return all(steps > 0 for steps in [debug_steps, profile_steps, memory_steps])


def main():
    """Run all tests."""
    print("🚀 WAX-ML Debugging and Profiling Demo Test")
    print("=" * 50)
    
    tests = [
        test_debugging_functionality,
        test_profiling_functionality,
        test_memory_tracking,
        test_combined_monitoring,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All debugging and profiling functionality works!")
        return True
    else:
        print("⚠️  Some tests failed - check implementation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)