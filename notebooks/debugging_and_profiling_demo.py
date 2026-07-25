# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Debugging and Profiling Streaming Computations in WAX-ML
#
# This notebook demonstrates WAX-ML's comprehensive debugging and profiling capabilities for streaming computations. The tools provide:
#
# - **Streaming Debugger**: Real-time state inspection, conditional breakpoints, and data flow tracing
# - **Performance Profiler**: Execution time analysis, bottleneck identification, and optimization recommendations
# - **Memory Tracker**: Memory usage monitoring, leak detection, and allocation analysis
#
# These tools are essential for developing, optimizing, and maintaining production streaming AI systems.

# %% [markdown]
# ## Setup and Imports

# %%
import jax
import jax.numpy as jnp
import time
import numpy as np

# WAX-ML debugging and profiling tools
from wax.flax.debug import (
    StreamingDebugger,
    DebugHook, 
    debug_streaming,
    StreamingProfiler,
    profile_streaming,
    ProfileResult,
    MemoryTracker,
    track_memory_usage,
    value_threshold,
    step_interval,
    state_change_detector,
    create_performance_report,
)

# WAX-ML streaming modules
from wax.flax.core.streaming_transforms import streaming_transform_with_state
from wax.flax.modules.ewma import EWMA
from wax.flax.modules.buffer import Buffer
from wax.flax.modules.compressed_buffer import CompressedBuffer, HierarchicalBuffer

# Set random seed for reproducibility
rng = jax.random.PRNGKey(42)

print("🔧 WAX-ML Debugging and Profiling Tools Demo")
print("=" * 55)

# %% [markdown]
# ## 1. Streaming Debugger: State Inspection and Breakpoints
#
# The StreamingDebugger provides real-time monitoring of streaming computations with conditional breakpoints, state inspection, and data flow tracing.

# %%
def demonstrate_streaming_debugger():
    """Demonstrate streaming debugger capabilities."""
    print("🐛 Streaming Debugger Demonstration")
    print("-" * 40)
    
    # Create debugger with comprehensive monitoring
    debugger = StreamingDebugger(
        enable_state_tracking=True,
        enable_performance_tracking=True,
        max_history=500
    )
    
    # Add various debug hooks
    print("🔧 Setting up debug hooks...")
    
    # 1. Monitor high input values
    debugger.add_breakpoint(
        name="high_value_alert",
        condition=value_threshold(8.0)
    )
    
    # 2. Regular state monitoring
    debugger.add_state_change_hook(
        name="regular_monitor",
        condition=step_interval(10)
    )
    
    # 3. Performance monitoring for slow operations
    debugger.add_performance_monitor(
        name="slow_operations",
        threshold_ms=5.0
    )
    
    # 4. State change detection
    debugger.add_hook(DebugHook(
        name="ewma_change_detector",
        condition=state_change_detector("ewma_state", tolerance=0.1),
        action="log"
    ))
    
    print(f"   Added {len(debugger.hooks)} debug hooks")
    
    # Create streaming function with debugging
    @debug_streaming(debugger, "ewma_processor")
    @streaming_transform_with_state
    def debug_ewma_processor(x):
        """EWMA processor with debugging."""
        return EWMA(alpha=0.2)(x)
    
    print("\n📊 Processing streaming data with debugging...")
    
    # Generate test data with some extreme values to trigger breakpoints
    test_sequence = []
    for i in range(30):
        if i in [8, 15, 22]:  # Add some high values
            value = 10.0 + jax.random.normal(rng, ()) * 2.0
        else:
            value = 5.0 + jax.random.normal(rng, ()) * 1.0
        test_sequence.append(float(value))
    
    # Initialize and run with debugging
    params, state = debug_ewma_processor.init(rng, test_sequence[0])
    current_state = state
    outputs = []
    
    for i, x in enumerate(test_sequence):
        # Simulate some processing delay for performance monitoring
        if i % 7 == 0:
            time.sleep(0.001)  # Intentional slow operation
        
        output, current_state = debug_ewma_processor.apply(
            params, current_state, None, x
        )
        outputs.append(output)
    
    # Analyze debugging results
    print(f"\n🎯 Debugging Session Results:")
    summary = debugger.get_summary()
    
    print(f"   Total steps processed: {summary['total_steps']}")
    print(f"   Session duration: {summary['session_duration']:.2f} seconds")
    
    print("\n📋 Hook Statistics:")
    for hook_name, stats in summary['hooks'].items():
        print(f"   {hook_name}:")
        print(f"     Hits: {stats['hit_count']}")
        print(f"     Events: {stats['events']}")
        print(f"     Enabled: {stats['enabled']}")
    
    if summary['performance']['total_steps'] > 0:
        print(f"\n⚡ Performance Summary:")
        perf = summary['performance']
        print(f"   Average time: {perf['avg_time_ms']:.2f} ms")
        print(f"   Max time: {perf['max_time_ms']:.2f} ms")
        print(f"   Min time: {perf['min_time_ms']:.2f} ms")
    
    # Show recent events from hooks
    print("\n📝 Recent Debug Events:")
    total_events = 0
    for hook_name, hook in debugger.hooks.items():
        if len(hook.events) > 0:
            recent_event = hook.events[-1]
            print(f"   {hook_name} (Step {recent_event.step}):")
            print(f"     Event type: {recent_event.event_type}")
            if recent_event.metadata:
                for key, value in recent_event.metadata.items():
                    if value is not None:
                        print(f"     {key}: {value}")
            total_events += len(hook.events)
    
    print(f"\n✅ Debugging complete. Total events captured: {total_events}")
    
    return debugger, outputs

# Run streaming debugger demonstration
debugger, debug_outputs = demonstrate_streaming_debugger()

# %% [markdown]
# ## 2. Performance Profiler: Bottleneck Analysis and Optimization
#
# The StreamingProfiler provides detailed performance analysis including execution time tracking, memory usage monitoring, and optimization recommendations.

# %%
def demonstrate_performance_profiler():
    """Demonstrate performance profiling capabilities."""
    print("\n\n⚡ Performance Profiler Demonstration")
    print("-" * 45)
    
    # Create profiler with comprehensive tracking
    profiler = StreamingProfiler(
        enable_memory_tracking=True,
        enable_cpu_tracking=True,
        bottleneck_threshold_ms=2.0,  # Lower threshold for demo
        memory_leak_threshold_mb=10.0
    )
    
    print("🔧 Setting up performance monitoring...")
    
    # Create multiple streaming functions with different performance characteristics
    @profile_streaming(profiler, "fast_ewma")
    @streaming_transform_with_state  
    def fast_ewma_processor(x):
        """Fast EWMA processor."""
        return EWMA(alpha=0.1)(x)
    
    @profile_streaming(profiler, "buffer_processor")
    @streaming_transform_with_state
    def buffer_processor(x):
        """Buffer processor with larger memory footprint."""
        return Buffer(maxlen=100)(x)
    
    @profile_streaming(profiler, "slow_compressed_buffer")
    @streaming_transform_with_state
    def compressed_buffer_processor(x):
        """Compressed buffer processor (potentially slower)."""
        time.sleep(0.001)  # Simulate slower computation
        return CompressedBuffer(maxlen=200, compression="quantile")(x)
    
    print("📊 Running performance analysis across multiple modules...")
    
    # Generate test data
    sequence_length = 50
    test_data = jax.random.normal(rng, (sequence_length,)) * 2.0 + 10.0
    
    # Initialize all processors
    processors = [
        ("fast_ewma", fast_ewma_processor),
        ("buffer_processor", buffer_processor), 
        ("slow_compressed_buffer", compressed_buffer_processor)
    ]
    
    processor_states = {}
    for name, processor in processors:
        params, state = processor.init(rng, test_data[0])
        processor_states[name] = (processor, params, state)
    
    # Process data through all processors
    all_outputs = {}
    
    for i, x in enumerate(test_data):
        if i % 10 == 0:
            print(f"   Processing step {i+1}/{sequence_length}")
        
        for name, (processor, params, state) in processor_states.items():
            output, new_state = processor.apply(params, state, None, x)
            processor_states[name] = (processor, params, new_state)
            
            if name not in all_outputs:
                all_outputs[name] = []
            all_outputs[name].append(output)
    
    print("\n🎯 Performance Analysis Results:")
    
    # Get comprehensive profiling results
    profile_result = profiler.finalize()
    summary = profile_result.get_summary()
    
    print(f"\n📊 Session Overview:")
    session = summary["session"]
    print(f"   Duration: {session['duration_seconds']:.2f} seconds")
    print(f"   Total steps: {session['total_steps']:,}")
    print(f"   Throughput: {session['avg_throughput']:.1f} steps/second")
    
    if "execution" in summary:
        print(f"\n⚡ Execution Performance:")
        exec_stats = summary["execution"]
        print(f"   Average time: {exec_stats['avg_time_ms']:.2f} ms")
        print(f"   Median time: {exec_stats['median_time_ms']:.2f} ms")
        print(f"   Max time: {exec_stats['max_time_ms']:.2f} ms")
        print(f"   Std deviation: {exec_stats['std_dev_ms']:.2f} ms")
    
    if "memory" in summary:
        print(f"\n💾 Memory Performance:")
        mem_stats = summary["memory"]
        print(f"   Peak usage: {mem_stats['peak_usage_mb']:.1f} MB")
        print(f"   Average usage: {mem_stats['avg_usage_mb']:.1f} MB")
        print(f"   Memory efficiency: {mem_stats['memory_efficiency']:.2f}")
    
    # Module-specific analysis
    print(f"\n🔍 Module Performance Breakdown:")
    for name, _ in processors:
        module_summary = profiler.get_module_summary(name)
        if "error" not in module_summary:
            print(f"   {name}:")
            print(f"     Invocations: {module_summary['total_invocations']:,}")
            print(f"     Avg time: {module_summary['avg_time_ms']:.2f} ms")
            print(f"     Max time: {module_summary['max_time_ms']:.2f} ms")
            print(f"     Peak memory: {module_summary['peak_memory_mb']:.1f} MB")
    
    # Bottlenecks and issues
    if summary.get('bottlenecks', 0) > 0:
        print(f"\n🚨 Performance Issues:")
        print(f"   Bottlenecks detected: {summary['bottlenecks']}")
        
        # Show bottleneck details
        for i, bottleneck in enumerate(profile_result.bottlenecks[:3]):
            print(f"   Bottleneck {i+1}:")
            print(f"     Module: {bottleneck['module']}")
            print(f"     Time: {bottleneck['execution_time_ms']:.2f} ms")
            print(f"     Step: {bottleneck['step']}")
    
    # Optimization recommendations
    if profile_result.optimization_recommendations:
        print(f"\n💡 Optimization Recommendations:")
        for i, rec in enumerate(profile_result.optimization_recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Memory leak detection
    memory_leaks = profiler.detect_memory_leaks()
    if memory_leaks:
        print(f"\n🔍 Memory Leak Analysis:")
        for leak in memory_leaks:
            print(f"   Type: {leak['type']}")
            print(f"   Growth: {leak['growth_mb']:.1f} MB")
            print(f"   Recommendation: {leak['recommendation']}")
    
    print(f"\n✅ Performance profiling complete.")
    
    return profiler, profile_result

# Run performance profiler demonstration
profiler, profile_result = demonstrate_performance_profiler()

# %% [markdown]
# ## 3. Memory Tracker: Memory Usage and Leak Detection
#
# The MemoryTracker provides detailed memory usage analysis, leak detection, and optimization insights for streaming computations.

# %%
def demonstrate_memory_tracker():
    """Demonstrate memory tracking capabilities."""
    print("\n\n💾 Memory Tracker Demonstration")
    print("-" * 40)
    
    # Create memory tracker with detailed monitoring
    memory_tracker = MemoryTracker(
        enable_detailed_tracking=True,
        enable_jax_tracking=True,
        snapshot_interval=5,  # Take snapshots every 5 steps
        leak_detection_threshold_mb=5.0,
        max_snapshots=100
    )
    
    print("🔧 Setting up memory monitoring...")
    
    # Create streaming functions with different memory patterns
    @track_memory_usage(memory_tracker, "memory_efficient_ewma")
    @streaming_transform_with_state
    def memory_efficient_processor(x):
        """Memory-efficient EWMA processor."""
        return EWMA(alpha=0.15)(x)
    
    @track_memory_usage(memory_tracker, "growing_buffer")
    @streaming_transform_with_state
    def growing_buffer_processor(x):
        """Buffer processor that accumulates data."""
        return Buffer(maxlen=200)(x)
    
    @track_memory_usage(memory_tracker, "hierarchical_memory")
    @streaming_transform_with_state
    def hierarchical_processor(x):
        """Hierarchical buffer processor."""
        return HierarchicalBuffer(
            recent_maxlen=20,
            medium_maxlen=100,
            long_maxlen=500
        )(x)
    
    print("📊 Running memory analysis across different usage patterns...")
    
    # Generate test data
    sequence_length = 60
    test_data = jax.random.normal(rng, (sequence_length,)) * 3.0 + 5.0
    
    # Initialize processors
    processors = [
        ("memory_efficient_ewma", memory_efficient_processor),
        ("growing_buffer", growing_buffer_processor),
        ("hierarchical_memory", hierarchical_processor)
    ]
    
    processor_states = {}
    for name, processor in processors:
        params, state = processor.init(rng, test_data[0])
        processor_states[name] = (processor, params, state)
    
    # Process data and track memory
    print("   Processing data with memory tracking...")
    
    for i, x in enumerate(test_data):
        if i % 15 == 0:
            print(f"     Step {i+1}/{sequence_length}")
        
        for name, (processor, params, state) in processor_states.items():
            output, new_state = processor.apply(params, state, None, x)
            processor_states[name] = (processor, params, new_state)
    
    print("\n🎯 Memory Analysis Results:")
    
    # Get comprehensive memory summary
    memory_summary = memory_tracker.get_memory_summary()
    
    if "error" not in memory_summary:
        print(f"\n📊 Session Overview:")
        session = memory_summary["session"]
        print(f"   Duration: {session['duration_seconds']:.1f} seconds")
        print(f"   Snapshots taken: {session['total_snapshots']:,}")
        print(f"   Total steps: {session['total_steps']:,}")
        
        print(f"\n🔍 Current Memory Usage:")
        current = memory_summary["current_usage"]
        print(f"   Process memory: {current['process_memory_mb']:.1f} MB")
        print(f"   JAX arrays: {current['jax_arrays_count']:,} ({current['jax_arrays_size_mb']:.1f} MB)")
        print(f"   State variables: {current['state_variables_mb']:.1f} MB")
        print(f"   Parameters: {current['parameters_mb']:.1f} MB")
        print(f"   Buffers: {current['buffers_mb']:.1f} MB")
        
        print(f"\n📈 Memory Statistics:")
        stats = memory_summary["statistics"]
        print(f"   Peak memory: {stats['peak_memory_mb']:.1f} MB")
        print(f"   Average memory: {stats['avg_memory_mb']:.1f} MB")
        print(f"   Total growth: {stats['total_growth_mb']:.1f} MB")
        print(f"   Peak arrays: {stats['peak_arrays_count']:,}")
        
        # Memory leak analysis
        leaks = memory_summary["leaks"]
        if leaks["total_detected"] > 0:
            print(f"\n🚨 Memory Leaks Detected:")
            print(f"   Total: {leaks['total_detected']}")
            print(f"   Critical: {leaks['critical_leaks']}")
            print(f"   High priority: {leaks['high_leaks']}")
            
            # Show leak details
            for leak in memory_tracker.detected_leaks[:2]:
                print(f"\n   🔴 {leak.severity.title()} Leak ({leak.leak_type}):")
                print(f"     Total leaked: {leak.total_leaked_mb:.1f} MB")
                print(f"     Growth rate: {leak.growth_rate_mb_per_step:.3f} MB/step")
                print(f"     Affected modules: {', '.join(leak.affected_modules)}")
                print(f"     Steps: {leak.first_detected_step} - {leak.last_detected_step}")
        else:
            print(f"\n✅ No Memory Leaks Detected")
    
    # Module-specific memory analysis
    print(f"\n🔍 Module Memory Breakdown:")
    for name, _ in processors:
        module_analysis = memory_tracker.get_module_memory_analysis(name)
        if "error" not in module_analysis:
            print(f"   {name}:")
            mem_usage = module_analysis["memory_usage"]
            print(f"     Current: {mem_usage['current_mb']:.1f} MB")
            print(f"     Peak: {mem_usage['peak_mb']:.1f} MB")
            print(f"     Growth: {mem_usage['growth_mb']:.1f} MB")
            
            state_mem = module_analysis["state_memory"]
            print(f"     State peak: {state_mem['peak_mb']:.1f} MB")
    
    # Generate comprehensive memory report
    print(f"\n📋 Comprehensive Memory Report:")
    memory_report = memory_tracker.generate_memory_report()
    print(memory_report)
    
    print(f"\n✅ Memory tracking complete.")
    
    return memory_tracker

# Run memory tracker demonstration
memory_tracker = demonstrate_memory_tracker()

# %% [markdown]
# ## 4. Combined Monitoring: Integrated Debugging, Profiling, and Memory Tracking
#
# Demonstrate using all monitoring tools together for comprehensive analysis of streaming computations.

# %%
def demonstrate_combined_monitoring():
    """Demonstrate combined debugging, profiling, and memory tracking."""
    print("\n\n🔬 Combined Monitoring Demonstration")
    print("-" * 45)
    
    # Create all monitoring tools
    debugger = StreamingDebugger(max_history=200)
    profiler = StreamingProfiler(bottleneck_threshold_ms=1.0)
    memory_tracker = MemoryTracker(snapshot_interval=8)
    
    print("🔧 Setting up comprehensive monitoring stack...")
    
    # Add debug hooks
    debugger.add_state_change_hook("state_monitor")
    debugger.add_breakpoint("anomaly_detector", value_threshold(15.0))
    debugger.add_performance_monitor("performance_watch", threshold_ms=2.0)
    
    # Create a complex streaming system with all monitoring
    @debug_streaming(debugger, "complex_system")
    @profile_streaming(profiler, "complex_system") 
    @track_memory_usage(memory_tracker, "complex_system")
    @streaming_transform_with_state
    def complex_streaming_system(x):
        """Complex streaming system with multiple components."""
        # Multi-level processing pipeline
        
        # Level 1: EWMA smoothing
        smoothed = EWMA(alpha=0.2)(x)
        
        # Level 2: Buffering for pattern detection
        buffered = Buffer(maxlen=30)(smoothed)
        
        # Level 3: Compressed long-term memory
        compressed = CompressedBuffer(
            maxlen=100, 
            compression="ewma",
            compression_params={"alpha": 0.05}
        )(jnp.mean(buffered))
        
        return {
            "smoothed": smoothed,
            "buffered": buffered,
            "compressed": compressed,
            "original": x
        }
    
    print("📊 Running complex streaming system with full monitoring...")
    
    # Generate complex test pattern
    sequence_length = 40
    
    # Create a pattern with different phases
    phase1 = jax.random.normal(rng, (15,)) * 2.0 + 8.0   # Normal operation
    phase2 = jnp.array([20.0, 18.0, 16.0])               # Anomaly spike
    phase3 = jax.random.normal(rng, (12,)) * 1.5 + 6.0   # Quiet period
    phase4 = jax.random.normal(rng, (10,)) * 3.0 + 12.0  # Higher variance
    
    test_sequence = jnp.concatenate([phase1, phase2, phase3, phase4])
    
    print(f"   Processing {len(test_sequence)} data points...")
    print("   Pattern: normal → anomaly → quiet → high-variance")
    
    # Initialize and run system
    params, state = complex_streaming_system.init(rng, test_sequence[0])
    current_state = state
    outputs = []
    
    for i, x in enumerate(test_sequence):
        if i % 10 == 0:
            print(f"     Step {i+1}/{len(test_sequence)}")
        
        # Add some processing delay variation
        if i in [5, 12, 25, 35]:
            time.sleep(0.002)  # Simulated slow operations
        
        output, current_state = complex_streaming_system.apply(
            params, current_state, None, x
        )
        outputs.append(output)
    
    print("\n🎯 Combined Monitoring Results:")
    print("=" * 40)
    
    # 1. Debug analysis
    print("\n🐛 Debug Analysis:")
    debug_summary = debugger.get_summary()
    print(f"   Steps processed: {debug_summary['total_steps']}")
    print(f"   Active hooks: {len([h for h in debug_summary['hooks'].values() if h['enabled']])}")
    
    triggered_hooks = [(name, stats) for name, stats in debug_summary['hooks'].items() 
                      if stats['hit_count'] > 0]
    if triggered_hooks:
        print("   Triggered hooks:")
        for name, stats in triggered_hooks:
            print(f"     {name}: {stats['hit_count']} hits")
    
    # 2. Performance analysis
    print("\n⚡ Performance Analysis:")
    profile_result = profiler.finalize()
    perf_summary = profile_result.get_summary()
    
    if "execution" in perf_summary:
        exec_stats = perf_summary["execution"]
        print(f"   Average time: {exec_stats['avg_time_ms']:.2f} ms")
        print(f"   Max time: {exec_stats['max_time_ms']:.2f} ms")
        print(f"   Throughput: {perf_summary['session']['avg_throughput']:.1f} steps/sec")
    
    if perf_summary.get('bottlenecks', 0) > 0:
        print(f"   Bottlenecks detected: {perf_summary['bottlenecks']}")
    
    if profile_result.optimization_recommendations:
        print("   Optimization recommendations:")
        for rec in profile_result.optimization_recommendations[:2]:
            print(f"     • {rec}")
    
    # 3. Memory analysis
    print("\n💾 Memory Analysis:")
    memory_summary = memory_tracker.get_memory_summary()
    
    if "error" not in memory_summary:
        current = memory_summary["current_usage"]
        stats = memory_summary["statistics"]
        
        print(f"   Current memory: {current['process_memory_mb']:.1f} MB")
        print(f"   Peak memory: {stats['peak_memory_mb']:.1f} MB")
        print(f"   Memory growth: {stats['total_growth_mb']:.1f} MB")
        print(f"   JAX arrays: {current['jax_arrays_count']:,}")
        
        leaks = memory_summary["leaks"]
        if leaks["total_detected"] > 0:
            print(f"   ⚠️  Memory leaks detected: {leaks['total_detected']}")
        else:
            print("   ✅ No memory leaks detected")
    
    # 4. System behavior analysis
    print("\n📊 System Behavior Analysis:")
    
    # Analyze output patterns
    smoothed_values = [out["smoothed"] for out in outputs]
    original_values = [out["original"] for out in outputs]
    
    smoothed_std = jnp.std(jnp.array(smoothed_values))
    original_std = jnp.std(jnp.array(original_values))
    smoothing_effectiveness = (original_std - smoothed_std) / original_std * 100
    
    print(f"   Input variance: {original_std:.3f}")
    print(f"   Smoothed variance: {smoothed_std:.3f}")
    print(f"   Smoothing effectiveness: {smoothing_effectiveness:.1f}%")
    
    # Detect anomalies in output
    mean_smoothed = jnp.mean(jnp.array(smoothed_values))
    std_smoothed = jnp.std(jnp.array(smoothed_values))
    anomaly_threshold = mean_smoothed + 2 * std_smoothed
    
    anomalous_steps = [i for i, val in enumerate(smoothed_values) 
                      if val > anomaly_threshold]
    
    if anomalous_steps:
        print(f"   Anomalous steps detected: {anomalous_steps}")
        print(f"   Anomaly detection threshold: {anomaly_threshold:.2f}")
    
    print("\n🏆 Combined monitoring analysis complete!")
    print("   All tools successfully captured system behavior")
    
    return debugger, profiler, memory_tracker, outputs

# Run combined monitoring demonstration
combined_debugger, combined_profiler, combined_memory_tracker, combined_outputs = demonstrate_combined_monitoring()

# %% [markdown]
# ## 5. Performance Report Generation
#
# Generate comprehensive reports combining insights from all monitoring tools.

# %%
def generate_comprehensive_reports():
    """Generate comprehensive monitoring reports."""
    print("\n\n📋 Comprehensive Monitoring Reports")
    print("-" * 45)
    
    # Generate individual tool reports
    print("🔧 Generating detailed reports...")
    
    # 1. Performance Report
    print("\n" + "="*60)
    print("⚡ PERFORMANCE ANALYSIS REPORT")
    print("="*60)
    
    performance_report = create_performance_report(combined_profiler.finalize())
    print(performance_report)
    
    # 2. Memory Report  
    print("\n" + "="*60)
    print("💾 MEMORY USAGE ANALYSIS REPORT")
    print("="*60)
    
    memory_report = combined_memory_tracker.generate_memory_report()
    print(memory_report)
    
    # 3. Debug Summary Report
    print("\n" + "="*60)
    print("🐛 DEBUGGING SESSION REPORT")
    print("="*60)
    
    debug_summary = combined_debugger.get_summary()
    
    print(f"\n📊 Debug Session Overview:")
    print(f"  Duration: {debug_summary['session_duration']:.2f} seconds")
    print(f"  Total steps: {debug_summary['total_steps']:,}")
    print(f"  Active hooks: {len(debug_summary['hooks'])}")
    
    print(f"\n🔍 Hook Activity Summary:")
    for hook_name, stats in debug_summary['hooks'].items():
        status = "✅ Active" if stats['enabled'] else "❌ Disabled"
        print(f"  {hook_name}: {status}")
        print(f"    Triggers: {stats['hit_count']}")
        print(f"    Events: {stats['events']}")
    
    if debug_summary['performance']['total_steps'] > 0:
        print(f"\n⚡ Debug Performance Metrics:")
        perf = debug_summary['performance']
        print(f"  Average execution time: {perf['avg_time_ms']:.2f} ms")
        print(f"  Performance range: {perf['min_time_ms']:.2f} - {perf['max_time_ms']:.2f} ms")
    
    print(f"\n📈 State Tracking:")
    print(f"  State history length: {debug_summary['state_history_length']}")
    print(f"  Global events captured: {debug_summary['global_events']}")
    
    # 4. Integrated Insights
    print("\n" + "="*60)
    print("🧠 INTEGRATED MONITORING INSIGHTS") 
    print("="*60)
    
    print("\n💡 Key Findings:")
    
    # Performance insights
    profile_final = combined_profiler.finalize()
    if profile_final.optimization_recommendations:
        print("\n⚡ Performance Optimizations:")
        for i, rec in enumerate(profile_final.optimization_recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Memory insights
    memory_final = combined_memory_tracker.get_memory_summary()
    if "error" not in memory_final:
        memory_efficiency = memory_final["statistics"]["avg_memory_mb"]
        if memory_efficiency < 50:
            print("\n💾 Memory Efficiency: Excellent (< 50 MB average)")
        elif memory_efficiency < 100:
            print("\n💾 Memory Efficiency: Good (50-100 MB average)")
        else:
            print("\n💾 Memory Efficiency: Monitor usage (> 100 MB average)")
    
    # Debug insights
    total_debug_events = sum(stats['events'] for stats in debug_summary['hooks'].values())
    if total_debug_events > 10:
        print(f"\n🐛 Debug Activity: High activity detected ({total_debug_events} events)")
        print("   Consider reviewing triggering conditions")
    else:
        print(f"\n🐛 Debug Activity: Normal activity ({total_debug_events} events)")
    
    # System health assessment
    print("\n🏥 System Health Assessment:")
    
    health_score = 100
    issues = []
    
    # Check for bottlenecks
    if len(profile_final.bottlenecks) > 0:
        health_score -= 20
        issues.append(f"Performance bottlenecks detected ({len(profile_final.bottlenecks)})")
    
    # Check for memory leaks
    if len(combined_memory_tracker.detected_leaks) > 0:
        health_score -= 30
        issues.append(f"Memory leaks detected ({len(combined_memory_tracker.detected_leaks)})")
    
    # Check for excessive debug triggers
    if total_debug_events > 20:
        health_score -= 10
        issues.append("High debug activity (may indicate issues)")
    
    if health_score >= 90:
        health_status = "🟢 Excellent"
    elif health_score >= 70:
        health_status = "🟡 Good" 
    elif health_score >= 50:
        health_status = "🟠 Fair"
    else:
        health_status = "🔴 Poor"
    
    print(f"  Overall Health: {health_status} ({health_score}/100)")
    
    if issues:
        print("  Issues identified:")
        for issue in issues:
            print(f"    • {issue}")
    else:
        print("  ✅ No significant issues detected")
    
    # Recommendations summary
    print("\n🎯 Recommended Actions:")
    
    if health_score >= 90:
        print("  • System operating optimally")
        print("  • Continue monitoring for trend analysis")
    elif health_score >= 70:
        print("  • Minor optimizations available")
        print("  • Review performance recommendations")
    else:
        print("  • Immediate attention required")
        print("  • Address memory leaks and bottlenecks")
        print("  • Consider system redesign if issues persist")
    
    print("\n" + "="*60)
    print("✅ COMPREHENSIVE MONITORING ANALYSIS COMPLETE")
    print("="*60)

# Generate comprehensive reports
generate_comprehensive_reports()

# %% [markdown]
# ## 6. Advanced Monitoring Patterns
#
# Demonstrate advanced monitoring techniques and patterns for production streaming systems.

# %%
def demonstrate_advanced_monitoring():
    """Demonstrate advanced monitoring patterns."""
    print("\n\n🚀 Advanced Monitoring Patterns")
    print("-" * 40)
    
    print("🔧 Setting up advanced monitoring scenarios...")
    
    # Create specialized monitoring setups for different scenarios
    
    # Scenario 1: Production monitoring with alerts
    production_debugger = StreamingDebugger()
    
    # Add production-ready hooks
    production_debugger.add_breakpoint(
        "critical_error",
        lambda step, state, inp, out: jnp.isnan(out).any() if hasattr(out, 'any') else jnp.isnan(out)
    )
    
    production_debugger.add_performance_monitor(
        "sla_violation", 
        threshold_ms=10.0  # SLA threshold
    )
    
    # Custom hook for business logic monitoring
    def business_logic_monitor(step, state, inp, out):
        # Monitor for business rule violations
        if hasattr(out, '__len__'):
            return any(val > 50.0 for val in out) if len(out) > 0 else False
        return float(out) > 50.0 if out is not None else False
    
    production_debugger.add_hook(DebugHook(
        name="business_rule_violation",
        condition=business_logic_monitor,
        action="alert"
    ))
    
    print("   ✅ Production monitoring setup complete")
    
    # Scenario 2: Development monitoring with detailed tracking
    dev_profiler = StreamingProfiler(
        enable_memory_tracking=True,
        enable_cpu_tracking=True,
        enable_jit_tracking=True,
        bottleneck_threshold_ms=1.0  # Very sensitive
    )
    
    print("   ✅ Development profiling setup complete")
    
    # Scenario 3: Research monitoring with comprehensive state tracking
    research_memory_tracker = MemoryTracker(
        enable_detailed_tracking=True,
        enable_jax_tracking=True,
        snapshot_interval=1,  # Every step
        leak_detection_threshold_mb=1.0  # Very sensitive
    )
    
    print("   ✅ Research memory tracking setup complete")
    
    # Create test streaming system with all advanced monitoring
    @debug_streaming(production_debugger, "production_system")
    @profile_streaming(dev_profiler, "development_profiler")
    @track_memory_usage(research_memory_tracker, "research_tracker")
    @streaming_transform_with_state
    def advanced_monitored_system(x):
        """Advanced system with comprehensive monitoring."""
        
        # Multi-stage processing with different characteristics
        
        # Stage 1: Input validation and normalization
        normalized = jnp.clip(x, -100.0, 100.0)  # Prevent extreme values
        
        # Stage 2: Feature extraction
        features = jnp.array([
            normalized,
            normalized**2,
            jnp.sin(normalized),
            jnp.exp(normalized / 10.0)
        ])
        
        # Stage 3: Streaming aggregation
        aggregated = EWMA(alpha=0.3)(jnp.mean(features))
        
        # Stage 4: Decision logic
        decision_score = jnp.tanh(aggregated / 5.0)
        
        return {
            "normalized": normalized,
            "features": features,
            "aggregated": aggregated,
            "decision_score": decision_score,
            "alert_flag": decision_score > 0.8
        }
    
    print("\n📊 Running advanced monitoring scenarios...")
    
    # Test different data patterns
    scenarios = [
        ("normal_operation", jax.random.normal(rng, (15,)) * 3.0 + 10.0),
        ("extreme_values", jnp.array([150.0, -200.0, 75.0, 25.0, 10.0])),
        ("nan_injection", jnp.array([5.0, jnp.nan, 10.0, 15.0])),
        ("gradual_drift", jnp.linspace(5.0, 45.0, 20)),
    ]
    
    # Initialize system
    params, state = advanced_monitored_system.init(rng, 1.0)
    current_state = state
    all_scenario_outputs = {}
    
    for scenario_name, test_data in scenarios:
        print(f"\n   Testing scenario: {scenario_name}")
        
        scenario_outputs = []
        for i, x in enumerate(test_data):
            if not jnp.isnan(x):  # Skip NaN values for actual processing
                output, current_state = advanced_monitored_system.apply(
                    params, current_state, None, x
                )
                scenario_outputs.append(output)
        
        all_scenario_outputs[scenario_name] = scenario_outputs
        print(f"     Processed {len(scenario_outputs)} data points")
    
    print("\n🎯 Advanced Monitoring Results:")
    print("=" * 40)
    
    # Production monitoring results
    print("\n🏭 Production Monitoring:")
    prod_summary = production_debugger.get_summary()
    
    critical_hooks = [(name, stats) for name, stats in prod_summary['hooks'].items() 
                     if stats['hit_count'] > 0 and 'critical' in name.lower()]
    
    if critical_hooks:
        print("   🚨 Critical alerts triggered:")
        for name, stats in critical_hooks:
            print(f"     {name}: {stats['hit_count']} times")
    else:
        print("   ✅ No critical alerts triggered")
    
    sla_violations = [(name, stats) for name, stats in prod_summary['hooks'].items() 
                     if 'sla' in name.lower() and stats['hit_count'] > 0]
    
    if sla_violations:
        print("   ⚠️  SLA violations detected:")
        for name, stats in sla_violations:
            print(f"     {name}: {stats['hit_count']} violations")
    else:
        print("   ✅ No SLA violations")
    
    # Development profiling results
    print("\n🔬 Development Profiling:")
    dev_result = dev_profiler.finalize()
    dev_summary = dev_result.get_summary()
    
    if "execution" in dev_summary:
        exec_stats = dev_summary["execution"]
        print(f"   Execution analysis:")
        print(f"     Mean time: {exec_stats['avg_time_ms']:.3f} ms")
        print(f"     Std deviation: {exec_stats['std_dev_ms']:.3f} ms")
        print(f"     Performance consistency: {100 - (exec_stats['std_dev_ms']/exec_stats['avg_time_ms']*100):.1f}%")
    
    if dev_result.bottlenecks:
        print(f"   Development bottlenecks: {len(dev_result.bottlenecks)}")
    else:
        print("   ✅ No development bottlenecks detected")
    
    # Research memory tracking results
    print("\n🔬 Research Memory Tracking:")
    research_summary = research_memory_tracker.get_memory_summary()
    
    if "error" not in research_summary:
        stats = research_summary["statistics"]
        print(f"   Memory behavior:")
        print(f"     Peak usage: {stats['peak_memory_mb']:.1f} MB")
        print(f"     Growth pattern: {stats['total_growth_mb']:.1f} MB total")
        print(f"     Efficiency ratio: {stats['min_memory_mb']/stats['peak_memory_mb']:.2f}")
        
        leaks = research_summary["leaks"]
        if leaks["total_detected"] > 0:
            print(f"   🔍 Research insights: {leaks['total_detected']} memory patterns detected")
        else:
            print("   ✅ Stable memory usage patterns")
    
    # Scenario-specific analysis
    print("\n📊 Scenario Analysis:")
    for scenario_name, outputs in all_scenario_outputs.items():
        if outputs:
            decision_scores = [out["decision_score"] for out in outputs]
            alert_flags = [out["alert_flag"] for out in outputs]
            
            print(f"   {scenario_name}:")
            print(f"     Samples: {len(outputs)}")
            print(f"     Decision range: {min(decision_scores):.3f} - {max(decision_scores):.3f}")
            print(f"     Alerts triggered: {sum(alert_flags)}")
    
    print("\n🎓 Advanced Monitoring Insights:")
    print("   • Production monitoring enables real-time alerting")
    print("   • Development profiling reveals optimization opportunities")
    print("   • Research tracking provides deep system understanding")
    print("   • Scenario-based testing validates system robustness")
    
    print("\n✅ Advanced monitoring demonstration complete!")
    
    return production_debugger, dev_profiler, research_memory_tracker

# Run advanced monitoring demonstration
advanced_debugger, advanced_profiler, advanced_memory_tracker = demonstrate_advanced_monitoring()

# %% [markdown]
# ## 7. Summary and Best Practices
#
# Summarize the debugging and profiling capabilities and provide best practices for production use.

# %%
print("\n\n🏆 WAX-ML DEBUGGING AND PROFILING SUMMARY")
print("=" * 60)

print("\n✨ Key Capabilities Demonstrated:")

print("\n🐛 Streaming Debugger:")
print("   • Real-time state inspection and monitoring")
print("   • Conditional breakpoints based on data or state conditions")
print("   • Interactive debugging sessions with step-by-step analysis")
print("   • Comprehensive event logging and history tracking")
print("   • Performance-aware debugging with execution time monitoring")

print("\n⚡ Performance Profiler:")
print("   • Detailed execution time analysis and bottleneck identification")
print("   • Memory usage tracking and leak detection")
print("   • Statistical performance analysis with optimization recommendations")
print("   • Module-specific performance breakdown and comparison")
print("   • JIT compilation tracking and optimization insights")

print("\n💾 Memory Tracker:")
print("   • Comprehensive memory usage monitoring and analysis")
print("   • JAX array lifecycle tracking and optimization")
print("   • State variable memory footprint analysis")
print("   • Memory leak detection with severity classification")
print("   • Detailed memory reports with optimization recommendations")

print("\n🔬 Advanced Monitoring:")
print("   • Production-ready alerting and SLA monitoring")
print("   • Development-focused detailed performance analysis")
print("   • Research-grade comprehensive system behavior tracking")
print("   • Scenario-based testing and validation")

print("\n🎯 Production Best Practices:")

print("\n1. 📊 Monitoring Strategy:")
print("   • Use StreamingDebugger for development and troubleshooting")
print("   • Deploy StreamingProfiler for performance optimization")
print("   • Implement MemoryTracker for long-running systems")
print("   • Combine all tools for comprehensive system health monitoring")

print("\n2. 🔧 Configuration Guidelines:")
print("   • Set appropriate thresholds based on system requirements")
print("   • Use sampling intervals to balance monitoring overhead")
print("   • Configure alert conditions for critical system events")
print("   • Implement automated reporting for continuous monitoring")

print("\n3. 🚀 Performance Optimization:")
print("   • Regular profiling to identify bottlenecks and inefficiencies")
print("   • Memory leak detection and prevention strategies")
print("   • JIT compilation optimization for streaming workloads")
print("   • Statistical analysis for performance trend identification")

print("\n4. 🏥 System Health:")
print("   • Continuous monitoring of execution time and memory usage")
print("   • Proactive alerting for performance degradation")
print("   • Historical analysis for capacity planning")
print("   • Automated health scoring and reporting")

print("\n📚 Integration Examples:")

print("\n• Financial Trading Systems:")
print("   - Monitor execution latency for high-frequency trading")
print("   - Track memory usage in long-running market data processors")
print("   - Debug anomalies in real-time risk management systems")

print("\n• IoT Data Processing:")
print("   - Profile sensor data ingestion pipelines")
print("   - Monitor memory usage in edge computing scenarios")
print("   - Debug data quality issues in streaming analytics")

print("\n• Research and Development:")
print("   - Comprehensive analysis of experimental streaming algorithms")
print("   - Performance comparison between different model architectures")
print("   - Memory optimization for large-scale streaming experiments")

print("\n🔮 Future Enhancements:")
print("   • Integration with external monitoring systems (Prometheus, Grafana)")
print("   • Distributed streaming monitoring across multiple devices")
print("   • Machine learning-based anomaly detection in monitoring data")
print("   • Real-time visualization of streaming computation graphs")

print("\n" + "=" * 60)
print("🎉 DEBUGGING AND PROFILING DEMO COMPLETE!")
print("   Comprehensive monitoring tools for production streaming AI")
print("   Built on WAX-ML's robust Flax streaming architecture")
print("=" * 60)

# %%
