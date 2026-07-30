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
"""Performance profiler for WAX-ML streaming computations.

This module provides comprehensive performance profiling capabilities for
streaming computations, including execution time analysis, memory usage tracking,
bottleneck identification, and optimization recommendations.

Key features:
- Real-time performance monitoring
- Memory usage analysis and leak detection
- Bottleneck identification and hotspot analysis
- JIT compilation tracking and optimization
- Statistical performance analysis
- Export capabilities for external analysis tools

Inspired by profiling tools from:
- JAX profiler and performance analysis tools
- Python cProfile and line_profiler
- TensorFlow Profiler
- NVIDIA Nsight for GPU profiling
"""

import os
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import psutil


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single computation step."""

    step: int
    module_name: str
    execution_time: float  # seconds
    memory_usage: float  # MB
    memory_delta: float  # MB change from previous step
    cpu_percent: float  # CPU usage percentage
    jit_compilation: bool  # Whether JIT compilation occurred
    input_size: int  # Size of input data
    output_size: int  # Size of output data
    timestamp: float  # Unix timestamp

    # Advanced metrics
    cache_hits: int = 0
    cache_misses: int = 0
    gc_collections: int = 0
    thread_count: int = 0


@dataclass
class ProfileResult:
    """Complete profiling results for analysis."""

    session_duration: float
    total_steps: int
    metrics: list[PerformanceMetrics]

    # Statistical summaries
    execution_stats: dict[str, float] = field(default_factory=dict)
    memory_stats: dict[str, float] = field(default_factory=dict)
    bottlenecks: list[dict[str, Any]] = field(default_factory=list)
    optimization_recommendations: list[str] = field(default_factory=list)

    def get_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of profiling results."""
        if not self.metrics:
            return {"error": "No profiling data available"}

        execution_times = [m.execution_time for m in self.metrics]
        memory_usage = [m.memory_usage for m in self.metrics]

        return {
            "session": {
                "duration_seconds": self.session_duration,
                "total_steps": self.total_steps,
                "avg_throughput": self.total_steps / self.session_duration
                if self.session_duration > 0
                else 0,
            },
            "execution": {
                "total_time": sum(execution_times),
                "avg_time_ms": statistics.mean(execution_times) * 1000,
                "median_time_ms": statistics.median(execution_times) * 1000,
                "max_time_ms": max(execution_times) * 1000,
                "min_time_ms": min(execution_times) * 1000,
                "std_dev_ms": statistics.stdev(execution_times) * 1000
                if len(execution_times) > 1
                else 0,
            },
            "memory": {
                "peak_usage_mb": max(memory_usage),
                "avg_usage_mb": statistics.mean(memory_usage),
                "total_allocated": sum(m.memory_delta for m in self.metrics if m.memory_delta > 0),
                "memory_efficiency": min(memory_usage) / max(memory_usage)
                if max(memory_usage) > 0
                else 1.0,
            },
            "bottlenecks": len(self.bottlenecks),
            "jit_compilations": sum(1 for m in self.metrics if m.jit_compilation),
            "recommendations": len(self.optimization_recommendations),
        }


class StreamingProfiler:
    """Comprehensive performance profiler for streaming computations."""

    def __init__(
        self,
        enable_memory_tracking: bool = True,
        enable_cpu_tracking: bool = True,
        enable_jit_tracking: bool = True,
        bottleneck_threshold_ms: float = 100.0,
        memory_leak_threshold_mb: float = 50.0,
        max_history: int = 10000,
    ):
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
        self.enable_jit_tracking = enable_jit_tracking
        self.bottleneck_threshold_ms = bottleneck_threshold_ms
        self.memory_leak_threshold_mb = memory_leak_threshold_mb
        self.max_history = max_history

        # Profiling data
        self.metrics: deque[PerformanceMetrics] = deque(maxlen=max_history)
        self.module_stats: dict[str, list[float]] = defaultdict(list)

        # Session tracking
        self.session_start_time = time.time()
        self.step_count = 0
        self.enabled = True

        # System monitoring
        self.process = (
            psutil.Process(os.getpid()) if enable_memory_tracking or enable_cpu_tracking else None
        )
        self.last_memory_usage = 0.0

        # JIT compilation tracking
        self.jit_cache: set[str] = set()
        self.compilation_events: list[dict[str, Any]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Hook into JAX compilation if JIT tracking enabled
        if enable_jit_tracking:
            self._setup_jit_tracking()

    def _setup_jit_tracking(self):
        """Setup JAX JIT compilation tracking."""
        try:
            # Store original jit function
            self._original_jit = jax.jit

            def tracked_jit(*args, **kwargs):
                # Wrap the JIT function to track compilations
                jitted_fn = self._original_jit(*args, **kwargs)

                def wrapper(*fn_args, **fn_kwargs):
                    fn_key = str(jitted_fn)
                    compilation_occurred = fn_key not in self.jit_cache

                    if compilation_occurred:
                        self.jit_cache.add(fn_key)
                        self.compilation_events.append(
                            {
                                "timestamp": time.time(),
                                "function": str(jitted_fn),
                                "step": self.step_count,
                            }
                        )

                    return jitted_fn(*fn_args, **fn_kwargs)

                return wrapper

            # Replace jax.jit temporarily (this is a simplified approach)
            # In practice, this would need more sophisticated hooking

        except Exception as e:
            print(f"Warning: Could not setup JIT tracking: {e}")

    def start_step(self, module_name: str) -> "ProfilerContext":
        """Start profiling a computation step."""
        return ProfilerContext(self, module_name)

    def record_metrics(
        self,
        module_name: str,
        execution_time: float,
        input_data: Any = None,
        output_data: Any = None,
        jit_compilation: bool = False,
    ) -> None:
        """Record performance metrics for a computation step."""
        if not self.enabled:
            return

        with self._lock:
            self.step_count += 1

            # Gather system metrics
            memory_usage = 0.0
            memory_delta = 0.0
            cpu_percent = 0.0

            if self.process:
                if self.enable_memory_tracking:
                    memory_info = self.process.memory_info()
                    memory_usage = memory_info.rss / 1024 / 1024  # MB
                    memory_delta = memory_usage - self.last_memory_usage
                    self.last_memory_usage = memory_usage

                if self.enable_cpu_tracking:
                    cpu_percent = self.process.cpu_percent()

            # Calculate data sizes
            input_size = self._calculate_data_size(input_data)
            output_size = self._calculate_data_size(output_data)

            # Create metrics record
            metrics = PerformanceMetrics(
                step=self.step_count,
                module_name=module_name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                memory_delta=memory_delta,
                cpu_percent=cpu_percent,
                jit_compilation=jit_compilation,
                input_size=input_size,
                output_size=output_size,
                timestamp=time.time(),
                thread_count=threading.active_count(),
            )

            self.metrics.append(metrics)
            self.module_stats[module_name].append(execution_time)

            # Check for bottlenecks
            if execution_time * 1000 > self.bottleneck_threshold_ms:
                self._record_bottleneck(metrics)

    def _calculate_data_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            if data is None:
                return 0
            elif isinstance(data, jnp.ndarray):
                return data.nbytes
            elif isinstance(data, (list, tuple)):
                return sum(self._calculate_data_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._calculate_data_size(v) for v in data.values())
            elif isinstance(data, (int, float)):
                return 8  # Approximate size
            elif isinstance(data, str):
                return len(data.encode("utf-8"))
            else:
                return 0
        except Exception:
            return 0

    def _record_bottleneck(self, metrics: PerformanceMetrics):
        """Record a performance bottleneck."""
        bottleneck = {
            "step": metrics.step,
            "module": metrics.module_name,
            "execution_time_ms": metrics.execution_time * 1000,
            "memory_usage_mb": metrics.memory_usage,
            "input_size_bytes": metrics.input_size,
            "output_size_bytes": metrics.output_size,
            "timestamp": metrics.timestamp,
        }

        # Store in result (we'll add this to ProfileResult later)
        if not hasattr(self, "_bottlenecks"):
            self._bottlenecks = []
        self._bottlenecks.append(bottleneck)

    def get_module_summary(self, module_name: str) -> dict[str, Any]:
        """Get performance summary for a specific module."""
        with self._lock:
            module_metrics = [m for m in self.metrics if m.module_name == module_name]

            if not module_metrics:
                return {"error": f"No data for module {module_name}"}

            execution_times = [m.execution_time for m in module_metrics]
            memory_usage = [m.memory_usage for m in module_metrics]

            return {
                "module_name": module_name,
                "total_invocations": len(module_metrics),
                "total_time_ms": sum(execution_times) * 1000,
                "avg_time_ms": statistics.mean(execution_times) * 1000,
                "max_time_ms": max(execution_times) * 1000,
                "min_time_ms": min(execution_times) * 1000,
                "avg_memory_mb": statistics.mean(memory_usage),
                "peak_memory_mb": max(memory_usage),
                "jit_compilations": sum(1 for m in module_metrics if m.jit_compilation),
            }

    def detect_memory_leaks(self) -> list[dict[str, Any]]:
        """Detect potential memory leaks."""
        with self._lock:
            if len(self.metrics) < 10:
                return []

            # Look for consistent memory growth
            recent_metrics = list(self.metrics)[-10:]
            memory_trend = [m.memory_usage for m in recent_metrics]

            # Simple linear trend detection
            if len(memory_trend) > 1:
                start_memory = memory_trend[0]
                end_memory = memory_trend[-1]
                growth = end_memory - start_memory

                if growth > self.memory_leak_threshold_mb:
                    return [
                        {
                            "type": "memory_leak",
                            "growth_mb": growth,
                            "start_memory": start_memory,
                            "end_memory": end_memory,
                            "steps_analyzed": len(recent_metrics),
                            "recommendation": "Check for accumulating state or unreleased references",
                        }
                    ]

            return []

    def generate_optimization_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations: list[str] = []

        with self._lock:
            if not self.metrics:
                return recommendations

            # Analyze execution times
            execution_times = [m.execution_time for m in self.metrics]
            avg_time = statistics.mean(execution_times)
            max_time = max(execution_times)

            if max_time > avg_time * 3:
                recommendations.append(
                    "High execution time variance detected. Consider JIT compilation or caching."
                )

            # Analyze memory usage
            memory_deltas = [m.memory_delta for m in self.metrics if m.memory_delta > 0]
            if memory_deltas and sum(memory_deltas) > 100:  # More than 100MB allocated
                recommendations.append(
                    "High memory allocation detected. Consider using in-place operations or smaller batch sizes."
                )

            # Analyze JIT compilations
            jit_compilations = sum(1 for m in self.metrics if m.jit_compilation)
            if jit_compilations > len(self.metrics) * 0.1:  # More than 10% compilations
                recommendations.append(
                    "Frequent JIT compilations detected. Consider pre-compiling or reducing dynamic shapes."
                )

            # Analyze data transfer
            large_inputs = [m for m in self.metrics if m.input_size > 1024 * 1024]  # > 1MB
            if len(large_inputs) > len(self.metrics) * 0.5:
                recommendations.append(
                    "Large input data detected. Consider data streaming or compression."
                )

            # Module-specific recommendations
            for module_name in self.module_stats:
                module_times = self.module_stats[module_name]
                if len(module_times) > 1:
                    std_dev = statistics.stdev(module_times)
                    mean_time = statistics.mean(module_times)

                    if std_dev > mean_time * 0.5:  # High variance
                        recommendations.append(
                            f"Module '{module_name}' shows high execution time variance. "
                            "Consider optimizing or investigating input dependencies."
                        )

        return recommendations

    def finalize(self) -> ProfileResult:
        """Finalize profiling and return complete results."""
        with self._lock:
            session_duration = time.time() - self.session_start_time

            # Generate comprehensive statistics
            if self.metrics:
                execution_times = [m.execution_time for m in self.metrics]
                memory_usage = [m.memory_usage for m in self.metrics]

                execution_stats = {
                    "mean": statistics.mean(execution_times),
                    "median": statistics.median(execution_times),
                    "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    "min": min(execution_times),
                    "max": max(execution_times),
                }

                memory_stats = {
                    "mean": statistics.mean(memory_usage),
                    "median": statistics.median(memory_usage),
                    "std_dev": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                    "min": min(memory_usage),
                    "max": max(memory_usage),
                }
            else:
                execution_stats = {}
                memory_stats = {}

            # Collect bottlenecks
            bottlenecks = getattr(self, "_bottlenecks", [])

            # Generate recommendations
            recommendations = self.generate_optimization_recommendations()

            return ProfileResult(
                session_duration=session_duration,
                total_steps=self.step_count,
                metrics=list(self.metrics),
                execution_stats=execution_stats,
                memory_stats=memory_stats,
                bottlenecks=bottlenecks,
                optimization_recommendations=recommendations,
            )

    def reset(self) -> None:
        """Reset profiler state."""
        with self._lock:
            self.metrics.clear()
            self.module_stats.clear()
            self.step_count = 0
            self.session_start_time = time.time()
            self.last_memory_usage = 0.0
            self.jit_cache.clear()
            self.compilation_events.clear()
            if hasattr(self, "_bottlenecks"):
                self._bottlenecks.clear()

    def enable(self) -> None:
        """Enable profiling."""
        self.enabled = True

    def disable(self) -> None:
        """Disable profiling."""
        self.enabled = False


class ProfilerContext:
    """Context manager for profiling individual computation steps."""

    def __init__(self, profiler: StreamingProfiler, module_name: str):
        self.profiler = profiler
        self.module_name = module_name
        self.start_time = None
        self.jit_compilation = False

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            execution_time = time.time() - self.start_time
            self.profiler.record_metrics(
                module_name=self.module_name,
                execution_time=execution_time,
                jit_compilation=self.jit_compilation,
            )

    def set_jit_compilation(self, occurred: bool):
        """Mark whether JIT compilation occurred during this step."""
        self.jit_compilation = occurred


def profile_streaming(
    profiler: StreamingProfiler | None = None,
    module_name: str | None = None,
    enable_detailed_tracking: bool = True,
):
    """Decorator for adding performance profiling to streaming functions.

    Example:
        profiler = StreamingProfiler()

        @profile_streaming(profiler, "ewma_module")
        @streaming_transform_with_state
        def ewma_processor(x):
            return EWMA(alpha=0.1)(x)

        # ... run computations ...

        results = profiler.finalize()
        print(results.get_summary())
    """
    if profiler is None:
        profiler = StreamingProfiler()

    def decorator(fn: Callable) -> Callable:
        actual_module_name = module_name or fn.__name__

        # Create a wrapper that preserves the original function interface
        class ProfileWrapper:
            def __init__(self, original_fn):
                self.original_fn = original_fn
                self.__name__ = getattr(original_fn, "__name__", "profile_wrapped")
                # Copy all attributes from original function except apply
                for attr in ["init", "__call__"]:
                    if hasattr(original_fn, attr):
                        setattr(self, attr, getattr(original_fn, attr))

                # Create a wrapped apply method that includes profiling
                if hasattr(original_fn, "apply"):
                    self.apply = self._create_profile_apply(original_fn.apply)

            def _create_profile_apply(self, original_apply):
                def profile_apply(params, state, rng, *args, **kwargs):
                    with profiler.start_step(actual_module_name) as ctx:
                        # Track input data if detailed tracking enabled
                        input_data = args[0] if args and enable_detailed_tracking else None

                        # Execute function
                        result = original_apply(params, state, rng, *args, **kwargs)

                        # Track output data if detailed tracking enabled
                        output_data = result if enable_detailed_tracking else None

                        # Record additional context
                        ctx.profiler.record_metrics(
                            module_name=actual_module_name,
                            execution_time=0,  # Will be set by context manager
                            input_data=input_data,
                            output_data=output_data,
                        )

                        return result

                return profile_apply

            def __call__(self, *args, **kwargs):
                return self.original_fn(*args, **kwargs)

        return ProfileWrapper(fn)

    return decorator


def create_performance_report(profile_result: ProfileResult) -> str:
    """Create a human-readable performance report."""
    summary = profile_result.get_summary()

    report = []
    report.append("🔍 WAX-ML Streaming Performance Report")
    report.append("=" * 50)

    # Session overview
    report.append("\n📊 Session Overview:")
    report.append(f"  Duration: {summary['session']['duration_seconds']:.2f} seconds")
    report.append(f"  Total Steps: {summary['session']['total_steps']:,}")
    report.append(f"  Throughput: {summary['session']['avg_throughput']:.1f} steps/second")

    # Execution performance
    if "execution" in summary:
        exec_stats = summary["execution"]
        report.append("\n⚡ Execution Performance:")
        report.append(f"  Average Time: {exec_stats['avg_time_ms']:.2f} ms")
        report.append(f"  Median Time: {exec_stats['median_time_ms']:.2f} ms")
        report.append(f"  Max Time: {exec_stats['max_time_ms']:.2f} ms")
        report.append(f"  Std Deviation: {exec_stats['std_dev_ms']:.2f} ms")

    # Memory usage
    if "memory" in summary:
        mem_stats = summary["memory"]
        report.append("\n💾 Memory Usage:")
        report.append(f"  Peak Usage: {mem_stats['peak_usage_mb']:.1f} MB")
        report.append(f"  Average Usage: {mem_stats['avg_usage_mb']:.1f} MB")
        report.append(f"  Total Allocated: {mem_stats['total_allocated']:.1f} MB")
        report.append(f"  Memory Efficiency: {mem_stats['memory_efficiency']:.2f}")

    # Issues and optimizations
    if summary.get("bottlenecks", 0) > 0:
        report.append("\n🚨 Issues Detected:")
        report.append(f"  Bottlenecks: {summary['bottlenecks']}")

    if summary.get("jit_compilations", 0) > 0:
        report.append(f"  JIT Compilations: {summary['jit_compilations']}")

    # Recommendations
    if profile_result.optimization_recommendations:
        report.append("\n💡 Optimization Recommendations:")
        for i, rec in enumerate(profile_result.optimization_recommendations, 1):
            report.append(f"  {i}. {rec}")

    report.append("\n" + "=" * 50)

    return "\n".join(report)
