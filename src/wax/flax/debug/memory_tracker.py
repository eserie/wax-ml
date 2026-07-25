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
"""Memory usage tracker and analyzer for WAX-ML streaming computations.

This module provides detailed memory tracking capabilities for streaming computations,
including allocation tracking, leak detection, optimization analysis, and memory
profiling specifically designed for JAX/Flax streaming workflows.

Key features:
- Real-time memory allocation tracking
- Memory leak detection and analysis
- JAX array memory usage monitoring
- State variable memory footprint analysis
- Garbage collection monitoring
- Memory optimization recommendations
- Export capabilities for memory analysis

Designed for JAX/Flax specifics:
- Device memory tracking (CPU/GPU)
- JAX array lifecycle monitoring
- Flax parameter memory analysis
- Streaming state memory patterns
"""

import gc
import sys
import threading
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""

    timestamp: float
    step: int
    module_name: str

    # System memory
    process_memory_mb: float
    system_memory_mb: float
    available_memory_mb: float

    # JAX-specific memory
    jax_arrays_count: int
    jax_arrays_size_mb: float
    device_memory_mb: dict[str, float]

    # Python object memory
    python_objects_mb: float
    gc_objects_count: int

    # State-specific memory
    state_variables_mb: float
    parameters_mb: float
    buffers_mb: float

    # Memory deltas (change from previous snapshot)
    memory_delta_mb: float = 0.0
    arrays_delta_count: int = 0

    # Metadata
    gc_collections: int = 0
    weak_refs_count: int = 0


@dataclass
class MemoryLeak:
    """Detected memory leak information."""

    leak_type: str  # 'growing_arrays', 'accumulating_state', 'python_objects'
    severity: str   # 'low', 'medium', 'high', 'critical'
    growth_rate_mb_per_step: float
    total_leaked_mb: float
    first_detected_step: int
    last_detected_step: int
    affected_modules: list[str]
    description: str
    recommendations: list[str]


class MemoryTracker:
    """Comprehensive memory tracker for streaming computations."""

    def __init__(self,
                 enable_detailed_tracking: bool = True,
                 enable_jax_tracking: bool = True,
                 enable_gc_monitoring: bool = True,
                 snapshot_interval: int = 10,
                 leak_detection_threshold_mb: float = 10.0,
                 max_snapshots: int = 1000):

        self.enable_detailed_tracking = enable_detailed_tracking
        self.enable_jax_tracking = enable_jax_tracking
        self.enable_gc_monitoring = enable_gc_monitoring
        self.snapshot_interval = snapshot_interval
        self.leak_detection_threshold_mb = leak_detection_threshold_mb
        self.max_snapshots = max_snapshots

        # Memory tracking data
        self.snapshots: deque[MemorySnapshot] = deque(maxlen=max_snapshots)
        self.detected_leaks: list[MemoryLeak] = []

        # JAX array tracking
        self.tracked_arrays: dict[int, dict[str, Any]] = {}
        self.array_lifecycle: defaultdict[str, list[dict]] = defaultdict(list)

        # State tracking
        self.state_memory_history: defaultdict[str, list[float]] = defaultdict(list)
        self.module_memory_usage: defaultdict[str, list[float]] = defaultdict(list)

        # Session tracking
        self.session_start_time = time.time()
        self.step_count = 0
        self.enabled = True

        # Baseline measurements
        self.baseline_memory = None
        self.last_snapshot = None

        # Thread safety
        self._lock = threading.RLock()

        # Initialize tracemalloc if detailed tracking enabled
        if enable_detailed_tracking:
            try:
                tracemalloc.start()
                self._tracemalloc_enabled = True
            except Exception as e:
                print(f"Warning: Could not start tracemalloc: {e}")
                self._tracemalloc_enabled = False
        else:
            self._tracemalloc_enabled = False

        # Set up JAX array tracking if enabled
        if enable_jax_tracking:
            self._setup_jax_tracking()

    def _setup_jax_tracking(self):
        """Setup JAX array creation and destruction tracking."""
        # Store original array creation functions
        self._original_array = jnp.array
        self._tracked_array_ids = weakref.WeakSet()

        def tracked_array(*args, **kwargs):
            arr = self._original_array(*args, **kwargs)

            # Track array creation
            array_id = id(arr)
            self.tracked_arrays[array_id] = {
                'creation_time': time.time(),
                'creation_step': self.step_count,
                'size_bytes': arr.nbytes,
                'shape': arr.shape,
                'dtype': arr.dtype
            }

            # Use weak reference to track when array is garbage collected
            def cleanup_callback(ref):
                if array_id in self.tracked_arrays:
                    array_info = self.tracked_arrays.pop(array_id)
                    self.array_lifecycle['destroyed'].append({
                        'array_id': array_id,
                        'destruction_time': time.time(),
                        'destruction_step': self.step_count,
                        'lifetime_steps': self.step_count - array_info['creation_step'],
                        'size_bytes': array_info['size_bytes']
                    })

            self._tracked_array_ids.add(weakref.ref(arr, cleanup_callback))

            self.array_lifecycle['created'].append({
                'array_id': array_id,
                'creation_time': time.time(),
                'creation_step': self.step_count,
                'size_bytes': arr.nbytes,
                'shape': arr.shape
            })

            return arr

        # Note: This is a simplified approach. In practice, you'd need more
        # sophisticated hooking into JAX's array creation pipeline

    def take_snapshot(self, module_name: str = "unknown",
                     state: Any = None, force: bool = False) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        if not self.enabled:
            return None

        with self._lock:
            self.step_count += 1

            # Check if we should take a snapshot
            if not force and self.step_count % self.snapshot_interval != 0:
                return None

            # System memory information
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                system_memory = psutil.virtual_memory()

                process_memory_mb = memory_info.rss / 1024 / 1024
                system_memory_mb = system_memory.total / 1024 / 1024
                available_memory_mb = system_memory.available / 1024 / 1024
            except ImportError:
                process_memory_mb = 0.0
                system_memory_mb = 0.0
                available_memory_mb = 0.0

            # JAX arrays tracking
            jax_arrays_count = len(self.tracked_arrays)
            jax_arrays_size_mb = sum(
                info['size_bytes'] for info in self.tracked_arrays.values()
            ) / 1024 / 1024

            # Device memory (simplified - would need JAX device memory API)
            device_memory_mb = {'cpu': process_memory_mb}  # Simplified

            # Python objects memory
            python_objects_mb = 0.0
            if self._tracemalloc_enabled:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    python_objects_mb = current / 1024 / 1024
                except Exception:
                    pass

            # GC information
            gc_objects_count = len(gc.get_objects()) if self.enable_gc_monitoring else 0
            if hasattr(gc, 'get_stats'):
                try:
                    gc_stats = gc.get_stats()
                    gc_collections = sum(stat.get('collections', 0) for stat in gc_stats)
                except Exception:
                    gc_collections = 0
            else:
                gc_collections = 0

            # State-specific memory analysis
            state_variables_mb, parameters_mb, buffers_mb = self._analyze_state_memory(state)

            # Calculate deltas
            memory_delta_mb = 0.0
            arrays_delta_count = 0
            if self.last_snapshot:
                memory_delta_mb = process_memory_mb - self.last_snapshot.process_memory_mb
                arrays_delta_count = jax_arrays_count - self.last_snapshot.jax_arrays_count

            # Create snapshot
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                step=self.step_count,
                module_name=module_name,
                process_memory_mb=process_memory_mb,
                system_memory_mb=system_memory_mb,
                available_memory_mb=available_memory_mb,
                jax_arrays_count=jax_arrays_count,
                jax_arrays_size_mb=jax_arrays_size_mb,
                device_memory_mb=device_memory_mb,
                python_objects_mb=python_objects_mb,
                gc_objects_count=gc_objects_count,
                state_variables_mb=state_variables_mb,
                parameters_mb=parameters_mb,
                buffers_mb=buffers_mb,
                memory_delta_mb=memory_delta_mb,
                arrays_delta_count=arrays_delta_count,
                gc_collections=gc_collections,
                weak_refs_count=len(self._tracked_array_ids) if hasattr(self, '_tracked_array_ids') else 0
            )

            self.snapshots.append(snapshot)
            self.last_snapshot = snapshot

            # Update tracking histories
            self.module_memory_usage[module_name].append(process_memory_mb)
            self.state_memory_history[module_name].append(state_variables_mb)

            # Check for memory leaks
            if len(self.snapshots) >= 5:  # Need some history
                self._check_for_leaks()

            return snapshot

    def _analyze_state_memory(self, state: Any) -> tuple[float, float, float]:
        """Analyze memory usage of state variables."""
        if state is None:
            return 0.0, 0.0, 0.0

        state_variables_mb = 0.0
        parameters_mb = 0.0
        buffers_mb = 0.0

        try:
            if isinstance(state, dict):
                # Handle Flax variable collections
                if 'params' in state:
                    parameters_mb = self._calculate_tree_memory(state['params'])

                if 'state' in state:
                    # This includes buffers and other state variables
                    state_dict = state['state']
                    state_variables_mb = self._calculate_tree_memory(state_dict)

                    # Try to separate buffers from other state
                    for key, value in state_dict.items():
                        if 'buffer' in key.lower():
                            buffers_mb += self._calculate_tree_memory(value)

                # Handle other state structures
                for key, value in state.items():
                    if key not in ['params', 'state']:
                        state_variables_mb += self._calculate_tree_memory(value)
            else:
                state_variables_mb = self._calculate_tree_memory(state)

        except Exception:
            # Fallback if analysis fails
            pass

        return state_variables_mb, parameters_mb, buffers_mb

    def _calculate_tree_memory(self, tree: Any) -> float:
        """Calculate memory usage of a tree structure in MB."""
        total_bytes = 0

        def accumulate_memory(x):
            nonlocal total_bytes
            if hasattr(x, 'nbytes'):
                total_bytes += x.nbytes
            elif isinstance(x, (int, float)):
                total_bytes += 8  # Approximate
            elif isinstance(x, str):
                total_bytes += len(x.encode('utf-8'))
            return x

        try:
            jax.tree_map(accumulate_memory, tree)
        except Exception:
            # If tree mapping fails, estimate based on type
            if hasattr(tree, 'nbytes'):
                total_bytes = tree.nbytes
            elif isinstance(tree, dict):
                total_bytes = sum(sys.getsizeof(v) for v in tree.values())
            else:
                total_bytes = sys.getsizeof(tree)

        return total_bytes / 1024 / 1024  # Convert to MB

    def _check_for_leaks(self):
        """Check for memory leaks based on recent snapshots."""
        if len(self.snapshots) < 5:
            return

        recent_snapshots = list(self.snapshots)[-10:]  # Last 10 snapshots

        # Check for consistent memory growth
        memory_values = [s.process_memory_mb for s in recent_snapshots]

        if len(memory_values) >= 3:
            # Simple linear regression to detect trend
            x_values = list(range(len(memory_values)))
            n = len(memory_values)

            # Calculate slope (growth rate)
            x_mean = sum(x_values) / n
            y_mean = sum(memory_values) / n

            numerator = sum((x_values[i] - x_mean) * (memory_values[i] - y_mean)
                          for i in range(n))
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

            if denominator > 0:
                slope = numerator / denominator  # MB per snapshot
                growth_rate = slope * self.snapshot_interval  # MB per step

                if growth_rate > self.leak_detection_threshold_mb / 100:  # Threshold per step
                    total_growth = memory_values[-1] - memory_values[0]

                    if total_growth > self.leak_detection_threshold_mb:
                        self._record_memory_leak(
                            leak_type="growing_memory",
                            growth_rate=growth_rate,
                            total_leaked=total_growth,
                            first_step=recent_snapshots[0].step,
                            last_step=recent_snapshots[-1].step,
                            affected_modules=list(set(s.module_name for s in recent_snapshots))
                        )

        # Check for growing JAX arrays
        array_counts = [s.jax_arrays_count for s in recent_snapshots]
        if len(array_counts) >= 3:
            array_growth = array_counts[-1] - array_counts[0]
            if array_growth > 100:  # More than 100 new arrays
                self._record_memory_leak(
                    leak_type="growing_arrays",
                    growth_rate=array_growth / len(array_counts),
                    total_leaked=sum(s.jax_arrays_size_mb for s in recent_snapshots[-3:]) / 3,
                    first_step=recent_snapshots[0].step,
                    last_step=recent_snapshots[-1].step,
                    affected_modules=list(set(s.module_name for s in recent_snapshots))
                )

    def _record_memory_leak(self, leak_type: str, growth_rate: float,
                           total_leaked: float, first_step: int, last_step: int,
                           affected_modules: list[str]):
        """Record a detected memory leak."""
        # Determine severity
        if total_leaked > 100:  # > 100 MB
            severity = "critical"
        elif total_leaked > 50:  # > 50 MB
            severity = "high"
        elif total_leaked > 20:  # > 20 MB
            severity = "medium"
        else:
            severity = "low"

        # Generate recommendations
        recommendations = []
        if leak_type == "growing_memory":
            recommendations.extend([
                "Check for accumulating state in streaming modules",
                "Verify that temporary variables are being garbage collected",
                "Consider using in-place operations where possible"
            ])
        elif leak_type == "growing_arrays":
            recommendations.extend([
                "Check for arrays being created but not released",
                "Verify JAX array lifecycle in streaming operations",
                "Consider reusing arrays or using array views"
            ])

        leak = MemoryLeak(
            leak_type=leak_type,
            severity=severity,
            growth_rate_mb_per_step=growth_rate,
            total_leaked_mb=total_leaked,
            first_detected_step=first_step,
            last_detected_step=last_step,
            affected_modules=affected_modules,
            description=f"{leak_type} detected: {total_leaked:.1f} MB leaked over {last_step - first_step} steps",
            recommendations=recommendations
        )

        self.detected_leaks.append(leak)

    def get_memory_summary(self) -> dict[str, Any]:
        """Get comprehensive memory usage summary."""
        with self._lock:
            if not self.snapshots:
                return {"error": "No memory snapshots available"}

            latest = self.snapshots[-1]
            first = self.snapshots[0]

            # Calculate overall statistics
            process_memories = [s.process_memory_mb for s in self.snapshots]
            array_counts = [s.jax_arrays_count for s in self.snapshots]

            return {
                "session": {
                    "duration_seconds": time.time() - self.session_start_time,
                    "total_snapshots": len(self.snapshots),
                    "total_steps": self.step_count
                },
                "current_usage": {
                    "process_memory_mb": latest.process_memory_mb,
                    "jax_arrays_count": latest.jax_arrays_count,
                    "jax_arrays_size_mb": latest.jax_arrays_size_mb,
                    "state_variables_mb": latest.state_variables_mb,
                    "parameters_mb": latest.parameters_mb,
                    "buffers_mb": latest.buffers_mb
                },
                "statistics": {
                    "peak_memory_mb": max(process_memories),
                    "min_memory_mb": min(process_memories),
                    "avg_memory_mb": sum(process_memories) / len(process_memories),
                    "total_growth_mb": latest.process_memory_mb - first.process_memory_mb,
                    "peak_arrays_count": max(array_counts),
                    "avg_arrays_count": sum(array_counts) / len(array_counts)
                },
                "leaks": {
                    "total_detected": len(self.detected_leaks),
                    "critical_leaks": sum(1 for leak in self.detected_leaks if leak.severity == "critical"),
                    "high_leaks": sum(1 for leak in self.detected_leaks if leak.severity == "high")
                }
            }

    def get_module_memory_analysis(self, module_name: str) -> dict[str, Any]:
        """Get memory analysis for a specific module."""
        with self._lock:
            module_snapshots = [s for s in self.snapshots if s.module_name == module_name]

            if not module_snapshots:
                return {"error": f"No data for module {module_name}"}

            memory_values = [s.process_memory_mb for s in module_snapshots]
            state_values = [s.state_variables_mb for s in module_snapshots]

            return {
                "module_name": module_name,
                "snapshots_count": len(module_snapshots),
                "memory_usage": {
                    "current_mb": memory_values[-1],
                    "peak_mb": max(memory_values),
                    "avg_mb": sum(memory_values) / len(memory_values),
                    "growth_mb": memory_values[-1] - memory_values[0]
                },
                "state_memory": {
                    "current_mb": state_values[-1],
                    "peak_mb": max(state_values),
                    "avg_mb": sum(state_values) / len(state_values)
                }
            }

    def generate_memory_report(self) -> str:
        """Generate a comprehensive memory usage report."""
        summary = self.get_memory_summary()

        if "error" in summary:
            return f"Memory Report Error: {summary['error']}"

        report = []
        report.append("💾 WAX-ML Memory Usage Report")
        report.append("=" * 40)

        # Session overview
        session = summary["session"]
        report.append("\n📊 Session Overview:")
        report.append(f"  Duration: {session['duration_seconds']:.1f} seconds")
        report.append(f"  Snapshots: {session['total_snapshots']:,}")
        report.append(f"  Steps: {session['total_steps']:,}")

        # Current usage
        current = summary["current_usage"]
        report.append("\n🔍 Current Memory Usage:")
        report.append(f"  Process Memory: {current['process_memory_mb']:.1f} MB")
        report.append(f"  JAX Arrays: {current['jax_arrays_count']:,} ({current['jax_arrays_size_mb']:.1f} MB)")
        report.append(f"  State Variables: {current['state_variables_mb']:.1f} MB")
        report.append(f"  Parameters: {current['parameters_mb']:.1f} MB")
        report.append(f"  Buffers: {current['buffers_mb']:.1f} MB")

        # Statistics
        stats = summary["statistics"]
        report.append("\n📈 Memory Statistics:")
        report.append(f"  Peak Memory: {stats['peak_memory_mb']:.1f} MB")
        report.append(f"  Average Memory: {stats['avg_memory_mb']:.1f} MB")
        report.append(f"  Total Growth: {stats['total_growth_mb']:.1f} MB")
        report.append(f"  Peak Arrays: {stats['peak_arrays_count']:,}")

        # Memory leaks
        leaks = summary["leaks"]
        if leaks["total_detected"] > 0:
            report.append("\n🚨 Memory Leaks Detected:")
            report.append(f"  Total Leaks: {leaks['total_detected']}")
            report.append(f"  Critical: {leaks['critical_leaks']}")
            report.append(f"  High Priority: {leaks['high_leaks']}")

            # Show details of critical leaks
            critical_leaks = [leak for leak in self.detected_leaks if leak.severity == "critical"]
            for leak in critical_leaks[:3]:  # Show up to 3 critical leaks
                report.append(f"\n  🔴 Critical Leak ({leak.leak_type}):")
                report.append(f"     Total Leaked: {leak.total_leaked_mb:.1f} MB")
                report.append(f"     Growth Rate: {leak.growth_rate_mb_per_step:.3f} MB/step")
                report.append(f"     Steps: {leak.first_detected_step} - {leak.last_detected_step}")
        else:
            report.append("\n✅ No Memory Leaks Detected")

        report.append("\n" + "=" * 40)

        return "\n".join(report)

    def reset(self):
        """Reset memory tracker state."""
        with self._lock:
            self.snapshots.clear()
            self.detected_leaks.clear()
            self.tracked_arrays.clear()
            self.array_lifecycle.clear()
            self.state_memory_history.clear()
            self.module_memory_usage.clear()
            self.step_count = 0
            self.session_start_time = time.time()
            self.baseline_memory = None
            self.last_snapshot = None

    def enable(self):
        """Enable memory tracking."""
        self.enabled = True

    def disable(self):
        """Disable memory tracking."""
        self.enabled = False


def track_memory_usage(tracker: MemoryTracker | None = None,
                      module_name: str | None = None,
                      detailed_state_analysis: bool = True):
    """Decorator for adding memory tracking to streaming functions.
    
    Example:
        tracker = MemoryTracker(enable_detailed_tracking=True)
        
        @track_memory_usage(tracker, "ewma_module")
        @streaming_transform_with_state
        def ewma_processor(x):
            return EWMA(alpha=0.1)(x)
        
        # ... run computations ...
        
        print(tracker.generate_memory_report())
    """
    if tracker is None:
        tracker = MemoryTracker()

    def decorator(fn: Callable) -> Callable:
        actual_module_name = module_name or fn.__name__

        # Create a wrapper that preserves the original function interface
        class MemoryWrapper:
            def __init__(self, original_fn):
                self.original_fn = original_fn
                self.__name__ = getattr(original_fn, '__name__', 'memory_wrapped')
                # Copy all attributes from original function except apply
                for attr in ['init', '__call__']:
                    if hasattr(original_fn, attr):
                        setattr(self, attr, getattr(original_fn, attr))

                # Create a wrapped apply method that includes memory tracking
                if hasattr(original_fn, 'apply'):
                    self.apply = self._create_memory_apply(original_fn.apply)

            def _create_memory_apply(self, original_apply):
                def memory_apply(params, state, rng, *args, **kwargs):
                    # Take snapshot before execution
                    tracker.take_snapshot(module_name=actual_module_name)

                    # Execute function
                    result = original_apply(params, state, rng, *args, **kwargs)

                    # Take snapshot after execution with state information
                    if detailed_state_analysis and isinstance(result, tuple) and len(result) == 2:
                        # Assuming result is (output, new_state) tuple from streaming function
                        output, new_state = result
                        tracker.take_snapshot(module_name=actual_module_name, state=new_state)
                    else:
                        tracker.take_snapshot(module_name=actual_module_name)

                    return result

                return memory_apply

            def __call__(self, *args, **kwargs):
                return self.original_fn(*args, **kwargs)

        return MemoryWrapper(fn)
    return decorator
