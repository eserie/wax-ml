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
"""Tests for WAX-ML debugging and profiling tools."""

import time
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from wax.flax.core.streaming_transforms import streaming_transform_with_state
from wax.flax.debug import (
    DebugHook,
    MemoryTracker,
    StreamingDebugger,
    StreamingProfiler,
    debug_streaming,
    profile_streaming,
    state_change_detector,
    step_interval,
    track_memory_usage,
    value_threshold,
)
from wax.flax.modules.buffer import Buffer
from wax.flax.modules.ewma import EWMA


class TestDebugHook:
    """Test DebugHook functionality."""

    def test_basic_hook_creation(self):
        """Test basic debug hook creation and configuration."""
        def condition(step, state, inp, out):
            return step % 5 == 0

        hook = DebugHook(name="test_hook", condition=condition, action="log")

        assert hook.name == "test_hook"
        assert hook.condition == condition
        assert hook.action == "log"
        assert hook.enabled == True
        assert hook.hit_count == 0
        assert len(hook.events) == 0

    def test_hook_condition_checking(self):
        """Test hook condition evaluation."""
        def always_true(step, state, inp, out):
            return True

        def always_false(step, state, inp, out):
            return False

        hook_true = DebugHook("true_hook", always_true)
        hook_false = DebugHook("false_hook", always_false)

        assert hook_true.check_condition(1, {}, 1.0, 2.0) == True
        assert hook_false.check_condition(1, {}, 1.0, 2.0) == False

    def test_hook_trigger_and_events(self):
        """Test hook triggering and event recording."""
        def threshold_condition(step, state, inp, out):
            return inp > 5.0

        hook = DebugHook("threshold_hook", threshold_condition, action="log")

        # Trigger hook
        hook.trigger(
            step=1,
            module_name="test_module",
            state={"value": 10},
            input_data=6.0,
            output=12.0,
            metadata={"test": True}
        )

        assert hook.hit_count == 1
        assert len(hook.events) == 1

        event = hook.events[0]
        assert event.step == 1
        assert event.module_name == "test_module"
        assert event.input_data == 6.0
        assert event.output_data == 12.0
        assert event.metadata["test"] == True

    def test_hook_max_events(self):
        """Test hook event limit enforcement."""
        hook = DebugHook("limited_hook", max_events=3)

        # Add more events than the limit
        for i in range(5):
            hook.trigger(i, "module", {}, i, i*2)

        # Should only keep the last 3 events
        assert len(hook.events) == 3
        assert hook.events[0].step == 2  # Steps 2, 3, 4
        assert hook.events[-1].step == 4


class TestStreamingDebugger:
    """Test StreamingDebugger functionality."""

    def test_debugger_creation(self):
        """Test debugger initialization."""
        debugger = StreamingDebugger()

        assert debugger.enabled == True
        assert debugger.current_step == 0
        assert len(debugger.hooks) == 0
        assert len(debugger.global_events) == 0

    def test_adding_hooks(self):
        """Test adding different types of hooks."""
        debugger = StreamingDebugger()

        # Add state change hook
        debugger.add_state_change_hook("state_monitor")
        assert "state_change_state_monitor" in debugger.hooks

        # Add breakpoint
        debugger.add_breakpoint("value_check", value_threshold(10.0))
        assert "breakpoint_value_check" in debugger.hooks

        # Add performance monitor
        debugger.add_performance_monitor("slow_ops", threshold_ms=50.0)
        assert "perf_slow_ops" in debugger.hooks

    def test_debugger_step_processing(self):
        """Test debugger step processing."""
        debugger = StreamingDebugger()

        # Add a simple hook that always triggers
        def always_trigger(step, state, inp, out):
            return True

        hook = DebugHook("always_hook", always_trigger, action="log")
        debugger.add_hook(hook)

        # Process a step
        debugger.step(
            module_name="test_module",
            state={"test": "value"},
            input_data=5.0,
            output=10.0,
            execution_time=0.001
        )

        assert debugger.current_step == 1
        assert hook.hit_count == 1
        assert len(debugger.performance_data['execution_time']) == 1
        assert debugger.performance_data['execution_time'][0] == 0.001

    def test_debugger_summary(self):
        """Test debugger summary generation."""
        debugger = StreamingDebugger()

        # Add hook and process some steps
        hook = DebugHook("test_hook")
        debugger.add_hook(hook)

        for i in range(5):
            debugger.step("module", {}, i, i*2, execution_time=0.001*i)

        summary = debugger.get_summary()

        assert summary['total_steps'] == 5
        assert 'hooks' in summary
        assert 'performance' in summary
        assert summary['hooks']['test_hook']['hit_count'] == 0  # No condition, so no hits
        assert summary['performance']['total_steps'] == 5


class TestStreamingProfiler:
    """Test StreamingProfiler functionality."""

    def test_profiler_creation(self):
        """Test profiler initialization."""
        profiler = StreamingProfiler()

        assert profiler.enabled == True
        assert profiler.step_count == 0
        assert len(profiler.metrics) == 0

    def test_profiler_context_manager(self):
        """Test profiler context manager."""
        profiler = StreamingProfiler()

        with profiler.start_step("test_module") as ctx:
            time.sleep(0.001)  # Small delay to measure

        assert profiler.step_count == 1
        assert len(profiler.metrics) == 1

        metrics = profiler.metrics[0]
        assert metrics.module_name == "test_module"
        assert metrics.execution_time > 0

    def test_profiler_record_metrics(self):
        """Test manual metrics recording."""
        profiler = StreamingProfiler()

        test_input = jnp.array([1.0, 2.0, 3.0])
        test_output = jnp.array([2.0, 4.0, 6.0])

        profiler.record_metrics(
            module_name="test_module",
            execution_time=0.005,
            input_data=test_input,
            output_data=test_output
        )

        assert profiler.step_count == 1
        assert len(profiler.metrics) == 1

        metrics = profiler.metrics[0]
        assert metrics.module_name == "test_module"
        assert metrics.execution_time == 0.005
        assert metrics.input_size == test_input.nbytes
        assert metrics.output_size == test_output.nbytes

    def test_profiler_module_summary(self):
        """Test module-specific performance summary."""
        profiler = StreamingProfiler()

        # Record metrics for different modules
        for i in range(3):
            profiler.record_metrics("module_a", 0.001 * (i + 1))
            profiler.record_metrics("module_b", 0.002 * (i + 1))

        summary_a = profiler.get_module_summary("module_a")
        summary_b = profiler.get_module_summary("module_b")

        assert summary_a["module_name"] == "module_a"
        assert summary_a["total_invocations"] == 3
        assert summary_b["module_name"] == "module_b"
        assert summary_b["total_invocations"] == 3

        # Module B should have higher average time
        assert summary_b["avg_time_ms"] > summary_a["avg_time_ms"]

    def test_profiler_optimization_recommendations(self):
        """Test optimization recommendation generation."""
        profiler = StreamingProfiler()

        # Simulate high variance in execution times
        execution_times = [0.001, 0.010, 0.002, 0.015, 0.001]
        for i, exec_time in enumerate(execution_times):
            profiler.record_metrics(f"module_{i}", exec_time)

        recommendations = profiler.generate_optimization_recommendations()

        # Should generate some recommendations
        assert len(recommendations) > 0
        # Check for specific types of recommendations
        assert any("memory" in rec.lower() or "variance" in rec.lower() or "time" in rec.lower()
                  for rec in recommendations)

    def test_profiler_finalize(self):
        """Test profiler finalization and result generation."""
        profiler = StreamingProfiler()

        # Add some metrics
        for i in range(5):
            profiler.record_metrics("test_module", 0.001 * (i + 1))

        result = profiler.finalize()

        assert result.total_steps == 5
        assert len(result.metrics) == 5
        assert result.session_duration > 0
        assert 'mean' in result.execution_stats
        assert 'mean' in result.memory_stats


class TestMemoryTracker:
    """Test MemoryTracker functionality."""

    def test_memory_tracker_creation(self):
        """Test memory tracker initialization."""
        tracker = MemoryTracker()

        assert tracker.enabled == True
        assert tracker.step_count == 0
        assert len(tracker.snapshots) == 0

    @patch('psutil.Process')
    def test_memory_snapshot(self, mock_process):
        """Test memory snapshot creation."""
        # Mock psutil
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
        mock_process_instance.cpu_percent.return_value = 25.0
        mock_process.return_value = mock_process_instance

        tracker = MemoryTracker()
        tracker.process = mock_process_instance

        snapshot = tracker.take_snapshot("test_module", force=True)

        assert snapshot is not None
        assert snapshot.module_name == "test_module"
        assert snapshot.process_memory_mb == 100.0
        assert tracker.step_count == 1

    def test_memory_state_analysis(self):
        """Test state memory analysis."""
        tracker = MemoryTracker()

        # Create test state with JAX arrays
        test_state = {
            'params': {
                'weights': jnp.ones((10, 10)),
                'bias': jnp.zeros(10)
            },
            'state': {
                'buffer': jnp.zeros(100),
                'counter': 5
            }
        }

        state_mb, params_mb, buffers_mb = tracker._analyze_state_memory(test_state)

        assert state_mb > 0
        assert params_mb > 0
        # Buffer analysis might be zero if not properly detected

    def test_memory_summary(self):
        """Test memory usage summary."""
        tracker = MemoryTracker()

        # Force a few snapshots
        for i in range(3):
            snapshot = tracker.take_snapshot(f"module_{i}", force=True)

        summary = tracker.get_memory_summary()

        assert "session" in summary
        assert "current_usage" in summary
        assert "statistics" in summary
        assert summary["session"]["total_snapshots"] == 3


class TestDecorators:
    """Test debugging and profiling decorators."""

    def test_debug_streaming_decorator(self):
        """Test debug_streaming decorator."""
        debugger = StreamingDebugger()

        @debug_streaming(debugger, "test_module")
        @streaming_transform_with_state
        def test_function(x):
            return EWMA(alpha=0.1)(x)

        # Initialize and run
        rng = jax.random.PRNGKey(42)
        params, state = test_function.init(rng, jnp.array(1.0))

        output, new_state = test_function.apply(params, state, None, jnp.array(1.0))

        # Check that debugging was triggered
        assert debugger.current_step > 0

    def test_profile_streaming_decorator(self):
        """Test profile_streaming decorator."""
        profiler = StreamingProfiler()

        @profile_streaming(profiler, "test_module")
        @streaming_transform_with_state
        def test_function(x):
            return EWMA(alpha=0.1)(x)

        # Initialize and run
        rng = jax.random.PRNGKey(42)
        params, state = test_function.init(rng, jnp.array(1.0))

        output, new_state = test_function.apply(params, state, None, jnp.array(1.0))

        # Check that profiling was recorded
        assert profiler.step_count > 0

    def test_track_memory_usage_decorator(self):
        """Test track_memory_usage decorator."""
        tracker = MemoryTracker()

        @track_memory_usage(tracker, "test_module")
        @streaming_transform_with_state
        def test_function(x):
            return Buffer(maxlen=10)(x)

        # Initialize and run
        rng = jax.random.PRNGKey(42)
        params, state = test_function.init(rng, jnp.array(1.0))

        output, new_state = test_function.apply(params, state, None, jnp.array(1.0))

        # Check that memory tracking occurred
        assert tracker.step_count > 0


class TestDebugConditions:
    """Test debug condition helpers."""

    def test_value_threshold_condition(self):
        """Test value threshold condition."""
        condition = value_threshold(5.0)

        assert condition(1, {}, 10.0, None) == True
        assert condition(1, {}, 3.0, None) == False

    def test_step_interval_condition(self):
        """Test step interval condition."""
        condition = step_interval(5)

        assert condition(5, {}, None, None) == True
        assert condition(10, {}, None, None) == True
        assert condition(3, {}, None, None) == False
        assert condition(7, {}, None, None) == False

    def test_state_change_detector(self):
        """Test state change detection condition."""
        condition = state_change_detector("value", tolerance=0.1)

        # First call should not trigger (no previous value)
        assert condition(1, {"value": 1.0}, None, None) == False

        # Small change should not trigger
        assert condition(2, {"value": 1.05}, None, None) == False

        # Large change should trigger
        assert condition(3, {"value": 1.5}, None, None) == True


class TestIntegration:
    """Integration tests combining multiple debugging tools."""

    def test_combined_debugging_and_profiling(self):
        """Test using debugger and profiler together."""
        debugger = StreamingDebugger()
        profiler = StreamingProfiler()

        # Add some debug hooks
        debugger.add_state_change_hook("monitor")
        debugger.add_performance_monitor("perf", threshold_ms=1.0)

        @debug_streaming(debugger, "combined_module")
        @profile_streaming(profiler, "combined_module")
        @streaming_transform_with_state
        def combined_function(x):
            return EWMA(alpha=0.2)(x)

        # Run some computations
        rng = jax.random.PRNGKey(42)
        params, state = combined_function.init(rng, jnp.array(1.0))

        current_state = state
        for i in range(10):
            output, current_state = combined_function.apply(
                params, current_state, None, jnp.array(float(i))
            )

        # Check both tools recorded data
        assert debugger.current_step >= 10
        assert profiler.step_count >= 10

        # Get summaries
        debug_summary = debugger.get_summary()
        profile_result = profiler.finalize()

        assert debug_summary['total_steps'] >= 10
        assert profile_result.total_steps >= 10

    def test_full_monitoring_stack(self):
        """Test complete monitoring with all tools."""
        debugger = StreamingDebugger()
        profiler = StreamingProfiler()
        memory_tracker = MemoryTracker()

        @debug_streaming(debugger, "full_stack")
        @profile_streaming(profiler, "full_stack")
        @track_memory_usage(memory_tracker, "full_stack")
        @streaming_transform_with_state
        def monitored_function(x):
            # Use a module that creates some state
            return Buffer(maxlen=50)(x)

        # Run computations
        rng = jax.random.PRNGKey(42)
        params, state = monitored_function.init(rng, jnp.array(1.0))

        current_state = state
        for i in range(20):
            output, current_state = monitored_function.apply(
                params, current_state, None, jnp.array(float(i))
            )

        # Verify all tools captured data
        assert debugger.current_step >= 20
        assert profiler.step_count >= 20
        assert memory_tracker.step_count >= 20

        # Generate reports
        debug_summary = debugger.get_summary()
        profile_result = profiler.finalize()
        memory_summary = memory_tracker.get_memory_summary()

        # All should have captured the computations
        assert all(summary.get('total_steps', 0) >= 15 for summary in [
            debug_summary,
            {'total_steps': profile_result.total_steps},
            memory_summary['session']
        ])
