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
"""Streaming computation debugger for WAX-ML.

This module provides comprehensive debugging capabilities for streaming computations,
including state inspection, data flow tracing, and interactive debugging hooks.

Key features:
- Real-time state inspection and logging
- Conditional breakpoints based on state conditions
- Data flow visualization and tracing
- Performance bottleneck identification
- Interactive debugging sessions
- Export capabilities for post-analysis

Based on debugging patterns from:
- TensorFlow Debugger (tfdbg) architecture
- PyTorch debugging utilities
- JAX debugging and inspection tools
- Streaming systems debugging (Apache Kafka, Flink)
"""

import threading
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp


class DebugCondition(Protocol):
    """Protocol for debug condition functions."""

    def __call__(self, step: int, state: Any, input_data: Any, output: Any) -> bool:
        """Return True if debug condition is met."""
        ...


@dataclass
class DebugEvent:
    """Represents a debugging event during streaming computation."""

    step: int
    timestamp: float
    event_type: str  # 'state_change', 'breakpoint', 'error', 'warning'
    module_name: str
    state_snapshot: Any
    input_data: Any
    output_data: Any
    metadata: dict[str, Any]
    stack_trace: list[str] | None = None


class DebugHook:
    """Debug hook for monitoring streaming computations."""

    def __init__(self,
                 name: str,
                 condition: DebugCondition | None = None,
                 action: str = "log",  # 'log', 'break', 'capture', 'alert'
                 max_events: int = 1000):
        self.name = name
        self.condition = condition
        self.action = action
        self.max_events = max_events
        self.events: deque[DebugEvent] = deque(maxlen=max_events)
        self.enabled = True
        self.hit_count = 0

    def check_condition(self, step: int, state: Any, input_data: Any, output: Any) -> bool:
        """Check if the debug condition is met."""
        if not self.enabled or self.condition is None:
            return False

        try:
            return self.condition(step, state, input_data, output)
        except Exception as e:
            print(f"Warning: Debug condition error in hook '{self.name}': {e}")
            return False

    def trigger(self, step: int, module_name: str, state: Any,
               input_data: Any, output: Any, metadata: dict[str, Any] | None = None):
        """Trigger the debug hook action."""
        self.hit_count += 1

        event = DebugEvent(
            step=step,
            timestamp=time.time(),
            event_type="breakpoint" if self.action == "break" else "state_change",
            module_name=module_name,
            state_snapshot=self._deep_copy_state(state),
            input_data=self._safe_copy(input_data),
            output_data=self._safe_copy(output),
            metadata=metadata or {},
            stack_trace=traceback.format_stack() if self.action == "break" else None
        )

        self.events.append(event)

        if self.action == "log":
            self._log_event(event)
        elif self.action == "break":
            self._interactive_break(event)
        elif self.action == "alert":
            self._alert_event(event)

    def _deep_copy_state(self, state: Any) -> Any:
        """Create a deep copy of state for debugging."""
        try:
            if hasattr(state, 'copy'):
                return state.copy()
            elif isinstance(state, dict):
                return {k: self._safe_copy(v) for k, v in state.items()}
            else:
                return state
        except Exception:
            return str(state)  # Fallback to string representation

    def _safe_copy(self, data: Any) -> Any:
        """Safely copy data for event storage."""
        try:
            if isinstance(data, jnp.ndarray):
                return jnp.copy(data)
            elif isinstance(data, (list, tuple)):
                return type(data)(self._safe_copy(item) for item in data)
            elif isinstance(data, dict):
                return {k: self._safe_copy(v) for k, v in data.items()}
            else:
                return data
        except Exception:
            return str(data)

    def _log_event(self, event: DebugEvent):
        """Log debug event."""
        print(f"[DEBUG {event.step:>6}] {event.module_name}: {event.event_type}")
        if event.metadata:
            for key, value in event.metadata.items():
                print(f"  {key}: {value}")

    def _interactive_break(self, event: DebugEvent):
        """Enter interactive debugging mode."""
        print(f"\n🛑 BREAKPOINT HIT: {self.name}")
        print(f"   Step: {event.step}")
        print(f"   Module: {event.module_name}")
        print(f"   Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}")

        # Simple interactive inspection
        while True:
            try:
                cmd = input("(debug) ").strip().lower()
                if cmd in ['c', 'continue']:
                    break
                elif cmd in ['s', 'state']:
                    print(f"State: {event.state_snapshot}")
                elif cmd in ['i', 'input']:
                    print(f"Input: {event.input_data}")
                elif cmd in ['o', 'output']:
                    print(f"Output: {event.output_data}")
                elif cmd in ['h', 'help']:
                    print("Commands: (c)ontinue, (s)tate, (i)nput, (o)utput, (h)elp, (q)uit")
                elif cmd in ['q', 'quit']:
                    raise KeyboardInterrupt("Debug session terminated")
                else:
                    print(f"Unknown command: {cmd}. Type 'h' for help.")
            except EOFError:
                break
            except KeyboardInterrupt:
                raise

    def _alert_event(self, event: DebugEvent):
        """Alert for important events."""
        print(f"🚨 ALERT: {self.name} triggered at step {event.step}")


class StreamingDebugger:
    """Comprehensive debugger for streaming computations."""

    def __init__(self, enable_state_tracking: bool = True,
                 enable_performance_tracking: bool = True,
                 max_history: int = 1000):
        self.enable_state_tracking = enable_state_tracking
        self.enable_performance_tracking = enable_performance_tracking
        self.max_history = max_history

        # Debug hooks and state
        self.hooks: dict[str, DebugHook] = {}
        self.global_events: deque[DebugEvent] = deque(maxlen=max_history)
        self.state_history: deque[Any] = deque(maxlen=max_history)
        self.performance_data: dict[str, list[float]] = defaultdict(list)

        # Tracking state
        self.current_step = 0
        self.session_start_time = time.time()
        self.enabled = True

        # Thread safety
        self._lock = threading.RLock()

    def add_hook(self, hook: DebugHook) -> 'StreamingDebugger':
        """Add a debug hook."""
        with self._lock:
            self.hooks[hook.name] = hook
        return self

    def remove_hook(self, name: str) -> 'StreamingDebugger':
        """Remove a debug hook."""
        with self._lock:
            self.hooks.pop(name, None)
        return self

    def add_state_change_hook(self, name: str, condition: DebugCondition | None = None) -> 'StreamingDebugger':
        """Add hook for state changes."""
        hook = DebugHook(name=f"state_change_{name}", condition=condition, action="log")
        return self.add_hook(hook)

    def add_breakpoint(self, name: str, condition: DebugCondition) -> 'StreamingDebugger':
        """Add conditional breakpoint."""
        hook = DebugHook(name=f"breakpoint_{name}", condition=condition, action="break")
        return self.add_hook(hook)

    def add_performance_monitor(self, name: str, threshold_ms: float = 100.0) -> 'StreamingDebugger':
        """Add performance monitoring hook."""
        def slow_condition(step, state, input_data, output):
            execution_times = self.performance_data.get('execution_time', [])
            return execution_times and execution_times[-1] * 1000 > threshold_ms

        hook = DebugHook(name=f"perf_{name}", condition=slow_condition, action="alert")
        return self.add_hook(hook)

    def step(self, module_name: str, state: Any, input_data: Any, output: Any,
            execution_time: float | None = None) -> None:
        """Process a debugging step."""
        if not self.enabled:
            return

        with self._lock:
            self.current_step += 1

            # Track state history
            if self.enable_state_tracking:
                self.state_history.append(self._safe_copy_state(state))

            # Track performance data
            if self.enable_performance_tracking and execution_time is not None:
                self.performance_data['execution_time'].append(execution_time)
                self.performance_data['step'].append(self.current_step)

            # Check all hooks
            for hook in self.hooks.values():
                if hook.check_condition(self.current_step, state, input_data, output):
                    metadata = {
                        'execution_time_ms': execution_time * 1000 if execution_time else None,
                        'session_time': time.time() - self.session_start_time
                    }
                    hook.trigger(self.current_step, module_name, state, input_data, output, metadata)

    def _safe_copy_state(self, state: Any) -> Any:
        """Safely copy state for history tracking."""
        try:
            if isinstance(state, dict):
                # Handle Flax variable collections
                if 'params' in state and 'state' in state:
                    return {
                        'params_shapes': self._get_tree_shapes(state['params']),
                        'state_values': self._safe_copy_tree(state['state'])
                    }
                else:
                    return {k: self._safe_copy_tree(v) for k, v in state.items()}
            else:
                return self._safe_copy_tree(state)
        except Exception as e:
            return f"<copy_error: {str(e)}>"

    def _get_tree_shapes(self, tree: Any) -> Any:
        """Get shapes of arrays in a tree structure."""
        def shape_fn(x):
            if hasattr(x, 'shape'):
                return x.shape
            else:
                return type(x).__name__

        return jax.tree_util.tree_map(shape_fn, tree)

    def _safe_copy_tree(self, tree: Any) -> Any:
        """Safely copy a tree structure."""
        try:
            if isinstance(tree, jnp.ndarray):
                return jnp.copy(tree)
            elif isinstance(tree, dict):
                return {k: self._safe_copy_tree(v) for k, v in tree.items()}
            elif isinstance(tree, (list, tuple)):
                return type(tree)(self._safe_copy_tree(item) for item in tree)
            else:
                return tree
        except Exception:
            return str(tree)

    def get_summary(self) -> dict[str, Any]:
        """Get debugging session summary."""
        with self._lock:
            hook_stats = {
                name: {
                    'hit_count': hook.hit_count,
                    'enabled': hook.enabled,
                    'events': len(hook.events)
                }
                for name, hook in self.hooks.items()
            }

            perf_summary = {}
            if 'execution_time' in self.performance_data:
                times = self.performance_data['execution_time']
                perf_summary = {
                    'total_steps': len(times),
                    'avg_time_ms': sum(times) / len(times) * 1000 if times else 0,
                    'max_time_ms': max(times) * 1000 if times else 0,
                    'min_time_ms': min(times) * 1000 if times else 0
                }

            return {
                'session_duration': time.time() - self.session_start_time,
                'total_steps': self.current_step,
                'hooks': hook_stats,
                'performance': perf_summary,
                'state_history_length': len(self.state_history),
                'global_events': len(self.global_events)
            }

    def clear_history(self) -> None:
        """Clear all debugging history."""
        with self._lock:
            self.global_events.clear()
            self.state_history.clear()
            self.performance_data.clear()
            for hook in self.hooks.values():
                hook.events.clear()
                hook.hit_count = 0

    def enable(self) -> None:
        """Enable debugging."""
        self.enabled = True

    def disable(self) -> None:
        """Disable debugging."""
        self.enabled = False

    def export_events(self, format: str = "dict") -> Any:
        """Export debugging events for analysis."""
        with self._lock:
            if format == "dict":
                return {
                    'summary': self.get_summary(),
                    'hooks': {
                        name: [
                            {
                                'step': event.step,
                                'timestamp': event.timestamp,
                                'event_type': event.event_type,
                                'metadata': event.metadata
                            }
                            for event in hook.events
                        ]
                        for name, hook in self.hooks.items()
                    },
                    'performance_data': dict(self.performance_data)
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")


def debug_streaming(debugger: StreamingDebugger | None = None,
                   module_name: str | None = None,
                   enable_timing: bool = True):
    """Decorator for adding debugging to streaming functions.
    
    Example:
        debugger = StreamingDebugger()
        debugger.add_breakpoint("high_value", lambda step, state, inp, out: inp > 100)
        
        @debug_streaming(debugger, "my_module")
        @streaming_transform_with_state
        def my_streaming_fn(x):
            return EWMA(alpha=0.1)(x)
    """
    if debugger is None:
        debugger = StreamingDebugger()

    def decorator(fn: Callable) -> Callable:
        actual_module_name = module_name or fn.__name__

        # Create a wrapper that preserves the original function interface
        class DebugWrapper:
            def __init__(self, original_fn):
                self.original_fn = original_fn
                self.__name__ = getattr(original_fn, '__name__', 'debug_wrapped')
                # Copy all attributes from original function except apply
                for attr in ['init', '__call__']:
                    if hasattr(original_fn, attr):
                        setattr(self, attr, getattr(original_fn, attr))

                # Create a wrapped apply method that includes debugging
                if hasattr(original_fn, 'apply'):
                    self.apply = self._create_debug_apply(original_fn.apply)

            def _create_debug_apply(self, original_apply):
                def debug_apply(params, state, rng, *args, **kwargs):
                    start_time = time.time() if enable_timing else None

                    result = original_apply(params, state, rng, *args, **kwargs)

                    execution_time = (time.time() - start_time) if enable_timing else None

                    # Extract input data (first argument after params/state/rng)
                    input_data = args[0] if args else None

                    # Extract output and new state
                    if isinstance(result, tuple) and len(result) == 2:
                        output, new_state = result
                    else:
                        output = result
                        new_state = state

                    debugger.step(
                        module_name=actual_module_name,
                        state=new_state,
                        input_data=input_data,
                        output=output,
                        execution_time=execution_time
                    )

                    return result

                return debug_apply

            def __call__(self, *args, **kwargs):
                return self.original_fn(*args, **kwargs)

        return DebugWrapper(fn)
    return decorator


# Convenience functions for common debug conditions

def value_threshold(threshold: float, field: str = None) -> DebugCondition:
    """Create condition for value threshold."""
    def condition(step, state, input_data, output):
        value = input_data
        if field and isinstance(input_data, dict):
            value = input_data.get(field, 0)
        return float(value) > threshold
    return condition


def step_interval(interval: int) -> DebugCondition:
    """Create condition for step intervals."""
    def condition(step, state, input_data, output):
        return step % interval == 0
    return condition


def state_change_detector(field: str, tolerance: float = 1e-6) -> DebugCondition:
    """Create condition for significant state changes."""
    last_value = None

    def condition(step, state, input_data, output):
        nonlocal last_value

        try:
            if isinstance(state, dict) and field in state:
                current_value = state[field]
            else:
                return False

            if last_value is None:
                last_value = current_value
                return False

            if hasattr(current_value, '__len__'):
                change = jnp.linalg.norm(current_value - last_value)
            else:
                change = abs(current_value - last_value)

            significant_change = change > tolerance
            if significant_change:
                last_value = current_value

            return significant_change

        except Exception:
            return False

    return condition
