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
"""Core streaming transform layer for functional streaming computation.

This module provides the fundamental abstractions that bridge stateful streaming
programming with JAX/Flax's functional paradigms, similar to how Haiku's
transform functions worked.
"""

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.core import FrozenDict

from .transform import flax_transform_with_state

# Type variables for generic transforms
F = TypeVar("F", bound=Callable[..., Any])
StreamingFn = TypeVar("StreamingFn", bound=Callable[..., Any])


class StreamingStatePattern(nn.Module):
    """Base class for streaming state patterns that enable hierarchical composition.

    This provides common functionality for:
    - Conditional state updates based on events
    - Hierarchical composition of streaming modules
    - Reset and initialization patterns
    """

    def setup(self):
        """Override in subclasses to define module structure."""
        pass

    def reset_state(self, state: Any) -> Any:
        """Reset state to initial values. Override in subclasses."""
        return state

    def should_update(self, *args, **kwargs) -> bool:
        """Determine if state should be updated. Override in subclasses."""
        return True

    def conditional_call(self, state: Any, *args, **kwargs):
        """Conditionally update state based on should_update logic."""
        if self.should_update(*args, **kwargs):
            return self(*args, **kwargs)
        else:
            # Return previous output with unchanged state
            # This needs to be implemented by concrete classes
            raise NotImplementedError("Conditional call must be implemented by subclasses")


class HierarchicalState(StreamingStatePattern):
    """Pattern for hierarchical composition of streaming modules.

    Enables building complex streaming systems from simpler components
    with proper state management and composition.
    """

    modules: dict[str, nn.Module]
    dependencies: dict[str, list[str]]  # Module dependency graph

    def __init__(
        self, modules: dict[str, nn.Module], dependencies: dict[str, list[str]] | None = None
    ):
        """Initialize hierarchical state manager.

        Args:
            modules: Dictionary mapping module names to module instances
            dependencies: Optional dependency graph for execution order
        """
        super().__init__()
        self.modules = modules
        self.dependencies = dependencies or {}

    def setup(self):
        """Register all modules for proper state management."""
        for name, module in self.modules.items():
            setattr(self, f"module_{name}", module)

    def execute_in_order(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute modules in dependency order."""
        outputs: dict[str, Any] = {}

        # Simple topological execution - can be enhanced with proper sorting
        for name, module in self.modules.items():
            # Get inputs for this module
            module_inputs = self._get_module_inputs(name, inputs, outputs)

            # Execute module
            output = module(**module_inputs)
            outputs[name] = output

        return outputs

    def _get_module_inputs(
        self, module_name: str, initial_inputs: dict[str, Any], computed_outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Get inputs for a specific module based on dependencies."""
        module_inputs = {}

        # Add dependencies from other modules
        if module_name in self.dependencies:
            for dep in self.dependencies[module_name]:
                if dep in computed_outputs:
                    module_inputs[dep] = computed_outputs[dep]

        # Add initial inputs that match parameter names
        # This is a simplified approach - can be enhanced
        module_inputs.update(initial_inputs)

        return module_inputs

    def __call__(self, **kwargs):
        """Execute the hierarchical computation."""
        return self.execute_in_order(kwargs)


class ConditionalStateUpdate(StreamingStatePattern):
    """Pattern for conditional state updates based on events or predicates.

    This enables event-driven computation where modules only update
    their state when specific conditions are met.
    """

    module: nn.Module
    condition_fn: Callable[..., bool]
    reset_on_condition: bool

    def __init__(
        self, module: nn.Module, condition_fn: Callable[..., bool], reset_on_condition: bool = False
    ):
        """Initialize conditional state updater.

        Args:
            module: The module to conditionally update
            condition_fn: Function that determines when to update
            reset_on_condition: Whether to reset state when condition is met
        """
        super().__init__()
        self.module = module
        self.condition_fn = condition_fn
        self.reset_on_condition = reset_on_condition

    def setup(self):
        """Register the wrapped module."""
        self.wrapped_module = self.module

    def should_update(self, *args, **kwargs) -> bool:
        """Check if state should be updated."""
        return self.condition_fn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Conditionally call the wrapped module."""
        # Note: This is a simplified implementation
        # In practice, we need to handle state preservation
        # when condition is False

        if self.should_update(*args, **kwargs):
            if self.reset_on_condition:
                # Reset module state before computation
                # This would need to be implemented with proper state handling
                pass
            return self.wrapped_module(*args, **kwargs)
        else:
            # Return some default or previous value
            # This needs to be enhanced with proper state management
            return args[0] if args else jnp.array(0.0)


class StreamingStateMachine(StreamingStatePattern):
    """Pattern for finite state machine-based streaming computation.

    Enables building streaming systems that have distinct states
    and transition between them based on inputs and conditions.
    """

    states: dict[str, nn.Module]
    transitions: dict[str, dict[str, Callable[..., bool]]]
    initial_state: str

    def __init__(
        self,
        states: dict[str, nn.Module],
        transitions: dict[str, dict[str, Callable[..., bool]]],
        initial_state: str,
    ):
        """Initialize streaming state machine.

        Args:
            states: Dictionary mapping state names to modules
            transitions: Transition conditions between states
            initial_state: Name of the initial state
        """
        super().__init__()
        self.states = states
        self.transitions = transitions
        self.initial_state = initial_state

    def setup(self):
        """Register all state modules."""
        for name, module in self.states.items():
            setattr(self, f"state_{name}", module)

        # Initialize current state tracking
        self.current_state = self.variable("state", "current", lambda: self.initial_state)

    def get_next_state(self, current_state: str, *args, **kwargs) -> str:
        """Determine next state based on transition conditions."""
        if current_state not in self.transitions:
            return current_state

        for next_state, condition_fn in self.transitions[current_state].items():
            if condition_fn(*args, **kwargs):
                return next_state

        return current_state

    def __call__(self, *args, **kwargs):
        """Execute current state and potentially transition."""
        current = self.current_state.value

        # Execute current state module
        output = self.states[current](*args, **kwargs)

        # Check for state transition
        next_state = self.get_next_state(current, *args, **kwargs)

        # Update current state if transition occurred
        if next_state != current:
            self.current_state.value = next_state

        return output, next_state


# Streaming state pattern decorators
def hierarchical_composition(**modules) -> Callable[[Callable], "StreamingTransform"]:
    """Decorator for creating hierarchical streaming compositions.

    Example:
        @hierarchical_composition(buffer=Buffer(maxlen=5), ewma=EWMA(alpha=0.1))
        def trading_signal(price):
            # Modules are available as self.buffer, self.ewma
            buffered = self.buffer(price)
            smoothed = self.ewma(price)
            return smoothed - jnp.mean(buffered)
    """

    def decorator(fn: Callable) -> "StreamingTransform":
        @streaming_transform_with_state
        def wrapper(*args, **kwargs):
            hierarchy = HierarchicalState(modules)
            return hierarchy(**kwargs)

        return wrapper

    return decorator


def conditional_update(
    condition_fn: Callable[..., bool], reset_on_condition: bool = False
) -> Callable[[Callable], "StreamingTransform"]:
    """Decorator for conditional state updates.

    Example:
        @conditional_update(lambda x: x > threshold)
        def threshold_detector(x):
            counter = Counter()
            return counter(1)  # Increment only when x > threshold
    """

    def decorator(fn: Callable) -> "StreamingTransform":
        @streaming_transform_with_state
        def wrapper(*args, **kwargs):
            # Create a simple module that wraps the function
            class ConditionalModule(nn.Module):
                def __call__(self, *args, **kwargs):
                    return fn(*args, **kwargs)

            module = ConditionalModule()
            conditional = ConditionalStateUpdate(module, condition_fn, reset_on_condition)
            return conditional(*args, **kwargs)

        return wrapper

    return decorator


def state_machine(
    states: dict[str, Callable],
    transitions: dict[str, dict[str, Callable[..., bool]]],
    initial_state: str,
) -> Callable[[Callable], "StreamingTransform"]:
    """Decorator for finite state machine-based streaming.

    Example:
        @state_machine(
            states={'warming': warming_fn, 'trading': trading_fn},
            transitions={'warming': {'trading': lambda x: x > warm_threshold}},
            initial_state='warming'
        )
        def trading_system(price):
            pass  # Logic handled by state machine
    """

    def decorator(fn: Callable) -> "StreamingTransform":
        @streaming_transform_with_state
        def wrapper(*args, **kwargs):
            # Convert functions to modules
            state_modules = {}
            for name, state_fn in states.items():

                class StateModule(nn.Module):
                    fn: Callable = state_fn

                    def __call__(self, *args, **kwargs):
                        return self.fn(*args, **kwargs)

                state_modules[name] = StateModule()

            machine = StreamingStateMachine(state_modules, transitions, initial_state)
            return machine(*args, **kwargs)

        return wrapper

    return decorator


class StreamingTransform:
    """Enhanced transform that provides streaming-specific functionality.

    This is the core abstraction that allows writing stateful-looking streaming
    code that compiles to pure functional JAX operations.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        auto_cache: bool = True,
        event_driven: bool = False,
        scan_mode: bool = False,
        state_pattern: str | None = None,
    ):
        """Initialize streaming transform.

        Args:
            fn: Function to transform
            auto_cache: Whether to automatically cache outputs when state unchanged
            event_driven: Whether this transform supports event-driven computation
            scan_mode: Whether to use scan semantics for sequence processing
        """
        self.fn = fn
        self.auto_cache = auto_cache
        self.event_driven = event_driven
        self.scan_mode = scan_mode

        # We need to wrap the function to make it compatible with flax_transform_with_state
        # Create a module that wraps the function
        class StreamingModule(nn.Module):
            @nn.compact
            def __call__(self, *args, **kwargs):
                return fn(*args, **kwargs)

        # Build the base transform with the module
        self._base_transform = flax_transform_with_state(StreamingModule())

        # Cache for last output when auto_cache is enabled
        self._output_cache = None
        self._state_cache = None

    def init(self, rng: jax.Array, *args, **kwargs) -> tuple[FrozenDict, FrozenDict]:
        """Initialize parameters and state."""
        params, state = self._base_transform.init(rng, *args, **kwargs)
        return params, state

    def apply(
        self, params: FrozenDict, state: FrozenDict, rng: jax.Array | None, *args, **kwargs
    ) -> tuple[Any, FrozenDict]:
        """Apply the streaming function with state management.

        Args:
            params: Model parameters
            state: Current state
            rng: Random key (can be None)
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (output, new_state)
        """
        # For now, delegate to base transform
        # Event-driven and scan semantics will be added in specialized transforms
        output, new_state = self._base_transform.apply(params, state, rng, *args, **kwargs)
        return output, new_state

    def scan(
        self,
        params: FrozenDict,
        state: FrozenDict,
        rng: jax.Array | None,
        inputs: jax.Array,
        **kwargs,
    ) -> tuple[jax.Array, FrozenDict]:
        """Apply function in scan mode over a sequence of inputs.

        Args:
            params: Model parameters
            state: Initial state
            rng: Random key (can be None)
            inputs: Sequence of inputs to process
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (outputs, final_state)
        """

        def scan_fn(carry_state, x):
            output, new_state = self.apply(params, carry_state, rng, x, **kwargs)
            return new_state, output

        final_state, outputs = jax.lax.scan(scan_fn, state, inputs)
        return outputs, final_state


def streaming_transform_with_state(fn: StreamingFn) -> StreamingTransform:
    """Transform a function into a streaming-aware pure function.

    This is the core transform that enables writing stateful-looking streaming
    code that compiles to pure JAX functions, similar to hk.transform_with_state.

    Args:
        fn: Function to transform. Should contain Flax modules with state.

    Returns:
        StreamingTransform object with init, apply, and scan methods.

    Example::

        @streaming_transform_with_state
        def streaming_model(x):
            buffer = Buffer(maxlen=10)
            ewma = EWMA(alpha=0.1)
            return ewma(buffer(x))

        # Use like any other transform
        params, state = streaming_model.init(rng, x0)
        output, new_state = streaming_model.apply(params, state, None, x1)
    """
    return StreamingTransform(fn, auto_cache=True)


class ConditionalComputation(nn.Module):
    """Module for conditional computation based on events.

    This is the building block for event-driven streaming computation.
    When an event_fn is provided, the output is only updated when
    event_fn returns True; otherwise the previously cached output is returned.

    Note: The inner module's state (e.g. EWMA running average) is always
    updated regardless of the event condition. Only the *output* is
    conditionally cached. This differs from the Haiku UpdateOnEvent which
    also rolls back module state; Flax does not expose an equivalent of
    Haiku's state_dict()/set_state_from_dict() for child modules.
    """

    update_fn: Callable
    event_fn: Callable | None = None
    initial_output_value: float = jnp.nan

    @nn.compact
    def __call__(self, *args, **kwargs) -> Any:
        """Execute conditional computation.

        Args:
            *args: Arguments to pass to update_fn and event_fn
            **kwargs: Keyword arguments

        Returns:
            Output from update_fn (if event occurred) or cached output
        """
        # Always compute new output so that child module variables are created
        # on every code path (required by Flax's variable system).
        new_output = self.update_fn(*args, **kwargs)

        # Initialize cached output with correct shape/dtype, filled with
        # initial_output_value (default NaN to signal "no event yet").
        cached_output = self.variable(
            "cache",
            "prev_output",
            lambda: jax.tree_util.tree_map(
                lambda x: jnp.full(x.shape, self.initial_output_value, dtype=x.dtype),
                new_output,
            ),
        )

        if self.event_fn is None:
            # No event function → always update
            cached_output.value = new_output
            return new_output

        should_update = self.event_fn(*args, **kwargs)

        # Select between new output and cached output based on event
        selected_output = jax.tree_util.tree_map(
            lambda new, cached: jnp.where(should_update, new, cached),
            new_output,
            cached_output.value,
        )

        # Persist selected output for next call
        cached_output.value = selected_output
        return selected_output


def update_on_event(
    event_fn: Callable[..., bool] | None = None,
    initial_output_value: float = jnp.nan,
) -> Callable[[StreamingFn], Callable]:
    """Transform decorator for event-driven conditional computation.

    This decorator enables writing streaming functions that only update their
    output when specific events occur, returning the cached output otherwise.

    Args:
        event_fn: Function that determines when to update. If None, always update.
        initial_output_value: Value used to fill the output cache before the first
            event fires.  Defaults to NaN.

    Returns:
        Decorator that transforms functions into event-driven streaming functions.

    Example::

        @update_on_event(event_fn=lambda x: x > 0)  # Only update on positive values
        def conditional_model(x):
            model = EWMA(alpha=0.1)
            return model(x)

        # Then wrap with streaming_transform_with_state
        streaming_fn = streaming_transform_with_state(conditional_model)
        params, state = streaming_fn.init(rng, x0)
        output, state = streaming_fn.apply(params, state, None, x1)
    """

    def decorator(update_fn: StreamingFn) -> Callable:
        def wrapped_fn(*args, **kwargs):
            conditional = ConditionalComputation(
                update_fn=update_fn,
                event_fn=event_fn,
                initial_output_value=initial_output_value,
            )
            return conditional(*args, **kwargs)

        return wrapped_fn

    return decorator


class StreamingScan(nn.Module):
    """Module for scan operations with streaming-specific features.

    This provides JAX-compatible scan operations with reset capabilities,
    essential for streaming time-series processing with episode boundaries.
    """

    fn: Callable
    reset_fn: Callable | None = None
    preserve_state: bool = True

    @nn.compact
    def __call__(self, inputs: jax.Array, initial_state: Any = None) -> tuple[jax.Array, Any]:
        """Apply function in scan mode with optional resets.

        Args:
            inputs: Sequence of inputs to process (shape: [seq_len, ...])
            initial_state: Initial state for the scan. If None, will be inferred.

        Returns:
            Tuple of (outputs, final_state)
        """
        # Handle initial state initialization
        if initial_state is None:
            # Create a consistent initial state for scan
            initial_state = 0  # Simple scalar initial state

        # Store reference to initial state for resets
        def scan_fn(carry, x):
            """Inner scan function with reset logic."""

            # Check if we should reset state
            should_reset = False
            if self.reset_fn is not None:
                should_reset = self.reset_fn(x)

            # Reset state if needed
            if should_reset and self.preserve_state:
                # Reset to initial state
                carry = initial_state

            # Apply the streaming function
            # For now, we'll implement a simple pass-through that maintains carry structure
            output = self.fn(x)

            # Keep carry structure consistent
            new_carry = carry

            return new_carry, output

        # Use JAX scan for efficient sequence processing
        final_carry, outputs = jax.lax.scan(scan_fn, initial_state, inputs)

        return outputs, final_carry


def streaming_scan(
    fn: StreamingFn = None,
    *,
    reset_on: Callable[..., bool] | None = None,
    preserve_state: bool = True,
) -> Callable:
    """Transform decorator for streaming scan operations with reset capabilities.

    This enables efficient processing of sequences while maintaining state across
    elements, with optional reset conditions for episode boundaries or other events.

    Args:
        fn: Function to apply in scan mode (if used as decorator without parentheses)
        reset_on: Function that determines when to reset state based on input
        preserve_state: Whether to preserve and reset state (vs. reinitialize)

    Returns:
        Streaming scan transform that can be used with JAX transformations

    Example::

        # As decorator with reset condition
        @streaming_scan(reset_on=lambda x: x > 10)
        def accumulator(x):
            buffer = Buffer(maxlen=5)
            return jnp.sum(buffer(x))

        # Manual usage
        scan_fn = streaming_scan(my_function, reset_on=lambda x: x == 0)
        outputs, final_state = scan_fn(sequence_data)
    """

    def decorator(scan_fn: StreamingFn):
        """Create a streaming scan transform from the function."""

        # First transform the function into a streaming function
        streaming_fn = streaming_transform_with_state(scan_fn)

        def scan_apply(inputs, params=None, state=None, rng=None):
            """Apply scan directly to inputs."""
            if params is None or state is None:
                # Initialize if needed
                sample_input = jax.tree_util.tree_map(lambda x: x[0], inputs)
                params, state = streaming_fn.init(rng or jax.random.PRNGKey(0), sample_input)

            # Create scan function that applies streaming function element by element
            def scan_body(carry_state, x):
                # Apply streaming function to single element
                output, new_state = streaming_fn.apply(params, carry_state, rng, x)

                # Check if we should reset state (JAX-compatible)
                if reset_on is not None:
                    should_reset = reset_on(x)
                    if preserve_state:
                        # Use JAX select for conditional reset
                        new_state = jax.tree_util.tree_map(
                            lambda new_val, init_val: jax.lax.select(
                                should_reset, init_val, new_val
                            ),
                            new_state,
                            state,
                        )

                return new_state, output

            # Apply scan
            final_state, outputs = jax.lax.scan(scan_body, state, inputs)
            return outputs, final_state

        # Create a simple object with scan_apply method for compatibility
        class StreamingScanResult:
            def __init__(self):
                self.scan_apply = scan_apply
                self.init = streaming_fn.init
                self.apply = streaming_fn.apply

        return StreamingScanResult()

    # Handle both @streaming_scan and @streaming_scan(...) usage patterns
    if fn is not None:
        return decorator(fn)  # type: ignore
    return decorator


class StreamingOptimizer(nn.Module):
    """Module for streaming optimization with automatic gradient flow.

    This provides a streaming-aware optimizer that maintains optimizer state
    across time steps and computes real gradients via ``jax.value_and_grad``
    for the given loss function.

    A learnable *scale* parameter modulates the model output.  Gradients of
    ``loss_fn(model_output * scale, target)`` w.r.t. ``scale`` are computed
    and used to update the optimizer state each step.
    """

    optimizer: optax.GradientTransformation
    loss_fn: Callable
    has_aux: bool = False

    @nn.compact
    def __call__(self, model_fn: Callable, *args, **kwargs) -> tuple[tuple[Any, ...], Any]:
        """Apply streaming optimization to a model function.

        Args:
            model_fn: Model function to optimize
            *args: Arguments to pass to model and loss functions
            **kwargs: Keyword arguments

        Returns:
            Tuple of (loss_info, updated_params) where loss_info contains
            loss value and optionally auxiliary outputs
        """
        # Run model to get prediction
        if self.has_aux:
            model_output, aux = model_fn(*args, **kwargs)
        else:
            model_output = model_fn(*args, **kwargs)
            aux = None

        # Extract target (second argument for supervised learning)
        target = args[1] if len(args) > 1 else args[0]

        # Learnable scale parameter stored in a mutable collection so it
        # persists across apply calls (the "params" collection is stripped
        # by flax_transform_with_state).
        scale_var = self.variable("learnable", "scale", lambda: jnp.array(1.0))
        scale = scale_var.value

        # Define loss as a function of scale so we can differentiate
        def scale_loss(scale_param):
            scaled_output = model_output * scale_param
            return self.loss_fn(scaled_output, target)

        # Compute real gradients via jax.value_and_grad
        loss_val, grads = jax.value_and_grad(scale_loss)(scale)

        # Maintain optimizer state across time steps
        opt_state = self.variable("opt_state", "state", lambda: self.optimizer.init(scale))

        # Apply optimizer update
        updates, new_opt_state = self.optimizer.update(grads, opt_state.value, scale)
        new_scale = optax.apply_updates(scale, updates)

        # Persist updated values
        scale_var.value = new_scale
        opt_state.value = new_opt_state

        # Format output
        loss_info: tuple[Any, Any] | tuple[Any, Any, Any]
        if self.has_aux:
            loss_info = (loss_val, model_output, aux)
        else:
            loss_info = (loss_val, model_output)

        params_info = {"scale": new_scale, "grads": grads}
        return loss_info, params_info


def streaming_optimizer(
    optimizer: optax.GradientTransformation, loss_fn: Callable, *, has_aux: bool = False
) -> Callable:
    """Transform decorator for streaming optimization with automatic gradient flow.

    This enables automatic gradient computation and parameter updates for streaming
    learning scenarios, maintaining optimizer state across time steps.

    Args:
        optimizer: Optax optimizer (e.g., optax.adam(0.001))
        loss_fn: Loss function that takes (predictions, targets, ...)
        has_aux: Whether model function returns auxiliary outputs

    Returns:
        Streaming optimizer transform that can be applied to model functions

    Example::

        # Create optimized streaming model
        @streaming_optimizer(optax.adam(0.001), mse_loss)
        def online_learner(x, y):
            model = EWMA(alpha=0.1)
            prediction = model(x)
            return prediction

        # Use with streaming transform
        learner = streaming_transform_with_state(online_learner)
        params, state = learner.init(rng, x0, y0)
        (loss, pred), new_state = learner.apply(params, state, None, x1, y1)
    """

    def decorator(model_fn: StreamingFn) -> StreamingTransform:
        """Create a streaming optimized function from the model function."""

        @streaming_transform_with_state
        def optimized_fn(*args, **kwargs):
            """Optimized streaming function with automatic gradients."""
            streaming_opt = StreamingOptimizer(
                optimizer=optimizer, loss_fn=loss_fn, has_aux=has_aux
            )

            # Apply streaming optimization
            loss_info, updated_params = streaming_opt(model_fn, *args, **kwargs)

            return loss_info

        return optimized_fn

    return decorator


# Export key functions for convenient imports
__all__ = [
    "StreamingTransform",
    "streaming_transform_with_state",
    "update_on_event",
    "streaming_scan",
    "streaming_optimizer",
    "ConditionalComputation",
    "StreamingScan",
    "StreamingOptimizer",
]
