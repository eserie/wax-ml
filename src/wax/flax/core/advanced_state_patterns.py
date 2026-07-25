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
"""Advanced state patterns for sophisticated streaming computation.

This module provides advanced state management patterns including:
- Hierarchical state machines with coordinated multi-level states
- Attention-based historical state selection
- Compositional state patterns with dependency management
- Dynamic state routing and coordination
"""

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import jax.numpy as jnp
from flax import linen as nn

from .streaming_transforms import StreamingTransform, streaming_transform_with_state

# Type variables
StateModule = TypeVar("StateModule", bound=nn.Module)
StateName = str
StateValue = Any


class HierarchicalStateMachine(nn.Module):
    """Hierarchical state machine for coordinated multi-level state management.

    This enables building systems with multiple interacting state machines,
    where higher-level states influence lower-level state transitions.

    Example:
        market_regime = MarketRegimeDetector()
        volatility_regime = VolatilityRegimeDetector()

        hsm = HierarchicalStateMachine(
            state_modules={'market': market_regime, 'volatility': volatility_regime},
            dependencies={'volatility': ['market']}
        )
    """

    state_modules: Mapping[StateName, nn.Module]
    dependencies: Mapping[StateName, list[StateName]] = None
    coordination_strategy: str = "sequential"  # or "parallel", "hierarchical"

    def setup(self):
        """Register all state modules and coordination mechanisms."""
        # Register state modules
        for name, module in self.state_modules.items():
            setattr(self, f"state_{name}", module)

        # Initialize coordination state (JAX-compatible)
        # Use numeric indices instead of strings for JAX compatibility
        state_names = list(self.state_modules.keys())
        num_states = len(state_names)
        self.coordination_state = self.variable(
            'state', 'coordination',
            lambda: {'num_active_states': num_states,
                    'state_priorities': jnp.ones(num_states),
                    'last_updates': jnp.zeros(num_states, dtype=jnp.int32)}
        )
        # Store the state name mapping as a class attribute (not in state)
        self.state_name_list = state_names

    def get_execution_order(self, active_states: list[StateName]) -> list[StateName]:
        """Determine execution order based on dependencies."""
        # Handle None dependencies
        dependencies = self.dependencies or {}

        # Topological sort considering dependencies
        ordered = []
        remaining = set(active_states)

        while remaining:
            # Find states with no unfulfilled dependencies
            ready = []
            for state in remaining:
                deps = dependencies.get(state, [])
                if all(dep in ordered or dep not in remaining for dep in deps):
                    ready.append(state)

            if not ready:
                # Circular dependency or invalid - use arbitrary order
                ready = [next(iter(remaining))]

            # Add ready states to execution order
            ordered.extend(ready)
            remaining -= set(ready)

        return ordered

    def coordinate_states(self, state_outputs: dict[StateName, Any],
                         coordination_info: dict) -> dict[StateName, Any]:
        """Coordinate state outputs and handle interactions."""
        if self.coordination_strategy == "sequential":
            # Sequential coordination - later states can see earlier outputs
            return state_outputs

        elif self.coordination_strategy == "hierarchical":
            # Hierarchical coordination - higher-level states influence lower-level
            coordinated: dict[StateName, Any] = {}

            # Determine hierarchy levels based on dependencies
            levels = self._get_hierarchy_levels()

            for level in sorted(levels.keys()):
                level_states = levels[level]
                for state_name in level_states:
                    if state_name in state_outputs:
                        output = state_outputs[state_name]

                        # Apply influence from higher-level states
                        for higher_state in self._get_higher_level_states(state_name, levels):
                            if higher_state in coordinated:
                                output = self._apply_hierarchical_influence(
                                    output, coordinated[higher_state], state_name, higher_state
                                )

                        coordinated[state_name] = output

            return coordinated

        else:  # parallel
            # Parallel coordination - all states updated independently
            return state_outputs

    def _get_hierarchy_levels(self) -> dict[int, list[StateName]]:
        """Compute hierarchy levels based on dependency depth."""
        dependencies = self.dependencies or {}
        levels: dict[int, list[StateName]] = {}

        def get_depth(state_name: StateName, visited: set) -> int:
            if state_name in visited:
                return 0  # Circular dependency
            visited.add(state_name)

            deps = dependencies.get(state_name, [])
            if not deps:
                return 0

            max_dep_depth = max(get_depth(dep, visited.copy()) for dep in deps)
            return max_dep_depth + 1

        for state_name in self.state_modules.keys():
            depth = get_depth(state_name, set())
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(state_name)

        return levels

    def _get_higher_level_states(self, state_name: StateName,
                                levels: dict[int, list[StateName]]) -> list[StateName]:
        """Get states at higher hierarchy levels."""
        current_level = None
        for level, states in levels.items():
            if state_name in states:
                current_level = level
                break

        higher_states = []
        if current_level is not None:
            for level in range(current_level):
                higher_states.extend(levels.get(level, []))

        return higher_states

    def _apply_hierarchical_influence(self, lower_output: Any, higher_output: Any,
                                    lower_state: StateName, higher_state: StateName) -> Any:
        """Apply influence from higher-level state to lower-level state."""
        # Simple influence mechanism - can be made more sophisticated
        if isinstance(lower_output, dict) and isinstance(higher_output, dict):
            # Apply influence factors if both outputs are dictionaries
            influence_factor = 0.1  # Configurable influence strength

            influenced = lower_output.copy()
            for key in influenced:
                if key in higher_output and isinstance(influenced[key], int | float | jnp.ndarray):
                    # Blend with higher-level signal
                    influenced[key] = (1 - influence_factor) * influenced[key] + \
                                    influence_factor * higher_output.get(key, 0)

            return influenced
        else:
            # Default: no influence for non-dict outputs
            return lower_output

    def __call__(self, *args, **kwargs) -> dict[StateName, Any]:
        """Execute hierarchical state machine."""
        # Get coordination state (JAX-compatible)
        coord_state = self.coordination_state.value
        active_states = self.state_name_list  # Use all registered states

        # Determine execution order
        execution_order = self.get_execution_order(active_states)

        # Execute state modules in order
        state_outputs = {}

        for state_name in execution_order:
            if state_name in self.state_modules:
                state_module = getattr(self, f"state_{state_name}")

                # Execute state module with original arguments only
                try:
                    output = state_module(*args, **kwargs)
                    state_outputs[state_name] = output

                except Exception as e:
                    # Handle state execution errors gracefully
                    state_outputs[state_name] = {"error": str(e), "status": "failed"}

        # Coordinate state outputs
        coordinated_outputs = self.coordinate_states(state_outputs, coord_state)

        # Update coordination state (JAX-compatible)
        new_last_updates = coord_state['last_updates'] + 1  # Increment all counters
        new_coord_state = {
            'num_active_states': coord_state['num_active_states'],
            'state_priorities': coord_state['state_priorities'],
            'last_updates': new_last_updates
        }
        self.coordination_state.value = new_coord_state

        return coordinated_outputs


class AttentionBasedStateSelector(nn.Module):
    """Attention mechanism for selecting relevant historical states.

    This module learns to attend to the most relevant historical states
    for making current decisions, enabling adaptive use of long-term memory.
    """

    embed_dim: int = 64
    num_heads: int = 4
    max_history_length: int = 100

    def setup(self):
        """Initialize attention mechanism and state storage."""
        # Multi-head attention for state selection
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim
        )

        # State embedding layers
        self.state_encoder = nn.Dense(self.embed_dim)
        self.query_encoder = nn.Dense(self.embed_dim)

        # Historical state buffer
        self.state_buffer = self.variable(
            'state', 'history',
            lambda: jnp.zeros((self.max_history_length, self.embed_dim))
        )

        self.buffer_pointer = self.variable('state', 'pointer', lambda: 0)
        self.buffer_size = self.variable('state', 'size', lambda: 0)

    def add_to_history(self, state_embedding: jnp.ndarray):
        """Add new state to circular buffer."""
        pointer = self.buffer_pointer.value
        size = self.buffer_size.value

        # Update buffer
        buffer = self.state_buffer.value
        buffer = buffer.at[pointer].set(state_embedding)
        self.state_buffer.value = buffer

        # Update pointer and size
        self.buffer_pointer.value = (pointer + 1) % self.max_history_length
        self.buffer_size.value = min(size + 1, self.max_history_length)

    def get_relevant_states(self, current_state: Any,
                           top_k: int = 5) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get most relevant historical states using attention."""
        # Encode current state as query
        if isinstance(current_state, dict):
            # Flatten dictionary state to vector
            state_vector = jnp.concatenate([
                jnp.atleast_1d(v) if isinstance(v, int | float | jnp.ndarray) else jnp.array([0.0])
                for v in current_state.values()
            ])
        else:
            state_vector = jnp.atleast_1d(current_state)

        # Pad or truncate to fixed size
        if len(state_vector) > self.embed_dim:
            state_vector = state_vector[:self.embed_dim]
        else:
            state_vector = jnp.pad(state_vector, (0, self.embed_dim - len(state_vector)))

        query = self.query_encoder(state_vector[None, None, :])  # [1, 1, embed_dim]

        # Get historical states as keys/values
        buffer_size = self.buffer_size.value
        if buffer_size == 0:
            # No history yet
            return jnp.zeros((1, self.embed_dim)), jnp.array([0.0])

        # Get valid history
        buffer = self.state_buffer.value
        if buffer_size < self.max_history_length:
            valid_history = buffer[:buffer_size]
        else:
            # Handle circular buffer
            pointer = self.buffer_pointer.value
            valid_history = jnp.concatenate([
                buffer[pointer:],
                buffer[:pointer]
            ])

        keys = values = valid_history[None, :, :]  # [1, history_len, embed_dim]

        # Apply attention
        self.attention(query, keys, values)

        # For top-k selection, use uniform weights (simplified approach)
        # In a real implementation, you could extract actual attention weights
        weights = jnp.ones(valid_history.shape[0]) / valid_history.shape[0]
        top_indices = jnp.arange(min(top_k, valid_history.shape[0]))

        relevant_states = valid_history[top_indices]
        relevant_weights = weights[top_indices]

        return relevant_states, relevant_weights

    def __call__(self, current_state: Any, use_attention: bool = True) -> dict[str, Any]:
        """Process current state with attention to historical context."""
        # Encode current state
        if isinstance(current_state, dict):
            state_vector = jnp.concatenate([
                jnp.atleast_1d(v) if isinstance(v, int | float | jnp.ndarray) else jnp.array([0.0])
                for v in current_state.values()
            ])
        else:
            state_vector = jnp.atleast_1d(current_state)

        # Pad or truncate
        if len(state_vector) > self.embed_dim:
            state_vector = state_vector[:self.embed_dim]
        else:
            state_vector = jnp.pad(state_vector, (0, self.embed_dim - len(state_vector)))

        state_embedding = self.state_encoder(state_vector)

        # Always initialize attention components (even if not used immediately)
        # This ensures parameters are available for later use
        dummy_query = self.query_encoder(state_embedding[None, None, :])  # [1, 1, embed_dim]

        # Check if we have history to use for attention
        buffer_size = self.buffer_size.value
        if use_attention and buffer_size > 1:  # Need at least 2 states for meaningful attention
            # Get valid history (excluding current state)
            buffer = self.state_buffer.value
            if buffer_size < self.max_history_length:
                valid_history = buffer[:buffer_size]
            else:
                # Handle circular buffer
                pointer = self.buffer_pointer.value
                valid_history = jnp.concatenate([
                    buffer[pointer:],
                    buffer[:pointer]
                ])

            # Use the actual query for attention
            query = dummy_query  # Already computed above
            keys = values = valid_history[None, :, :]  # [1, history_len, embed_dim]

            # Apply attention (Flax MultiHeadDotProductAttention API)
            attended_states = self.attention(query, keys, values)

            # Use attended states directly (shape: [1, 1, embed_dim])
            context = attended_states[0, 0, :]  # Extract the attended representation
            enhanced_state = state_embedding + 0.1 * context  # Additive context

            # For outputs, use the valid history and uniform weights (simplified)
            relevant_states = valid_history
            relevant_weights = jnp.ones(valid_history.shape[0]) / valid_history.shape[0]
        else:
            # No attention when buffer is empty or disabled
            # But we still need to initialize attention to ensure parameters exist
            dummy_keys = dummy_values = jnp.zeros((1, 1, self.embed_dim))
            _ = self.attention(dummy_query, dummy_keys, dummy_values)

            relevant_states = jnp.zeros((1, self.embed_dim))
            relevant_weights = jnp.array([0.0])
            enhanced_state = state_embedding

        # Add current state to history (after processing)
        self.add_to_history(state_embedding)

        return {
            "enhanced_state": enhanced_state,
            "original_state": state_embedding,
            "relevant_history": relevant_states,
            "attention_weights": relevant_weights,
            "current_state": current_state
        }


class CompositeStateManager(nn.Module):
    """Utility for composing multiple state patterns with dependency management.

    This provides a framework for building complex state systems by composing
    simpler state patterns with automatic dependency resolution and coordination.
    """

    state_patterns: Mapping[str, nn.Module]
    composition_strategy: str = "pipeline"  # or "parallel", "hierarchical"

    def setup(self):
        """Register all state pattern modules."""
        for name, pattern in self.state_patterns.items():
            setattr(self, f"pattern_{name}", pattern)

    def __call__(self, input_data: Any) -> dict[str, Any]:
        """Execute composite state patterns."""
        if self.composition_strategy == "pipeline":
            # Sequential pipeline: output of one feeds into next
            current_data = input_data
            outputs = {}

            for name, _pattern in self.state_patterns.items():
                pattern_module = getattr(self, f"pattern_{name}")
                output = pattern_module(current_data)
                outputs[name] = output

                # Use output as input for next pattern (if it's a dict, merge)
                if isinstance(output, dict) and isinstance(current_data, dict):
                    current_data = {**current_data, **output}
                else:
                    current_data = output

            return outputs

        elif self.composition_strategy == "parallel":
            # Parallel execution: all patterns process same input
            outputs = {}

            for name, _pattern in self.state_patterns.items():
                pattern_module = getattr(self, f"pattern_{name}")
                outputs[name] = pattern_module(input_data)

            return outputs

        elif self.composition_strategy == "hierarchical":
            # Hierarchical: patterns at different levels
            # This is a simplified version - could be more sophisticated
            outputs = {}
            processed_data = input_data

            for name, _pattern in self.state_patterns.items():
                pattern_module = getattr(self, f"pattern_{name}")
                output = pattern_module(processed_data)
                outputs[name] = output

                # Higher-level patterns influence lower-level data
                if isinstance(output, dict) and "influence" in output:
                    processed_data = output["influence"]

            return outputs
        else:
            raise ValueError(f"Unknown composition strategy: {self.composition_strategy}")


# Decorator functions for convenient usage

def streaming_state_machine(state_modules: Mapping[StateName, nn.Module],
                           dependencies: Mapping[StateName, list[StateName]] | None = None,
                           coordination_strategy: str = "sequential"):
    """Decorator for creating hierarchical state machines.

    Example:
        @streaming_state_machine({
            'market': MarketRegimeDetector(),
            'volatility': VolatilityRegimeDetector()
        }, dependencies={'volatility': ['market']})
        def multi_regime_trading_system(price, volume):
            # State machine handles coordination automatically
            pass
    """
    def decorator(fn: Callable) -> StreamingTransform:
        @streaming_transform_with_state
        def wrapper(*args, **kwargs):
            hsm = HierarchicalStateMachine(
                state_modules, dependencies, coordination_strategy
            )
            state_outputs = hsm(*args, **kwargs)

            # Call original function with state outputs
            return fn(state_outputs, *args, **kwargs)
        return wrapper
    return decorator


def streaming_attention_state(embed_dim: int = 64,
                             num_heads: int = 4,
                             max_history: int = 100):
    """Decorator for attention-based state selection.

    Example:
        @streaming_attention_state(embed_dim=128, max_history=200)
        def adaptive_context_processor(x):
            # Function receives enhanced state with attention context
            pass
    """
    def decorator(fn: Callable) -> StreamingTransform:
        @streaming_transform_with_state
        def wrapper(*args, **kwargs):
            attention_selector = AttentionBasedStateSelector(
                embed_dim=embed_dim,
                num_heads=num_heads,
                max_history_length=max_history
            )

            # Use first argument as current state
            current_state = args[0] if args else kwargs
            attention_output = attention_selector(current_state)

            # Call original function with attention-enhanced context
            return fn(attention_output, *args, **kwargs)
        return wrapper
    return decorator


def streaming_compose_states(*state_modules: nn.Module,
                           strategy: str = "pipeline"):
    """Decorator for composing multiple state patterns.

    Example:
        @streaming_compose_states(
            HierarchicalStateMachine(...),
            AttentionBasedStateSelector(...),
            strategy="pipeline"
        )
        def complex_state_system(data):
            # All state patterns are composed automatically
            pass
    """
    def decorator(fn: Callable) -> StreamingTransform:
        @streaming_transform_with_state
        def wrapper(*args, **kwargs):
            state_patterns = {f"pattern_{i}": module for i, module in enumerate(state_modules)}
            composer = CompositeStateManager(state_patterns, strategy)

            input_data = args[0] if args else kwargs
            composed_output = composer(input_data)

            # Call original function with composed outputs
            return fn(composed_output, *args, **kwargs)
        return wrapper
    return decorator
