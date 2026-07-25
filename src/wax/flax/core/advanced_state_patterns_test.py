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
"""Tests for advanced state patterns."""

import jax
import jax.numpy as jnp
from flax import linen as nn

from wax.flax.core.advanced_state_patterns import (
    AttentionBasedStateSelector,
    CompositeStateManager,
    HierarchicalStateMachine,
    streaming_attention_state,
    streaming_compose_states,
    streaming_state_machine,
)
from wax.flax.core.streaming_transforms import streaming_transform_with_state
from wax.flax.modules.ewma import EWMA


class SimpleRegimeDetector(nn.Module):
    """Simple regime detector for testing."""

    threshold: float = 1.0
    regime_name: str = "default"

    def setup(self):
        self.ewma = EWMA(alpha=0.2)

    def __call__(self, x):
        signal = self.ewma(x)
        regime_active = jnp.abs(signal) > self.threshold

        return {
            "regime": self.regime_name,
            "signal": signal,
            "active": regime_active,
            "input": x
        }


class TestHierarchicalStateMachine:
    """Test hierarchical state machine functionality."""

    def test_basic_hierarchical_state_machine(self):
        """Test basic hierarchical state machine without dependencies."""
        # Create simple state modules
        market_detector = SimpleRegimeDetector(threshold=0.5, regime_name="market")
        volatility_detector = SimpleRegimeDetector(threshold=1.0, regime_name="volatility")

        state_modules = {
            "market": market_detector,
            "volatility": volatility_detector
        }

        # Create hierarchical state machine
        hsm = HierarchicalStateMachine(state_modules)

        # Initialize
        rng = jax.random.PRNGKey(42)
        variables = hsm.init(rng, jnp.array(1.0))

        # Should initialize successfully
        assert "params" in variables
        assert "state" in variables

        # Apply the state machine
        output, new_variables = hsm.apply(variables, jnp.array(1.5), mutable=['state'])

        # Should have outputs for both state modules
        assert "market" in output
        assert "volatility" in output

        # Each output should have expected structure
        for state_name in ["market", "volatility"]:
            state_output = output[state_name]
            assert "regime" in state_output
            assert "signal" in state_output
            assert "active" in state_output
            assert jnp.isfinite(state_output["signal"])

    def test_hierarchical_state_machine_with_dependencies(self):
        """Test hierarchical state machine with dependency ordering."""
        # Create state modules
        market_detector = SimpleRegimeDetector(threshold=0.5, regime_name="market")
        volatility_detector = SimpleRegimeDetector(threshold=1.0, regime_name="volatility")

        state_modules = {
            "market": market_detector,
            "volatility": volatility_detector
        }

        # Volatility depends on market
        dependencies = {"volatility": ["market"]}

        # Create hierarchical state machine
        hsm = HierarchicalStateMachine(state_modules, dependencies)

        # Test execution order computation
        execution_order = hsm.get_execution_order(["market", "volatility"])
        assert execution_order.index("market") < execution_order.index("volatility")

        # Initialize and test
        rng = jax.random.PRNGKey(42)
        variables = hsm.init(rng, jnp.array(1.0))

        output, new_variables = hsm.apply(variables, jnp.array(1.5), mutable=['state'])

        assert "market" in output
        assert "volatility" in output

    def test_hierarchical_coordination_strategies(self):
        """Test different coordination strategies."""
        market_detector = SimpleRegimeDetector(threshold=0.5, regime_name="market")
        volatility_detector = SimpleRegimeDetector(threshold=1.0, regime_name="volatility")

        state_modules = {
            "market": market_detector,
            "volatility": volatility_detector
        }

        # Test different strategies
        for strategy in ["sequential", "parallel", "hierarchical"]:
            hsm = HierarchicalStateMachine(state_modules, coordination_strategy=strategy)

            rng = jax.random.PRNGKey(42)
            variables = hsm.init(rng, jnp.array(1.0))

            output, _ = hsm.apply(variables, jnp.array(1.5), mutable=['state'])

            # Should work with all strategies
            assert "market" in output
            assert "volatility" in output

    def test_state_machine_decorator(self):
        """Test the @streaming_state_machine decorator."""

        @streaming_state_machine({
            "trend": SimpleRegimeDetector(threshold=0.3, regime_name="trend"),
            "momentum": SimpleRegimeDetector(threshold=0.7, regime_name="momentum")
        })
        def multi_regime_processor(state_outputs, price):
            """Process with multiple regime detection."""
            # Combine regime signals
            trend_signal = state_outputs["trend"]["signal"]
            momentum_signal = state_outputs["momentum"]["signal"]

            combined_signal = (trend_signal + momentum_signal) / 2

            return {
                "price": price,
                "trend_signal": trend_signal,
                "momentum_signal": momentum_signal,
                "combined_signal": combined_signal,
                "regimes": {name: output["regime"] for name, output in state_outputs.items()}
            }

        # Test the decorated function
        rng = jax.random.PRNGKey(42)
        params, state = multi_regime_processor.init(rng, jnp.array(100.0))

        output, new_state = multi_regime_processor.apply(
            params, state, None, jnp.array(105.0)
        )

        # Check output structure
        assert "price" in output
        assert "trend_signal" in output
        assert "momentum_signal" in output
        assert "combined_signal" in output
        assert "regimes" in output

        # Check regime information
        assert "trend" in output["regimes"]
        assert "momentum" in output["regimes"]


class TestAttentionBasedStateSelector:
    """Test attention-based state selection functionality."""

    def test_basic_attention_state_selector(self):
        """Test basic attention mechanism functionality."""
        # Create attention-based state selector
        attention_selector = AttentionBasedStateSelector(
            embed_dim=32, num_heads=2, max_history_length=10
        )

        # Initialize
        rng = jax.random.PRNGKey(42)
        variables = attention_selector.init(rng, jnp.array(1.0))

        # Test with simple state
        current_state = jnp.array([1.0, 2.0, 3.0])
        output, new_variables = attention_selector.apply(
            variables, current_state, mutable=['state']
        )

        # Check output structure
        assert "enhanced_state" in output
        assert "original_state" in output
        assert "relevant_history" in output
        assert "attention_weights" in output
        assert "current_state" in output

        # Check shapes
        assert output["enhanced_state"].shape == (32,)  # embed_dim
        assert output["original_state"].shape == (32,)

    def test_attention_with_history_buildup(self):
        """Test attention mechanism as history accumulates."""
        attention_selector = AttentionBasedStateSelector(
            embed_dim=16, num_heads=1, max_history_length=5
        )

        rng = jax.random.PRNGKey(42)
        # Initialize with the same type of input we'll use in the test
        variables = attention_selector.init(rng, jnp.array([1.0, 2.0, 3.0]))

        # Process sequence of states to build history
        states = [jnp.array([float(i), float(i+1), float(i+2)]) for i in range(7)]
        outputs = []
        current_variables = variables

        for state in states:
            output, new_state = attention_selector.apply(
                current_variables, state, mutable=['state']
            )
            # Update only the state collection, keeping params intact
            current_variables = {**current_variables, 'state': new_state['state']}
            outputs.append(output)

        # Check that attention weights have the expected shapes and are meaningful
        # Early in the sequence, buffer is small
        early_weights = outputs[1]["attention_weights"]  # Should have 1 weight (1 state in buffer)
        late_weights = outputs[-1]["attention_weights"]   # Should have max 5 weights (max buffer size)

        # Verify that attention weights are valid probabilities
        assert jnp.all(early_weights >= 0)
        assert jnp.all(late_weights >= 0)
        assert jnp.allclose(jnp.sum(early_weights), 1.0, atol=1e-6)
        assert jnp.allclose(jnp.sum(late_weights), 1.0, atol=1e-6)

        # Buffer should grow with more states (up to max_history_length=5)
        assert len(late_weights) >= len(early_weights)
        assert len(late_weights) <= 5  # max_history_length

    def test_attention_state_decorator(self):
        """Test the @streaming_attention_state decorator."""

        @streaming_attention_state(embed_dim=32, max_history=10)
        def context_aware_processor(attention_output, x):
            """Process with attention-enhanced context."""
            enhanced_state = attention_output["enhanced_state"]
            attention_weights = attention_output["attention_weights"]

            # Simple processing using enhanced state
            processed = jnp.mean(enhanced_state) + x

            return {
                "input": x,
                "processed": processed,
                "enhanced_state": enhanced_state,
                "attention_strength": jnp.sum(attention_weights)
            }

        # Test the decorated function
        rng = jax.random.PRNGKey(42)
        params, state = context_aware_processor.init(rng, jnp.array(1.0))

        # Process sequence to build context
        inputs = [1.0, 2.0, 3.0, 4.0, 5.0]
        outputs = []
        current_state = state

        for x in inputs:
            output, current_state = context_aware_processor.apply(
                params, current_state, None, jnp.array(x)
            )
            outputs.append(output)

        # Check that processing evolves with context
        assert len(outputs) == len(inputs)

        # Later outputs should show influence of context
        for output in outputs:
            assert "processed" in output
            assert "attention_strength" in output
            assert jnp.isfinite(output["processed"])

    def test_attention_with_dict_states(self):
        """Test attention mechanism with dictionary states."""
        attention_selector = AttentionBasedStateSelector(embed_dim=16)

        rng = jax.random.PRNGKey(42)
        variables = attention_selector.init(rng, {"price": 100.0, "volume": 1000.0})

        # Test with dictionary state
        dict_state = {"price": 105.0, "volume": 1200.0, "signal": 0.5}
        output, _ = attention_selector.apply(variables, dict_state, mutable=['state'])

        # Should handle dictionary states correctly
        assert "enhanced_state" in output
        assert output["current_state"] == dict_state


class TestCompositeStateManager:
    """Test composite state management functionality."""

    def test_pipeline_composition(self):
        """Test pipeline composition strategy."""
        # Create simple state patterns
        class SimpleProcessor(nn.Module):
            name: str

            def __call__(self, x):
                if isinstance(x, dict):
                    return {f"{self.name}_processed": x.get("value", 0) + 1}
                else:
                    return {f"{self.name}_processed": x + 1, "value": x + 1}

        state_patterns = {
            "first": SimpleProcessor(name="first"),
            "second": SimpleProcessor(name="second")
        }

        composer = CompositeStateManager(state_patterns, "pipeline")

        # Initialize
        rng = jax.random.PRNGKey(42)
        variables = composer.init(rng, jnp.array(1.0))

        # Apply pipeline
        output, _ = composer.apply(variables, jnp.array(1.0), mutable=['state'])

        # Should have outputs from both patterns
        assert "first" in output
        assert "second" in output

        # Pipeline: second should process output of first
        # Flax creates scoped names like "state_patterns_first_processed"
        assert "state_patterns_first_processed" in output["first"]
        assert "state_patterns_second_processed" in output["second"]

        # Verify the pipeline behavior: second processes output of first
        assert output["first"]["state_patterns_first_processed"] == 2.0  # 1.0 + 1
        assert output["second"]["state_patterns_second_processed"] == 3.0  # value from first (2.0) + 1

    def test_parallel_composition(self):
        """Test parallel composition strategy."""
        class IdentityProcessor(nn.Module):
            name: str

            def __call__(self, x):
                return {f"{self.name}_result": x}

        state_patterns = {
            "pattern_a": IdentityProcessor(name="a"),
            "pattern_b": IdentityProcessor(name="b")
        }

        composer = CompositeStateManager(state_patterns, "parallel")

        rng = jax.random.PRNGKey(42)
        variables = composer.init(rng, jnp.array(2.0))

        output, _ = composer.apply(variables, jnp.array(2.0), mutable=['state'])

        # Both patterns should process the same input
        assert "pattern_a" in output
        assert "pattern_b" in output

        # Check that Flax scoped the names correctly
        assert "state_patterns_pattern_a_result" in output["pattern_a"]
        assert "state_patterns_pattern_b_result" in output["pattern_b"]

        # Both should process the same input (2.0)
        assert output["pattern_a"]["state_patterns_pattern_a_result"] == 2.0
        assert output["pattern_b"]["state_patterns_pattern_b_result"] == 2.0

    def test_compose_states_decorator(self):
        """Test the @streaming_compose_states decorator."""

        # Create simple modules for composition
        class AddOneProcessor(nn.Module):
            def __call__(self, x):
                return {"added": x + 1}

        class MultiplyTwoProcessor(nn.Module):
            def __call__(self, x):
                if isinstance(x, dict) and "added" in x:
                    return {"multiplied": x["added"] * 2}
                else:
                    return {"multiplied": x * 2}

        @streaming_compose_states(
            AddOneProcessor(),
            MultiplyTwoProcessor(),
            strategy="pipeline"
        )
        def composed_processor(composed_output, x):
            """Process with composed state patterns."""
            # Extract results from composition
            added_result = composed_output["pattern_0"]["added"]
            multiplied_result = composed_output["pattern_1"]["multiplied"]

            return {
                "input": x,
                "added": added_result,
                "multiplied": multiplied_result,
                "final": added_result + multiplied_result
            }

        # Test the decorated function
        rng = jax.random.PRNGKey(42)
        params, state = composed_processor.init(rng, jnp.array(5.0))

        output, _ = composed_processor.apply(params, state, None, jnp.array(5.0))

        # Check composition results
        assert "input" in output
        assert "added" in output
        assert "multiplied" in output
        assert "final" in output

        # Verify computation: (5 + 1) * 2 = 12, final = 6 + 12 = 18
        assert output["added"] == 6.0
        assert output["multiplied"] == 12.0
        assert output["final"] == 18.0


class TestAdvancedStateIntegration:
    """Test integration between different advanced state patterns."""

    def test_hierarchical_with_attention(self):
        """Test combining hierarchical state machine with attention."""

        @streaming_transform_with_state
        def integrated_processor(price, volume):
            """Processor combining hierarchical states and attention."""

            # Hierarchical state machine
            market_detector = SimpleRegimeDetector(threshold=0.5, regime_name="market")
            volatility_detector = SimpleRegimeDetector(threshold=1.0, regime_name="volatility")

            hsm = HierarchicalStateMachine({
                "market": market_detector,
                "volatility": volatility_detector
            })

            regime_outputs = hsm(price)

            # Attention-based state selection
            attention_selector = AttentionBasedStateSelector(embed_dim=16, max_history_length=5)
            attention_output = attention_selector(regime_outputs)

            # Combine information
            enhanced_state = attention_output["enhanced_state"]
            market_signal = regime_outputs["market"]["signal"]
            volatility_signal = regime_outputs["volatility"]["signal"]

            combined_signal = jnp.mean(enhanced_state) + 0.5 * (market_signal + volatility_signal)

            return {
                "price": price,
                "volume": volume,
                "regimes": regime_outputs,
                "attention": attention_output,
                "combined_signal": combined_signal
            }

        # Test integrated processor
        rng = jax.random.PRNGKey(42)
        params, state = integrated_processor.init(rng, jnp.array(100.0), jnp.array(1000.0))

        # Process sequence
        prices = [100.0, 102.0, 99.0, 105.0, 98.0]
        volumes = [1000.0, 1100.0, 900.0, 1200.0, 800.0]

        outputs = []
        current_state = state

        for price, volume in zip(prices, volumes, strict=False):
            output, current_state = integrated_processor.apply(
                params, current_state, None, jnp.array(price), jnp.array(volume)
            )
            outputs.append(output)

        # Check that integration works
        assert len(outputs) == len(prices)

        for output in outputs:
            assert "regimes" in output
            assert "attention" in output
            assert "combined_signal" in output
            assert jnp.isfinite(output["combined_signal"])

    def test_jax_transformations_compatibility(self):
        """Test that advanced state patterns work with JAX transformations."""

        @streaming_state_machine({
            "detector": SimpleRegimeDetector(threshold=1.0, regime_name="test")
        })
        def simple_state_system(state_outputs, x):
            signal = state_outputs["detector"]["signal"]
            return {"input": x, "output": signal * 2}

        # Test JIT compilation
        jitted_init = jax.jit(simple_state_system.init)
        jitted_apply = jax.jit(simple_state_system.apply)

        rng = jax.random.PRNGKey(42)
        x0 = jnp.array(1.0)

        # Initialize with JIT
        params, state = jitted_init(rng, x0)

        # Apply with JIT
        output, new_state = jitted_apply(params, state, None, x0)

        # Should work correctly
        assert "input" in output
        assert "output" in output
        assert jnp.isfinite(output["output"])


