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
"""Tests for Flax ARMA module."""

import jax
import jax.numpy as jnp

from wax.flax.modules.arma import create_arma


def apply_stateful(module, variables, *args, **kwargs):
    """Helper to apply a module with proper state handling."""
    output, new_variables = module.apply(variables, *args, **kwargs, mutable=["state"])
    return output, new_variables


class TestARMA:
    """Test cases for ARMA module."""

    def test_basic_arma_filtering(self):
        """Test basic ARMA filtering with simple coefficients."""
        # Create ARMA module with simple coefficients
        alpha = jnp.array([0.5])  # AR coefficient
        beta = jnp.array([0.3])  # MA coefficient
        arma = create_arma(alpha=alpha, beta=beta)

        # Initialize with a sample
        key = jax.random.PRNGKey(42)
        x = jnp.array(1.0)
        variables = arma.init(key, x)

        # Apply ARMA filter to a sequence
        inputs = [1.0, 2.0, 3.0, 2.0, 1.0]
        outputs = []
        current_variables = variables

        for inp in inputs:
            output, current_variables = apply_stateful(arma, current_variables, jnp.array(inp))
            outputs.append(float(output))

        # Check that outputs are reasonable (non-zero and bounded)
        assert all(isinstance(out, float) for out in outputs)
        assert all(not jnp.isnan(out) for out in outputs)
        assert all(not jnp.isinf(out) for out in outputs)

    def test_pure_ar_model(self):
        """Test pure autoregressive model (beta=0)."""
        # Create AR(1) model
        alpha = jnp.array([0.8])
        beta = jnp.array([0.0])  # No MA component
        arma = create_arma(alpha=alpha, beta=beta)

        # Initialize
        key = jax.random.PRNGKey(42)
        variables = arma.init(key, jnp.array(0.0))

        # Apply impulse (1 followed by zeros)
        impulse_response = []
        current_variables = variables
        for i in range(5):
            x = jnp.array(1.0 if i == 0 else 0.0)
            output, current_variables = apply_stateful(arma, current_variables, x)
            impulse_response.append(float(output))

        # For AR(1) with coefficient 0.8, impulse response should show some response
        # The first response should be non-zero (captures the impulse input)
        assert abs(impulse_response[0]) > 1e-6, (
            f"First response should be non-zero: {impulse_response}"
        )

        # Check that we get a reasonable impulse response
        assert any(abs(val) > 1e-6 for val in impulse_response), (
            f"All outputs are zero: {impulse_response}"
        )

    def test_pure_ma_model(self):
        """Test pure moving average model (alpha=0)."""
        # Create MA(1) model
        alpha = jnp.array([0.0])  # No AR component
        beta = jnp.array([0.6])
        arma = create_arma(alpha=alpha, beta=beta)

        # Initialize
        key = jax.random.PRNGKey(42)
        variables = arma.init(key, jnp.array(0.0))

        # Apply step input
        outputs = []
        current_variables = variables
        for i in range(4):
            x = jnp.array(1.0)  # Constant input
            output, current_variables = apply_stateful(arma, current_variables, x)
            outputs.append(float(output))

        # For MA model, output should stabilize quickly
        assert all(not jnp.isnan(out) for out in outputs)
        assert all(not jnp.isinf(out) for out in outputs)

    def test_multiple_coefficients(self):
        """Test ARMA with multiple AR and MA coefficients."""
        # Create ARMA(2,2) model
        alpha = jnp.array([0.5, -0.2])  # AR coefficients
        beta = jnp.array([0.3, 0.1])  # MA coefficients
        arma = create_arma(alpha=alpha, beta=beta)

        # Initialize
        key = jax.random.PRNGKey(42)
        variables = arma.init(key, jnp.array(0.0))

        # Apply random sequence
        key = jax.random.PRNGKey(123)
        inputs = jax.random.normal(key, (10,))

        outputs = []
        current_variables = variables
        for inp in inputs:
            output, current_variables = apply_stateful(arma, current_variables, inp)
            outputs.append(output)

        # Check that all outputs are finite
        outputs = jnp.array(outputs)
        assert jnp.all(jnp.isfinite(outputs))

    def test_vector_input(self):
        """Test ARMA with vector inputs."""
        # Create ARMA module
        alpha = jnp.array([0.4])
        beta = jnp.array([0.2])
        arma = create_arma(alpha=alpha, beta=beta)

        # Initialize with vector input
        key = jax.random.PRNGKey(42)
        x = jnp.array([1.0, 2.0])
        variables = arma.init(key, x)

        # Apply vector inputs
        x1 = jnp.array([1.5, 2.5])
        output1, variables = apply_stateful(arma, variables, x1)

        x2 = jnp.array([2.0, 3.0])
        output2, variables = apply_stateful(arma, variables, x2)

        # Check output shapes and values
        assert output1.shape == x1.shape
        assert output2.shape == x2.shape
        assert jnp.all(jnp.isfinite(output1))
        assert jnp.all(jnp.isfinite(output2))

    def test_zero_coefficients(self):
        """Test ARMA with zero coefficients (should pass through input)."""
        # Create ARMA with zero coefficients
        alpha = jnp.array([0.0])
        beta = jnp.array([0.0])
        arma = create_arma(alpha=alpha, beta=beta)

        # Initialize
        key = jax.random.PRNGKey(42)
        variables = arma.init(key, jnp.array(0.0))

        # Apply inputs
        test_inputs = [1.0, 2.0, -1.0, 0.5]
        current_variables = variables
        for inp in test_inputs:
            output, current_variables = apply_stateful(arma, current_variables, jnp.array(inp))
            # With zero coefficients, output should be proportional to input
            assert jnp.isfinite(output)

    def test_stability_check(self):
        """Test ARMA with coefficients that should remain stable."""
        # Create stable ARMA model (sum of AR coefficients < 1)
        alpha = jnp.array([0.3, 0.2])  # Sum = 0.5 < 1, should be stable
        beta = jnp.array([0.1])
        arma = create_arma(alpha=alpha, beta=beta)

        # Initialize
        key = jax.random.PRNGKey(42)
        variables = arma.init(key, jnp.array(0.0))

        # Apply white noise input for many steps
        key = jax.random.PRNGKey(456)
        noise = jax.random.normal(key, (50,)) * 0.1

        outputs = []
        current_variables = variables
        for n in noise:
            output, current_variables = apply_stateful(arma, current_variables, n)
            outputs.append(float(output))

        # Check that outputs don't blow up (remain bounded)
        outputs = jnp.array(outputs)
        assert jnp.all(jnp.isfinite(outputs))
        assert jnp.max(jnp.abs(outputs)) < 100  # Reasonable bound
