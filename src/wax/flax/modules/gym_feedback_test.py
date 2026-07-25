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
"""Tests for Flax GymFeedback module."""

import jax
import jax.numpy as jnp
import pytest

from wax.flax.modules.gym_feedback import (
    GymFeedback,
    create_gym_feedback,
    create_gym_feedback_loop,
)


def apply_stateful(module, variables, *args, **kwargs):
    """Helper to apply a module with proper state handling."""
    output, new_variables = module.apply(variables, *args, **kwargs, mutable=["state"])
    return output, new_variables


def simple_agent(obs):
    """Simple agent that returns scaled observation as action."""
    return obs * 0.5


def linear_agent(obs):
    """Linear agent that applies simple transformation."""
    return jnp.sum(obs) * jnp.ones_like(obs)


def reward_aware_agent(obs, reward):
    """Agent that considers previous reward."""
    return obs + reward * 0.1


class TestGymFeedback:
    """Test cases for GymFeedback module."""

    def test_simple_agent_interaction(self):
        """Test basic agent-environment interaction."""
        # Create GymFeedback with simple agent
        gym_feedback = create_gym_feedback(simple_agent)

        # Initialize
        key = jax.random.PRNGKey(42)
        obs = jnp.array([1.0, 2.0])
        variables = gym_feedback.init(key, obs)

        # Apply agent to observation
        action, new_variables = apply_stateful(gym_feedback, variables, obs)

        # Check that action has correct shape and values
        expected_action = obs * 0.5
        assert action.shape == obs.shape
        assert jnp.allclose(action, expected_action)

    def test_agent_with_reward(self):
        """Test agent that can handle reward input."""
        # Create GymFeedback with reward-aware agent
        gym_feedback = create_gym_feedback(reward_aware_agent)

        # Initialize
        key = jax.random.PRNGKey(42)
        obs = jnp.array([1.0])
        reward = jnp.array(0.5)
        variables = gym_feedback.init(key, obs, reward)

        # Apply agent with observation and reward
        action, new_variables = apply_stateful(gym_feedback, variables, obs, reward)

        # Check that action incorporates reward
        expected_action = obs + reward * 0.1
        assert jnp.allclose(action, expected_action)

    def test_sequential_interactions(self):
        """Test multiple sequential agent interactions."""
        # Create GymFeedback
        gym_feedback = create_gym_feedback(simple_agent)

        # Initialize
        key = jax.random.PRNGKey(42)
        variables = gym_feedback.init(key, jnp.array([0.0]))

        # Apply sequence of observations
        observations = [jnp.array([1.0]), jnp.array([2.0]), jnp.array([-1.0]), jnp.array([0.5])]

        actions = []
        current_variables = variables
        for obs in observations:
            action, current_variables = apply_stateful(gym_feedback, current_variables, obs)
            actions.append(action)

        # Check that all actions are reasonable
        for i, (obs, action) in enumerate(zip(observations, actions, strict=False)):
            expected = obs * 0.5
            assert jnp.allclose(action, expected), f"Mismatch at step {i}"

    def test_vector_observations(self):
        """Test with multi-dimensional observations."""
        # Create GymFeedback with linear agent
        gym_feedback = create_gym_feedback(linear_agent)

        # Initialize
        key = jax.random.PRNGKey(42)
        obs = jnp.array([1.0, 2.0, 3.0])
        variables = gym_feedback.init(key, obs)

        # Apply agent
        action, new_variables = apply_stateful(gym_feedback, variables, obs)

        # Linear agent should return sum of obs times ones
        expected_sum = jnp.sum(obs)  # 6.0
        expected_action = expected_sum * jnp.ones_like(obs)

        assert action.shape == obs.shape
        assert jnp.allclose(action, expected_action)

    def test_state_persistence(self):
        """Test that previous action state is maintained."""
        # Create GymFeedback
        gym_feedback = create_gym_feedback(simple_agent)

        # Initialize
        key = jax.random.PRNGKey(42)
        obs1 = jnp.array([1.0])
        variables = gym_feedback.init(key, obs1)

        # First interaction
        action1, variables = apply_stateful(gym_feedback, variables, obs1)

        # Second interaction
        obs2 = jnp.array([2.0])
        action2, variables = apply_stateful(gym_feedback, variables, obs2)

        # Actions should be different based on observations
        assert not jnp.allclose(action1, action2)
        assert jnp.allclose(action1, obs1 * 0.5)
        assert jnp.allclose(action2, obs2 * 0.5)


class TestGymFeedbackLoop:
    """Test cases for GymFeedbackLoop module."""

    def test_feedback_loop_output_format(self):
        """Test that feedback loop returns full interaction tuple."""
        # Create GymFeedbackLoop
        gym_loop = create_gym_feedback_loop(simple_agent)

        # Initialize
        key = jax.random.PRNGKey(42)
        obs = jnp.array([1.0])
        reward = jnp.array(0.5)
        variables = gym_loop.init(key, obs, reward)

        # Apply feedback loop
        result, new_variables = apply_stateful(gym_loop, variables, obs, reward)

        # Should return tuple of (obs, action, reward)
        assert isinstance(result, tuple)
        assert len(result) == 3

        obs_out, action_out, reward_out = result
        assert jnp.allclose(obs_out, obs)
        assert jnp.allclose(action_out, obs * 0.5)  # simple_agent output
        assert jnp.allclose(reward_out, reward)

    def test_feedback_loop_without_reward(self):
        """Test feedback loop when no reward is provided."""
        # Create GymFeedbackLoop
        gym_loop = create_gym_feedback_loop(simple_agent)

        # Initialize
        key = jax.random.PRNGKey(42)
        obs = jnp.array([2.0])
        variables = gym_loop.init(key, obs)

        # Apply without reward
        result, new_variables = apply_stateful(gym_loop, variables, obs)

        # Should return tuple with default reward
        obs_out, action_out, reward_out = result
        assert jnp.allclose(obs_out, obs)
        assert jnp.allclose(action_out, obs * 0.5)
        assert jnp.allclose(reward_out, 0.0)  # Default reward

    def test_feedback_loop_sequential(self):
        """Test sequential feedback loop interactions."""
        # Create GymFeedbackLoop
        gym_loop = create_gym_feedback_loop(simple_agent)

        # Initialize
        key = jax.random.PRNGKey(42)
        variables = gym_loop.init(key, jnp.array([0.0]))

        # Apply sequence
        observations = [jnp.array([1.0]), jnp.array([2.0]), jnp.array([3.0])]
        rewards = [jnp.array(0.1), jnp.array(0.2), jnp.array(0.3)]

        results = []
        current_variables = variables
        for obs, reward in zip(observations, rewards, strict=False):
            result, current_variables = apply_stateful(gym_loop, current_variables, obs, reward)
            results.append(result)

        # Check all results
        for i, (result, expected_obs, expected_reward) in enumerate(
            zip(results, observations, rewards, strict=False)
        ):
            obs_out, action_out, reward_out = result
            assert jnp.allclose(obs_out, expected_obs), f"Obs mismatch at step {i}"
            assert jnp.allclose(action_out, expected_obs * 0.5), f"Action mismatch at step {i}"
            assert jnp.allclose(reward_out, expected_reward), f"Reward mismatch at step {i}"

    def test_agent_error_handling(self):
        """Test error handling for invalid agent."""
        # Test with non-callable agent
        with pytest.raises(ValueError, match="Agent must be callable"):
            gym_feedback = GymFeedback(agent="not_callable")
            key = jax.random.PRNGKey(42)
            variables = gym_feedback.init(key, jnp.array([1.0]))
            action, new_vars = apply_stateful(gym_feedback, variables, jnp.array([1.0]))
