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
"""Flax-based GymFeedback module for RL agent-environment interaction."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class GymFeedback(nn.Module):
    """Flax-based module for agent-environment interaction in RL settings.

    This module implements the feedback loop between an agent and environment,
    managing action state between timesteps and providing flexible output options.
    """

    agent: Any  # Agent module/function

    @nn.compact
    def __call__(self, obs: jnp.ndarray, reward: jnp.ndarray | None = None) -> Any:
        """Process observation and return agent action.

        Args:
            obs: Current observation from environment
            reward: Previous reward (optional)

        Returns:
            Agent action or tuple of (obs, action, reward) depending on usage
        """
        # Ensure observation is an array
        obs = jnp.asarray(obs)

        # Initialize previous action state
        prev_action = self.variable(
            "state",
            "prev_action",
            lambda: jnp.zeros_like(obs),  # Initialize with zero action
        )

        # Get agent's action based on current observation
        # The agent can be a function or a module
        if callable(self.agent):
            if reward is not None:
                # Agent that takes both observation and reward
                try:
                    action = self.agent(obs, reward)
                except TypeError:
                    # Agent only takes observation
                    action = self.agent(obs)
            else:
                # Agent only takes observation
                action = self.agent(obs)
        else:
            raise ValueError("Agent must be callable")

        # Ensure action is an array
        action = jnp.asarray(action)

        # Store current action for next timestep
        prev_action.value = action

        # Return the action (can be extended to return more info if needed)
        return action


class GymFeedbackLoop(nn.Module):
    """Extended GymFeedback that returns full interaction tuple."""

    agent: Any  # Agent module/function

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray, reward: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Process observation and return full interaction data.

        Args:
            obs: Current observation from environment
            reward: Previous reward (optional)

        Returns:
            Tuple of (observation, action, reward)
        """
        # Use base GymFeedback to get action
        gym_feedback = GymFeedback(agent=self.agent)
        action = gym_feedback(obs, reward)

        # Return full interaction tuple
        if reward is None:
            reward = jnp.array(0.0)  # Default reward

        return obs, action, reward


def create_gym_feedback(agent: Any) -> GymFeedback:
    """Factory function to create GymFeedback module.

    Args:
        agent: Agent module/function that maps observations to actions

    Returns:
        GymFeedback module instance
    """
    return GymFeedback(agent=agent)


def create_gym_feedback_loop(agent: Any) -> GymFeedbackLoop:
    """Factory function to create GymFeedbackLoop module.

    Args:
        agent: Agent module/function that maps observations to actions

    Returns:
        GymFeedbackLoop module instance
    """
    return GymFeedbackLoop(agent=agent)
