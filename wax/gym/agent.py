# Copyright 2021 Google LLC
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
"""Define API for Gym agents."""
import logging
from abc import ABCMeta, abstractmethod
from typing import Any

from wax.gym.entity import Entity
from wax.gym.gym_unroll import gym_unroll

logger = logging.getLogger(__name__)


# @dataclass
class Agent(Entity, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, obs: Any) -> Any:
        ...

    def reset(self):
        logger.warning(f"`reset` for agent `{self.name}` not implemented")

    def feed(self, reward):
        logger.warning(f"`feed` for agent `{self.name}` not implemented")

    def unroll(
        self,
        env,
        *,
        obs=None,
        callbacks=None,
    ):
        """Unroll agent with a gym environment.

        Parameters
        ----------
        env
            Gym environment used to unroll the data and feed the agent.

        obs
            if passed, start the Gym unroll loop with this observation instead of
            the observation returned by 'env.reset()', else 'obs = env.reset()' is
            used as first observation.

        callbacks
            Additional callback to use


        Returns
        -------
        gym_state
            nested data structure
        """
        return gym_unroll(self, env, obs=obs, callbacks=callbacks)
