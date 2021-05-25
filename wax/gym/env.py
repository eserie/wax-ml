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
"""Define API for Gym environments."""

import logging
from abc import ABCMeta, abstractmethod
from dataclasses import field
from typing import Any, Tuple

from wax.gym.entity import Entity

logger = logging.getLogger(__name__)


class Env(Entity, metaclass=ABCMeta):
    # Set these in ALL subclasses
    action_space: Any = field(init=False, default=None)
    observation_space: Any = field(init=False, default=None)

    @abstractmethod
    def reset(self) -> Any:
        ...

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, Any, bool, Any]:
        """
        Parameters
        ----------
        action
            action of the agent.

        Returns
        -------
        obs
            Modified observation
        rw
            Reward
        done
            true if the run is over.
        info
            Any data structure containing some general informations
            about the running simulation.

        """

    def close(self):
        ...
