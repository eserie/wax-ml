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
"""Gym feedback between an agent and a Gym environment."""
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import haiku as hk


@dataclass
class GymOutput:
    """Dynamically define GymOutput namedtuple structure.

    Args:
        return_obs: if true, add 'obs' field to output structure.
        return_action: if true, action 'obs' field to output structure.
    """

    return_obs: bool = False
    return_action: bool = False
    gym_output_struct: Any = field(init=False)

    def __post_init__(self):
        outputs = ["reward"]
        if self.return_obs:
            outputs += ["obs"]
        if self.return_action:
            outputs += ["action"]
        self.gym_output_struct = namedtuple("GymOutput", outputs)

    def format(self, reward, obs=None, action=None):
        outputs = [reward]
        if self.return_obs:
            outputs += [obs]
        if self.return_action:
            outputs += [action]
        return self.gym_output_struct(*outputs)


class GymInfo(NamedTuple):
    env: Any
    agent: Any


class GymFeedback(hk.Module):
    """Gym feedback between an agent and a Gym environment."""

    GymState = namedtuple("GymState", "action")

    def __init__(
        self,
        agent,
        env,
        return_obs=False,
        return_action=False,
        init_action=None,
        name=None,
    ):
        """Initialize module.

        Args:
            agent : Gym environment used to unroll the data and feed the agent.
            env : Gym environment used to unroll the data and feed the agent.
            return_obs : if true return environment observation
            return_action : if true return agent action
            name : name of the module

        """
        super().__init__(name=name)
        self.agent = agent
        self.env = env
        self.return_obs = return_obs
        self.return_action = return_action
        self.init_action = init_action

        self.GymOutput = GymOutput(self.return_obs, self.return_action)

    def __call__(self, raw_obs):
        """Compute Gym feedback loop.

        Args:
            raw_obs: raw observations.
        Returns:
            gym_output : reward.
                Use return_obs=True to also return env observations.
                Use return_action=True to aslo return agent actions.
        """

        def init_action(shape, dtype):
            if self.init_action is not None:
                return self.GymState(self.init_action)
            else:
                # call
                return self.GymState(self.agent(raw_obs)[0])

        action = hk.get_state("action", shape=[], init=init_action)

        rw, obs, info_env = self.env(action.action, raw_obs)
        action, info_agent = self.agent(obs)

        info = GymInfo(info_env, info_agent)
        outputs = self.GymOutput.format(rw, obs, action)
        state = self.GymState(action)
        hk.set_state("action", state)
        return outputs, info
