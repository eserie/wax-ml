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
"""Gym environment defined from an Haiku module"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generator

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
import haiku as hk

from wax.gym.env import Env


@dataclass
class HaikuEnv(Env):
    """Wrap a pair of init/apply function as a standard python class
    with state management and Gym env API.

    Args:
        data_generator :
    """

    data_generator: Generator
    env: hk.TransformedWithState
    name: Any = None

    _info: Dict = field(init=False, default_factory=dict)
    _seq: hk.PRNGSequence = field(
        init=False, default_factory=lambda: hk.PRNGSequence(42)
    )
    _params: Any = field(init=False, default=None)
    _state: Any = field(init=False, default=None)
    _initialized: bool = field(init=False, default=False)

    def reset(self):
        raw_obs, self._info = next(self.data_generator)
        self._initialized = False
        return raw_obs

    def step(self, action):
        try:
            raw_obs, self._info = next(self.data_generator)
            done = False
        except StopIteration:
            obs = None
            rw = 0
            done = True
            return obs, rw, done, self._info

        if not self._initialized:
            self._params, self._state = self.env.init(next(self._seq), action, raw_obs)
            self._initialized = True

        (rw, obs), self._state = self.env.apply(
            self._params, self._state, next(self._seq), action, raw_obs
        )

        return obs, rw, done, self._info

    def __len__(self):
        return self._info.get("maxlen") if isinstance(self._info, dict) else None
