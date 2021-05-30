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
"""Gym agent defined from an Haiku module"""

from dataclasses import dataclass, field
from typing import Any

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

from wax.gym.agent import Agent


@dataclass
class HaikuAgent(Agent):
    """Wrap a pair of init/apply function as a standard python class with state management."""

    agent: hk.TransformedWithState
    name: Any = None
    _seq: hk.PRNGSequence = field(
        init=False, default_factory=lambda: hk.PRNGSequence(42)
    )
    _params: Any = field(init=False, default=None)
    _state: Any = field(init=False, default=None)
    _initialized: bool = field(init=False, default=False)

    def _initialize(self, obs):
        self._params, self._state = self.agent.init(next(self._seq), obs)
        self._initialized = True

    def reset(self):
        self._params = None
        self._state = None
        self._initialized = False

    def __call__(self, obs):

        if not self._initialized:
            self._initialize(obs)

        action, self._state = self.agent.apply(
            self._params, self._state, next(self._seq), obs
        )
        return action
