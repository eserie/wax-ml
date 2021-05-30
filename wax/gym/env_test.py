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
from abc import ABCMeta
from dataclasses import dataclass

from wax.gym.env import Env


def test_env():
    Env.__abstractmethods__ = set()

    @dataclass
    class DummyEnv(Env):
        ...

    env = DummyEnv()
    obs = env.reset()
    assert isinstance(Env, ABCMeta)
    assert obs is None
