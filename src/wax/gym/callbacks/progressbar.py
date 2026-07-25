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
from tqdm.auto import tqdm

from wax.gym.callbacks.callbacks import Callback


class ProgressBar(Callback):
    def __init__(self):
        # state
        self._pbar = None

    def __enter__(self):
        super().__enter__()

    def _on_train_start(self, env, agent, obs):
        if self._pbar is None:
            try:
                self._pbar = tqdm(total=len(env))
            except TypeError:
                self._pbar = tqdm()

    def _on_step(self, env, agent, gym_state):
        self._pbar.update()

    def _on_train_end(self, env, agent):
        if self._pbar is not None:
            self._pbar.close()
