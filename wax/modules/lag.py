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
"""Delay operator."""
import jax.numpy as jnp

from wax.modules.buffer import Buffer


class Lag(Buffer):
    """Delay operator."""

    def __init__(self, lag, fill_value=jnp.nan, return_state=False, name=None):
        super().__init__(
            maxlen=lag + 1, fill_value=fill_value, return_state=return_state, name=name
        )
        self.lag = lag

    def __call__(self, x):
        buffer = super().__call__(x)
        return buffer[0]
