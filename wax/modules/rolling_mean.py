# Copyright 2021 The Wax Authors
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
"""Rolling mean."""
import haiku as hk
import jax.numpy as jnp

from wax.modules.buffer import Buffer


class RollingMean(hk.Module):
    """Rolling mean."""

    def __init__(self, horizon, min_periods=1, name=None):
        super().__init__(name=name)
        self.horizon = horizon
        self.min_periods = min_periods

    def __call__(self, x):
        buffer, attrs = Buffer(self.horizon, return_state=True)(x)
        sum = jnp.where(jnp.logical_not(jnp.isnan(buffer)), buffer, 0.0).sum(axis=0)
        count = jnp.where(jnp.logical_not(jnp.isnan(buffer)), 1, 0).sum(axis=0)
        mean = jnp.where(count >= self.min_periods, sum / count, jnp.nan)
        return mean
