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
"""Relative change between the current and a prior element."""
import haiku as hk

from wax.modules import Ffill, Lag


class PctChange(hk.Module):
    """Relative change between the current and a prior element."""

    def __init__(
        self,
        periods: int = 1,
        fill_method: str = "pad",
        limit: int = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.periods = periods
        self.fill_method = fill_method
        self.limit = limit
        assert periods == 1, "periods > 1 not implemented."

    def __call__(self, x):
        if self.fill_method in ["ffill", "pad"]:
            previous_x = Lag(self.periods)(Ffill()(x))
        else:
            previous_x = Lag(self.periods)()
        pct_change = x / previous_x - 1.0
        return pct_change
