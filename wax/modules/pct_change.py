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
from typing import Optional

import haiku as hk
import jax.numpy as jnp

from wax.modules import Ffill, Lag


class PctChange(hk.Module):
    """Relative change between the current and a prior element.

    Computes the relative change from the immediately previous observation by
    default. This is useful in comparing the relative of change in a time
    series of elements.

    Args:
        periods : Periods to shift for forming relative change.
        fill_method : How to handle NAs before computing percent changes.
        limit : The number of consecutive NAs to fill before stopping.
        fillna_zero: if true (default), behave as in pandas: the module return 0
            where the current observation is NA and the previous observation is
            not NA; else return NA.
    """

    def __init__(
        self,
        periods: int = 1,
        fill_method: str = "pad",
        limit: Optional[int] = None,
        fillna_zero: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.periods = periods
        self.fill_method = fill_method
        self.limit = limit
        self.fillna_zero = fillna_zero
        assert periods == 1, "periods > 1 not implemented."

    def __call__(self, x):
        if self.fill_method in ["ffill", "pad"]:
            previous_x = Lag(self.periods)(Ffill()(x))
        else:
            previous_x = Lag(self.periods)()
        pct_change = x / previous_x - 1.0
        if self.fillna_zero:
            pct_change = jnp.where(
                jnp.isnan(x) & ~jnp.isnan(previous_x), 0.0, pct_change
            )
        return pct_change
