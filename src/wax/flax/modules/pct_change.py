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
"""Flax-based PctChange module for computing percentage changes."""

import flax.linen as nn
import jax.numpy as jnp

from .ffill import Ffill
from .lag import Lag


class PctChange(nn.Module):
    """Flax-based module for computing percentage changes in time series.

    Computes the relative change between the current and a prior element.
    This is useful for comparing relative changes in time series data.
    """

    periods: int = 1
    fill_method: str = "pad"
    limit: int | None = None
    fillna_zero: bool = True

    def setup(self):
        """Setup the PctChange module."""
        assert self.periods == 1, "periods > 1 not implemented."

        # Create internal modules for forward fill and lag operations
        self.ffill = Ffill()
        self.lag = Lag(lag=self.periods)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute percentage change.

        Args:
            x: Input data value

        Returns:
            Percentage change: (current / previous) - 1.0
        """
        # Get previous value, with optional forward-filling
        if self.fill_method in ["ffill", "pad"]:
            # Forward fill NaN values before computing lag
            x_filled = self.ffill(x)
            previous_x = self.lag(x_filled)
        else:
            # Use raw values without forward-filling
            previous_x = self.lag(x)

        # Compute percentage change: (current / previous) - 1.0
        pct_change = x / previous_x - 1.0

        # Handle special case: if current is NaN but previous is valid,
        # return 0.0 (pandas-compatible behavior when fillna_zero=True)
        if self.fillna_zero:
            pct_change = jnp.where(jnp.isnan(x) & ~jnp.isnan(previous_x), 0.0, pct_change)

        return jnp.array(pct_change)


def create_pct_change(
    periods: int = 1,
    fill_method: str = "pad",
    limit: int | None = None,
    fillna_zero: bool = True,
) -> PctChange:
    """Factory function to create PctChange module.

    Args:
        periods: Periods to shift for forming relative change
        fill_method: How to handle NAs before computing percent changes
        limit: The number of consecutive NAs to fill before stopping (not implemented)
        fillna_zero: Return 0 when current is NaN but previous is valid

    Returns:
        PctChange module instance
    """
    return PctChange(
        periods=periods,
        fill_method=fill_method,
        limit=limit,
        fillna_zero=fillna_zero,
    )
