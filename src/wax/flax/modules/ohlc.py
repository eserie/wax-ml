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
"""Flax-based OHLC module for Open-High-Low-Close binning."""

from typing import NamedTuple

import flax.linen as nn
import jax.numpy as jnp

from .has_changed import HasChanged


class OHLCData(NamedTuple):
    """Named tuple for OHLC data structure."""

    OPEN: jnp.ndarray
    HIGH: jnp.ndarray
    LOW: jnp.ndarray
    CLOSE: jnp.ndarray


class OHLC(nn.Module):
    """Flax-based OHLC (Open-High-Low-Close) binning module.

    This module maintains running OHLC statistics for time-series data,
    with support for periodic resets (e.g., daily OHLC bars).
    """

    def setup(self):
        """Setup the OHLC module."""
        self.has_changed = HasChanged()

    @nn.compact
    def __call__(self, input: jnp.ndarray, reset_on: jnp.ndarray) -> OHLCData:
        """Process input data and maintain OHLC statistics.

        Args:
            input: Input time-series data value
            reset_on: Boolean flag indicating when to reset bins

        Returns:
            OHLCData containing current OPEN, HIGH, LOW, CLOSE values
        """
        # Ensure input is an array
        input = jnp.asarray(input)
        reset_on = jnp.asarray(reset_on)

        # Use reset signal directly
        should_reset = reset_on

        # Initialize OHLC state variables
        open_val = self.variable(
            "state", "open", lambda: jnp.full(input.shape, jnp.nan, input.dtype)
        )
        high_val = self.variable(
            "state", "high", lambda: jnp.full(input.shape, jnp.nan, input.dtype)
        )
        low_val = self.variable("state", "low", lambda: jnp.full(input.shape, jnp.nan, input.dtype))
        close_val = self.variable(
            "state", "close", lambda: jnp.full(input.shape, jnp.nan, input.dtype)
        )

        # Get current state values
        current_open = open_val.value
        current_high = high_val.value
        current_low = low_val.value
        # The previous close is never needed: CLOSE is always the latest input.

        # Determine if this is the first valid input or a reset
        is_first_or_reset = should_reset | jnp.isnan(current_open)

        # Update OPEN: set to input if first/reset, otherwise keep current
        new_open = jnp.where(is_first_or_reset, input, current_open)

        # Update HIGH: max of current high and input, or input if first/reset
        new_high = jnp.where(is_first_or_reset, input, jnp.maximum(current_high, input))

        # Update LOW: min of current low and input, or input if first/reset
        new_low = jnp.where(is_first_or_reset, input, jnp.minimum(current_low, input))

        # Update CLOSE: always set to current input
        new_close = input

        # Update state variables
        open_val.value = new_open
        high_val.value = new_high
        low_val.value = new_low
        close_val.value = new_close

        # Return OHLC data structure
        return OHLCData(OPEN=new_open, HIGH=new_high, LOW=new_low, CLOSE=new_close)


def create_ohlc() -> OHLC:
    """Factory function to create OHLC module.

    Returns:
        OHLC module instance
    """
    return OHLC()
