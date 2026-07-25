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
"""Open-High-Low-Close binning."""

from typing import NamedTuple

import haiku as hk
import jax.numpy as jnp


class OHLCData(NamedTuple):
    OPEN: jnp.ndarray
    HIGH: jnp.ndarray
    LOW: jnp.ndarray
    CLOSE: jnp.ndarray


class OHLC(hk.Module):
    """Open-High-Low-Close binning."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, input: jnp.ndarray, *, reset_on: bool):
        def init_array(shape, dtype):
            return jnp.full(shape, fill_value=jnp.nan, dtype=dtype)

        # states
        BIN_OPEN = hk.get_state("BIN_OPEN", input.shape, input.dtype, init_array)
        BIN_HIGH = hk.get_state("BIN_HIGH", input.shape, input.dtype, init_array)
        BIN_LOW = hk.get_state("BIN_LOW", input.shape, input.dtype, init_array)
        BIN_CLOSE = hk.get_state("BIN_CLOSE", input.shape, input.dtype, init_array)

        def daily_call(operand):
            input, BIN_CLOSE = operand
            # initialize OPEN, HIGH, LOW, CLOSE
            BIN_CLOSE = init_array(input.shape, input.dtype)
            BIN_OPEN = init_array(input.shape, input.dtype)
            BIN_HIGH = init_array(input.shape, input.dtype)
            BIN_LOW = init_array(input.shape, input.dtype)
            return OHLCData(BIN_OPEN, BIN_HIGH, BIN_LOW, BIN_CLOSE)

        operand = input, BIN_CLOSE
        BIN_state = OHLCData(BIN_OPEN, BIN_HIGH, BIN_LOW, BIN_CLOSE)
        updated_BIN_state = hk.cond(
            pred=reset_on,
            true_operand=operand,
            true_fun=daily_call,
            false_operand=operand,
            false_fun=(lambda operand: BIN_state),
        )
        BIN_OPEN, BIN_HIGH, BIN_LOW, BIN_CLOSE = updated_BIN_state

        # update BIN_CLOSE
        BIN_CLOSE = jnp.where(jnp.logical_not(jnp.isnan(input)), input, BIN_CLOSE)

        # initialize BIN_{OPEN,HIGH,LOW} with the first non nan BIN_CLOSE
        BIN_OPEN = jnp.where(
            jnp.isnan(BIN_OPEN),
            BIN_CLOSE,
            BIN_OPEN,
        )

        BIN_HIGH = jnp.where(
            jnp.isnan(BIN_HIGH),
            BIN_CLOSE,
            BIN_HIGH,
        )

        BIN_LOW = jnp.where(
            jnp.isnan(BIN_LOW),
            BIN_CLOSE,
            BIN_LOW,
        )

        # update BIN_HIGH and BIN_LOW
        BIN_HIGH = jnp.where(
            BIN_CLOSE > BIN_HIGH,
            BIN_CLOSE,
            BIN_HIGH,
        )
        BIN_LOW = jnp.where(
            BIN_CLOSE < BIN_LOW,
            BIN_CLOSE,
            BIN_LOW,
        )
        # update state
        hk.set_state("BIN_OPEN", BIN_OPEN)
        hk.set_state("BIN_HIGH", BIN_HIGH)
        hk.set_state("BIN_LOW", BIN_LOW)
        hk.set_state("BIN_CLOSE", BIN_CLOSE)

        return OHLCData(BIN_OPEN, BIN_HIGH, BIN_LOW, BIN_CLOSE)
