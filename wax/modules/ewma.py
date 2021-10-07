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
"""Compute exponentioal moving average."""
import haiku as hk
import jax.numpy as jnp


class EWMA(hk.Module):
    """Compute exponentioal moving average."""

    def __init__(
        self, alpha: float, adjust: bool = True, initial_value=jnp.nan, name: str = None
    ):
        """Initialize module.

        Args:
            alpha: alpha parameter of the exponential moving average.
            adjust: if true, implement a non-stationary filter with exponential initialization
                scheme. If "linear", implement a non-stationary filter with linear initialization.
            initial_value: initial value for the state.
            name : name of the module instance.
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.adjust = adjust
        self.initial_value = initial_value

    def __call__(self, x):
        """Compute EWMA.

        Args:
            x: input data.
        """
        mean = hk.get_state(
            "mean",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, self.initial_value, dtype),
        )
        is_initialized = hk.get_state(
            "is_initialized",
            shape=x.shape,
            dtype=bool,
            init=lambda shape, dtype: jnp.full(shape, False, dtype),
        )
        count = hk.get_state(
            "count",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, 0.0, dtype),
        )

        alpha = hk.get_parameter(
            "alpha",
            shape=[],
            dtype=x.dtype,
            init=lambda *_: jnp.array(self.alpha),
        )

        # initialization on first non-nan value
        mean = jnp.where(is_initialized, mean, x)
        is_initialized = jnp.where(jnp.isnan(x), is_initialized, True)
        hk.set_state("is_initialized", is_initialized)

        isnan_x = jnp.isnan(x)
        x = jnp.nan_to_num(x)
        mean = jnp.nan_to_num(mean)

        count = jnp.where(isnan_x, count, count + 1)

        # alpha adjustement scheme
        if self.adjust == "linear":
            tscale = 1.0 / alpha
            tscale = jnp.where(count < tscale, count, tscale)
            alpha = jnp.where(tscale > 0, 1.0 / tscale, jnp.nan)
        elif self.adjust:
            # exponential scheme (as in pandas)
            eps = jnp.finfo(x.dtype).resolution
            alpha = jnp.where(
                count > 0, alpha / (eps + 1.0 - (1.0 - alpha) ** (count)), 1.0
            )
            # alpha = alpha / (1.0 - (1.0 - alpha) ** (count))

        # update mean  if x is not nan
        mean = jnp.where(isnan_x, mean, (1.0 - alpha) * mean + alpha * x)

        mean = jnp.where(is_initialized, mean, self.initial_value)
        hk.set_state("mean", mean)
        hk.set_state("count", count)
        return mean
