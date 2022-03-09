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
from jax import numpy as jnp


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
        alpha = hk.get_parameter(
            "alpha",
            shape=[],
            dtype=x.dtype,
            init=lambda *_: jnp.array(self.alpha),
        )

        # initialization on first non-nan value
        mean = jnp.where(jnp.isnan(mean), x, mean)
        isnan_x = jnp.isnan(x)
        isnan_mean = jnp.isnan(mean)

        # fillna by zero to avoid nans in gradient computations
        x = jnp.nan_to_num(x)
        mean = jnp.nan_to_num(mean)

        # alpha adjustement scheme
        if self.adjust == "linear":
            count = hk.get_state(
                "count",
                shape=x.shape,
                dtype=x.dtype,
                init=lambda shape, dtype: jnp.full(shape, 0.0, dtype),
            )
            count = jnp.where(isnan_x, count, count + 1)
            hk.set_state("count", count)

            tscale = 1.0 / alpha
            tscale = jnp.where(count < tscale, count, tscale)
            alpha_eff = jnp.where(tscale > 0, 1.0 / tscale, jnp.nan)
        elif self.adjust:
            rho_pow = hk.get_state(
                "rho_pow",
                shape=x.shape,
                dtype=x.dtype,
                init=lambda shape, dtype: jnp.full(shape, 1.0, dtype),
            )
            # exponential scheme (as in pandas)
            eps = jnp.finfo(x.dtype).resolution
            rho_pow = jnp.where(isnan_x, rho_pow, rho_pow * (1 - alpha))
            alpha_eff = jnp.where(isnan_x, alpha, alpha / (eps + 1.0 - rho_pow))
            hk.set_state("rho_pow", rho_pow)
        else:
            alpha_eff = alpha

        # update mean  if x is not nan
        mean = jnp.where(isnan_x, mean, (1.0 - alpha_eff) * mean + alpha_eff * x)

        # restore nan
        mean = jnp.where(jnp.logical_and(isnan_x, isnan_mean), jnp.nan, mean)

        hk.set_state("mean", mean)
        return mean
