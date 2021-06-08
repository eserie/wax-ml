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
"""Compute exponentially weighted variance."""
import haiku as hk
import jax.numpy as jnp


class EWMVar(hk.Module):
    """Compute exponentially weighted variance.

    To calculate the variance we use the fact that Var(X) = Mean(x^2) - Mean(x)^2 and internally
    we use the exponentially weighted mean of x/x^2 to calculate this.

    References
    ----------
    [^1]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf) # noqa
    """

    def __init__(self, alpha: float = 0.5, adjust: bool = True, name: str = None):
        """Initialize module.

        Args:
            alpha: alpha parameter of the exponential moving average.
            adjust: if true, implement a non-stationary filter with exponential initialization
                scheme. If "linear", implement a non-stationary filter with linear initialization.
            name : name of the module instance.
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.adjust = adjust

    def __call__(self, x):
        """Implement last formula in [1]:
        .. maths::
            diff := x - mean
            incr := alpha * diff
            mean := mean + incr
            variance := (1 - alpha) * (variance + diff * incr)
        """
        mean = hk.get_state(
            "mean",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, jnp.nan, dtype),
        )
        variance = hk.get_state(
            "variance",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, jnp.nan, dtype),
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
        mean = jnp.where(jnp.isnan(mean), x, mean)
        variance = jnp.where(jnp.isnan(variance), 0.0, variance)

        mask = jnp.logical_not(jnp.isnan(x))
        count = jnp.where(mask, count + 1, count)

        # alpha adjustement scheme
        if self.adjust == "linear":
            tscale = 1.0 / alpha
            tscale = jnp.where(count < tscale, count, tscale)
            alpha = jnp.where(tscale > 0, 1.0 / tscale, jnp.nan)
        elif self.adjust:
            # exponential scheme (as in pandas)
            alpha = alpha / (1.0 - (1.0 - alpha) ** (count))

        diff = x - mean
        incr = alpha * diff

        # update state
        mean = jnp.where(mask, mean + incr, mean)
        variance = jnp.where(mask, (1 - alpha) * (variance + diff * incr), variance)

        hk.set_state("mean", mean)
        hk.set_state("variance", variance)
        hk.set_state("count", count)

        return variance
