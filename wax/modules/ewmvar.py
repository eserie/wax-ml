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
from typing import Optional

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

    def __init__(
        self,
        *,
        com: Optional[float] = None,
        alpha: Optional[float] = None,
        adjust: bool = True,
        name: Optional[str] = None
    ):
        r"""Initialize the module.

        Args:
            com : Specify decay in terms of center of mass
                :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.

            alpha:  Specify smoothing factor :math:`\alpha` directly
                :math:`0 < \alpha \leq 1`.

            adjust: if true, implement a non-stationary filter with exponential initialization
                scheme. If "linear", implement a non-stationary filter with linear initialization.

            name : name of the module instance.
        """
        super().__init__(name=name)
        assert (
            com is not None or alpha is not None
        ), "com or alpha parameters must be specified."
        if com is not None:
            assert alpha is None
        elif alpha is not None:
            assert com is None
            com = 1.0 / alpha - 1.0

        self.com = com
        self.adjust = adjust

    def __call__(self, x):
        """Implement last formula in [1]:
        .. maths::
            diff := x - mean
            incr := alpha * diff
            mean := mean + incr
            variance := (1 - alpha) * (variance + diff * incr)
        """

        logcom = hk.get_parameter(
            "logcom",
            shape=[],
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.array(jnp.log(self.com), dtype),
        )
        com = jnp.exp(logcom)
        alpha = 1.0 / (1.0 + com)

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
        nobs = hk.get_state(
            "nobs",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, 0.0, dtype),
        )

        # initialization on first non-nan value
        mean = jnp.where(jnp.isnan(mean), x, mean)
        variance = jnp.where(jnp.isnan(variance), 0.0, variance)

        mask = jnp.logical_not(jnp.isnan(x))
        nobs = jnp.where(mask, nobs + 1, nobs)

        # alpha adjustement scheme
        if self.adjust == "linear":
            tscale = 1.0 / alpha
            tscale = jnp.where(nobs < tscale, nobs, tscale)
            alpha = jnp.where(tscale > 0, 1.0 / tscale, jnp.nan)
        elif self.adjust:
            # exponential scheme (as in pandas)
            alpha = alpha / (1.0 - (1.0 - alpha) ** (nobs))

        diff = x - mean
        incr = alpha * diff

        # update state
        mean = jnp.where(mask, mean + incr, mean)
        variance = jnp.where(mask, (1 - alpha) * (variance + diff * incr), variance)

        hk.set_state("mean", mean)
        hk.set_state("variance", variance)
        hk.set_state("nobs", nobs)

        return variance
