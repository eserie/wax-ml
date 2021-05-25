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
"""Exponentially weighted variance module."""
import haiku as hk
import jax.numpy as jnp

from wax.modules.ewma import EWMA


class EWMCov(hk.Module):
    """Exponentially weighted covariance.
    To calculate the variance we use the fact that Var(X) = Mean(x^2) - Mean(x)^2 and internally
    we use the exponentially weighted mean of x/x^2 to calculate this.

    Arguments:
        alpha : The closer `alpha` is to 1 the more the statistic will adapt to recent values.

    Attributes:
        variance : The running exponentially weighted variance.

    References
    ----------
    [^1]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf) # noqa
    """

    def __init__(self, alpha=0.5, adjust=True, assume_centered=False, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.adjust = adjust
        self.assume_centered = assume_centered

    def __call__(self, data):
        x, y = data
        mean_xy = EWMA(self.alpha, self.adjust, initial_value=jnp.nan, name="mean_xy")(
            jnp.outer(x, y)
        )
        if self.assume_centered:
            cov = mean_xy
        else:
            mean_x = EWMA(
                self.alpha, self.adjust, initial_value=jnp.nan, name="mean_x"
            )(x)
            mean_y = EWMA(
                self.alpha, self.adjust, initial_value=jnp.nan, name="mean_y"
            )(y)
            cov = mean_xy - jnp.outer(mean_x, mean_y)
        return cov
