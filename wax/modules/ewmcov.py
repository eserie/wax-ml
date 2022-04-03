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
"""Compute exponentially weighted covariance."""
from typing import Optional

import haiku as hk
import jax.numpy as jnp

from wax.modules.ewma import EWMA


class EWMCov(hk.Module):
    """Compute exponentially weighted covariance.
    To calculate the variance we use the fact that Var(X) = Mean(x^2) - Mean(x)^2 and internally
    we use the exponentially weighted mean of x/x^2 to calculate this.

    References
    ----------
    [^1]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf) # noqa
    """

    def __init__(
        self,
        *,
        alpha: Optional[float] = None,
        com: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        initial_value: float = jnp.nan,
        assume_centered: bool = False,
        name: str = None,
    ):
        """

        Args:
            alpha: alpha parameter of the exponential moving average.
            adjust: if true, implement a non-stationary filter with exponential initialization
                scheme. If "linear", implement a non-stationary filter with linear initialization.
            assume_centered: if true, assume that the mean estimator is zero.
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
        self.min_periods = min_periods
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.initial_value = initial_value
        self.assume_centered = assume_centered

    def __call__(self, x, y=None):
        """Compute"""
        if y is None:
            import warnings

            warnings.warn(
                "x and y arguments may be passed directly"
                "instead as a tuple argument. The tuple argument syntax"
                "may be removed in the future."
            )
            x, y = x
        mean_xy = EWMA(
            com=self.com,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            initial_value=self.initial_value,
            name="mean_xy",
        )(jnp.outer(x, y))
        if self.assume_centered:
            cov = mean_xy
        else:
            mean_x = EWMA(
                com=self.com,
                min_periods=self.min_periods,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                initial_value=self.initial_value,
                name="mean_x",
            )(x)
            mean_y = EWMA(
                com=self.com,
                min_periods=self.min_periods,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                initial_value=self.initial_value,
                name="mean_y",
            )(y)
            cov = mean_xy - jnp.outer(mean_x, mean_y)
        return cov
