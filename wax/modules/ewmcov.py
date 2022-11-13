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
        com: Optional[float] = None,
        alpha: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        initial_value: float = jnp.nan,
        assume_centered: bool = False,
        name: Optional[str] = None,
    ):
        r"""Initialize the module.

        Args:
            com : Specify decay in terms of center of mass
                :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.

            alpha:  Specify smoothing factor :math:`\alpha` directly
                :math:`0 < \alpha \leq 1`.

            min_periods : Minimum number of observations in window required to have a value;
                otherwise, result is ``np.nan``.

            adjust : Divide by decaying adjustment factor in beginning periods to account
                for imbalance in relative weightings (viewing EWMA as a moving average).
                - When ``adjust=True`` (default), the EW function is calculated using weights
                  :math:`w_i = (1 - \alpha)^i`. For example, the EW moving average of the series
                  [:math:`x_0, x_1, ..., x_t`] would be:
                .. math::
                    y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ... + (1 -
                    \alpha)^t x_0}{1 + (1 - \alpha) + (1 - \alpha)^2 + ... + (1 - \alpha)^t}
                - When ``adjust=False``, the exponentially weighted function is calculated
                  recursively:
                .. math::
                    \begin{split}
                        y_0 &= x_0\\
                        y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,
                    \end{split}
                The effective  center of mass (com) interpolate exponentially between 0 and the
                nominal center of mass.

                - When ``adjust='linear'`` the effective  center of mass (com) interpolate linearly
                between 0 and the nominal center of mass.

            ignore_na : Ignore missing values when calculating weights.
                - When ``ignore_na=False`` (default), weights are based on absolute positions.
                  For example, the weights of :math:`x_0` and :math:`x_2` used in calculating
                  the final weighted average of [:math:`x_0`, None, :math:`x_2`] are
                  :math:`(1-\alpha)^2` and :math:`1` if ``adjust=True``, and
                  :math:`(1-\alpha)^2` and :math:`\alpha` if ``adjust=False``.
                - When ``ignore_na=True``, weights are based
                  on relative positions. For example, the weights of :math:`x_0` and :math:`x_2`
                  used in calculating the final weighted average of
                  [:math:`x_0`, None, :math:`x_2`] are :math:`1-\alpha` and :math:`1` if
                  ``adjust=True``, and :math:`1-\alpha` and :math:`\alpha` if ``adjust=False``.


            initial_value : initial value for the state.

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
