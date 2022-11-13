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
from typing import Dict, Optional, Tuple, Union, cast

import haiku as hk
from jax import numpy as jnp


class EWMA(hk.Module):
    """Compute exponential moving average."""

    def __init__(
        self,
        *,
        com: Optional[float] = None,
        alpha: Optional[float] = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        initial_value: float = jnp.nan,
        return_info: bool = False,
        name: Optional[str] = None,
    ):
        r"""Initialize module.

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

            return_info : if true, a dictionary is returned in addition to the module output which
                contains additional variables.

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

        self.com = cast(float, com)
        self.min_periods = min_periods
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.initial_value = initial_value
        self.return_info = return_info

    def __call__(
        self, x: jnp.ndarray
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]:
        """Compute EWMA.

        Args:
            x: input data.

        Returns:
            last_mean : value of the mean

            info: A dictionnary with additionnal variables. It is returned if `return_info` is true.
        """
        info = {}

        logcom = hk.get_parameter(
            "logcom",
            shape=[],
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.array(jnp.log(self.com), dtype),
        )
        com = jnp.exp(logcom)

        mean = hk.get_state(
            "mean",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, self.initial_value, dtype),
        )

        # get status
        is_observation = ~jnp.isnan(x)
        isnan_mean = jnp.isnan(mean)

        # fillna by zero to avoid nans in gradient computations
        x = jnp.nan_to_num(x)
        mean = jnp.nan_to_num(mean)

        alpha = 1.0 / (1.0 + com)

        if self.adjust:
            new_wt = jnp.array(1.0)
        else:
            new_wt = alpha

        old_wt = hk.get_state(
            "old_wt",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, 1.0, dtype),
        )

        if self.adjust == "linear":
            # com_eff grow linearly when there is observation but
            # decrease linearly when there is nans.
            old_wt_factor = jnp.where(
                is_observation, 1.0, jnp.maximum(0.0, (old_wt - 1.0) / old_wt)
            )
            old_wt = jnp.minimum(old_wt, com)
        else:
            old_wt_factor = 1.0 - alpha

        if self.ignore_na:
            old_wt = jnp.where(is_observation, old_wt * old_wt_factor, old_wt)
        else:
            old_wt = old_wt * old_wt_factor

        old_wt = jnp.where(isnan_mean, 0.0, old_wt)

        mean = jnp.where(
            is_observation, (old_wt * mean + new_wt * x) / (old_wt + new_wt), mean
        )

        if self.return_info:
            info["com_eff"] = old_wt / new_wt

        if self.adjust:
            old_wt = jnp.where(is_observation, old_wt + new_wt, old_wt)
        else:
            old_wt = jnp.where(is_observation, 1.0, old_wt)

        # restore nan
        mean = jnp.where(jnp.logical_and(~is_observation, isnan_mean), jnp.nan, mean)

        hk.set_state("old_wt", old_wt)
        hk.set_state("mean", mean)

        nobs = hk.get_state(
            "nobs",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, 0, dtype=int),
        )
        nobs = jnp.where(is_observation, nobs + 1, nobs)
        if self.return_info:
            info["nobs"] = nobs
        hk.set_state("nobs", nobs)

        if self.min_periods:
            result = jnp.where(nobs >= self.min_periods, mean, jnp.nan)
        else:
            result = mean

        if self.return_info:
            return result, info
        else:
            return result
