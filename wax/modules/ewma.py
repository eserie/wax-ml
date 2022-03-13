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
from typing import Dict, Tuple, Union, cast

import haiku as hk
from jax import numpy as jnp


class EWMA(hk.Module):
    """Compute exponentioal moving average."""

    def __init__(
        self,
        alpha: float = None,
        com: float = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        initial_value=jnp.nan,
        return_info: bool = False,
        name: str = None,
    ):
        """Initialize module.

        Args:
            alpha:  Specify smoothing factor :math:`\alpha` directly
                :math:`0 < \alpha \leq 1`.
            com : Specify decay in terms of center of mass
                :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.

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
        assert cast(float, com) > 0.0

        self.com = com
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

        # initialization on first non-nan value if initial_value is nan
        mean = jnp.where(jnp.isnan(mean), x, mean)

        # get status
        is_observation = ~jnp.isnan(x)
        isnan_mean = jnp.isnan(mean)

        # fillna by zero to avoid nans in gradient computations
        x = jnp.nan_to_num(x)
        mean = jnp.nan_to_num(mean)

        alpha = 1.0 / (1.0 + com)

        if not self.ignore_na or self.min_periods:
            is_initialized = hk.get_state(
                "is_initialized",
                shape=x.shape,
                dtype=bool,
                init=lambda shape, dtype: jnp.full(shape, False, dtype),
            )

            is_initialized = jnp.where(is_initialized, is_initialized, is_observation)
            if self.return_info:
                info["is_initialized"] = is_initialized
            hk.set_state("is_initialized", is_initialized)

        if self.min_periods:
            count = hk.get_state(
                "count",
                shape=x.shape,
                dtype=x.dtype,
                init=lambda shape, dtype: jnp.full(shape, 0.0, dtype),
            )
            if self.return_info:
                info["count"] = count
            if self.ignore_na:
                count = jnp.where(is_observation, count + 1, count)
            else:
                count = jnp.where(is_initialized, count + 1, count)
            hk.set_state("count", count)

        if self.adjust:
            # adjustement scheme
            com_eff = hk.get_state(
                "com_eff",
                shape=x.shape,
                dtype=x.dtype,
                init=lambda shape, dtype: jnp.full(shape, 0.0, dtype),
            )
            if self.return_info:
                info["com_eff"] = com_eff
            alpha_eff = 1.0 / (1.0 + com_eff)

            if self.adjust == "linear":
                if self.ignore_na:
                    com_eff = jnp.where(is_observation, com_eff + 1, com_eff)
                else:
                    com_eff = jnp.where(is_initialized, com_eff + 1, com_eff)
                com_eff = jnp.minimum(com_eff, com)
            else:
                # exponential scheme (as in pandas)
                if self.ignore_na:
                    com_eff = jnp.where(
                        is_observation, alpha * com + (1 - alpha) * com_eff, com_eff
                    )
                else:
                    com_eff = jnp.where(
                        is_initialized, alpha * com + (1 - alpha) * com_eff, com_eff
                    )
            hk.set_state("com_eff", com_eff)
        else:
            alpha_eff = alpha

        if self.return_info:
            info["alpha_eff"] = alpha_eff

        # update mean  if x is not nan
        if self.ignore_na:
            mean = jnp.where(
                is_observation, (1.0 - alpha_eff) * mean + alpha_eff * x, mean
            )
        else:
            norm = hk.get_state(
                "norm",
                shape=x.shape,
                dtype=x.dtype,
                init=lambda shape, dtype: jnp.full(shape, 1.0, dtype),
            )

            mean = jnp.where(
                is_initialized, (1.0 - alpha_eff) * mean + alpha_eff * x, mean
            )
            norm = jnp.where(
                is_initialized,
                (1.0 - alpha_eff) * norm + alpha_eff * is_observation,
                norm,
            )

            if self.return_info:
                info["mean"] = mean
                info["norm"] = norm

            hk.set_state("norm", norm)

        # restore nan
        mean = jnp.where(jnp.logical_and(~is_observation, isnan_mean), jnp.nan, mean)

        hk.set_state("mean", mean)

        if self.ignore_na:
            last_mean = mean
        else:
            last_mean = hk.get_state(
                "last_mean",
                shape=x.shape,
                dtype=x.dtype,
                init=lambda shape, dtype: jnp.full(shape, self.initial_value, dtype),
            )

            # update only if
            last_mean = jnp.where(is_observation, mean / norm, last_mean)
            hk.set_state("last_mean", last_mean)

        if self.min_periods:
            last_mean = jnp.where(count < self.min_periods, jnp.nan, last_mean)

        if self.return_info:
            return last_mean, info
        else:
            return last_mean
