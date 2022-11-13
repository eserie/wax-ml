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
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, cast

import numba
import numpy as np
import pandas as pd


class EWMAState(NamedTuple):
    mean: Any
    old_wt: Any
    nobs: Any


def ewma(
    *,
    com: Optional[float] = None,
    alpha: Optional[float] = None,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
    initial_value=np.nan,
):
    r"""Compute exponentioal moving average.

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
    """
    assert (
        com is not None or alpha is not None
    ), "com or alpha parameters must be specified."
    if com is not None:
        assert alpha is None
    elif alpha is not None:
        assert com is None
        com = 1.0 / alpha - 1.0
    com = cast(float, com)
    assert com > 0.0
    alpha = 1.0 / (1.0 + com)

    def init_ewma_state(x):
        x = x[0]

        dtype = x.dtype
        shape = x.shape

        state = EWMAState(
            mean=np.full(shape, initial_value, dtype),
            old_wt=np.full(shape, 1.0, dtype),
            nobs=np.full(shape, 0.0, dtype=np.int64),
        )
        return state

    def apply(values: np.ndarray, state: Optional[EWMAState] = None):
        is_1d = False
        if values.ndim == 1:
            values = values.reshape(-1, 1)
            is_1d = True

        if state is None:
            state = init_ewma_state(values)
        mean = state.mean
        old_wt = state.old_wt
        nobs = state.nobs

        res, mean, old_wt, nobs = numba_apply(values, mean, old_wt, nobs)
        state = EWMAState(mean, old_wt, nobs)

        if is_1d:
            res = res.reshape(values.shape[0])
        return res, state

    @numba.jit(nopython=True, nogil=True, parallel=False)
    def numba_apply(values, mean, old_wt, nobs):

        """
        Compute online exponentially weighted mean per column over 2D values.

        Takes the first observation as is, then computes the subsequent
        exponentially weighted mean accounting minimum periods.
        """
        minimum_periods = min_periods

        if adjust:
            new_wt = 1.0
        else:
            new_wt = alpha

        # deltas = np.ones(values.shape)
        result = np.empty(values.shape)
        weighted_avg = mean

        for i in range(len(values)):
            cur = values[i]
            is_observations = ~np.isnan(cur)
            nobs += is_observations.astype(np.int64)
            for j in numba.prange(len(cur)):
                if not np.isnan(weighted_avg[j]):
                    if adjust == "linear":
                        if is_observations[j]:
                            old_wt_factor = 1.0
                        else:
                            if old_wt[j] > 0:
                                old_wt_factor = np.maximum(
                                    0.0, (old_wt[j] - 1.0) / old_wt[j]
                                )
                            else:
                                old_wt_factor = 0.0
                        old_wt[j] = np.minimum(old_wt[j], com)
                    else:
                        old_wt_factor = 1.0 - alpha

                    if is_observations[j] or not ignore_na:
                        # note that len(deltas) = len(vals) - 1 and deltas[i] is to be
                        # used in conjunction with vals[i+1]
                        old_wt[j] *= old_wt_factor  # ** deltas[j - 1]
                        if is_observations[j]:
                            # avoid numerical errors on constant series
                            if weighted_avg[j] != cur[j]:
                                weighted_avg[j] = (
                                    (old_wt[j] * weighted_avg[j]) + (new_wt * cur[j])
                                ) / (old_wt[j] + new_wt)
                            if adjust:
                                old_wt[j] += new_wt
                            else:
                                old_wt[j] = 1.0
                elif is_observations[j]:
                    weighted_avg[j] = cur[j]

            result[i] = np.where(nobs >= minimum_periods, weighted_avg, np.nan)
            mean = weighted_avg
        return result, mean, old_wt, nobs

    return apply


class WaxAccessor:
    def ewm(self, *args, **kwargs):
        return NumbaExponentialMovingWindow(accessor=self, *args, **kwargs)


@dataclass(frozen=True)
class NumbaExponentialMovingWindow:
    accessor: WaxAccessor
    com: Optional[float] = None
    alpha: Optional[float] = None
    min_periods: int = 0
    adjust: bool = True
    ignore_na: bool = False
    initial_value: float = np.nan
    return_state: bool = False

    def mean(self, state=None):
        def _apply_ema(
            *,
            state,
            obj,
            com,
            alpha,
            min_periods,
            adjust,
            ignore_na,
            initial_value,
            return_state,
        ):
            if state is not None:
                state = state.reindex(obj.columns)
                state = EWMAState(
                    mean=state["mean"].values,
                    old_wt=state["old_wt"].values,
                    nobs=state["nobs"].values,
                )

            res, state = ewma(
                alpha=alpha,
                com=com,
                min_periods=min_periods,
                adjust=adjust,
                ignore_na=ignore_na,
                initial_value=initial_value,
            )(obj.values, state)

            res = pd.DataFrame(res, obj.index, obj.columns)

            state = state._replace(
                mean=pd.Series(state.mean, obj.columns),
                old_wt=pd.Series(state.old_wt, obj.columns),
                nobs=pd.Series(state.nobs, obj.columns),
            )
            state = pd.concat(state._asdict(), axis=1)
            if return_state:
                return res, state
            else:
                return res

        kwargs = self.__dict__.copy()
        obj = kwargs.pop("accessor")._obj

        if isinstance(obj, pd.Series):
            kwargs["obj"] = obj.to_frame()
            res = _apply_ema(state=state, **kwargs)
            if self.return_state:
                res, state = res
            res = res.iloc[:, 0]
            if self.return_state:
                return res, state
            else:
                return res
        else:
            kwargs["obj"] = obj
            return _apply_ema(state=state, **kwargs)


class WaxNumbaEWMAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def ewm(self, *args, **kwargs):
        return NumbaExponentialMovingWindow(accessor=self, *args, **kwargs)


def register_wax_numba():
    pd.api.extensions.register_dataframe_accessor("wax_numba")(WaxNumbaEWMAccessor)
    pd.api.extensions.register_series_accessor("wax_numba")(WaxNumbaEWMAccessor)
