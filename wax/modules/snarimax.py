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
import haiku as hk
import jax.numpy as jnp

from wax.modules.buffer import Buffer
from wax.modules.fill_nan_inf import FillNanInf


class SNARIMAX(hk.Module):
    """SNARIMAX Adaptive filter.

    It can be used to forecast timeseries with a ARMA dynamic using the online learning
    method as described in [^4].

    The API of this module is similar to the one of SNARIMAX model in the river Python library.

    Parameters
    ----------
    p
        Order of the autoregressive part. This is the number of past target values that will be
        included as features.
    d
        Differencing order.
    q
        Order of the moving average part. This is the number of past error terms that will be
        included as features.
    m
        Season length used for extracting seasonal features. If you believe your data has a
        seasonal pattern, then set this accordingly. For instance, if the data seems to exhibit
        a yearly seasonality, and that your data is spaced by month, then you should set this
        to 12. Note that for this parameter to have any impact you should also set at least one
        of the `p`, `d`, and `q` parameters.
    sp
        Seasonal order of the autoregressive part. This is the number of past target values
        that will be included as features.
    sd
        Seasonal differencing order.
    sq
        Seasonal order of the moving average part. This is the number of past error terms that
        will be included as features.

    regressor
        The online regression model to use. By default, a haiku `Linear` module.

    References
    ----------
    [^1]: [Wikipedia page on ARMA](https://www.wikiwand.com/en/Autoregressive%E2%80%93moving-average_model)
    [^2]: [Wikipedia page on NARX](https://www.wikiwand.com/en/Nonlinear_autoregressive_exogenous_model)
    [^3]: [ARIMA models](https://otexts.com/fpp2/arima.html)
    [^4] [Anava, O., Hazan, E., Mannor, S. and Shamir, O., 2013, June.
    Online learning for time series prediction. In Conference on learning theory (pp. 172-184)]
    (https://arxiv.org/pdf/1302.6927.pdf)

    """

    def __init__(
        self,
        p: int,
        d: int = 0,
        q: int = 0,
        m: int = 1,
        sp: int = 0,
        sd: int = 0,
        sq: int = 0,
        regressor=None,
        name=None,
    ):
        super().__init__(name=name)
        self.p = p
        self.d = d
        self.q = q
        self.m = m
        self.sp = sp
        self.sd = sd
        self.sq = sq
        self.regressor = regressor if regressor is not None else hk.Linear(1)
        if d != 0 or sp != 0 or sd != 0 or sq != 0:
            raise NotImplementedError(
                "Options 'd', 'sp', 'sd', 'sq' are not yet implemented."
            )

    def __call__(self, y, X=None):
        yp = Buffer(self.p + 1, name="y_trues")(y)[1:]

        errp = hk.get_state(
            "errp",
            [],
            init=lambda *_: jnp.full((self.q + 1,) + y.shape, 0.0, y.dtype)[1:],
        )

        X = [X] if X is not None else []
        X += [errp.flatten(), yp.flatten()]
        X = jnp.concatenate(X)
        X = FillNanInf()(X)

        y_pred = self.regressor(X).reshape(y.shape)
        err = y - y_pred
        errp = Buffer(self.q + 1, name="err_lag")(err)[1:]
        hk.set_state("errp", errp)
        return y_pred, {}
