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

    It can be used to forecast timeseries with a ARMA dynamic using online learning as described in [^1].

    The API of this module is similar to the one of SNARIMAX model in the river Python library.

    References
    ----------
    [^1] [Anava, O., Hazan, E., Mannor, S. and Shamir, O., 2013, June.
    Online learning for time series prediction. In Conference on learning theory (pp. 172-184)]
    (https://arxiv.org/pdf/1302.6927.pdf)

    """

    def __init__(
        self,
        p: int,  # AR
        d: int,  # differenciate
        q: int,  # MA
        m: int = 1,  # Seasonal part
        sp: int = 0,  # Seasonal AR
        sd: int = 0,
        sq: int = 0,  # Seasonal MA
        model=None,
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
        self.model = model if model is not None else hk.Linear(1, with_bias=False)

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

        y_pred = self.model(X).reshape(y.shape)
        err = y - y_pred
        errp = Buffer(self.q + 1, name="err_lag")(err)[1:]
        hk.set_state("errp", errp)
        return y_pred, {}
