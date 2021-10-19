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
"""ARMA linear filter."""
import haiku as hk
import jax.numpy as jnp

from wax.modules.buffer import Buffer
from wax.modules.fill_nan_inf import FillNanInf


class ARMA(hk.Module):
    """ARMA linear filter."""

    def __init__(self, alpha, beta, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, eps):
        eps_buffer = Buffer(len(self.beta) + 1, 0.0, name="eps_buffer")(eps)[:-1]

        y_buffer = hk.get_state(
            "y_buffer",
            eps.shape,
            eps.dtype,
            init=lambda shape, dtype: jnp.zeros(
                ((len(self.alpha),) + shape), dtype=dtype
            ),
        )

        y = self.alpha @ y_buffer + self.beta @ eps_buffer + eps

        y = FillNanInf()(y)

        # reshape with shape of eps (for scalar case)
        y = y.reshape(eps.shape)

        y_buffer = Buffer(len(self.alpha), 0, name="y_buffer")(y)

        hk.set_state("y_buffer", y_buffer)
        return y
