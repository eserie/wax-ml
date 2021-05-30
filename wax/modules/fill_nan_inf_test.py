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
import jax
import jax.numpy as jnp

from wax.modules.fill_nan_inf import FillNanInf


def test_fill_nan_inf():

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(100, 5), key=next(seq), dtype=jnp.float64)
    mask_nan = jax.random.choice(next(seq), 2, shape=x.shape)
    x = jnp.where(mask_nan, x, jnp.nan)
    del mask_nan

    mask_inf = jax.random.choice(next(seq), 2, shape=x.shape)
    x = jnp.where(mask_inf, x, jnp.inf)
    del mask_inf

    mask_ninf = jax.random.choice(next(seq), 2, shape=x.shape)
    x = jnp.where(mask_ninf, x, -jnp.inf)
    del mask_ninf

    fun = hk.transform(lambda x: FillNanInf()(x))

    params = fun.init(next(seq), x)
    x_fill = fun.apply(params, next(seq), x)
    assert jnp.allclose(x_fill, jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
