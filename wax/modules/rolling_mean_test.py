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
import pandas as pd
from jax.config import config

from wax.compile import jit_init_apply
from wax.modules.rolling_mean import RollingMean
from wax.unroll import dynamic_unroll


def test_rolling_mean_init_apply():
    @jit_init_apply
    @hk.transform_with_state
    def rolling_mean(x):
        return RollingMean(2)(x)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(next(seq), (2, 3))
    params, state = rolling_mean.init(next(seq), x)
    output, state = rolling_mean.apply(params, state, next(seq), x)
    assert output is not x
    assert (output == x).all()
    x1 = x

    x = jax.random.normal(next(seq), (2, 3))
    output, state = rolling_mean.apply(params, state, next(seq), x)
    assert ((x1 + x) / 2 == output).all()
    x2 = x

    x = jax.random.normal(next(seq), (2, 3))
    output, state = rolling_mean.apply(params, state, next(seq), x)
    assert ((x2 + x) / 2 == output).all()


def test_run_ema_vs_pandas_not_adjust(window=10, min_periods=5):

    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(100, 5), key=next(seq), dtype=jnp.float64)
    mask = jax.random.choice(next(seq), 2, shape=x.shape)
    x = jnp.where(mask, x, jnp.nan)

    @jit_init_apply
    @hk.transform_with_state
    def rolling_mean(x):
        return RollingMean(window, min_periods)(x)

    mean, state = dynamic_unroll(rolling_mean, None, None, next(seq), False, x)

    mean_pandas = pd.DataFrame(x).rolling(window, min_periods).mean()

    assert jnp.allclose(mean, mean_pandas.values, equal_nan=True)
